# -*- coding: utf-8 -*-
from __future__ import annotations
"""视觉子进程(TRACK-FIX + POINT-02 + PLAY-01):Face(每帧)+ Hand(自适应提频)。

独立进程 = 独立 GIL:六线程融合后视觉循环曾被饿到 41→19fps,挪进程后真并行。
Face Landmarker 每帧跑(人脸跟随要实时);Hand Landmarker 平时每 HAND_EVERY 帧跑一次
(指向是偶发事件,~7Hz 足够);一旦出现"近手"(score≥0.6 且 size≥0.22,逗它候选)
→ 之后 HAND_BOOST_S 秒内每帧跑(跟手要全帧率,7Hz 会卡),近手消失自动降回省 CPU。

协议(result_q,dict):
  一次 {"kind":"ready"}
  每帧 {"kind":"det", "t":t_grab,
        "face":(u,v,h)|None, "n_faces":int, "face_ms":float,
        "hand":{"angle":deg,"extended":bool,"tip":(u,v),
                "u":f,"v":f,"size":f,"score":f}|None }            # hand 仅在跑了的帧带
MediaPipe VIDEO 模式时间戳严格递增(CALIBRATION §9 坑)。
"""

import math
import os
import time

WRIST, IDX_MCP, IDX_PIP, IDX_TIP = 0, 5, 6, 8
HAND_EVERY = 4         # 平时每 N 帧跑一次手部检测
HAND_NEAR_SCORE = 0.6  # "近手"双门:handedness score(真手>0.9,背景误检<0.6)
HAND_NEAR_SIZE = 0.22  # "近手"双门:bbox 最大边占画面比(逗它的手 0.5+,误检 0.06~0.15)
HAND_BOOST_S = 2.0     # 见到近手后,这么多秒内手检测提频到每帧(跟手用)

# ── M1.5-c sticky 选脸(跨帧粘滞,两张相近脸不再跳)──
STICKY_MATCH_DIST = 0.18    # 匹配上帧脸的最大欧几里得距离(归一化坐标)
STICKY_SWITCH_RATIO = 1.20  # 另一张脸 h > 当前 × ratio 才开始"切换压力"计数
STICKY_SWITCH_FRAMES = 8    # 另一张脸连续 N 帧明显更大才真切(~0.3s@27fps)

# MediaPipe landmark → arcface 5-point 关键点索引
# MP "LEFT"=画面左侧=人的右侧, 与 arcface _ARC_REF_POINTS 顺序一致(右眼,左眼,鼻,右嘴,左嘴)
_MP_LEFT_EYE = (33, 133)    # 画面左眼(人右眼) → arcface[0]
_MP_RIGHT_EYE = (362, 263)  # 画面右眼(人左眼) → arcface[1]
_MP_NOSE = 1                # arcface[2]
_MP_MOUTH_LEFT = 61         # 画面左嘴角(人右嘴) → arcface[3]
_MP_MOUTH_RIGHT = 291       # 画面右嘴角(人左嘴) → arcface[4]

# 关键点索引（MediaPipe Hands 21点）
_THUMB_CMC, _THUMB_MCP, _THUMB_IP, _THUMB_TIP = 1, 2, 3, 4
_IDX_MCP2, _IDX_PIP2, _IDX_TIP2 = 5, 6, 8
_MID_MCP, _MID_PIP, _MID_TIP = 9, 10, 12
_RING_MCP, _RING_PIP, _RING_TIP = 13, 14, 16
_PINKY_MCP, _PINKY_PIP, _PINKY_TIP = 17, 18, 20


def pick_main_face(result):
    """返回最大人脸的 (u, v, 高度占比);没有人脸返回 None。(无状态版,--no-sticky 回退用)"""
    if not result.face_landmarks:
        return None
    best = None
    best_h = -1.0
    for lms in result.face_landmarks:
        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        h = max(ys) - min(ys)
        if h > best_h:
            best_h = h
            best = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, h)
    return best


def pick_main_face_yunet(faces, W: int, H: int):
    """YuNet 版:从 detect() 结果选最大脸,返回 (u, v, h_ratio) 或 None。"""
    if faces is None or len(faces) == 0:
        return None
    best = None
    best_area = -1.0
    for f in faces:
        area = f[2] * f[3]
        if area > best_area:
            best_area = area
            best = f
    if best is None:
        return None
    x, y, w, h = best[0], best[1], best[2], best[3]
    return ((x + w / 2) / W, (y + h / 2) / H, h / H)


class FaceSelector:
    """M1.5-c 跨帧粘滞选脸:锁住一张脸就跟着,除非它消失或另一张持续明显更大。
    支持 YuNet 格式: faces 是 Nx15 数组, 每行 [x,y,w,h, kp*10, conf]。"""

    def __init__(self, sticky: bool = True):
        self._sticky = sticky
        self._prev_u = None
        self._prev_v = None
        self._rival_count = 0
        self.selected_face = None  # 选中脸的原始 YuNet 行

    def reset(self):
        self._prev_u = self._prev_v = None
        self._rival_count = 0
        self.selected_face = None

    def select_yunet(self, faces, W: int, H: int) -> tuple | None:
        """从 YuNet detect() 结果选脸。返回 (u, v, h_ratio) 或 None。"""
        if faces is None or len(faces) == 0:
            self._prev_u = self._prev_v = None
            self._rival_count = 0
            self.selected_face = None
            return None

        parsed = []
        for idx, f in enumerate(faces):
            x, y, w, h = f[0], f[1], f[2], f[3]
            u = (x + w / 2) / W
            v = (y + h / 2) / H
            h_ratio = h / H
            parsed.append((u, v, h_ratio, idx))

        def _pick(p):
            self._prev_u, self._prev_v = p[0], p[1]
            self._rival_count = 0
            self.selected_face = faces[p[3]]
            return (float(p[0]), float(p[1]), float(p[2]))

        if not self._sticky or len(parsed) == 1 or self._prev_u is None:
            best = max(parsed, key=lambda f: f[2])
            return _pick(best)

        def _dist(f):
            return math.hypot(f[0] - self._prev_u, f[1] - self._prev_v)

        matched = min(parsed, key=_dist)
        if _dist(matched) > STICKY_MATCH_DIST:
            best = max(parsed, key=lambda f: f[2])
            return _pick(best)

        biggest = max(parsed, key=lambda f: f[2])
        if biggest is not matched and biggest[2] > matched[2] * STICKY_SWITCH_RATIO:
            self._rival_count += 1
            if self._rival_count >= STICKY_SWITCH_FRAMES:
                return _pick(biggest)
        else:
            self._rival_count = 0

        self._prev_u, self._prev_v = matched[0], matched[1]
        self.selected_face = faces[matched[3]]
        return (float(matched[0]), float(matched[1]), float(matched[2]))


def index_dir(lms):
    """单手 21 点 → (食指角度°[画面系:0右/-90上/+90下/±180左], 是否明显伸出, 指尖(u,v))。

    伸出判定按手的尺度相对(POINT 路由纠偏教训):原绝对阈值 seg>0.08 对远手恒 False
    (指着 1m 外物体时手指段占画面不到 0.08)→ 改 seg > 0.30×手bbox最大边,远近同标准。
    """
    mcp, pip, tip = lms[IDX_MCP], lms[IDX_PIP], lms[IDX_TIP]
    xs = [p.x for p in lms]
    ys = [p.y for p in lms]
    hand_size = max(max(xs) - min(xs), max(ys) - min(ys)) + 1e-6
    dx = tip.x - mcp.x
    dy = tip.y - mcp.y
    angle = math.degrees(math.atan2(dy, dx))
    seg = math.hypot(dx, dy)
    v1 = (pip.x - mcp.x, pip.y - mcp.y)
    v2 = (tip.x - pip.x, tip.y - pip.y)
    n1 = math.hypot(*v1) + 1e-6
    n2 = math.hypot(*v2) + 1e-6
    cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    extended = seg > 0.30 * hand_size and seg > 0.025 and cosang > 0.6
    return angle, extended, (tip.x, tip.y)


def _classify_gesture(lms):
    """21 关键点 → (fingers: int, gesture: str|None)。

    fingers: 0-5 伸直手指数
    gesture: "fist"|"point"|"two"|"three"|"four"|"five"|"ok"|None
    """
    # 拇指：尖(4) vs IP关节(3)，用 X 水平距离（左右手都用绝对距离 vs MCP距离）
    thumb_open = abs(lms[_THUMB_TIP].x - lms[_THUMB_CMC].x) > abs(lms[_THUMB_IP].x - lms[_THUMB_CMC].x)
    # 四指：尖的 Y < PIP 的 Y（图像坐标 Y 向下，更小=更高=伸直）
    margin = 0.02  # 归一化容差
    idx_open   = lms[_IDX_TIP2].y  < lms[_IDX_PIP2].y  - margin
    mid_open   = lms[_MID_TIP].y   < lms[_MID_PIP].y   - margin
    ring_open  = lms[_RING_TIP].y  < lms[_RING_PIP].y  - margin
    pinky_open = lms[_PINKY_TIP].y < lms[_PINKY_PIP].y - margin

    fingers = sum([thumb_open, idx_open, mid_open, ring_open, pinky_open])

    gesture = None
    if fingers == 0:
        gesture = "fist"
    elif fingers == 1:
        gesture = "point" if idx_open else "one"
    elif fingers == 2 and idx_open and mid_open:
        gesture = "two"
    elif fingers == 3:
        gesture = "three"
    elif fingers == 4:
        gesture = "four"
    elif fingers == 5:
        gesture = "five"

    # OK：拇指尖与食指尖靠拢，其余三指伸直
    tip_dist = math.hypot(lms[_THUMB_TIP].x - lms[_IDX_TIP2].x,
                          lms[_THUMB_TIP].y - lms[_IDX_TIP2].y)
    if tip_dist < 0.10 and mid_open and ring_open and pinky_open:
        gesture = "ok"
        fingers = 5

    return fingers, gesture


YUNET_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "face_detection_yunet_2023mar.onnx")
YUNET_SCORE_THR = 0.5
YUNET_NMS_THR = 0.3
YUNET_TOP_K = 10

# ── SCRFD(InsightFace)人脸检测:默认后端(关键点质量更稳,原生 per-face 置信)──
SCRFD_PACK = os.environ.get("SCRFD_PACK", "buffalo_sc")  # det_500m(SCRFD) + w600k_mbf
SCRFD_DET_SIZE = int(os.environ.get("SCRFD_DET_SIZE", "640"))
SCRFD_THRESH = 0.5
SCRFD_MAX_FACES = 10


_MODEL_GESTURE_MAP = {
    "Closed_Fist": "fist",
    "Open_Palm": "five",
    "Pointing_Up": "point",
    "Victory": "two",
    "Thumb_Up": "thumbup",
    "Thumb_Down": "thumbdown",
    "ILoveYou": "ily",
}

_GESTURE_FINGER_MAP = {
    "fist": 0, "point": 1, "two": 2, "five": 5,
    "thumbup": 1, "thumbdown": 1, "ily": 3,
}


def vision_worker(face_model: str, hand_model: str, frame_q, result_q,
                   gesture_model: str = None) -> None:
    """子进程入口:Face(每帧) + Hand(自适应提频)。

    人脸后端通过环境变量 FACE_BACKEND 切换:
      yunet (默认) — OpenCV YuNet, 零额外依赖, 全角度高检出率
      mediapipe    — MediaPipe FaceLandmarker VIDEO 模式(含 blendshape 表情)

    手势识别:
      gesture_model 存在时用 GestureRecognizer(模型优先 + 规则 fallback)
      否则用 HandLandmarker + 纯规则 _classify_gesture
    """
    import cv2

    face_backend = os.environ.get("FACE_BACKEND", "scrfd").lower()
    no_sticky = os.environ.get("VISION_NO_STICKY", "") == "1"
    face_sel = FaceSelector(sticky=not no_sticky)

    # ── 人脸后端初始化(默认 SCRFD/InsightFace;yunet/mediapipe 仍可选)──
    use_scrfd = (face_backend == "scrfd")
    use_yunet = (face_backend == "yunet")
    use_mp = (face_backend == "mediapipe")
    yunet = None
    face_lm = None
    scrfd = None
    yunet_size = None  # (W, H) 缓存,尺寸变化时重建

    if use_scrfd:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name=SCRFD_PACK, allowed_modules=["detection"],
                            providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(SCRFD_DET_SIZE, SCRFD_DET_SIZE),
                     det_thresh=SCRFD_THRESH)
        scrfd = _app.models.get("detection") or getattr(_app, "det_model", None)
        print(f"[vision_worker] 人脸后端: SCRFD/InsightFace ({SCRFD_PACK})", flush=True)
    elif use_yunet:
        print(f"[vision_worker] 人脸后端: YuNet ({YUNET_MODEL})", flush=True)
    else:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        face_lm = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_model),
                running_mode=mp_vision.RunningMode.VIDEO, num_faces=2,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_face_blendshapes=True))
        print(f"[vision_worker] 人脸后端: MediaPipe ({face_model})", flush=True)

    # ── 手部初始化(始终用 MediaPipe)──
    hand_lm = None
    gesture_rec = None
    use_gesture_rec = False
    try:
        if not use_mp:   # SCRFD/YuNet 人脸后端时,手部仍需 MediaPipe → 这里导入
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        if gesture_model and os.path.exists(gesture_model):
            gesture_rec = mp_vision.GestureRecognizer.create_from_options(
                mp_vision.GestureRecognizerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=gesture_model),
                    running_mode=mp_vision.RunningMode.VIDEO, num_hands=1,
                    min_hand_detection_confidence=0.7,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.4))
            use_gesture_rec = True
            print(f"[vision_worker] 手势后端: GestureRecognizer ({gesture_model})", flush=True)
        else:
            hand_lm = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                    running_mode=mp_vision.RunningMode.VIDEO, num_hands=1,
                    min_hand_detection_confidence=0.7,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.4))
            print(f"[vision_worker] 手势后端: HandLandmarker + 规则 ({hand_model})", flush=True)
    except Exception as _e:
        print(f"[vision_worker] 手部/手势模型加载失败: {_e}", flush=True)
        hand_lm = None
        gesture_rec = None
        use_gesture_rec = False

    result_q.put({"kind": "ready"})

    last_face_ts = -1
    last_hand_ts = -1
    n = 0
    hand_boost_until = -1.0

    while True:
        item = frame_q.get()
        if item is None:
            break
        if item == "sticky_reset":
            face_sel.reset()
            continue
        t_grab, rgb = item
        n += 1
        H, W = rgb.shape[:2]
        out = {"kind": "det", "t": t_grab, "face": None, "n_faces": 0, "face_ms": 0.0,
               "face_box": None, "face_kps": None, "all_faces": None, "hand": None}
        try:
            # ── 人脸检测 ──
            t0 = time.monotonic()
            if use_scrfd:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                bboxes, kpss = scrfd.detect(bgr, max_num=SCRFD_MAX_FACES, metric="default")
                out["face_ms"] = (time.monotonic() - t0) * 1000.0
                n_faces = 0 if bboxes is None else int(len(bboxes))
                out["n_faces"] = n_faces
                if n_faces > 0:
                    _all = []
                    for _i in range(n_faces):
                        x1, y1, x2, y2 = (float(bboxes[_i][0]), float(bboxes[_i][1]),
                                          float(bboxes[_i][2]), float(bboxes[_i][3]))
                        _conf = float(bboxes[_i][4])
                        _bw, _bh = x2 - x1, y2 - y1
                        _kp = ([(float(kpss[_i][k][0]), float(kpss[_i][k][1])) for k in range(5)]
                               if kpss is not None and len(kpss) > _i else None)
                        _all.append({
                            "u": float((x1 + x2) / 2 / W),
                            "v": float((y1 + y2) / 2 / H),
                            "h": float(_bh / H),
                            "box": (int(x1), int(y1), int(_bw), int(_bh)),
                            "kps": _kp,
                            "conf": _conf,
                        })
                    out["all_faces"] = _all
                    # 主脸 = 最大框(过渡:旧身份路径仍用 face/face_box;ByteTracker 接入后改用 all_faces)
                    _big = max(_all, key=lambda a: a["h"])
                    out["face"] = (_big["u"], _big["v"], _big["h"])
                    out["face_box"] = _big["box"]
                    out["face_kps"] = _big["kps"]
                else:
                    out["face"] = None
            elif use_yunet:
                if yunet is None or yunet_size != (W, H):
                    yunet = cv2.FaceDetectorYN.create(
                        YUNET_MODEL, "", (W, H), YUNET_SCORE_THR, YUNET_NMS_THR, YUNET_TOP_K)
                    yunet_size = (W, H)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                _, faces_raw = yunet.detect(bgr)
                out["face_ms"] = (time.monotonic() - t0) * 1000.0
                out["face"] = face_sel.select_yunet(faces_raw, W, H)
                out["n_faces"] = len(faces_raw) if faces_raw is not None else 0
                # all_faces: 所有检测到的脸(供主进程 DOA 选人)
                if faces_raw is not None and len(faces_raw) > 0:
                    _all = []
                    for _af in faces_raw:
                        _ax, _ay, _aw, _ah = _af[0], _af[1], _af[2], _af[3]
                        _all.append({
                            "u": float((_ax + _aw / 2) / W),
                            "v": float((_ay + _ah / 2) / H),
                            "h": float(_ah / H),
                            "box": (int(_ax), int(_ay), int(_aw), int(_ah)),
                            "kps": [(float(_af[4 + i * 2]), float(_af[5 + i * 2]))
                                    for i in range(5)],
                        })
                    out["all_faces"] = _all
                if out["face"] is not None and face_sel.selected_face is not None:
                    f = face_sel.selected_face
                    out["face_box"] = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
                    out["face_kps"] = [(float(f[4 + i * 2]), float(f[5 + i * 2]))
                                       for i in range(5)]
            else:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                last_face_ts = max(last_face_ts + 1, int(t_grab * 1000))
                fres = face_lm.detect_for_video(mp_img, last_face_ts)
                out["face_ms"] = (time.monotonic() - t0) * 1000.0
                out["face"] = face_sel.select_yunet(None, W, H)  # placeholder
                # MediaPipe 路径:用旧 select 逻辑
                if fres.face_landmarks:
                    mp_faces = []
                    _all_mp = []
                    for lms in fres.face_landmarks:
                        xs = [p.x for p in lms]
                        ys = [p.y for p in lms]
                        fh = max(ys) - min(ys)
                        fu = (min(xs) + max(xs)) / 2.0
                        fv = (min(ys) + max(ys)) / 2.0
                        mp_faces.append((fu, fv, fh))
                        pxs = [p.x * W for p in lms]
                        pys = [p.y * H for p in lms]
                        bx, by = min(pxs), min(pys)
                        bw, bh = max(pxs) - bx, max(pys) - by
                        _all_mp.append({
                            "u": float(fu), "v": float(fv), "h": float(fh),
                            "box": (int(bx), int(by), int(bw), int(bh)),
                            "kps": None,
                        })
                    best = max(mp_faces, key=lambda f: f[2])
                    out["face"] = best
                    out["n_faces"] = len(fres.face_landmarks)
                    out["all_faces"] = _all_mp
                else:
                    out["face"] = None
                    out["n_faces"] = 0
                if fres.face_blendshapes and out["face"] is not None:
                    _bs = fres.face_blendshapes[0]
                    _smile = _frown = 0.0
                    for cat in _bs:
                        if cat.category_name == "mouthSmileLeft" or cat.category_name == "mouthSmileRight":
                            _smile += cat.score * 0.5
                        elif cat.category_name == "mouthFrownLeft" or cat.category_name == "mouthFrownRight":
                            _frown += cat.score * 0.5
                    out["smile"] = _smile
                    out["frown"] = _frown
                if out["face"] is not None and fres.face_landmarks:
                    lms = fres.face_landmarks[0]
                    pxs = [p.x * W for p in lms]
                    pys = [p.y * H for p in lms]
                    bx, by = min(pxs), min(pys)
                    bw, bh = max(pxs) - bx, max(pys) - by
                    out["face_box"] = (int(bx), int(by), int(bw), int(bh))
                    def _eye_center(i1, i2):
                        return ((lms[i1].x + lms[i2].x) * 0.5 * W,
                                (lms[i1].y + lms[i2].y) * 0.5 * H)
                    out["face_kps"] = [
                        _eye_center(*_MP_LEFT_EYE),
                        _eye_center(*_MP_RIGHT_EYE),
                        (lms[_MP_NOSE].x * W, lms[_MP_NOSE].y * H),
                        (lms[_MP_MOUTH_LEFT].x * W, lms[_MP_MOUTH_LEFT].y * H),
                        (lms[_MP_MOUTH_RIGHT].x * W, lms[_MP_MOUTH_RIGHT].y * H),
                    ]

            # ── 手部检测(MediaPipe,降频)──
            _hand_ready = (hand_lm is not None or use_gesture_rec)
            if _hand_ready and (n % HAND_EVERY == 0 or t_grab <= hand_boost_until):
                mp_img_h = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                last_hand_ts = max(last_hand_ts + 1, int(t_grab * 1000) + 1)
                if use_mp:   # 仅 MediaPipe 人脸后端才与 face 时间戳耦合(同一单调时钟)
                    last_hand_ts = max(last_hand_ts, last_face_ts + 1)
                _hand_landmarks = None
                _handedness = None
                _model_gesture = None
                _model_gesture_score = 0.0
                if use_gesture_rec:
                    gres = gesture_rec.recognize_for_video(mp_img_h, last_hand_ts)
                    if gres.hand_landmarks:
                        _hand_landmarks = gres.hand_landmarks
                        _handedness = gres.handedness
                        if gres.gestures and gres.gestures[0]:
                            _model_gesture = gres.gestures[0][0].category_name
                            _model_gesture_score = gres.gestures[0][0].score
                else:
                    hres = hand_lm.detect_for_video(mp_img_h, last_hand_ts)
                    if hres.hand_landmarks:
                        _hand_landmarks = hres.hand_landmarks
                        _handedness = hres.handedness
                if not use_yunet:
                    last_face_ts = last_hand_ts
                if _hand_landmarks:
                    lms0 = _hand_landmarks[0]
                    angle, extended, tip = index_dir(lms0)
                    mapped = _MODEL_GESTURE_MAP.get(_model_gesture or "")
                    if mapped and _model_gesture_score >= 0.6:
                        gesture = mapped
                        fingers = _GESTURE_FINGER_MAP.get(gesture, -1)
                        if fingers < 0:
                            fingers, _ = _classify_gesture(lms0)
                    else:
                        fingers, gesture = _classify_gesture(lms0)
                    xs = [p.x for p in lms0]
                    ys = [p.y for p in lms0]
                    size = max(max(xs) - min(xs), max(ys) - min(ys))
                    score = _handedness[0][0].score if _handedness else 1.0
                    hand_out = {"angle": angle, "extended": extended, "tip": tip,
                                "u": (min(xs) + max(xs)) / 2.0,
                                "v": (min(ys) + max(ys)) / 2.0,
                                "size": size, "score": score,
                                "fingers": fingers, "gesture": gesture}
                    if _model_gesture:
                        hand_out["gesture_model"] = _model_gesture
                        hand_out["gesture_model_score"] = _model_gesture_score
                    out["hand"] = hand_out
                    if score >= HAND_NEAR_SCORE and size >= HAND_NEAR_SIZE:
                        hand_boost_until = t_grab + HAND_BOOST_S
            try:
                result_q.put_nowait(out)
            except Exception:
                pass
        except Exception:
            continue
