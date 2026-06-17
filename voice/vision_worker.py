# -*- coding: utf-8 -*-
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


class FaceSelector:
    """M1.5-c 跨帧粘滞选脸:锁住一张脸就跟着,除非它消失或另一张持续明显更大。"""

    def __init__(self, sticky: bool = True):
        self._sticky = sticky
        self._prev_u = None       # 上帧选中脸的中心 u
        self._prev_v = None       # 上帧选中脸的中心 v
        self._rival_count = 0     # 另一张脸"明显更大"连续帧计数

    def reset(self):
        """清除粘滞状态(切换对话对象时调用)。"""
        self._prev_u = self._prev_v = None
        self._rival_count = 0

    def select(self, result) -> tuple | None:
        """从 MediaPipe 结果选脸。返回 (u, v, h) 或 None。"""
        if not result.face_landmarks:
            self._prev_u = self._prev_v = None
            self._rival_count = 0
            return None

        faces = []
        for lms in result.face_landmarks:
            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            h = max(ys) - min(ys)
            u = (min(xs) + max(xs)) / 2.0
            v = (min(ys) + max(ys)) / 2.0
            faces.append((u, v, h))

        if not self._sticky or len(faces) == 1 or self._prev_u is None:
            best = max(faces, key=lambda f: f[2])
            self._prev_u, self._prev_v = best[0], best[1]
            self._rival_count = 0
            return best

        # 多脸 + 有前帧 → 找距离上帧最近的(粘住)
        def _dist(f):
            return math.hypot(f[0] - self._prev_u, f[1] - self._prev_v)

        matched = min(faces, key=_dist)
        if _dist(matched) > STICKY_MATCH_DIST:
            # 所有脸都离上帧太远(人移走了?),退回 argmax
            best = max(faces, key=lambda f: f[2])
            self._prev_u, self._prev_v = best[0], best[1]
            self._rival_count = 0
            return best

        # 检查是否有另一张脸持续明显更大(该切换了)
        biggest = max(faces, key=lambda f: f[2])
        if biggest is not matched and biggest[2] > matched[2] * STICKY_SWITCH_RATIO:
            self._rival_count += 1
            if self._rival_count >= STICKY_SWITCH_FRAMES:
                self._prev_u, self._prev_v = biggest[0], biggest[1]
                self._rival_count = 0
                return biggest
        else:
            self._rival_count = 0

        self._prev_u, self._prev_v = matched[0], matched[1]
        return matched


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


def vision_worker(face_model: str, hand_model: str, frame_q, result_q) -> None:
    """子进程入口:Face 每帧 + Hand 降频检测 frame_q 里的最新帧。"""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    no_sticky = os.environ.get("VISION_NO_STICKY", "") == "1"
    face_sel = FaceSelector(sticky=not no_sticky)

    face_lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_model),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=2))
    hand_lm = None
    try:
        hand_lm = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                running_mode=mp_vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=0.5,   # 初捕获要真(防背景误检泛滥)
                min_hand_presence_confidence=0.4,    # 跟踪期放宽:远手/运动模糊不丢锁
                min_tracking_confidence=0.3))        # (standalone PLAY-01 实测调校)
    except Exception as _e:
        print(f"[vision_worker] HandLandmarker 加载失败: {_e}", flush=True)
        hand_lm = None  # 手模型缺失也不影响人脸跟随
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
        out = {"kind": "det", "t": t_grab, "face": None, "n_faces": 0, "face_ms": 0.0,
               "hand": None}
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.monotonic()
            last_face_ts = max(last_face_ts + 1, int(t_grab * 1000))
            fres = face_lm.detect_for_video(mp_img, last_face_ts)
            out["face_ms"] = (time.monotonic() - t0) * 1000.0
            out["face"] = face_sel.select(fres)
            out["n_faces"] = len(fres.face_landmarks) if fres.face_landmarks else 0

            if hand_lm is not None and (n % HAND_EVERY == 0 or t_grab <= hand_boost_until):
                # 手部检测要求时间戳严格 > 上次,且与 face 流不冲突 → 用独立递增计数
                last_hand_ts = max(last_hand_ts + 1, last_face_ts + 1)
                hres = hand_lm.detect_for_video(mp_img, last_hand_ts)
                last_face_ts = last_hand_ts  # 两个检测器共用单调时钟,继续递增
                if hres.hand_landmarks:
                    lms0 = hres.hand_landmarks[0]
                    angle, extended, tip = index_dir(lms0)
                    fingers, gesture = _classify_gesture(lms0)
                    xs = [p.x for p in lms0]
                    ys = [p.y for p in lms0]
                    size = max(max(xs) - min(xs), max(ys) - min(ys))
                    score = hres.handedness[0][0].score if hres.handedness else 1.0
                    out["hand"] = {"angle": angle, "extended": extended, "tip": tip,
                                   "u": (min(xs) + max(xs)) / 2.0,
                                   "v": (min(ys) + max(ys)) / 2.0,
                                   "size": size, "score": score,
                                   "fingers": fingers, "gesture": gesture}
                    if score >= HAND_NEAR_SCORE and size >= HAND_NEAR_SIZE:
                        hand_boost_until = t_grab + HAND_BOOST_S  # 近手 → 提频跟手
            try:
                result_q.put_nowait(out)
            except Exception:
                pass
        except Exception:
            continue
