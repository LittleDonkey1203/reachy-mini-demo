# -*- coding: utf-8 -*-
"""视觉子进程(TRACK-FIX + POINT-02):Face(每帧,跟随用)+ Hand(降频,指向用)。

独立进程 = 独立 GIL:六线程融合后视觉循环曾被饿到 41→19fps,挪进程后真并行。
Face Landmarker 每帧跑(人脸跟随要实时);Hand Landmarker 每 HAND_EVERY 帧跑一次
(指向是偶发事件,~7Hz 足够,省 CPU 不拖累跟随)。

协议(result_q,dict):
  一次 {"kind":"ready"}
  每帧 {"kind":"det", "t":t_grab,
        "face":(u,v,h)|None, "n_faces":int, "face_ms":float,
        "hand":{"angle":deg,"extended":bool,"tip":(u,v)}|None }   # hand 仅在跑了的帧带
MediaPipe VIDEO 模式时间戳严格递增(CALIBRATION §9 坑)。
"""

import math
import time

WRIST, IDX_MCP, IDX_PIP, IDX_TIP = 0, 5, 6, 8
HAND_EVERY = 4  # 每 N 帧跑一次手部检测


def pick_main_face(result):
    """返回最大人脸的 (u, v, 高度占比);没有人脸返回 None。"""
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


def index_dir(lms):
    """单手 21 点 → (食指角度°[画面系:0右/-90上/+90下/±180左], 是否明显伸出, 指尖(u,v))。"""
    mcp, pip, tip = lms[IDX_MCP], lms[IDX_PIP], lms[IDX_TIP]
    dx = tip.x - mcp.x
    dy = tip.y - mcp.y
    angle = math.degrees(math.atan2(dy, dx))
    seg = math.hypot(dx, dy)
    v1 = (pip.x - mcp.x, pip.y - mcp.y)
    v2 = (tip.x - pip.x, tip.y - pip.y)
    n1 = math.hypot(*v1) + 1e-6
    n2 = math.hypot(*v2) + 1e-6
    cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    extended = seg > 0.08 and cosang > 0.6
    return angle, extended, (tip.x, tip.y)


def vision_worker(face_model: str, hand_model: str, frame_q, result_q) -> None:
    """子进程入口:Face 每帧 + Hand 降频检测 frame_q 里的最新帧。"""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    face_lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_model),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=2))
    hand_lm = None
    try:
        hand_lm = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                running_mode=mp_vision.RunningMode.VIDEO, num_hands=1))
    except Exception:
        hand_lm = None  # 手模型缺失也不影响人脸跟随
    result_q.put({"kind": "ready"})

    last_face_ts = -1
    last_hand_ts = -1
    n = 0
    while True:
        item = frame_q.get()
        if item is None:
            break
        t_grab, rgb = item
        n += 1
        out = {"kind": "det", "t": t_grab, "face": None, "n_faces": 0, "face_ms": 0.0, "hand": None}
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.monotonic()
            last_face_ts = max(last_face_ts + 1, int(t_grab * 1000))
            fres = face_lm.detect_for_video(mp_img, last_face_ts)
            out["face_ms"] = (time.monotonic() - t0) * 1000.0
            face = pick_main_face(fres)
            out["face"] = face
            out["n_faces"] = len(fres.face_landmarks) if fres.face_landmarks else 0

            if hand_lm is not None and (n % HAND_EVERY == 0):
                # 手部检测要求时间戳严格 > 上次,且与 face 流不冲突 → 用独立递增计数
                last_hand_ts = max(last_hand_ts + 1, last_face_ts + 1)
                hres = hand_lm.detect_for_video(mp_img, last_hand_ts)
                last_face_ts = last_hand_ts  # 两个检测器共用单调时钟,继续递增
                if hres.hand_landmarks:
                    angle, extended, tip = index_dir(hres.hand_landmarks[0])
                    out["hand"] = {"angle": angle, "extended": extended, "tip": tip}
            try:
                result_q.put_nowait(out)
            except Exception:
                pass
        except Exception:
            continue
