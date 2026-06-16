# -*- coding: utf-8 -*-
"""OpenCV Haar Cascade 人脸检测 — mediapipe 不可用时(如 macOS x86_64)的后备视觉子进程。

协议与 vision_worker.py 完全兼容:
  {"kind":"ready"}   — 初始化完成
  {"kind":"det", "t":t_grab, "face":(u,v,h)|None, "n_faces":int, "face_ms":float, "hand":None}

不检测手部(hand 恒 None),face 使用 haarcascade_frontalface_default.xml。
"""

import time


def vision_worker(face_model: str, hand_model: str, frame_q, result_q) -> None:
    """子进程入口(OpenCV fallback):无需 mediapipe,model 参数被忽略。"""
    import cv2

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    result_q.put({"kind": "ready"})

    while True:
        item = frame_q.get()
        if item is None:
            break
        t_grab, rgb = item
        out = {"kind": "det", "t": t_grab, "face": None, "n_faces": 0, "face_ms": 0.0, "hand": None}
        try:
            t0 = time.monotonic()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            h_px, w_px = gray.shape[:2]
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            out["face_ms"] = (time.monotonic() - t0) * 1000.0
            if len(faces) > 0:
                areas = [w * h for (_, _, w, h) in faces]
                x, y, w, h = faces[int(areas.index(max(areas)))]
                out["face"] = (
                    (x + w * 0.5) / w_px,
                    (y + h * 0.5) / h_px,
                    h / h_px,
                )
                out["n_faces"] = len(faces)
            try:
                result_q.put_nowait(out)
            except Exception:
                pass
        except Exception:
            continue
