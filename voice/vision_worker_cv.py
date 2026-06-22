# -*- coding: utf-8 -*-
"""OpenCV 后备视觉子进程 — mediapipe 不可用时(如 macOS x86_64)使用。

人脸检测: YuNet ONNX (face_detection_yunet_2023mar.onnx, ~227KB)
  - 多角度人脸检测(正面/侧脸/仰头), CPU ~5ms/帧
  - 输出 5 关键点(右眼/左眼/鼻尖/右嘴角/左嘴角), 可用于 arcface 对齐
手部检测: HSV 肤色分割(同原来)

协议与 vision_worker.py 完全兼容:
  {"kind":"ready"}
  {"kind":"det", "t":t_grab,
   "face":(u,v,h)|None, "n_faces":int, "face_ms":float,
   "face_box":(x,y,w,h)|None,         # 像素坐标, arcface 裁剪用
   "face_kps":[(x,y),...5]|None,       # 5 关键点(像素), arcface 对齐用
   "hand":{"angle":0,"extended":True,"tip":(u,v),
           "u":f,"v":f,"size":f,"score":f}|None}
"""

import os
import time

_S1_H_LO, _S1_H_HI = 0, 25
_S2_H_LO, _S2_H_HI = 160, 180
_S_LO, _S_HI = 30, 255
_V_LO, _V_HI = 50, 255

HAND_SIZE_MIN = 0.04
FACE_EXCLUDE_MARGIN = 1.5
HAND_EVERY = 3

_YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "face_detection_yunet_2023mar.onnx")


def vision_worker(face_model: str, hand_model: str, frame_q, result_q) -> None:
    """子进程入口(OpenCV fallback): YuNet 人脸 + HSV 手部。"""
    import cv2
    import numpy as np

    yunet_ok = os.path.exists(_YUNET_PATH)
    if yunet_ok:
        yunet = cv2.FaceDetectorYN.create(_YUNET_PATH, "", (320, 240),
                                          score_threshold=0.65,
                                          nms_threshold=0.3,
                                          top_k=10)
    else:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    result_q.put({"kind": "ready"})

    n = 0
    last_hand = None
    prev_w, prev_h = 0, 0

    while True:
        item = frame_q.get()
        if item is None:
            break
        if item == "sticky_reset":
            continue
        t_grab, rgb = item
        n += 1
        out = {"kind": "det", "t": t_grab, "face": None, "n_faces": 0,
               "face_ms": 0.0, "face_box": None, "face_kps": None, "hand": None}
        try:
            h_px, w_px = rgb.shape[:2]
            t0 = time.monotonic()

            face_rect = None

            if yunet_ok:
                if w_px != prev_w or h_px != prev_h:
                    yunet.setInputSize((w_px, h_px))
                    prev_w, prev_h = w_px, h_px
                _, faces = yunet.detect(rgb)
                out["face_ms"] = (time.monotonic() - t0) * 1000.0

                if faces is not None and len(faces) > 0:
                    areas = faces[:, 2] * faces[:, 3]
                    best = int(np.argmax(areas))
                    f = faces[best]
                    x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    face_rect = (x, y, fw, fh)
                    out["face"] = (
                        (x + fw * 0.5) / w_px,
                        (y + fh * 0.5) / h_px,
                        fh / h_px,
                    )
                    out["n_faces"] = len(faces)
                    out["face_box"] = (x, y, fw, fh)
                    kps = []
                    for ki in range(5):
                        kps.append((float(f[4 + ki * 2]), float(f[5 + ki * 2])))
                    out["face_kps"] = kps
            else:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                faces_haar = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                out["face_ms"] = (time.monotonic() - t0) * 1000.0

                if len(faces_haar) > 0:
                    areas = [fw * fh for (_, _, fw, fh) in faces_haar]
                    best = int(np.argmax(areas))
                    x, y, fw, fh = faces_haar[best]
                    face_rect = (x, y, fw, fh)
                    out["face"] = (
                        (x + fw * 0.5) / w_px,
                        (y + fh * 0.5) / h_px,
                        fh / h_px,
                    )
                    out["n_faces"] = len(faces_haar)
                    out["face_box"] = (x, y, fw, fh)

            # ── 手部检测(每 HAND_EVERY 帧) ──
            if n % HAND_EVERY == 0:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

                m1 = cv2.inRange(hsv,
                                 np.array([_S1_H_LO, _S_LO, _V_LO]),
                                 np.array([_S1_H_HI, _S_HI, _V_HI]))
                m2 = cv2.inRange(hsv,
                                 np.array([_S2_H_LO, _S_LO, _V_LO]),
                                 np.array([_S2_H_HI, _S_HI, _V_HI]))
                mask = cv2.bitwise_or(m1, m2)

                if face_rect is not None:
                    fx, fy, fw, fh = face_rect
                    mx = int(fw * FACE_EXCLUDE_MARGIN)
                    my = int(fh * FACE_EXCLUDE_MARGIN)
                    x0 = max(0, fx - (mx - fw) // 2)
                    y0 = max(0, fy - (my - fh) // 2)
                    x1 = min(w_px, x0 + mx)
                    y1 = min(h_px, y0 + my)
                    mask[y0:y1, x0:x1] = 0

                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    mask, connectivity=8)

                best_hand = None
                best_size = 0.0
                for i in range(1, n_labels):
                    bw = stats[i, cv2.CC_STAT_WIDTH]
                    bh = stats[i, cv2.CC_STAT_HEIGHT]
                    size = max(bw / w_px, bh / h_px)
                    if size < HAND_SIZE_MIN:
                        continue
                    if size > best_size:
                        best_size = size
                        cx, cy = centroids[i]
                        best_hand = (cx / w_px, cy / h_px, size)

                if best_hand is not None:
                    u, v, size = best_hand
                    score = min(1.0, size / 0.4 + 0.5)
                    last_hand = {
                        "angle": 0.0,
                        "extended": True,
                        "tip": (u, v),
                        "u": u, "v": v,
                        "size": size,
                        "score": score,
                    }
                else:
                    last_hand = None

            out["hand"] = last_hand

            try:
                result_q.put_nowait(out)
            except Exception:
                pass
        except Exception:
            continue
