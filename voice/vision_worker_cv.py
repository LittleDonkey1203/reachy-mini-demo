# -*- coding: utf-8 -*-
"""OpenCV 后备视觉子进程 — mediapipe 不可用时(如 macOS x86_64)使用。

人脸检测: Haar Cascade (haarcascade_frontalface_default.xml)
手部检测: HSV 肤色分割 — 无需模型文件。逻辑:
  1. 转 HSV,用宽松肤色范围 mask(兼顾浅肤/深肤双段)
  2. 形态学 close 填洞,找最大连通域
  3. 当最大域占画面比 ≥ HAND_SIZE_MIN(0.04)且无人脸重叠时当作"手"
  4. 输出 angle=0(无方向估计)、extended=True(保守)、tip=中心

协议与 vision_worker.py 完全兼容:
  {"kind":"ready"}
  {"kind":"det", "t":t_grab,
   "face":(u,v,h)|None, "n_faces":int, "face_ms":float,
   "hand":{"angle":0,"extended":True,"tip":(u,v),
           "u":f,"v":f,"size":f,"score":f}|None}

局限:
  - 肤色分割对光线变化敏感,强背光/荧光灯偏色时误检会增加
  - 无手指方向,angle 恒 0、extended 恒 True → 指向功能仍不可用(与之前相同)
  - 逗它跟手(PLAYING)功能依赖 size/u/v,肤色分割已足够驱动
"""

import math
import time

# 肤色分割参数(HSV 双段,覆盖浅肤/中肤/深肤)
# 段 1: H 0-25  (橘红系)
_S1_H_LO, _S1_H_HI = 0, 25
# 段 2: H 160-180(紫红回绕系,深肤补充)
_S2_H_LO, _S2_H_HI = 160, 180
_S_LO, _S_HI = 30, 255   # 饱和度:排除灰白
_V_LO, _V_HI = 50, 255   # 亮度:排除黑色

# 最小手域面积(占画面比):低于此忽略(背景小肤色块)
HAND_SIZE_MIN = 0.04
# 伪手过滤:人脸区域向外扩展系数(避免把脸的皮肤认成手)
FACE_EXCLUDE_MARGIN = 1.5
# 手部检测每 N 帧运行一次(降低 CPU;逗它 0.30 size 手不会瞬间消失)
HAND_EVERY = 3


def vision_worker(face_model: str, hand_model: str, frame_q, result_q) -> None:
    """子进程入口(OpenCV fallback):无需 mediapipe,model 参数被忽略。"""
    import cv2
    import numpy as np

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 形态学核(填洞+平滑边缘)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    result_q.put({"kind": "ready"})

    n = 0
    last_hand = None  # 缓存上一次手部结果,跳帧时复用

    while True:
        item = frame_q.get()
        if item is None:
            break
        t_grab, rgb = item
        n += 1
        out = {"kind": "det", "t": t_grab, "face": None, "n_faces": 0,
               "face_ms": 0.0, "hand": None}
        try:
            h_px, w_px = rgb.shape[:2]
            t0 = time.monotonic()

            # ── 人脸检测(每帧) ──
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            out["face_ms"] = (time.monotonic() - t0) * 1000.0

            face_rect = None  # (x,y,w,h) 最大人脸
            if len(faces) > 0:
                areas = [fw * fh for (_, _, fw, fh) in faces]
                best = int(np.argmax(areas))
                x, y, fw, fh = faces[best]
                face_rect = (x, y, fw, fh)
                out["face"] = (
                    (x + fw * 0.5) / w_px,
                    (y + fh * 0.5) / h_px,
                    fh / h_px,
                )
                out["n_faces"] = len(faces)

            # ── 手部检测(每 HAND_EVERY 帧) ──
            if n % HAND_EVERY == 0:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

                # 双段肤色 mask
                m1 = cv2.inRange(hsv,
                                 np.array([_S1_H_LO, _S_LO, _V_LO]),
                                 np.array([_S1_H_HI, _S_HI, _V_HI]))
                m2 = cv2.inRange(hsv,
                                 np.array([_S2_H_LO, _S_LO, _V_LO]),
                                 np.array([_S2_H_HI, _S_HI, _V_HI]))
                mask = cv2.bitwise_or(m1, m2)

                # 排除人脸区域(防止把脸当手)
                if face_rect is not None:
                    fx, fy, fw, fh = face_rect
                    mx = int(fw * FACE_EXCLUDE_MARGIN)
                    my = int(fh * FACE_EXCLUDE_MARGIN)
                    x0 = max(0, fx - (mx - fw) // 2)
                    y0 = max(0, fy - (my - fh) // 2)
                    x1 = min(w_px, x0 + mx)
                    y1 = min(h_px, y0 + my)
                    mask[y0:y1, x0:x1] = 0

                # 形态学填洞
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # 找最大连通域
                n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    mask, connectivity=8)

                best_hand = None
                best_size = 0.0
                for i in range(1, n_labels):  # 0 是背景
                    area_px = stats[i, cv2.CC_STAT_AREA]
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
                    # score 用 size 线性估算(够大的手置信度高)
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
