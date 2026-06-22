# -*- coding: utf-8 -*-
"""PLAY-01 假手诊断:抓 10s 帧,把手检测(含低分小检测)画框存图,人工看是什么。

用户不伸手、保持测试坐姿 → 凡检出"手"都是误检,存图看误检落在哪(脸?衣服?背景?)。
输出 vision/debug/ghost_*.jpg(gitignored,含人像不推送)。
"""

import os
import sys
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from reachy_mini import ReachyMini

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "models", "hand_landmarker.task")
OUT = os.path.join(ROOT, "debug")
os.makedirs(OUT, exist_ok=True)
DECIMATE = 3
RUN_S = 10.0
HAND_CONF = 0.3  # 与 play01 相同的放宽置信度,复现同样的误检


def main() -> int:
    lm = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            running_mode=mp_vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=HAND_CONF,
            min_hand_presence_confidence=HAND_CONF,
            min_tracking_confidence=HAND_CONF))
    saved = 0
    last_ts = -1
    t0 = time.monotonic()
    with ReachyMini(connection_mode="localhost_only", media_backend="default") as mini:
        warm = None
        dl = time.monotonic() + 10
        while warm is None and time.monotonic() < dl:
            warm = mini.media.get_frame()
            time.sleep(0.05)
        if warm is None:
            print("no frame", flush=True)
            return 1
        print("开始 10s 诊断(请勿伸手)…", flush=True)
        end = time.monotonic() + RUN_S
        while time.monotonic() < end and saved < 4:
            frame = mini.media.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            rgb = np.ascontiguousarray(frame[::DECIMATE, ::DECIMATE, ::-1])
            last_ts = max(last_ts + 1, int((time.monotonic() - t0) * 1000))
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ts)
            if res.hand_landmarks:
                lms = res.hand_landmarks[0]
                h_img, w_img = rgb.shape[:2]
                xs = [p.x for p in lms]
                ys = [p.y for p in lms]
                size = max(max(xs) - min(xs), max(ys) - min(ys))
                bgr = rgb[:, :, ::-1].copy()
                cv2.rectangle(bgr, (int(min(xs) * w_img), int(min(ys) * h_img)),
                              (int(max(xs) * w_img), int(max(ys) * h_img)), (0, 0, 255), 2)
                for p in lms:
                    cv2.circle(bgr, (int(p.x * w_img), int(p.y * h_img)), 2, (0, 255, 0), -1)
                path = os.path.join(OUT, f"ghost_{saved}_size{size:.2f}.jpg")
                cv2.imwrite(path, bgr)
                print(f"误检!size={size:.2f} u={(min(xs)+max(xs))/2:.2f} "
                      f"v={(min(ys)+max(ys))/2:.2f} → {path}", flush=True)
                saved += 1
                time.sleep(0.8)  # 隔开存图时间点
            time.sleep(0.01)
    print(f"完成,共存 {saved} 张", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
