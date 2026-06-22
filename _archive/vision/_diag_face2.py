# -*- coding: utf-8 -*-
"""VIS-01 诊断2:① 标准人脸图能否检出(验证 MediaPipe 安装);② 实拍当前帧存档。"""

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import sys
import time

import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from reachy_mini import ReachyMini

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "face_landmarker.task")
DBG = os.path.join(HERE, "debug")


def log(m):
    print(m, flush=True)


def detect_np(landmarker, rgb: np.ndarray, tag: str) -> None:
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    n = len(res.face_landmarks) if res.face_landmarks else 0
    log(f"  {tag}:{'✅ 检出 ' + str(n) + ' 张脸' if n else '❌ 0 张'}")


def main() -> int:
    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=2,
        )
    )

    # ① 标准人脸测试图(MediaPipe 官方示例同款)
    p = Image.open(os.path.join(DBG, "portrait_test.jpg")).convert("RGB")
    log(f"标准测试图:{p.size}")
    detect_np(landmarker, np.asarray(p).copy(), "portrait 原尺寸")
    detect_np(landmarker, np.asarray(p.resize((640, int(640 * p.size[1] / p.size[0])))).copy(), "portrait 缩到640宽")

    # ② 实拍当前帧
    log("连接机器人抓实拍帧…")
    with ReachyMini(connection_mode="localhost_only", media_backend="default") as mini:
        frame = None
        wdl = time.monotonic() + 10.0
        while frame is None and time.monotonic() < wdl:
            frame = mini.media.get_frame()
            if frame is None:
                time.sleep(0.05)
        if frame is None:
            log("❌ 无帧")
            return 1
        log("READY_LIVE(3 秒后抓帧,请正对镜头)")
        time.sleep(3.0)
        for _ in range(5):
            f = mini.media.get_frame()
            if f is not None:
                frame = f
            time.sleep(0.03)

    rgb_full = np.ascontiguousarray(frame[:, :, ::-1])
    Image.fromarray(rgb_full[::3, ::3]).save(os.path.join(DBG, "diag2_live.jpg"), quality=90)
    log(f"实拍帧已存:{DBG}\\diag2_live.jpg")
    detect_np(landmarker, np.ascontiguousarray(rgb_full[::3, ::3]), "实拍 640×360")
    detect_np(landmarker, rgb_full, "实拍 1920×1080 全尺寸")
    return 0


if __name__ == "__main__":
    sys.exit(main())
