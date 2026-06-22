# -*- coding: utf-8 -*-
"""VIS-01 诊断:为什么 MediaPipe 零检出。

抓当前帧 → 打印亮度统计 → 存 jpg 人工核对 →
同一帧 4 个变体跑 FaceLandmarker(IMAGE 模式)看哪个能检出:
  A) 抽样降采样 + BGR→RGB(主脚本的做法)
  B) 抽样降采样,不换通道(BGR 直喂)
  C) PIL 高质量缩放 + BGR→RGB(排除抽样锯齿)
  D) A 旋转 180°(排除相机倒装)
"""

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


def main() -> int:
    os.makedirs(DBG, exist_ok=True)
    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=2,
        )
    )
    log("MediaPipe 就绪(IMAGE 模式),连接机器人…")
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
        log("READY_FACE_DIAG(5 秒后抓正式帧,请坐在镜头正前方不动)")
        time.sleep(5.0)
        for _ in range(5):  # 取最新
            f = mini.media.get_frame()
            if f is not None:
                frame = f
            time.sleep(0.03)

    log(f"帧:shape={frame.shape} dtype={frame.dtype} 亮度 mean={frame.mean():.1f} std={frame.std():.1f}"
        f"(全黑≈0;正常室内 60~140)")

    dec = np.ascontiguousarray(frame[::3, ::3])            # BGR 640×360
    a = np.ascontiguousarray(dec[:, :, ::-1])              # RGB
    b = dec                                                # BGR 直喂
    pil_full = Image.fromarray(frame[:, :, ::-1])          # RGB 全幅
    c = np.asarray(pil_full.resize((640, 360))).copy()     # 高质量缩放 RGB
    d = np.ascontiguousarray(a[::-1, ::-1])                # 旋转 180°

    Image.fromarray(a).save(os.path.join(DBG, "diag_A_decimate_rgb.jpg"), quality=90)
    Image.fromarray(c).save(os.path.join(DBG, "diag_C_pilresize_rgb.jpg"), quality=90)
    log(f"已存人工核对图:{DBG}\\diag_A_decimate_rgb.jpg / diag_C_pilresize_rgb.jpg")

    for name, img in (("A 抽样+RGB(主脚本)", a), ("B 抽样+BGR直喂", b),
                      ("C PIL缩放+RGB", c), ("D A旋转180°", d)):
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
        n = len(res.face_landmarks) if res.face_landmarks else 0
        if n:
            xs = [p.x for p in res.face_landmarks[0]]
            ys = [p.y for p in res.face_landmarks[0]]
            log(f"  {name}:✅ 检出 {n} 张脸,中心 u={(min(xs)+max(xs))/2:.2f} v={(min(ys)+max(ys))/2:.2f}")
        else:
            log(f"  {name}:❌ 0 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
