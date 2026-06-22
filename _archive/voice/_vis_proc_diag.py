# -*- coding: utf-8 -*-
"""TRACK-FIX 零检出诊断:同一帧走两条路检测,定位"人不在画面"还是"进程管线坏了"。

1. 抓帧存 vision/debug/procdiag.jpg(人工看画面里有没有人)
2. 同一帧:A. 主进程内直接 MediaPipe 检测(老路,F-01 验证过)
          B. 经 multiprocessing.Queue 传给 vision_worker 子进程检测(新路)
3. 对比 A/B 结果是否一致 → 一致=画面问题;A有B无=进程传输坏数据
"""

import multiprocessing
import os
import sys
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "vision", "models", "face_landmarker.task")
DBG = os.path.join(ROOT, "vision", "debug")


def main() -> int:
    from reachy_mini import ReachyMini
    from vision_worker import vision_worker, pick_main_face

    os.makedirs(DBG, exist_ok=True)
    print("=== 视觉进程化零检出诊断 ===", flush=True)

    with ReachyMini(connection_mode="localhost_only", media_backend="default",
                    automatic_body_yaw=False) as mini:
        mini.media.start_recording()  # 与正式脚本同条件(录音管线共存)
        warm = None
        dl = time.monotonic() + 10
        while warm is None and time.monotonic() < dl:
            warm = mini.media.get_frame()
            if warm is None:
                time.sleep(0.05)
        if warm is None:
            print("❌ 无帧", flush=True)
            return 1
        time.sleep(2.0)  # 稳两秒再抓正式帧
        frame = mini.media.get_frame()
        mini.media.stop_recording()

    rgb = np.ascontiguousarray(frame[::3, ::3, ::-1])
    Image.fromarray(rgb).save(os.path.join(DBG, "procdiag.jpg"), "JPEG", quality=90)
    print(f"帧已存 vision/debug/procdiag.jpg,降采样 {rgb.shape},均值亮度 {rgb.mean():.0f}", flush=True)

    # A 路:主进程内直接检测
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=2))
    res_a = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), 1)
    face_a = pick_main_face(res_a)
    print(f"A 路(主进程直检):{face_a}", flush=True)

    # B 路:经队列给子进程
    fq: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
    rq: multiprocessing.Queue = multiprocessing.Queue(maxsize=8)
    p = multiprocessing.Process(target=vision_worker, args=(MODEL, fq, rq), daemon=True)
    p.start()
    ready = rq.get(timeout=30)
    print(f"子进程 ready:{ready[0]}", flush=True)
    fq.put((time.monotonic(), rgb))
    out = rq.get(timeout=10)
    print(f"B 路(子进程队列):t={out[0]:.1f} u={out[1]} v={out[2]} n_faces={out[4]} infer={out[5]:.1f}ms", flush=True)
    fq.put(None)

    a_hit = face_a is not None
    b_hit = out[1] is not None
    if a_hit == b_hit:
        print(f"=== A/B 一致(都{'检出' if a_hit else '未检出'})→ "
              f"{'管线正常' if a_hit else '管线正常,是画面里没有正脸——看 procdiag.jpg'} ===", flush=True)
    else:
        print("=== ❌ A/B 不一致 → 进程传输路径有问题 ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
