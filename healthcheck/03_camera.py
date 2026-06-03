# -*- coding: utf-8 -*-
"""第 3 项:视频流体检 — 用 SDK 的 media.get_frame() 抓帧,不另开 cv2.VideoCapture。

抓 60 帧,打印每帧分辨率/数据类型/帧间隔,统计真实 FPS,
并保存第 1/30/60 帧为 jpg 供肉眼确认。退出时上下文管理器自动释放摄像头。
"""

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import sys
import time
import numpy as np
from PIL import Image
from reachy_mini import ReachyMini

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
N_FRAMES = 60
SAVE_AT = {1, 30, 60}            # 1-indexed:保存这几帧
WARMUP_TIMEOUT = 10.0           # 等首帧的最长时间(秒)


def save_bgr_as_jpg(frame_bgr: np.ndarray, path: str) -> None:
    """get_frame() 返回的是 BGR,转 RGB 再用 Pillow 存 jpg。"""
    rgb = frame_bgr[:, :, ::-1]   # BGR -> RGB
    Image.fromarray(np.ascontiguousarray(rgb)).save(path, quality=92)


def main() -> bool:
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== 第3项:视频流体检 ===")
    ok = True

    # 默认 backend(LOCAL):走 daemon 的本地 IPC 摄像头,不抢占设备
    with ReachyMini(connection_mode="localhost_only", media_backend="default") as mini:
        # --- 预热:等第一帧出来(GStreamer 管线启动需要一点时间)---
        print("预热中,等待首帧 ...", flush=True)
        t_warm = time.time()
        first = None
        while time.time() - t_warm < WARMUP_TIMEOUT:
            first = mini.media.get_frame()
            if first is not None:
                break
            time.sleep(0.01)
        if first is None:
            print("⚠ 预热超时:没拿到任何帧,摄像头可能没就绪")
            print("=== 失败 ===")
            return False

        h, w = first.shape[0], first.shape[1]
        print(f"首帧就绪:分辨率 {w}x{h},dtype={first.dtype},shape={first.shape}\n")

        # --- 正式抓 60 帧,统计 FPS ---
        frames = {}            # idx -> frame,用于保存
        none_count = 0
        t_prev = None
        t_start = None
        got = 0
        print(f"开始连续抓 {N_FRAMES} 帧:", flush=True)
        while got < N_FRAMES:
            frame = mini.media.get_frame()
            if frame is None:
                none_count += 1
                continue
            now = time.time()
            if t_start is None:
                t_start = now
            got += 1
            dt_ms = (now - t_prev) * 1000 if t_prev is not None else 0.0
            t_prev = now

            # 每帧:分辨率、数据类型、与上一帧间隔
            print(f"  帧 {got:>2}: {frame.shape[1]}x{frame.shape[0]}  dtype={frame.dtype}  "
                  f"帧间隔={dt_ms:6.1f} ms", flush=True)

            if got in SAVE_AT:
                frames[got] = frame.copy()

        t_end = time.time()
        elapsed = t_end - t_start
        fps = (got - 1) / elapsed if elapsed > 0 else 0.0   # got-1 个间隔

        # --- 保存指定帧 ---
        print("\n保存抽样帧:")
        for idx in sorted(frames):
            path = os.path.join(OUT_DIR, f"camera_frame_{idx:02d}.jpg")
            save_bgr_as_jpg(frames[idx], path)
            print(f"  第 {idx} 帧 -> {path}")

        # --- 显著打印 FPS ---
        print("\n" + "=" * 44)
        print(f"   实测抓帧帧率 FPS = {fps:.2f}")
        print(f"   ({got} 帧有效 / {elapsed:.2f} 秒;期间 None 次数={none_count})")
        print("=" * 44)

        if got < N_FRAMES:
            ok = False

    print("\n摄像头资源已通过上下文管理器释放。")
    print("=== 通过 ===" if ok else "=== 失败 ===")
    print("\n请肉眼确认 output 下的 3 张 jpg 画面是否正常(清晰、不偏色、不全黑/全花)。")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
