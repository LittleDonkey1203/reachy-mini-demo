# -*- coding: utf-8 -*-
"""POINT-02-a:手指方向检测(独立测,只打印角度,不转头)。

MediaPipe Hand Landmarker(和 Face 同源)→ 取食指根 MCP(landmark 5)+ 指尖 TIP(8)
→ 算指尖相对指根的 2D 方向向量(画面坐标系)→ 角度 + 方向标签。

画面坐标:x 右、y 下。角度 atan2(dy, dx):
  0°=指向画面右,90°=指向下,±180°=左,-90°=上。
方向标签按 8 向量化(右/右下/下/左下/左/左上/上/右上)。
食指是否伸出:用 TIP/PIP/MCP 共线 + 远离手腕 粗判,过滤握拳误读。

运行 60s 自动结束:在镜头前用食指指不同方向,看打印角度/方向跟不跟手。
$env:PYTHONUTF8=1; python voice\\_point_dir_test.py
"""

import math
import os
import sys
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from reachy_mini import ReachyMini

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "vision", "models", "hand_landmarker.task")
RUN_SECONDS = 60.0
DECIMATE = 3
T0 = time.monotonic()

WRIST, IDX_MCP, IDX_PIP, IDX_TIP = 0, 5, 6, 8


def log(m: str) -> None:
    print(f"[t+{time.monotonic() - T0:6.2f}s] {m}", flush=True)


def dir_label(angle: float) -> str:
    """画面系角度(0右/90下/-90上/±180左)→ 8 向中文标签(从用户视角)。"""
    # 用户视角:画面右是用户的左手边,但这里先按"画面方向"报,转头映射留给 b
    labels = ["右", "右下", "下", "左下", "左", "左上", "上", "右上"]  # 0=右,顺时针(y 向下)
    a = (angle + 360) % 360
    return labels[int((a + 22.5) / 45) % 8]


def index_dir(lms):
    """返回 (角度°, 食指是否明显伸出, tip坐标);lms 为单手 21 点。"""
    mcp = lms[IDX_MCP]
    tip = lms[IDX_TIP]
    pip = lms[IDX_PIP]
    wrist = lms[WRIST]
    dx = tip.x - mcp.x
    dy = tip.y - mcp.y
    angle = math.degrees(math.atan2(dy, dx))
    # 伸出判定:tip 离 mcp 的距离 > tip 离 wrist 距离的一定比例,且 tip-pip-mcp 大致共线
    seg = math.hypot(dx, dy)
    v1 = (pip.x - mcp.x, pip.y - mcp.y)
    v2 = (tip.x - pip.x, tip.y - pip.y)
    n1 = math.hypot(*v1) + 1e-6
    n2 = math.hypot(*v2) + 1e-6
    cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)  # 两段方向一致性
    extended = seg > 0.08 and cosang > 0.6
    return angle, extended, (tip.x, tip.y)


def main() -> int:
    if not os.path.exists(MODEL):
        log(f"❌ 手部模型不存在:{MODEL}")
        return 1
    print("=== POINT-02-a:食指方向检测(只打印,不转头)===", flush=True)
    landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
    )
    log("✅ MediaPipe HandLandmarker 就绪")

    last_ts_ms = -1
    n_det = n_hand = n_point = 0
    infer_acc: list[float] = []
    stat_t = time.monotonic()
    last_print = 0.0

    with ReachyMini(connection_mode="localhost_only", media_backend="default",
                    automatic_body_yaw=False) as mini:
        try:
            warm = None
            dl = time.monotonic() + 10
            while warm is None and time.monotonic() < dl:
                warm = mini.media.get_frame()
                if warm is None:
                    time.sleep(0.05)
            if warm is None:
                log("❌ 摄像头无帧")
                return 1
            log(f"READY_FOR_HAND(开始 {RUN_SECONDS:.0f}s:用食指指不同方向)")
            end = time.monotonic() + RUN_SECONDS
            while time.monotonic() < end:
                frame = mini.media.get_frame()
                now = time.monotonic()
                if frame is None:
                    time.sleep(0.005)
                    continue
                rgb = np.ascontiguousarray(frame[::DECIMATE, ::DECIMATE, ::-1])
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t_inf = time.monotonic()
                last_ts_ms = max(last_ts_ms + 1, int((now - T0) * 1000))
                result = landmarker.detect_for_video(mp_img, last_ts_ms)
                infer_acc.append((time.monotonic() - t_inf) * 1000)
                n_det += 1

                if result.hand_landmarks:
                    n_hand += 1
                    lms = result.hand_landmarks[0]
                    angle, extended, (tx, ty) = index_dir(lms)
                    if extended:
                        n_point += 1
                    # 限频打印(4Hz),避免刷屏
                    if now - last_print >= 0.25:
                        flag = "食指伸出 ✋→" if extended else "未明显伸出  "
                        log(f"🖐 检测到手|{flag}|食指角度 {angle:+6.1f}°({dir_label(angle)})"
                            f"|指尖 u={tx:.2f} v={ty:.2f}")
                        last_print = now

                if now - stat_t >= 10.0:
                    fps = n_det / (now - stat_t)
                    log(f"—— 统计:检测 {fps:.1f}fps|有手 {100*n_hand/max(1,n_det):.0f}%|"
                        f"食指伸出 {100*n_point/max(1,n_det):.0f}%|推理 {np.mean(infer_acc):.1f}ms")
                    stat_t = now
                    n_det = n_hand = n_point = 0
                    infer_acc = []
        except KeyboardInterrupt:
            log("Ctrl+C 提前结束")
        finally:
            try:
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass
    print("=== 完成,等待肉眼确认:检测到手?食指方向对不对? ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
