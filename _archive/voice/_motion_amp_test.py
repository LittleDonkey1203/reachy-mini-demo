# -*- coding: utf-8 -*-
"""O-01a 修复1:加大动作幅度后的安全验证(纯硬件,不连 Qwen)。

新幅度:点头 +15/-10°,摇头 ±15°,看向 ±16°,歪头 roll 15°,天线 ±0.8rad。
每个动作跑 2 遍,逐个 try/except 捕获 IK/限位异常;人需在旁听异响。
"""

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:6.2f}s] {msg}", flush=True)


INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]

# ── 新幅度 ──
NOD_DOWN, NOD_UP = 15, -10
SHAKE = 15
LOOK = 16
TILT = 15
ANT = 0.8


def head_pose(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


def act_nod(m):
    for _ in range(2):
        m.goto_target(head_pose(pitch_deg=+NOD_DOWN), duration=0.35, body_yaw=0.0)
        m.goto_target(head_pose(pitch_deg=NOD_UP), duration=0.35, body_yaw=0.0)
    m.goto_target(INIT_HEAD_POSE, duration=0.35, body_yaw=0.0)


def act_shake(m):
    for _ in range(2):
        m.goto_target(head_pose(yaw_deg=+SHAKE), duration=0.35, body_yaw=0.0)
        m.goto_target(head_pose(yaw_deg=-SHAKE), duration=0.35, body_yaw=0.0)
    m.goto_target(INIT_HEAD_POSE, duration=0.35, body_yaw=0.0)


def _look(m, **kw):
    m.goto_target(head_pose(**kw), duration=0.6, body_yaw=0.0)
    time.sleep(0.6)
    m.goto_target(INIT_HEAD_POSE, duration=0.6, body_yaw=0.0)


def act_wiggle(m):
    for _ in range(2):
        m.goto_target(antennas=[+ANT, -ANT], duration=0.3, body_yaw=0.0)
        m.goto_target(antennas=[-ANT, +ANT], duration=0.3, body_yaw=0.0)
    m.goto_target(antennas=INIT_ANTENNAS, duration=0.35, body_yaw=0.0)


def act_tilt(m):
    m.goto_target(head_pose(roll_deg=TILT), duration=0.5, body_yaw=0.0)
    time.sleep(0.6)
    m.goto_target(INIT_HEAD_POSE, duration=0.5, body_yaw=0.0)


SEQ = [
    ("nod(+15/-10°)", act_nod),
    ("shake_head(±15°)", act_shake),
    ("look_left(+16°)", lambda m: _look(m, yaw_deg=+LOOK)),
    ("look_right(-16°)", lambda m: _look(m, yaw_deg=-LOOK)),
    ("look_up(-16°)", lambda m: _look(m, pitch_deg=-LOOK)),
    ("look_down(+16°)", lambda m: _look(m, pitch_deg=+LOOK)),
    ("wiggle_antennas(±0.8rad)", act_wiggle),
    ("tilt_head(roll 15°)", act_tilt),
]


def main() -> int:
    print("=== 加大幅度安全验证 ===", flush=True)
    errors = []
    with ReachyMini(connection_mode="localhost_only",
                    media_backend="no_media",
                    automatic_body_yaw=False) as mini:
        try:
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            log("WATCH_NOW(5 秒后开始全套动作,请在旁观察/听异响)")
            time.sleep(5.0)
            for rnd in (1, 2):
                log(f"── 第 {rnd} 遍 ──")
                for name, fn in SEQ:
                    log(f"现在执行:{name}")
                    try:
                        fn(mini)
                    except Exception as e:
                        errors.append((name, f"{type(e).__name__}: {e}"))
                        log(f"❌ {name} 异常:{type(e).__name__}: {e}")
                    time.sleep(0.6)
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
        finally:
            try:
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass

    print("\n========== 汇总 ==========", flush=True)
    if errors:
        for n, e in errors:
            print(f"❌ {n}:{e}", flush=True)
        print("=== 有异常,幅度需回调 ===", flush=True)
        return 1
    print("全部 16 次动作无异常(IK/限位均通过)。异响与流畅度请人耳/肉眼确认。", flush=True)
    print("=== 安全验证通过 ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
