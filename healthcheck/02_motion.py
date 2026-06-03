# -*- coding: utf-8 -*-
"""第 2 项:动作体检 — 逐个验证头部 / 天线 / body 的运动通道。

方向约定(Reachy/Pollen 标准头部坐标系:X 朝前、Y 朝左、Z 朝上):
  - yaw 绕 Z:  + = 看左, - = 看右
  - pitch 绕 Y: + = 看下, - = 看上
该约定下序列只用到 yaw / pitch(无单独 roll),符号几何无歧义。

动作序列:reset → 点头 → 摇头 → 看左 → 看右 → 看上 → 看下 → 天线摆动 → body转动 → reset
全程 goto_target min-jerk 平滑插值,幅度保守。每个动作之间间隔 2 秒。
"""

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini

# 与 SDK 一致的初始姿态/天线位
INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]   # SDK 默认中立(约 ±10°,减少竖直抖动)

PAUSE = 2.0       # 每个动作之间的间隔,方便肉眼观察
MOVE_DUR = 1.0    # 单段平滑插值时长
HEAD_AMP = 12     # 头部看向类幅度(度)
NOD_AMP = 10      # 点头/摇头幅度(度)
BODY_AMP = 15     # body 转动幅度(度)
ANT_AMP = 0.5     # 天线摆动幅度(rad,约 28°)


def head_pose(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """按 euler 'xyz'=[roll, pitch, yaw] 构造 4x4 头部位姿矩阵。"""
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


def main() -> bool:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== 第2项:动作体检 ===")
    print("方向约定:yaw + = 看左 / - = 看右;pitch + = 看下 / - = 看上\n")
    ok = True

    # automatic_body_yaw=False:头部动作纯靠 Stewart 平台,身体不跟随,通道隔离干净
    with ReachyMini(connection_mode="localhost_only",
                    media_backend="no_media",
                    automatic_body_yaw=False) as mini:
        try:
            # ---- reset ----
            print("现在执行:reset(回到初始中立位)", flush=True)
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.5, body_yaw=0.0)
            time.sleep(PAUSE)

            # ---- 点头(pitch 上下摆,2 个循环)----
            print(f"现在执行:点头(pitch 下{NOD_AMP}° ↔ 上{NOD_AMP}°,2 次)", flush=True)
            for _ in range(2):
                mini.goto_target(head_pose(pitch_deg=+NOD_AMP), duration=0.5, body_yaw=0.0)   # 低头
                mini.goto_target(head_pose(pitch_deg=-NOD_AMP), duration=0.5, body_yaw=0.0)   # 抬头
            mini.goto_target(INIT_HEAD_POSE, duration=0.5, body_yaw=0.0)
            time.sleep(PAUSE)

            # ---- 摇头(yaw 左右摆,2 个循环)----
            print(f"现在执行:摇头(yaw 左{NOD_AMP}° ↔ 右{NOD_AMP}°,2 次)", flush=True)
            for _ in range(2):
                mini.goto_target(head_pose(yaw_deg=+NOD_AMP), duration=0.5, body_yaw=0.0)     # 看左
                mini.goto_target(head_pose(yaw_deg=-NOD_AMP), duration=0.5, body_yaw=0.0)     # 看右
            mini.goto_target(INIT_HEAD_POSE, duration=0.5, body_yaw=0.0)
            time.sleep(PAUSE)

            # ---- 看左 ----
            print(f"现在执行:看左(yaw +{HEAD_AMP}°)", flush=True)
            mini.goto_target(head_pose(yaw_deg=+HEAD_AMP), duration=MOVE_DUR, body_yaw=0.0)
            time.sleep(PAUSE)

            # ---- 看右 ----
            print(f"现在执行:看右(yaw -{HEAD_AMP}°)", flush=True)
            mini.goto_target(head_pose(yaw_deg=-HEAD_AMP), duration=MOVE_DUR, body_yaw=0.0)
            time.sleep(PAUSE)

            # ---- 看上 ----
            print(f"现在执行:看上(pitch -{HEAD_AMP}°)", flush=True)
            mini.goto_target(head_pose(pitch_deg=-HEAD_AMP), duration=MOVE_DUR, body_yaw=0.0)
            time.sleep(PAUSE)

            # ---- 看下 ----
            print(f"现在执行:看下(pitch +{HEAD_AMP}°)", flush=True)
            mini.goto_target(head_pose(pitch_deg=+HEAD_AMP), duration=MOVE_DUR, body_yaw=0.0)
            time.sleep(PAUSE)

            # 回中,准备天线动作
            mini.goto_target(INIT_HEAD_POSE, duration=MOVE_DUR, body_yaw=0.0)
            time.sleep(PAUSE)

            # ---- 天线摆动(左右交替挥动,2 个循环)----
            print(f"现在执行:天线摆动(左右交替 ±{ANT_AMP}rad,2 次)", flush=True)
            for _ in range(2):
                mini.goto_target(antennas=[+ANT_AMP, -ANT_AMP], duration=0.4, body_yaw=0.0)
                mini.goto_target(antennas=[-ANT_AMP, +ANT_AMP], duration=0.4, body_yaw=0.0)
            mini.goto_target(antennas=INIT_ANTENNAS, duration=0.5, body_yaw=0.0)
            time.sleep(PAUSE)

            # ---- body 转动(只转身体,头保持中立)----
            print(f"现在执行:body 转动(身体 yaw 左{BODY_AMP}° → 右{BODY_AMP}° → 回正)", flush=True)
            mini.goto_target(INIT_HEAD_POSE, duration=MOVE_DUR, body_yaw=np.radians(+BODY_AMP))   # 身体向左
            time.sleep(1.0)
            mini.goto_target(INIT_HEAD_POSE, duration=MOVE_DUR, body_yaw=np.radians(-BODY_AMP))   # 身体向右
            time.sleep(1.0)
            mini.goto_target(INIT_HEAD_POSE, duration=MOVE_DUR, body_yaw=0.0)                     # 回正
            time.sleep(PAUSE)

            # ---- reset ----
            print("现在执行:reset(回到初始中立位)", flush=True)
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.5, body_yaw=0.0)
            time.sleep(1.0)

        except KeyboardInterrupt:
            print("\n⚠ 收到中断,停止动作并保持当前位置。", flush=True)
            ok = False
        except Exception as e:
            print(f"\n⚠ 动作执行出错:{type(e).__name__}: {e}", flush=True)
            ok = False
        finally:
            # 把自动 body yaw 恢复成默认,避免影响后续脚本
            try:
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass

    print("=== 通过 ===" if ok else "=== 失败 ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
