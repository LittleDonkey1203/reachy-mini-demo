# -*- coding: utf-8 -*-
"""动作库：头部姿态矩阵工具 + act_* 手势动作 + ACTIONS 字典。

所有 act_* 签名统一为 (mini, base_yaw, base_pitch, body_yaw_deg)，
以当前跟随姿态为基准做动作、做完回基准。
body_yaw 必须传当前身体朝向（传 0 会把转过去的身体拽回正前）。
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini

from voice.config import GES_PITCH_BOX, GES_YAW_BOX

# ── 常量 ──
INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]


# ── 姿态矩阵 ──
def head_pose(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


def gpose(yaw: float, pitch: float, body: float, roll: float = 0.0) -> np.ndarray:
    """手势姿态 = 跟随基准 + 手势 offset;yaw 裁剪到身体±箱(颈不顶限),pitch 绝对裁剪。"""
    return head_pose(
        pitch_deg=float(np.clip(pitch, -GES_PITCH_BOX, GES_PITCH_BOX)),
        yaw_deg=float(np.clip(yaw, body - GES_YAW_BOX, body + GES_YAW_BOX)),
        roll_deg=roll,
    )


# ── 手势动作 ──
def act_nod(m: ReachyMini, by: float, bp: float, body: float) -> None:
    brad = math.radians(body)
    for _ in range(2):
        m.goto_target(gpose(by, bp + 15, body), duration=0.35, body_yaw=brad)
        m.goto_target(gpose(by, bp - 10, body), duration=0.35, body_yaw=brad)
    m.goto_target(gpose(by, bp, body), duration=0.35, body_yaw=brad)


def act_shake(m: ReachyMini, by: float, bp: float, body: float) -> None:
    brad = math.radians(body)
    for _ in range(2):
        m.goto_target(gpose(by + 15, bp, body), duration=0.35, body_yaw=brad)
        m.goto_target(gpose(by - 15, bp, body), duration=0.35, body_yaw=brad)
    m.goto_target(gpose(by, bp, body), duration=0.35, body_yaw=brad)


def _look(m: ReachyMini, by: float, bp: float, body: float,
          yaw_off: float = 0.0, pitch_off: float = 0.0) -> None:
    """看向某方向(相对身体正前的偏向),看完回跟随基准。"""
    brad = math.radians(body)
    m.goto_target(gpose(body + yaw_off, pitch_off, body), duration=0.6, body_yaw=brad)
    time.sleep(0.8)
    m.goto_target(gpose(by, bp, body), duration=0.6, body_yaw=brad)


def act_wiggle(m: ReachyMini, by: float, bp: float, body: float) -> None:
    brad = math.radians(body)
    for _ in range(2):
        m.goto_target(antennas=[+0.8, -0.8], duration=0.3, body_yaw=brad)
        m.goto_target(antennas=[-0.8, +0.8], duration=0.3, body_yaw=brad)
    m.goto_target(antennas=INIT_ANTENNAS, duration=0.35, body_yaw=brad)


def act_tilt(m: ReachyMini, by: float, bp: float, body: float) -> None:
    brad = math.radians(body)
    m.goto_target(gpose(by, bp, body, roll=15), duration=0.5, body_yaw=brad)
    time.sleep(0.8)
    m.goto_target(gpose(by, bp, body), duration=0.5, body_yaw=brad)


# ── 注视微动作:小幅度、慢速、像小动物的无意识小动作 ──

def act_micro_tilt(m: ReachyMini, by: float, bp: float, body: float) -> None:
    """微歪头:roll 5°,轻轻歪一下就回来。"""
    brad = math.radians(body)
    m.goto_target(gpose(by, bp, body, roll=5), duration=0.6, body_yaw=brad)
    time.sleep(0.4)
    m.goto_target(gpose(by, bp, body), duration=0.6, body_yaw=brad)


def act_micro_nod(m: ReachyMini, by: float, bp: float, body: float) -> None:
    """微点头:pitch 只动 5°,像不经意地点了一下。"""
    brad = math.radians(body)
    m.goto_target(gpose(by, bp + 5, body), duration=0.5, body_yaw=brad)
    m.goto_target(gpose(by, bp, body), duration=0.5, body_yaw=brad)


def act_micro_wiggle(m: ReachyMini, by: float, bp: float, body: float) -> None:
    """微摆天线:幅度减半,只摆一次,像耳朵抖了一下。"""
    brad = math.radians(body)
    m.goto_target(antennas=[+0.35, -0.35], duration=0.35, body_yaw=brad)
    m.goto_target(antennas=[-0.25, +0.25], duration=0.35, body_yaw=brad)
    m.goto_target(antennas=INIT_ANTENNAS, duration=0.4, body_yaw=brad)


def act_micro_perk(m: ReachyMini, by: float, bp: float, body: float) -> None:
    """竖天线:两根天线同时轻轻竖起,像小动物竖耳朵。"""
    brad = math.radians(body)
    m.goto_target(antennas=[-0.5, -0.5], duration=0.4, body_yaw=brad)
    time.sleep(0.3)
    m.goto_target(antennas=INIT_ANTENNAS, duration=0.5, body_yaw=brad)


def act_micro_glance(m: ReachyMini, by: float, bp: float, body: float) -> None:
    """微瞥:yaw 偏 4° 再回来,像不经意地瞥了一眼。"""
    brad = math.radians(body)
    m.goto_target(gpose(by + 4, bp - 2, body), duration=0.5, body_yaw=brad)
    time.sleep(0.2)
    m.goto_target(gpose(by, bp, body), duration=0.5, body_yaw=brad)


# ── 动作分发字典 ──
ACTIONS = {
    "nod": act_nod,
    "shake_head": act_shake,
    "look_left": lambda m, by, bp, body: _look(m, by, bp, body, yaw_off=+16),
    "look_right": lambda m, by, bp, body: _look(m, by, bp, body, yaw_off=-16),
    "look_up": lambda m, by, bp, body: _look(m, by, bp, body, pitch_off=-16),
    "look_down": lambda m, by, bp, body: _look(m, by, bp, body, pitch_off=+16),
    "wiggle_antennas": act_wiggle,
    "tilt_head": act_tilt,
    "micro_tilt": act_micro_tilt,
    "micro_nod": act_micro_nod,
    "micro_wiggle": act_micro_wiggle,
    "micro_perk": act_micro_perk,
    "micro_glance": act_micro_glance,
}
