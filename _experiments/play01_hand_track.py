# -*- coding: utf-8 -*-
"""PLAY-01-a:手部互动("逗它")检测 + 跟手走(独立脚本,先不加表情)。

链路:media.get_frame() → 降采样 640×360 → MediaPipe HandLandmarker(VIDEO,每帧)
  → 手 bbox 中心 (u,v) + 大小(占画面比例 = 近距判据)
  → "在逗它"迟滞判定(够大持续 0.3s 进入;变小/丢手持续 0.8s 退出)
  → 逗它时:One Euro + 时间常数 τ 型增益跟手中心(完全复用人脸跟踪机制)
  → 非逗它:保持当前朝向,超时缓慢回中。

近距判据:手 bbox 最大边占画面比例 size。粗标定(65°FOV):
  伸开的手 ~18cm,30cm 距离 ≈ 0.5+,60cm ≈ 0.3,1m ≈ 0.2 → ON=0.30 / OFF=0.22。
打印实测 size,不准就照数据调。

方向约定同 vis01:摄像头不镜像,画面右(u>0.5)= 机器人右 → yaw 负;画面下 → pitch 正。

运行 60s:手伸近晃 → 头平滑跟手;手拿远/移开 → 不跟、缓慢回中。
$env:PYTHONUTF8=1; python vision\\play01_hand_track.py
"""

import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import math
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from reachy_mini import ReachyMini

# ───────────────────────── 配置 ─────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
RUN_SECONDS = 60.0
DECIMATE = 3
FOV_X_DEG = 65.0
FOV_Y_DEG = 40.0

# 跟踪控制(时间常数型同 vis01,但调"灵敏"档:跟脸要稳重,逗它要跟得上快手。
# 首轮用户实测:平滑但快手跟不上 → TAU 0.40→0.25 / MAX_STEP 1.5→3.0(30fps≈90°/s,
# 与听声转头同速)/ One Euro beta 0.08→0.25(高速时降滤波延迟))
YAW_SIGN = -1.0
PITCH_SIGN = +1.0
TAU = 0.25
DEADBAND_DEG = 2.0
MAX_STEP_DEG = 3.0
AMP = 0.90          # 转动幅度系数(round4 用户验收:整体挺好,幅度收 10% 更自然)
YAW_LIMIT = 25.0 * AMP
PITCH_LIMIT = 15.0 * AMP

# "在逗它"判定(近距 = 手 bbox 最大边占画面比例)
PLAY_SIZE_ON = 0.30    # 手够大(够近)才算逗
PLAY_SIZE_OFF = 0.22   # 退出阈值(迟滞防边界抖)
PLAY_ON_S = 0.3        # 持续够大才进入(防路过挥手误触)
PLAY_OFF_S = 1.5       # 持续变小/丢手才退出(首轮实测:手怼太近 size>1.2 时检测
                       # 会连丢 1s+,0.8s 窗导致 60s 内 6 次误退出 → 加长吸收)
LOST_HOLD_S = 1.0      # 退出逗它后保持朝向的时长,再缓慢回中
MISS_RESET_N = 3       # 连续丢 N 帧才 reset One Euro(快速晃手时检测会偶丢)
COAST_S = 0.35         # 检测丢失期惯性外推时长(round2:快手丢检出→头愣住→猛追;
                       # 用 One Euro 的速度估计把目标点继续向前推,像猫预判逗猫棒)
COAST_DU_MAX = 0.20    # 外推位移上限(画面占比。round3:不设限时快手速度估计大,
                       # 目标点直接飞到画面边,头在限位间乱甩 → 只许往前"探一小步")
COAST_VEL_MAX = 2.0    # 速度钳位(/s):防检出位置跳变瞬间的速度尖峰被拿去外推
# 两道门在检测源头过滤(round5 用户定:手要有置信度 + 要够大,逗它的手不会小):
# 背景小物会被低分误检成"手"(实拍 size 0.06~0.15 / 真手 0.6+),曾把头拽去看角落
HAND_SCORE_MIN = 0.6   # handedness score 当置信度用(真手 >0.9,误检低)
HAND_SIZE_MIN = 0.22   # 小于此的"手"直接当不存在
HAND_TRACK_CONF = 0.3  # 仅跟踪置信度保持放宽(快手运动模糊不丢锁);检测/存在恢复严格

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:7.2f}s] {msg}", flush=True)


INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]


def head_pose(pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [0.0, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


class OneEuroFilter:
    """标准 One Euro:低速强平滑防抖,高速低延迟跟手。"""

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.08, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0
        self.t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(1e-3, t - self.t_prev)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


def hand_center_size(lms) -> tuple[float, float, float]:
    """单手 21 点 → (中心u, 中心v, bbox最大边占比)。"""
    xs = [p.x for p in lms]
    ys = [p.y for p in lms]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0,
            max(max(xs) - min(xs), max(ys) - min(ys)))


def main() -> int:
    if not os.path.exists(MODEL_PATH):
        log(f"❌ 手部模型不存在:{MODEL_PATH}")
        return 1

    print("=== PLAY-01-a:手部互动检测 + 跟手(只跟,不加表情)===", flush=True)
    landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=HAND_TRACK_CONF,
        )
    )
    log("✅ MediaPipe HandLandmarker 就绪")

    fx = OneEuroFilter(min_cutoff=0.8, beta=0.25)  # beta 调高:快手低延迟
    fy = OneEuroFilter(min_cutoff=0.8, beta=0.25)

    yaw_t = pitch_t = 0.0
    playing = False          # "在逗它"状态(迟滞)
    big_since = None         # 手够大的起始时刻
    small_since = None       # 手变小/丢失的起始时刻
    play_left_at = 0.0       # 退出逗它的时刻(LOST_HOLD 用)
    miss_run = 0
    last_u = last_v = None   # 最近一次滤波后的手位置(惯性外推用)
    vel_u = vel_v = 0.0      # One Euro 速度估计
    last_hand_t = 0.0
    last_ts_ms = -1
    t_prev_ctrl = time.monotonic()

    n_det = n_hand = n_play_frames = n_coast = 0
    infer_acc: list[float] = []
    stat_t = time.monotonic()
    last_print = 0.0

    log("连接 Reachy Mini(media_backend=default)…")
    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="default",
        automatic_body_yaw=False,
    ) as mini:
        try:
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            warm = None
            wdl = time.monotonic() + 10.0
            while warm is None and time.monotonic() < wdl:
                warm = mini.media.get_frame()
                if warm is None:
                    time.sleep(0.05)
            if warm is None:
                log("❌ 10s 内摄像头无帧,中止")
                return 1

            log(f"READY_FOR_PLAY(开始 {RUN_SECONDS:.0f}s:手伸近晃→应跟手;拿远/移开→应不跟)")
            end = time.monotonic() + RUN_SECONDS
            while time.monotonic() < end:
                frame = mini.media.get_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue
                now = time.monotonic()
                rgb = np.ascontiguousarray(frame[::DECIMATE, ::DECIMATE, ::-1])
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t_inf = time.monotonic()
                last_ts_ms = max(last_ts_ms + 1, int((now - T0) * 1000))
                result = landmarker.detect_for_video(mp_img, last_ts_ms)
                infer_acc.append((time.monotonic() - t_inf) * 1000)
                n_det += 1

                hand = None
                hand_score = 0.0
                if result.hand_landmarks:
                    cand = hand_center_size(result.hand_landmarks[0])
                    hand_score = result.handedness[0][0].score if result.handedness else 1.0
                    # 两道门:够自信 + 够大,否则当不存在(背景误检在源头掐掉)
                    if hand_score >= HAND_SCORE_MIN and cand[2] >= HAND_SIZE_MIN:
                        hand = cand
                        n_hand += 1

                # ── "在逗它"迟滞状态机 ──
                big_now = hand is not None and hand[2] >= (PLAY_SIZE_OFF if playing else PLAY_SIZE_ON)
                if big_now:
                    small_since = None
                    if big_since is None:
                        big_since = now
                    if not playing and now - big_since >= PLAY_ON_S:
                        playing = True
                        log(f"🎾 进入逗它(size={hand[2]:.2f})")
                else:
                    big_since = None
                    if small_since is None:
                        small_since = now
                    if playing and now - small_since >= PLAY_OFF_S:
                        playing = False
                        play_left_at = now
                        log("💤 退出逗它(手远/移开)")

                # ── 跟踪控制 ──
                dt = max(1e-3, now - t_prev_ctrl)
                t_prev_ctrl = now

                def steer(tu: float, tv: float) -> None:
                    nonlocal yaw_t, pitch_t
                    err_yaw = YAW_SIGN * (tu - 0.5) * FOV_X_DEG * AMP
                    err_pitch = PITCH_SIGN * (tv - 0.5) * FOV_Y_DEG * AMP
                    if abs(err_yaw) < DEADBAND_DEG:
                        err_yaw = 0.0
                    if abs(err_pitch) < DEADBAND_DEG:
                        err_pitch = 0.0
                    k = 1.0 - math.exp(-dt / TAU)
                    yaw_t = float(np.clip(yaw_t + np.clip(k * err_yaw, -MAX_STEP_DEG, MAX_STEP_DEG),
                                          -YAW_LIMIT, YAW_LIMIT))
                    pitch_t = float(np.clip(pitch_t + np.clip(k * err_pitch, -MAX_STEP_DEG, MAX_STEP_DEG),
                                            -PITCH_LIMIT, PITCH_LIMIT))
                    mini.set_target(head=head_pose(pitch_deg=pitch_t, yaw_deg=yaw_t))

                if playing and hand is not None:
                    miss_run = 0
                    n_play_frames += 1
                    u = fx(hand[0], now)
                    v = fy(hand[1], now)
                    last_u, last_v = u, v
                    vel_u, vel_v = fx.dx_prev, fy.dx_prev  # One Euro 速度估计
                    last_hand_t = now
                    steer(u, v)
                elif playing:
                    age = now - last_hand_t
                    if age <= COAST_S and last_u is not None:
                        # 惯性外推:检测短暂丢失时按手速度继续前推目标点(防愣住→猛追)
                        # 位移封顶 + 速度钳位:只往前探一小步,不许飞到画面边(round3 乱甩教训)
                        n_coast += 1
                        cu = float(np.clip(vel_u, -COAST_VEL_MAX, COAST_VEL_MAX)) * age
                        cv = float(np.clip(vel_v, -COAST_VEL_MAX, COAST_VEL_MAX)) * age
                        cu = float(np.clip(cu, -COAST_DU_MAX, COAST_DU_MAX))
                        cv = float(np.clip(cv, -COAST_DU_MAX, COAST_DU_MAX))
                        steer(min(1.0, max(0.0, last_u + cu)),
                              min(1.0, max(0.0, last_v + cv)))
                    else:
                        miss_run += 1
                        if miss_run >= MISS_RESET_N:
                            fx.reset()
                            fy.reset()
                else:
                    fx.reset()
                    fy.reset()
                    if (now - play_left_at > LOST_HOLD_S
                            and (abs(yaw_t) > 0.5 or abs(pitch_t) > 0.5)):
                        yaw_t *= 0.97
                        pitch_t *= 0.97
                        mini.set_target(head=head_pose(pitch_deg=pitch_t, yaw_deg=yaw_t))

                # ── 打印(4Hz)──
                if now - last_print >= 0.25:
                    if hand is not None:
                        flag = "🎾 逗它中" if playing else "·  无视  "
                        log(f"{flag}|手 size={hand[2]:.2f} conf={hand_score:.2f} "
                            f"u={hand[0]:.2f} v={hand[1]:.2f}"
                            f"|头目标 yaw={yaw_t:+.1f}° pitch={pitch_t:+.1f}°")
                    elif playing:
                        log(f"🎾 逗它中|手暂时丢失|头目标 yaw={yaw_t:+.1f}° pitch={pitch_t:+.1f}°")
                    last_print = now

                if now - stat_t >= 10.0:
                    fps = n_det / (now - stat_t)
                    log(f"—— 统计:{fps:.1f}fps|有手 {100*n_hand/max(1,n_det):.0f}%|"
                        f"逗它跟踪帧 {n_play_frames}|惯性外推帧 {n_coast}|推理 {np.mean(infer_acc):.1f}ms")
                    stat_t = now
                    n_det = n_hand = n_play_frames = n_coast = 0
                    infer_acc = []

        except KeyboardInterrupt:
            log("Ctrl+C 提前结束")
        finally:
            try:
                mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass

    print("=== 完成:手近时跟手了吗?平滑吗?手远/移开时停了吗? ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
