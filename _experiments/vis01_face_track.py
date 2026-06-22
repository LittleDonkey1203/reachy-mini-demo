# -*- coding: utf-8 -*-
"""VIS-01:本地视觉看脸 + 转头(MediaPipe Face Landmarker,独立脚本,不接对话)。

链路:media.get_frame()(1080p BGR,实测 ~49FPS)
  → 整数抽样降采样 640×360(1080p 直接喂太慢)
  → MediaPipe FaceLandmarker(VIDEO 模式)检测人脸
  → 取最大人脸中心 (u,v) → One Euro 滤波 → P 控制映射头部 yaw/pitch
  → set_target 连续平滑转头(闭环:摄像头在头上,转头后误差自然收敛)

方向约定(CALIBRATION.md §2):yaw+ = 看左,pitch+ = 看下。
摄像头不镜像 → 人在画面右(u>0.5)= 在机器人右边 → 应右转 = yaw 负。
安全幅度:yaw 限 ±25°,pitch 限 ±15°(均小于手势验证过的上限)。

运行 60 秒自动结束,期间在镜头前移动;Ctrl+C 可提前退出。
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
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")
RUN_SECONDS = 60.0

# 1920×1080 → ::3 整数抽样 → 640×360(零拷贝级开销,比 resize 快)
DECIMATE = 3

# 相机视场角估计(只影响 P 增益的标定,闭环下不需要精确)
FOV_X_DEG = 65.0
FOV_Y_DEG = 40.0

# 控制参数
# 教训(首轮实测):按"每帧吃掉固定比例误差"设增益,在 47fps 下等效 ~190°/s,
# 叠加摄像头管线 ~100ms 延迟 → pitch 在 ±15° 限位间打摆("一直点头")。
# 改为时间常数型:step = err × (1 − exp(−dt/TAU)),与帧率解耦。
YAW_SIGN = -1.0     # u>0.5(画面右)→ 机器人右转(yaw 负);若实测方向反了改成 +1.0
PITCH_SIGN = +1.0   # v>0.5(画面下)→ 低头(pitch 正)
TAU = 0.40          # 收敛时间常数(s):约 0.4s 吃掉 63% 误差,对 100ms 延迟稳定
DEADBAND_DEG = 2.0  # 误差死区,防微抖
MAX_STEP_DEG = 1.5  # 单帧最大步进(47fps 时 ~70°/s 上限)
YAW_LIMIT = 25.0
PITCH_LIMIT = 15.0
LOST_HOLD_S = 1.5   # 人脸丢失后保持当前朝向的时长,超时缓慢回中

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

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
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


def pick_main_face(result) -> tuple[float, float, float] | None:
    """返回最大人脸的 (u, v, 高度占比);没有人脸返回 None。"""
    if not result.face_landmarks:
        return None
    best = None
    best_h = -1.0
    for lms in result.face_landmarks:
        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        h = max(ys) - min(ys)
        if h > best_h:
            best_h = h
            best = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, h)
    return best


def main() -> int:
    if not os.path.exists(MODEL_PATH):
        log(f"❌ 模型不存在:{MODEL_PATH}")
        return 1

    print("=== VIS-01:MediaPipe 看脸 + 转头 ===", flush=True)
    log("初始化 MediaPipe FaceLandmarker(VIDEO 模式)…")
    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=2,
        )
    )
    log("✅ MediaPipe 就绪")

    fx = OneEuroFilter(min_cutoff=0.8, beta=0.08)
    fy = OneEuroFilter(min_cutoff=0.8, beta=0.08)

    yaw_t = 0.0     # 当前头部目标角(控制状态)
    pitch_t = 0.0
    last_seen = 0.0
    last_ts_ms = -1   # MediaPipe VIDEO 模式要求时间戳严格递增
    t_prev_ctrl = time.monotonic()  # 时间常数型增益用

    # 统计
    n_frames = 0          # 处理的帧数(成功 get_frame)
    n_detect_calls = 0    # MediaPipe 调用次数
    n_face_hits = 0       # 检出人脸的次数
    infer_ms_acc: list[float] = []
    stat_t = time.monotonic()
    stat_frames = 0

    log("连接 Reachy Mini(media_backend=default)…")
    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="default",
        automatic_body_yaw=False,
    ) as mini:
        try:
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            # 摄像头预热
            warm = None
            wdl = time.monotonic() + 10.0
            while warm is None and time.monotonic() < wdl:
                warm = mini.media.get_frame()
                if warm is None:
                    time.sleep(0.05)
            if warm is None:
                log("❌ 10s 内摄像头无帧,中止")
                return 1
            log(f"✅ 摄像头出帧 {warm.shape};降采样后 {warm[::DECIMATE, ::DECIMATE].shape}")

            log(f"READY_FOR_FACE(开始跟踪 {RUN_SECONDS:.0f}s,请在镜头前移动)")
            end = time.monotonic() + RUN_SECONDS
            while time.monotonic() < end:
                frame = mini.media.get_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue
                n_frames += 1
                now = time.monotonic()

                # 1080p BGR → 640×360 RGB(整数抽样 + 通道反转)
                rgb = np.ascontiguousarray(frame[::DECIMATE, ::DECIMATE, ::-1])
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t_inf = time.monotonic()
                last_ts_ms = max(last_ts_ms + 1, int((now - T0) * 1000))  # 同毫秒帧防撞
                result = landmarker.detect_for_video(mp_img, last_ts_ms)
                infer_ms_acc.append((time.monotonic() - t_inf) * 1000)
                n_detect_calls += 1
                stat_frames += 1

                face = pick_main_face(result)
                if face is not None:
                    n_face_hits += 1
                    last_seen = now
                    u_raw, v_raw, h = face
                    u = fx(u_raw, now)
                    v = fy(v_raw, now)
                    # 画面误差 → 角度误差(闭环,时间常数型增益,与帧率解耦)
                    err_yaw = YAW_SIGN * (u - 0.5) * FOV_X_DEG
                    err_pitch = PITCH_SIGN * (v - 0.5) * FOV_Y_DEG
                    if abs(err_yaw) < DEADBAND_DEG:
                        err_yaw = 0.0
                    if abs(err_pitch) < DEADBAND_DEG:
                        err_pitch = 0.0
                    dt = max(1e-3, now - t_prev_ctrl)
                    k = 1.0 - math.exp(-dt / TAU)
                    step_yaw = float(np.clip(k * err_yaw, -MAX_STEP_DEG, MAX_STEP_DEG))
                    step_pitch = float(np.clip(k * err_pitch, -MAX_STEP_DEG, MAX_STEP_DEG))
                    yaw_t = float(np.clip(yaw_t + step_yaw, -YAW_LIMIT, YAW_LIMIT))
                    pitch_t = float(np.clip(pitch_t + step_pitch, -PITCH_LIMIT, PITCH_LIMIT))
                    mini.set_target(head=head_pose(pitch_deg=pitch_t, yaw_deg=yaw_t))
                else:
                    fx.reset()
                    fy.reset()
                    if now - last_seen > LOST_HOLD_S and (abs(yaw_t) > 0.5 or abs(pitch_t) > 0.5):
                        yaw_t *= 0.97   # 丢脸超时:缓慢回中
                        pitch_t *= 0.97
                        mini.set_target(head=head_pose(pitch_deg=pitch_t, yaw_deg=yaw_t))
                t_prev_ctrl = now

                # 每 5s 报一次状态
                if now - stat_t >= 5.0:
                    fps = stat_frames / (now - stat_t)
                    avg_inf = float(np.mean(infer_ms_acc[-stat_frames:])) if stat_frames else 0.0
                    n_faces = len(result.face_landmarks) if result.face_landmarks else 0
                    if face is not None:
                        log(f"检测 FPS={fps:.1f}|推理 {avg_inf:.1f}ms|人脸 {n_faces} 个|"
                            f"中心 u={u_raw:.2f} v={v_raw:.2f}|头部目标 yaw={yaw_t:+.1f}° pitch={pitch_t:+.1f}°")
                    else:
                        log(f"检测 FPS={fps:.1f}|推理 {avg_inf:.1f}ms|当前无人脸|"
                            f"头部 yaw={yaw_t:+.1f}° pitch={pitch_t:+.1f}°")
                    stat_t = now
                    stat_frames = 0

        except KeyboardInterrupt:
            log("收到 Ctrl+C,提前结束")
        finally:
            try:
                mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass

    # ── 汇总 ──
    elapsed = time.monotonic() - T0
    print("\n========== 汇总 ==========", flush=True)
    print(f"处理帧数:{n_frames}|MediaPipe 调用:{n_detect_calls}", flush=True)
    if n_detect_calls:
        print(f"端到端检测 FPS(全程均值):{n_detect_calls / max(1e-6, elapsed - 12):.1f}", flush=True)
        print(f"单帧推理耗时:均值 {np.mean(infer_ms_acc):.1f}ms / P95 {np.percentile(infer_ms_acc, 95):.1f}ms", flush=True)
        print(f"人脸检出率:{100.0 * n_face_hits / n_detect_calls:.0f}%({n_face_hits}/{n_detect_calls})", flush=True)
    ok = n_detect_calls > 0 and n_face_hits > 0
    print(f"=== {'视觉跟踪闭环完成,平滑度请肉眼确认' if ok else '未检出人脸,需排查'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
