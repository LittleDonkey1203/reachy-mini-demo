# -*- coding: utf-8 -*-
"""TRACK-FIX 1d 跟踪质量验证(75s 三段式,纯跟踪不连对话)。

视觉独立进程版(vision_worker 子进程)+ 时间常数型跟随,固定时序:
  第1段  0-25s 静止测试:用户坐正前 1m 不动 → 量化静止抖动(track std)
  第2段 25-50s 移动测试:用户缓慢左右平移 → 看跟随平滑度/滞后
  第3段 50-75s 侧脸测试:用户转头侧脸再回正 → 看丢脸缓冲/回正是否乱甩

每 5s 打实时 FPS/检出率;结束出每段汇总(FPS/检出率/丢脸事件/最大漏检连击/静止抖动)。
运行:$env:PYTHONUTF8=1; python voice\\_track_verify.py
"""

import math
import multiprocessing
import os
import queue
import sys
import threading
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from vision_worker import vision_worker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "vision", "models", "face_landmarker.task")

VIS_MAX_FPS = 40.0
VIS_MISS_N = 5
DECIMATE = 3
FOV_X_DEG = 65.0
FOV_Y_DEG = 40.0
TRACK_TAU = 0.40
TRACK_DEADBAND = 2.0
TRACK_MAX_STEP = 1.5
YAW_LIMIT = 23.0
PITCH_LIMIT = 15.0
LOST_HOLD_S = 1.5
RETURN_TAU = 0.8
YAW_SIGN = -1.0
PITCH_SIGN = +1.0

PHASES = [
    ("STILL  静止(坐正前1m不动)", 0.0, 25.0),
    ("MOVE   移动(缓慢左右平移)", 25.0, 50.0),
    ("PROFILE 侧脸(转头再回正)", 50.0, 75.0),
]

T0 = time.monotonic()


def log(m: str) -> None:
    print(f"[t+{time.monotonic() - T0:6.2f}s] {m}", flush=True)


INIT_HEAD = np.eye(4)


def head_pose(pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [0.0, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.08, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

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
        a = self._alpha(self.min_cutoff + self.beta * abs(dx_hat), dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class S:
    lock = threading.Lock()
    track_yaw = 0.0
    track_pitch = 0.0
    face_seen_at = 0.0
    phase_idx = 0
    # 每段统计
    stats = [dict(n=0, hit=0, loss_events=0, max_streak=0, infer=[], yaw_samples=[], pitch_samples=[])
             for _ in PHASES]


def frame_pump(mini, fq, stop):
    t_last = 0.0
    while not stop.is_set():
        frame = mini.media.get_frame()
        now = time.monotonic()
        if frame is None:
            time.sleep(0.005)
            continue
        if now - t_last < 1.0 / VIS_MAX_FPS:
            continue
        t_last = now
        rgb = np.ascontiguousarray(frame[::DECIMATE, ::DECIMATE, ::-1])
        try:
            fq.put_nowait((now, rgb))
        except Exception:
            try:
                fq.get_nowait()
                fq.put_nowait((now, rgb))
            except Exception:
                pass


def result_loop(rq, stop):
    fx = OneEuroFilter()
    fy = OneEuroFilter()
    t_prev = time.monotonic()
    miss = 0
    n5 = 0
    hit5 = 0
    t5 = time.monotonic()
    while not stop.is_set():
        try:
            t_grab, u_raw, v_raw, _h, _n, infer_ms = rq.get(timeout=0.2)
        except queue.Empty:
            continue
        if t_grab == "ready":
            log("👁 视觉子进程就绪")
            continue
        now = time.monotonic()
        with S.lock:
            ph = S.stats[S.phase_idx]
        ph["n"] += 1
        ph["infer"].append(infer_ms)
        n5 += 1
        if u_raw is not None:
            ph["hit"] += 1
            hit5 += 1
            if miss >= VIS_MISS_N:
                pass  # 刚从真丢脸恢复
            miss = 0
            u = fx(u_raw, now)
            v = fy(v_raw, now)
            err_yaw = YAW_SIGN * (u - 0.5) * FOV_X_DEG
            err_pitch = PITCH_SIGN * (v - 0.5) * FOV_Y_DEG
            if abs(err_yaw) < TRACK_DEADBAND:
                err_yaw = 0.0
            if abs(err_pitch) < TRACK_DEADBAND:
                err_pitch = 0.0
            dt = max(1e-3, now - t_prev)
            t_prev = now
            k = 1.0 - math.exp(-dt / TRACK_TAU)
            with S.lock:
                S.track_yaw = float(np.clip(S.track_yaw + np.clip(k * err_yaw, -TRACK_MAX_STEP, TRACK_MAX_STEP), -YAW_LIMIT, YAW_LIMIT))
                S.track_pitch = float(np.clip(S.track_pitch + np.clip(k * err_pitch, -TRACK_MAX_STEP, TRACK_MAX_STEP), -PITCH_LIMIT, PITCH_LIMIT))
                S.face_seen_at = now
                ph["yaw_samples"].append(S.track_yaw)
                ph["pitch_samples"].append(S.track_pitch)
        else:
            miss += 1
            ph["max_streak"] = max(ph["max_streak"], miss)
            if miss == VIS_MISS_N:
                ph["loss_events"] += 1  # 真丢脸事件(连续 N 帧)计一次
            dt = max(1e-3, now - t_prev)
            t_prev = now
            if miss >= VIS_MISS_N:
                fx.reset()
                fy.reset()
                with S.lock:
                    if now - S.face_seen_at > LOST_HOLD_S:
                        decay = math.exp(-dt / RETURN_TAU)
                        S.track_yaw *= decay
                        S.track_pitch *= decay
        if now - t5 >= 5.0:
            log(f"  实时:FPS {n5 / (now - t5):.1f}|检出 {100.0 * hit5 / max(1, n5):.0f}%|"
                f"yaw {S.track_yaw:+.1f}° pitch {S.track_pitch:+.1f}°")
            n5 = 0
            hit5 = 0
            t5 = now


def head_loop(mini, stop):
    while not stop.is_set():
        with S.lock:
            y, p = S.track_yaw, S.track_pitch
        try:
            mini.set_target(head=head_pose(pitch_deg=p, yaw_deg=y))
        except Exception:
            time.sleep(1.0)
        time.sleep(1.0 / 25.0)


def main() -> int:
    if not os.path.exists(MODEL):
        print("❌ 模型不存在", flush=True)
        return 1
    print("=== TRACK-FIX 跟踪质量验证(75s 三段式)===", flush=True)

    fq: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
    rq: multiprocessing.Queue = multiprocessing.Queue(maxsize=64)
    multiprocessing.Process(target=vision_worker, args=(MODEL, fq, rq), daemon=True).start()

    stop = threading.Event()
    with ReachyMini(connection_mode="localhost_only", media_backend="default",
                    automatic_body_yaw=False) as mini:
        try:
            mini.goto_target(INIT_HEAD, duration=1.0, body_yaw=0.0)
            warm = None
            dl = time.monotonic() + 10
            while warm is None and time.monotonic() < dl:
                warm = mini.media.get_frame()
                if warm is None:
                    time.sleep(0.05)
            if warm is None:
                log("❌ 摄像头无帧")
                return 1
            threading.Thread(target=frame_pump, args=(mini, fq, stop), daemon=True).start()
            threading.Thread(target=result_loop, args=(rq, stop), daemon=True).start()
            threading.Thread(target=head_loop, args=(mini, stop), daemon=True).start()

            log("READY_FOR_TRACK(75s 开始)")
            t_run = time.monotonic()
            cur = -1
            while time.monotonic() - t_run < PHASES[-1][2]:
                t = time.monotonic() - t_run
                idx = next(i for i, (_, a, b) in enumerate(PHASES) if a <= t < b)
                if idx != cur:
                    cur = idx
                    with S.lock:
                        S.phase_idx = idx
                    print(f"\n████ 第{idx + 1}段 {PHASES[idx][0]} ████\n", flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            log("Ctrl+C 提前结束")
        finally:
            stop.set()
            try:
                fq.put_nowait(None)
            except Exception:
                pass
            time.sleep(0.2)
            try:
                mini.goto_target(INIT_HEAD, duration=1.0, body_yaw=0.0)
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass

    print("\n========== 三段汇总 ==========", flush=True)
    print(f"{'段':28s} {'FPS':>6s} {'检出率':>7s} {'丢脸事件':>8s} {'最大连击':>8s} {'抖动std(yaw/pitch)':>20s}", flush=True)
    for (name, a, b), ph in zip(PHASES, S.stats):
        dur = b - a
        fps = ph["n"] / dur
        hitp = 100.0 * ph["hit"] / max(1, ph["n"])
        ys = ph["yaw_samples"][len(ph["yaw_samples"]) // 3:]  # 去掉前 1/3 收敛期
        ps = ph["pitch_samples"][len(ph["pitch_samples"]) // 3:]
        jit = f"{np.std(ys):.2f}°/{np.std(ps):.2f}°" if ys else "—"
        print(f"{name:28s} {fps:6.1f} {hitp:6.0f}% {ph['loss_events']:8d} {ph['max_streak']:8d} {jit:>20s}", flush=True)
    print("=== 完成,等待肉眼结论 ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
