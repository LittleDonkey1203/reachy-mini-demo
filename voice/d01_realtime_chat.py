# -*- coding: utf-8 -*-
"""Reachy Mini × Qwen3.5-Omni-Realtime 语音对话(D-01 + O-01a + V-01 + F-01:完整体)。

对着机器人说话 → Qwen 全双工识别并生成语音 → 从机器人扬声器播放;
说话时可随时插话打断(barge-in);模型自主调用动作工具做身体语言
(点头/摇头/看向/摆天线/歪头),可边说边动;说话时有 idle 微动;
让它"看"时调 take_snapshot 抓当前画面 → chat.completions 看图 → 语音转述;
本地视觉(MediaPipe)持续看脸,聊天时头温和地跟着人转(F-01 融合)。

三层动作仲裁(F-01,优先级从高到低,头部 channel 唯一写入口是 head_control_loop):
  Primary  明确手势(function_call):motion 线程 goto_target 独占,
           其他层让位(head_control 停发);手势以"当前跟随姿态"为基准做
           (点头时继续看着人),做完回基准 → 跟随无缝接管,不突跳。
  Tracking 人脸跟随:视觉线程 41fps 检测、积分出 track_yaw/pitch 目标;
           ⭐ 增益必须时间常数型 step=err×(1−exp(−dt/τ)),与帧率解耦
           (帧率比例增益在 47fps 下打摆,见 CALIBRATION §9 教训)。
  Idle     说话微动:跟随之上小幅叠加(跟随时缩到 40%,不打架);
           无人脸时回原幅度,行为同 O-01a-2。

音频链路(实测验证,见 ../CALIBRATION.md §6):
  上行:麦克风 16kHz 原生 → 取 audio[:,0] → int16 → base64(零重采样)
  下行:24kHz PCM16 → resample_poly 24k→16k → 抖动缓冲 ~300ms → push_audio_sample
  打断:speech_started → 队列代际作废 + audio.clear_player() + 必要时 cancel_response

get_frame 协调(F-01):视觉线程是唯一持续抓帧者,最新帧共享在 State;
take_snapshot 直接读共享帧(≤25ms 新),不再和跟随抢 get_frame。

运行(需 daemon 已启动、DASHSCOPE_API_KEY 已配):
  $env:PYTHONUTF8=1
  & "C:\\Users\\ldkji\\AppData\\Local\\Reachy Mini Control\\.venv\\Scripts\\python.exe" voice\\d01_realtime_chat.py [秒数]
可选参数 [秒数]:到时自动干净退出(编排测试用);不带参数则 Ctrl+C 退出。
"""

import os

# ── 代理隔离:必须在 import reachy_mini / dashscope 之前 ──
_no_proxy = "localhost,127.0.0.1,::1,.aliyuncs.com,aliyuncs.com"
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

import base64
import io
import json
import math
import queue
import sys
import threading
import time

import numpy as np
from PIL import Image
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation as R

import dashscope
from dashscope.audio.qwen_omni import (
    AudioFormat,
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)
from openai import OpenAI
from reachy_mini import ReachyMini

import mediapipe as mp  # 只用库做推理;禁开 cv2.VideoCapture / sounddevice 流
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ───────────────────────── 配置 ─────────────────────────
MODEL = "qwen3.5-omni-plus-realtime"
VISION_MODEL = "qwen3.5-omni-plus"   # take_snapshot 看图用(chat.completions 回合制)
VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VOICE = "Ethan"
INSTRUCTIONS = (
    "你是桌面机器人 Reachy Mini,有真实的身体(头、天线)和一台摄像头。"
    "用简体中文、口语化、简短地回答,一般不超过两三句话。"
    "回答时自然地配合动作工具表达身体语言:打招呼/同意时点头,否定时摇头,"
    "开心/兴奋/被夸时摆天线,好奇/疑惑时歪头。"
    "重要:做动作时必须同时用语音回应,边说边做;绝不要默默做动作不说话。"
    "用户让你看东西时调用 take_snapshot,拿到画面描述后用自己的话自然地告诉用户你看到了什么。"
)

SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")  # 快照存放(已 gitignore)

OUT_SR = 24000   # Realtime 下行采样率
PLAY_SR = 16000  # Reachy 播放管线 appsrc 固定 16kHz
JITTER_S = 0.30
JITTER_WALL_S = 0.50

# idle 微动(说话时的"活着感",O-01a-2)
IDLE_HZ = 25.0
IDLE_YAW_AMP = 2.5    # 度(无人脸时的原幅度)
IDLE_PITCH_AMP = 1.5  # 度
IDLE_YAW_F = 0.20     # Hz
IDLE_PITCH_F = 0.30   # Hz
IDLE_TAU = 0.5        # 包络时间常数(s)
TRACK_SWAY_SCALE = 0.4  # 跟随中微动缩放(小幅叠加,不和跟随打架)

# ── 本地视觉跟随(VIS-01 → F-01 融合,参数与教训见 CALIBRATION §9)──
VIS_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vision", "models", "face_landmarker.task",
)
VIS_MAX_FPS = 30.0   # 检测限频(基线 41fps/12ms,降到 30 给对话留余量;τ 控制与帧率解耦)
DECIMATE = 3         # 1920×1080 → ::3 整数抽样 → 640×360
FOV_X_DEG = 65.0
FOV_Y_DEG = 40.0
# ⭐ 铁律:增益必须时间常数型 step = err × (1 − exp(−dt/TAU)),与帧率解耦。
#   按"每帧吃固定比例"在高帧率下等效角速度爆表 + 相机 ~100ms 延迟 → 限位间打摆。
TRACK_TAU = 0.40       # 收敛时间常数(s)
TRACK_DEADBAND = 2.0   # 误差死区(度),防微抖
TRACK_MAX_STEP = 1.5   # 单帧最大步进(度)
TRACK_YAW_LIMIT = 25.0
TRACK_PITCH_LIMIT = 15.0
LOST_HOLD_S = 1.5      # 丢脸后保持朝向时长,超时缓慢回中
RETURN_TAU = 0.8       # 回中时间常数(s)(同样与帧率解耦)
YAW_SIGN = -1.0        # 画面右(u>0.5)= 机器人右边 → yaw 负(摄像头不镜像)
PITCH_SIGN = +1.0      # 画面下 → pitch 正(低头)
# 手势合成安全箱:手势 offset 叠加跟随基准后裁剪到此范围(幅度均在已验证上限内)
GES_YAW_BOX = 25.0
GES_PITCH_BOX = 16.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────── 动作库(CALIBRATION.md §2 标定参数)─────────────────────────
INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]


def head_pose(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


def gpose(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """手势姿态 = 跟随基准 + 手势 offset,裁剪进安全箱(F-01:点头时继续看着人)。"""
    return head_pose(
        pitch_deg=float(np.clip(pitch, -GES_PITCH_BOX, GES_PITCH_BOX)),
        yaw_deg=float(np.clip(yaw, -GES_YAW_BOX, GES_YAW_BOX)),
        roll_deg=roll,
    )


# 所有手势签名 (mini, base_yaw, base_pitch):以当前跟随姿态为基准,做完回基准
def act_nod(m: ReachyMini, by: float, bp: float) -> None:
    for _ in range(2):
        m.goto_target(gpose(by, bp + 15), duration=0.35, body_yaw=0.0)
        m.goto_target(gpose(by, bp - 10), duration=0.35, body_yaw=0.0)
    m.goto_target(gpose(by, bp), duration=0.35, body_yaw=0.0)


def act_shake(m: ReachyMini, by: float, bp: float) -> None:
    for _ in range(2):
        m.goto_target(gpose(by + 15, bp), duration=0.35, body_yaw=0.0)
        m.goto_target(gpose(by - 15, bp), duration=0.35, body_yaw=0.0)
    m.goto_target(gpose(by, bp), duration=0.35, body_yaw=0.0)


def _look(m: ReachyMini, by: float, bp: float, **kw) -> None:
    """看向某方向是绝对指令(离开人脸),看完回跟随基准。"""
    m.goto_target(head_pose(**kw), duration=0.6, body_yaw=0.0)
    time.sleep(0.8)
    m.goto_target(gpose(by, bp), duration=0.6, body_yaw=0.0)


def act_wiggle(m: ReachyMini, by: float, bp: float) -> None:
    for _ in range(2):
        m.goto_target(antennas=[+0.8, -0.8], duration=0.3, body_yaw=0.0)
        m.goto_target(antennas=[-0.8, +0.8], duration=0.3, body_yaw=0.0)
    m.goto_target(antennas=INIT_ANTENNAS, duration=0.35, body_yaw=0.0)


def act_tilt(m: ReachyMini, by: float, bp: float) -> None:
    m.goto_target(gpose(by, bp, roll=15), duration=0.5, body_yaw=0.0)
    time.sleep(0.8)
    m.goto_target(gpose(by, bp), duration=0.5, body_yaw=0.0)


ACTIONS = {
    "nod": act_nod,
    "shake_head": act_shake,
    "look_left": lambda m, by, bp: _look(m, by, bp, yaw_deg=+16),
    "look_right": lambda m, by, bp: _look(m, by, bp, yaw_deg=-16),
    "look_up": lambda m, by, bp: _look(m, by, bp, pitch_deg=-16),
    "look_down": lambda m, by, bp: _look(m, by, bp, pitch_deg=+16),
    "wiggle_antennas": act_wiggle,
    "tilt_head": act_tilt,
}

_NOPARAM = {"type": "object", "properties": {}}
TOOLS = [
    {"type": "function", "name": "nod", "description": "点头。打招呼、同意、确认、答应请求时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "shake_head", "description": "摇头。否定、拒绝、不同意、说'不'时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_left", "description": "把头转向左边看。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_right", "description": "把头转向右边看。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_up", "description": "抬头看上方。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_down", "description": "低头看下方。", "parameters": _NOPARAM},
    {"type": "function", "name": "wiggle_antennas", "description": "欢快地摆动头顶天线。表达开心、兴奋、被夸奖、热情时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "tilt_head", "description": "歪头。表达好奇、疑惑、思考、没听懂时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "take_snapshot",
     "description": "用摄像头拍一张当前画面并理解内容。当用户让你看东西、问'你看到什么''我手里是什么''这是什么'等需要视觉的问题时调用。",
     "parameters": _NOPARAM},
]


# ───────────────────────── One Euro 滤波(VIS-01 验证参数)─────────────────────────
class OneEuroFilter:
    """标准 One Euro:低速强平滑防抖,高速低延迟跟手。丢脸后必须 reset。"""

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


# ───────────────────────── 共享状态 ─────────────────────────
class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_updated = threading.Event()
        # 播放 / 打断
        self.play_gen = 0
        self.drop_audio = False
        self.in_flight = 0
        self.playback_end_estimate = 0.0
        # function calling 协调(O-01a 修复2:即时回 output + 纯动作响应立即补话)
        self.resp_audio_count = 0
        self.fc_seen_this_resp = False
        self.fc_gen = 0
        # 三层仲裁
        self.action_active = False   # Primary 手势执行中 → Tracking/Idle 让位
        self.track_yaw = 0.0         # Tracking 目标角(视觉线程积分;head_control 读)
        self.track_pitch = 0.0
        self.face_seen_at = 0.0      # 最近一次检出人脸的时刻
        # 帧共享(视觉线程是唯一持续 get_frame 者;take_snapshot 读这里)
        self.latest_frame = None
        self.latest_frame_t = 0.0
        # take_snapshot:进行中的快照数(挂起时 response.done 不补话,等描述回来)
        self.snapshot_pending = 0


# ───────────────────────── 回调:收服务端事件 ─────────────────────────
class ChatCallback(OmniRealtimeCallback):
    def __init__(self, st: State, play_q: "queue.Queue", motion_q: "queue.Queue",
                 snap_q: "queue.Queue", mini: ReachyMini):
        self.st = st
        self.play_q = play_q
        self.motion_q = motion_q
        self.snap_q = snap_q
        self.mini = mini
        self.conv: OmniRealtimeConversation | None = None

    def on_open(self) -> None:
        log("✅ WebSocket 已连接 dashscope.aliyuncs.com")

    def on_close(self, close_status_code, close_msg) -> None:
        log(f"🔌 连接关闭:code={close_status_code} msg={close_msg}")

    def _do_barge_in(self, in_flight: bool) -> None:
        """打断:作废队列 → flush 管线残余 → 必要时取消在途回复。动作/跟随不中断。"""
        st = self.st
        with st.lock:
            st.play_gen += 1
            st.drop_audio = True
            st.playback_end_estimate = time.monotonic()
        while True:
            try:
                self.play_q.get_nowait()
            except queue.Empty:
                break
        try:
            self.mini.media.audio.clear_player()
        except Exception as e:
            log(f"⚠ clear_player 失败:{type(e).__name__}: {e}")
        if in_flight and self.conv is not None:
            self.conv.cancel_response()
        log("⛔ 打断:已停止播放" + (",并取消在途回复" if in_flight else ""))

    def on_event(self, event) -> None:  # SDK 实际传入已解析的 dict
        st = self.st
        try:
            etype = event.get("type", "")
            now = time.monotonic()
            if etype == "session.created":
                log(f"✅ 会话已建立 session_id={event['session']['id']}")
            elif etype == "session.updated":
                log("✅ 会话配置生效(semantic_vad / 8 个动作工具 + take_snapshot 已注册)")
                log("▶ 可以对机器人说话了;它说话时可随时插话打断(Ctrl+C 退出)")
                st.session_updated.set()
            elif etype == "input_audio_buffer.speech_started":
                with st.lock:
                    playing = (now < st.playback_end_estimate) or (not self.play_q.empty())
                    in_flight = st.in_flight > 0
                log("🎤 检测到你开始说话…")
                if playing or in_flight:
                    self._do_barge_in(in_flight)
            elif etype == "input_audio_buffer.speech_stopped":
                log("🤫 检测到你说完了,等模型回应…")
            elif etype == "conversation.item.input_audio_transcription.completed":
                log(f"📝 听到的是:「{(event.get('transcript') or '').strip()}」")
            elif etype == "response.created":
                with st.lock:
                    st.in_flight += 1
                    st.drop_audio = False
                    st.resp_audio_count = 0
                    st.fc_seen_this_resp = False
                log("💭 模型开始生成回复…")
            elif etype == "response.function_call_arguments.done":
                name = event.get("name", "")
                call_id = event.get("call_id", "")
                with st.lock:
                    st.fc_seen_this_resp = True
                    st.fc_gen = st.play_gen
                log(f"🤖 模型调用工具: {name}")
                if name == "take_snapshot":
                    # 快照:必须等图像理解结果才回 output(worker 完成后回 + 补话)
                    with st.lock:
                        st.snapshot_pending += 1
                    self.snap_q.put({"call_id": call_id, "gen": st.fc_gen})
                else:
                    # 手势:乐观即时回 output → 说话不等动作做完
                    self.motion_q.put({"name": name, "call_id": call_id})
                    try:
                        self.conv.create_item({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"success": True, "action": name}, ensure_ascii=False),
                        })
                    except Exception as e:
                        log(f"⚠ 回 function_call_output 失败:{e}")
            elif etype == "response.audio_transcript.delta":
                print(event.get("delta", ""), end="", flush=True)
            elif etype == "response.audio_transcript.done":
                print(flush=True)
            elif etype == "response.audio.delta":
                with st.lock:
                    if st.drop_audio:
                        return
                    gen = st.play_gen
                    st.resp_audio_count += 1
                b64 = event.get("delta") or event.get("audio") or ""
                pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                f16k = resample_poly(pcm.astype(np.float32) / 32768.0, PLAY_SR, OUT_SR).astype(np.float32)
                self.play_q.put((gen, f16k))
            elif etype == "response.done":
                fire_rc = False
                with st.lock:
                    st.in_flight = max(0, st.in_flight - 1)
                    # 纯动作响应(无音频且没被打断)→ 马上补话,不等动作做完
                    # 快照挂起时跳过:等图像描述回来再让模型开口
                    if (
                        st.fc_seen_this_resp
                        and st.resp_audio_count == 0
                        and st.fc_gen == st.play_gen
                        and st.snapshot_pending == 0
                    ):
                        fire_rc = True
                d = self.conv.get_last_first_audio_delay() if self.conv else None
                log(f"✅ 本轮回复完成{f'(首音频延迟 {d:.0f}ms)' if d else ''}")
                if fire_rc and self.conv is not None:
                    self.conv.create_response()
            elif etype == "error":
                log(f"❌ 服务端错误事件:{event}")
        except Exception as e:
            log(f"❌ on_event 处理异常:{type(e).__name__}: {e}\n   原始事件:{str(event)[:300]}")


# ───────────────────────── 动作线程:Primary 手势,串行执行 ─────────────────────────
def motion_loop(mini: ReachyMini, st: State, motion_q: "queue.Queue", stop: threading.Event) -> None:
    """只管执行手势(function_call_output 已在 ws 线程即时回过)。
    F-01:进场读跟随基准 → 手势相对基准做、做完回基准;期间 head_control/视觉积分让位。"""
    while not stop.is_set():
        try:
            job = motion_q.get(timeout=0.1)
        except queue.Empty:
            continue
        name = job["name"]
        fn = ACTIONS.get(name)
        with st.lock:
            st.action_active = True  # Tracking/Idle 让位
            by, bp = st.track_yaw, st.track_pitch
        try:
            if fn is None:
                log(f"⚠ 未知动作 {name}")
            else:
                fn(mini, by, bp)
                log(f"✅ 动作完成: {name}(基准 yaw={by:+.1f}° pitch={bp:+.1f}°,跟随恢复)")
        except Exception as e:
            log(f"⚠ 动作 {name} 执行失败:{type(e).__name__}: {e}")
        finally:
            with st.lock:
                st.action_active = False


# ───────────────────────── 快照线程:共享帧 → 看图 → 回结果 ─────────────────────────
def snapshot_loop(mini: ReachyMini, st: State, cb: ChatCallback, oai: OpenAI,
                  snap_q: "queue.Queue", stop: threading.Event) -> None:
    """take_snapshot:优先读视觉线程共享的最新帧(≤25ms 新,不抢 get_frame);
    视觉线程没帧时退回直接抓。→ 640×360 jpg → chat.completions 看图
    → 描述作为 function_call_output 回 Realtime → response.create 让模型语音转述。"""
    os.makedirs(SNAP_DIR, exist_ok=True)
    snap_idx = 0
    while not stop.is_set():
        try:
            job = snap_q.get(timeout=0.1)
        except queue.Empty:
            continue
        call_id, gen0 = job["call_id"], job["gen"]
        snap_idx += 1
        t0 = time.monotonic()
        log("📸 拍照:取当前画面…")
        with st.lock:
            frame = st.latest_frame
            fresh = (time.monotonic() - st.latest_frame_t) < 1.0
        if frame is None or not fresh:
            frame = None  # 视觉线程未供帧 → 退回直接抓(连抓取最新,防旧帧)
            got = 0
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and got < 3:
                f = mini.media.get_frame()
                if f is not None:
                    frame = f
                    got += 1
                else:
                    time.sleep(0.02)
        desc = ""
        ok = frame is not None
        if not ok:
            desc = "拍照失败,没有抓到画面。"
            log("❌ 没有可用画面帧")
        else:
            img = Image.fromarray(frame[:, :, ::-1]).resize((640, 360))  # BGR→RGB,降采样
            img.save(os.path.join(SNAP_DIR, f"snapshot_{snap_idx:02d}.jpg"), "JPEG", quality=85)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            jpg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            log(f"📸 取到帧并压缩({len(buf.getvalue()) / 1024:.0f}KB),送看图…")
            try:
                comp = oai.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{jpg_b64}"}},
                        {"type": "text",
                         "text": "你是机器人的眼睛。用简体中文两三句话描述画面主要内容,特别是人手里举着或拿着的物体(若有)。"},
                    ]}],
                    stream=True,  # omni 必须流式
                    stream_options={"include_usage": True},
                    extra_body={"modalities": ["text"]},
                )
                parts = []
                for chunk in comp:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                desc = "".join(parts).strip()
                log(f"🖼 图像理解({(time.monotonic() - t0) * 1000:.0f}ms 全程):「{desc}」")
            except Exception as e:
                ok = False
                desc = f"看图服务调用失败:{type(e).__name__}"
                log(f"❌ chat.completions 失败:{type(e).__name__}: {e}")
        fire_rc = False
        with st.lock:
            st.snapshot_pending = max(0, st.snapshot_pending - 1)
            fire_rc = st.play_gen == gen0  # 期间被打断则不补话
        try:
            cb.conv.create_item({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"success": ok, "scene_description": desc}, ensure_ascii=False),
            })
        except Exception as e:
            log(f"⚠ 回 function_call_output 失败:{e}")
            continue
        if fire_rc:
            try:
                cb.conv.create_response()  # 让模型用语音转述所见
            except Exception as e:
                log(f"⚠ response.create 失败:{e}")


# ───────────────────────── 视觉线程:看脸 + 积分跟随目标 ─────────────────────────
def vision_loop(mini: ReachyMini, st: State, stop: threading.Event) -> None:
    """MediaPipe 看脸:唯一持续 get_frame 者(最新帧共享给 take_snapshot)。
    只负责"感知 + 积分目标角",不直接动头(头部唯一写入口在 head_control_loop)。
    手势执行中(action_active)冻结积分,手势结束后从基准平滑收敛到人脸。"""
    if not os.path.exists(VIS_MODEL_PATH):
        log(f"⚠ 视觉模型不存在({VIS_MODEL_PATH}),本次无人脸跟随(其余功能不受影响)")
        return
    try:
        landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=VIS_MODEL_PATH),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=2,
            )
        )
    except Exception as e:
        log(f"⚠ MediaPipe 初始化失败:{type(e).__name__}: {e};本次无人脸跟随")
        return
    log(f"👁 视觉跟随就绪(MediaPipe VIDEO 模式,检测限频 {VIS_MAX_FPS:.0f}fps)")

    fx = OneEuroFilter(min_cutoff=0.8, beta=0.08)
    fy = OneEuroFilter(min_cutoff=0.8, beta=0.08)
    t_start = time.monotonic()
    last_ts_ms = -1            # VIDEO 模式时间戳必须严格递增(同毫秒帧防撞)
    t_prev_ctrl = t_start
    t_last_det = 0.0
    n_det = 0
    n_hit = 0
    infer_acc: list[float] = []
    stat_t = t_start

    while not stop.is_set():
        frame = mini.media.get_frame()
        now = time.monotonic()
        if frame is None:
            time.sleep(0.005)
            continue
        with st.lock:
            st.latest_frame = frame
            st.latest_frame_t = now
        if now - t_last_det < 1.0 / VIS_MAX_FPS:
            continue  # 限频:帧照常共享,检测降频省 CPU
        t_last_det = now

        rgb = np.ascontiguousarray(frame[::DECIMATE, ::DECIMATE, ::-1])
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t_inf = time.monotonic()
        last_ts_ms = max(last_ts_ms + 1, int((now - t_start) * 1000))
        try:
            result = landmarker.detect_for_video(mp_img, last_ts_ms)
        except Exception as e:
            log(f"⚠ MediaPipe 检测异常:{type(e).__name__}: {e}")
            continue
        infer_acc.append((time.monotonic() - t_inf) * 1000)
        n_det += 1

        face = pick_main_face(result)
        with st.lock:
            action = st.action_active
        if face is not None:
            n_hit += 1
            u_raw, v_raw, _h = face
            u = fx(u_raw, now)
            v = fy(v_raw, now)
            if action:
                t_prev_ctrl = now  # 手势期间冻结积分(滤波继续,保持可见性)
                with st.lock:
                    st.face_seen_at = now
            else:
                err_yaw = YAW_SIGN * (u - 0.5) * FOV_X_DEG
                err_pitch = PITCH_SIGN * (v - 0.5) * FOV_Y_DEG
                if abs(err_yaw) < TRACK_DEADBAND:
                    err_yaw = 0.0
                if abs(err_pitch) < TRACK_DEADBAND:
                    err_pitch = 0.0
                dt = max(1e-3, now - t_prev_ctrl)
                t_prev_ctrl = now
                k = 1.0 - math.exp(-dt / TRACK_TAU)
                with st.lock:
                    sy = float(np.clip(k * err_yaw, -TRACK_MAX_STEP, TRACK_MAX_STEP))
                    sp = float(np.clip(k * err_pitch, -TRACK_MAX_STEP, TRACK_MAX_STEP))
                    st.track_yaw = float(np.clip(st.track_yaw + sy, -TRACK_YAW_LIMIT, TRACK_YAW_LIMIT))
                    st.track_pitch = float(np.clip(st.track_pitch + sp, -TRACK_PITCH_LIMIT, TRACK_PITCH_LIMIT))
                    st.face_seen_at = now
        else:
            fx.reset()
            fy.reset()
            dt = max(1e-3, now - t_prev_ctrl)
            t_prev_ctrl = now
            with st.lock:
                if not action and now - st.face_seen_at > LOST_HOLD_S:
                    decay = math.exp(-dt / RETURN_TAU)  # 丢脸超时:时间常数型缓慢回中
                    st.track_yaw *= decay
                    st.track_pitch *= decay

        if now - stat_t >= 10.0:
            fps = n_det / (now - stat_t)
            avg_inf = float(np.mean(infer_acc)) if infer_acc else 0.0
            hit = 100.0 * n_hit / max(1, n_det)
            with st.lock:
                ty, tp = st.track_yaw, st.track_pitch
            log(f"👁 视觉:检测 {fps:.1f}fps|推理均值 {avg_inf:.1f}ms|检出率 {hit:.0f}%|"
                f"跟随目标 yaw={ty:+.1f}° pitch={tp:+.1f}°")
            stat_t = now
            n_det = 0
            n_hit = 0
            infer_acc = []


# ───────────────────────── 头部控制线程:三层仲裁唯一写入口 ─────────────────────────
def head_control_loop(mini: ReachyMini, st: State, stop: threading.Event) -> None:
    """25Hz set_target,头部姿态 = Tracking 目标 + Idle 微动叠加。
    Primary 手势执行中(action_active)完全让位(不发包,motion 线程 goto_target 独占);
    手势结束于跟随基准 → 本线程接管时姿态一致,无突跳。
    微动:说话时包络升起;跟随中缩到 40% 小幅叠加,无人脸时回原幅度(O-01a-2 行为)。"""
    dt = 1.0 / IDLE_HZ
    amp = 0.0
    sway_scale = 1.0
    while not stop.is_set():
        now = time.monotonic()
        with st.lock:
            action = st.action_active
            speaking = now < st.playback_end_estimate
            ty, tp = st.track_yaw, st.track_pitch
            tracked = (now - st.face_seen_at) < LOST_HOLD_S
        if action:
            amp = 0.0  # 硬让位:立即停发,包络清零
            time.sleep(dt)
            continue
        amp += ((1.0 if speaking else 0.0) - amp) * (dt / IDLE_TAU)
        target_scale = TRACK_SWAY_SCALE if tracked else 1.0
        sway_scale += (target_scale - sway_scale) * (dt / IDLE_TAU)  # 幅度切换也平滑
        sway_yaw = amp * sway_scale * IDLE_YAW_AMP * math.sin(2 * math.pi * IDLE_YAW_F * now)
        sway_pitch = amp * sway_scale * IDLE_PITCH_AMP * math.sin(2 * math.pi * IDLE_PITCH_F * now + 1.0)
        try:
            mini.set_target(head=head_pose(pitch_deg=tp + sway_pitch, yaw_deg=ty + sway_yaw))
        except Exception:
            time.sleep(1.0)
        time.sleep(dt)


# ───────────────────────── 播放线程:队列 → 扬声器 ─────────────────────────
def player_loop(mini: ReachyMini, st: State, play_q: "queue.Queue", stop: threading.Event) -> None:
    def current_gen() -> int:
        with st.lock:
            return st.play_gen

    def push(chunk: np.ndarray) -> None:
        mini.media.push_audio_sample(chunk)
        with st.lock:
            base = max(st.playback_end_estimate, time.monotonic())
            st.playback_end_estimate = base + len(chunk) / PLAY_SR

    buffering = True
    while not stop.is_set():
        try:
            gen, chunk = play_q.get(timeout=0.1)
        except queue.Empty:
            buffering = True
            continue
        if gen != current_gen():
            continue
        if buffering:
            stash = [(gen, chunk)]
            dur = len(chunk) / PLAY_SR
            t_start = time.monotonic()
            while dur < JITTER_S and time.monotonic() - t_start < JITTER_WALL_S:
                try:
                    g2, c2 = play_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if g2 != current_gen():
                    continue
                stash.append((g2, c2))
                dur += len(c2) / PLAY_SR
            g_now = current_gen()
            valid = [c for g, c in stash if g == g_now]
            if not valid:
                continue
            for c in valid:
                push(c)
            buffering = False
        else:
            push(chunk)


# ───────────────────────── 主流程 ─────────────────────────
def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        log("❌ 环境变量 DASHSCOPE_API_KEY 未配置,退出。")
        return 1
    dashscope.api_key = api_key
    oai = OpenAI(api_key=api_key, base_url=VISION_BASE_URL)  # take_snapshot 看图

    run_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else None  # 编排测试用:到时干净退出

    print("=== Reachy Mini 语音对话:可打断 + 动作工具 + 看图 + 人脸跟随 ===", flush=True)
    log(f"模型:{MODEL}|semantic_vad|16k上行|24k→16k下行|8 动作 + take_snapshot|三层仲裁")

    st = State()
    play_q: "queue.Queue" = queue.Queue()
    motion_q: "queue.Queue" = queue.Queue()
    snap_q: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    log("连接 Reachy Mini(media_backend=default, automatic_body_yaw=False)…")
    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="default",
        automatic_body_yaw=False,
    ) as mini:
        try:
            mini.media.start_recording()
            mini.media.start_playing()
            log("✅ 录音/播放管线已启动;回中立位…")
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            time.sleep(0.8)
            # 摄像头预热(顺便验证与录音管线并存)
            warm = None
            wdl = time.monotonic() + 10.0
            while warm is None and time.monotonic() < wdl:
                warm = mini.media.get_frame()
                if warm is None:
                    time.sleep(0.05)
            log(f"摄像头:{'✅ 出帧 ' + str(warm.shape) if warm is not None else '⚠ 10s 无帧(跟随/take_snapshot 可能失败)'}")

            callback = ChatCallback(st, play_q, motion_q, snap_q, mini)
            conv = OmniRealtimeConversation(model=MODEL, callback=callback)
            callback.conv = conv
            log("连接 Qwen-Omni-Realtime(北京端点)…")
            conv.connect()
            conv.update_session(
                output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                voice=VOICE,
                input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                enable_input_audio_transcription=True,
                enable_turn_detection=True,
                turn_detection_type="semantic_vad",
                instructions=INSTRUCTIONS,
                tools=TOOLS,
            )
            if not st.session_updated.wait(timeout=10):
                log("❌ 10s 内未收到 session.updated,中止")
                conv.close()
                return 1

            threading.Thread(target=player_loop, args=(mini, st, play_q, stop), daemon=True).start()
            threading.Thread(target=motion_loop, args=(mini, st, motion_q, stop), daemon=True).start()
            threading.Thread(target=vision_loop, args=(mini, st, stop), daemon=True).start()
            threading.Thread(target=head_control_loop, args=(mini, st, stop), daemon=True).start()
            threading.Thread(target=snapshot_loop, args=(mini, st, callback, oai, snap_q, stop), daemon=True).start()

            # 排空预热期旧音频(限时 3s,防排空循环被持续来帧拖死)
            drain_dl = time.monotonic() + 3.0
            while time.monotonic() < drain_dl and mini.media.get_audio_sample() is not None:
                pass

            # 主循环:麦克风 → Realtime 上行(每 10s 报一次电平,便于排查"说话没被听见")
            sent_samples = 0
            rms_acc: list[float] = []
            rms_t = time.monotonic()
            t_run0 = time.monotonic()
            try:
                while True:
                    if run_seconds is not None and time.monotonic() - t_run0 >= run_seconds:
                        log(f"⏱ 到达预设时长 {run_seconds:.0f}s,自动退出")
                        break
                    chunk = mini.media.get_audio_sample()
                    if chunk is None or len(chunk) == 0:
                        time.sleep(0.01)
                        continue
                    mono = chunk[:, 0]
                    rms_acc.append(float(np.sqrt(np.mean(mono**2))))
                    pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
                    conv.append_audio(base64.b64encode(pcm16.tobytes()).decode("ascii"))
                    sent_samples += len(mono)
                    if time.monotonic() - rms_t >= 10.0:
                        rms = float(np.mean(rms_acc)) if rms_acc else 0.0
                        if rms < 0.005:
                            log(f"🎙 近10s 上行电平偏低(RMS={rms:.4f}),说话请大声靠近")
                        rms_acc = []
                        rms_t = time.monotonic()
            except KeyboardInterrupt:
                print(flush=True)
                log(f"收到 Ctrl+C,退出。本次共上行音频 {sent_samples / 16000:.1f} 秒")
            finally:
                stop.set()
                time.sleep(0.15)  # 让 head_control 最后一帧 set_target 落地,避免与回中 goto 抢
                try:
                    conv.close()
                except Exception:
                    pass
                try:
                    mini.media.stop_recording()
                    mini.media.stop_playing()
                    mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
                except Exception:
                    pass
                log("已释放 Realtime 连接与 Reachy 媒体资源。")
        finally:
            try:
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
