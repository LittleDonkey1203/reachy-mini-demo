# -*- coding: utf-8 -*-
"""V-01-1 编排版:take_snapshot 单帧看图(chat.completions 回合制)。

在 _o01a3(对话+打断+动作+idle 微动)基础上新增,已跑通链路不动:

1. 注册 take_snapshot 工具;模型调用 → 快照 worker 线程:
   media.get_frame()(连抓几帧取最新,保证是当前画面)
   → 1080p 缩到 640×360 jpg(≤256KB)→ base64
   → chat.completions(qwen3.5-omni-plus,image_url,stream,只要文本)
   → 描述作为 function_call_output 回 Realtime → response.create 让模型语音转述。
2. 与手势的协调差异:手势=乐观即时回 output;快照=等理解结果才回。
   response.done 的"纯动作立即补话"逻辑对快照挂起的响应跳过
   (否则模型没拿到描述就开口)。
3. 抓帧与录音上行并存性验证:快照期间上行 RMS 日志应不间断,对话不中断。
4. 快照存 voice/output/ 供人工核对是否当前画面。
"""

import os

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

MODEL = "qwen3.5-omni-plus-realtime"
VISION_MODEL = "qwen3.5-omni-plus"
VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VOICE = "Ethan"
INSTRUCTIONS = (
    "你是桌面机器人 Reachy Mini,有真实的身体(头、天线)和一台摄像头。"
    "用简体中文、口语化、简短地回答。"
    "回答时自然地配合动作工具;做动作时必须同时用语音回应,边说边做。"
    "用户让你看东西时调用 take_snapshot,拿到画面描述后用自己的话自然地告诉用户你看到了什么。"
)

OUT_SR = 24000
PLAY_SR = 16000
JITTER_S = 0.30
JITTER_WALL_S = 0.50
REC_WINDOW_S = 75.0
RESPONSE_TIMEOUT_S = 25.0

IDLE_HZ = 25.0
IDLE_YAW_AMP = 2.5
IDLE_PITCH_AMP = 1.5
IDLE_YAW_F = 0.20
IDLE_PITCH_F = 0.30
IDLE_TAU = 0.5

SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:7.2f}s] {msg}", flush=True)


INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]


def head_pose(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


def act_nod(m):
    for _ in range(2):
        m.goto_target(head_pose(pitch_deg=+15), duration=0.35, body_yaw=0.0)
        m.goto_target(head_pose(pitch_deg=-10), duration=0.35, body_yaw=0.0)
    m.goto_target(INIT_HEAD_POSE, duration=0.35, body_yaw=0.0)


def act_shake(m):
    for _ in range(2):
        m.goto_target(head_pose(yaw_deg=+15), duration=0.35, body_yaw=0.0)
        m.goto_target(head_pose(yaw_deg=-15), duration=0.35, body_yaw=0.0)
    m.goto_target(INIT_HEAD_POSE, duration=0.35, body_yaw=0.0)


def _look(m, **kw):
    m.goto_target(head_pose(**kw), duration=0.6, body_yaw=0.0)
    time.sleep(0.8)
    m.goto_target(INIT_HEAD_POSE, duration=0.6, body_yaw=0.0)


def act_wiggle(m):
    for _ in range(2):
        m.goto_target(antennas=[+0.8, -0.8], duration=0.3, body_yaw=0.0)
        m.goto_target(antennas=[-0.8, +0.8], duration=0.3, body_yaw=0.0)
    m.goto_target(antennas=INIT_ANTENNAS, duration=0.35, body_yaw=0.0)


def act_tilt(m):
    m.goto_target(head_pose(roll_deg=15), duration=0.5, body_yaw=0.0)
    time.sleep(0.8)
    m.goto_target(INIT_HEAD_POSE, duration=0.5, body_yaw=0.0)


ACTIONS = {
    "nod": act_nod,
    "shake_head": act_shake,
    "look_left": lambda m: _look(m, yaw_deg=+16),
    "look_right": lambda m: _look(m, yaw_deg=-16),
    "look_up": lambda m: _look(m, pitch_deg=-16),
    "look_down": lambda m: _look(m, pitch_deg=+16),
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


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_updated = threading.Event()
        self.play_gen = 0
        self.drop_audio = False
        self.in_flight = 0
        self.playback_end_estimate = 0.0
        self.resp_audio_count = 0
        self.fc_seen_this_resp = False
        self.fc_gen = 0
        self.action_active = False
        self.snapshot_pending = 0    # 进行中的快照数(response.done 补话要跳过)
        # 留痕
        self.input_transcripts: list[str] = []
        self.reply_parts: list[str] = []
        self.fc_records: list[dict] = []
        self.motion_records: list[dict] = []
        self.snapshot_records: list[dict] = []
        self.barge_ins: list[dict] = []
        self.errors: list[dict] = []
        self.audio_delta_count = 0


class ChatCallback(OmniRealtimeCallback):
    def __init__(self, st: State, play_q, motion_q, snap_q, mini):
        self.st = st
        self.play_q = play_q
        self.motion_q = motion_q
        self.snap_q = snap_q
        self.mini = mini
        self.conv = None

    def on_open(self) -> None:
        log("✅ WebSocket 已连接")

    def on_close(self, code, msg) -> None:
        log(f"🔌 连接关闭 code={code} msg={msg}")

    def _do_barge_in(self, in_flight: bool) -> None:
        st = self.st
        with st.lock:
            st.play_gen += 1
            st.drop_audio = True
            residual = max(0.0, st.playback_end_estimate - time.monotonic())
            st.playback_end_estimate = time.monotonic()
            st.barge_ins.append({"t": round(time.monotonic() - T0, 2), "residual_s": round(residual, 2)})
        while True:
            try:
                self.play_q.get_nowait()
            except queue.Empty:
                break
        try:
            self.mini.media.audio.clear_player()
        except Exception as e:
            log(f"⚠ clear_player 失败:{e}")
        if in_flight and self.conv is not None:
            self.conv.cancel_response()
        log(f"⛔ BARGE-IN(残余 {residual:.2f}s 已清)")

    def on_event(self, event) -> None:
        st = self.st
        try:
            etype = event.get("type", "")
            now = time.monotonic()
            if etype == "session.created":
                log(f"✅ session.created id={event['session']['id']}")
            elif etype == "session.updated":
                log("✅ session.updated(9 工具含 take_snapshot)")
                st.session_updated.set()
            elif etype == "input_audio_buffer.speech_started":
                with st.lock:
                    playing = (now < st.playback_end_estimate) or (not self.play_q.empty())
                    in_flight = st.in_flight > 0
                log(f"🎤 speech_started(播放中={playing},生成中={in_flight})")
                if playing or in_flight:
                    self._do_barge_in(in_flight)
            elif etype == "input_audio_buffer.speech_stopped":
                log("🤫 speech_stopped")
            elif etype == "conversation.item.input_audio_transcription.completed":
                t = (event.get("transcript") or "").strip()
                with st.lock:
                    st.input_transcripts.append(t)
                log(f"📝 输入转写:「{t}」")
            elif etype == "response.created":
                with st.lock:
                    st.in_flight += 1
                    st.drop_audio = False
                    st.resp_audio_count = 0
                    st.fc_seen_this_resp = False
                log("💭 response.created")
            elif etype == "response.function_call_arguments.done":
                name = event.get("name", "")
                call_id = event.get("call_id", "")
                with st.lock:
                    st.fc_seen_this_resp = True
                    st.fc_gen = st.play_gen
                    st.fc_records.append({"t": round(now - T0, 2), "name": name})
                log(f"🤖 模型调用工具: {name}")
                if name == "take_snapshot":
                    # 快照:等图像理解结果才回 output(worker 完成后回 + 补话)
                    with st.lock:
                        st.snapshot_pending += 1
                    self.snap_q.put({"call_id": call_id, "gen": st.fc_gen})
                else:
                    # 手势:乐观即时回 output
                    self.motion_q.put({"name": name, "call_id": call_id})
                    try:
                        self.conv.create_item({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"success": True, "action": name}, ensure_ascii=False),
                        })
                    except Exception as e:
                        log(f"⚠ 回 output 失败:{e}")
            elif etype == "response.audio_transcript.delta":
                with st.lock:
                    st.reply_parts.append(event.get("delta", ""))
            elif etype == "response.audio.delta":
                with st.lock:
                    if st.drop_audio:
                        return
                    gen = st.play_gen
                    st.resp_audio_count += 1
                    st.audio_delta_count += 1
                b64 = event.get("delta") or event.get("audio") or ""
                pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                f16k = resample_poly(pcm.astype(np.float32) / 32768.0, PLAY_SR, OUT_SR).astype(np.float32)
                self.play_q.put((gen, f16k))
            elif etype == "response.done":
                fire_rc = False
                with st.lock:
                    st.in_flight = max(0, st.in_flight - 1)
                    # 纯动作响应立即补话——但快照挂起时跳过(等描述回来再说)
                    if (
                        st.fc_seen_this_resp
                        and st.resp_audio_count == 0
                        and st.fc_gen == st.play_gen
                        and st.snapshot_pending == 0
                    ):
                        fire_rc = True
                log(f"✅ response.done{'(快照挂起,等理解结果)' if st.snapshot_pending else ''}")
                if fire_rc and self.conv is not None:
                    self.conv.create_response()
            elif etype == "error":
                with st.lock:
                    st.errors.append(event)
                log(f"❌ error 事件:{event}")
        except Exception as e:
            log(f"❌ on_event 异常 {type(e).__name__}: {e} | 事件:{str(event)[:300]}")


# ───────────────────────── 快照 worker:抓帧 → 看图 → 回结果 ─────────────────────────
def snapshot_loop(mini, st: State, cb: ChatCallback, oai: OpenAI, snap_q, stop: threading.Event) -> None:
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
        log("📸 拍照:抓取当前画面…")
        # 连抓几帧取最新(appsink drop=True 只留最新,多抓保证非旧帧/黑帧)
        frame = None
        got = 0
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and got < 3:
            f = mini.media.get_frame()
            if f is not None:
                frame = f
                got += 1
            else:
                time.sleep(0.02)
        capture_ms = (time.monotonic() - t0) * 1000
        desc = ""
        ok = frame is not None
        if not ok:
            log("❌ 3s 内未抓到有效帧")
            desc = "拍照失败,没有抓到画面。"
        else:
            # BGR→RGB,1080p→640×360,jpg
            img = Image.fromarray(frame[:, :, ::-1]).resize((640, 360))
            jpg_path = os.path.join(SNAP_DIR, f"v01_snapshot_{snap_idx:02d}.jpg")
            img.save(jpg_path, "JPEG", quality=85)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            jpg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            kb = len(buf.getvalue()) / 1024
            log(f"📸 抓到 {frame.shape[1]}×{frame.shape[0]} 帧(第{got}抓,{capture_ms:.0f}ms),压成 640×360 jpg {kb:.0f}KB → 已存 {os.path.basename(jpg_path)}")
            # 单发 chat.completions 看图
            t1 = time.monotonic()
            try:
                comp = oai.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{jpg_b64}"}},
                        {"type": "text",
                         "text": "你是机器人的眼睛。用简体中文两三句话描述画面主要内容,特别是人手里举着或拿着的物体(若有)。"},
                    ]}],
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={"modalities": ["text"]},
                )
                parts = []
                for chunk in comp:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                desc = "".join(parts).strip()
                llm_ms = (time.monotonic() - t1) * 1000
                log(f"🖼 图像理解({llm_ms:.0f}ms):「{desc}」")
            except Exception as e:
                ok = False
                desc = f"看图服务调用失败:{type(e).__name__}"
                log(f"❌ chat.completions 失败:{type(e).__name__}: {e}")
        total_ms = (time.monotonic() - t0) * 1000
        with st.lock:
            st.snapshot_records.append({
                "t": round(t0 - T0, 2), "ok": ok, "capture_ms": round(capture_ms),
                "total_ms": round(total_ms), "desc": desc[:100],
            })
        # 回 output;没被打断则补话让模型转述
        fire_rc = False
        with st.lock:
            st.snapshot_pending = max(0, st.snapshot_pending - 1)
            fire_rc = st.play_gen == gen0
        try:
            cb.conv.create_item({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"success": ok, "scene_description": desc}, ensure_ascii=False),
            })
            log(f"↩ 已回 function_call_output(take_snapshot,耗时 {total_ms:.0f}ms)")
        except Exception as e:
            log(f"⚠ 回 output 失败:{e}")
            continue
        if fire_rc:
            try:
                cb.conv.create_response()
                log("▶ 已发 response.create(让模型语音转述所见)")
            except Exception as e:
                log(f"⚠ response.create 失败:{e}")


def motion_loop(mini, st: State, motion_q, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            job = motion_q.get(timeout=0.1)
        except queue.Empty:
            continue
        name = job["name"]
        fn = ACTIONS.get(name)
        with st.lock:
            st.action_active = True
        t_start = time.monotonic()
        with st.lock:
            speaking = t_start < st.playback_end_estimate
        log(f"🦾 动作开始: {name}(此刻{'正在' if speaking else '没在'}说话)")
        ok = True
        try:
            if fn is None:
                ok = False
            else:
                fn(mini)
        except Exception as e:
            ok = False
            log(f"⚠ 动作 {name} 失败:{e}")
        finally:
            with st.lock:
                st.action_active = False
        with st.lock:
            st.motion_records.append({"name": name, "t_start": round(t_start - T0, 2), "ok": ok})
        log(f"✅ 动作完成: {name}")


def idle_sway_loop(mini, st: State, stop: threading.Event) -> None:
    dt = 1.0 / IDLE_HZ
    amp = 0.0
    active = False
    while not stop.is_set():
        now = time.monotonic()
        with st.lock:
            speaking = now < st.playback_end_estimate
            action = st.action_active
        if action:
            amp = 0.0
            active = False
            time.sleep(dt)
            continue
        target = 1.0 if speaking else 0.0
        amp += (target - amp) * (dt / IDLE_TAU)
        if amp > 0.02:
            yaw = amp * IDLE_YAW_AMP * math.sin(2 * math.pi * IDLE_YAW_F * now)
            pitch = amp * IDLE_PITCH_AMP * math.sin(2 * math.pi * IDLE_PITCH_F * now + 1.0)
            try:
                mini.set_target(head=head_pose(pitch_deg=pitch, yaw_deg=yaw))
            except Exception:
                time.sleep(1.0)
            active = True
        elif active:
            try:
                mini.set_target(head=INIT_HEAD_POSE)
            except Exception:
                pass
            active = False
        time.sleep(dt)


def player_loop(mini, st: State, play_q, stop: threading.Event) -> None:
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


def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        log("❌ 无 DASHSCOPE_API_KEY")
        return 1
    dashscope.api_key = api_key
    oai = OpenAI(api_key=api_key, base_url=VISION_BASE_URL)

    print("=== V-01-1 编排版:take_snapshot 单帧看图 ===", flush=True)
    st = State()
    play_q: "queue.Queue" = queue.Queue()
    motion_q: "queue.Queue" = queue.Queue()
    snap_q: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    log("连接 Reachy Mini…")
    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="default",
        automatic_body_yaw=False,
    ) as mini:
        try:
            mini.media.start_recording()
            mini.media.start_playing()
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            time.sleep(0.8)
            # 摄像头预热:确认能出帧(顺便验证与录音并存)
            warm = None
            wdl = time.monotonic() + 10.0
            while warm is None and time.monotonic() < wdl:
                warm = mini.media.get_frame()
                if warm is None:
                    time.sleep(0.05)
            log(f"摄像头预热:{'✅ 出帧 ' + str(warm.shape) if warm is not None else '❌ 10s 无帧'}")

            cb = ChatCallback(st, play_q, motion_q, snap_q, mini)
            conv = OmniRealtimeConversation(model=MODEL, callback=cb)
            cb.conv = conv
            log("连接 Qwen-Omni-Realtime…")
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
                log("❌ session.updated 超时,中止")
                conv.close()
                return 1

            threading.Thread(target=player_loop, args=(mini, st, play_q, stop), daemon=True).start()
            threading.Thread(target=motion_loop, args=(mini, st, motion_q, stop), daemon=True).start()
            threading.Thread(target=idle_sway_loop, args=(mini, st, stop), daemon=True).start()
            threading.Thread(target=snapshot_loop, args=(mini, st, cb, oai, snap_q, stop), daemon=True).start()

            drain_dl = time.monotonic() + 3.0  # 排空旧音频(限时,防上次那种 40s 排不完)
            while time.monotonic() < drain_dl and mini.media.get_audio_sample() is not None:
                pass

            log(f"READY_FOR_SPEECH(录音窗口 {REC_WINDOW_S:.0f}s 现在打开)")
            win_end = time.monotonic() + REC_WINDOW_S
            sent = 0
            rms_acc: list[float] = []
            rms_t = time.monotonic()
            while time.monotonic() < win_end:
                chunk = mini.media.get_audio_sample()
                if chunk is None or len(chunk) == 0:
                    time.sleep(0.01)
                    continue
                mono = chunk[:, 0]
                rms_acc.append(float(np.sqrt(np.mean(mono**2))))
                pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
                conv.append_audio(base64.b64encode(pcm16.tobytes()).decode("ascii"))
                sent += len(mono)
                if time.monotonic() - rms_t >= 5.0:
                    rms = float(np.mean(rms_acc)) if rms_acc else 0.0
                    log(f"🎙 近5s 上行 RMS={rms:.4f}({'有声' if rms > 0.005 else '偏弱,请大声靠近'})")
                    rms_acc = []
                    rms_t = time.monotonic()
            log(f"录音窗口关闭,共上行 {sent / PLAY_SR:.1f}s;补 1.5s 静音帧")
            silence = base64.b64encode(np.zeros(8000, dtype=np.int16).tobytes()).decode("ascii")
            for _ in range(3):
                conv.append_audio(silence)
                time.sleep(0.5)

            deadline = time.monotonic() + RESPONSE_TIMEOUT_S
            while time.monotonic() < deadline:
                with st.lock:
                    pending = st.in_flight > 0 or st.snapshot_pending > 0
                if not pending and play_q.empty() and motion_q.empty() and snap_q.empty():
                    break
                time.sleep(0.2)

            with st.lock:
                tail = st.playback_end_estimate - time.monotonic()
            if tail > 0:
                log(f"等待播放余量 {tail:.1f}s…")
                time.sleep(tail + 1.5)

            stop.set()
            time.sleep(0.2)
            try:
                conv.close()
            except Exception:
                pass
            mini.media.stop_recording()
            mini.media.stop_playing()
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
        finally:
            try:
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass

    # ── 汇总 ──
    print("\n========== 汇总 ==========", flush=True)
    print(f"输入转写:{st.input_transcripts}", flush=True)
    print(f"回复全文:「{''.join(st.reply_parts).strip()[:300]}」", flush=True)
    print(f"工具调用:{[(r['t'], r['name']) for r in st.fc_records]}", flush=True)
    print("快照记录:", flush=True)
    for r in st.snapshot_records:
        print(f"  t+{r['t']}s ok={r['ok']} 抓帧{r['capture_ms']}ms 总耗时{r['total_ms']}ms", flush=True)
        print(f"    描述:「{r['desc']}」", flush=True)
    print(f"动作执行:{[(r['t_start'], r['name']) for r in st.motion_records]}", flush=True)
    print(f"barge-in:{st.barge_ins if st.barge_ins else '无'}|error:{st.errors if st.errors else '无'}", flush=True)
    snaps_ok = bool(st.snapshot_records) and all(r["ok"] for r in st.snapshot_records)
    ok = snaps_ok and not st.errors
    print(f"=== {'看图闭环成功' if ok else '未达验收,看快照记录'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
