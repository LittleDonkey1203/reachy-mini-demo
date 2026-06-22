# -*- coding: utf-8 -*-
"""O-01a-2 编排版:说话时 idle 微动(secondary 微动层雏形)。

在 O-01a-1(动作工具)基础上新增 idle 线程,其余链路不动:

- 25Hz tick,用 set_target 连续控制(不与 goto_target 插值打架)。
- 姿态 = 幅度包络 × 双频慢正弦:yaw ±2.5°@0.20Hz、pitch ±1.5°@0.30Hz,
  两轴不同频不同相 → 缓慢自然游移,不是抖。
- 包络 τ≈0.5s 平滑升降:开始说话渐起,说完渐落;正弦×包络→0 自然滑回中立。
- 明确动作优先:motion 线程执行期间置 action_active,idle 立即停发指令(让位),
  动作结束且仍在说话则渐起恢复。
- "在说话"判据复用 playback_end_estimate。
"""

import os

_no_proxy = "localhost,127.0.0.1,::1,.aliyuncs.com,aliyuncs.com"
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

import base64
import json
import math
import queue
import sys
import threading
import time

import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation as R

import dashscope
from dashscope.audio.qwen_omni import (
    AudioFormat,
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)
from reachy_mini import ReachyMini

MODEL = "qwen3.5-omni-plus-realtime"
VOICE = "Ethan"
INSTRUCTIONS = (
    "你是桌面机器人 Reachy Mini,有真实的身体(头、天线)。"
    "用简体中文、口语化回答。让你自我介绍或讲东西时可以说四五句话,稍微展开。"
    "回答时自然地配合动作工具:打招呼/同意时点头,否定时摇头,开心时摆天线,好奇时歪头。"
)

OUT_SR = 24000
PLAY_SR = 16000
JITTER_S = 0.30
JITTER_WALL_S = 0.50
REC_WINDOW_S = 60.0
RESPONSE_TIMEOUT_S = 20.0

# idle 微动参数
IDLE_HZ = 25.0          # tick 频率
IDLE_YAW_AMP = 2.5      # 度
IDLE_PITCH_AMP = 1.5    # 度
IDLE_YAW_F = 0.20       # Hz
IDLE_PITCH_F = 0.30     # Hz
IDLE_TAU = 0.5          # 包络时间常数(s)

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:7.2f}s] {msg}", flush=True)


INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]


def head_pose(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


def act_nod(m: ReachyMini) -> None:
    for _ in range(2):
        m.goto_target(head_pose(pitch_deg=+10), duration=0.35, body_yaw=0.0)
        m.goto_target(head_pose(pitch_deg=-6), duration=0.35, body_yaw=0.0)
    m.goto_target(INIT_HEAD_POSE, duration=0.35, body_yaw=0.0)


def act_shake(m: ReachyMini) -> None:
    for _ in range(2):
        m.goto_target(head_pose(yaw_deg=+10), duration=0.35, body_yaw=0.0)
        m.goto_target(head_pose(yaw_deg=-10), duration=0.35, body_yaw=0.0)
    m.goto_target(INIT_HEAD_POSE, duration=0.35, body_yaw=0.0)


def _look(m: ReachyMini, **kw) -> None:
    m.goto_target(head_pose(**kw), duration=0.6, body_yaw=0.0)
    time.sleep(0.8)
    m.goto_target(INIT_HEAD_POSE, duration=0.6, body_yaw=0.0)


def act_wiggle(m: ReachyMini) -> None:
    for _ in range(2):
        m.goto_target(antennas=[+0.5, -0.5], duration=0.3, body_yaw=0.0)
        m.goto_target(antennas=[-0.5, +0.5], duration=0.3, body_yaw=0.0)
    m.goto_target(antennas=INIT_ANTENNAS, duration=0.35, body_yaw=0.0)


def act_tilt(m: ReachyMini) -> None:
    m.goto_target(head_pose(roll_deg=12), duration=0.5, body_yaw=0.0)
    time.sleep(0.8)
    m.goto_target(INIT_HEAD_POSE, duration=0.5, body_yaw=0.0)


ACTIONS = {
    "nod": act_nod,
    "shake_head": act_shake,
    "look_left": lambda m: _look(m, yaw_deg=+12),
    "look_right": lambda m: _look(m, yaw_deg=-12),
    "look_up": lambda m: _look(m, pitch_deg=-12),
    "look_down": lambda m: _look(m, pitch_deg=+12),
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
        self.fc_pending = 0
        self.fc_seen_this_resp = False
        self.fc_response_done = False
        self.fc_needs_rc = False
        self.fc_rc_sent = False
        self.fc_gen = 0
        self.action_active = False     # ★ 明确动作执行中,idle 让位
        # 留痕
        self.input_transcripts: list[str] = []
        self.reply_parts: list[str] = []
        self.fc_records: list[dict] = []
        self.motion_records: list[dict] = []
        self.idle_records: list[dict] = []   # idle 微动起停/让位时间线
        self.barge_ins: list[dict] = []
        self.errors: list[dict] = []
        self.audio_delta_count = 0


class ChatCallback(OmniRealtimeCallback):
    def __init__(self, st: State, play_q: "queue.Queue", motion_q: "queue.Queue", mini: ReachyMini):
        self.st = st
        self.play_q = play_q
        self.motion_q = motion_q
        self.mini = mini
        self.conv: OmniRealtimeConversation | None = None

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

    def _maybe_continue_locked(self) -> bool:
        st = self.st
        if (
            st.fc_pending == 0
            and st.fc_response_done
            and st.fc_needs_rc
            and not st.fc_rc_sent
            and st.fc_gen == st.play_gen
        ):
            st.fc_rc_sent = True
            return True
        return False

    def on_event(self, event) -> None:
        st = self.st
        try:
            etype = event.get("type", "")
            now = time.monotonic()
            if etype == "session.created":
                log(f"✅ session.created id={event['session']['id']}")
            elif etype == "session.updated":
                log("✅ session.updated(工具 + idle 微动就绪)")
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
                    if st.fc_pending == 0:
                        st.fc_seen_this_resp = False
                        st.fc_response_done = False
                        st.fc_needs_rc = False
                        st.fc_rc_sent = False
                log("💭 response.created")
            elif etype == "response.function_call_arguments.done":
                name = event.get("name", "")
                call_id = event.get("call_id", "")
                with st.lock:
                    st.fc_pending += 1
                    st.fc_seen_this_resp = True
                    st.fc_gen = st.play_gen
                    st.fc_records.append({"t": round(now - T0, 2), "name": name})
                log(f"🤖 模型调用动作: {name}")
                self.motion_q.put({"name": name, "call_id": call_id})
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
                    if st.fc_seen_this_resp and not st.fc_response_done:
                        st.fc_response_done = True
                        st.fc_needs_rc = st.resp_audio_count == 0
                        fire_rc = self._maybe_continue_locked()
                log("✅ response.done")
                if fire_rc and self.conv is not None:
                    self.conv.create_response()
                    log("▶ 已发 response.create(纯动作响应)")
            elif etype == "error":
                with st.lock:
                    st.errors.append(event)
                log(f"❌ error 事件:{event}")
        except Exception as e:
            log(f"❌ on_event 异常 {type(e).__name__}: {e} | 事件:{str(event)[:300]}")


def motion_loop(mini: ReachyMini, st: State, cb: ChatCallback, motion_q: "queue.Queue", stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            job = motion_q.get(timeout=0.1)
        except queue.Empty:
            continue
        name, call_id = job["name"], job["call_id"]
        fn = ACTIONS.get(name)
        with st.lock:
            st.action_active = True   # ★ idle 让位
        t_start = time.monotonic()
        with st.lock:
            speaking = t_start < st.playback_end_estimate
        log(f"🦾 动作开始: {name}(此刻机器人{'正在' if speaking else '没在'}说话)")
        ok = True
        try:
            if fn is None:
                ok = False
            else:
                fn(mini)
        except Exception as e:
            ok = False
            log(f"⚠ 动作 {name} 执行失败:{e}")
        finally:
            with st.lock:
                st.action_active = False  # ★ idle 可恢复
        dur = time.monotonic() - t_start
        with st.lock:
            st.motion_records.append(
                {"name": name, "t_start": round(t_start - T0, 2), "dur_s": round(dur, 2),
                 "speaking_at_start": speaking, "ok": ok}
            )
        log(f"✅ 动作完成: {name}({dur:.1f}s)")
        fire_rc = False
        try:
            cb.conv.create_item({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"success": ok, "action": name}, ensure_ascii=False),
            })
            with st.lock:
                st.fc_pending = max(0, st.fc_pending - 1)
                fire_rc = cb._maybe_continue_locked()
            log(f"↩ 已回 function_call_output({name})")
        except Exception as e:
            log(f"⚠ 回 output 失败:{e}")
        if fire_rc:
            try:
                cb.conv.create_response()
                log("▶ 已发 response.create(纯动作响应)")
            except Exception as e:
                log(f"⚠ response.create 失败:{e}")


def idle_sway_loop(mini: ReachyMini, st: State, stop: threading.Event) -> None:
    """说话时的低幅慢速头部游移(secondary 微动层)。

    amp 包络 0→1 渐起 / 1→0 渐落;明确动作执行中硬让位(立即停发并清零包络)。
    """
    dt = 1.0 / IDLE_HZ
    amp = 0.0
    active = False       # 当前是否在发 set_target
    state_name = "off"   # off / sway / yield
    while not stop.is_set():
        now = time.monotonic()
        with st.lock:
            speaking = now < st.playback_end_estimate
            action = st.action_active
        if action:
            if state_name != "yield":
                st.idle_records.append({"t": round(now - T0, 2), "event": "让位(明确动作)"})
                log("〰 idle 让位(明确动作优先)")
                state_name = "yield"
            amp = 0.0       # 硬让位:立即停发,包络清零
            active = False
            time.sleep(dt)
            continue
        target = 1.0 if speaking else 0.0
        amp += (target - amp) * (dt / IDLE_TAU)
        if amp > 0.02:
            t = now - T0
            yaw = amp * IDLE_YAW_AMP * math.sin(2 * math.pi * IDLE_YAW_F * t)
            pitch = amp * IDLE_PITCH_AMP * math.sin(2 * math.pi * IDLE_PITCH_F * t + 1.0)
            try:
                mini.set_target(head=head_pose(pitch_deg=pitch, yaw_deg=yaw))
            except Exception as e:
                log(f"⚠ idle set_target 失败:{e}")
                time.sleep(1.0)
            if not active:
                st.idle_records.append({"t": round(now - T0, 2), "event": "微动开始(说话中)"})
                log("〰 idle 微动开始(说话中)")
                active = True
                state_name = "sway"
        else:
            if active:
                # 包络已衰减到位,补一帧精确中立再停发
                try:
                    mini.set_target(head=INIT_HEAD_POSE)
                except Exception:
                    pass
                st.idle_records.append({"t": round(now - T0, 2), "event": "微动停止(回中立)"})
                log("〰 idle 微动停止(说完,回中立)")
                active = False
                state_name = "off"
        time.sleep(dt)


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


def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        log("❌ 无 DASHSCOPE_API_KEY")
        return 1
    dashscope.api_key = api_key

    print("=== O-01a-2 编排版:说话时 idle 微动 ===", flush=True)
    st = State()
    play_q: "queue.Queue" = queue.Queue()
    motion_q: "queue.Queue" = queue.Queue()
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
            log("录音/播放管线已启动;回中立位…")
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            time.sleep(0.8)

            cb = ChatCallback(st, play_q, motion_q, mini)
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
            threading.Thread(target=motion_loop, args=(mini, st, cb, motion_q, stop), daemon=True).start()
            threading.Thread(target=idle_sway_loop, args=(mini, st, stop), daemon=True).start()

            while mini.media.get_audio_sample() is not None:
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
                    pending = st.in_flight > 0 or st.fc_pending > 0
                if not pending and play_q.empty() and motion_q.empty():
                    break
                time.sleep(0.2)

            with st.lock:
                tail = st.playback_end_estimate - time.monotonic()
            if tail > 0:
                log(f"等待播放余量 {tail:.1f}s…")
                time.sleep(tail + 1.5)  # 多等 0.5s 让 idle 包络落完

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
    print(f"回复全文:「{''.join(st.reply_parts).strip()[:200]}」", flush=True)
    print(f"动作调用:{[(r['t'], r['name']) for r in st.fc_records]}", flush=True)
    print("idle 微动时间线:", flush=True)
    for r in st.idle_records:
        print(f"  t+{r['t']}s {r['event']}", flush=True)
    print(f"动作执行:{[(r['t_start'], r['name'], r['dur_s']) for r in st.motion_records]}", flush=True)
    print(f"barge-in:{st.barge_ins if st.barge_ins else '无'}|error:{st.errors if st.errors else '无'}", flush=True)
    sway_started = any("微动开始" in r["event"] for r in st.idle_records)
    sway_stopped = any("微动停止" in r["event"] for r in st.idle_records)
    ok = sway_started and sway_stopped and not st.errors
    print(f"=== {'idle 微动闭环成功' if ok else '未达验收,看时间线'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
