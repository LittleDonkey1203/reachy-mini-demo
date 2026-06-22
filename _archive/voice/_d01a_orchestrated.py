# -*- coding: utf-8 -*-
"""D-01a 编排版:固定时序跑一轮语音闭环,全程留痕,结束输出汇总日志。

时序:
  setup(daemon + 录音/播放管线 + Realtime 连接 + session 配置)
  → 打印 READY_FOR_SPEECH 标记(外部据此喊"现在说话")
  → 25 秒录音窗口:麦克风持续上行,semantic_vad 自动判说话起止
  → 窗口结束补 1.5s 静音帧(防止话音收在窗口边缘 VAD 不闭合)
  → 等 response.done(最多 20s)→ 等播放排空 → 输出汇总
"""

import os

_no_proxy = "localhost,127.0.0.1,::1,.aliyuncs.com,aliyuncs.com"
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

import base64
import queue
import sys
import threading
import time

import numpy as np
from scipy.signal import resample_poly

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
INSTRUCTIONS = "你是桌面机器人 Reachy Mini。用简体中文、口语化、简短地回答,一般不超过两三句话。"

OUT_SR = 24000
PLAY_SR = 16000
REC_WINDOW_S = 25.0       # 录音窗口
RESPONSE_TIMEOUT_S = 20.0  # 窗口后等回复

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:7.2f}s] {msg}", flush=True)


class State:
    """跨线程共享的事件账本。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_updated = threading.Event()
        self.speech_started_times: list[float] = []
        self.speech_stopped_times: list[float] = []
        self.response_done_times: list[float] = []
        self.input_transcripts: list[str] = []
        self.reply_transcript_parts: list[str] = []
        self.errors: list[dict] = []
        self.first_audio_delay_ms: float | None = None
        self.audio_delta_count = 0
        # 播放进度估计(用于判断"机器人正在出声"的时间段)
        self.playback_end_estimate = 0.0  # monotonic 时刻


class ChatCallback(OmniRealtimeCallback):
    def __init__(self, st: State, play_q: "queue.Queue[np.ndarray]"):
        self.st = st
        self.play_q = play_q
        self.conv: OmniRealtimeConversation | None = None

    def on_open(self) -> None:
        log("✅ WebSocket 已连接")

    def on_close(self, code, msg) -> None:
        log(f"🔌 连接关闭 code={code} msg={msg}")

    def on_event(self, event) -> None:
        st = self.st
        try:
            etype = event.get("type", "")
            now = time.monotonic()
            if etype == "session.created":
                log(f"✅ session.created id={event['session']['id']}")
            elif etype == "session.updated":
                log("✅ session.updated(配置生效)")
                st.session_updated.set()
            elif etype == "input_audio_buffer.speech_started":
                with st.lock:
                    st.speech_started_times.append(now)
                    playing = now < st.playback_end_estimate
                log(f"🎤 speech_started(此刻机器人{'正在' if playing else '没在'}出声)")
            elif etype == "input_audio_buffer.speech_stopped":
                with st.lock:
                    st.speech_stopped_times.append(now)
                log("🤫 speech_stopped")
            elif etype == "conversation.item.input_audio_transcription.completed":
                t = (event.get("transcript") or "").strip()
                with st.lock:
                    st.input_transcripts.append(t)
                log(f"📝 输入转写:「{t}」")
            elif etype == "response.created":
                log("💭 response.created")
            elif etype == "response.audio_transcript.delta":
                with st.lock:
                    st.reply_transcript_parts.append(event.get("delta", ""))
            elif etype == "response.audio.delta":
                b64 = event.get("delta") or event.get("audio") or ""
                pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                f32 = pcm.astype(np.float32) / 32768.0
                f16k = resample_poly(f32, PLAY_SR, OUT_SR).astype(np.float32)
                with st.lock:
                    st.audio_delta_count += 1
                    if st.audio_delta_count == 1:
                        log("🔉 收到首个 response.audio.delta")
                self.play_q.put(f16k)
            elif etype == "response.done":
                with st.lock:
                    st.response_done_times.append(now)
                if self.conv is not None:
                    st.first_audio_delay_ms = self.conv.get_last_first_audio_delay()
                log("✅ response.done")
            elif etype == "error":
                with st.lock:
                    st.errors.append(event)
                log(f"❌ error 事件:{event}")
        except Exception as e:
            log(f"❌ on_event 异常 {type(e).__name__}: {e} | 事件:{str(event)[:300]}")


def player_loop(mini: ReachyMini, st: State, play_q: "queue.Queue[np.ndarray]", stop: threading.Event) -> None:
    logged = False
    while not stop.is_set():
        try:
            chunk = play_q.get(timeout=0.1)
        except queue.Empty:
            logged = False
            continue
        if not logged:
            log("🔊 开始向机器人扬声器推流播放")
            logged = True
        mini.media.push_audio_sample(chunk)
        dur = len(chunk) / PLAY_SR
        with st.lock:
            base = max(st.playback_end_estimate, time.monotonic())
            st.playback_end_estimate = base + dur


def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        log("❌ 无 DASHSCOPE_API_KEY")
        return 1
    dashscope.api_key = api_key

    print("=== D-01a 编排版:单轮语音闭环 ===", flush=True)
    st = State()
    play_q: "queue.Queue[np.ndarray]" = queue.Queue()
    stop = threading.Event()

    log("连接 Reachy Mini(media_backend=default)…")
    with ReachyMini(connection_mode="localhost_only", media_backend="default") as mini:
        mini.media.start_recording()
        mini.media.start_playing()
        log("录音/播放管线已启动,预热 1.5s…")
        time.sleep(1.5)

        cb = ChatCallback(st, play_q)
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
        )
        if not st.session_updated.wait(timeout=10):
            log("❌ 10s 内未收到 session.updated,中止")
            conv.close()
            return 1

        player = threading.Thread(target=player_loop, args=(mini, st, play_q, stop), daemon=True)
        player.start()

        # 丢弃预热期囤的旧音频
        while mini.media.get_audio_sample() is not None:
            pass

        log(f"READY_FOR_SPEECH(录音窗口 {REC_WINDOW_S:.0f}s 现在打开)")
        win_end = time.monotonic() + REC_WINDOW_S
        sent = 0
        while time.monotonic() < win_end:
            chunk = mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue
            mono = chunk[:, 0]
            pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
            conv.append_audio(base64.b64encode(pcm16.tobytes()).decode("ascii"))
            sent += len(mono)
        log(f"录音窗口关闭,共上行 {sent / PLAY_SR:.1f}s 音频;补 1.5s 静音帧")
        silence = base64.b64encode(np.zeros(8000, dtype=np.int16).tobytes()).decode("ascii")
        for _ in range(3):  # 3 × 0.5s
            conv.append_audio(silence)
            time.sleep(0.5)

        # 等在途回复完成:speech_started 数 > response.done 数 → 还有未完成轮次
        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        while time.monotonic() < deadline:
            with st.lock:
                pending = len(st.speech_started_times) > len(st.response_done_times)
            if not pending:
                break
            time.sleep(0.2)

        # 等播放排空
        while not play_q.empty():
            time.sleep(0.2)
        with st.lock:
            tail = st.playback_end_estimate - time.monotonic()
        if tail > 0:
            log(f"等待播放余量 {tail:.1f}s…")
            time.sleep(tail + 1.0)

        stop.set()
        try:
            conv.close()
        except Exception:
            pass
        mini.media.stop_recording()
        mini.media.stop_playing()

    # ── 汇总 ──
    print("\n========== 汇总 ==========", flush=True)
    reply = "".join(st.reply_transcript_parts).strip()
    print(f"speech_started 次数:{len(st.speech_started_times)}", flush=True)
    print(f"speech_stopped 次数:{len(st.speech_stopped_times)}", flush=True)
    print(f"response.done 次数:{len(st.response_done_times)}", flush=True)
    print(f"输入转写:{st.input_transcripts}", flush=True)
    print(f"回复文本:「{reply}」", flush=True)
    fad = st.first_audio_delay_ms
    print(f"首音频延迟:{fad:.0f}ms" if fad else "首音频延迟:N/A", flush=True)
    print(f"audio.delta 块数:{st.audio_delta_count}", flush=True)
    print(f"error 事件:{st.errors if st.errors else '无'}", flush=True)
    ok = bool(st.input_transcripts) and bool(reply) and st.audio_delta_count > 0 and not st.errors
    print(f"=== {'闭环成功' if ok else '闭环未完成,见上方日志'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
