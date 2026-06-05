# -*- coding: utf-8 -*-
"""D-01b 编排版:barge-in 打断 + 播放抖动缓冲,固定时序单次测试。

在 D-01a 闭环(连接/上行/VAD/重采样不动)基础上新增:

1. barge-in:播放中(或回复生成中)收到 speech_started →
   - 丢弃 Python 播放队列里的待播块(代际计数器作废旧块)
   - mini.media.audio.clear_player():flush GStreamer appsrc,残余立即静音
   - 回复仍在生成则 cancel_response()
   - 此后到下一个 response.created 之间的 audio.delta 全部丢弃(在途旧块)
2. 抖动缓冲:每段回复先攒 ~300ms(或 0.5s 兜底超时)再开播,减少句中停顿。

测试时序:
  READY_FOR_SPEECH → 用户让机器人讲长内容 → 机器人开讲 →
  用户中途插话 → 应立即闭嘴并回应新话 → 窗口结束输出汇总
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
# 打断测试需要长回复,放开篇幅限制
INSTRUCTIONS = "你是桌面机器人 Reachy Mini。用简体中文口语化回答。用户让你讲故事或长内容时,就放开篇幅认真讲。"

OUT_SR = 24000
PLAY_SR = 16000
REC_WINDOW_S = 45.0        # 比 D-01a 长:要容纳 提问→开讲→打断→新回复→播完
RESPONSE_TIMEOUT_S = 20.0
JITTER_S = 0.30            # 开播前攒的缓冲时长
JITTER_WALL_S = 0.50       # 攒不够 300ms 时的兜底等待上限

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:7.2f}s] {msg}", flush=True)


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_updated = threading.Event()
        self.speech_started_times: list[float] = []
        self.response_created_count = 0
        self.response_done_times: list[float] = []
        self.input_transcripts: list[str] = []
        self.reply_parts: list[str] = []          # (代际, 文本) 拼出每轮回复
        self.reply_gen_marks: list[int] = []
        self.errors: list[dict] = []
        self.first_audio_delay_ms: float | None = None
        self.audio_delta_count = 0
        self.dropped_delta_count = 0
        self.playback_end_estimate = 0.0
        # barge-in
        self.play_gen = 0          # 代际:打断时 +1,旧代际音频块作废
        self.drop_audio = False    # 打断后到下一个 response.created 前丢弃 delta
        self.barge_ins: list[dict] = []


class ChatCallback(OmniRealtimeCallback):
    def __init__(self, st: State, play_q: "queue.Queue", mini: ReachyMini):
        self.st = st
        self.play_q = play_q
        self.mini = mini
        self.conv: OmniRealtimeConversation | None = None

    def on_open(self) -> None:
        log("✅ WebSocket 已连接")

    def on_close(self, code, msg) -> None:
        log(f"🔌 连接关闭 code={code} msg={msg}")

    def _do_barge_in(self, now: float) -> None:
        """打断:作废队列 → flush 管线 → 必要时 cancel 在途回复。"""
        st = self.st
        with st.lock:
            in_flight = st.response_created_count > len(st.response_done_times)
            residual = max(0.0, st.playback_end_estimate - now)
            st.play_gen += 1
            st.drop_audio = True
            st.playback_end_estimate = now
            st.barge_ins.append(
                {"t": now - T0, "in_flight": in_flight, "residual_s": round(residual, 2)}
            )
        # 1) 清 Python 播放队列
        drained = 0
        while True:
            try:
                self.play_q.get_nowait()
                drained += 1
            except queue.Empty:
                break
        # 2) flush GStreamer appsrc(清掉已推未播的残余)
        try:
            self.mini.media.audio.clear_player()
            flushed = "appsrc 已 flush"
        except Exception as e:
            flushed = f"clear_player 失败:{type(e).__name__}: {e}"
        # 3) 在途回复直接取消
        if in_flight and self.conv is not None:
            self.conv.cancel_response()
        log(
            f"⛔ BARGE-IN:丢弃队列 {drained} 块,{flushed},"
            f"残余估计 {residual:.2f}s,{'已 cancel_response' if in_flight else '回复已完成无需 cancel'}"
        )

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
                    playing = (now < st.playback_end_estimate) or (not self.play_q.empty())
                    in_flight = st.response_created_count > len(st.response_done_times)
                log(f"🎤 speech_started(播放中={playing},生成中={in_flight})")
                if playing or in_flight:
                    self._do_barge_in(now)
            elif etype == "input_audio_buffer.speech_stopped":
                log("🤫 speech_stopped")
            elif etype == "conversation.item.input_audio_transcription.completed":
                t = (event.get("transcript") or "").strip()
                with st.lock:
                    st.input_transcripts.append(t)
                log(f"📝 输入转写:「{t}」")
            elif etype == "response.created":
                with st.lock:
                    st.response_created_count += 1
                    st.drop_audio = False        # 新回复的音频从这里开始有效
                    gen = st.play_gen
                    st.reply_gen_marks.append(len(st.reply_parts))
                log(f"💭 response.created(代际 {gen})")
            elif etype == "response.audio_transcript.delta":
                with st.lock:
                    st.reply_parts.append(event.get("delta", ""))
            elif etype == "response.audio.delta":
                with st.lock:
                    if st.drop_audio:
                        st.dropped_delta_count += 1
                        return
                    gen = st.play_gen
                    st.audio_delta_count += 1
                    first = st.audio_delta_count == 1
                b64 = event.get("delta") or event.get("audio") or ""
                pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                f32 = pcm.astype(np.float32) / 32768.0
                f16k = resample_poly(f32, PLAY_SR, OUT_SR).astype(np.float32)
                if first:
                    log("🔉 收到首个 response.audio.delta")
                self.play_q.put((gen, f16k))
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


def player_loop(mini: ReachyMini, st: State, play_q: "queue.Queue", stop: threading.Event) -> None:
    """消费播放队列:每段先攒 JITTER_S 再开播;代际不符的块直接丢。"""

    def current_gen() -> int:
        with st.lock:
            return st.play_gen

    def push(chunk: np.ndarray) -> None:
        mini.media.push_audio_sample(chunk)
        dur = len(chunk) / PLAY_SR
        with st.lock:
            base = max(st.playback_end_estimate, time.monotonic())
            st.playback_end_estimate = base + dur

    buffering = True
    while not stop.is_set():
        try:
            gen, chunk = play_q.get(timeout=0.1)
        except queue.Empty:
            buffering = True   # 队列放空 → 下一段重新攒缓冲
            continue
        if gen != current_gen():
            continue           # 被打断作废的旧块
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
                continue       # 攒的过程中被打断,全部作废
            log(f"🔊 开播(缓冲 {sum(len(c) for c in valid) / PLAY_SR:.2f}s)")
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

    print("=== D-01b 编排版:barge-in 打断 + 抖动缓冲 ===", flush=True)
    st = State()
    play_q: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    log("连接 Reachy Mini(media_backend=default)…")
    with ReachyMini(connection_mode="localhost_only", media_backend="default") as mini:
        mini.media.start_recording()
        mini.media.start_playing()
        log("录音/播放管线已启动,预热 1.5s…")
        time.sleep(1.5)

        cb = ChatCallback(st, play_q, mini)
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
        for _ in range(3):
            conv.append_audio(silence)
            time.sleep(0.5)

        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        while time.monotonic() < deadline:
            with st.lock:
                pending = len(st.speech_started_times) > len(st.response_done_times)
            if not pending:
                break
            time.sleep(0.2)

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
    # 按 response.created 的分段标记拆出每轮回复文本
    marks = st.reply_gen_marks + [len(st.reply_parts)]
    replies = [
        "".join(st.reply_parts[marks[i]:marks[i + 1]]).strip()
        for i in range(len(marks) - 1)
    ]
    print(f"speech_started 次数:{len(st.speech_started_times)}", flush=True)
    print(f"response.created / done:{st.response_created_count} / {len(st.response_done_times)}", flush=True)
    print(f"输入转写:{st.input_transcripts}", flush=True)
    for i, r in enumerate(replies, 1):
        print(f"回复{i}:「{r[:120]}{'…' if len(r) > 120 else ''}」", flush=True)
    print(f"barge-in 记录:{st.barge_ins if st.barge_ins else '无'}", flush=True)
    print(f"播放 delta:{st.audio_delta_count} 块|打断后丢弃 delta:{st.dropped_delta_count} 块", flush=True)
    fad = st.first_audio_delay_ms
    print(f"最后一轮首音频延迟:{fad:.0f}ms" if fad else "首音频延迟:N/A", flush=True)
    print(f"error 事件:{st.errors if st.errors else '无'}", flush=True)

    # 验收:发生过打断,且最后一次打断之后还有完成的新回复(机器人听了新话并答了)
    barged = bool(st.barge_ins)
    post_round_done = False
    if barged:
        last_barge_t = st.barge_ins[-1]["t"]
        post_round_done = any((t - T0) > last_barge_t for t in st.response_done_times)
    ok = barged and post_round_done and not st.errors
    print(f"=== {'打断闭环成功' if ok else '未达验收(看上方 barge-in 记录与时间线)'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
