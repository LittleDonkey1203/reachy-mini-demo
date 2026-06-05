# -*- coding: utf-8 -*-
"""D-01:Reachy Mini × Qwen3.5-Omni-Realtime 语音对话(含 barge-in 打断)。

对着机器人说话 → Qwen 全双工识别并生成语音 → 从机器人扬声器播放;
机器人说话时你开口插话,它会立即闭嘴听你说(barge-in)。

音频链路(均已实测验证,见 ../CALIBRATION.md):
  上行:Reachy 麦克风 16kHz float32 双声道(实为单声道复制,A-01)
        → 取 audio[:,0] → int16 → base64 → input_audio_buffer.append
        (与 Realtime 要求的 16kHz/PCM16/单声道 原生一致,零重采样)
  下行:response.audio.delta(24kHz PCM16 base64)
        → 重采样 24k→16k(播放管线 appsrc 固定 16kHz)
        → 播放队列(攒 ~300ms 抖动缓冲再开播)→ push_audio_sample
        (收包回调只入队,播放由独立线程消费,不阻塞 WebSocket)
  打断:speech_started 且(播放中 或 回复生成中)→
        作废播放队列(代际计数)+ audio.clear_player() flush 管线残余
        + 生成中则 cancel_response();实测打断到静音 ~20ms。

运行(需 daemon 已启动、DASHSCOPE_API_KEY 已配):
  $env:PYTHONUTF8=1
  & "C:\\Users\\ldkji\\AppData\\Local\\Reachy Mini Control\\.venv\\Scripts\\python.exe" voice\\d01_realtime_chat.py
按 Ctrl+C 退出。
"""

import os

# ── 代理隔离:必须在 import reachy_mini / dashscope 之前 ──
# localhost → daemon 直连;*.aliyuncs.com → 大陆直连,别让 7897 本机代理劫持 wss
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

# ───────────────────────── 配置 ─────────────────────────
MODEL = "qwen3.5-omni-plus-realtime"  # 北京地域,SDK 默认端点 dashscope.aliyuncs.com
VOICE = "Ethan"
INSTRUCTIONS = "你是桌面机器人 Reachy Mini。用简体中文、口语化、简短地回答,一般不超过两三句话。"

OUT_SR = 24000   # Realtime 下行音频采样率
PLAY_SR = 16000  # Reachy 播放管线 appsrc 固定 16kHz(audio_gstreamer.py caps)
JITTER_S = 0.30      # 每段回复开播前攒的缓冲
JITTER_WALL_S = 0.50  # 攒不够时的兜底等待上限


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class State:
    """跨线程共享状态(barge-in 代际 + 播放进度估计)。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_updated = threading.Event()
        self.play_gen = 0            # 打断时 +1,旧代际音频块作废
        self.drop_audio = False      # 打断后到下一个 response.created 前丢弃 delta
        self.in_flight = 0           # response.created 未 done 的数量
        self.playback_end_estimate = 0.0  # monotonic 时刻:已推音频预计播完时间


# ───────────────────────── 回调:收服务端事件 ─────────────────────────
class ChatCallback(OmniRealtimeCallback):
    """只做轻活:解析事件、打日志、音频解码入队、打断处置。播放交给独立线程。"""

    def __init__(self, st: State, play_q: "queue.Queue", mini: ReachyMini):
        self.st = st
        self.play_q = play_q
        self.mini = mini
        self.conv: OmniRealtimeConversation | None = None  # connect 后回填

    def on_open(self) -> None:
        log("✅ WebSocket 已连接 dashscope.aliyuncs.com")

    def on_close(self, close_status_code, close_msg) -> None:
        log(f"🔌 连接关闭:code={close_status_code} msg={close_msg}")

    def _do_barge_in(self, in_flight: bool) -> None:
        """打断:作废队列 → flush 管线残余 → 必要时取消在途回复。"""
        st = self.st
        with st.lock:
            st.play_gen += 1
            st.drop_audio = True
            st.playback_end_estimate = time.monotonic()
        while True:  # 清 Python 播放队列
            try:
                self.play_q.get_nowait()
            except queue.Empty:
                break
        try:  # flush GStreamer appsrc 里已推未播的残余
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
                log("✅ 会话配置生效(semantic_vad / 16k入 / 24k出 / 音色 " + VOICE + ")")
                log("▶ 现在可以对机器人说话了;它说话时你随时可以插话打断(Ctrl+C 退出)")
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
                    st.drop_audio = False  # 新回复的音频从这里开始有效
                log("💭 模型开始生成回复…")
            elif etype == "response.audio_transcript.delta":
                print(event.get("delta", ""), end="", flush=True)  # 回复文本流
            elif etype == "response.audio_transcript.done":
                print(flush=True)
            elif etype == "response.audio.delta":
                with st.lock:
                    if st.drop_audio:
                        return  # 被打断回复的在途残块,丢
                    gen = st.play_gen
                b64 = event.get("delta") or event.get("audio") or ""
                pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                f32 = pcm.astype(np.float32) / 32768.0
                f16k = resample_poly(f32, PLAY_SR, OUT_SR).astype(np.float32)
                self.play_q.put((gen, f16k))
            elif etype == "response.done":
                with st.lock:
                    st.in_flight = max(0, st.in_flight - 1)
                d = self.conv.get_last_first_audio_delay() if self.conv else None
                log(f"✅ 本轮回复完成{f'(首音频延迟 {d:.0f}ms)' if d else ''}")
            elif etype == "error":
                log(f"❌ 服务端错误事件:{event}")
            # 其余事件(rate_limits、item.created 等)静默
        except Exception as e:  # 回调里出错别让 ws 线程裸崩,打出来排查
            log(f"❌ on_event 处理异常:{type(e).__name__}: {e}\n   原始事件:{str(event)[:300]}")


# ───────────────────────── 播放线程:队列 → 扬声器 ─────────────────────────
def player_loop(mini: ReachyMini, st: State, play_q: "queue.Queue", stop: threading.Event) -> None:
    """每段回复先攒 JITTER_S 抖动缓冲再开播;代际不符(被打断)的块直接丢。"""

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
            buffering = True  # 队列放空 → 下一段重新攒缓冲
            continue
        if gen != current_gen():
            continue  # 被打断作废的旧块
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
                continue  # 攒的过程中被打断,全部作废
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

    print("=== D-01:Qwen3.5-Omni-Realtime 语音对话(可打断) ===", flush=True)
    log(f"模型:{MODEL}|VAD:semantic_vad|上行 16kHz PCM16|下行 24kHz→16kHz")

    st = State()
    play_q: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    log("连接 Reachy Mini(media backend = default)…")
    with ReachyMini(connection_mode="localhost_only", media_backend="default") as mini:
        # 先把录音管线拉起来(A-02:启动有 1~2s 延迟,常开)
        mini.media.start_recording()
        mini.media.start_playing()
        log("✅ Reachy 录音/播放管线已启动,预热中…")
        time.sleep(1.5)

        callback = ChatCallback(st, play_q, mini)
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
            turn_detection_type="semantic_vad",  # 官方推荐用于 qwen3.5-omni-realtime
            instructions=INSTRUCTIONS,
        )
        if not st.session_updated.wait(timeout=10):
            log("❌ 10s 内未收到 session.updated,中止")
            conv.close()
            return 1

        player = threading.Thread(target=player_loop, args=(mini, st, play_q, stop), daemon=True)
        player.start()

        # 把预热期囤的旧音频丢掉,从"现在"开始上行
        while mini.media.get_audio_sample() is not None:
            pass

        # 主循环:麦克风 → Realtime 上行
        sent_samples = 0
        try:
            while True:
                chunk = mini.media.get_audio_sample()  # (N,2) float32 16kHz 或 None
                if chunk is None or len(chunk) == 0:
                    time.sleep(0.01)
                    continue
                mono = chunk[:, 0]  # A-01:双声道为单声道复制,取一路即可
                pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
                conv.append_audio(base64.b64encode(pcm16.tobytes()).decode("ascii"))
                sent_samples += len(mono)
        except KeyboardInterrupt:
            print(flush=True)
            log(f"收到 Ctrl+C,退出。本次共上行音频 {sent_samples / 16000:.1f} 秒")
        finally:
            stop.set()
            try:
                conv.close()
            except Exception:
                pass
            try:
                mini.media.stop_recording()
                mini.media.stop_playing()
            except Exception:
                pass
            log("已释放 Realtime 连接与 Reachy 媒体资源。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
