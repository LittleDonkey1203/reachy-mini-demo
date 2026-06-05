# -*- coding: utf-8 -*-
"""O-01 诊断:定位"上行 75s 无任何服务端事件"。

A 段:session 带 tools → 12s 窗口,逐秒打印上行 RMS + 收到的所有事件类型
B 段:同一连接 update_session 去掉 tools → 12s 窗口,同样打印
结论矩阵:
  RMS≈0            → 麦克风通道问题(与 tools 无关)
  RMS 正常,A 无事件 B 有 → tools 配置打挂了会话
  RMS 正常,A B 都无事件  → 连接/账号/服务端问题
"""

import os

_no_proxy = "localhost,127.0.0.1,::1,.aliyuncs.com,aliyuncs.com"
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

import base64
import sys
import threading
import time

import numpy as np

import dashscope
from dashscope.audio.qwen_omni import (
    AudioFormat,
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)
from reachy_mini import ReachyMini

MODEL = "qwen3.5-omni-plus-realtime"
T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:7.2f}s] {msg}", flush=True)


_NOPARAM = {"type": "object", "properties": {}}
TOOLS = [
    {"type": "function", "name": "nod", "description": "点头。打招呼、同意时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "shake_head", "description": "摇头。否定时使用。", "parameters": _NOPARAM},
]

events_a: list[str] = []
events_b: list[str] = []
current_sink = events_a
session_updated = threading.Event()
lock = threading.Lock()


class CB(OmniRealtimeCallback):
    def on_open(self) -> None:
        log("✅ ws open")

    def on_close(self, code, msg) -> None:
        log(f"🔌 ws close code={code} msg={msg}")

    def on_event(self, ev) -> None:
        t = ev.get("type", "?")
        with lock:
            current_sink.append(t)
        if t == "session.updated":
            session_updated.set()
        if t == "error":
            log(f"❌ error 事件全文:{ev}")
        else:
            log(f"  事件: {t}")


def stream_window(mini, conv, seconds: float, tag: str) -> None:
    """上行 seconds 秒麦克风音频,逐秒打印 RMS。"""
    log(f"── {tag}:窗口 {seconds:.0f}s 打开,请持续说话 ──")
    buf_cnt = 0
    sec_acc: list[float] = []
    sec_t = time.monotonic()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        chunk = mini.media.get_audio_sample()
        if chunk is None or len(chunk) == 0:
            time.sleep(0.01)
            continue
        mono = chunk[:, 0]
        sec_acc.append(float(np.sqrt(np.mean(mono**2))))
        pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        conv.append_audio(base64.b64encode(pcm16.tobytes()).decode("ascii"))
        buf_cnt += 1
        if time.monotonic() - sec_t >= 1.0:
            rms = float(np.mean(sec_acc)) if sec_acc else 0.0
            log(f"  {tag} 上行 RMS={rms:.4f}({'有声' if rms > 0.002 else '静音级'})")
            sec_acc = []
            sec_t = time.monotonic()
    log(f"── {tag}:窗口关闭(共 {buf_cnt} 块)──")


def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        log("❌ 无 key")
        return 1
    dashscope.api_key = api_key

    print("=== O-01 诊断 ===", flush=True)
    global current_sink
    with ReachyMini(connection_mode="localhost_only", media_backend="default") as mini:
        mini.media.start_recording()
        time.sleep(1.5)

        conv = OmniRealtimeConversation(model=MODEL, callback=CB())
        conv.connect()

        # ── A 段:带 tools ──
        session_updated.clear()
        conv.update_session(
            output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
            voice="Ethan",
            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            enable_input_audio_transcription=True,
            enable_turn_detection=True,
            turn_detection_type="semantic_vad",
            instructions="测试。简短回答。",
            tools=TOOLS,
        )
        if not session_updated.wait(timeout=10):
            log("❌ A 段 session.updated 超时")
        while mini.media.get_audio_sample() is not None:
            pass
        log("READY_A")
        stream_window(mini, conv, 12.0, "A(带tools)")
        time.sleep(3.0)  # 给在途事件一点时间

        # ── B 段:去掉 tools(tools=[] 显式清空)──
        with lock:
            current_sink = events_b
        session_updated.clear()
        conv.update_session(
            output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
            voice="Ethan",
            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            enable_input_audio_transcription=True,
            enable_turn_detection=True,
            turn_detection_type="semantic_vad",
            instructions="测试。简短回答。",
        )
        session_updated.wait(timeout=10)
        log("READY_B")
        stream_window(mini, conv, 12.0, "B(无tools)")
        time.sleep(3.0)

        try:
            conv.close()
        except Exception:
            pass
        mini.media.stop_recording()

    print("\n========== 诊断汇总 ==========", flush=True)
    print(f"A 段(带 tools)事件:{events_a}", flush=True)
    print(f"B 段(无 tools)事件:{events_b}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
