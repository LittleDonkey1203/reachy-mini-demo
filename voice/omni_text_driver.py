# -*- coding: utf-8 -*-
"""
OmniTextDriver —— ASR 级联重构(docs/ASR_CASCADE_REDESIGN.md)阶段0 骨架。

把 Omni 从"语音直入(S2S)"改为"文本入(带说话人)+ 中和音频":
- 主 Omni 连接只收【静音锚点 + 视频帧 + 我方带说话人标签的文本】,turn_detection=None、
  转写关 → Omni 永不据音频自动回复;视频仍被参考(真机 spike 已验证)。
- 真实音频交给独立 ASR 连接(见 §3.2.1,本类不管),ASR 断句 + ASD 归属后调 speak_turn()。

阶段0 边界:本模块【独立、未接线进 d01】,不改现有 S2S 行为。锚点机制已在
scratchpad/spike_continuous.py 真机验证(连续帧稳定、每轮画面新鲜、clear 后无缝重锚、0 报错)。
阶段2 才把 d01 主循环的 append_audio(真实音频)替换为本类的静音锚点 + speak_turn 驱动。

已验证参数(spike):静音 100ms/帧、视频 ~200ms/帧、中途 input_audio_buffer.clear 安全。
"""
import base64
import threading
import time

import numpy as np
from dashscope.audio.qwen_omni import (
    OmniRealtimeConversation, AudioFormat, MultiModality,
)

from voice.config import MODEL, VOICE
from voice.state import log

# 100ms @ 16k 单声道 PCM16 静音,作视频锚点(Omni 视频帧必须挂在音频 buffer 之后)
_SILENCE_B64 = base64.b64encode(np.zeros(1600, dtype=np.int16).tobytes()).decode("ascii")


class OmniTextDriver:
    """封装:脑连接配置 + 静音锚点 sender + 视频转发 + 带标签文本轮次驱动。"""

    def __init__(self, callback, instructions: str, tools=None,
                 get_frame_b64=None, is_playing=None,
                 silence_period_s: float = 0.1, video_every: int = 2,
                 clear_every_s: float = 30.0):
        """
        callback: OmniRealtimeCallback(复用 ChatCallback,处理下行音频/工具/transcript)
        instructions: 基础人设(system,稳定部分)
        tools: function-calling 工具表(不变)
        get_frame_b64: () -> str|None,取当前视频帧 base64(JPEG);None=不送视频
        is_playing: () -> bool,机器人是否在放 TTS(留给 ASR 侧回声门,本类暂不用)
        silence_period_s: 静音锚点周期(已验证 0.1)
        video_every: 每 N 个静音周期送 1 帧视频(2 → ~200ms/帧,已验证)
        clear_every_s: 周期 input_audio_buffer.clear 防 buffer 膨胀(0=不清)
        """
        self.callback = callback
        self.instructions = instructions
        self.tools = tools or []
        self.get_frame_b64 = get_frame_b64
        self.is_playing = is_playing
        self.silence_period_s = silence_period_s
        self.video_every = max(1, video_every)
        self.clear_every_s = clear_every_s

        self.conv: OmniRealtimeConversation | None = None
        self._anchor_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ctx_seq = 0

    # ── 连接 ──────────────────────────────────────────────────────────
    def connect(self, timeout: float = 10.0) -> bool:
        """建连并把会话配成"中和音频"模式:turn_detection 关、转写关、audio+text 输出。"""
        c = OmniRealtimeConversation(model=MODEL, callback=self.callback)
        c.connect()
        c.update_session(
            output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],  # AUDIO=TTS 下行
            voice=VOICE,
            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            enable_input_audio_transcription=False,   # 关:不用 Omni 的 ASR(避免无标签 history item)
            enable_turn_detection=False,              # 关:Omni 永不据音频自动回复
            instructions=self.instructions,
            tools=self.tools,
        )
        self.conv = c
        if self.callback is not None:
            self.callback.conv = c
        log("✅ OmniTextDriver 已连接(turn_detection=None / 转写关 / 静音锚点模式)")
        return True

    # ── 静音锚点 + 视频 sender(后台线程)────────────────────────────
    def start_anchor(self):
        if self._anchor_thread is not None:
            return
        self._stop.clear()
        self._anchor_thread = threading.Thread(target=self._anchor_loop, daemon=True)
        self._anchor_thread.start()
        log(f"📎 静音锚点已启动(silence={self.silence_period_s*1000:.0f}ms video每{self.video_every}拍)")

    def _anchor_loop(self):
        n = 0
        last_clear = time.monotonic()
        while not self._stop.is_set():
            c = self.conv
            if c is not None:
                try:
                    c.append_audio(_SILENCE_B64)                       # 锚点静音
                    if self.get_frame_b64 is not None and n % self.video_every == 0:
                        frame = self.get_frame_b64()
                        if frame:
                            c.append_video(frame)                      # 当前视频帧
                    if self.clear_every_s > 0 and (time.monotonic() - last_clear) > self.clear_every_s:
                        # clear 后 sender 下一拍立刻补静音重锚(spike 验证无缝,不破坏视频)
                        c.clear_appended_audio()
                        last_clear = time.monotonic()
                except Exception as e:
                    log(f"⚠ 锚点 sender 异常(忽略继续):{type(e).__name__}: {e}")
            n += 1
            time.sleep(self.silence_period_s)

    # ── 一轮:带说话人标签的文本 + 带记忆的单次回复指令 ──────────────
    def speak_turn(self, speaker_label: str, transcript: str,
                   response_instructions: str | None = None):
        """
        speaker_label: 说话人显示名(已命名)或"访客A"(未命名);进历史当稳定标签。
        transcript: ASR 定稿文本。
        response_instructions: 本轮易变记忆/消歧(只作用这次回复,不进历史)。
        """
        c = self.conv
        if c is None:
            log("⚠ speak_turn:未连接"); return
        text = f"「{speaker_label}」:{transcript}"
        self._ctx_seq += 1
        c.create_item({
            "id": f"u{self._ctx_seq}",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        })
        c.create_response(instructions=response_instructions)
        log(f"🗣 speak_turn 「{speaker_label}」:{transcript[:20]}… "
            f"{'(带记忆指令)' if response_instructions else ''}")

    def cancel(self):
        c = self.conv
        if c is not None:
            try:
                c.cancel_response()
            except Exception as e:
                log(f"⚠ cancel_response 失败:{type(e).__name__}: {e}")

    # ── 关闭 ─────────────────────────────────────────────────────────
    def close(self):
        self._stop.set()
        t = self._anchor_thread
        if t is not None:
            t.join(timeout=1.0)
        self._anchor_thread = None
        c = self.conv
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
        self.conv = None
        if self.callback is not None:
            self.callback.conv = None
        log("🔌 OmniTextDriver 已关闭")
