# -*- coding: utf-8 -*-
"""
独立流式 ASR —— ASR 级联重构(docs/ASR_CASCADE_REDESIGN.md)阶段1。

职责单一:**音频字节 → 带时间戳的句子 / 轮次**。
刻意【不依赖】ASD / memory / Omni —— 说话人归属、记忆注入、回复驱动都由调用方做,
本模块只经注入的 `resolve_speaker(start_mono)->pid` 回调判"说话人是否切换",保持解耦。

两个类:
- `AsrStream`     —— 包 dashscope `Recognition`,`feed(pcm16_bytes)` 入 → `on_sentence` 出。
- `TurnAggregator` —— 句子入 → `on_turn` 出;gap 静音 / 超长 / 说话人切换 三条触发。

阶段1 用法(只打日志,不驱动 Omni):
    agg = TurnAggregator(on_turn=lambda t: log(f"🧾 轮:{t.text}"))
    asr = AsrStream(on_sentence=agg.add)
    asr.start()
    # 送帧循环:asr.feed(pcm16.tobytes())
    asr.stop(); agg.close()
"""
import threading
import time
from dataclasses import dataclass

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

from voice.state import log

# 华北2(北京)地域 Recognition WebSocket 端点(与 Omni 的 url 互不影响)。
_ASR_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

_NO_SPEAKER = object()   # 哨兵:区分"本轮还没定说话人"与"说话人=None(画外)"


@dataclass
class Sentence:
    """一句 ASR 定稿。时间戳两套:ASR 毫秒(相对流启动)+ 本地 monotonic(供 ASD 对齐)。"""
    text: str
    begin_time: int | None   # ASR ms
    end_time: int | None     # ASR ms
    start_mono: float        # 本句首个 partial 到达的 monotonic
    end_mono: float          # 定稿的 monotonic


@dataclass
class Turn:
    """一轮聚合(可能多句)。start_mono 供调用方 ASD.speaker_window 对齐归属。"""
    text: str
    start_mono: float
    end_mono: float
    n_sentences: int
    reason: str              # 触发原因:静音 / 超长 / 说话人切换


class AsrStream:
    """流式 ASR:喂 PCM16 裸字节,回调 on_sentence(定稿) / on_partial(中间稿)。"""

    def __init__(self, on_sentence, on_partial=None, *,
                 model: str = "paraformer-realtime-v2", sample_rate: int = 16000,
                 semantic_punctuation: bool = True, disfluency_removal: bool = True,
                 phrase_id: str | None = None, set_ws_url: bool = True):
        self.on_sentence = on_sentence
        self.on_partial = on_partial
        self.model = model
        self.sample_rate = sample_rate
        self.semantic_punctuation = semantic_punctuation
        self.disfluency_removal = disfluency_removal
        self.phrase_id = phrase_id
        self.set_ws_url = set_ws_url

        self._rec: Recognition | None = None
        self._cb = self._Cb(self)
        self._cur_start_mono: float | None = None   # 当前句起点(首 partial 时记)
        self._started = False
        self._alive = False        # 连接是否活着(on_close/on_error 置 False → feed 时节流重连)
        self._stopping = False     # 主动 stop 中(区别于错误断连,不触发重连)
        self._last_restart = 0.0

    def _open(self):
        if self.set_ws_url:
            dashscope.base_websocket_api_url = _ASR_WS_URL
        self._rec = Recognition(
            model=self.model,
            callback=self._cb,
            format="pcm",
            sample_rate=self.sample_rate,
            semantic_punctuation_enabled=self.semantic_punctuation,
            disfluency_removal_enabled=self.disfluency_removal,
        )
        if self.phrase_id:
            self._rec.start(phrase_id=self.phrase_id)
        else:
            self._rec.start()
        self._alive = True

    def start(self):
        if self._started:
            return
        self._open()
        self._started = True

    def feed(self, pcm16_bytes: bytes):
        """送一帧裸 PCM16(即 d01 里 base64 前那份 `pcm16.tobytes()`)。连接死了则节流重连。"""
        if not self._started or self._stopping:
            return
        if not self._alive:
            now = time.monotonic()
            if now - self._last_restart > 3.0:   # 节流:最多 3s 重连一次
                self._last_restart = now
                try:
                    self._open()
                    log("🎙 ASR 自动重连")
                except Exception as e:
                    log(f"⚠ ASR 重连失败:{type(e).__name__}: {e}")
            if not self._alive:
                return
        try:
            self._rec.send_audio_frame(pcm16_bytes)
        except Exception:
            self._alive = False   # 发帧异常 → 标记断连,下次 feed 触发重连

    def stop(self):
        self._stopping = True
        self._alive = False
        if self._rec is not None and self._started:
            try:
                self._rec.stop()
            except Exception as e:
                log(f"⚠ ASR stop 异常:{type(e).__name__}: {e}")
        self._started = False

    def update_phrase(self, phrase_id: str | None):
        """热词表变更(有人被命名后刷新在场人名);下次 start 生效。"""
        self.phrase_id = phrase_id

    class _Cb(RecognitionCallback):
        def __init__(self, outer: "AsrStream"):
            self.o = outer

        def on_open(self):
            log("🎙 ASR 连接已开")

        def on_close(self):
            self.o._alive = False
            log("🎙 ASR 连接已关" + ("" if self.o._stopping else "(非主动→将自动重连)"))

        def on_complete(self):
            log("🎙 ASR 识别完成")

        def on_error(self, message):
            self.o._alive = False
            log(f"⚠ ASR 错误 req={getattr(message, 'request_id', '?')}: "
                f"{getattr(message, 'message', message)}")

        def on_event(self, result: RecognitionResult):
            try:
                s = result.get_sentence()
            except Exception as e:
                log(f"⚠ ASR get_sentence 异常:{type(e).__name__}: {e}")
                return
            if not s or "text" not in s:
                return
            now = time.monotonic()
            if self.o._cur_start_mono is None:
                self.o._cur_start_mono = now   # 本句起点 = 首个 partial 到达时刻
            text = s["text"]
            if self.o.on_partial is not None:
                self.o.on_partial(text)
            if RecognitionResult.is_sentence_end(s):
                sent = Sentence(
                    text=text,
                    begin_time=s.get("begin_time"),
                    end_time=s.get("end_time"),
                    start_mono=self.o._cur_start_mono,
                    end_mono=now,
                )
                self.o._cur_start_mono = None
                if self.o.on_sentence is not None:
                    self.o.on_sentence(sent)


class TurnAggregator:
    """句子 → 轮次。触发:轮末静音 > turn_gap_s / 超长 / 说话人切换(经 resolve_speaker)。"""

    def __init__(self, on_turn, resolve_speaker=None, *,
                 turn_gap_s: float = 1.2, max_turn_s: float = 15.0, tick_s: float = 0.2):
        self.on_turn = on_turn
        self.resolve_speaker = resolve_speaker   # (start_mono) -> pid|None;None=不按说话人切
        self.turn_gap_s = turn_gap_s
        self.max_turn_s = max_turn_s
        self.tick_s = tick_s

        self._buf: list[Sentence] = []
        self._turn_speaker = _NO_SPEAKER
        self._last_activity = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
        self._ticker.start()

    def add(self, sent: Sentence):
        """AsrStream.on_sentence 回调入口。可能产出 0/1/2 个轮次(切换时先冲旧轮)。"""
        spk = self.resolve_speaker(sent.start_mono) if self.resolve_speaker else None
        pending: list[Turn] = []
        with self._lock:
            if self._buf and self.resolve_speaker is not None and spk != self._turn_speaker:
                t = self._build_locked("说话人切换")
                if t:
                    pending.append(t)
            if not self._buf:
                self._turn_speaker = spk
            self._buf.append(sent)
            self._last_activity = time.monotonic()
            if self._buf and (self._buf[-1].end_mono - self._buf[0].start_mono) >= self.max_turn_s:
                t = self._build_locked("超长")
                if t:
                    pending.append(t)
        for t in pending:            # 回调在锁外调,避免回调回头调 add 造成死锁
            self._emit(t)

    def _tick_loop(self):
        while not self._stop.wait(self.tick_s):
            t = None
            with self._lock:
                if self._buf and (time.monotonic() - self._last_activity) > self.turn_gap_s:
                    t = self._build_locked("静音")
            if t:
                self._emit(t)

    def _build_locked(self, reason: str) -> Turn | None:
        if not self._buf:
            return None
        text = "".join(s.text for s in self._buf).strip()
        turn = Turn(
            text=text,
            start_mono=self._buf[0].start_mono,
            end_mono=self._buf[-1].end_mono,
            n_sentences=len(self._buf),
            reason=reason,
        )
        self._buf = []
        self._turn_speaker = _NO_SPEAKER
        return turn

    def _emit(self, turn: Turn):
        if turn.text and self.on_turn is not None:
            try:
                self.on_turn(turn)
            except Exception as e:
                log(f"⚠ on_turn 回调异常:{type(e).__name__}: {e}")

    def flush(self):
        """强制冲出当前缓冲(如会话结束)。"""
        t = None
        with self._lock:
            t = self._build_locked("强制")
        if t:
            self._emit(t)

    def close(self):
        self._stop.set()
        self.flush()
