# -*- coding: utf-8 -*-
"""WAKE-01 唤醒词门控 — 本地 sherpa-onnx KeywordSpotter。"""

import os
import time

import numpy as np
import sherpa_onnx

from voice.config import (
    KWS_MODEL_DIR, KWS_KEYWORDS, KWS_FORMS,
    KWS_SINGLE_THR, KWS_DEBOUNCE_S, KWS_REFRACTORY_S,
)
from voice.state import log


class KwsGate:
    """本地 sherpa-onnx KeywordSpotter(单字"小艺"三形态)+ 去抖/不应期。
    feed(mono_f32_16k) → True 表示一次真唤醒。armed/engaged 都喂(engaged 命中忽略)。"""

    def __init__(self) -> None:
        os.makedirs(os.path.dirname(KWS_KEYWORDS), exist_ok=True)
        with open(KWS_KEYWORDS, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{form} #{KWS_SINGLE_THR:.2f} @小艺" for form in KWS_FORMS) + "\n")
        tag = "epoch-12-avg-2-chunk-16-left-64"
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=os.path.join(KWS_MODEL_DIR, "tokens.txt"),
            encoder=os.path.join(KWS_MODEL_DIR, f"encoder-{tag}.int8.onnx"),
            decoder=os.path.join(KWS_MODEL_DIR, f"decoder-{tag}.int8.onnx"),
            joiner=os.path.join(KWS_MODEL_DIR, f"joiner-{tag}.int8.onnx"),
            num_threads=1, max_active_paths=4,
            keywords_file=KWS_KEYWORDS,
            keywords_score=1.0, keywords_threshold=0.10, num_trailing_blanks=1,
            provider="cpu",
        )
        self.stream = self.kws.create_stream()
        self._last_raw = -1e9
        self._last_wake = -1e9
        self._diag_t = time.monotonic()
        self._diag_chunks = self._diag_dec = 0
        self._diag_rms_sq = self._diag_rms_n = 0
        self._diag_ch_done = False

    def feed(self, mono: "np.ndarray", chunk_full: "np.ndarray | None" = None) -> bool:
        if not self._diag_ch_done and chunk_full is not None and chunk_full.ndim == 2:
            ch_rms = [(c, float((chunk_full[:, c].astype(np.float64) ** 2).mean()) ** 0.5)
                      for c in range(chunk_full.shape[1])]
            log(f"[KWS通道诊断] shape={chunk_full.shape} dtype={chunk_full.dtype} "
                f"min={float(chunk_full.min()):.5f} max={float(chunk_full.max()):.5f} | "
                + " ".join(f"ch{c}={r:.5f}" for c, r in ch_rms))
            self._diag_ch_done = True
        self.stream.accept_waveform(16000, np.ascontiguousarray(mono, dtype=np.float32))
        hit = False
        n_dec = 0
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            n_dec += 1
        self._diag_chunks += 1
        self._diag_dec += n_dec
        self._diag_rms_sq += float(np.dot(mono, mono))
        self._diag_rms_n += len(mono)
        now = time.monotonic()
        if now - self._diag_t > 3.0:
            rms = (self._diag_rms_sq / max(1, self._diag_rms_n)) ** 0.5
            log(f"[KWS诊断] chunks={self._diag_chunks} dec={self._diag_dec} "
                f"RMS={rms:.4f} {'⚠ 静音?' if rms < 0.001 else '✅ 有声'}")
            self._diag_chunks = self._diag_dec = 0
            self._diag_rms_sq = self._diag_rms_n = 0
            self._diag_t = now
        result = self.kws.get_result(self.stream)
        if result:
            log(f"[KWS] 原始命中: {result!r}")
            self.kws.reset_stream(self.stream)
            hit = True
        if not hit:
            return False
        t = time.monotonic()
        if t - self._last_wake < KWS_REFRACTORY_S or t - self._last_raw < KWS_DEBOUNCE_S:
            self._last_raw = t
            return False
        self._last_raw = t
        self._last_wake = t
        return True
