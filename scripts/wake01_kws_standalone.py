# -*- coding: utf-8 -*-
"""WAKE-01 · 唤醒词真机 standalone 验证 + 标定工具(完全不碰 d01)。

最终结论(详见 CALIBRATION.md §14 WAKE-01):
  唤醒词 = 单字「小艺」,keywords 留 yī/yìn/yì 三声调形态,--single-thr 0.17,命中即唤醒。
  叠词「小艺小艺」+ 双单确认在真人远场音下不成立(token 解码不出),已弃用。
  模型:int8 sherpa-onnx-kws-zipformer-wenetspeech-3.3M(放 tools/_kws_models/,gitignore)。

复用 d01 的上行麦克风路径:同设备、同 16kHz mono float32 流(d01 没跑时本脚本独占麦)。

运行(daemon 须已启动;$env:PYTHONUTF8=1):
  # ★ 生产配置(锁定):单字「小艺」命中即唤醒,跑召回+5min背景误触两关
  tools/wake01_kws_standalone.py --prod
  tools/wake01_kws_standalone.py --prod --free [--diag]   # 自由监听,我喊你读日志(--diag 开漏检诊断旁路)
  tools/wake01_kws_standalone.py --prod --wake-only        # 只跑召回
标定/探查模式(留痕,定 0.17 的依据):
  --probe       叠词/单字 × 阈值阶梯(读不到内部分数的替代:看最高触发档=score 落点)
  --varprobe    声调/韵尾/连读各形态 × {0.20,0.40},看真人发音落到哪个 token 行
  --free        旧:单关键词 + 诊断 ASR(实时打 RMS + 听到的 token,分清"没听见"vs"没认出")
可调参:--single-thr 0.17  --debounce 0.3  --refractory 2.0  --bg-min 5  --fp32(默认 int8)
"""

import os

# ── 代理隔离:必须在 import reachy_mini 之前(localhost 连接被本机代理截断)──
_no_proxy = "localhost,127.0.0.1,::1,.aliyuncs.com,aliyuncs.com"
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

import argparse
import threading
import time
from collections import deque

import numpy as np
import sherpa_onnx

from reachy_mini import ReachyMini

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "_kws_models", "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01")
KEYWORDS = os.path.join(HERE, "_kws_models", "keywords_xiaoyi.txt")  # 叠词 "小艺小艺"(配置B,实测误触 0/6)
SR = 16000

_t0 = time.monotonic()
def log(msg: str) -> None:
    print(f"[{time.monotonic() - _t0:7.2f}s] {msg}", flush=True)


def _model_paths(fp32: bool):
    tag = "epoch-12-avg-2-chunk-16-left-64"
    suf = ".onnx" if fp32 else ".int8.onnx"
    enc = os.path.join(MODEL, f"encoder-{tag}{suf}")
    dec = os.path.join(MODEL, f"decoder-{tag}{suf}")
    joi = os.path.join(MODEL, f"joiner-{tag}{suf}")
    tok = os.path.join(MODEL, "tokens.txt")
    for p in (enc, dec, joi, tok):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    return tok, enc, dec, joi


def build_kws(args) -> "sherpa_onnx.KeywordSpotter":
    tok, enc, dec, joi = _model_paths(args.fp32)
    if not os.path.exists(KEYWORDS):
        raise FileNotFoundError(KEYWORDS)
    with open(KEYWORDS, encoding="utf-8") as f:
        kw_lines = [ln.strip() for ln in f if ln.strip()]
    log(f"KWS 模型:{'fp32' if args.fp32 else 'int8'} | thr={args.threshold} "
        f"score={args.score} trailing_blanks={args.trailing_blanks}")
    log(f"   keywords({os.path.basename(KEYWORDS)}):{kw_lines}")   # 点4:确认 token 行真加载
    return sherpa_onnx.KeywordSpotter(
        tokens=tok,
        encoder=enc, decoder=dec, joiner=joi,
        num_threads=1,
        max_active_paths=4,
        keywords_file=KEYWORDS,
        keywords_score=args.score,
        keywords_threshold=args.threshold,
        num_trailing_blanks=args.trailing_blanks,
        provider="cpu",
    )


def build_asr(args):
    """诊断用:同一 transducer 当流式 ASR 跑,把实时认出的 token 打出来。
    认出 'xiǎo yì' = 音频好(KWS 阈值/token 问题);认不出 = 采集问题。失败则降级返回 None。"""
    try:
        tok, enc, dec, joi = _model_paths(args.fp32)
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tok, encoder=enc, decoder=dec, joiner=joi,
            num_threads=1, provider="cpu", decoding_method="greedy_search",
            enable_endpoint_detection=True,
        )
        log("🔎 诊断 ASR 就绪(同模型流式识别,实时打印听到的 token)")
        return rec
    except Exception as e:
        log(f"⚠ 诊断 ASR 构建失败({e}),降级:只打 RMS/峰值/解码活动")
        return None


def build_kws_th(fp32: bool, keywords_file: str, threshold: float):
    """阈值阶梯探测用:同一模型 + 单关键词文件 + 指定 threshold 造一个 spotter。"""
    tok, enc, dec, joi = _model_paths(fp32)
    return sherpa_onnx.KeywordSpotter(
        tokens=tok, encoder=enc, decoder=dec, joiner=joi,
        num_threads=1, max_active_paths=4,
        keywords_file=keywords_file,
        keywords_score=1.0, keywords_threshold=threshold, num_trailing_blanks=1,
        provider="cpu",
    )


# 探测阶梯:叠词(D)/单字(S)× 阈值;看每次说话"最高触发到哪一档"=best score 落点
PROBE_THRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
KW_PROBE_D = os.path.join(HERE, "_kws_models", "keywords_probe_D.txt")  # x iǎo y ì x iǎo y ì
KW_PROBE_S = os.path.join(HERE, "_kws_models", "keywords_probe_S.txt")  # x iǎo y ì


class ScoreProbeReader(threading.Thread):
    """阈值阶梯探测:把 100ms 块同时喂给 (叠词/单字 × 各阈值) 一排 spotter,
    哪一档触发就打 🎯,据此圈出 best score 落点。不接 ASR(已证音频好,腾出算力)。"""
    def __init__(self, mini, fp32: bool, stop: threading.Event):
        super().__init__(daemon=True)
        self.mini = mini
        self.fp32 = fp32
        self.stop = stop
        self.ladder = []   # (label, thr, spotter, stream)
        for label, kwf in (("叠", KW_PROBE_D), ("单", KW_PROBE_S)):
            for thr in PROBE_THRS:
                sp = build_kws_th(fp32, kwf, thr)
                self.ladder.append((label, thr, sp, sp.create_stream()))
        log(f"🪜 阈值阶梯就绪:叠词/单字 × {PROBE_THRS}({len(self.ladder)} 个 spotter)")

    def run(self) -> None:
        BLOCK = 1600  # 100ms@16k:按块喂,降每帧开销
        buf = np.empty(0, dtype=np.float32)
        diag_done = False
        last_status = time.monotonic()
        peak_win = 0.0; sq = 0.0; n = 0

        while not self.stop.is_set():
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.005); continue
            mono = np.ascontiguousarray(chunk[:, 0], dtype=np.float32)
            if not diag_done:
                try: native_sr = self.mini.media.get_input_audio_samplerate()
                except Exception: native_sr = "?"
                log(f"🔬 首帧:shape={chunk.shape} dtype={chunk.dtype} 采样率={native_sr}Hz")
                diag_done = True
            peak_win = max(peak_win, float(np.abs(mono).max()))
            sq += float(np.sum(mono.astype(np.float64) ** 2)); n += len(mono)
            buf = np.concatenate([buf, mono])
            if len(buf) < BLOCK:
                pass
            else:
                block, buf = buf[:BLOCK], buf[BLOCK:]
                fired = []  # (label, thr)
                for label, thr, sp, st_ in self.ladder:
                    sp_ = sp
                    st_.accept_waveform(SR, block)
                    while sp_.is_ready(st_):
                        sp_.decode_stream(st_)
                    if sp_.get_result(st_):
                        fired.append((label, thr))
                        sp_.reset_stream(st_)
                if fired:
                    # 汇总:每类(叠/单)最高触发阈值 = best score 下界
                    best = {}
                    for label, thr in fired:
                        best[label] = max(best.get(label, 0.0), thr)
                    parts = " ".join(f"{lb}≥{th:.2f}" for lb, th in sorted(best.items()))
                    log(f"🎯 命中 → {parts}   (越高=匹配越好)")
            now = time.monotonic()
            if now - last_status >= 0.3:
                rms = (sq / n) ** 0.5 if n else 0.0
                bar = "#" * min(20, int(rms * 400))
                log(f"📊 RMS={rms:.4f} peak={peak_win:.3f} {bar}")
                last_status = now; peak_win = 0.0; sq = 0.0; n = 0


# 变体阶梯:真人发"小艺"实际落到哪个 token 形态(声调/韵尾/连读)→ 定生产要保留哪几行
# token 已核对全部在词表(toneless 'in' 不存在,故 yin 只用带调 ìn)
VAR_FORMS = [
    ("S_yi4",  "x iǎo y ì"),            # 单·四声(目标)
    ("S_yi1",  "x iǎo y ī"),            # 单·一声(ASR 常出 xiǎoyī)
    ("S_yi2",  "x iǎo y í"),            # 单·二声
    ("S_yi3",  "x iǎo y ǐ"),            # 单·三声
    ("S_yi0",  "x iǎo y i"),            # 单·轻声/无调兜底
    ("S_yin4", "x iǎo y ìn"),           # 单·带鼻音(ASR 常出 xiǎoyìn)
    ("D_yi4",  "x iǎo y ì x iǎo y ì"),  # 叠·四声
    ("D_yi1",  "x iǎo y ī x iǎo y ī"),  # 叠·一声
]
VAR_THRS = [0.20, 0.40]
VAR_DIR = os.path.join(HERE, "_kws_models", "_var")


class VariantProbeReader(threading.Thread):
    """变体阶梯:每个 token 形态 × {0.20,0.40} 各一个 spotter,命中打出
    具体形态 + 触发档(≈score 区间)+ 该形态累计命中数(=可靠度排序)。"""
    def __init__(self, mini, fp32: bool, stop: threading.Event):
        super().__init__(daemon=True)
        self.mini = mini
        self.stop = stop
        os.makedirs(VAR_DIR, exist_ok=True)
        self.ladder = []     # (name, thr, spotter, stream)
        self.count = {}      # name -> 累计命中(任意档)
        for name, toks in VAR_FORMS:
            kwf = os.path.join(VAR_DIR, f"kw_{name}.txt")
            with open(kwf, "w", encoding="utf-8") as f:
                f.write(f"{toks} @{name}\n")
            for thr in VAR_THRS:
                sp = build_kws_th(fp32, kwf, thr)
                self.ladder.append((name, thr, sp, sp.create_stream()))
            self.count[name] = 0
        log(f"🪜 变体阶梯就绪:{len(VAR_FORMS)} 形态 × {VAR_THRS}({len(self.ladder)} spotter)")
        for name, toks in VAR_FORMS:
            log(f"     {name:7s} = {toks}")

    def run(self) -> None:
        BLOCK = 1600
        buf = np.empty(0, dtype=np.float32)
        diag_done = False
        last_status = time.monotonic()
        peak_win = 0.0; sq = 0.0; n = 0
        while not self.stop.is_set():
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.005); continue
            mono = np.ascontiguousarray(chunk[:, 0], dtype=np.float32)
            if not diag_done:
                try: native_sr = self.mini.media.get_input_audio_samplerate()
                except Exception: native_sr = "?"
                log(f"🔬 首帧:shape={chunk.shape} 采样率={native_sr}Hz"); diag_done = True
            peak_win = max(peak_win, float(np.abs(mono).max()))
            sq += float(np.sum(mono.astype(np.float64) ** 2)); n += len(mono)
            buf = np.concatenate([buf, mono])
            while len(buf) >= BLOCK:
                block, buf = buf[:BLOCK], buf[BLOCK:]
                best = {}  # name -> 最高触发档
                for name, thr, sp, st_ in self.ladder:
                    st_.accept_waveform(SR, block)
                    while sp.is_ready(st_):
                        sp.decode_stream(st_)
                    if sp.get_result(st_):
                        best[name] = max(best.get(name, 0.0), thr)
                        sp.reset_stream(st_)
                for name in sorted(best):
                    self.count[name] += 1
                    log(f"🎯 {name:7s} ≥{best[name]:.2f}   (累计 {self.count[name]})")
            now = time.monotonic()
            if now - last_status >= 0.3:
                rms = (sq / n) ** 0.5 if n else 0.0
                bar = "#" * min(20, int(rms * 400))
                log(f"📊 RMS={rms:.4f} peak={peak_win:.3f} {bar}")
                last_status = now; peak_win = 0.0; sq = 0.0; n = 0


# ════════════ 生产配置(A 定稿):单字「小艺」多声调命中即唤醒(叠词已被数据否掉,不保留)═══════════
# 变体落点实测:yī(一声)/yìn(鼻音)/yì(四声)是真人发"小艺"的前三命中形态
PROD_SINGLE_FORMS = ["x iǎo y ī", "x iǎo y ìn", "x iǎo y ì"]
PROD_KW = os.path.join(HERE, "_kws_models", "keywords_prod.txt")
# 漏检诊断旁路:全 6 单字形态 × {0.10,0.20},只记录不唤醒 → 区分"落到没挂形态"vs"卡 0.20 差一点"
DIAG_FORMS = [
    ("yi4", "x iǎo y ì"), ("yi1", "x iǎo y ī"), ("yi2", "x iǎo y í"),
    ("yi3", "x iǎo y ǐ"), ("yi0", "x iǎo y i"), ("yin4", "x iǎo y ìn"),
]
DIAG_THRS = [0.10, 0.20]
DIAG_DIR = os.path.join(HERE, "_kws_models", "_var")


def write_prod_keywords(single_thr: float) -> str:
    lines = [f"{f} #{single_thr:.2f} @S" for f in PROD_SINGLE_FORMS]
    with open(PROD_KW, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    return PROD_KW


class WakeConfirmer:
    """A 定稿:单字命中即唤醒。仅保留去抖(同一声"小艺"点亮多变体行/多块算一次)
    + 不应期(唤醒后 refractory 秒吞掉余波,防同次喊话重复唤醒)。"""
    def __init__(self, debounce: float, refractory: float):
        self.debounce = debounce
        self.refr = refractory
        self.last_raw = -1e9
        self.last_wake = -1e9

    def feed(self, t: float):
        if t - self.last_wake < self.refr:
            return None
        if t - self.last_raw < self.debounce:
            return None
        self.last_raw = t
        self.last_wake = t
        return "单字唤醒"


class ProdReader(threading.Thread):
    """生产读取器:单字 spotter(per-line 阈值)命中即唤醒。可选诊断旁路看漏检 score。"""
    def __init__(self, mini, fp32: bool, kwfile: str, conf: WakeConfirmer,
                 stop: threading.Event, diag: bool = False):
        super().__init__(daemon=True)
        self.mini = mini
        self.conf = conf
        self.stop = stop
        self.spotter = build_kws_th(fp32, kwfile, 0.10)  # 全局兜底低阈,per-line 实际把关
        self.stream = self.spotter.create_stream()
        self.lock = threading.Lock()
        self.wakes: "list[tuple[float, str]]" = []   # (t, 路径)
        self.raw: "list[tuple[float, str]]" = []     # (t, S)
        # 诊断旁路
        self.diag_ladder = []
        if diag:
            os.makedirs(DIAG_DIR, exist_ok=True)
            for name, toks in DIAG_FORMS:
                kwf = os.path.join(DIAG_DIR, f"kw_{name}.txt")
                with open(kwf, "w", encoding="utf-8") as f:
                    f.write(f"{toks} @{name}\n")
                for thr in DIAG_THRS:
                    sp = build_kws_th(fp32, kwf, thr)
                    self.diag_ladder.append((name, thr, sp, sp.create_stream()))
            log(f"🔎 漏检诊断旁路:{len(DIAG_FORMS)} 形态 × {DIAG_THRS}({len(self.diag_ladder)} spotter)")

    def run(self) -> None:
        BLOCK = 1600
        buf = np.empty(0, dtype=np.float32)
        last_status = time.monotonic(); peak_win = 0.0; sq = 0.0; n = 0
        log("🎙 生产读取线程启动(单字「小艺」命中即唤醒)…")
        while not self.stop.is_set():
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.005); continue
            mono = np.ascontiguousarray(chunk[:, 0], dtype=np.float32)
            peak_win = max(peak_win, float(np.abs(mono).max()))
            sq += float(np.sum(mono.astype(np.float64) ** 2)); n += len(mono)
            buf = np.concatenate([buf, mono])
            while len(buf) >= BLOCK:
                block, buf = buf[:BLOCK], buf[BLOCK:]
                self.stream.accept_waveform(SR, block)
                while self.spotter.is_ready(self.stream):
                    self.spotter.decode_stream(self.stream)
                r = self.spotter.get_result(self.stream)
                if r:
                    t = time.monotonic()
                    self.spotter.reset_stream(self.stream)
                    with self.lock:
                        self.raw.append((t, r))
                    path = self.conf.feed(t)
                    log(f"·命中 {r}" + (f"  → 🔔🔔 唤醒({path})" if path else "  (去抖/不应期内,忽略)"))
                    if path:
                        with self.lock:
                            self.wakes.append((t, path))
                # 诊断旁路:只记录"哪个形态在哪个阈值会触发",不参与唤醒
                if self.diag_ladder:
                    dbest = {}
                    for name, thr, sp, st_ in self.diag_ladder:
                        st_.accept_waveform(SR, block)
                        while sp.is_ready(st_):
                            sp.decode_stream(st_)
                        if sp.get_result(st_):
                            dbest[name] = max(dbest.get(name, 0.0), thr)
                            sp.reset_stream(st_)
                    if dbest:
                        parts = " ".join(f"{nm}≥{th:.2f}" for nm, th in sorted(dbest.items()))
                        log(f"   🔎诊断 {parts}")
            now = time.monotonic()
            if now - last_status >= 1.0:
                rms = (sq / n) ** 0.5 if n else 0.0
                # 安静时不刷屏:仅在有点动静时打
                if rms > 0.02:
                    log(f"📊 RMS={rms:.4f} peak={peak_win:.3f}")
                last_status = now; peak_win = 0.0; sq = 0.0; n = 0

    def wakes_in(self, t0, t1):
        with self.lock:
            return [(t, p) for (t, p) in self.wakes if t0 <= t <= t1]

    def raw_in(self, t0, t1):
        with self.lock:
            return [(t, k) for (t, k) in self.raw if t0 <= t <= t1]


def prod_phase_recall(reader: ProdReader, n_trials: int, gap: float):
    log("=" * 64)
    log(f"【召回 ×{n_trials}】每次提示说一遍「小艺小艺」,间隔 {gap:.0f}s。2 秒后开始 …")
    time.sleep(2.0)
    results = []
    for i in range(1, n_trials + 1):
        t0 = time.monotonic()
        log(f"  ▶ 第 {i}/{n_trials} 次:现在说「小艺小艺」")
        time.sleep(gap)
        wk = reader.wakes_in(t0, time.monotonic())
        if wk:
            results.append(True)
            log(f"    ✅ 真唤醒({wk[0][1]})")
        else:
            rw = reader.raw_in(t0, time.monotonic())
            results.append(False)
            log(f"    ❌ 未唤醒(原始命中 {len(rw)} 次:{[k for _,k in rw]})")
    return results


def prod_phase_bg(reader: ProdReader, minutes: float):
    log("=" * 64)
    log(f"【误触 · 背景声 {minutes:.0f} 分钟】请现在【开始播放】电视/日常说话(不含唤醒词)。")
    log("   3 秒后开始计时 …")
    time.sleep(3.0)
    t0 = time.monotonic()
    dur = minutes * 60.0
    nxt = t0 + 60.0
    while time.monotonic() - t0 < dur:
        time.sleep(1.0)
        if time.monotonic() >= nxt:
            el = (time.monotonic() - t0) / 60.0
            nw = len(reader.wakes_in(t0, time.monotonic()))
            log(f"   …已 {el:.0f} 分钟,累计误唤醒 {nw}")
            nxt += 60.0
    wk = reader.wakes_in(t0, time.monotonic())
    log(f"背景声结束:误唤醒 {len(wk)} 次 {[p for _,p in wk]}。可【停止】背景声。")
    return len(wk)


def prod_summarize(recall, bg_false):
    log("=" * 64); log("【生产配置验收汇总】")
    n_ok = sum(1 for x in recall if x)
    log(f"  召回真唤醒 : {n_ok}/{len(recall)}(目标 ≥9)")
    log(f"  背景误唤醒 : {bg_false}(目标 ≤1)")
    passed = n_ok >= 9 and bg_false <= 1
    log("-" * 64)
    log(f"  两关:{'✅ 都达标 → 可进 d01 集成' if passed else '❌ 未达标,在 standalone 继续调'}")


class KwsReader(threading.Thread):
    """常驻线程:持续读麦克风 16k mono → 喂 KWS(命中记 (t,kw))+ 喂诊断 ASR;
    每 ~0.3s 打一行读数:RMS / 峰值 / 解码块数 / ASR 实时识别文本。"""
    def __init__(self, mini, kws, asr, stop: threading.Event, readout: bool):
        super().__init__(daemon=True)
        self.mini = mini
        self.kws = kws
        self.asr = asr
        self.stop = stop
        self.readout = readout            # 是否打 0.3s 读数(诊断期开)
        self.lock = threading.Lock()
        self.hits: "list[tuple[float, str]]" = []          # 全程命中留痕
        self.chunk_ms: "deque[float]" = deque(maxlen=2000)  # 单块推理耗时
        self.samples = 0

    def run(self) -> None:
        stream = self.kws.create_stream()
        asr_stream = self.asr.create_stream() if self.asr is not None else None
        log("🎙 麦克风读取线程启动,持续喂 KWS …")

        diag_done = False
        last_status = time.monotonic()
        peak_win = 0.0
        sq_acc = 0.0
        n_acc = 0
        decode_blocks = 0
        asr_text = ""

        while not self.stop.is_set():
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.005)
                continue
            mono = np.ascontiguousarray(chunk[:, 0], dtype=np.float32)

            # 点1/2:一次性把"真相"打出来 —— 实际采样率 + 帧形状/类型/取值范围
            if not diag_done:
                try:
                    native_sr = self.mini.media.get_input_audio_samplerate()
                except Exception:
                    native_sr = "?"
                log(f"🔬 首帧:shape={chunk.shape} dtype={chunk.dtype} "
                    f"min={float(mono.min()):.3f} max={float(mono.max()):.3f} | "
                    f"SDK 报告采样率={native_sr}Hz(喂 KWS 用 {SR}Hz)")
                diag_done = True

            self.samples += len(mono)
            # 读数累计
            peak_win = max(peak_win, float(np.abs(mono).max()))
            sq_acc += float(np.sum(mono.astype(np.float64) ** 2))
            n_acc += len(mono)

            # 喂 KWS
            tc = time.perf_counter()
            stream.accept_waveform(SR, mono)
            while self.kws.is_ready(stream):
                self.kws.decode_stream(stream)
                decode_blocks += 1
            r = self.kws.get_result(stream)
            self.chunk_ms.append((time.perf_counter() - tc) * 1000.0)
            if r:
                t = time.monotonic()
                with self.lock:
                    self.hits.append((t, r))
                log(f"🔔 命中「{r}」")
                self.kws.reset_stream(stream)

            # 喂诊断 ASR
            if asr_stream is not None:
                asr_stream.accept_waveform(SR, mono)
                while self.asr.is_ready(asr_stream):
                    self.asr.decode_stream(asr_stream)
                asr_text = self.asr.get_result(asr_stream)
                if self.asr.is_endpoint(asr_stream):
                    self.asr.reset(asr_stream)

            # 每 ~0.3s 打读数
            now = time.monotonic()
            if self.readout and now - last_status >= 0.3:
                rms = (sq_acc / n_acc) ** 0.5 if n_acc else 0.0
                bar = "#" * min(20, int(rms * 400))   # RMS 粗略可视化
                log(f"📊 RMS={rms:.4f} peak={peak_win:.3f} dec={decode_blocks:>2} "
                    f"asr=「{asr_text}」 {bar}")
                last_status = now
                peak_win = 0.0
                sq_acc = 0.0
                n_acc = 0
                decode_blocks = 0

    def hits_in(self, t_start: float, t_end: float) -> "list[tuple[float, str]]":
        with self.lock:
            return [(t, k) for (t, k) in self.hits if t_start <= t <= t_end]


def phase_quiet(reader: KwsReader, dur: float) -> int:
    log("=" * 64)
    log(f"【阶段1 · 安静基线 {dur:.0f}s】请保持安静,不要说话。期望:0 触发")
    t0 = time.monotonic()
    time.sleep(dur)
    n = len(reader.hits_in(t0, time.monotonic()))
    log(f"阶段1 完成:{n} 次触发({'✅ 通过' if n == 0 else '❌ 有误触'})")
    return n


def phase_bg(reader: KwsReader, dur: float) -> int:
    log("=" * 64)
    log(f"【阶段2 · 背景人声/电视 {dur:.0f}s】请现在【开始播放】电视/人声(别说唤醒词)。期望:0 误触")
    log("   3 秒后开始计时 …")
    time.sleep(3.0)
    t0 = time.monotonic()
    time.sleep(dur)
    n = len(reader.hits_in(t0, time.monotonic()))
    log(f"阶段2 完成:{n} 次误触({'✅ 通过' if n == 0 else '❌ 有误触'})。可以【停止】背景声了。")
    return n


def phase_wake(reader: KwsReader, n_trials: int, gap: float):
    log("=" * 64)
    log(f"【阶段3 · 唤醒检出 ×{n_trials}】每次提示后请清晰说一遍「小艺小艺」,间隔 {gap:.0f}s。")
    log("   2 秒后开始 …")
    time.sleep(2.0)
    results = []  # (命中?, 检出延迟 or None)
    for i in range(1, n_trials + 1):
        t_prompt = time.monotonic()
        log(f"  ▶ 第 {i}/{n_trials} 次:现在说「小艺小艺」")
        time.sleep(gap)
        hits = reader.hits_in(t_prompt, time.monotonic())
        if hits:
            lat = hits[0][0] - t_prompt
            results.append((True, lat))
            log(f"    ✅ 检出(提示→检出 {lat:.2f}s{',多次命中 '+str(len(hits)) if len(hits)>1 else ''})")
        else:
            results.append((False, None))
            log("    ❌ 未检出")
    return results


def summarize(args, reader, quiet_trips, bg_trips, wake_results):
    log("=" * 64)
    log("【汇总】")
    n_ok = sum(1 for ok, _ in wake_results if ok)
    lats = [lat for ok, lat in wake_results if ok and lat is not None]
    log(f"  唤醒检出率 : {n_ok}/{len(wake_results)}")
    if lats:
        log(f"  检出延迟   : 均值 {np.mean(lats):.2f}s  中位 {np.median(lats):.2f}s "
            f"(含人耳反应+说话时长,非纯引擎延迟)")
    log(f"  误触发     : 安静 {quiet_trips} + 背景声 {bg_trips} = {quiet_trips + bg_trips}")
    if reader.chunk_ms:
        ct = np.array(reader.chunk_ms)
        log(f"  单块推理   : 均值 {ct.mean():.2f}ms  p95 {np.percentile(ct,95):.2f}ms  max {ct.max():.2f}ms")
    log(f"  累计上行   : {reader.samples / SR:.1f}s 音频")
    passed = (n_ok >= 9) and (quiet_trips + bg_trips == 0)
    log("-" * 64)
    log(f"  验收门(检出≥9/10 且 误触=0):{'✅ 通过 → 可进 M1b' if passed else '❌ 未达标,在 standalone 调参再测'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--score", type=float, default=1.0)
    ap.add_argument("--trailing-blanks", type=int, default=1, dest="trailing_blanks")
    ap.add_argument("--fp32", action="store_true", help="用 fp32 模型(默认 int8)")
    ap.add_argument("--quiet-s", type=float, default=30.0, dest="quiet_s")
    ap.add_argument("--bg-s", type=float, default=60.0, dest="bg_s")
    ap.add_argument("--n-wake", type=int, default=10, dest="n_wake")
    ap.add_argument("--wake-gap", type=float, default=3.0, dest="wake_gap")
    ap.add_argument("--free", action="store_true", help="只持续监听打印命中,不跑编排测试")
    ap.add_argument("--wake-only", action="store_true", dest="wake_only",
                    help="跳过安静/背景两段,直接进唤醒×N(诊断快捷)")
    ap.add_argument("--no-asr", action="store_true", dest="no_asr",
                    help="不挂诊断 ASR(只看 RMS/峰值/解码活动)")
    ap.add_argument("--probe", action="store_true",
                    help="阈值阶梯探测:叠词/单字×各阈值并跑,打最高触发档(读不到内部分数的替代)")
    ap.add_argument("--varprobe", action="store_true",
                    help="变体阶梯:声调/韵尾/连读各形态×{0.20,0.40}并跑,看真人发音落到哪个 token 行")
    # 生产配置(第二步)
    ap.add_argument("--prod", action="store_true",
                    help="生产配置:单字多声调高阈+叠词低阈+二次确认;跑召回+背景误触两关")
    ap.add_argument("--single-thr", type=float, default=0.17, dest="single_thr")  # WAKE-01 锁定值
    ap.add_argument("--debounce", type=float, default=0.3)
    ap.add_argument("--refractory", type=float, default=2.0)
    ap.add_argument("--bg-min", type=float, default=5.0, dest="bg_min")
    ap.add_argument("--diag", action="store_true", help="prod 召回时开漏检诊断旁路")
    args = ap.parse_args()

    light = args.probe or args.varprobe or args.prod
    kws = None if light else build_kws(args)
    asr = None if (args.no_asr or light) else build_asr(args)
    stop = threading.Event()

    log("连接 Reachy Mini(media_backend=default)…")
    with ReachyMini(connection_mode="localhost_only", media_backend="default",
                    automatic_body_yaw=False) as mini:
        mini.media.start_recording()
        log("✅ 录音管线已启动")
        # 排空预热期旧音频(限时 2s)
        dl = time.monotonic() + 2.0
        while time.monotonic() < dl and mini.media.get_audio_sample() is not None:
            pass

        if args.prod:
            kwf = write_prod_keywords(args.single_thr)
            log(f"📝 生产 keywords({os.path.basename(kwf)}):单字#{args.single_thr}(命中即唤醒)")
            with open(kwf, encoding="utf-8") as f:
                for ln in f:
                    log(f"     {ln.rstrip()}")
            conf = WakeConfirmer(args.debounce, args.refractory)
            log(f"🧩 单字唤醒:去抖 {args.debounce}s · 不应期 {args.refractory}s")
            reader = ProdReader(mini, args.fp32, kwf, conf, stop, diag=args.diag)
        elif args.varprobe:
            reader = VariantProbeReader(mini, args.fp32, stop)
        elif args.probe:
            reader = ScoreProbeReader(mini, args.fp32, stop)
        else:
            reader = KwsReader(mini, kws, asr, stop, readout=True)
        reader.start()
        try:
            if args.prod and args.free:
                log("【生产·自由监听】我来给喊话节拍,持续记录原始命中 + 真唤醒。Ctrl+C 退出 …")
                while True:
                    time.sleep(1.0)
            elif args.prod:
                r = prod_phase_recall(reader, args.n_wake, max(args.wake_gap, 4.0))
                if args.wake_only:
                    log("(--wake-only:跳过背景误触段)")
                    prod_summarize(r, 0)
                else:
                    b = prod_phase_bg(reader, args.bg_min)
                    prod_summarize(r, b)
            elif args.varprobe:
                log("【变体阶梯】请说「小艺」(单遍和连读两遍穿插),我读触发的 token 形态。Ctrl+C 退出 …")
                while True:
                    time.sleep(1.0)
            elif args.probe:
                log("【阈值阶梯探测】请按提示说「小艺小艺」,我读最高触发档。Ctrl+C 退出 …")
                while True:
                    time.sleep(1.0)
            elif args.free:
                log("【自由监听】持续打印 RMS/ASR 读数 + 命中,Ctrl+C 退出 …")
                while True:
                    time.sleep(1.0)
            elif args.wake_only:
                log("(--wake-only:跳过安静/背景两段)")
                w = phase_wake(reader, args.n_wake, args.wake_gap)
                summarize(args, reader, 0, 0, w)
            else:
                q = phase_quiet(reader, args.quiet_s)
                b = phase_bg(reader, args.bg_s)
                w = phase_wake(reader, args.n_wake, args.wake_gap)
                summarize(args, reader, q, b, w)
        except KeyboardInterrupt:
            print(flush=True)
            log("收到 Ctrl+C,退出")
        finally:
            stop.set()
            time.sleep(0.2)
            try:
                mini.media.stop_recording()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
