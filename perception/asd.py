# -*- coding: utf-8 -*-
"""Active Speaker Detection(谁在说话)——移植 asd-demo 的 LR-ASD(IJCV 2025, AVA mAP 94.45)。

完全保留原实现:LR-ASD 模型代码原样拷在 perception/lr_asd/(model/ + loss.py),
本文件只做"加载 + 预处理 + 前向"的薄封装,前向逻辑与 asd-demo 的 lr_asd_adapter.py 一致:
    forward_audio_frontend → forward_visual_frontend → forward_audio_visual_backend
    → lossAV.forward(labels=None) → 逐帧 logit(>0 = 说话)。

输入(单个人脸轨迹):
  - 视频:若干 112×112 灰度帧(25fps),值域 [0,255]
  - 音频:同步的 16kHz mono float 音频
输出:逐帧 speaking 分数(signed logit),以及二值 speaking、窗口均分。

依赖:torch(CPU)、python_speech_features。缺任一则 available=False,调用方退化(只用 DOA)。
"""
from __future__ import annotations

import os
import sys
import time
import threading
import collections
import numpy as np
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_LR_ASD_DIR = _THIS / "lr_asd"                       # 原样移植的 LR-ASD 模型代码
_DEFAULT_WEIGHT = _THIS.parent / "models" / "lr_asd_finetuning_TalkSet.model"

# asd-demo 固定参数(勿改)
MFCC_NUMCEP = 13
MFCC_WINLEN = 0.025      # 25ms
MFCC_WINSTEP = 0.010     # 10ms → 100fps
AUDIO_SR = 16000
VIDEO_FPS = 25
FACE_SIZE = 112


def preprocess_face(rgb_crop: np.ndarray) -> np.ndarray:
    """人脸 ROI(RGB)→ 112×112 灰度 float32(asd-demo 预处理:resize224→中心裁112)。"""
    import cv2
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (224, 224))
    gray = gray[56:168, 56:168]                      # 中心裁 112×112
    return gray.astype(np.float32)


def syncnet_crop(full_rgb: np.ndarray, bbox_xyxy, crop_scale: float = 0.4):
    """asd-demo(SyncNet/TalkNet)式裁剪几何:让"嘴"落在 112 窗口正中央(模型训练时的位置)。
    bbox_xyxy = 全分辨率 [x1,y1,x2,y2]。纵向不对称(上 bs / 下 1.8bs),横向 ±1.4bs。
    返回 112×112 灰度 float32 或 None。"""
    import cv2
    H, W = full_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bs = max(x2 - x1, y2 - y1) / 2.0                  # 半边长
    if bs <= 4:
        return None
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0          # 脸中心
    cs = crop_scale
    ty0 = int(round(my - bs))                          # 纵向不对称:上 bs
    ty1 = int(round(my + bs * (1 + 2 * cs)))           #            下 1.8bs → 嘴居中
    tx0 = int(round(mx - bs * (1 + cs)))               # 横向对称 ±1.4bs
    tx1 = int(round(mx + bs * (1 + cs)))
    cy0, cy1 = max(0, ty0), min(H, ty1)
    cx0, cx1 = max(0, tx0), min(W, tx1)
    if cy1 - cy0 < 8 or cx1 - cx0 < 8:
        return None
    face = full_rgb[cy0:cy1, cx0:cx1]
    pt, pb, pl, pr = cy0 - ty0, ty1 - cy1, cx0 - tx0, tx1 - cx1   # 越界补边(复制)
    if pt or pb or pl or pr:
        face = cv2.copyMakeBorder(face, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (224, 224))
    return gray[56:168, 56:168].astype(np.float32)     # 中心裁 112 → 嘴在正中


class SpeakerDetector:
    """LR-ASD 单轨迹主动说话人检测(CPU)。"""

    def __init__(self, weight_path: str | None = None, device: str = "auto",
                 speak_thresh: float = 0.0, smooth_window: int = 2):
        self.available = False
        self.speak_thresh = speak_thresh
        self.smooth_window = smooth_window
        self._mfcc = None
        try:
            import torch
            import python_speech_features
            self._torch = torch
            self._mfcc = python_speech_features.mfcc
        except Exception as e:
            print(f"[asd] torch/python_speech_features 缺失 → ASD 不可用({type(e).__name__}: {e})", flush=True)
            return

        weight = weight_path or os.environ.get("LR_ASD_WEIGHT", str(_DEFAULT_WEIGHT))
        if not os.path.exists(weight):
            print(f"[asd] LR-ASD 权重不存在: {weight} → ASD 不可用", flush=True)
            return
        if str(_LR_ASD_DIR) not in sys.path:
            sys.path.insert(0, str(_LR_ASD_DIR))     # 让 from model.Model / from loss 解析到移植副本
        try:
            from model.Model import ASD_Model
            from loss import lossAV
            if device == "cpu":
                self.device = torch.device("cpu")
            else:                                            # auto/cuda:有就用 GPU(LR-ASD 0.22ms vs CPU 18ms)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ASD_Model()
            self.lossAV = lossAV()
            sd = torch.load(weight, map_location="cpu", weights_only=False)
            m_sd, l_sd = {}, {}
            for k, v in sd.items():
                k = k.replace("module.", "")
                if k.startswith("model."):
                    m_sd[k[len("model."):]] = v
                elif k.startswith("lossAV."):
                    l_sd[k[len("lossAV."):]] = v
            self.model.load_state_dict(m_sd, strict=True)
            self.lossAV.load_state_dict(l_sd, strict=True)
            self.model.to(self.device).eval()
            self.lossAV.to(self.device).eval()
            self.available = True
            print(f"[asd] LR-ASD 就绪({os.path.basename(weight)}, {self.device})", flush=True)
        except Exception as e:
            print(f"[asd] LR-ASD 加载失败({type(e).__name__}: {e}) → ASD 不可用", flush=True)

    # ── 推理(与 asd-demo lr_asd_adapter / vision_processer 一致)──
    def score(self, gray_frames: list[np.ndarray], audio_16k: np.ndarray) -> dict | None:
        """gray_frames: list of 112×112 float32 灰度脸(25fps);audio_16k: 同步 16k float。
        返回 {scores[Tv], scores_smooth[Tv], speaking[Tv], mean_score} 或 None。"""
        if not self.available or not gray_frames or audio_16k is None:
            return None
        torch = self._torch
        try:
            video_feature = np.expand_dims(np.stack(gray_frames), axis=0).astype(np.float32)  # [1,Tv,112,112]
            audio_feature = self._mfcc(audio_16k, AUDIO_SR, numcep=MFCC_NUMCEP,
                                       winlen=MFCC_WINLEN, winstep=MFCC_WINSTEP)               # [Ta,13]
            # 时序对齐 4:1(100fps MFCC : 25fps 视频),与 asd-demo 一致
            length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100.0,
                         video_feature.shape[1] / float(VIDEO_FPS))
            if length <= 0.2:
                return None
            audio_feature = audio_feature[:int(round(length * 100))]
            video_feature = video_feature[:, :int(round(length * VIDEO_FPS))]
            audio_feature = np.expand_dims(audio_feature, axis=0).astype(np.float32)           # [1,Ta,13]

            with torch.no_grad():
                a = torch.from_numpy(np.ascontiguousarray(audio_feature)).float().to(self.device)
                v = torch.from_numpy(np.ascontiguousarray(video_feature)).float().to(self.device)
                embA = self.model.forward_audio_frontend(a)
                embV = self.model.forward_visual_frontend(v)
                outsAV = self.model.forward_audio_visual_backend(embA, embV)
                scores = self.lossAV.forward(outsAV, labels=None)  # numpy [Tv], raw logit x[:,1]
            scores = np.asarray(scores, dtype=float).reshape(-1)
            if scores.size == 0:
                return None
            # ±smooth_window 帧平滑(asd-demo vision_processer:279)
            w = self.smooth_window
            sm = np.array([float(np.mean(scores[max(i - w, 0):min(i + w + 1, len(scores))]))
                           for i in range(len(scores))])
            return {
                "scores": scores,
                "scores_smooth": sm,
                "speaking": sm >= self.speak_thresh,
                "mean_score": float(np.mean(sm[-6:])),   # 最近 6 帧均分(webcam demo 口径)
            }
        except Exception as e:
            print(f"[asd] score 异常({type(e).__name__}: {e})", flush=True)
            return None


SAMPLES_PER_FRAME = AUDIO_SR // VIDEO_FPS    # 640(16000/25)


class AudioRing:
    """线程安全 16kHz mono 环形缓冲(供 ASD 取同步音频窗口)。"""

    def __init__(self, seconds: float = 3.0):
        self.maxlen = int(AUDIO_SR * seconds)
        self.buf = np.zeros(self.maxlen, dtype=np.float32)
        self.filled = 0
        self.pos = 0
        self.t_last_mono = 0.0       # 最后 push 的单调时刻(≈最新样本时间),用于时间↔样本映射
        self._lock = threading.Lock()

    def push(self, mono):
        x = np.asarray(mono, dtype=np.float32).reshape(-1)
        n = x.size
        if n == 0:
            return
        if n >= self.maxlen:
            x = x[-self.maxlen:]; n = self.maxlen
        with self._lock:
            end = self.pos + n
            if end <= self.maxlen:
                self.buf[self.pos:end] = x
            else:
                k = self.maxlen - self.pos
                self.buf[self.pos:] = x[:k]
                self.buf[:end - self.maxlen] = x[k:]
            self.pos = end % self.maxlen
            self.filled = min(self.maxlen, self.filled + n)
            self.t_last_mono = time.monotonic()

    def get_last(self, n):
        with self._lock:
            n = min(n, self.filled)
            if n <= 0:
                return None
            start = (self.pos - n) % self.maxlen
            if start + n <= self.maxlen:
                return self.buf[start:start + n].copy()
            return np.concatenate([self.buf[start:], self.buf[:(start + n) % self.maxlen]])

    def get_window(self, t0, t1):
        """取 [t0,t1] 时间段的样本(按 t_last_mono 把单调时间映射到样本)。"""
        if self.t_last_mono <= 0:
            return None
        n_after = max(0, int(round((self.t_last_mono - t1) * AUDIO_SR)))   # t1 之后的样本数
        n_win = int(round((t1 - t0) * AUDIO_SR))
        if n_win < AUDIO_SR // 10:                                         # <0.1s 不取
            return None
        seg = self.get_last(n_after + n_win)
        if seg is None or seg.size < n_win:
            return None
        return seg[:n_win]


class AsdEngine:
    """编排:per-track 灰度脸累积 + 独立线程跑 LR-ASD → per-track 说话分(EMA)。
    d01 用法:feed_audio(mic mono) / feed_crop(track_id, rgb_face, now) / start();
              读 scores() 或 speaker()。"""

    def __init__(self, detector: "SpeakerDetector | None" = None,
                 win_frames: int = 48, min_frames: int = 12, win_seconds: float = 1.3,
                 score_interval_s: float = 0.16, crop_fps: int = VIDEO_FPS,
                 ema: float = 0.5, speak_thresh: float = 0.0, stale_s: float = 1.0):
        self.detector = detector or SpeakerDetector()
        self.win = win_frames
        self.win_seconds = win_seconds
        self.min_frames = min_frames
        self.score_interval = score_interval_s
        self.crop_dt = 1.0 / float(crop_fps)
        self.ema = ema
        self.speak_thresh = speak_thresh
        self.stale_s = stale_s
        self.audio = AudioRing()
        # 键 = 说话人标识(调用方传 person_id 或 f"t{track_id}"):按身份聚合,抗 track churn
        self._crops: dict = {}
        self._last_crop_t: dict = {}
        self._scores: dict = {}
        self._score_t: dict = {}
        self._last_pos_t: dict = {}     # 每 key 最近一次"在说话"的时刻
        self._last_tid: dict = {}       # key → 最近 track_id(仅供显示/归属标签)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def available(self) -> bool:
        return self.detector.available

    def feed_audio(self, mono):
        self.audio.push(mono)

    def feed_crop(self, key, full_rgb: np.ndarray, bbox_xyxy, now: float, track_id=None):
        """对每个 confirmed 说话人喂一帧:从全分辨率帧按 SyncNet 几何裁脸(嘴居中)→112 灰度。
        key = 说话人标识(person_id 或 f"t{track_id}"):按身份聚合,track churn 换 id 不丢累积。
        bbox_xyxy = 全分辨率 [x1,y1,x2,y2]。内部限到 ~crop_fps。"""
        last = self._last_crop_t.get(key, 0.0)
        if now - last < self.crop_dt:
            return
        g = syncnet_crop(full_rgb, bbox_xyxy)
        if g is None:
            return
        with self._lock:
            dq = self._crops.get(key)
            if dq is None:
                dq = collections.deque(maxlen=self.win)
                self._crops[key] = dq
            dq.append((now, g))          # 存(时间戳, 灰度脸),供 25fps 重采样 + 音频对齐
            if track_id is not None:
                self._last_tid[key] = track_id
        self._last_crop_t[key] = now

    def start(self):
        if self._thread is not None or not self.available:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def gc(self, active_keys):
        """丢弃已消失说话人(key)的状态。"""
        alive = set(active_keys)
        with self._lock:
            for d in (self._crops, self._scores):
                for k in [k for k in d if k not in alive]:
                    d.pop(k, None)
            for k in [k for k in self._last_crop_t if k not in alive]:
                self._last_crop_t.pop(k, None)
                self._score_t.pop(k, None)
                self._last_pos_t.pop(k, None)
                self._last_tid.pop(k, None)

    def last_track(self, key):
        """key 对应的最近 track_id(仅供显示/归属标签);无则 None。"""
        return self._last_tid.get(key)

    def scores(self) -> dict[int, float]:
        with self._lock:
            return dict(self._scores)

    def speaker(self):
        """返回 (track_id, score):分最高且 >阈值 的 track;否则 None。"""
        now = time.monotonic()
        with self._lock:
            cand = [(tid, s) for tid, s in self._scores.items()
                    if s > self.speak_thresh and (now - self._score_t.get(tid, 0.0)) < self.stale_s]
        if not cand:
            return None
        return max(cand, key=lambda kv: kv[1])

    def speaking_ids(self):
        """当前确信在说话的 track 集合(>阈值且新鲜)——绿框用,带 speaker() 同款新鲜度门,
        治"说完停了 EMA 残留正值绿框不灭"。不改任何参数(保持判断敏感)。"""
        now = time.monotonic()
        with self._lock:
            return {tid for tid, s in self._scores.items()
                    if s > self.speak_thresh and (now - self._score_t.get(tid, 0.0)) < self.stale_s}

    def speaker_window(self, t_start: float):
        """本句归属(耐 ASD 延迟):自 t_start 起任意时刻被判为说话的 track 中分最高者。
        即使'说完才出分',只要这句话期间检测到在说就能归对。返回 (track_id, score)|None。"""
        with self._lock:
            cand = [(tid, self._scores.get(tid, -9.9))
                    for tid, t in self._last_pos_t.items() if t >= t_start]
        if not cand:
            return None
        return max(cand, key=lambda kv: kv[1])

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(self.score_interval)
            if not self.detector.available:
                continue
            with self._lock:
                items = [(tid, list(dq)) for tid, dq in self._crops.items()]
            for tid, crops in items:    # crops: [(t, gray), ...]
                if len(crops) < self.min_frames:
                    continue
                t1 = crops[-1][0]
                crops = [c for c in crops if c[0] >= t1 - self.win_seconds]  # 时间窗封顶,低 fps 不稀释
                if len(crops) < self.min_frames:
                    continue
                t0 = crops[0][0]
                span = t1 - t0
                if span < 0.4:                                  # 不足 ~0.4s 不评
                    continue
                # GPU 让检测回 ~25fps → 直接喂真帧(不重采样,webcam 同款,避免重采样引入失同步)
                vid = [c[1] for c in crops]
                audio = self.audio.get_window(t0, t1)           # 同一时间段的音频
                if audio is None or audio.size < SAMPLES_PER_FRAME:
                    continue
                if float(np.abs(audio).max()) <= 4.0:           # 归一化 float → int16 量级(MFCC 期望)
                    audio = audio * 32768.0
                r = self.detector.score(vid, audio)
                if r is None:
                    continue
                sc = r["mean_score"]
                _nowm = time.monotonic()
                with self._lock:
                    prev = self._scores.get(tid)
                    self._scores[tid] = sc if prev is None else (self.ema * sc + (1 - self.ema) * prev)
                    self._score_t[tid] = _nowm
                    if self._scores[tid] > self.speak_thresh:
                        self._last_pos_t[tid] = _nowm           # 最近一次"在说话"的时刻(供归属耐延迟)
