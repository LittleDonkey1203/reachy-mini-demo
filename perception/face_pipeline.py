# -*- coding: utf-8 -*-
"""人脸 ReID 集成层(主进程)——把 ByteTracker + 全分辨率 ArcFace + 三区间 IdentityStore
串起来,供 d01 vision_result_loop 调用。完全参考 face-tracker-demo/pipeline.py 的编排,
但适配机器人架构:

  - 检测在子进程(SCRFD,见 vision_worker),这里只消费 all_faces(降采样像素 bbox/kps/conf)
  - 识别用注入的 embedder(d01 传 ArcFace,全分辨率裁脸;CI 传 fake)
  - 出口给 behavior 的是**归一化 u/v/h**(由像素 bbox / 检测帧 W,H 换算)
  - 身份主键 = IdentityStore identity_id(d01 用作 person_id 接 OwnerManager/记忆)

铁律:本模块纯逻辑,不写 st.state、不调 head_control;由 d01 在主进程视觉线程调用。
"""
from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from perception.face_config import FaceSystemConfig
from perception.face_tracker import ByteTracker, Detection, STrack
from perception.quality import compute_quality_proxy
from identity.identity_store import IdentityStore

# embedder(full_rgb, box_xywh_fullres, kps_fullres|None) -> 512d L2-normed | None
Embedder = Callable[[np.ndarray, tuple, Optional[list]], Optional[np.ndarray]]


@dataclass
class TrackView:
    """对外暴露的单 track 视图(归一化坐标 + 身份 + 注视)。"""
    track_id: int
    u: float
    v: float
    h: float
    bbox_px: tuple            # 检测(降采样)帧像素 [x1,y1,x2,y2]
    state: str                # confirmed | tentative
    person_id: Optional[str]
    person_name: Optional[str]
    zone: str                 # known | unsure | unknown | ""
    confidence: float
    is_confirmed: bool = False   # gallery 身份是否已用户确认(命名)
    gaze_yaw: float = 0.0       # L2 注视 yaw (度)
    gaze_pitch: float = 0.0     # L2 注视 pitch (度)
    mutual_gaze: bool = False   # 是否在看摄像头
    head_yaw: float = 0.0       # L0 几何头姿 yaw (度)
    head_pitch: float = 0.0     # L0 几何头姿 pitch (度)


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


class FaceReIDPipeline:
    """ByteTracker + 三区间 IdentityStore 集成。"""

    def __init__(self, embedder: Embedder, config: FaceSystemConfig | None = None,
                 emb_per_frame_budget: int = 2, per_track_emb_interval_s: float = 0.35,
                 refresh_interval_s: float = 2.0, log_fn=None):
        self.cfg = config or FaceSystemConfig()
        self.embedder = embedder
        self.tracker = ByteTracker(self.cfg.tracking)
        self.store = IdentityStore(self.cfg.identity)
        self._emb_budget = emb_per_frame_budget
        self._track_interval = per_track_emb_interval_s
        self._refresh_interval = refresh_interval_s
        self._log_fn = log_fn        # 识别可观测:每次 track 绑定/新建身份时打印(含距离)
        # per-track 状态:上次提特征时刻 / 连续 unknown 帧数
        self._last_emb: dict[int, float] = {}
        self._unknown_cnt: dict[int, int] = {}
        self._gaze = None  # GazeModule; d01 注入

    def load_gallery(self) -> int:
        return self.store.load()

    def save_gallery(self):
        self.store.save()

    # ── 每帧调用 ──────────────────────────────────────────
    def process(self, all_faces: Optional[list], det_wh: tuple,
                full_rgb: Optional[np.ndarray], decimate: int,
                now: Optional[float] = None, doa_idx: Optional[int] = None
                ) -> tuple[Optional[TrackView], list[TrackView]]:
        """消费一帧检测,返回 (primary, all_track_views)。
        all_faces 项:{u,v,h,box(x,y,w,h 降采样px),kps(5×2 降采样px)|None,conf}。"""
        if now is None:
            now = time.monotonic()
        W, H = det_wh
        dets: list[Detection] = []
        for af in (all_faces or []):
            kps = af.get("kps")
            lmk = (np.asarray(kps, dtype=np.float32) if kps
                   else np.zeros((5, 2), dtype=np.float32))
            dets.append(Detection(bbox=_xywh_to_xyxy(af["box"]),
                                  confidence=float(af.get("conf", 0.9)),
                                  landmarks=lmk, embedding=None,
                                  quality=0.0))

        # 子进程检测不带 embedding(主进程事后懒算)→ 这条路 dets 恒无 embedding。
        # 必须在关联前把 embedding_weight 清零,否则 embedding_distance 返回全 1.0,
        # 给每对关联加 0.3 惩罚,把 IoU 门从名义 0.30 抬到 0.429 → 低 fps 下丢轨重建
        # (镜像 face-tracker-demo pipeline.py 的 split-path 守卫)。
        _emb_w = self.tracker.cfg.embedding_weight
        if all(d.embedding is None for d in dets):
            self.tracker.cfg.embedding_weight = 0.0
        try:
            self.tracker.update(dets)
        finally:
            self.tracker.cfg.embedding_weight = _emb_w
        active = self.tracker.get_all_active()

        # ── 懒提特征 + 三区间身份(只对 confirmed track,限频 + 每帧预算)──
        if full_rgb is not None:
            budget = self._emb_budget
            # DOA 选中的脸优先(把它对应的 track 排前)
            ordered = self._order_by_doa(active, all_faces, doa_idx)
            for trk in ordered:
                if budget <= 0:
                    break
                if not trk.is_confirmed():
                    continue
                if not self._needs_embedding(trk, now):
                    continue
                emb, quality = self._extract(trk, full_rgb, decimate)
                self._last_emb[trk.track_id] = now
                budget -= 1
                if emb is None:
                    continue
                trk.inject_embedding(emb, quality)
                self._assign_identity(trk, quality)

        self._gc_state(active)

        # ── per-track 注视估计(L0+L1+L2 级联) ──
        _gaze_results: dict = {}
        if self._gaze is not None and self._gaze.available and full_rgb is not None:
            alive_keys = {trk.identity_id or f"t{trk.track_id}" for trk in active}
            self._gaze.gc(alive_keys)
            for trk in active:
                if trk.landmarks is None or not np.any(trk.landmarks):
                    continue
                gr = self._gaze.update(
                    trk.track_id, trk.landmarks, full_rgb, trk.bbox, decimate,
                    identity_key=trk.identity_id,
                    frame_w=full_rgb.shape[1] if full_rgb is not None else 0)
                _gaze_results[trk.track_id] = gr

        views = [self._view(t, _gaze_results) for t in active]
        prim = self.tracker.get_primary_target()
        primary = self._view(prim, _gaze_results) if (prim is not None and prim.is_confirmed()) else None
        # 归一化:用检测帧 W,H
        return self._normalize(primary, W, H), [self._normalize(v, W, H) for v in views]

    # ── 身份命名(d01 在 remember_fact 时调)──────────────
    def name_track(self, track_id: int, name: str) -> bool:
        """把某 track 的身份命名为 name(provisional→confirmed,或新建 confirmed)。"""
        trk = self.tracker.find_track(track_id)
        if trk is None or trk.smooth_embedding is None:
            return False
        if trk.identity_id and trk.identity_id in self.store.identities:
            self.store.confirm_identity(trk.identity_id, name)
            trk.identity_name = name
            return True
        pid = self.store.register_identity(name, [trk.smooth_embedding],
                                           [trk.best_quality], confirmed=True)
        trk.identity_id = pid
        trk.identity_name = name
        return True

    # ── 内部 ──────────────────────────────────────────────
    def _needs_embedding(self, trk: STrack, now: float) -> bool:
        # Q4 身份冻结:已绑定 track 不再识别(只有新/未绑定 track 才跑识别);
        # 配合稳定 track(DECIMATE=3),"track_id 变了才识别、track 在则身份不变"。
        if trk.identity_id is not None:
            return False
        last = self._last_emb.get(trk.track_id, 0.0)
        return (now - last) >= self._track_interval            # 未识别:积极取特征直到绑定

    def _extract(self, trk: STrack, full_rgb: np.ndarray, decimate: int):
        x1, y1, x2, y2 = trk.bbox
        box_full = (int(x1 * decimate), int(y1 * decimate),
                    int((x2 - x1) * decimate), int((y2 - y1) * decimate))
        kps_full = None
        if trk.landmarks is not None and np.any(trk.landmarks):
            kps_full = [(float(px * decimate), float(py * decimate)) for px, py in trk.landmarks]
        quality = compute_quality_proxy(
            np.array([box_full[0], box_full[1], box_full[0] + box_full[2], box_full[1] + box_full[3]],
                     dtype=np.float32),
            trk.confidence, np.asarray(kps_full) if kps_full else trk.landmarks, full_rgb.shape)
        try:
            emb = self.embedder(full_rgb, box_full, kps_full)
        except Exception:
            emb = None
        return emb, quality

    def _assign_identity(self, trk: STrack, quality: float):
        probe = trk.smooth_embedding
        if probe is None:
            return
        _old = trk.identity_id
        r = self.store.match(probe, quality)
        if r.zone == "known":
            trk.identity_id = r.identity_id
            trk.identity_name = r.identity_name
            trk.identity_confidence = r.confidence
            trk.identity_zone = "known"
            self._unknown_cnt[trk.track_id] = 0
            if quality >= self.cfg.identity.min_quality:
                self.store.match_and_update(probe, quality)   # 刷 gallery
            if self._log_fn and r.identity_id != _old:        # 仅在绑定/换人时打印
                self._log_fn(f"🔍 track {trk.track_id} → {r.identity_name} "
                             f"(known dist={r.distance:.3f} q={quality:.2f})")
        elif r.zone == "unsure":
            trk.identity_zone = "unsure"                       # 不提交身份,继续跟踪
        else:  # unknown
            trk.identity_zone = "unknown"
            self._unknown_cnt[trk.track_id] = self._unknown_cnt.get(trk.track_id, 0) + 1
            if (trk.identity_id is None
                    and self._unknown_cnt[trk.track_id] >= self.cfg.identity.min_confirm_frames
                    and quality >= self.cfg.identity.min_quality):
                ur = self.store.register_unknown(probe, quality)
                trk.identity_id = ur.identity_id
                trk.identity_name = ur.identity_name
                trk.identity_confidence = ur.confidence
                if self._log_fn:
                    self._log_fn(f"🔍 track {trk.track_id} → 新建 {ur.identity_name} "
                                 f"(unknown dist={r.distance:.3f})")

    def _order_by_doa(self, active, all_faces, doa_idx):
        if doa_idx is None or not all_faces or doa_idx >= len(all_faces):
            return active
        af = all_faces[doa_idx]
        tx, ty = af["box"][0] + af["box"][2] / 2, af["box"][1] + af["box"][3] / 2
        def d(t):
            b = t.bbox
            return (((b[0] + b[2]) / 2 - tx) ** 2 + ((b[1] + b[3]) / 2 - ty) ** 2)
        return sorted(active, key=d)

    def _gc_state(self, active):
        alive = {t.track_id for t in active}
        for d in (self._last_emb, self._unknown_cnt):
            for tid in [k for k in d if k not in alive]:
                d.pop(tid, None)

    def _view(self, t: STrack, gaze_results: dict | None = None) -> TrackView:
        b = t.bbox
        _conf = bool(t.identity_id and t.identity_id in self.store.identities
                     and self.store.identities[t.identity_id].is_confirmed)
        gr = (gaze_results or {}).get(t.track_id)
        return TrackView(
            track_id=t.track_id, u=0.0, v=0.0, h=0.0,
            bbox_px=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            state="confirmed" if t.is_confirmed() else "tentative",
            person_id=t.identity_id, person_name=t.identity_name,
            zone=getattr(t, "identity_zone", ""), confidence=t.identity_confidence,
            is_confirmed=_conf,
            gaze_yaw=gr.gaze_yaw if gr else 0.0,
            gaze_pitch=gr.gaze_pitch if gr else 0.0,
            mutual_gaze=gr.mutual_gaze if gr else False,
            head_yaw=gr.head_yaw if gr else 0.0,
            head_pitch=gr.head_pitch if gr else 0.0)

    @staticmethod
    def _normalize(v: Optional[TrackView], W: int, H: int) -> Optional[TrackView]:
        if v is None:
            return None
        x1, y1, x2, y2 = v.bbox_px
        v.u = float((x1 + x2) / 2 / max(W, 1))
        v.v = float((y1 + y2) / 2 / max(H, 1))
        v.h = float((y2 - y1) / max(H, 1))
        return v
