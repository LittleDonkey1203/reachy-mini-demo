# -*- coding: utf-8 -*-
"""ByteTrack 式多人脸跟踪器(完全参考 face-tracker-demo/tracker.py)。

核心:
  1. Kalman 滤波(匀速模型,bbox 状态)做运动预测
  2. 两阶段 BYTE 关联:
     - Stage1: 高置信 det ↔ 现有 track(IoU + embedding 融合)
     - Stage2: 低置信 det ↔ 剩余 track(纯 IoU)
     - Stage3: 未匹配高置信 det ↔ lost track(纯 embedding 距离,ReID 找回)
  3. 生命周期:Tentative → Confirmed → Lost → Deleted

Reference: Zhang et al., "ByteTrack", ECCV 2022.

注:坐标全程用**像素 bbox** [x1,y1,x2,y2];机器人侧 behavior/head_control 需要的
归一化 u/v/h 由调用方(d01 出口适配器)从 bbox 换算,本模块不掺归一化坐标。
"""
from __future__ import annotations

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from perception.face_config import TrackingConfig


class TrackState(Enum):
    Tentative = auto()
    Confirmed = auto()
    Lost = auto()


@dataclass
class Detection:
    """单帧人脸检测(来自 SCRFD)。"""
    bbox: np.ndarray            # [x1, y1, x2, y2] 像素
    confidence: float
    landmarks: np.ndarray       # (5, 2) 5 点关键点(像素)
    embedding: Optional[np.ndarray] = None   # 512-d ArcFace,后填
    quality: float = 0.0        # FIQA 代理分


# ── Kalman(匀速 bbox 模型)─────────────────────────────────
# 状态 [cx, cy, s, r, vcx, vcy, vs, 0]:中心 x/y、面积 s=w*h、宽高比 r=w/h、各速度

class KalmanBoxFilter:
    """跟踪 (cx, cy, area, aspect_ratio) 的 Kalman 滤波。"""

    _motion_mat = np.eye(8, dtype=np.float32)
    _update_mat = np.eye(4, 8, dtype=np.float32)

    def __init__(self, bbox: np.ndarray):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        s = w * h
        r = w / max(h, 1e-6)

        self.mean = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
        std = [
            2 * max(w, h), 2 * max(w, h), 10 * s, 0.1,
            10 * max(w, h), 10 * max(w, h), 10 * s, 0.01,
        ]
        self.covariance = np.diag(np.square(std).astype(np.float32))
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def predict(self) -> np.ndarray:
        F = self._motion_mat.copy()
        for i in range(4):
            F[i, i + 4] = 1.0

        std_pos = self._std_weight_position * max(self.mean[2], 1)
        std_vel = self._std_weight_velocity * max(self.mean[2], 1)
        Q = np.diag(np.square([
            std_pos, std_pos, std_pos * 2, 1e-2,
            std_vel, std_vel, std_vel * 2, 1e-5,
        ]).astype(np.float32))

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q
        self.mean[2] = max(self.mean[2], 1.0)
        return self._state_to_bbox(self.mean)

    def update(self, bbox: np.ndarray):
        H = self._update_mat
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        measurement = np.array([cx, cy, w * h, w / max(h, 1e-6)], dtype=np.float32)

        std = self._std_weight_position * max(self.mean[2], 1)
        R = np.diag(np.square([std, std, std * 2, 1e-2]).astype(np.float32))

        y = measurement - H @ self.mean
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.mean = self.mean + K @ y
        I_KH = np.eye(8, dtype=np.float32) - K @ H
        self.covariance = I_KH @ self.covariance

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> np.ndarray:
        cx, cy, s, r = state[:4]
        s = max(s, 1.0)
        r = max(r, 0.1)
        h = np.sqrt(s / r)
        w = r * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


# ── 单个 Track ─────────────────────────────────────────────

class STrack:
    """带 Kalman 状态、embedding 历史与身份的单个跟踪人脸。"""

    _id_counter = 0

    def __init__(self, det: Detection, track_id: Optional[int] = None):
        STrack._id_counter += 1
        self.track_id: int = track_id or STrack._id_counter
        self.kalman = KalmanBoxFilter(det.bbox)
        self.state = TrackState.Tentative

        self.hits: int = 1
        self.age: int = 0
        self.time_since_update: int = 0

        self.bbox: np.ndarray = det.bbox.copy()
        self.confidence: float = det.confidence
        self.landmarks: np.ndarray = det.landmarks.copy()

        self.embeddings: list[np.ndarray] = []
        if det.embedding is not None:
            self.embeddings.append(det.embedding)
        self.smooth_embedding: Optional[np.ndarray] = det.embedding

        self.identity_id: Optional[str] = None
        self.identity_name: Optional[str] = None
        self.identity_confidence: float = 0.0

        self.best_quality: float = det.quality

    @property
    def predicted_bbox(self) -> np.ndarray:
        return KalmanBoxFilter._state_to_bbox(self.kalman.mean)

    def predict(self):
        self.bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection, frame_idx: int = 0):
        self.kalman.update(det.bbox)
        self.bbox = det.bbox.copy()
        self.confidence = det.confidence
        self.landmarks = det.landmarks.copy()
        self.hits += 1
        self.time_since_update = 0

        if det.embedding is not None:
            self._absorb_embedding(det.embedding)
        if det.quality > self.best_quality:
            self.best_quality = det.quality

    def _absorb_embedding(self, emb: np.ndarray, alpha: float = 0.7):
        """ring buffer(10)+ EMA 平滑。供 update() 与外部 inject_embedding 共用。"""
        self.embeddings.append(emb)
        if len(self.embeddings) > 10:
            self.embeddings = self.embeddings[-10:]
        if self.smooth_embedding is not None:
            s = alpha * emb + (1 - alpha) * self.smooth_embedding
            self.smooth_embedding = (s / (np.linalg.norm(s) + 1e-8)).astype(np.float32)
        else:
            self.smooth_embedding = emb

    def inject_embedding(self, emb: np.ndarray, quality: float = 0.0):
        """外部(主进程懒提 ArcFace 后)灌入 embedding。机器人侧 det/rec 解耦用。"""
        if emb is None:
            return
        self._absorb_embedding(emb)
        if quality > self.best_quality:
            self.best_quality = quality

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_confirmed(self):
        self.state = TrackState.Confirmed

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0


# ── IoU 与 cost 工具 ───────────────────────────────────────

def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """两组 bbox 的 IoU 矩阵 (M, N)。"""
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.empty((len(bboxes_a), len(bboxes_b)), dtype=np.float32)

    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0].T)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def embedding_distance(tracks: list[STrack], detections: list[Detection]) -> np.ndarray:
    """track.smooth_embedding ↔ det.embedding 的余弦距离矩阵(缺则 1.0=最大距离)。"""
    M, N = len(tracks), len(detections)
    cost = np.ones((M, N), dtype=np.float32)
    for i, trk in enumerate(tracks):
        if trk.smooth_embedding is None:
            continue
        te = trk.smooth_embedding / (np.linalg.norm(trk.smooth_embedding) + 1e-8)
        for j, det in enumerate(detections):
            if det.embedding is None:
                continue
            de = det.embedding / (np.linalg.norm(det.embedding) + 1e-8)
            cost[i, j] = 1.0 - float(np.dot(te, de))
    return cost


def linear_assignment(cost_matrix: np.ndarray, threshold: float):
    """Hungarian 指派,超过 threshold 视为未匹配。
    返回 (matches[N,2], unmatched_rows, unmatched_cols)。"""
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(cost_matrix.shape[0]),
                np.arange(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > threshold:
            continue
        matches.append([r, c])
        unmatched_rows.discard(r)
        unmatched_cols.discard(c)
    matches = np.array(matches, dtype=int).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(sorted(unmatched_rows)), np.array(sorted(unmatched_cols))


# ── ByteTracker ───────────────────────────────────────────

class ByteTracker:
    """ByteTrack 式多人脸跟踪 + embedding ReID。"""

    def __init__(self, config: TrackingConfig | None = None):
        self.cfg = config or TrackingConfig()
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.removed: list[STrack] = []
        self.frame_id: int = 0

    def update(self, detections: list[Detection]) -> list[STrack]:
        """每帧调用。返回当前 confirmed track 列表。两阶段 BYTE + Stage3 ReID。"""
        self.frame_id += 1

        high_dets = [d for d in detections if d.confidence >= self.cfg.high_thresh]
        low_dets = [d for d in detections if self.cfg.low_thresh <= d.confidence < self.cfg.high_thresh]

        for trk in self.tracked + self.lost:
            trk.predict()

        # ── Stage1:高置信 det ↔ tracked,融合 IoU + embedding ──
        tracked_stracks = [t for t in self.tracked if t.state != TrackState.Lost]
        if len(tracked_stracks) > 0 and len(high_dets) > 0:
            trk_bboxes = np.array([t.predicted_bbox for t in tracked_stracks])
            det_bboxes = np.array([d.bbox for d in high_dets])
            iou_cost = 1.0 - iou_batch(trk_bboxes, det_bboxes)
            emb_cost = embedding_distance(tracked_stracks, high_dets)
            w = self.cfg.embedding_weight
            cost = (1 - w) * iou_cost + w * emb_cost
            matches_s1, unmatched_trks_s1, unmatched_dets_s1 = linear_assignment(
                cost, threshold=1.0 - self.cfg.iou_threshold)
        else:
            matches_s1 = np.empty((0, 2), dtype=int)
            unmatched_trks_s1 = np.arange(len(tracked_stracks))
            unmatched_dets_s1 = np.arange(len(high_dets))

        for m in matches_s1:
            tracked_stracks[m[0]].update(high_dets[m[1]], self.frame_id)

        # ── Stage2:低置信 det ↔ 剩余 tracked,纯 IoU ──
        remaining_trks = [tracked_stracks[i] for i in unmatched_trks_s1]
        if len(remaining_trks) > 0 and len(low_dets) > 0:
            trk_bboxes = np.array([t.predicted_bbox for t in remaining_trks])
            det_bboxes = np.array([d.bbox for d in low_dets])
            iou_cost = 1.0 - iou_batch(trk_bboxes, det_bboxes)
            matches_s2, unmatched_trks_s2, _ = linear_assignment(
                iou_cost, threshold=1.0 - self.cfg.iou_threshold)
        else:
            matches_s2 = np.empty((0, 2), dtype=int)
            unmatched_trks_s2 = np.arange(len(remaining_trks))

        for m in matches_s2:
            remaining_trks[m[0]].update(low_dets[m[1]], self.frame_id)

        for trk in [remaining_trks[i] for i in unmatched_trks_s2]:
            trk.mark_lost()

        # ── Stage3:未匹配高置信 det ↔ lost track,纯 embedding ReID ──
        unmatched_high_dets = [high_dets[i] for i in unmatched_dets_s1]
        if len(self.lost) > 0 and len(unmatched_high_dets) > 0:
            emb_cost = embedding_distance(self.lost, unmatched_high_dets)
            matches_s3, _, unmatched_dets_s3 = linear_assignment(
                emb_cost, threshold=self.cfg.embedding_threshold)
        else:
            matches_s3 = np.empty((0, 2), dtype=int)
            unmatched_dets_s3 = np.arange(len(unmatched_high_dets))

        for m in matches_s3:
            self.lost[m[0]].update(unmatched_high_dets[m[1]], self.frame_id)
            self.lost[m[0]].state = TrackState.Confirmed
            self.tracked.append(self.lost[m[0]])
        matched_lost_idx = set(m[0] for m in matches_s3)
        self.lost = [t for i, t in enumerate(self.lost) if i not in matched_lost_idx]

        # ── 真正未匹配的高置信 det → 新 track ──
        for i in unmatched_dets_s3:
            det = unmatched_high_dets[i]
            if det.confidence >= self.cfg.high_thresh:
                new_track = STrack(det)
                new_track.state = TrackState.Tentative
                self.tracked.append(new_track)

        # ── 生命周期 ──
        new_tracked = []
        for trk in self.tracked:
            if trk.state == TrackState.Lost:
                self.lost.append(trk)
            else:
                if trk.state == TrackState.Tentative and trk.hits >= self.cfg.min_hits:
                    trk.mark_confirmed()
                new_tracked.append(trk)
        self.tracked = new_tracked

        new_lost = []
        for trk in self.lost:
            if trk.time_since_update > self.cfg.max_age:
                self.removed.append(trk)
            else:
                new_lost.append(trk)
        self.lost = new_lost
        if len(self.removed) > 100:
            self.removed = self.removed[-50:]

        return [t for t in self.tracked if t.is_confirmed()]

    def get_all_active(self) -> list[STrack]:
        return list(self.tracked)

    def get_primary_target(self) -> Optional[STrack]:
        """当前最佳目标:confirmed 中最大脸(bbox 面积),无则取最大 tentative。"""
        def area(t: STrack) -> float:
            b = t.bbox
            return float((b[2] - b[0]) * (b[3] - b[1]))
        confirmed = [t for t in self.tracked if t.is_confirmed()]
        if confirmed:
            return max(confirmed, key=area)
        tentative = [t for t in self.tracked if t.state == TrackState.Tentative]
        return max(tentative, key=area) if tentative else None

    def find_track(self, track_id: int) -> Optional[STrack]:
        for t in self.tracked:
            if t.track_id == track_id:
                return t
        return None

    def reset(self):
        self.tracked.clear()
        self.lost.clear()
        self.removed.clear()
        self.frame_id = 0
        STrack.reset_id_counter()

    def stats(self) -> dict:
        return {
            "frame": self.frame_id,
            "tracked": len(self.tracked),
            "confirmed": sum(1 for t in self.tracked if t.is_confirmed()),
            "lost": len(self.lost),
        }
