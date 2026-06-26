# -*- coding: utf-8 -*-
"""Embedding 平滑 + gallery 聚类(完全参考 face-tracker-demo/clustering.py)。

1. 时序平滑(track 级):质量加权 EMA + 离群门 + 年龄自适应 momentum
2. gallery 聚类(身份级):pose-mode 聚类 / 代表点匹配 / 跨身份合并 / 压缩

EmbeddingSmoother 供 tracker 之外的可选平滑;ByteTracker.STrack 已内置 EMA,
此处 Smoother 可用于 d01 懒提特征时做额外离群门(拒错脸/突变),或离线分析。
GalleryClustering 作用于 identity_store.Identity 字典。
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from perception.face_config import SmoothingConfig, ClusteringConfig

logger = logging.getLogger(__name__)


# ═══ 1. 时序平滑 ═══════════════════════════════════════════

class EmbeddingSmoother:
    """质量加权 EMA + 离群门,做 per-track embedding 平滑。"""

    def __init__(self, config: SmoothingConfig | None = None):
        self.cfg = config or SmoothingConfig()
        self.smooth: Optional[np.ndarray] = None
        self.sample_count: int = 0
        self.rejected_count: int = 0
        self._recent: list[np.ndarray] = []

    def update(self, embedding: np.ndarray, quality: float = 1.0) -> Optional[np.ndarray]:
        if embedding is None:
            return self.smooth
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)

        if self.smooth is None:
            self.smooth = emb.copy()
            self.sample_count = 1
            self._recent.append(emb)
            return self.smooth

        # 离群门(攒够样本后)
        if self.sample_count >= self.cfg.min_samples_for_gating:
            cos_dist = 1.0 - float(np.dot(emb, self.smooth))
            if cos_dist > self.cfg.outlier_threshold:
                self.rejected_count += 1
                return self.smooth

        alpha = self.cfg.base_alpha + self.cfg.quality_boost * quality
        alpha = min(alpha, 0.95)
        alpha *= self.cfg.momentum_decay ** min(self.sample_count, 100)

        self.smooth = alpha * emb + (1.0 - alpha) * self.smooth
        self.smooth /= np.linalg.norm(self.smooth) + 1e-8
        self.sample_count += 1
        self._recent.append(emb)
        if len(self._recent) > 20:
            self._recent = self._recent[-20:]
        return self.smooth

    @property
    def embedding_variance(self) -> float:
        if len(self._recent) < 3:
            return 1.0
        arr = np.array(self._recent)
        centroid = arr.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        return float(np.std([1.0 - np.dot(e, centroid) for e in arr]))

    @property
    def stats(self) -> dict:
        return {"samples": self.sample_count, "rejected": self.rejected_count,
                "variance": round(self.embedding_variance, 6)}


# ═══ 2. gallery 聚类 ═══════════════════════════════════════

@dataclass
class EmbeddingMode:
    """一个身份 gallery 内的聚类(姿态/表情 mode)。"""
    centroid: np.ndarray
    members: list[np.ndarray]
    qualities: list[float]
    count: int = 0

    def add(self, emb: np.ndarray, quality: float, max_members: int = 3):
        self.members.append(emb)
        self.qualities.append(quality)
        self.count += 1
        if len(self.members) > max_members:
            worst = int(np.argmin(self.qualities))
            self.members.pop(worst)
            self.qualities.pop(worst)
        self._update_centroid()

    def _update_centroid(self):
        if self.members:
            c = np.mean(self.members, axis=0)
            self.centroid = c / (np.linalg.norm(c) + 1e-8)


class GalleryClustering:
    """身份内 embedding 聚类 + 跨身份合并 + 压缩。"""

    def __init__(self, config: ClusteringConfig | None = None):
        self.cfg = config or ClusteringConfig()

    # ── 身份内聚类 ──
    def cluster_identity(self, embeddings: list[np.ndarray],
                         qualities: list[float] | None = None) -> list[EmbeddingMode]:
        """增量贪心聚类成 1~max_modes 个 mode。"""
        if not embeddings:
            return []
        if qualities is None:
            qualities = [1.0] * len(embeddings)
        modes: list[EmbeddingMode] = []
        for emb, q in zip(embeddings, qualities):
            en = emb / (np.linalg.norm(emb) + 1e-8)
            if not modes:
                modes.append(EmbeddingMode(en.copy(), [en], [q], 1))
                continue
            dists = [1.0 - np.dot(en, m.centroid) for m in modes]
            ni = int(np.argmin(dists))
            if dists[ni] < self.cfg.min_mode_distance:
                modes[ni].add(en, q, self.cfg.compaction_max_per_mode)
            elif len(modes) < self.cfg.max_modes:
                modes.append(EmbeddingMode(en.copy(), [en], [q], 1))
            else:
                modes[ni].add(en, q, self.cfg.compaction_max_per_mode)
        return modes

    def match_against_modes(self, query: np.ndarray,
                            modes: list[EmbeddingMode]) -> tuple[float, int]:
        if not modes:
            return 999.0, -1
        qn = query / (np.linalg.norm(query) + 1e-8)
        dists = [1.0 - np.dot(qn, m.centroid) for m in modes]
        bi = int(np.argmin(dists))
        return dists[bi], bi

    def add_to_modes(self, embedding: np.ndarray, quality: float,
                     modes: list[EmbeddingMode]) -> list[EmbeddingMode]:
        en = embedding / (np.linalg.norm(embedding) + 1e-8)
        if not modes:
            modes.append(EmbeddingMode(en.copy(), [en], [quality], 1))
            return modes
        dists = [1.0 - np.dot(en, m.centroid) for m in modes]
        ni = int(np.argmin(dists))
        if dists[ni] < self.cfg.min_mode_distance:
            modes[ni].add(en, quality, self.cfg.compaction_max_per_mode)
        elif len(modes) < self.cfg.max_modes:
            modes.append(EmbeddingMode(en.copy(), [en], [quality], 1))
        else:
            modes[ni].add(en, quality, self.cfg.compaction_max_per_mode)
        return modes

    # ── 跨身份合并 ──
    def find_mergeable_pairs(self, identities: dict) -> list[tuple[str, str, float]]:
        """找应合并的 provisional 对(同一人不同时段)。返回 (id_a,id_b,dist) 按距离排序。"""
        provisionals = [(i, ident) for i, ident in identities.items()
                        if not ident.is_confirmed and ident.embeddings]
        pairs = []
        for i in range(len(provisionals)):
            ida, ia = provisionals[i]
            ca = ia.centroid
            if ca is None:
                continue
            for j in range(i + 1, len(provisionals)):
                idb, ib = provisionals[j]
                cb = ib.centroid
                if cb is None:
                    continue
                dist = 1.0 - float(np.dot(ca, cb))
                if dist < self.cfg.merge_threshold:
                    pairs.append((ida, idb, dist))
        return sorted(pairs, key=lambda x: x[2])

    def merge_identities(self, identities: dict, id_keep: str, id_remove: str) -> bool:
        """把 id_remove 合并进 id_keep。"""
        if id_keep not in identities or id_remove not in identities:
            return False
        keeper = identities[id_keep]
        removed = identities[id_remove]
        for emb, q in zip(removed.embeddings, removed.quality_scores):
            keeper.add_embedding(emb, q, max_n=15)
        keeper.total_sightings += removed.total_sightings
        keeper.last_seen = max(keeper.last_seen, removed.last_seen)
        del identities[id_remove]
        logger.info(f"Merged identity '{removed.name}' into '{keeper.name}'")
        return True

    # ── gallery 压缩 ──
    def compact_gallery(self, identities: dict) -> dict:
        """按质量过滤 + 重聚类,只留 mode 代表点。"""
        stats = {"identities_processed": 0, "embeddings_before": 0,
                 "embeddings_after": 0, "modes_total": 0}
        for ident in identities.values():
            if not ident.embeddings:
                continue
            stats["identities_processed"] += 1
            stats["embeddings_before"] += len(ident.embeddings)
            filtered = [(e, q) for e, q in zip(ident.embeddings, ident.quality_scores)
                        if q >= self.cfg.compaction_min_quality]
            if not filtered:
                bi = int(np.argmax(ident.quality_scores)) if ident.quality_scores else 0
                filtered = [(ident.embeddings[bi],
                             ident.quality_scores[bi] if ident.quality_scores else 1.0)]
            embs = [e for e, _ in filtered]
            quals = [q for _, q in filtered]
            modes = self.cluster_identity(embs, quals)
            stats["modes_total"] += len(modes)
            new_e, new_q = [], []
            for mode in modes:
                for m, q in zip(mode.members, mode.qualities):
                    new_e.append(m)
                    new_q.append(q)
            ident.embeddings = new_e
            ident.quality_scores = new_q
            stats["embeddings_after"] += len(new_e)
        logger.info(f"Compaction: {stats['embeddings_before']} → {stats['embeddings_after']} "
                    f"embeddings, {stats['modes_total']} modes")
        return stats
