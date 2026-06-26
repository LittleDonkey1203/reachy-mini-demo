# -*- coding: utf-8 -*-
"""持久化身份 gallery + 开集识别(完全参考 face-tracker-demo/identity_store.py)。

- 每个身份多 embedding(覆盖姿态/表情)
- 质量门入库(FIQA 代理)
- 三区间开集匹配(known / unsure / unknown)
- provisional(自动 Unknown-N)vs confirmed(用户命名)
- distance_log(bounded deque)供阈值校准
- JSON 持久化,跨会话身份

距离语义:cosine distance(1 - cos_sim),0=同一人。
"""
from __future__ import annotations

import json
import time
import logging
import collections
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from perception.face_config import IdentityConfig

logger = logging.getLogger(__name__)


@dataclass
class Identity:
    """一个已知或临时(provisional)身份。"""
    identity_id: str
    name: str                                  # 显示名(provisional 为 "Unknown-N")
    embeddings: list[np.ndarray] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)
    created_at: float = 0.0
    last_seen: float = 0.0
    total_sightings: int = 0
    is_confirmed: bool = False                 # True = 用户命名;False = 临时

    @property
    def centroid(self) -> Optional[np.ndarray]:
        if not self.embeddings:
            return None
        c = np.mean(self.embeddings, axis=0)
        return c / (np.linalg.norm(c) + 1e-8)

    def add_embedding(self, emb: np.ndarray, quality: float, max_n: int = 10):
        """加一个 embedding,只保留质量最高的 N 个。"""
        self.embeddings.append(emb)
        self.quality_scores.append(quality)
        self.total_sightings += 1
        self.last_seen = time.time()
        if len(self.embeddings) > max_n:
            worst_idx = int(np.argmin(self.quality_scores))
            self.embeddings.pop(worst_idx)
            self.quality_scores.pop(worst_idx)


@dataclass
class MatchResult:
    identity_id: Optional[str]
    identity_name: Optional[str]
    distance: float
    confidence: float                # 1 - distance
    is_known: bool
    is_provisional: bool = False
    zone: str = "unknown"            # known | unsure | unknown


class IdentityStore:
    """持久化身份 gallery,开集识别。"""

    def __init__(self, config: IdentityConfig | None = None):
        self.cfg = config or IdentityConfig()
        self.identities: dict[str, Identity] = {}
        self._next_unknown_id: int = 0
        self.distance_log: "collections.deque[dict]" = collections.deque(
            maxlen=self.cfg.distance_log_max)
        self._log_enabled: bool = True

    def _log_distance(self, distance: float, zone: str, identity_name: Optional[str]):
        if self._log_enabled:
            self.distance_log.append({
                "distance": round(float(distance), 6),
                "zone": zone,
                "identity": identity_name or "unknown",
                "timestamp": time.time(),
            })

    # ── 核心匹配:三区间 ──────────────────────────────────
    def match(self, embedding: np.ndarray, quality: float = 1.0) -> MatchResult:
        if not self.identities:
            return MatchResult(None, None, 999.0, 0.0, False, zone="unknown")

        best_id = None
        best_name = None
        best_dist = float("inf")
        best_provisional = False
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        for ident in self.identities.values():
            if not ident.embeddings:
                continue
            distances = []
            for stored in ident.embeddings:
                sn = stored / (np.linalg.norm(stored) + 1e-8)
                distances.append(1.0 - float(np.dot(emb_norm, sn)))
            centroid = ident.centroid
            if centroid is not None:
                distances.append(1.0 - float(np.dot(emb_norm, centroid)))
            min_dist = min(distances)
            if min_dist < best_dist:
                best_dist = min_dist
                best_id = ident.identity_id
                best_name = ident.name
                best_provisional = not ident.is_confirmed

        if best_dist <= self.cfg.match_threshold:
            zone = "known"
        elif best_dist < self.cfg.unknown_threshold:
            zone = "unsure"
        else:
            zone = "unknown"

        self._log_distance(best_dist, zone, best_name)
        return MatchResult(
            identity_id=best_id if zone == "known" else None,
            identity_name=best_name if zone == "known" else None,
            distance=best_dist,
            confidence=max(0.0, 1.0 - best_dist),
            is_known=(zone == "known"),
            is_provisional=best_provisional if zone == "known" else False,
            zone=zone,
        )

    def match_and_update(self, embedding: np.ndarray, quality: float = 1.0) -> MatchResult:
        """匹配;命中则按质量门更新该身份 gallery。"""
        result = self.match(embedding, quality)
        if result.is_known and result.identity_id:
            ident = self.identities[result.identity_id]
            if quality >= self.cfg.min_quality:
                ident.add_embedding(embedding, quality, self.cfg.max_gallery_per_id)
            else:
                ident.total_sightings += 1
                ident.last_seen = time.time()
        return result

    # ── 注册 ──────────────────────────────────────────────
    def register_identity(self, name: str, embeddings: list[np.ndarray],
                          qualities: list[float] | None = None,
                          confirmed: bool = True) -> str:
        identity_id = f"id_{int(time.time()*1000)}_{name.lower().replace(' ', '_')}"
        if qualities is None:
            qualities = [1.0] * len(embeddings)
        ident = Identity(
            identity_id=identity_id, name=name,
            embeddings=list(embeddings), quality_scores=list(qualities),
            created_at=time.time(), last_seen=time.time(),
            total_sightings=len(embeddings), is_confirmed=confirmed,
        )
        self.identities[identity_id] = ident
        logger.info(f"Registered identity: {name} (id={identity_id}, {len(embeddings)} embeddings)")
        return identity_id

    def register_unknown(self, embedding: np.ndarray, quality: float = 1.0) -> MatchResult:
        """陌生脸:先与已有 provisional 比对(质心 ≤ match_threshold 则合并),否则建新 Unknown-N。"""
        best_prov_id = None
        best_prov_dist = float("inf")
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        for ident in self.identities.values():
            if ident.is_confirmed:
                continue
            centroid = ident.centroid
            if centroid is None:
                continue
            dist = 1.0 - float(np.dot(emb_norm, centroid))
            if dist < best_prov_dist:
                best_prov_dist = dist
                best_prov_id = ident.identity_id

        if best_prov_id and best_prov_dist <= self.cfg.match_threshold:
            ident = self.identities[best_prov_id]
            if quality >= self.cfg.min_quality:
                ident.add_embedding(embedding, quality, self.cfg.max_gallery_per_id)
            return MatchResult(best_prov_id, ident.name, best_prov_dist,
                               max(0.0, 1.0 - best_prov_dist), True, is_provisional=True, zone="known")

        self._next_unknown_id += 1
        prov_name = f"Unknown-{self._next_unknown_id}"
        prov_id = self.register_identity(prov_name, [embedding], [quality], confirmed=False)
        logger.info(f"Created provisional identity: {prov_name}")
        return MatchResult(prov_id, prov_name, 0.0, 1.0, True, is_provisional=True, zone="known")

    def confirm_identity(self, identity_id: str, name: str) -> bool:
        """provisional → confirmed(用户命名)。"""
        if identity_id not in self.identities:
            return False
        ident = self.identities[identity_id]
        ident.name = name
        ident.is_confirmed = True
        logger.info(f"Confirmed identity: {identity_id} → {name}")
        return True

    def remove_identity(self, identity_id: str) -> bool:
        if identity_id in self.identities:
            del self.identities[identity_id]
            return True
        return False

    # ── 持久化 ────────────────────────────────────────────
    def save(self, path: Path | None = None):
        path = Path(path or self.cfg.gallery_path)
        data = {}
        for ident_id, ident in self.identities.items():
            data[ident_id] = {
                "identity_id": ident.identity_id,
                "name": ident.name,
                "embeddings": [np.asarray(e).tolist() for e in ident.embeddings],
                "quality_scores": [float(q) for q in ident.quality_scores],
                "created_at": ident.created_at,
                "last_seen": ident.last_seen,
                "total_sightings": ident.total_sightings,
                "is_confirmed": ident.is_confirmed,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False,
                      default=lambda x: float(x) if hasattr(x, "item") else str(x))
        logger.info(f"Gallery saved: {len(data)} identities → {path}")

    def load(self, path: Path | None = None) -> int:
        path = Path(path or self.cfg.gallery_path)
        if not path.exists():
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupted gallery {path}: {e}, starting fresh.")
            return 0
        self.identities.clear()
        for ident_id, d in data.items():
            self.identities[ident_id] = Identity(
                identity_id=d["identity_id"], name=d["name"],
                embeddings=[np.array(e, dtype=np.float32) for e in d["embeddings"]],
                quality_scores=d.get("quality_scores", []),
                created_at=d.get("created_at", 0), last_seen=d.get("last_seen", 0),
                total_sightings=d.get("total_sightings", 0),
                is_confirmed=d.get("is_confirmed", False),
            )
        max_unk = 0
        for ident in self.identities.values():
            if ident.name.startswith("Unknown-"):
                try:
                    max_unk = max(max_unk, int(ident.name.split("-")[1]))
                except (ValueError, IndexError):
                    pass
        self._next_unknown_id = max_unk
        logger.info(f"Gallery loaded: {len(self.identities)} identities from {path}")
        return len(self.identities)

    # ── 查询 / 统计 ───────────────────────────────────────
    def list_identities(self) -> list[dict]:
        return sorted([{
            "id": i.identity_id, "name": i.name, "confirmed": i.is_confirmed,
            "num_embeddings": len(i.embeddings), "total_sightings": i.total_sightings,
            "last_seen": i.last_seen,
        } for i in self.identities.values()], key=lambda x: x["last_seen"], reverse=True)

    def get_stats(self) -> dict:
        confirmed = sum(1 for i in self.identities.values() if i.is_confirmed)
        return {
            "total": len(self.identities), "confirmed": confirmed,
            "provisional": len(self.identities) - confirmed,
            "total_embeddings": sum(len(i.embeddings) for i in self.identities.values()),
        }

    # ── distance log(阈值校准)────────────────────────────
    def export_distance_log(self, path: Path | str = "data/distance_log.json") -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(self.distance_log), f, indent=2,
                      default=lambda x: float(x) if hasattr(x, "item") else str(x))
        return len(self.distance_log)

    def print_distance_summary(self):
        if not self.distance_log:
            print("  No distance log entries yet.")
            return
        zones = {"known": [], "unsure": [], "unknown": []}
        for e in self.distance_log:
            zones[e["zone"]].append(e["distance"])
        print(f"\n  Distance Log Summary ({len(self.distance_log)} matches)")
        for z in ["known", "unsure", "unknown"]:
            d = np.array(zones[z]) if zones[z] else None
            if d is not None:
                print(f"  {z:<8s} N={len(d):>4d} min={d.min():.4f} mean={d.mean():.4f} max={d.max():.4f}")
            else:
                print(f"  {z:<8s} N=   0")
        print(f"  thresholds: match={self.cfg.match_threshold:.2f} unknown={self.cfg.unknown_threshold:.2f}")
