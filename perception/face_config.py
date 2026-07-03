# -*- coding: utf-8 -*-
"""人脸检测 + 跟踪 + ReID 子系统配置(完全参考 face-tracker-demo/config.py)。

算法选型:
  - 检测:   InsightFace SCRFD(buffalo_sc 的 det_500m)
  - 识别:   ArcFace(buffalo_sc 的 w600k_mbf,512-d)
  - 跟踪:   ByteTrack 式(Kalman + Hungarian + BYTE 两阶段)
  - 身份:   余弦距离 gallery + 三区间 + 质量门

阈值直接复用参考工程(同一套 buffalo 模型 → embedding 距离分布一致,实测合理)。
距离语义:cosine distance(0=同一人,2=相反),全程用距离,不用相似度。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectionConfig:
    """SCRFD 人脸检测(InsightFace)。"""
    model_pack: str = "buffalo_sc"      # det_500m(SCRFD) + w600k_mbf(ArcFace MobileFaceNet)
    det_size: tuple[int, int] = (640, 640)
    det_thresh: float = 0.5             # 最低检测置信
    max_faces: int = 10                 # 每帧上限


@dataclass
class TrackingConfig:
    """ByteTrack 式多人脸跟踪。"""
    high_thresh: float = 0.6            # BYTE 一阶段置信门
    low_thresh: float = 0.1            # BYTE 二阶段下界
    iou_threshold: float = 0.15         # IoU 匹配门(放宽:低fps帧间位移大,IoU低但仍是同一人)
    embedding_threshold: float = 0.45  # lost track 用 embedding 找回的余弦距离门
    max_age: int = 60                  # lost track 删除前的帧数(低fps=8时60帧≈7.5s)
    min_hits: int = 2                  # tentative → confirmed 所需命中帧(快确认,减少churn)
    embedding_weight: float = 0.3      # 一阶段融合 cost 里 embedding 的权重


@dataclass
class RecognitionConfig:
    """ArcFace embedding 设置。"""
    embed_dim: int = 512
    embed_interval: int = 3            # 每个稳定 track 每 N 帧提一次 embedding
    similarity_metric: str = "cosine"


@dataclass
class IdentityConfig:
    """持久化身份 gallery。"""
    match_threshold: float = 0.65      # cosine 距离 ≤ 此 → 同一人(known)
    unknown_threshold: float = 0.80    # cosine 距离 ≥ 此 → 肯定陌生(unknown)
    # match~unknown 之间 = unsure 区:继续跟踪,不提交身份
    max_gallery_per_id: int = 10       # 每个身份最多存的 embedding 数
    min_quality: float = 0.40          # FIQA 代理入库门
    min_confirm_frames: int = 5        # 稳定 unknown 帧数 → 自动注册 provisional
    gallery_path: Path = Path("data/gallery.json")
    distance_log_max: int = 5000       # distance_log 上限(deque)


@dataclass
class SmoothingConfig:
    """per-track embedding 时序平滑(参考 clustering.EmbeddingSmoother)。"""
    base_alpha: float = 0.6            # 新帧 EMA 基础权重
    quality_boost: float = 0.3         # 高质量帧的额外 alpha 加成
    outlier_threshold: float = 0.55    # 余弦距离超此 → 拒绝该帧(错脸/检测错;机器人转头帧略放宽)
    min_samples_for_gating: int = 3    # 攒够 N 个 embedding 才开始离群门
    momentum_decay: float = 0.98       # 成熟 track: alpha *= decay^age


@dataclass
class ClusteringConfig:
    """gallery 聚类(参考 clustering.GalleryClustering)。"""
    max_modes: int = 4                 # 每个身份最多聚成几个 pose mode
    min_mode_distance: float = 0.25    # mode 质心间最小余弦距离
    merge_threshold: float = 0.55      # 跨身份合并门(仅 provisional)
    compaction_min_quality: float = 0.3
    compaction_max_per_mode: int = 3


@dataclass
class FaceSystemConfig:
    """顶层配置,聚合各子配置。"""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
