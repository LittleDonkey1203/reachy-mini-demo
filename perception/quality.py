# -*- coding: utf-8 -*-
"""人脸质量评估 FIQA 代理(完全参考 face-tracker-demo/pipeline.compute_quality_proxy)。

无需额外模型,用 bbox 大小 + 检测置信 + 关键点对称/正脸度 估一个 [0,1] 质量分,
作为 embedding 入库/注册的门(min_quality)。生产可换 CR-FIQA / SER-FIQ。
"""
from __future__ import annotations

import numpy as np


def compute_quality_proxy(
    bbox: np.ndarray,
    confidence: float,
    landmarks: np.ndarray,
    frame_shape: tuple,
) -> float:
    """返回 [0,1] 质量分:0.3·置信 + 0.25·大小 + 0.2·对称 + 0.25·正脸。

    bbox: [x1,y1,x2,y2];landmarks: (5,2)[右眼,左眼,鼻,右嘴,左嘴];frame_shape: (H,W,...)。
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    frame_h, frame_w = frame_shape[:2]

    # 大小分:脸宽占帧宽 5~30% 给满分
    size_ratio = w / max(frame_w, 1)
    size_score = min(1.0, size_ratio / 0.15)

    conf_score = float(confidence)

    # 对称:两眼间距应 ≈ 脸宽 35%
    if landmarks is not None and len(landmarks) >= 2:
        eye_dist = np.linalg.norm(np.asarray(landmarks[1]) - np.asarray(landmarks[0]))
        sym_ratio = eye_dist / max(w, 1)
        sym_score = 1.0 - abs(sym_ratio - 0.35) * 3
        sym_score = max(0.0, min(1.0, sym_score))
    else:
        sym_score = 0.5

    # 正脸:鼻尖应大致居于两眼中线
    if landmarks is not None and len(landmarks) >= 3:
        eye_center_x = (landmarks[0][0] + landmarks[1][0]) / 2
        nose_x = landmarks[2][0]
        deviation = abs(nose_x - eye_center_x) / max(w, 1)
        front_score = max(0.0, 1.0 - deviation * 5)
    else:
        front_score = 0.5

    return float(0.3 * conf_score + 0.25 * size_score + 0.2 * sym_score + 0.25 * front_score)
