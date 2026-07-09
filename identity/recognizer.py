# -*- coding: utf-8 -*-
"""ArcFace embedding 提取 + 人脸对齐工具函数。

FaceDB / IdentityRecognizer 已废弃，身份管理统一使用
identity.identity_store.IdentityStore。
"""

import os

import cv2
import numpy as np

# ── 模型路径 ──
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YUNET_PATH = os.path.join(_REPO, "models", "face_detection_yunet_2023mar.onnx")
_ARCFACE_PATH = os.path.join(_REPO, "models", "w600k_mbf.onnx")

# ── 匹配阈值 ──
COSINE_THRESHOLD = 0.35
MAX_EMBEDDINGS_PER_PERSON = 10
IDENTITY_COOLDOWN_S = 2.0
MIN_FACE_PX = 60
NEW_PERSON_CONFIRM_FRAMES = 3

# ── arcface 输入标准 ──
_ARC_SIZE = 112
_ARC_MEAN = 127.5
_ARC_STD = 127.5

# arcface 标准目标关键点(112x112 图上的 5 点坐标)
_ARC_REF_POINTS = np.array([
    [38.2946, 51.6963],   # 右眼
    [73.5318, 51.5014],   # 左眼
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 右嘴角
    [70.7299, 92.2041],   # 左嘴角
], dtype=np.float32)


def _align_face(rgb: np.ndarray, kps: list[tuple[float, float]]) -> np.ndarray:
    """用 5 关键点仿射对齐到 112×112 arcface 标准。"""
    src = np.array(kps, dtype=np.float32)
    M = cv2.estimateAffinePartial2D(src, _ARC_REF_POINTS)[0]
    if M is None:
        cx = int(np.mean([k[0] for k in kps]))
        cy = int(np.mean([k[1] for k in kps]))
        half = 56
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        crop = rgb[y0:y0 + 112, x0:x0 + 112]
        if crop.shape[0] != 112 or crop.shape[1] != 112:
            crop = cv2.resize(crop, (112, 112))
        return crop
    return cv2.warpAffine(rgb, M, (112, 112))


def _crop_face(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """无关键点时，从 bbox 裁剪并 resize 到 112×112。"""
    x, y, w, h = box
    margin = int(max(w, h) * 0.15)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(rgb.shape[1], x + w + margin)
    y1 = min(rgb.shape[0], y + h + margin)
    crop = rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (112, 112))


class ArcFaceONNX:
    """arcface embedding 提取器(onnxruntime)。"""

    def __init__(self, model_path: str = _ARCFACE_PATH):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, face_112: np.ndarray) -> np.ndarray:
        """输入 112×112 RGB → 输出 L2 归一化的 512d embedding。"""
        img = face_112.astype(np.float32)
        img = (img - _ARC_MEAN) / _ARC_STD
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # (1,3,112,112)
        out = self.session.run(None, {self.input_name: img})[0][0]
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out


