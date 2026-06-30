# -*- coding: utf-8 -*-
"""三级级联注视估计: L0 头姿预过滤 + L1 时间降频 + L2 L2CS-Net ONNX。

L0: 从 SCRFD 5-point landmarks 几何估计 head yaw/pitch, ~0.05ms/face
L1: NOT_LOOKING tracks 每 N 帧检一次; LOOKING/新 track 每帧跑
L2: L2CS-Net MobileNetV2 ONNX (448×448), ~10-15ms/face on macOS Intel CPU

铁律: 纯感知,不写 st.state、不调 head_control。
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import cv2

_log = logging.getLogger(__name__)


@dataclass
class GazeResult:
    track_id: int
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    gaze_yaw: float = 0.0
    gaze_pitch: float = 0.0
    mutual_gaze: bool = False
    l2_ran: bool = False


@dataclass
class _TrackGazeState:
    last_result: str = "UNKNOWN"
    frames_since_check: int = 0
    gaze_yaw: float = 0.0
    gaze_pitch: float = 0.0


class HeadPoseFilter:
    """L0: 5-point landmarks → 几何头姿估计 + 阈值过滤。"""

    def __init__(self, yaw_thresh: float = 45.0, pitch_thresh: float = 35.0):
        self._yaw_thresh = yaw_thresh
        self._pitch_thresh = pitch_thresh

    def estimate(self, kps5: np.ndarray) -> tuple[float, float]:
        """从 SCRFD 5 点(le, re, nose, lm, rm)几何估计 (yaw_deg, pitch_deg)。"""
        le, re, nose = kps5[0], kps5[1], kps5[2]
        eye_center = (le + re) * 0.5
        inter_eye = np.linalg.norm(re - le)
        if inter_eye < 1e-6:
            return 0.0, 0.0
        yaw = float(np.degrees(np.arctan2(nose[0] - eye_center[0], inter_eye)))
        pitch = float(np.degrees(np.arctan2(nose[1] - eye_center[1], inter_eye)))
        return yaw, pitch

    def is_candidate(self, yaw_deg: float, pitch_deg: float) -> bool:
        return abs(yaw_deg) <= self._yaw_thresh and abs(pitch_deg) <= self._pitch_thresh


class GazeEstimator:
    """L2: L2CS-Net ONNX 推理。"""

    def __init__(self, model_path: str, input_size: int = 448,
                 num_bins: int = 90, bin_width: float = 4.0, offset: float = 180.0,
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225)):
        self._input_size = input_size
        self._num_bins = num_bins
        self._idx = np.arange(num_bins, dtype=np.float32) * bin_width - offset
        self._mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.available = False
        self._session = None
        self._input_name = ""
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self.available = True
            _log.info("L2CS-Net ONNX loaded: %s", model_path)
        except Exception as e:
            _log.warning("L2CS-Net ONNX not available: %s", e)

    def _preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_rgb, (self._input_size, self._input_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self._mean) / self._std
        return img[np.newaxis]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def predict(self, face_rgb: np.ndarray) -> tuple[float, float]:
        blob = self._preprocess(face_rgb)
        yaw_bins, pitch_bins = self._session.run(None, {self._input_name: blob})
        yaw_deg = float(self._softmax(yaw_bins[0]) @ self._idx)
        pitch_deg = float(self._softmax(pitch_bins[0]) @ self._idx)
        return yaw_deg, pitch_deg


def _crop_face(full_rgb: np.ndarray, bbox_xyxy: np.ndarray,
               decimate: int, margin: float = 0.15) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox_xyxy
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin * decimate), int(h * margin * decimate)
    fh, fw = full_rgb.shape[:2]
    fx1 = max(0, int(x1 * decimate) - mx)
    fy1 = max(0, int(y1 * decimate) - my)
    fx2 = min(fw, int(x2 * decimate) + mx)
    fy2 = min(fh, int(y2 * decimate) + my)
    if fx2 - fx1 < 10 or fy2 - fy1 < 10:
        return None
    return full_rgb[fy1:fy2, fx1:fx2]


class GazeModule:
    """三级级联: L0 头姿 + L1 降频 + L2 ONNX。每帧对每个 track 调 update()。"""

    def __init__(self, model_path: str,
                 head_yaw_thresh: float = 45.0, head_pitch_thresh: float = 35.0,
                 not_looking_interval: int = 5,
                 mutual_yaw_thresh: float = 12.0, mutual_pitch_thresh: float = 15.0,
                 min_face_px: int = 40,
                 input_size: int = 448, num_bins: int = 90,
                 bin_width: float = 4.0, offset: float = 180.0,
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225)):
        self._head_filter = HeadPoseFilter(head_yaw_thresh, head_pitch_thresh)
        self._estimator = GazeEstimator(model_path, input_size, num_bins,
                                        bin_width, offset, mean, std)
        self._mutual_yaw = mutual_yaw_thresh
        self._mutual_pitch = mutual_pitch_thresh
        self._not_looking_interval = not_looking_interval
        self._min_face_px = min_face_px
        self._states: dict[int, _TrackGazeState] = {}

    @property
    def available(self) -> bool:
        return self._estimator.available

    def update(self, track_id: int, landmarks_5x2: np.ndarray,
               full_rgb: Optional[np.ndarray], bbox_xyxy: np.ndarray,
               decimate: int) -> GazeResult:
        st = self._states.get(track_id)
        if st is None:
            st = _TrackGazeState()
            self._states[track_id] = st

        head_yaw, head_pitch = self._head_filter.estimate(landmarks_5x2)
        res = GazeResult(track_id=track_id, head_yaw=head_yaw, head_pitch=head_pitch)

        if not self._head_filter.is_candidate(head_yaw, head_pitch):
            st.last_result = "NOT_LOOKING"
            st.frames_since_check = 0
            return res

        needs_l2 = self._needs_l2(st)
        if not needs_l2:
            res.gaze_yaw = st.gaze_yaw
            res.gaze_pitch = st.gaze_pitch
            res.mutual_gaze = (abs(st.gaze_yaw) < self._mutual_yaw
                               and abs(st.gaze_pitch) < self._mutual_pitch)
            return res

        if not self._estimator.available or full_rgb is None:
            return res

        face_w = (bbox_xyxy[2] - bbox_xyxy[0])
        if face_w < self._min_face_px:
            return res

        crop = _crop_face(full_rgb, bbox_xyxy, decimate)
        if crop is None:
            return res

        gaze_yaw, gaze_pitch = self._estimator.predict(crop)
        res.gaze_yaw = gaze_yaw
        res.gaze_pitch = gaze_pitch
        res.mutual_gaze = (abs(gaze_yaw) < self._mutual_yaw
                           and abs(gaze_pitch) < self._mutual_pitch)
        res.l2_ran = True

        st.gaze_yaw = gaze_yaw
        st.gaze_pitch = gaze_pitch
        st.last_result = "LOOKING" if res.mutual_gaze else "NOT_LOOKING"
        st.frames_since_check = 0
        return res

    def _needs_l2(self, st: _TrackGazeState) -> bool:
        if st.last_result == "UNKNOWN":
            return True
        if st.last_result == "LOOKING":
            return True
        st.frames_since_check += 1
        if st.frames_since_check >= self._not_looking_interval:
            st.frames_since_check = 0
            return True
        return False

    def gc(self, alive_ids: set[int]) -> None:
        for tid in [k for k in self._states if k not in alive_ids]:
            del self._states[tid]
