# -*- coding: utf-8 -*-
"""三级级联注视估计: L0 头姿预过滤 + L1 时间降频 + L2 L2CS-Net ONNX。

L0: 从 SCRFD 5-point landmarks 几何估计 head yaw/pitch, ~0.05ms/face
L1: NOT_LOOKING tracks 每 N 帧检一次; LOOKING/新 track 每帧跑
L2: L2CS-Net MobileNetV2 ONNX (448×448), ~10-15ms/face on macOS Intel CPU

铁律: 纯感知,不写 st.state、不调 head_control。
"""
from __future__ import annotations

import json
import logging
import os
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
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
    # EMA 平滑后的 gaze(供 mutual_gaze 判定)
    smooth_yaw: float = 0.0
    smooth_pitch: float = 0.0
    # 连续帧迟滞计数器
    looking_streak: int = 0     # 连续"原始 mutual" 帧数
    not_looking_streak: int = 0  # 连续"原始 非 mutual" 帧数
    confirmed_mutual: bool = False  # 迟滞后的稳定 mutual_gaze
    l2_count: int = 0  # L2 推理次数(warm-start: 前几帧用高 alpha)


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


# ── 5 点仿射对齐(复用 ArcFace 标准模板,缩放到 L2CS-Net 输入尺寸) ──

# arcface 标准目标关键点(112×112 图上的 5 点坐标)
_ARC_REF_112 = np.array([
    [38.2946, 51.6963],   # 右眼
    [73.5318, 51.5014],   # 左眼
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 右嘴角
    [70.7299, 92.2041],   # 左嘴角
], dtype=np.float32)


def _align_face(full_rgb: np.ndarray, kps5: np.ndarray,
                decimate: int, output_size: int = 448) -> Optional[np.ndarray]:
    """用 5 关键点仿射对齐到 output_size×output_size(L2CS-Net 输入)。

    kps5 是降采样坐标系,需 ×decimate 还原到 full_rgb 坐标系。
    """
    src = kps5.astype(np.float32) * decimate          # 还原全分辨率坐标
    ref = _ARC_REF_112 * (output_size / 112.0)        # 模板缩放到目标尺寸
    M = cv2.estimateAffinePartial2D(src, ref)[0]
    if M is None:
        return _crop_face_fallback(full_rgb, kps5, decimate)
    return cv2.warpAffine(full_rgb, M, (output_size, output_size))


def _crop_face_fallback(full_rgb: np.ndarray, kps5: np.ndarray,
                        decimate: int, margin: float = 0.25) -> Optional[np.ndarray]:
    """对齐失败时的 fallback: 以关键点中心裁正方形 crop。"""
    pts = kps5.astype(np.float32) * decimate
    cx, cy = pts.mean(axis=0)
    spread = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))  # 关键点跨度
    half = int(spread * (1.0 + margin))
    fh, fw = full_rgb.shape[:2]
    x0 = max(0, int(cx) - half)
    y0 = max(0, int(cy) - half)
    x1 = min(fw, int(cx) + half)
    y1 = min(fh, int(cy) + half)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return full_rgb[y0:y1, x0:x1]


class GazeModule:
    """三级级联: L0 头姿 + L1 降频 + L2 ONNX。每帧对每个 track 调 update()。"""

    def __init__(self, model_path: str,
                 head_yaw_thresh: float = 45.0, head_pitch_thresh: float = 35.0,
                 not_looking_interval: int = 5,
                 looking_interval: int = 3,
                 mutual_yaw_thresh: float = 12.0, mutual_pitch_thresh: float = 15.0,
                 gaze_dir_deadband: float = 8.0,
                 fov_x_deg: float = 65.0,
                 min_face_px: int = 40,
                 l2_ema_alpha: float = 0.35,
                 mutual_confirm_frames: int = 3,
                 mutual_drop_frames: int = 5,
                 input_size: int = 448, num_bins: int = 90,
                 bin_width: float = 4.0, offset: float = 180.0,
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225)):
        self._head_filter = HeadPoseFilter(head_yaw_thresh, head_pitch_thresh)
        self._estimator = GazeEstimator(model_path, input_size, num_bins,
                                        bin_width, offset, mean, std)
        self._mutual_yaw = mutual_yaw_thresh
        self._mutual_pitch = mutual_pitch_thresh
        self._gaze_dir_deadband = gaze_dir_deadband
        self._fov_x = fov_x_deg
        self._not_looking_interval = not_looking_interval
        self._looking_interval = looking_interval
        self._min_face_px = min_face_px
        self._ema_alpha = l2_ema_alpha
        self._confirm_frames = mutual_confirm_frames
        self._drop_frames = mutual_drop_frames
        self._states: dict[str, _TrackGazeState] = {}
        # 样本采集: GAZE_SAVE_SAMPLES=1 时保存 L2 输入 crop + 模型输出
        self._save_samples = os.environ.get("GAZE_SAVE_SAMPLES") == "1"
        self._sample_dir: Optional[Path] = None
        self._sample_seq = 0
        if self._save_samples:
            repo = Path(__file__).resolve().parent.parent
            self._sample_dir = repo / "data" / "gaze_samples"
            self._sample_dir.mkdir(parents=True, exist_ok=True)
            _log.info("📸 Gaze 样本采集已开启 → %s", self._sample_dir)

    @property
    def available(self) -> bool:
        return self._estimator.available

    def update(self, track_id: int, landmarks_5x2: np.ndarray,
               full_rgb: Optional[np.ndarray], bbox_xyxy: np.ndarray,
               decimate: int, identity_key: Optional[str] = None,
               frame_w: int = 0) -> GazeResult:
        key = identity_key or f"t{track_id}"
        st = self._states.get(key)
        if st is None:
            st = _TrackGazeState()
            self._states[key] = st

        head_yaw, head_pitch = self._head_filter.estimate(landmarks_5x2)
        res = GazeResult(track_id=track_id, head_yaw=head_yaw, head_pitch=head_pitch)

        if not self._head_filter.is_candidate(head_yaw, head_pitch):
            # 大侧脸:直接判定非注视,更新迟滞计数
            st.last_result = "NOT_LOOKING"
            st.frames_since_check = 0
            st.looking_streak = 0
            st.not_looking_streak += 1
            if st.not_looking_streak >= self._drop_frames:
                st.confirmed_mutual = False
            res.mutual_gaze = st.confirmed_mutual
            return res

        needs_l2 = self._needs_l2(st)
        if not needs_l2:
            # 复用上次平滑值
            res.gaze_yaw = st.smooth_yaw
            res.gaze_pitch = st.smooth_pitch
            res.mutual_gaze = st.confirmed_mutual
            return res

        if not self._estimator.available or full_rgb is None:
            res.mutual_gaze = st.confirmed_mutual
            return res

        face_w = (bbox_xyxy[2] - bbox_xyxy[0])
        if face_w < self._min_face_px:
            res.mutual_gaze = st.confirmed_mutual
            return res

        crop = _align_face(full_rgb, landmarks_5x2, decimate, self._estimator._input_size)
        if crop is None:
            res.mutual_gaze = st.confirmed_mutual
            return res

        gaze_yaw, gaze_pitch = self._estimator.predict(crop)

        # ── 样本采集 ──
        if self._save_samples and self._sample_dir is not None:
            self._sample_seq += 1
            sid = f"{self._sample_seq:06d}"
            cv2.imwrite(str(self._sample_dir / f"{sid}.jpg"),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            meta = {
                "id": sid, "ts": time.time(), "track_id": track_id,
                "head_yaw": round(head_yaw, 2), "head_pitch": round(head_pitch, 2),
                "gaze_yaw_raw": round(gaze_yaw, 2), "gaze_pitch_raw": round(gaze_pitch, 2),
                "smooth_yaw": round(st.smooth_yaw, 2), "smooth_pitch": round(st.smooth_pitch, 2),
                "bbox": [round(float(v), 1) for v in bbox_xyxy[:4]],
                "decimate": decimate, "frame_w": frame_w,
                "label": None,  # 待标注: "looking" / "not_looking"
            }
            with open(self._sample_dir / f"{sid}.json", "w") as f:
                json.dump(meta, f, ensure_ascii=False)

        # ── EMA 平滑 L2 输出 ──
        st.l2_count += 1
        a = 0.6 if st.l2_count <= 3 else self._ema_alpha  # warm-start: 前3帧高alpha快收敛
        if st.last_result == "UNKNOWN":
            st.smooth_yaw = gaze_yaw
            st.smooth_pitch = gaze_pitch
        else:
            st.smooth_yaw = a * gaze_yaw + (1.0 - a) * st.smooth_yaw
            st.smooth_pitch = a * gaze_pitch + (1.0 - a) * st.smooth_pitch

        res.gaze_yaw = st.smooth_yaw
        res.gaze_pitch = st.smooth_pitch
        res.l2_ran = True

        # 保存原始值(用于降频复用)
        st.gaze_yaw = gaze_yaw
        st.gaze_pitch = gaze_pitch

        # ── 位置补偿 + 连续帧迟滞判定 mutual_gaze ──
        # L2CS-Net 输出约定(经 5 点对齐后的实测):
        #   正 yaw = 人眼向自己左边看(画面右侧方向)
        #   负 yaw = 人眼向自己右边看(画面左侧方向)
        #
        # 人偏离画面中心时,看相机的 gaze ≠ 0:
        #   人在画面左侧(pos_offset<0) → 看相机需眼球右转 → gaze_yaw 偏负
        #   人在画面右侧(pos_offset>0) → 看相机需眼球左转 → gaze_yaw 偏正
        #   => mutual 时 gaze_yaw ≈ -pos_offset
        #   => corrected = smooth - pos_offset → mutual 时 corrected ≈ 0
        pos_offset_yaw = 0.0
        pos_offset_pitch = 0.0
        if frame_w > 0:
            face_cx = (bbox_xyxy[0] + bbox_xyxy[2]) * 0.5 * decimate
            pos_offset_yaw = (face_cx / frame_w - 0.5) * self._fov_x
        if full_rgb is not None and frame_w > 0:
            frame_h = full_rgb.shape[0]
            face_cy = (bbox_xyxy[1] + bbox_xyxy[3]) * 0.5 * decimate
            fov_y = self._fov_x * frame_h / frame_w   # 近似垂直 FOV
            pos_offset_pitch = (face_cy / frame_h - 0.5) * fov_y
        corrected_yaw = st.smooth_yaw - pos_offset_yaw
        corrected_pitch = st.smooth_pitch - pos_offset_pitch

        raw_mutual = (abs(corrected_yaw) < self._mutual_yaw
                      and abs(corrected_pitch) < self._mutual_pitch)
        if raw_mutual:
            st.looking_streak += 1
            st.not_looking_streak = 0
            if st.looking_streak >= self._confirm_frames:
                st.confirmed_mutual = True
        else:
            st.not_looking_streak += 1
            st.looking_streak = 0
            if st.not_looking_streak >= self._drop_frames:
                st.confirmed_mutual = False

        res.mutual_gaze = st.confirmed_mutual
        st.last_result = "LOOKING" if st.confirmed_mutual else "NOT_LOOKING"
        st.frames_since_check = 0

        # ── 诊断日志(每5帧打一次,不刷屏) ──
        _diag_ctr = getattr(st, '_diag_ctr', 0) + 1
        st._diag_ctr = _diag_ctr
        if _diag_ctr % 5 == 0:
            _log.info("T%d gaze raw=%.1f/%.1f smooth=%.1f/%.1f "
                      "corr=%.1f/%.1f posoff=%.1f/%.1f head=%.1f/%.1f "
                      "look=%d notlook=%d mutual=%s",
                      track_id, gaze_yaw, gaze_pitch,
                      st.smooth_yaw, st.smooth_pitch,
                      corrected_yaw, corrected_pitch,
                      pos_offset_yaw, pos_offset_pitch,
                      head_yaw, head_pitch,
                      st.looking_streak, st.not_looking_streak,
                      st.confirmed_mutual)

        return res

    def _needs_l2(self, st: _TrackGazeState) -> bool:
        if st.last_result == "UNKNOWN":
            return True
        st.frames_since_check += 1
        interval = (self._looking_interval if st.last_result == "LOOKING"
                    else self._not_looking_interval)
        if st.frames_since_check >= interval:
            st.frames_since_check = 0
            return True
        return False

    def gc(self, alive_keys: set[str]) -> None:
        for k in [k for k in self._states if k not in alive_keys]:
            del self._states[k]
