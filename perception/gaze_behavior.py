# -*- coding: utf-8 -*-
"""注视行为状态机 (GazeBehaviorFSM)。

4 态:
  IDLE          — 无人/无人看我,自主探索动画
  CURIOUS_LOOK  — 1人看我,好奇地注视对方
  SCANNING      — 多人看我,视线在看我的人之间缓慢扫过
  GLANCING      — 有人在场但没人看我,偶尔瞥一下最近的人脸

铁律: 纯逻辑,不写 st.state、不调 head_control。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class GazeBehavior(Enum):
    IDLE = auto()
    CURIOUS_LOOK = auto()
    SCANNING = auto()
    GLANCING = auto()


@dataclass
class GazeCommand:
    behavior: GazeBehavior
    target_track_id: Optional[int] = None
    scan_targets: list[int] = field(default_factory=list)
    scan_index: int = 0


class GazeBehaviorFSM:
    def __init__(self, idle_timeout_s: float = 2.0,
                 scan_period_s: float = 2.5,
                 glance_interval_s: float = 4.0):
        self.state = GazeBehavior.IDLE
        self._idle_timeout = idle_timeout_s
        self._scan_period = scan_period_s
        self._glance_interval = glance_interval_s
        self._no_face_since: float = 0.0
        self._scan_switch_t: float = 0.0
        self._scan_idx: int = 0

    def update(self, track_views: list, now: float | None = None) -> GazeCommand:
        if now is None:
            now = time.monotonic()

        faces = [v for v in track_views if v.state == "confirmed"]
        looking = [v for v in faces if v.mutual_gaze]

        if not faces:
            if self._no_face_since == 0.0:
                self._no_face_since = now
            if (now - self._no_face_since) >= self._idle_timeout:
                self.state = GazeBehavior.IDLE
                return GazeCommand(behavior=GazeBehavior.IDLE)
            return GazeCommand(behavior=self.state)

        self._no_face_since = 0.0

        if len(looking) == 1:
            self.state = GazeBehavior.CURIOUS_LOOK
            return GazeCommand(behavior=GazeBehavior.CURIOUS_LOOK,
                               target_track_id=looking[0].track_id)

        if len(looking) > 1:
            self.state = GazeBehavior.SCANNING
            scan_ids = [v.track_id for v in looking]
            if (now - self._scan_switch_t) >= self._scan_period:
                self._scan_switch_t = now
                self._scan_idx = (self._scan_idx + 1) % len(scan_ids)
            idx = self._scan_idx % len(scan_ids)
            return GazeCommand(behavior=GazeBehavior.SCANNING,
                               target_track_id=scan_ids[idx],
                               scan_targets=scan_ids, scan_index=idx)

        self.state = GazeBehavior.GLANCING
        biggest = max(faces, key=lambda v: abs(
            (v.bbox_px[2] - v.bbox_px[0]) * (v.bbox_px[3] - v.bbox_px[1])))
        return GazeCommand(behavior=GazeBehavior.GLANCING,
                           target_track_id=biggest.track_id)
