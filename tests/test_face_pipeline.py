# -*- coding: utf-8 -*-
"""FaceReIDPipeline 集成层单测(fake embedder,纯逻辑,可 CI)。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.face_config import FaceSystemConfig
from perception.face_pipeline import FaceReIDPipeline
from perception.face_tracker import STrack

W, H, DEC = 640, 480, 3
FULL = np.zeros((H * DEC, W * DEC, 3), dtype=np.uint8)


def _nrm(seed):
    rng = np.random.RandomState(seed)
    e = rng.randn(512).astype(np.float32)
    return e / np.linalg.norm(e)


def _face(cx=320, cy=240, w=90):
    """一个 all_faces 项(检测帧像素)。"""
    return {"u": cx / W, "v": cy / H, "h": w / H,
            "box": (cx - w // 2, cy - w // 2, w, w),
            "kps": [[cx - 0.18 * w, cy - 0.15 * w], [cx + 0.18 * w, cy - 0.15 * w],
                    [cx, cy], [cx - 0.12 * w, cy + 0.2 * w], [cx + 0.12 * w, cy + 0.2 * w]],
            "conf": 0.9}


def _cfg(min_confirm=2, min_hits=2, min_quality=0.0):
    c = FaceSystemConfig()
    c.identity.min_confirm_frames = min_confirm
    c.identity.min_quality = min_quality
    c.tracking.min_hits = min_hits
    return c


@pytest.fixture(autouse=True)
def _reset():
    STrack.reset_id_counter()
    yield


def _run(pipe, n, face=None, t0=1000.0, dt=0.4):
    face = face or _face()
    prim = None
    for i in range(n):
        prim, _ = pipe.process([face], (W, H), FULL, DEC, now=t0 + i * dt)
    return prim


def test_unknown_registers_provisional():
    e = _nrm(1)
    pipe = FaceReIDPipeline(lambda rgb, b, k: e, _cfg(min_confirm=2, min_hits=2))
    prim = _run(pipe, 8)
    assert prim is not None and prim.person_id is not None
    assert len(pipe.store.identities) == 1
    assert not list(pipe.store.identities.values())[0].is_confirmed   # provisional


def test_known_match_binds():
    e = _nrm(2)
    pipe = FaceReIDPipeline(lambda rgb, b, k: e, _cfg())
    pid = pipe.store.register_identity("Alice", [e], confirmed=True)
    prim = _run(pipe, 6)
    assert prim is not None and prim.person_id == pid
    assert prim.zone == "known" and prim.person_name == "Alice"
    # 不应再新建身份
    assert len(pipe.store.identities) == 1


def test_unsure_does_not_commit():
    base = _nrm(3)
    # embedder 返回与库中人 cos≈0.28(dist 0.72)→ unsure
    o = _nrm(33)
    o = o - np.dot(o, base) * base
    o /= np.linalg.norm(o)
    mid = (0.28 * base + np.sqrt(1 - 0.28 ** 2) * o).astype(np.float32)
    pipe = FaceReIDPipeline(lambda rgb, b, k: mid, _cfg(min_confirm=99))  # 不让它注册
    pipe.store.register_identity("Bob", [base], confirmed=True)
    prim = _run(pipe, 6)
    assert prim is not None and prim.person_id is None       # unsure 不提交
    assert prim.zone == "unsure"
    assert len(pipe.store.identities) == 1                   # 没新建


def test_primary_normalized_coords():
    e = _nrm(4)
    pipe = FaceReIDPipeline(lambda rgb, b, k: e, _cfg())
    prim = _run(pipe, 6, face=_face(cx=160, cy=120, w=80))
    assert prim is not None
    assert prim.u == pytest.approx(160 / W, abs=0.02)
    assert prim.v == pytest.approx(120 / H, abs=0.02)
    assert 0.0 < prim.h < 1.0


def test_name_track_confirms():
    e = _nrm(5)
    pipe = FaceReIDPipeline(lambda rgb, b, k: e, _cfg())
    prim = _run(pipe, 6)
    assert prim is not None and prim.person_id is not None
    assert pipe.name_track(prim.track_id, "Carol")
    ident = pipe.store.identities[prim.tracker_pid if hasattr(prim, "tracker_pid") else prim.person_id]
    assert ident.is_confirmed and ident.name == "Carol"


def test_budget_limits_embeds_per_frame():
    e = _nrm(6)
    calls = {"n": 0}
    def emb(rgb, b, k):
        calls["n"] += 1
        return e
    pipe = FaceReIDPipeline(emb, _cfg(), emb_per_frame_budget=1)
    faces = [_face(cx=160, cy=240), _face(cx=480, cy=240)]
    pipe.tracker.cfg.min_hits = 1
    # 两脸都 confirmed 后,单帧预算=1 → 一帧最多 1 次 embedder
    for i in range(4):
        before = calls["n"]
        pipe.process(faces, (W, H), FULL, DEC, now=1000 + i)
        assert calls["n"] - before <= 1
