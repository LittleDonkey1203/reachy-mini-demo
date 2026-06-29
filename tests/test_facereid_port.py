# -*- coding: utf-8 -*-
"""移植自 face-tracker-demo 的核心算法单测(ByteTracker / IdentityStore / clustering / quality)。

纯 numpy+scipy,零硬件依赖,可 CI。运行:
  python -m pytest tests/test_facereid_port.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.face_config import TrackingConfig, IdentityConfig, SmoothingConfig, ClusteringConfig
from perception.face_tracker import ByteTracker, STrack, Detection, iou_batch, linear_assignment
from perception.quality import compute_quality_proxy
from identity.identity_store import IdentityStore
from identity.clustering import GalleryClustering, EmbeddingSmoother


# ── helpers ──────────────────────────────────────────────

def _box(cx, cy, size=60):
    h = size / 2
    return np.array([cx - h, cy - h, cx + h, cy + h], dtype=np.float32)


def _kps(cx, cy, w=60):
    """正脸 5 点:右眼/左眼/鼻/右嘴/左嘴(像素)。"""
    return np.array([[cx - 0.18 * w, cy - 0.15 * w], [cx + 0.18 * w, cy - 0.15 * w],
                     [cx, cy], [cx - 0.12 * w, cy + 0.2 * w], [cx + 0.12 * w, cy + 0.2 * w]],
                    dtype=np.float32)


def _det(cx, cy, size=60, conf=0.9, emb=None, q=0.6):
    return Detection(bbox=_box(cx, cy, size), confidence=conf,
                     landmarks=_kps(cx, cy, size), embedding=emb, quality=q)


def _nrm(seed):
    rng = np.random.RandomState(seed)
    e = rng.randn(512).astype(np.float32)
    return e / np.linalg.norm(e)


def _orth(base, seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(len(base)).astype(np.float32)
    v = v - np.dot(v, base) * base
    return v / (np.linalg.norm(v) + 1e-8)


def _at_cos(base, c, seed=0):
    """返回与 base 余弦≈c 的单位向量。"""
    o = _orth(base, seed)
    e = c * base + np.sqrt(max(0.0, 1 - c * c)) * o
    return (e / np.linalg.norm(e)).astype(np.float32)


@pytest.fixture(autouse=True)
def _reset_ids():
    STrack.reset_id_counter()
    yield


# ── ByteTracker ──────────────────────────────────────────

def test_iou_batch_basic():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    iou = iou_batch(a, b)
    assert iou.shape == (1, 2)
    assert iou[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert iou[0, 1] == pytest.approx(0.0, abs=1e-3)


def test_track_lifecycle_tentative_to_confirmed():
    tr = ByteTracker(TrackingConfig(min_hits=3))
    for _ in range(2):
        out = tr.update([_det(100, 100)])
        assert out == []                      # 还没 confirmed
    out = tr.update([_det(100, 100)])
    assert len(out) == 1 and out[0].is_confirmed()
    tid = out[0].track_id
    for _ in range(3):
        out = tr.update([_det(100, 100)])
        assert out[0].track_id == tid


def test_two_faces_independent():
    tr = ByteTracker(TrackingConfig(min_hits=1))
    out = tr.update([_det(100, 100), _det(400, 100)])
    assert len({t.track_id for t in out}) == 2


def test_byte_low_conf_keeps_track_no_new():
    """低置信 det 只能在 Stage2 续上已有 track,不创建新 track。"""
    tr = ByteTracker(TrackingConfig(min_hits=1))
    out = tr.update([_det(100, 100, conf=0.9)])
    tid = out[0].track_id
    out = tr.update([_det(102, 100, conf=0.3)])   # 0.1<0.3<0.6 低置信,同位置
    assert len(out) == 1 and out[0].track_id == tid   # 续上,无新 track


def test_stage3_lost_reid_by_embedding():
    """track 丢失后,带相似 embedding 的高置信 det 在 Stage3 按外观找回。"""
    e = _nrm(1)
    tr = ByteTracker(TrackingConfig(min_hits=1, max_age=10, embedding_threshold=0.45))
    out = tr.update([_det(100, 100, emb=e)])
    tid = out[0].track_id
    for _ in range(3):
        tr.update([])                          # 丢几帧 → lost
    # 远处再现 + 相似 embedding → Stage3 ReID 复活同 id
    out = tr.update([_det(300, 300, emb=_at_cos(e, 0.9, seed=5))])
    assert any(t.track_id == tid for t in out)


def test_stage3_lost_reid_by_iou_no_embedding():
    """方案B(检测无 embedding):track 漏检一帧丢失后,同位置重检靠 IoU 在 Stage3 找回同 id。
    这是治 churn 的关键——否则无 embedding 的 lost 永远找不回,漏检一帧就新建 track。"""
    tr = ByteTracker(TrackingConfig(min_hits=1, max_age=10))
    out = tr.update([_det(100, 100)])              # 无 embedding
    tid = out[0].track_id
    tr.update([])                                  # 漏检一帧 → lost
    out = tr.update([_det(101, 100)])             # 同位置重检(无 embedding)→ 应按 IoU 找回
    assert any(t.track_id == tid for t in out), "无 embedding 时应按 IoU 位置找回 lost,不新建"


def test_smooth_embedding_ema_normalized():
    tr = ByteTracker(TrackingConfig(min_hits=1))
    e = _nrm(2)
    for i in range(5):
        tr.update([_det(100, 100, emb=_at_cos(e, 0.97, seed=i))])
    trk = tr.get_primary_target()
    assert trk.smooth_embedding is not None
    assert np.linalg.norm(trk.smooth_embedding) == pytest.approx(1.0, abs=1e-3)


# ── IdentityStore 三区间 ──────────────────────────────────

def _store():
    return IdentityStore(IdentityConfig())


def test_identity_known_match():
    s = _store()
    base = _nrm(10)
    pid = s.register_identity("Alice", [base])
    r = s.match(_at_cos(base, 0.92, seed=1))   # dist≈0.08 ≤0.65 → known
    assert r.zone == "known" and r.identity_id == pid and r.identity_name == "Alice"


def test_identity_unsure_zone_no_commit():
    s = _store()
    base = _nrm(11)
    s.register_identity("Bob", [base])
    r = s.match(_at_cos(base, 0.28, seed=2))   # dist≈0.72 ∈(0.65,0.80) → unsure
    assert r.zone == "unsure" and r.identity_id is None


def test_identity_unknown_zone():
    s = _store()
    s.register_identity("Carol", [_nrm(12)])
    r = s.match(_at_cos(_nrm(12), 0.05, seed=3))  # dist≈0.95 ≥0.80 → unknown
    assert r.zone == "unknown" and r.identity_id is None


def test_register_unknown_merges_provisional():
    s = _store()
    base = _nrm(13)
    r1 = s.register_unknown(base)
    r2 = s.register_unknown(_at_cos(base, 0.95, seed=4))   # 同人 → 合并进同一 provisional
    assert r1.identity_id == r2.identity_id
    assert len(s.identities) == 1 and not list(s.identities.values())[0].is_confirmed


def test_confirm_identity():
    s = _store()
    r = s.register_unknown(_nrm(14))
    assert s.confirm_identity(r.identity_id, "Dave")
    assert s.identities[r.identity_id].is_confirmed
    assert s.identities[r.identity_id].name == "Dave"


def test_quality_gate_blocks_low_quality_enroll():
    s = _store()
    base = _nrm(15)
    pid = s.register_identity("Eve", [base])
    n0 = len(s.identities[pid].embeddings)
    s.match_and_update(_at_cos(base, 0.9, seed=6), quality=0.1)   # 质量<0.40 不入库
    assert len(s.identities[pid].embeddings) == n0
    s.match_and_update(_at_cos(base, 0.9, seed=7), quality=0.8)   # 达标 → 入库
    assert len(s.identities[pid].embeddings) == n0 + 1


def test_gallery_save_load_roundtrip(tmp_path):
    s = _store()
    base = _nrm(16)
    pid = s.register_identity("Frank", [base], [0.9], confirmed=True)
    p = tmp_path / "gallery.json"
    s.save(p)
    s2 = IdentityStore(IdentityConfig())
    assert s2.load(p) == 1
    assert s2.identities[pid].name == "Frank" and s2.identities[pid].is_confirmed


def test_distance_log_bounded():
    cfg = IdentityConfig()
    cfg.distance_log_max = 10
    s = IdentityStore(cfg)
    s.register_identity("G", [_nrm(17)])
    for i in range(30):
        s.match(_at_cos(_nrm(17), 0.5, seed=i))
    assert len(s.distance_log) == 10        # deque 上限生效


# ── GalleryClustering ────────────────────────────────────

def test_cluster_identity_finds_modes():
    gc = GalleryClustering(ClusteringConfig(max_modes=4, min_mode_distance=0.25))
    a = _nrm(20)
    b = _at_cos(a, 0.3, seed=21)              # 与 a 差很大(另一姿态 mode)
    embs = [a, _at_cos(a, 0.98, 1), b, _at_cos(b, 0.98, 2)]
    modes = gc.cluster_identity(embs)
    assert 2 <= len(modes) <= 4


def test_find_mergeable_and_merge():
    s = _store()
    base = _nrm(22)
    id1 = s.register_unknown(base).identity_id
    # 第二个 provisional,质心与第一个很近(同人跨会话)
    id2 = s.register_identity("Unknown-99", [_at_cos(base, 0.92, 8)], confirmed=False)
    gc = GalleryClustering(ClusteringConfig(merge_threshold=0.55))
    pairs = gc.find_mergeable_pairs(s.identities)
    assert pairs and {pairs[0][0], pairs[0][1]} == {id1, id2}
    assert gc.merge_identities(s.identities, id1, id2)
    assert id2 not in s.identities and id1 in s.identities


def test_merge_skips_confirmed():
    s = _store()
    base = _nrm(23)
    s.register_identity("Named", [base], confirmed=True)
    s.register_identity("AlsoNamed", [_at_cos(base, 0.93, 9)], confirmed=True)
    gc = GalleryClustering()
    assert gc.find_mergeable_pairs(s.identities) == []   # confirmed 不参与合并


def test_compact_gallery_runs():
    s = _store()
    base = _nrm(24)
    embs = [_at_cos(base, 0.99, i) for i in range(8)]
    pid = s.register_identity("H", embs, [0.6] * 8)
    gc = GalleryClustering()
    stats = gc.compact_gallery(s.identities)
    assert stats["embeddings_after"] <= stats["embeddings_before"]
    assert len(s.identities[pid].embeddings) >= 1


# ── EmbeddingSmoother ────────────────────────────────────

def test_smoother_rejects_outlier():
    sm = EmbeddingSmoother(SmoothingConfig(outlier_threshold=0.55, min_samples_for_gating=3))
    base = _nrm(30)
    for i in range(4):
        sm.update(_at_cos(base, 0.99, seed=i), quality=0.8)   # 一致帧
    rej0 = sm.rejected_count
    sm.update(_at_cos(base, 0.1, seed=99), quality=0.8)        # 突变(dist≈0.9>0.55)→拒
    assert sm.rejected_count == rej0 + 1


# ── FIQA quality ─────────────────────────────────────────

def test_quality_frontal_higher_than_offcenter():
    frame = (480, 640, 3)
    bbox = _box(320, 240, 120)
    good = compute_quality_proxy(bbox, 0.95, _kps(320, 240, 120), frame)
    # 鼻尖严重偏离两眼中线 → 正脸分低
    bad_kps = _kps(320, 240, 120).copy()
    bad_kps[2][0] += 60
    bad = compute_quality_proxy(bbox, 0.95, bad_kps, frame)
    assert 0.0 <= bad < good <= 1.0
