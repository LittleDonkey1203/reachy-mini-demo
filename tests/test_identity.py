#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""身份识别基础组件测试 — ArcFaceONNX + 对齐函数。

FaceDB / IdentityRecognizer 已废弃，其测试已移除。
身份管理测试见 tests/test_facereid_port.py (IdentityStore)。

运行:
  cd reachy-mini-demo
  python tests/test_identity.py
  python tests/test_identity.py -k test_01
"""

import os
import sys

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from identity.recognizer import (
    ArcFaceONNX,
    _align_face,
    _crop_face,
    _YUNET_PATH,
    _ARCFACE_PATH,
    COSINE_THRESHOLD,
)

PASS = 0
FAIL = 0
SKIP = 0


def _check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def _skip(name: str, reason: str):
    global SKIP
    SKIP += 1
    print(f"  ⏭️  {name} — SKIP: {reason}")


def _make_face_rgb(seed: int = 42, size: int = 200) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.randint(60, 200, (size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, (200, 180, 160), -1)
    cv2.circle(img, (size // 3, size // 3), size // 12, (50, 50, 50), -1)
    cv2.circle(img, (2 * size // 3, size // 3), size // 12, (50, 50, 50), -1)
    cv2.ellipse(img, (size // 2, 2 * size // 3), (size // 6, size // 12),
                0, 0, 180, (150, 80, 80), 2)
    return img


def _make_face_112(seed: int = 42) -> np.ndarray:
    img = _make_face_rgb(seed, 200)
    return cv2.resize(img, (112, 112))


def test_01_model_files():
    print("\n[Test 01] 模型文件与加载")
    _check("YuNet 模型文件存在", os.path.exists(_YUNET_PATH),
           f"缺少 {_YUNET_PATH}")
    _check("arcface 模型文件存在", os.path.exists(_ARCFACE_PATH),
           f"缺少 {_ARCFACE_PATH}")

    arcface_size = os.path.getsize(_ARCFACE_PATH) if os.path.exists(_ARCFACE_PATH) else 0
    _check("arcface 模型大小合理 (>1MB)", arcface_size > 1_000_000,
           f"实际 {arcface_size} bytes，可能下载损坏")

    if arcface_size < 1_000_000:
        return False

    try:
        arc = ArcFaceONNX()
        _check("arcface ONNX 加载成功", True)
    except Exception as e:
        _check("arcface ONNX 加载成功", False, str(e))
        return False

    face = _make_face_112()
    emb = arc.get_embedding(face)
    _check("embedding 维度 = 512", emb.shape == (512,), f"shape={emb.shape}")
    norm = np.linalg.norm(emb)
    _check("embedding L2 归一化 ≈ 1.0", abs(norm - 1.0) < 0.01, f"norm={norm:.4f}")
    return True


def test_02_embedding_consistency(arc: ArcFaceONNX):
    print("\n[Test 02] embedding 一致性")
    face = _make_face_112(seed=100)
    emb1 = arc.get_embedding(face)
    emb2 = arc.get_embedding(face)
    sim = float(np.dot(emb1, emb2))
    _check("同图两次 embedding 相同", sim > 0.999, f"sim={sim:.6f}")

    face_b = _make_face_112(seed=9999)
    emb_b = arc.get_embedding(face_b)
    cross_sim = float(np.dot(emb1, emb_b))
    print(f"    不同 seed 的 cross sim: {cross_sim:.4f}")
    _check("不同图 embedding 有差异", cross_sim < 0.99)


def test_03_alignment():
    print("\n[Test 03] 人脸对齐")
    img = _make_face_rgb(seed=1200, size=400)

    kps = [
        (140.0, 130.0),
        (260.0, 130.0),
        (200.0, 200.0),
        (155.0, 260.0),
        (245.0, 260.0),
    ]
    aligned = _align_face(img, kps)
    _check("对齐输出 112×112", aligned.shape[:2] == (112, 112))
    _check("对齐输出 3 通道", aligned.shape[2] == 3)

    box = (100, 80, 200, 250)
    cropped = _crop_face(img, box)
    _check("裁剪输出 112×112", cropped.shape[:2] == (112, 112))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=str, help="只运行匹配的用例 (e.g. test_01)")
    args = parser.parse_args()

    print("=" * 60)
    print("  身份识别基础组件测试 — ArcFaceONNX + 对齐")
    print("=" * 60)

    all_tests = {
        "test_01": (test_01_model_files, []),
        "test_02": (test_02_embedding_consistency, ["arc"]),
        "test_03": (test_03_alignment, []),
    }

    arc = None

    for name, (fn, deps) in all_tests.items():
        if args.k and args.k not in name:
            continue
        if "arc" in deps and arc is None:
            try:
                arc = ArcFaceONNX()
            except Exception as e:
                print(f"\n⚠️  arcface 模型加载失败，跳过需要 arc 的测试: {e}")
                break
        try:
            if "arc" in deps:
                fn(arc)
            else:
                fn()
        except Exception as e:
            print(f"\n  💥 {name} 异常: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  结果: ✅ {PASS}  ❌ {FAIL}  ⏭️  {SKIP}")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
