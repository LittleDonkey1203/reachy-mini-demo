#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gaze estimation CPU benchmark — L0 头姿 + L2 ONNX latency (p50/p95/p99)。

用法:
    python scripts/benchmark_gaze.py [--n 200] [--model models/l2csnet_mobilenetv2.onnx]
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.gaze import HeadPoseFilter, GazeEstimator


def report(name: str, times: list[float]):
    arr = np.array(times)
    print(f"  {name}: p50={np.percentile(arr, 50):.2f}ms  "
          f"p95={np.percentile(arr, 95):.2f}ms  "
          f"p99={np.percentile(arr, 99):.2f}ms  "
          f"mean={arr.mean():.2f}ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="iterations")
    parser.add_argument("--model", default="models/l2csnet_mobilenetv2.onnx")
    args = parser.parse_args()

    kps5 = np.array([[80, 90], [160, 90], [120, 130], [95, 160], [145, 160]],
                    dtype=np.float32)
    face_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    print(f"Benchmark: {args.n} iterations\n")

    hpf = HeadPoseFilter(45.0, 35.0)
    times_l0 = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        hpf.estimate(kps5)
        times_l0.append((time.perf_counter() - t0) * 1000)
    report("L0 (head pose 5pt)", times_l0)

    est = GazeEstimator(args.model)
    if not est.available:
        print(f"\n  L2 model not found: {args.model}")
        print("  Download from: https://github.com/yakhyo/gaze-estimation/releases")
        return

    for _ in range(5):
        est.predict(face_rgb)
    times_l2 = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        est.predict(face_rgb)
        times_l2.append((time.perf_counter() - t0) * 1000)
    report("L2 (ONNX MobileNetV2 448)", times_l2)

    print(f"\n  Effective per-face cost ≈ {np.mean(times_l0) + np.mean(times_l2):.1f}ms "
          f"(L0 always + L2 when candidate)")


if __name__ == "__main__":
    main()
