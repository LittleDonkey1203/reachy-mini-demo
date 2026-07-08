#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注视样本标注 + 评估工具。

用法:
  1. 采集: GAZE_SAVE_SAMPLES=1 bash start_mac.sh  (运行一段时间后 Ctrl+C)
  2. 标注: python scripts/gaze_eval.py label       (逐张看图,按 y/n 标注)
  3. 评估: python scripts/gaze_eval.py eval         (算准确率 + 找最优阈值)
  4. 查看: python scripts/gaze_eval.py stats        (当前样本统计)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "gaze_samples"


def _load_samples():
    samples = []
    for jf in sorted(SAMPLE_DIR.glob("*.json")):
        with open(jf) as f:
            meta = json.load(f)
        img_path = jf.with_suffix(".jpg")
        if img_path.exists():
            meta["_img_path"] = str(img_path)
            meta["_json_path"] = str(jf)
            samples.append(meta)
    return samples


def cmd_stats():
    samples = _load_samples()
    labeled = [s for s in samples if s.get("label") is not None]
    looking = [s for s in labeled if s["label"] == "looking"]
    not_looking = [s for s in labeled if s["label"] == "not_looking"]
    print(f"总样本: {len(samples)}")
    print(f"已标注: {len(labeled)} (looking={len(looking)}, not_looking={len(not_looking)})")
    print(f"未标注: {len(samples) - len(labeled)}")


def cmd_label():
    samples = _load_samples()
    unlabeled = [s for s in samples if s.get("label") is None]
    if not unlabeled:
        print("所有样本已标注完毕!")
        cmd_stats()
        return

    print(f"待标注: {len(unlabeled)} 张")
    print("操作: y=looking  n=not_looking  s=跳过  q=退出")
    print("---")

    labeled_count = 0
    for s in unlabeled:
        img = cv2.imread(s["_img_path"])
        if img is None:
            continue

        info = (f"[{s['id']}] head={s['head_yaw']:+.1f}/{s['head_pitch']:+.1f}  "
                f"gaze_raw={s['gaze_yaw_raw']:+.1f}/{s['gaze_pitch_raw']:+.1f}")
        h, w = img.shape[:2]
        scale = max(1, 300 // min(h, w))
        display = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Gaze Sample - y/n/s/q", display)
        cv2.setWindowTitle("Gaze Sample - y/n/s/q", info)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                s["label"] = "looking"
                break
            elif key == ord('n'):
                s["label"] = "not_looking"
                break
            elif key == ord('s'):
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print(f"\n标注了 {labeled_count} 张")
                cmd_stats()
                return

        if s.get("label") is not None:
            with open(s["_json_path"], "w") as f:
                json.dump({k: v for k, v in s.items() if not k.startswith("_")},
                          f, ensure_ascii=False)
            labeled_count += 1

    cv2.destroyAllWindows()
    print(f"\n标注完成! 共 {labeled_count} 张")
    cmd_stats()


def cmd_eval():
    samples = _load_samples()
    labeled = [s for s in samples if s.get("label") is not None]
    if len(labeled) < 5:
        print(f"已标注样本太少({len(labeled)}),至少需要 5 个")
        return

    gt = np.array([1 if s["label"] == "looking" else 0 for s in labeled])
    raw_yaw = np.array([abs(s["gaze_yaw_raw"]) for s in labeled])
    raw_pitch = np.array([abs(s["gaze_pitch_raw"]) for s in labeled])

    print(f"=== 样本统计 ({len(labeled)} 张, looking={gt.sum()}, not_looking={len(gt)-gt.sum()}) ===\n")

    # 分布
    look_mask = gt == 1
    print("  looking 样本:")
    print(f"    |yaw|  均值={raw_yaw[look_mask].mean():.1f}° 中位={np.median(raw_yaw[look_mask]):.1f}° "
          f"std={raw_yaw[look_mask].std():.1f}°  [min={raw_yaw[look_mask].min():.1f}, max={raw_yaw[look_mask].max():.1f}]")
    print(f"    |pitch| 均值={raw_pitch[look_mask].mean():.1f}° 中位={np.median(raw_pitch[look_mask]):.1f}° "
          f"std={raw_pitch[look_mask].std():.1f}°  [min={raw_pitch[look_mask].min():.1f}, max={raw_pitch[look_mask].max():.1f}]")

    nolook_mask = gt == 0
    if nolook_mask.sum() > 0:
        print("  not_looking 样本:")
        print(f"    |yaw|  均值={raw_yaw[nolook_mask].mean():.1f}° 中位={np.median(raw_yaw[nolook_mask]):.1f}° "
              f"std={raw_yaw[nolook_mask].std():.1f}°  [min={raw_yaw[nolook_mask].min():.1f}, max={raw_yaw[nolook_mask].max():.1f}]")
        print(f"    |pitch| 均值={raw_pitch[nolook_mask].mean():.1f}° 中位={np.median(raw_pitch[nolook_mask]):.1f}° "
              f"std={raw_pitch[nolook_mask].std():.1f}°  [min={raw_pitch[nolook_mask].min():.1f}, max={raw_pitch[nolook_mask].max():.1f}]")

    # 网格搜索最优阈值
    print("\n=== 阈值网格搜索 ===\n")
    best_f1 = 0
    best_params = (0, 0)
    results = []

    for yaw_t in range(5, 35, 2):
        for pitch_t in range(5, 35, 2):
            pred = ((raw_yaw < yaw_t) & (raw_pitch < pitch_t)).astype(int)
            tp = ((pred == 1) & (gt == 1)).sum()
            fp = ((pred == 1) & (gt == 0)).sum()
            fn = ((pred == 0) & (gt == 1)).sum()
            tn = ((pred == 0) & (gt == 0)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            acc = (tp + tn) / len(gt)
            results.append((yaw_t, pitch_t, acc, prec, rec, f1, tp, fp, fn, tn))
            if f1 > best_f1:
                best_f1 = f1
                best_params = (yaw_t, pitch_t)

    # top-5 by F1
    results.sort(key=lambda x: -x[5])
    print(f"  {'yaw_t':>5} {'pitch_t':>7} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6}  TP/FP/FN/TN")
    for r in results[:10]:
        print(f"  {r[0]:5d} {r[1]:7d} {r[2]:6.1%} {r[3]:6.1%} {r[4]:6.1%} {r[5]:6.3f}  {r[6]}/{r[7]}/{r[8]}/{r[9]}")

    by, bp = best_params
    print(f"\n★ 最优阈值: yaw={by}° pitch={bp}° (F1={best_f1:.3f})")
    print(f"  → config.py 中设置:")
    print(f"    GAZE_MUTUAL_YAW_THRESH = {by}.0")
    print(f"    GAZE_MUTUAL_PITCH_THRESH = {bp}.0")

    # 用当前阈值评估
    cur_yaw_t, cur_pitch_t = 15.0, 15.0
    pred_cur = ((raw_yaw < cur_yaw_t) & (raw_pitch < cur_pitch_t)).astype(int)
    tp = ((pred_cur == 1) & (gt == 1)).sum()
    fp = ((pred_cur == 1) & (gt == 0)).sum()
    fn = ((pred_cur == 0) & (gt == 1)).sum()
    tn = ((pred_cur == 0) & (gt == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc = (tp + tn) / len(gt)
    print(f"\n  当前阈值 (yaw={cur_yaw_t}° pitch={cur_pitch_t}°): "
          f"acc={acc:.1%} prec={prec:.1%} rec={rec:.1%} F1={f1:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "label":
        cmd_label()
    elif cmd == "eval":
        cmd_eval()
    elif cmd == "stats":
        cmd_stats()
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)
        sys.exit(1)
