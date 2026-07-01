#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大大 人脸重新采集工具（纯终端，无 GUI）
用法: .venv/bin/python tools/recapture_face.py

配合 dashboard 看画面，终端按 Enter 采集当前帧的 embedding。
"""

import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import cv2
import numpy as np
import json
from reachy_mini import ReachyMini

from identity.recognizer import ArcFaceONNX, _align_face

DB_PATH = os.path.join(ROOT, "data", "face_db.json")
DADA_PID = "person_d810a436"
KUNKUN_PID = "person_ea67b91b"
YUNET_PATH = os.path.join(ROOT, "models", "face_detection_yunet_2023mar.onnx")
DECIMATE = 3


def load_db():
    with open(DB_PATH) as f:
        return json.load(f)


def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def main():
    db = load_db()
    dada = db.get(DADA_PID)
    if not dada:
        print(f"❌ 找不到 {DADA_PID}")
        return

    kk_embs = np.array(db.get(KUNKUN_PID, {}).get("embeddings", []), dtype=np.float32)
    print(f"坤坤: {len(kk_embs)} embeddings")

    old_count = len(dada.get("embeddings", []))
    print(f"大大: {old_count} embeddings → 即将清空并重新采集")
    print()
    print("操作: Enter=采集  q=保存退出  Ctrl+C=放弃")
    print("请从不同角度采集 8-10 个（正面、左右转、抬低头）")
    print()
    input("按 Enter 连接 Reachy Mini…")

    arcface = ArcFaceONNX()
    yunet = cv2.FaceDetectorYN.create(YUNET_PATH, "", (640, 480), 0.65, 0.3)

    print("连接 Reachy Mini…")
    mini = ReachyMini(connection_mode="localhost_only", media_backend="default")
    mini.__enter__()
    time.sleep(1.0)

    test = mini.media.get_frame()
    if test is None:
        print("❌ 摄像头无画面")
        mini.__exit__(None, None, None)
        return
    print(f"✅ 摄像头已连接 ({test.shape[1]}x{test.shape[0]})")
    print()

    collected = []

    try:
        while True:
            cmd = input(f"  [{len(collected)}/10] Enter=采集, q=保存退出 > ").strip().lower()
            if cmd == "q":
                break

            frame = mini.media.get_frame()
            if frame is None:
                print("  ⚠ 无画面，重试")
                continue

            small = frame[::DECIMATE, ::DECIMATE]
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            h, w = small.shape[:2]
            yunet.setInputSize((w, h))
            _, faces = yunet.detect(small)

            if faces is None or len(faces) == 0:
                print("  ⚠ 未检测到人脸，请正对摄像头")
                continue

            face = faces[0]
            kps = [(float(face[4 + i * 2]), float(face[5 + i * 2])) for i in range(5)]
            aligned = _align_face(rgb, kps)
            emb = arcface.get_embedding(aligned)

            # 与坤坤交叉
            cross_max = 0.0
            if len(kk_embs) > 0:
                cross_max = max(float(np.dot(emb, k)) for k in kk_embs)

            # 与已采集的内部 sim
            internal_max = 0.0
            if collected:
                internal_max = max(float(np.dot(emb, np.array(c, dtype=np.float32)))
                                   for c in collected)

            if internal_max > 0.85:
                print(f"  ⚠ 与已采集 sim={internal_max:.3f} 太近，请换个角度")
                continue

            collected.append(emb.tolist())
            warn = " ⚠坤坤交叉高!" if cross_max > 0.50 else ""
            print(f"  ✅ #{len(collected)} 坤坤sim={cross_max:.3f} 内部sim={internal_max:.3f}{warn}")

    except KeyboardInterrupt:
        print("\n❌ 放弃采集")
        mini.__exit__(None, None, None)
        return

    mini.__exit__(None, None, None)

    if len(collected) < 3:
        print(f"⚠ 至少需要 3 个，当前只有 {len(collected)}，未保存")
        return

    dada["embeddings"] = collected
    dada["last_seen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_db(db)
    print(f"\n✅ 已保存 {len(collected)} 个新 embeddings")

    embs_d = np.array(collected, dtype=np.float32)
    if len(kk_embs) > 0:
        cross = [float(np.dot(d, k)) for d in embs_d for k in kk_embs]
        print(f"   与坤坤: avg={np.mean(cross):.3f} max={max(cross):.3f}")
    internal = [float(np.dot(embs_d[i], embs_d[j]))
                for i in range(len(embs_d)) for j in range(i + 1, len(embs_d))]
    if internal:
        print(f"   内部多样性: avg={np.mean(internal):.3f} min={min(internal):.3f}")


if __name__ == "__main__":
    main()
