#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸重新采集工具（纯终端，无 GUI）
用法: .venv/bin/python scripts/recapture_face.py --name 大大

配合 dashboard 看画面，终端按 Enter 采集当前帧的 embedding。
使用 IdentityStore (gallery.json) 存储。
"""

import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import cv2
import numpy as np
from reachy_mini import ReachyMini

from identity.recognizer import ArcFaceONNX, _align_face
from identity.identity_store import IdentityStore

GALLERY_PATH = os.path.join(ROOT, "data", "gallery.json")
YUNET_PATH = os.path.join(ROOT, "models", "face_detection_yunet_2023mar.onnx")
DECIMATE = 3


def main():
    import argparse
    parser = argparse.ArgumentParser(description="人脸重新采集工具")
    parser.add_argument("--name", required=True, help="目标人名")
    parser.add_argument("--gallery", default=GALLERY_PATH, help="gallery.json 路径")
    args = parser.parse_args()

    store = IdentityStore(args.gallery)
    target = store.find_by_name(args.name)
    if target is None:
        print(f"❌ 在 gallery 中找不到「{args.name}」")
        print(f"已有身份: {[i.name for i in store.identities.values() if i.name]}")
        return

    pid = target.identity_id
    old_count = len(target.embeddings)
    print(f"目标: {args.name} ({pid}), 当前 {old_count} embeddings → 即将清空并重新采集")
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

            # 与其他人交叉检查
            cross_max = 0.0
            cross_who = ""
            for oid, ident in store.identities.items():
                if oid == pid:
                    continue
                for oe in ident.embeddings:
                    s = float(np.dot(emb, np.array(oe, dtype=np.float32)))
                    if s > cross_max:
                        cross_max = s
                        cross_who = ident.name or oid[:12]

            # 与已采集的内部 sim
            internal_max = 0.0
            if collected:
                internal_max = max(float(np.dot(emb, np.array(c, dtype=np.float32)))
                                   for c in collected)

            if internal_max > 0.85:
                print(f"  ⚠ 与已采集 sim={internal_max:.3f} 太近，请换个角度")
                continue

            collected.append(emb.tolist())
            warn = f" ⚠{cross_who}交叉高!" if cross_max > 0.50 else ""
            print(f"  ✅ #{len(collected)} 交叉sim={cross_max:.3f}({cross_who}) 内部sim={internal_max:.3f}{warn}")

    except KeyboardInterrupt:
        print("\n❌ 放弃采集")
        mini.__exit__(None, None, None)
        return

    mini.__exit__(None, None, None)

    if len(collected) < 3:
        print(f"⚠ 至少需要 3 个，当前只有 {len(collected)}，未保存")
        return

    target.embeddings = collected
    store.save()
    print(f"\n✅ 已保存 {len(collected)} 个新 embeddings 到 gallery.json")


if __name__ == "__main__":
    main()
