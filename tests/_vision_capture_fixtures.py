# -*- coding: utf-8 -*-
"""从 Reachy Mini 摄像头交互式采集视觉测试夹具图。

为 tests/_vision_model_test.py 准备一批带「期望标注」的固定测试图：
人脸、各手势（fist/point/two/three/four/five/ok）、空场景负样本。
采集结果存入 tests/fixtures/，并写入/合并 manifest.json。

约定（CALIBRATION.md §3）：只用 mini.media.get_frame()，先 start_recording + 预热。

运行：
  cd reachy-mini-demo
  python tests/_vision_capture_fixtures.py            # 采集默认全套
  python tests/_vision_capture_fixtures.py five ok    # 只补拍指定项
"""

import json
import os
import sys
import time

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

import numpy as np
from PIL import Image

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MANIFEST_PATH = os.path.join(FIXTURES_DIR, "manifest.json")
DECIMATE = 3

# (文件名, 提示语, expect)  ——  期望标注，供测试断言
PLAN = [
    ("face_front.jpg", "正脸对准摄像头（不要露手）",
     {"face": True, "n_faces": 1, "hand": False, "gesture": None}),
    ("hand_fist.jpg", "握拳（fist）对准摄像头，靠近一些",
     {"face": True, "n_faces": 1, "hand": True, "gesture": "fist"}),
    ("hand_point.jpg", "只伸食指指向镜头（point）",
     {"face": True, "n_faces": 1, "hand": True, "gesture": "point"}),
    ("hand_two.jpg", "伸食指+中指（two / 剪刀手）",
     {"face": True, "n_faces": 1, "hand": True, "gesture": "two"}),
    ("hand_three.jpg", "伸三指（three）",
     {"face": True, "n_faces": 1, "hand": True, "gesture": "three"}),
    ("hand_four.jpg", "伸四指（four）",
     {"face": True, "n_faces": 1, "hand": True, "gesture": "four"}),
    ("hand_five.jpg", "张开五指（five）",
     {"face": True, "n_faces": 1, "hand": True, "gesture": "five"}),
    ("hand_ok.jpg", "比 OK 手势（拇指+食指捏圈，其余三指伸直）",
     {"face": True, "n_faces": 1, "hand": True, "gesture": "ok"}),
    ("no_target.jpg", "把镜头对准空白墙面/桌面（无脸无手）",
     {"face": False, "n_faces": 0, "hand": False, "gesture": None}),
]


def load_manifest() -> list:
    if os.path.isfile(MANIFEST_PATH):
        try:
            return json.loads(open(MANIFEST_PATH, encoding="utf-8").read())
        except Exception:
            pass
    return []


def save_manifest(entries: list) -> None:
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def upsert(entries: list, file: str, expect: dict) -> None:
    """按文件名去重合并。"""
    for e in entries:
        if e.get("file") == file:
            e["expect"] = expect
            return
    entries.append({"file": file, "expect": expect})


def grab_stable(mini, settle_s: float = 0.4) -> np.ndarray:
    """连续取帧丢弃前几帧，返回一张稳定帧（RGB, 已降采样）。"""
    t = time.monotonic()
    last = None
    while time.monotonic() - t < settle_s:
        raw = mini.media.get_frame()
        if raw is not None:
            last = raw
        time.sleep(0.03)
    if last is None:
        last = mini.media.get_frame()
    return np.ascontiguousarray(last[::DECIMATE, ::DECIMATE, ::-1])


def main() -> int:
    # 过滤要采集的项：无参=全套；有参=按手势/名匹配
    sel = sys.argv[1:]
    plan = PLAN
    if sel:
        keys = set(sel)
        plan = [p for p in PLAN
                if any(k in p[0] or k == (p[2].get("gesture") or "") for k in keys)]
        if not plan:
            print(f"❌ 没有匹配的采集项: {sel}")
            print("   可选: " + ", ".join(p[0] for p in PLAN))
            return 1

    try:
        from reachy_mini import ReachyMini
    except ImportError:
        print("❌ reachy_mini 不可导入，请在机器人环境运行。")
        return 1

    os.makedirs(FIXTURES_DIR, exist_ok=True)
    entries = load_manifest()
    print(f"=== 夹具采集 → {FIXTURES_DIR} ===")
    print(f"待拍 {len(plan)} 张。每项按提示摆好后回车拍摄，输入 s 跳过，q 退出。")

    with ReachyMini(connection_mode="localhost_only", media_backend="default",
                    automatic_body_yaw=False) as mini:
        mini.media.start_recording()
        print("预热 3s...", end="", flush=True)
        t_w = time.monotonic()
        while time.monotonic() - t_w < 3.0:
            mini.media.get_frame(); time.sleep(0.05)
        print(" 就绪")

        try:
            for fname, prompt, expect in plan:
                ans = input(f"\n[{fname}] {prompt}\n  回车拍摄 / s 跳过 / q 退出: ").strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    print("  跳过")
                    continue
                rgb = grab_stable(mini)
                fpath = os.path.join(FIXTURES_DIR, fname)
                Image.fromarray(rgb).save(fpath, "JPEG", quality=92)
                upsert(entries, fname, expect)
                save_manifest(entries)
                print(f"  ✅ 已存 {fpath}  ({rgb.shape[1]}×{rgb.shape[0]})  期望={expect}")
        except KeyboardInterrupt:
            print("\n中断")
        finally:
            mini.media.stop_recording()

    save_manifest(entries)
    print(f"\n完成。manifest: {MANIFEST_PATH} （{len(entries)} 条）")
    print("下一步: python tests/_vision_model_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
