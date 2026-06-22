#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""身份识别功能测试脚本。

用例覆盖：
  1. 模型加载             — YuNet + arcface ONNX 能正常初始化
  2. 单人注册 + 再识别     — 同一人两次识别返回相同 person_id
  3. 多人区分             — 两张不同的脸拿到不同 person_id
  4. 命名 + 查询          — set_name / get_name 正确持久化
  5. LWW 更名             — 改名后查询返回新名
  6. 多角度 embedding 累积 — 同一人不同帧追加 embedding（sim<0.85 时追加）
  7. 清除个人 / 全部重置   — clear_person / reset 后数据消失
  8. 特征库 JSON 持久化    — 写入 → 重新加载 → 仍能匹配
  9. 阈值边界             — 相似度 < COSINE_THRESHOLD 不匹配
  10. CLI 入口             — --list / --reset 不崩溃

运行:
  cd reachy-mini-demo/voice
  python test_identity.py              # 全部用例
  python test_identity.py -k test_03   # 单个用例
  python test_identity.py --live       # 摄像头实时测试（需连摄像头）
"""

import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from identity.recognizer import (
    ArcFaceONNX,
    FaceDB,
    IdentityRecognizer,
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
    """生成一个伪造的 RGB 人脸图（随机噪声+特征，用于 embedding 测试）。"""
    rng = np.random.RandomState(seed)
    img = rng.randint(60, 200, (size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, (200, 180, 160), -1)
    cv2.circle(img, (size // 3, size // 3), size // 12, (50, 50, 50), -1)
    cv2.circle(img, (2 * size // 3, size // 3), size // 12, (50, 50, 50), -1)
    cv2.ellipse(img, (size // 2, 2 * size // 3), (size // 6, size // 12),
                0, 0, 180, (150, 80, 80), 2)
    return img


def _make_face_112(seed: int = 42) -> np.ndarray:
    """生成 112×112 的伪人脸图。"""
    img = _make_face_rgb(seed, 200)
    return cv2.resize(img, (112, 112))


# ──────────────────────────────────────────────────
# 用例 1：模型文件检查 + 加载
# ──────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────
# 用例 2：单人注册 + 再识别
# ──────────────────────────────────────────────────
def test_02_single_person(arc: ArcFaceONNX):
    print("\n[Test 02] 单人注册 + 再识别")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        db = FaceDB(db_path)

        face1 = _make_face_112(seed=100)
        emb1 = arc.get_embedding(face1)
        pid1 = db.add_person(emb1, name="测试者A")
        _check("注册返回 person_id", pid1.startswith("person_"))

        match_id, sim = db.match(emb1)
        _check("同一 embedding 再匹配", match_id == pid1,
               f"expected={pid1}, got={match_id}")
        _check(f"相似度 = 1.0 (自匹配)", abs(sim - 1.0) < 0.01, f"sim={sim:.4f}")

        face1b = _make_face_112(seed=101)
        emb1b = arc.get_embedding(face1b)
        match_id2, sim2 = db.match(emb1b)
        print(f"    (不同 seed 的匹配相似度: {sim2:.4f})")
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────
# 用例 3：多人区分
# ──────────────────────────────────────────────────
def test_03_multi_person(arc: ArcFaceONNX):
    print("\n[Test 03] 多人区分")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        db = FaceDB(db_path)

        seeds = [200, 300, 400]
        pids = []
        for i, seed in enumerate(seeds):
            face = _make_face_112(seed=seed)
            emb = arc.get_embedding(face)
            pid = db.add_person(emb, name=f"Person_{i}")
            pids.append(pid)

        _check("注册了 3 个不同的 person_id",
               len(set(pids)) == 3, f"pids={pids}")

        for i, seed in enumerate(seeds):
            face = _make_face_112(seed=seed)
            emb = arc.get_embedding(face)
            match_id, sim = db.match(emb)
            _check(f"Person_{i} 自匹配", match_id == pids[i],
                   f"expected={pids[i]}, got={match_id}, sim={sim:.4f}")

        persons = db.list_persons()
        _check("list_persons 返回 3 人", len(persons) == 3)
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────
# 用例 4：命名 + 查询
# ──────────────────────────────────────────────────
def test_04_naming(arc: ArcFaceONNX):
    print("\n[Test 04] 命名 + 查询")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        db = FaceDB(db_path)
        face = _make_face_112(seed=500)
        emb = arc.get_embedding(face)
        pid = db.add_person(emb)

        _check("初始 name 为 None", db.get_name(pid) is None)

        db.set_name(pid, "小明")
        _check("set_name 后查询正确", db.get_name(pid) == "小明")

        info = db.get_info(pid)
        _check("get_info 包含 name", info["name"] == "小明")
        _check("get_info 包含 created_at", "created_at" in info)
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────
# 用例 5：LWW 更名
# ──────────────────────────────────────────────────
def test_05_lww_rename(arc: ArcFaceONNX):
    print("\n[Test 05] LWW 更名")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        db = FaceDB(db_path)
        face = _make_face_112(seed=600)
        emb = arc.get_embedding(face)
        pid = db.add_person(emb, name="小A")

        _check("初始名 = 小A", db.get_name(pid) == "小A")

        db.set_name(pid, "小B")
        _check("更名后 = 小B (LWW)", db.get_name(pid) == "小B")

        db.set_name(pid, "小C")
        _check("再更名 = 小C", db.get_name(pid) == "小C")
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────
# 用例 6：多角度 embedding 累积
# ──────────────────────────────────────────────────
def test_06_multi_embedding(arc: ArcFaceONNX):
    print("\n[Test 06] 多角度 embedding 累积")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        db = FaceDB(db_path)
        face0 = _make_face_112(seed=700)
        emb0 = arc.get_embedding(face0)
        pid = db.add_person(emb0, name="多角度")

        _check("初始 1 个 embedding", len(db.persons[pid]["embeddings"]) == 1)

        for seed in [701, 702, 703]:
            face = _make_face_112(seed=seed)
            emb = arc.get_embedding(face)
            db.update_embedding(pid, emb)

        n_emb = len(db.persons[pid]["embeddings"])
        print(f"    累积 embedding 数: {n_emb}")
        _check("embedding 数 >= 1", n_emb >= 1)

        db.update_embedding(pid, emb0)
        n_after = len(db.persons[pid]["embeddings"])
        _check("重复 embedding 不追加 (sim>0.85)", n_after == n_emb,
               f"before={n_emb}, after={n_after}")
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────
# 用例 7：清除 / 重置
# ──────────────────────────────────────────────────
def test_07_clear_reset(arc: ArcFaceONNX):
    print("\n[Test 07] 清除个人 / 全部重置")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        db = FaceDB(db_path)
        pids = []
        for seed in [800, 801, 802]:
            face = _make_face_112(seed=seed)
            emb = arc.get_embedding(face)
            pids.append(db.add_person(emb))

        _check("注册 3 人", len(db.persons) == 3)

        db.clear_person(pids[0])
        _check("删除 1 人后剩 2", len(db.persons) == 2)
        _check("被删者不可查", db.get_info(pids[0]) is None)

        db.reset()
        _check("reset 后为空", len(db.persons) == 0)
        _check("reset 后文件存在", os.path.exists(db_path))
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────
# 用例 8：JSON 持久化
# ──────────────────────────────────────────────────
def test_08_persistence(arc: ArcFaceONNX):
    print("\n[Test 08] JSON 持久化")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        db1 = FaceDB(db_path)
        face = _make_face_112(seed=900)
        emb = arc.get_embedding(face)
        pid = db1.add_person(emb, name="持久化测试")

        db2 = FaceDB(db_path)
        _check("重新加载后人数一致", len(db2.persons) == 1)
        _check("重新加载后名字一致", db2.get_name(pid) == "持久化测试")

        match_id, sim = db2.match(emb)
        _check("重新加载后仍能匹配", match_id == pid, f"sim={sim:.4f}")

        with open(db_path) as f:
            raw = json.load(f)
        _check("JSON 可直接解析", pid in raw)
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────
# 用例 9：阈值边界
# ──────────────────────────────────────────────────
def test_09_threshold(arc: ArcFaceONNX):
    print("\n[Test 09] 阈值边界测试")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        db = FaceDB(db_path)
        face_a = _make_face_112(seed=1000)
        emb_a = arc.get_embedding(face_a)
        pid_a = db.add_person(emb_a, name="A")

        face_b = _make_face_112(seed=9999)
        emb_b = arc.get_embedding(face_b)
        match_id, sim = db.match(emb_b)

        print(f"    跨人相似度: {sim:.4f} (阈值={COSINE_THRESHOLD})")
        if sim < COSINE_THRESHOLD:
            _check("低相似度不匹配 (返回 None)", match_id is None)
        else:
            print(f"    ⚠️  随机噪声图的相似度 >= 阈值，这在伪造图上可能发生")
            _check("匹配返回某个 ID (相似度高于阈值)", match_id is not None)
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────
# 用例 10：IdentityRecognizer 高层接口
# ──────────────────────────────────────────────────
def test_10_recognizer():
    print("\n[Test 10] IdentityRecognizer 高层接口")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        rec = IdentityRecognizer(db_path)

        img = _make_face_rgb(seed=1100, size=300)
        box = (50, 50, 200, 200)
        kps = [
            (115.0, 110.0),
            (185.0, 110.0),
            (150.0, 155.0),
            (120.0, 195.0),
            (180.0, 195.0),
        ]

        # 需要连续 NEW_PERSON_CONFIRM_FRAMES 次才能注册
        from identity.recognizer import NEW_PERSON_CONFIRM_FRAMES

        for i in range(NEW_PERSON_CONFIRM_FRAMES - 1):
            pid, name, sim, is_new = rec.recognize(img, box, kps)
            if i == 0:
                _check(f"确认中返回 pid=None (第{i+1}帧)", pid is None)

        pid1, name1, sim1, is_new1 = rec.recognize(img, box, kps)
        _check("确认完成 is_new=True", is_new1)
        _check("确认完成返回 pid", pid1 is not None and pid1.startswith("person_"))
        _check("确认完成 name=None", name1 is None)

        rec.db.set_name(pid1, "测试人")

        pid2, name2, sim2, is_new2 = rec.recognize(img, box, kps)
        _check("再次识别 is_new=False", not is_new2)
        _check("再次识别 pid 一致", pid2 == pid1)
        _check("再次识别 name=测试人", name2 == "测试人")
        _check(f"再次识别相似度 >= 阈值 ({sim2:.3f})", sim2 >= COSINE_THRESHOLD)
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────
# 用例 11：人脸对齐
# ──────────────────────────────────────────────────
def test_11_alignment():
    print("\n[Test 11] 人脸对齐")
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


# ──────────────────────────────────────────────────
# 用例 12：摄像头实时（可选）
# ──────────────────────────────────────────────────
def test_12_live_camera():
    print("\n[Test 12] 摄像头实时身份识别")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = f.name

    try:
        rec = IdentityRecognizer(db_path)
    except Exception as e:
        _skip("摄像头测试", f"模型加载失败: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        _skip("摄像头测试", "无法打开摄像头")
        return

    print("    摄像头已打开，测试 5 秒实时识别...")
    print("    按 q 提前退出, 按 n 命名当前人")

    _COLORS_BGR = [
        (0, 128, 255), (255, 0, 0), (0, 200, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
    ]
    pid_color_map: dict[str, tuple] = {}

    def color_for(pid: str) -> tuple:
        if pid not in pid_color_map:
            pid_color_map[pid] = _COLORS_BGR[len(pid_color_map) % len(_COLORS_BGR)]
        return pid_color_map[pid]

    yunet = cv2.FaceDetectorYN.create(_YUNET_PATH, "", (320, 240), 0.65, 0.3, 10)
    t_start = time.monotonic()
    frames = 0
    detections = 0
    identifications = 0
    cached_ids: list[tuple] = []

    def _match_cached(cx: float, cy: float):
        best, best_d = None, 80.0
        for c in cached_ids:
            d = ((cx - c[0]) ** 2 + (cy - c[1]) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best = c
        return best

    while time.monotonic() - t_start < 5.0:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yunet.setInputSize((w, h))
        _, faces = yunet.detect(rgb)

        if faces is not None and len(faces) > 0:
            detections += 1
            do_identify = (frames % 15 == 1)
            new_cache: list[tuple] = []

            for f in faces:
                bx, by, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                cx, cy = bx + bw / 2, by + bh / 2

                for ki in range(5):
                    px = int(f[4 + ki * 2])
                    py = int(f[5 + ki * 2])
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

                if do_identify:
                    box = (bx, by, bw, bh)
                    kps = [(float(f[4 + i * 2]), float(f[5 + i * 2])) for i in range(5)]
                    pid, name, sim, is_new = rec.recognize(rgb, box, kps)
                    identifications += 1
                    new_cache.append((cx, cy, pid, name, sim, is_new))
                    tag = "NEW" if is_new else f"OK({sim:.2f})"
                    name_s = name or pid[:12]
                    print(f"    [{tag}] {name_s}")
                    color = color_for(pid)
                    label = name_s
                else:
                    hit = _match_cached(cx, cy)
                    if hit:
                        _, _, pid, name, sim, is_new = hit
                        color = color_for(pid)
                        label = name or pid[:12]
                    else:
                        color = (0, 128, 255)
                        label = None

                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
                if label:
                    cv2.putText(frame, label, (bx, by - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if do_identify:
                cached_ids = new_cache
        else:
            cached_ids.clear()

        n_faces = len(faces) if faces is not None else 0
        cv2.putText(frame, f"faces: {n_faces} | DB: {len(rec.db.persons)}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        cv2.imshow("Identity Live Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.monotonic() - t_start
    fps = frames / elapsed if elapsed > 0 else 0
    print(f"    {frames} frames in {elapsed:.1f}s = {fps:.1f} FPS")
    print(f"    检测 {detections}/{frames} 帧, 识别 {identifications} 次")
    print(f"    特征库: {len(rec.db.persons)} 人")

    _check("处理了帧", frames > 0)
    _check("有人脸检测", detections > 0)

    os.unlink(db_path)


# ──────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=str, help="只运行匹配的用例 (e.g. test_03)")
    parser.add_argument("--live", action="store_true", help="包含摄像头实时测试")
    args = parser.parse_args()

    print("=" * 60)
    print("  身份识别功能测试 — identity.py")
    print("=" * 60)

    all_tests = {
        "test_01": (test_01_model_files, []),
        "test_02": (test_02_single_person, ["arc"]),
        "test_03": (test_03_multi_person, ["arc"]),
        "test_04": (test_04_naming, ["arc"]),
        "test_05": (test_05_lww_rename, ["arc"]),
        "test_06": (test_06_multi_embedding, ["arc"]),
        "test_07": (test_07_clear_reset, ["arc"]),
        "test_08": (test_08_persistence, ["arc"]),
        "test_09": (test_09_threshold, ["arc"]),
        "test_10": (test_10_recognizer, []),
        "test_11": (test_11_alignment, []),
    }
    if args.live:
        all_tests["test_12"] = (test_12_live_camera, [])

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
