# -*- coding: utf-8 -*-
"""视觉模型测试（人脸 / 手 / 手势）——诊断脚本风格，跨平台精度对比。

动机：三个检测模型在不同平台精度差异大：
  - Windows / macOS-arm64：MediaPipe（正脸 ~100%、手部可用）
  - macOS Intel：无 MediaPipe wheel → OpenCV Haar 后备（人脸 25-67%、手部不可用）
本模块用「固定夹具图」做可复现的自动断言，在各平台跑同一批图即可直接对比精度，
并复用项目仪表盘风格的可视化（摄像头帧 + 标注框 + 置信度/尺寸/手势 + 门控状态）。

复用（不重写推理逻辑）：
  - voice/vision_worker.py  : pick_main_face / index_dir / _classify_gesture + 生产常量
  - voice/_hand_model_diag.py: gate_status / annotate_image + 门控阈值

三种模式：
  1. 夹具模式（默认，无参）：遍历 tests/fixtures/manifest.json，断言期望检测，
       存标注图到 tests/output/，打印 ✅/❌/⏭️ 与汇总通过率，exit(0/1)。
         python tests/_vision_model_test.py
  2. 实时摄像头模式：从 Reachy Mini 取帧实时检测 + 标注，滚动统计，供人工查看。
         python tests/_vision_model_test.py --live [秒]
  3. 临时静态图模式：对任意图片跑检测 + 标注 + 打印（无需 manifest）。
         python tests/_vision_model_test.py /path/a.jpg /path/b.jpg
"""

import json
import os
import sys
import time

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

import numpy as np
from PIL import Image, ImageDraw

# ── 路径：把 repo root 和 _archive/voice 加进 sys.path 以复用生产纯函数与可视化 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_ARCHIVE_VOICE = os.path.join(ROOT, "_archive", "voice")
if _ARCHIVE_VOICE not in sys.path:
    sys.path.insert(0, _ARCHIVE_VOICE)

# 生产推理纯函数 + 常量（与线上 vision_worker 完全一致）
from perception.vision_worker import (  # noqa: E402
    pick_main_face,
    index_dir,
    _classify_gesture,
    HAND_NEAR_SCORE,
    HAND_NEAR_SIZE,
)
# 可视化 + 门控（复用诊断工具，避免重复造轮子）
from _hand_model_diag import (  # noqa: E402
    gate_status,
    annotate_image,
    PLAY_SCORE_MIN,
    PLAY_SIZE_OFF,
    PLAY_HAND_V_MAX,
)

FACE_MODEL_PATH = os.path.join(ROOT, "models", "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(ROOT, "models", "hand_landmarker.task")
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MANIFEST_PATH = os.path.join(FIXTURES_DIR, "manifest.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# 手势近义集：分类器对 one/point 边界敏感，互相视为通过
GESTURE_ALIASES = {
    "point": {"point", "one"},
    "one": {"one", "point"},
}

T0 = time.monotonic()


def ts() -> str:
    return f"[{time.monotonic()-T0:7.2f}s]"


# ══════════════════════════════════════════════════════════════════
#  MediaPipe 后端：复刻 vision_worker.py 的「生产」landmarker 参数
# ══════════════════════════════════════════════════════════════════
def create_face_landmarker():
    """与 vision_worker.vision_worker() 中一致：num_faces=2, VIDEO 模式。"""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    return mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=2))


def create_hand_landmarker():
    """与 vision_worker.vision_worker() 中一致的「生产」阈值（非诊断宽松阈值）。"""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.4))


class Detector:
    """封装 MediaPipe 模型实例，提供 run_frame(rgb, ts_ms) → det dict。

    det = {"face":(u,v,h)|None, "n_faces":int, "face_ms":float,
           "hand":{u,v,size,score,label,fingers,gesture,angle,extended}|None}
    """

    def __init__(self):
        self._face = create_face_landmarker()
        self._hand = None
        try:
            self._hand = create_hand_landmarker()
        except Exception as e:
            print(f"⚠️  HandLandmarker 加载失败（手/手势用例将跳过）: {e}")

    @property
    def hand_supported(self) -> bool:
        return self._hand is not None

    def run_frame(self, rgb: np.ndarray, ts_ms: int) -> dict:
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        out = {"face": None, "n_faces": 0, "face_ms": 0.0, "hand": None}

        t0 = time.monotonic()
        fres = self._face.detect_for_video(mp_img, ts_ms)
        out["face_ms"] = (time.monotonic() - t0) * 1000.0
        out["face"] = pick_main_face(fres)
        out["n_faces"] = len(fres.face_landmarks) if fres.face_landmarks else 0

        if self._hand is not None:
            hres = self._hand.detect_for_video(mp_img, ts_ms + 1)
            if hres.hand_landmarks:
                lms = hres.hand_landmarks[0]
                angle, extended, tip = index_dir(lms)
                fingers, gesture = _classify_gesture(lms)
                xs = [p.x for p in lms]
                ys = [p.y for p in lms]
                size = max(max(xs) - min(xs), max(ys) - min(ys))
                score = hres.handedness[0][0].score if hres.handedness else 1.0
                label = hres.handedness[0][0].display_name if hres.handedness else "?"
                out["hand"] = {"u": (min(xs) + max(xs)) / 2.0,
                               "v": (min(ys) + max(ys)) / 2.0,
                               "size": size, "score": score, "label": label,
                               "fingers": fingers, "gesture": gesture,
                               "angle": angle, "extended": extended, "tip": tip}
        return out


# ══════════════════════════════════════════════════════════════════
#  可视化：复用 _hand_model_diag.annotate_image（手），叠加人脸框 + 手势
# ══════════════════════════════════════════════════════════════════
def annotate_frame(rgb: np.ndarray, det: dict, frame_idx: int, backend: str,
                   title: str = "") -> np.ndarray:
    H, W = rgb.shape[:2]
    hand = det.get("hand")
    # 手部检测列表交给现成的 annotate_image（画底部线 + bbox + score/size + 门控状态）
    hand_dets = []
    if hand is not None:
        hand_dets.append({"u": hand["u"], "v": hand["v"], "size": hand["size"],
                          "score": hand["score"], "label": hand.get("label", "?"),
                          "xs": [], "ys": []})
    base = annotate_image(rgb, hand_dets, frame_idx)

    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # 人脸框（青色），与 d01 仪表盘风格一致
    face = det.get("face")
    if face is not None:
        fu, fv, fh = face
        fw = fh * 0.85
        x0, y0 = int((fu - fw / 2) * W), int((fv - fh / 2) * H)
        x1, y1 = int((fu + fw / 2) * W), int((fv + fh / 2) * H)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 255), width=2)
        draw.text((x0, max(0, y0 - 14)),
                  f"FACE u={fu:.2f} v={fv:.2f} h={fh:.2f} n={det.get('n_faces',0)}",
                  fill=(0, 200, 255))

    # 手势 / 手指数（写在手框下方一行）
    if hand is not None:
        g = hand.get("gesture") or "-"
        draw.text((int(hand["u"] * W) - 40, min(H - 12, int(hand["v"] * H) + 16)),
                  f"gesture={g} fingers={hand.get('fingers','-')}", fill=(0, 220, 0))

    # 左上角元信息
    info = f"[{backend}] {title}".strip()
    draw.text((4, 18), info, fill=(255, 255, 0))
    return np.array(img)


# ══════════════════════════════════════════════════════════════════
#  断言：把一帧检测结果与期望标注比对
# ══════════════════════════════════════════════════════════════════
def check_case(det: dict, expect: dict, hand_supported: bool):
    """返回 (verdict, lines)。verdict ∈ {'pass','fail','skip'}。"""
    lines = []
    ok = True

    # ── 人脸 ──
    want_face = bool(expect.get("face"))
    n_faces = det.get("n_faces", 0)
    if want_face:
        need = int(expect.get("n_faces", 1))
        if n_faces >= need:
            lines.append(f"  ✅ 人脸: 检出 {n_faces} 张 (期望≥{need})")
        else:
            ok = False
            lines.append(f"  ❌ 人脸: 检出 {n_faces} 张 (期望≥{need})")
    else:
        if det.get("face") is None:
            lines.append("  ✅ 人脸: 正确未检出（负样本）")
        else:
            ok = False
            lines.append(f"  ❌ 人脸: 误检出 {n_faces} 张（期望无脸）")

    # ── 手 / 手势 ──
    want_hand = bool(expect.get("hand"))
    want_gesture = expect.get("gesture")
    if want_hand or want_gesture:
        if not hand_supported:
            # OpenCV 后端 / 手模型缺失 → 跳过手相关断言
            lines.append("  ⏭️  手/手势: 当前后端不支持手部模型，跳过")
            return ("skip" if ok else "fail"), lines
        hand = det.get("hand")
        if hand is None:
            ok = False
            lines.append("  ❌ 手: 未检出（期望有手）")
        else:
            gpass, gstatus = gate_status(hand)
            if hand["score"] >= PLAY_SCORE_MIN and hand["size"] >= PLAY_SIZE_OFF:
                lines.append(f"  ✅ 手: score={hand['score']:.2f} size={hand['size']:.2f} {gstatus}")
            else:
                ok = False
                lines.append(f"  ❌ 手: score={hand['score']:.2f} size={hand['size']:.2f} 未过门 {gstatus}")
            if want_gesture is not None:
                got = hand.get("gesture")
                allowed = GESTURE_ALIASES.get(want_gesture, {want_gesture})
                if got in allowed:
                    lines.append(f"  ✅ 手势: {got} (期望 {want_gesture})")
                else:
                    ok = False
                    lines.append(f"  ❌ 手势: {got} (期望 {want_gesture})")
    else:
        # 期望无手 → 不应误检手
        if hand_supported and det.get("hand") is not None:
            ok = False
            h = det["hand"]
            lines.append(f"  ❌ 手: 误检出 score={h['score']:.2f} size={h['size']:.2f}（期望无手）")
        elif hand_supported:
            lines.append("  ✅ 手: 正确未检出（负样本）")

    return ("pass" if ok else "fail"), lines


# ══════════════════════════════════════════════════════════════════
#  模式 1：夹具模式（默认）
# ══════════════════════════════════════════════════════════════════
def run_fixtures() -> int:
    print(f"=== 视觉模型测试 · 夹具模式 ===")
    print(f"后端: mediapipe   人脸模型: {FACE_MODEL_PATH}")
    if not os.path.isfile(MANIFEST_PATH):
        print(f"❌ 找不到夹具清单 {MANIFEST_PATH}")
        print("   先采集夹具: python tests/_vision_capture_fixtures.py")
        return 1
    try:
        manifest = json.loads(open(MANIFEST_PATH, encoding="utf-8").read())
    except Exception as e:
        print(f"❌ 解析 manifest 失败: {e}")
        return 1
    if not manifest:
        print("⚠️  manifest 为空，无用例。先采集夹具。")
        return 1

    try:
        det_engine = Detector()
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    n_pass = n_fail = n_skip = 0
    for i, case in enumerate(manifest):
        fname = case.get("file", "")
        expect = case.get("expect", {})
        fpath = os.path.join(FIXTURES_DIR, fname)
        print(f"\n--- [{i+1}/{len(manifest)}] {fname} ---")
        if not os.path.isfile(fpath):
            print(f"  ❌ 夹具图缺失: {fpath}")
            n_fail += 1
            continue
        try:
            rgb = np.array(Image.open(fpath).convert("RGB"))
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            n_fail += 1
            continue

        det = det_engine.run_frame(rgb, ts_ms=(i + 1) * 100)
        verdict, lines = check_case(det, expect, det_engine.hand_supported)
        for ln in lines:
            print(ln)

        ann = annotate_frame(rgb, det, i, "mediapipe", title=fname)
        out_path = os.path.join(OUT_DIR, f"fixture_{i:02d}_{verdict}_{fname}")
        try:
            Image.fromarray(ann).save(out_path, "JPEG", quality=88)
        except Exception:
            pass

        if verdict == "pass":
            n_pass += 1
        elif verdict == "skip":
            n_skip += 1
        else:
            n_fail += 1

    total = len(manifest)
    rate = 100.0 * n_pass / max(total - n_skip, 1)
    print("\n" + "═" * 56)
    print(f"后端=mediapipe  用例={total}  ✅通过={n_pass}  ❌失败={n_fail}  ⏭️跳过={n_skip}")
    print(f"通过率(不含跳过)={rate:.0f}%   标注图: {OUT_DIR}")
    print("═" * 56)
    return 0 if n_fail == 0 else 1


# ══════════════════════════════════════════════════════════════════
#  模式 2：实时摄像头
# ══════════════════════════════════════════════════════════════════
def run_live(max_seconds: float = 20.0, save_every: int = 30) -> int:
    print(f"=== 视觉模型测试 · 实时模式 (mediapipe) ===")
    try:
        from reachy_mini import ReachyMini
    except ImportError:
        print("❌ reachy_mini 不可导入，请在机器人环境运行。")
        return 1
    try:
        det_engine = Detector()
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    DECIMATE = 3
    frame_idx = saved = 0
    face_hits = hand_hits = 0
    gesture_counts: dict = {}
    t_start = time.monotonic()
    last_ts = -1

    with ReachyMini(connection_mode="localhost_only", media_backend="default",
                    automatic_body_yaw=False) as mini:
        mini.media.start_recording()
        print("预热 3s...", end="", flush=True)
        t_w = time.monotonic()
        while time.monotonic() - t_w < 3.0:
            mini.media.get_frame(); time.sleep(0.05)
        print(" 开始")
        try:
            while time.monotonic() - t_start < max_seconds:
                raw = mini.media.get_frame()
                if raw is None:
                    time.sleep(0.02); continue
                rgb = np.ascontiguousarray(raw[::DECIMATE, ::DECIMATE, ::-1])
                frame_idx += 1
                ts_ms = max(last_ts + 1, int((time.monotonic() - t_start) * 1000))
                last_ts = ts_ms + 1  # 给手部时间戳留一位
                det = det_engine.run_frame(rgb, ts_ms)
                if det.get("face") is not None:
                    face_hits += 1
                hand = det.get("hand")
                if hand is not None:
                    hand_hits += 1
                    g = hand.get("gesture") or "-"
                    gesture_counts[g] = gesture_counts.get(g, 0) + 1
                    gp, gs = gate_status(hand)
                    print(f"{ts()} 👋 score={hand['score']:.2f} size={hand['size']:.2f} "
                          f"gesture={g} fingers={hand.get('fingers')} {gs}")
                if frame_idx % save_every == 0:
                    ann = annotate_frame(rgb, det, frame_idx, "mediapipe", title="live")
                    p = os.path.join(OUT_DIR, f"live_{saved:03d}.jpg")
                    Image.fromarray(ann).save(p, "JPEG", quality=85)
                    saved += 1
                    fps = frame_idx / (time.monotonic() - t_start)
                    print(f"{ts()} 📸 {p} fps={fps:.1f} 人脸{face_hits}/{frame_idx} 手{hand_hits}")
        except KeyboardInterrupt:
            print("\nCtrl+C 停止")
        finally:
            mini.media.stop_recording()

    elapsed = time.monotonic() - t_start
    print(f"\n=== 统计 ({elapsed:.1f}s, {frame_idx}帧) ===")
    print(f"  人脸检出: {face_hits}/{frame_idx} ({100*face_hits/max(frame_idx,1):.0f}%)")
    print(f"  手检出: {hand_hits}  手势分布: {gesture_counts}")
    print(f"  标注图: {OUT_DIR}")
    return 0


# ══════════════════════════════════════════════════════════════════
#  模式 3：临时静态图
# ══════════════════════════════════════════════════════════════════
def run_adhoc(paths: list) -> int:
    print(f"=== 视觉模型测试 · 临时图模式 (mediapipe) ===")
    try:
        det_engine = Detector()
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, path in enumerate(paths):
        try:
            rgb = np.array(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"❌ 读取 {path}: {e}"); continue
        det = det_engine.run_frame(rgb, ts_ms=(i + 1) * 100)
        face = det.get("face"); hand = det.get("hand")
        print(f"\n=== {os.path.basename(path)} ({rgb.shape[1]}×{rgb.shape[0]}) ===")
        if face is None:
            print("  人脸: 无")
        else:
            print(f"  人脸: u={face[0]:.2f} v={face[1]:.2f} h={face[2]:.2f} n={det.get('n_faces')}")
        if hand is None:
            print("  手: 无")
        else:
            gp, gs = gate_status(hand)
            print(f"  手: score={hand['score']:.2f} size={hand['size']:.2f} "
                  f"gesture={hand.get('gesture')} fingers={hand.get('fingers')} {gs}")
        ann = annotate_frame(rgb, det, i, "mediapipe", title=os.path.basename(path))
        p = os.path.join(OUT_DIR, f"adhoc_{i:02d}_{os.path.basename(path)}")
        Image.fromarray(ann).save(p, "JPEG", quality=88)
        print(f"  标注图: {p}")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return run_fixtures()
    if args[0] == "--live":
        secs = float(args[1]) if len(args) > 1 else 20.0
        return run_live(max_seconds=secs)
    return run_adhoc(args)


if __name__ == "__main__":
    sys.exit(main())
