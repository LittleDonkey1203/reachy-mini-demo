# -*- coding: utf-8 -*-
"""手部模型精度诊断工具（离线 + 实时两模式）。

用途：排查 "手检测误触发导致机器人低头" 问题。
  - 打印每帧检测结果：score / size / (u,v) 位置 / 是否通过各道门
  - 标注 "通过 v-max 门（≤0.80）" 还是 "被底部过滤"，帮助定位误检区域
  - 保存标注图到 voice/output/hand_diag_NNN.jpg（可离线查看）

模式：
  1. 实时（默认）：从 Reachy Mini 相机取帧，实时检测，Ctrl+C 停止
  2. 静态图片：python _hand_model_diag.py <image.jpg> [...]
     对每张图片跑检测并保存标注结果

运行（实时）：
  cd reachy-mini-demo/voice
  python _hand_model_diag.py

运行（静态）：
  python _hand_model_diag.py /path/to/img1.jpg /path/to/img2.jpg
"""

import math
import os
import sys
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "vision", "models", "hand_landmarker.task")
HAND_MODEL_PATH = os.path.join(ROOT, "vision", "models", "hand_landmarker.task")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── 门槛（与 d01_realtime_chat.py 保持一致）──
PLAY_SCORE_MIN = 0.6
PLAY_SIZE_OFF  = 0.22   # 跟踪/保持下限
PLAY_SIZE_ON   = 0.30   # 进入逗它阈值
PLAY_HAND_V_MAX = 0.80  # 底部过滤阈值（v > 此值视为误检区）

T0 = time.monotonic()


def ts() -> str:
    return f"[{time.monotonic()-T0:7.2f}s]"


def detect_once(hand_lm, rgb: np.ndarray, frame_ts_ms: int):
    """跑一帧手部检测，返回 list[dict] 每个手的信息。"""
    import mediapipe as mp
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hres = hand_lm.detect_for_video(mp_img, frame_ts_ms)
    results = []
    if not hres.hand_landmarks:
        return results
    for i, lms in enumerate(hres.hand_landmarks):
        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        u = (min(xs) + max(xs)) / 2.0
        v = (min(ys) + max(ys)) / 2.0
        size = max(max(xs) - min(xs), max(ys) - min(ys))
        score = hres.handedness[i][0].score if hres.handedness else 1.0
        label = hres.handedness[i][0].display_name if hres.handedness else "?"
        results.append({"u": u, "v": v, "size": size, "score": score, "label": label,
                         "xs": xs, "ys": ys})
    return results


def gate_status(det: dict) -> tuple[bool, str]:
    """返回 (通过所有门, 原因字符串)。"""
    reasons = []
    ok = True
    if det["score"] < PLAY_SCORE_MIN:
        ok = False; reasons.append(f"score {det['score']:.2f}<{PLAY_SCORE_MIN}")
    if det["size"] < PLAY_SIZE_OFF:
        ok = False; reasons.append(f"size {det['size']:.2f}<{PLAY_SIZE_OFF}")
    if det["v"] > PLAY_HAND_V_MAX:
        ok = False; reasons.append(f"v {det['v']:.2f}>{PLAY_HAND_V_MAX}(底部过滤)")
    status = "✅ PASS" if ok else f"❌ BLOCKED ({', '.join(reasons)})"
    return ok, status


def annotate_image(rgb: np.ndarray, detections: list, frame_idx: int) -> np.ndarray:
    """在图上画出所有检测结果和门控状态，返回标注后的 RGB 数组。"""
    H, W = rgb.shape[:2]
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    # 画底部过滤线
    y_thresh = int(PLAY_HAND_V_MAX * H)
    draw.line([(0, y_thresh), (W, y_thresh)], fill=(255, 100, 0), width=2)
    draw.text((4, y_thresh - 14), f"v={PLAY_HAND_V_MAX} 底部过滤线", fill=(255, 160, 0))

    for i, d in enumerate(detections):
        ok, status = gate_status(d)
        color = (0, 220, 0) if ok else (220, 80, 0)
        half = int(d["size"] * max(W, H) / 2)
        cx, cy = int(d["u"] * W), int(d["v"] * H)
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], outline=color, width=2)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color)
        label = f"#{i} {d['label']} sc={d['score']:.2f} sz={d['size']:.2f} u={d['u']:.2f} v={d['v']:.2f}"
        draw.text((cx - half, cy - half - 14), label, fill=color)
        draw.text((cx - half, cy + half + 2), status, fill=color)

    draw.text((4, 4), f"Frame #{frame_idx}  {len(detections)} hand(s)", fill=(200, 200, 200))
    return np.array(img)


def create_hand_landmarker():
    """创建 HandLandmarker 实例（VIDEO 模式）。"""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,                          # 诊断时检测所有手（含误检）
            min_hand_detection_confidence=0.3,    # 低阈值，让误检也显出来
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.2))


def run_static(paths: list[str]) -> None:
    """对静态图片文件逐一检测。"""
    print(f"[模型] {HAND_MODEL_PATH}")
    print(f"[输出] {OUT_DIR}")
    try:
        hand_lm = create_hand_landmarker()
    except Exception as e:
        print(f"❌ HandLandmarker 加载失败: {e}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, path in enumerate(paths):
        try:
            img = Image.open(path).convert("RGB")
            rgb = np.array(img)
        except Exception as e:
            print(f"❌ 无法读取 {path}: {e}"); continue

        ts_ms = (i + 1) * 100
        dets = detect_once(hand_lm, rgb, ts_ms)
        print(f"\n=== {os.path.basename(path)} ({rgb.shape[1]}×{rgb.shape[0]}) ===")
        if not dets:
            print("  无检测")
        for j, d in enumerate(dets):
            ok, status = gate_status(d)
            print(f"  手#{j}: u={d['u']:.3f} v={d['v']:.3f} size={d['size']:.3f} "
                  f"score={d['score']:.3f} [{d['label']}]  → {status}")

        ann = annotate_image(rgb, dets, i)
        out_path = os.path.join(OUT_DIR, f"hand_diag_{i:03d}_{os.path.basename(path)}")
        Image.fromarray(ann).save(out_path, "JPEG", quality=88)
        print(f"  标注图已存: {out_path}")

    print("\n完成。")


def run_realtime(max_seconds: float = 120.0, save_every: int = 30) -> None:
    """从 Reachy Mini 相机实时检测，每隔 save_every 帧保存一张标注图。"""
    try:
        from reachy_mini import ReachyMini
    except ImportError:
        print("❌ reachy_mini 不可导入，请在机器人环境运行，或用静态图模式: python _hand_model_diag.py <img.jpg>")
        return

    print(f"[模型] {HAND_MODEL_PATH}")
    print(f"[输出] {OUT_DIR}  (每 {save_every} 帧保存一张)")
    print(f"[时限] {max_seconds:.0f}s  Ctrl+C 可提前停止")
    print("──────────────────────────────────────────────────────")

    try:
        hand_lm = create_hand_landmarker()
    except Exception as e:
        print(f"❌ HandLandmarker 加载失败: {e}"); return

    os.makedirs(OUT_DIR, exist_ok=True)
    DECIMATE = 3
    frame_idx = 0
    saved_idx = 0
    pass_count = hit_count = bot_filter_count = 0
    t_start = time.monotonic()
    last_hand_ts = -1

    with ReachyMini(connection_mode="localhost_only", media_backend="default",
                    automatic_body_yaw=False) as mini:
        mini.media.start_recording()
        # 预热
        print("预热 3s...", end="", flush=True)
        t_w = time.monotonic()
        while time.monotonic() - t_w < 3.0:
            mini.media.get_frame(); time.sleep(0.05)
        print(" 开始检测")

        try:
            while time.monotonic() - t_start < max_seconds:
                raw = mini.media.get_frame()
                if raw is None:
                    time.sleep(0.02); continue
                rgb = np.ascontiguousarray(raw[::DECIMATE, ::DECIMATE, ::-1])
                frame_idx += 1
                ts_ms = max(last_hand_ts + 1, int((time.monotonic() - t_start) * 1000))
                last_hand_ts = ts_ms

                dets = detect_once(hand_lm, rgb, ts_ms)
                hit_count += len(dets)
                for d in dets:
                    ok, status = gate_status(d)
                    if ok:
                        pass_count += 1
                    elif d["v"] > PLAY_HAND_V_MAX:
                        bot_filter_count += 1
                    sym = "✅" if ok else ("🚫" if d["v"] > PLAY_HAND_V_MAX else "⚠️")
                    print(f"{ts()} {sym} u={d['u']:.2f} v={d['v']:.2f} "
                          f"sz={d['size']:.2f} sc={d['score']:.2f} [{d['label']}]  {status}")

                if frame_idx % save_every == 0:
                    ann = annotate_image(rgb, dets, frame_idx)
                    out_path = os.path.join(OUT_DIR, f"hand_diag_{saved_idx:03d}.jpg")
                    Image.fromarray(ann).save(out_path, "JPEG", quality=85)
                    saved_idx += 1
                    fps = frame_idx / (time.monotonic() - t_start)
                    print(f"{ts()} 📸 保存 {out_path}  fps={fps:.1f} "
                          f"累计: 检出{hit_count} 通过{pass_count} 底部过滤{bot_filter_count}")

        except KeyboardInterrupt:
            print("\nCtrl+C 停止")
        finally:
            mini.media.stop_recording()

    elapsed = time.monotonic() - t_start
    fps = frame_idx / max(elapsed, 0.1)
    print(f"\n=== 统计 ({elapsed:.1f}s, {frame_idx}帧, {fps:.1f}fps) ===")
    print(f"  总检测次数: {hit_count}")
    print(f"  通过所有门: {pass_count} ({100*pass_count/max(hit_count,1):.0f}%)")
    print(f"  被底部过滤(v>{PLAY_HAND_V_MAX}): {bot_filter_count} ({100*bot_filter_count/max(hit_count,1):.0f}%)")
    print(f"  标注图已存入: {OUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_static(sys.argv[1:])
    else:
        run_realtime()
