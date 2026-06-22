# -*- coding: utf-8 -*-
"""身份识别模块 — YuNet 人脸检测 + arcface 特征提取 + 特征库匹配。

用法(独立测试):
  python identity.py                    # 摄像头实时识别
  python identity.py --list             # 列出已知人脸
  python identity.py --reset            # 清空特征库

集成到 d01:
  identity_q: vision_worker 检出人脸后扔 (t, rgb, face_box, face_kps)
  识别线程异步跑, 结果写 st.current_identity / st.identity_name
"""

import json
import os
import time
import uuid
from typing import Optional

import cv2
import numpy as np

# ── 模型路径 ──
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YUNET_PATH = os.path.join(_REPO, "models", "face_detection_yunet_2023mar.onnx")
_ARCFACE_PATH = os.path.join(_REPO, "models", "w600k_mbf.onnx")
_DB_PATH = os.path.join(_REPO, "data", "face_db.json")

# ── 匹配阈值 ──
COSINE_THRESHOLD = 0.35
MAX_EMBEDDINGS_PER_PERSON = 10
IDENTITY_COOLDOWN_S = 2.0
MIN_FACE_PX = 60
NEW_PERSON_CONFIRM_FRAMES = 3

# ── arcface 输入标准 ──
_ARC_SIZE = 112
_ARC_MEAN = 127.5
_ARC_STD = 127.5

# arcface 标准目标关键点(112x112 图上的 5 点坐标)
_ARC_REF_POINTS = np.array([
    [38.2946, 51.6963],   # 右眼
    [73.5318, 51.5014],   # 左眼
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 右嘴角
    [70.7299, 92.2041],   # 左嘴角
], dtype=np.float32)


def _align_face(rgb: np.ndarray, kps: list[tuple[float, float]]) -> np.ndarray:
    """用 5 关键点仿射对齐到 112×112 arcface 标准。"""
    src = np.array(kps, dtype=np.float32)
    M = cv2.estimateAffinePartial2D(src, _ARC_REF_POINTS)[0]
    if M is None:
        cx = int(np.mean([k[0] for k in kps]))
        cy = int(np.mean([k[1] for k in kps]))
        half = 56
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        crop = rgb[y0:y0 + 112, x0:x0 + 112]
        if crop.shape[0] != 112 or crop.shape[1] != 112:
            crop = cv2.resize(crop, (112, 112))
        return crop
    return cv2.warpAffine(rgb, M, (112, 112))


def _crop_face(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """无关键点时，从 bbox 裁剪并 resize 到 112×112。"""
    x, y, w, h = box
    margin = int(max(w, h) * 0.15)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(rgb.shape[1], x + w + margin)
    y1 = min(rgb.shape[0], y + h + margin)
    crop = rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (112, 112))


class ArcFaceONNX:
    """arcface embedding 提取器(onnxruntime)。"""

    def __init__(self, model_path: str = _ARCFACE_PATH):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, face_112: np.ndarray) -> np.ndarray:
        """输入 112×112 RGB → 输出 L2 归一化的 512d embedding。"""
        img = face_112.astype(np.float32)
        img = (img - _ARC_MEAN) / _ARC_STD
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # (1,3,112,112)
        out = self.session.run(None, {self.input_name: img})[0][0]
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out


class FaceDB:
    """人脸特征库 — JSON 文件持久化。"""

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        self.persons: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.db_path) and os.path.getsize(self.db_path) > 0:
            with open(self.db_path, "r") as f:
                self.persons = json.load(f)
        else:
            self.persons = {}

    def _save(self):
        tmp = self.db_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.persons, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.db_path)

    def match(self, embedding: np.ndarray) -> tuple[Optional[str], float]:
        """匹配最相似的人。返回 (person_id, similarity) 或 (None, 0)。"""
        best_id = None
        best_sim = -1.0
        emb = np.array(embedding, dtype=np.float32)
        for pid, info in self.persons.items():
            for stored in info.get("embeddings", []):
                sim = float(np.dot(emb, np.array(stored, dtype=np.float32)))
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
        if best_sim >= COSINE_THRESHOLD:
            return best_id, best_sim
        return None, best_sim

    def add_person(self, embedding: np.ndarray, name: str = None) -> str:
        """创建新人。返回 person_id。"""
        pid = f"person_{uuid.uuid4().hex[:8]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.persons[pid] = {
            "name": name,
            "embeddings": [embedding.tolist()],
            "facts": [],
            "created_at": now,
            "last_seen_at": now,
        }
        self._save()
        return pid

    def update_embedding(self, person_id: str, embedding: np.ndarray):
        """追加新 embedding（不同角度），只收高质量的。"""
        info = self.persons.get(person_id)
        if not info:
            return
        embs = info.get("embeddings", [])
        sims = [float(np.dot(embedding, np.array(s, dtype=np.float32))) for s in embs]
        max_sim = max(sims) if sims else 0.0
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        if max_sim > 0.85:
            return
        if avg_sim < COSINE_THRESHOLD:
            return
        embs.append(embedding.tolist())
        if len(embs) > MAX_EMBEDDINGS_PER_PERSON:
            embs.pop(0)
        info["last_seen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save()

    def set_name(self, person_id: str, name: str):
        if person_id in self.persons:
            self.persons[person_id]["name"] = name
            self._save()

    def get_name(self, person_id: str) -> Optional[str]:
        info = self.persons.get(person_id)
        return info.get("name") if info else None

    def get_info(self, person_id: str) -> Optional[dict]:
        return self.persons.get(person_id)

    def list_persons(self) -> list[dict]:
        result = []
        for pid, info in self.persons.items():
            result.append({
                "id": pid,
                "name": info.get("name"),
                "n_embeddings": len(info.get("embeddings", [])),
                "created_at": info.get("created_at"),
                "last_seen_at": info.get("last_seen_at"),
            })
        return result

    def clear_person(self, person_id: str):
        if person_id in self.persons:
            del self.persons[person_id]
            self._save()

    def reset(self):
        self.persons = {}
        self._save()


class IdentityRecognizer:
    """组合 YuNet + ArcFace + FaceDB 的高层接口。"""

    def __init__(self, db_path: str = _DB_PATH):
        self.arcface = ArcFaceONNX()
        self.db = FaceDB(db_path)
        self._last_id = None
        self._last_t = 0.0
        self._pending_new: dict[str, list[np.ndarray]] = {}

    def _face_key(self, box: tuple[int, int, int, int]) -> str:
        cx, cy = box[0] + box[2] // 2, box[1] + box[3] // 2
        return f"{cx // 40}_{cy // 40}"

    def recognize(self, rgb: np.ndarray,
                  face_box: tuple[int, int, int, int],
                  face_kps: list[tuple[float, float]] = None,
                  det_score: float = 1.0
                  ) -> tuple[Optional[str], Optional[str], float, bool]:
        """识别一张脸。

        Returns:
            (person_id, name, similarity, is_new)
            person_id=None 表示人脸太小或正在确认中
        """
        x, y, w, h = face_box
        if w < MIN_FACE_PX or h < MIN_FACE_PX:
            return None, None, 0.0, False

        if face_kps and len(face_kps) == 5:
            aligned = _align_face(rgb, face_kps)
        else:
            aligned = _crop_face(rgb, face_box)

        embedding = self.arcface.get_embedding(aligned)

        pid, sim = self.db.match(embedding)
        is_new = False
        if pid is not None:
            self.db.update_embedding(pid, embedding)
            self._pending_new.clear()
        else:
            fk = self._face_key(face_box)
            pending = self._pending_new.get(fk, [])
            pending.append(embedding)
            self._pending_new[fk] = pending

            if len(pending) >= NEW_PERSON_CONFIRM_FRAMES:
                avg_emb = np.mean(pending, axis=0)
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
                re_pid, re_sim = self.db.match(avg_emb)
                if re_pid is not None:
                    pid = re_pid
                    sim = re_sim
                    self.db.update_embedding(pid, avg_emb)
                else:
                    pid = self.db.add_person(avg_emb)
                    is_new = True
                    sim = 1.0
                del self._pending_new[fk]
            else:
                return None, None, 0.0, False

        name = self.db.get_name(pid)
        self._last_id = pid
        self._last_t = time.monotonic()
        return pid, name, sim, is_new

    def detect_and_recognize(self, rgb: np.ndarray
                             ) -> Optional[tuple[str, Optional[str], float, bool, tuple]]:
        """从完整帧中检测最大人脸并识别。返回 (pid, name, sim, is_new, face_box) 或 None。"""
        results = self.detect_and_recognize_all(rgb)
        if not results:
            return None
        areas = [r[4][2] * r[4][3] for r in results]
        best = int(np.argmax(areas))
        return results[best]

    def detect_and_recognize_all(self, rgb: np.ndarray
                                 ) -> list[tuple[str, Optional[str], float, bool, tuple, list]]:
        """检测并识别所有人脸。返回 [(pid, name, sim, is_new, face_box, face_kps), ...]。"""
        if not os.path.exists(_YUNET_PATH):
            return []
        h, w = rgb.shape[:2]
        yunet = cv2.FaceDetectorYN.create(_YUNET_PATH, "", (w, h), 0.65, 0.3, 10)
        _, faces = yunet.detect(rgb)
        if faces is None or len(faces) == 0:
            return []
        results = []
        for f in faces:
            box = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
            kps = [(float(f[4 + i * 2]), float(f[5 + i * 2])) for i in range(5)]
            pid, name, sim, is_new = self.recognize(rgb, box, kps)
            results.append((pid, name, sim, is_new, box, kps))
        return results


# ── CLI 测试入口 ──
def _main():
    import argparse
    parser = argparse.ArgumentParser(description="身份识别测试工具")
    parser.add_argument("--list", action="store_true", help="列出已知人脸")
    parser.add_argument("--reset", action="store_true", help="清空特征库")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID")
    parser.add_argument("--image", type=str, help="识别单张图片")
    parser.add_argument("--db", type=str, default=_DB_PATH, help="特征库路径")
    args = parser.parse_args()

    db = FaceDB(args.db)

    if args.reset:
        confirm = input("确定清空所有人脸数据? (y/N): ")
        if confirm.lower() == "y":
            db.reset()
            print("已清空。")
        return

    if args.list:
        persons = db.list_persons()
        if not persons:
            print("特征库为空。")
            return
        print(f"共 {len(persons)} 人：")
        for p in persons:
            name_s = p["name"] or "(未命名)"
            print(f"  {p['id']}  {name_s}  "
                  f"embeddings={p['n_embeddings']}  "
                  f"last_seen={p['last_seen_at']}")
        return

    try:
        rec = IdentityRecognizer(args.db)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确认 models/ 下有 face_detection_yunet_2023mar.onnx 和 w600k_mbf.onnx")
        return

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"无法读取图片: {args.image}")
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = rec.detect_and_recognize(rgb)
        if result is None:
            print("未检测到人脸。")
        else:
            pid, name, sim, is_new, box = result
            tag = "新建" if is_new else "匹配"
            name_s = name or "(未命名)"
            print(f"[{tag}] {pid}  name={name_s}  sim={sim:.3f}  box={box}")
        return

    # ── 摄像头实时模式 ──
    print(f"打开摄像头 {args.camera}，按 q 退出，按 n 给当前人命名，按 r 重新识别")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    _COLORS_BGR = [
        (0, 128, 255),   # 橙
        (255, 0, 0),     # 蓝
        (0, 200, 0),     # 绿
        (0, 255, 255),   # 黄
        (255, 0, 255),   # 品红
        (255, 255, 0),   # 青
    ]
    _pid_color_map: dict[str, tuple] = {}

    def _color_for_pid(pid: str) -> tuple:
        if pid not in _pid_color_map:
            _pid_color_map[pid] = _COLORS_BGR[len(_pid_color_map) % len(_COLORS_BGR)]
        return _pid_color_map[pid]

    yunet = cv2.FaceDetectorYN.create(_YUNET_PATH, "", (320, 240), 0.65, 0.3, 10)
    cached_ids: list[tuple] = []  # [(cx, cy, pid, name, sim, is_new), ...]
    frame_n = 0
    identify_every = 15
    last_named_pid = None

    def _match_cached(cx: float, cy: float) -> tuple | None:
        best, best_d = None, 80.0
        for c in cached_ids:
            d = ((cx - c[0]) ** 2 + (cy - c[1]) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best = c
        return best

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yunet.setInputSize((w, h))
        _, faces = yunet.detect(rgb)

        if faces is not None and len(faces) > 0:
            do_identify = (frame_n % identify_every == 1)
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
                    if pid is not None:
                        new_cache.append((cx, cy, pid, name, sim, is_new))
                        last_named_pid = pid
                        color = _color_for_pid(pid)
                        tag = "NEW" if is_new else "OK"
                        name_s = name or pid[:12]
                        label = f"{tag} {name_s} ({sim:.2f})"
                    else:
                        color = (128, 128, 128)
                        label = "identifying..."
                else:
                    hit = _match_cached(cx, cy)
                    if hit:
                        _, _, pid, name, sim, is_new = hit
                        color = _color_for_pid(pid)
                        tag = "NEW" if is_new else "OK"
                        name_s = name or pid[:12]
                        label = f"{tag} {name_s} ({sim:.2f})"
                    else:
                        color = (0, 128, 255)
                        label = None

                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
                if label:
                    cv2.putText(frame, label, (bx, by - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if do_identify:
                cached_ids = new_cache
        else:
            cached_ids.clear()

        n_persons = len(rec.db.persons)
        n_faces = len(faces) if faces is not None else 0
        cv2.putText(frame, f"DB: {n_persons} persons | faces: {n_faces} | frame {frame_n}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        cv2.imshow("Identity Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n") and last_named_pid:
            cv2.destroyWindow("Identity Test")
            name = input(f"给 {last_named_pid} 起名: ").strip()
            if name:
                rec.db.set_name(last_named_pid, name)
                print(f"已命名: {last_named_pid} → {name}")
        elif key == ord("r"):
            frame_n = 0
            cached_ids.clear()
            print("强制重新识别")

    cap.release()
    cv2.destroyAllWindows()

    print("\n最终特征库：")
    for p in rec.db.list_persons():
        name_s = p["name"] or "(未命名)"
        print(f"  {p['id']}  {name_s}  embeddings={p['n_embeddings']}")


if __name__ == "__main__":
    _main()
