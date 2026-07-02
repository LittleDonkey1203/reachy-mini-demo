# -*- coding: utf-8 -*-
"""Reachy Mini × Qwen3.5-Omni-Realtime 语音对话(D-01+O-01a+V-01+F-01+FUSION-03+PLAY-01:完整体)。

⭐ 架构地图(四能力边界/线程清单/状态机图/仲裁/改码铁律)见同目录 ARCHITECTURE.md;
   实测参数与踩坑沿革见 ../CALIBRATION.md §6-§13。四个核心能力:
   ① 对话(main上行 + ChatCallback + player/motion/snapshot)② 头部跟踪(vision_* TRACKING积分)
   ③ 听声转向(doa_sensor + behavior ENGAGING)④ 指向理解(两段式 judge→point)

对着机器人说话 → Qwen 全双工识别并生成语音 → 从机器人扬声器播放;
说话时可随时插话打断(barge-in);模型自主调用动作工具做身体语言
(点头/摇头/看向/摆天线/歪头),可边说边动;说话时有 idle 微动;
让它"看"时调 take_snapshot 抓当前画面 → chat.completions 看图 → 语音转述;
本地视觉(MediaPipe)持续看脸,聊天时头温和地跟着人转(F-01 融合);
在视场外(背后/侧后)叫它 → DOA 声源转向 → 人脸进视野 → 视觉接管(FUSION-02);
手凑近逗它 → 像猫被逗猫棒吸引:开心地跟着手走,手离开回到跟脸(PLAY-01)。

五层动作仲裁(PLAY-01 更新,优先级从高到低,头部唯一 set_target 写入口 head_control_loop):
  Primary  明确手势(function_call)/ 指向转头(POINTING):motion goto_target
           独占或 behavior 驱动,其他层让位;手势以"当前跟随姿态"为基准做,
           做完回基准 → 无缝接管不突跳。
  Playing  逗它跟手(PLAYING):近处**晃动的**大手(score≥0.6 + size≥0.30 + 0.8s 内
           位移≥0.08;托下巴的静止手不算)持续 0.3s → 注意力被手吸引,头灵敏跟手
           (τ=0.25/步进3.0/幅度0.9,standalone 调校);手静止 4s = 没意思 → 回跟脸;
           开心表达克制:持续逗 5s 后才第一次摇天线、之后每 ~7s 小摇(进入不动天线,
           短暂中断重入不重置节拍);近手消失 1.5s → 回跟脸/待命。
  SoundTurn 声源转向(事件性,DOA REST 10Hz):丢脸 且 DOA 残差>25°(视场外
           有人说话)→ 闭环链式转向(头给世界系完整角 + 身体分担,匀速 ramp,
           每步同步状态);转向中看到脸立即中止交还视觉;最多 3 跳(后方镜像角
           逐跳收缩,天然可达背后),仍无脸则放弃进冷却。有脸在跟时绝不抢。
  Tracking 人脸跟随:视觉线程积分 track_yaw(头的世界朝向目标,限身体±23°);
           ⭐ 增益必须时间常数型 step=err×(1−exp(−dt/τ))(CALIBRATION §9);
           丢脸衰减回"身体正前"而不是世界 0°。
  Idle     说话微动:跟随之上小幅叠加(跟随时缩到 40%,不打架)。

⭐ 参考系(CALIBRATION §11 教训):head pose 是世界系;body_yaw 转动被 Stewart
反向补偿,头世界朝向不变 → 大角度转向 head 给完整目标角,body_yaw 只是分担量。

音频链路(实测验证,见 ../CALIBRATION.md §6):
  上行:麦克风 16kHz 原生 → 取 audio[:,0] → int16 → base64(零重采样)
  下行:24kHz PCM16 → resample_poly 24k→16k → 抖动缓冲 ~300ms → push_audio_sample
  打断:speech_started → 队列代际作废 + audio.clear_player() + 必要时 cancel_response

get_frame 协调(F-01):视觉线程是唯一持续抓帧者,最新帧共享在 State;
take_snapshot 直接读共享帧(≤25ms 新),不再和跟随抢 get_frame。

运行(需 daemon 已启动、DASHSCOPE_API_KEY 已配):
  $env:PYTHONUTF8=1
  & "C:\\Users\\ldkji\\AppData\\Local\\Reachy Mini Control\\.venv\\Scripts\\python.exe" voice\\d01_realtime_chat.py [秒数]
可选参数 [秒数]:到时自动干净退出(编排测试用);不带参数则 Ctrl+C 退出。
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# repo root → sys.path，让 perception/identity/memory 包可导入
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# 加载 demo 目录的 .env（voice/ 向上一级 = reachy-mini-demo/）
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

# ── 代理隔离:必须在 import reachy_mini / dashscope 之前 ──
_no_proxy = "localhost,127.0.0.1,::1,.aliyuncs.com,aliyuncs.com"
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

import base64
import io
import json
import collections
import math
import multiprocessing
import queue
import re
import sys
import threading
import time
import random
from collections import deque
from datetime import datetime, timezone

import cv2  # noqa: E402,F401  必须在 numpy 之前 import:规避 Windows spawn 下 cv2/numpy 循环 import 崩溃
import numpy as np
from PIL import Image
import pytweening

import dashscope
from openai import OpenAI
from reachy_mini import ReachyMini

# MediaPipe 不在主进程导入(TRACK-FIX):检测在 vision_worker 子进程跑,独立 GIL。
from perception.vision_worker import vision_worker as _vision_worker_fn

from identity.recognizer import IdentityRecognizer, IDENTITY_COOLDOWN_S
from identity.owner import OwnerManager
from memory.manager import MemoryManager, QWEN_TOOLS
from perception.face_pipeline import FaceReIDPipeline
from perception.face_config import FaceSystemConfig
from perception.asd import AsdEngine

from voice.config import (                          # ← 配置常量集中管理
    MODEL, VISION_MODEL, VISION_BASE_URL, VOICE, INSTRUCTIONS,
    SNAP_DIR, _MODELS_DIR, VIS_MODEL_PATH, HAND_MODEL_PATH, GESTURE_MODEL_PATH,
    _DATA_DIR, PROFILE_PATH, MEMORY_PATH,
    OUT_SR, PLAY_SR,
    IDLE_HZ, IDLE_YAW_AMP, IDLE_PITCH_AMP, IDLE_YAW_F, IDLE_PITCH_F, IDLE_TAU,
    TRACK_SWAY_SCALE,
    VIS_MAX_FPS, VIS_MISS_N, DECIMATE, FOV_X_DEG, FOV_Y_DEG,
    TRACK_TAU, TRACK_DEADBAND, TRACK_MAX_STEP, TRACK_YAW_LIMIT, TRACK_PITCH_LIMIT,
    HEAD_ASD_EMA, HEAD_SPK_ON, HEAD_SWITCH_MARGIN,
    DOA_GLANCE_DEG, GLANCE_MAX_DEG, GLANCE_MIN_HOLD_S, FPS_FREEZE_BELOW, ASD_MAX_TRACKS,
    GLANCE_TIMEOUT_S, GLANCE_COOLDOWN_S, GLANCE_SPEECH_GRACE_S,
    GLANCE_MIN_TURN_DEG, GLANCE_LOCAL_RMS,
    GROUP_FRAMING_ENABLED, GROUP_MIN_H, GROUP_EDGE_MARGIN, GROUP_FIT_SPAN,
    LOST_HOLD_S, RETURN_TAU, YAW_SIGN, PITCH_SIGN,
    SND_RESID_MIN, SND_DONE_RESID,
    DOA_DEBUG, GATE_DEG, DOA_GATE_FRESH_S,
    SWITCH_COOLDOWN_S, SWITCH_AWAY_DEG, SWITCH_SETTLE, SWITCH_TIMEOUT_S, SWITCH_COARSE_DEG,
    SND_FACE_FRESH_S, SND_MAX_HOPS, SND_WAIT_FACE_S, SND_COOLDOWN_S,
    SND_SPEED_DPS, SND_TARGET_LIMIT, BODY_LIMIT_DEG, NECK_REL_LIMIT,
    BODY_FOLLOW_THRESHOLD, BODY_FOLLOW_SPEED_DPS,
    CLEAR_VERIFY_COUNT, CLEAR_VERIFY_SIM, CLEAR_TIMEOUT_S,
    ST_ARMED, ST_IDLE, ST_ENGAGING, ST_TRACKING, ST_SEARCHING, ST_RETURNING,
    ST_POINTING, ST_PLAYING,
    FACE_FRESH_S, LOCK_WIN, LOCK_ON_RATE, LOCK_OFF_RATE,
    ENGAGE_TIMEOUT_S, ENGAGE_SCAN_RANGE, ENGAGE_SCAN_TIME_S,
    SEARCH_TIMEOUT_S, NO_INTERACT_S, FSM_HZ,
    KWS_SINGLE_THR, CONNECT_TIMEOUT_S,
    ARMED_BREATH_F, ARMED_BREATH_PITCH, CUE,
    SPREAD_BAD, WIDE_SCAN_RANGE, WIDE_SCAN_HZ, WIDE_SCAN_TIME_S,
    DOA_WAKE_FRESH_S, SEEK_PITCH_UP, SEEK_PITCH_AMP, SEEK_PITCH_HZ,
    SEEK_NEARBY_DEG, SEEK_NEARBY_TIME_S, SEEK_SUPPRESS_DEG,
    GREET_PHRASES, EXIT_MIN_S, EXIT_MAX_S,
    POINT_FRESH_S, POINT_YAW_GAIN, POINT_PITCH_GAIN,
    POINT_TURN_TIMEOUT_S, POINT_SETTLE_S, POINT_HOLD_MAX_S,
    PLAY_SIZE_ON, PLAY_SIZE_OFF, PLAY_SCORE_MIN, PLAY_HAND_V_MAX,
    PLAY_ON_S, PLAY_OFF_S, PLAY_FRESH_S,
    PLAY_MOVE_WIN_S, PLAY_MOVE_MIN, PLAY_STILL_S,
    PLAY_TAU, PLAY_MAX_STEP, PLAY_AMP, PLAY_YAW_LIMIT, PLAY_PITCH_LIMIT,
    PLAY_COAST_S, PLAY_COAST_DU, PLAY_COAST_VEL,
    PLAY_JOY_DELAY_S, PLAY_JOY_PERIOD_S, PLAY_JOY_FLICK_S, PLAY_REENTRY_S,
    EASE_ATTACK_FRAC, CUE_VARIATION,
    THINK_ROLL_AMP, THINK_ROLL_F, THINK_PITCH, THINK_ANT_AMP, THINK_ANT_F,
    THINK_BLEND_TAU,
    EXPR_SMILE_ANT, EXPR_FROWN_ANT, EXPR_BLEND_TAU,
    AUDIO_GATE_TIMEOUT_S,
    _NOPARAM, BASE_TOOLS, SNAP_PROMPTS, _DIR_MAP,
    greet_prompt, parse_judge,
)
from voice.state import (                           # ← 共享状态 + 日志 + 录制
    State, OneEuroFilter, log,
    _vis_log_buf, _vis_log_seq,
    _conv_events, _conv_turns, _conv_seq,
    _feedback_notes, _feedback_seq,
    _record_event, _record_snap_result, _record_vis_event,
)
import voice.state as _st_mod

_id_recognizer: IdentityRecognizer | None = None
_memory_mgr: MemoryManager | None = None
_owner_mgr: OwnerManager | None = None
_face_pipeline = None   # FaceReIDPipeline(ByteTrack + 三区间);main() 初始化
_asd_engine = None      # AsdEngine(LR-ASD 谁在说话);main() 初始化

# 仿真模式摄像头源切换:USE_WEBCAM=1 时 frame_pump_loop 从 Mac 摄像头取帧,
# 绕过 MuJoCo 虚拟摄像头(空棋盘格场景无人脸,人脸检测永远失败)。
# 用 --sim 启动 daemon 时在 .env 设 USE_WEBCAM=1;真机时不需要。
USE_WEBCAM = os.environ.get("USE_WEBCAM", "").lower() in ("1", "true", "yes")

# NO_VOICE=1:跳过麦克风/扬声器/Qwen 连接,只跑视觉跟随 + 行为状态机,便于单独调试人脸跟踪/指向/逗它。
NO_VOICE = os.environ.get("NO_VOICE", "").lower() in ("1", "true", "yes")

# VIS_DEBUG=1：启动 MJPEG HTTP 服务，浏览器实时查看视觉子进程实际处理的帧 + 检测结果标注。
# 头部照常运动。打开 http://localhost:VIS_DEBUG_PORT 即可。
VIS_DEBUG = os.environ.get("VIS_DEBUG", "").lower() in ("1", "true", "yes")
VIS_DEBUG_PORT = int(os.environ.get("VIS_DEBUG_PORT", "7654"))

# ───────────────────────── 组合工具列表 ─────────────────────────
TOOLS = BASE_TOOLS + QWEN_TOOLS

from voice.actions import (
    INIT_HEAD_POSE, INIT_ANTENNAS, head_pose, gpose,
    act_nod, act_shake, _look, act_wiggle, act_tilt, ACTIONS,
)
from voice.audio import doa_sensor_loop, _fresh_sound, player_loop
from voice.realtime import RealtimeDialog
from voice.kws import KwsGate
from voice.asr_stream import AsrStream, TurnAggregator   # ASR 级联 shadow(env ASR_SHADOW=1,阶段1 只打日志)
from memory.safety import inject_clear_msg
from perception.fusion import select_face_by_doa

# ───────────────────────── M3-c 记忆:读写(旧单用户启动数据,仅 main 用)─────────────────────────
def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)

def load_profile() -> dict | None:
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def load_memories() -> list:
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("items", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []



# ───────────────────────── 动作线程:Primary 手势,串行执行 ─────────────────────────
def motion_loop(mini: ReachyMini, st: State, motion_q: "queue.Queue", stop: threading.Event) -> None:
    """只管执行手势(function_call_output 已在 ws 线程即时回过)。
    F-01:进场读跟随基准 → 手势相对基准做、做完回基准;期间 head_control/视觉积分让位。"""
    while not stop.is_set():
        try:
            job = motion_q.get(timeout=0.1)
        except queue.Empty:
            continue
        name = job["name"]
        fn = ACTIONS.get(name)
        with st.lock:
            st.action_active = True  # SoundTurn/Tracking/Idle 让位
            by, bp = st.track_yaw, st.track_pitch
            body = st.body_yaw_deg   # ⭐ 手势期间身体保持当前朝向(传 0 会拽回正前)
        try:
            if fn is None:
                log(f"⚠ 未知动作 {name}")
            else:
                fn(mini, by, bp, body)
                log(f"✅ 动作完成: {name}(基准 yaw={by:+.1f}° pitch={bp:+.1f}° body={body:+.1f}°,跟随恢复)")
        except Exception as e:
            log(f"⚠ 动作 {name} 执行失败:{type(e).__name__}: {e}")
        finally:
            with st.lock:
                st.action_active = False


# ───────────────── ①对话+④指向:快照线程,共享帧 → Qwen-VL → 回结果(judge 轮可升级转头)─────────────────
def snapshot_loop(mini: ReachyMini, st: State, cb: "ChatCallback", oai: OpenAI,
                  snap_q: "queue.Queue", stop: threading.Event) -> None:
    """take_snapshot:优先读视觉线程共享的最新帧(≤25ms 新,不抢 get_frame);
    视觉线程没帧时退回直接抓。→ 640×360 jpg → chat.completions 看图
    → 描述作为 function_call_output 回 Realtime → response.create 让模型语音转述。"""
    os.makedirs(SNAP_DIR, exist_ok=True)
    snap_idx = 0
    while not stop.is_set():
        try:
            job = snap_q.get(timeout=0.1)
        except queue.Empty:
            continue
        call_id, gen0 = job["call_id"], job["gen"]
        mode = job.get("mode", "scene")
        snap_idx += 1
        t0 = time.monotonic()
        _label = {"point": "指向理解", "judge": "指向判断", "scene": "场景描述"}.get(mode, mode)
        log(f"📸 拍照:取当前画面…({_label})")
        with st.lock:
            frame = st.latest_frame
            fresh = (time.monotonic() - st.latest_frame_t) < 1.0
        if frame is None or not fresh:
            frame = None  # 视觉线程未供帧 → 退回直接抓(连抓取最新,防旧帧)
            got = 0
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and got < 3:
                f = mini.media.get_frame()
                if f is not None:
                    frame = f
                    got += 1
                else:
                    time.sleep(0.02)
        desc = ""
        ok = frame is not None
        if not ok:
            desc = "拍照失败,没有抓到画面。"
            log("❌ 没有可用画面帧")
        else:
            img = Image.fromarray(frame[:, :, ::-1]).resize((640, 360))  # BGR→RGB,降采样
            img.save(os.path.join(SNAP_DIR, f"snapshot_{snap_idx:02d}.jpg"), "JPEG", quality=85)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            jpg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            with st.lock:
                st.snap_grabbed = True  # 帧已落盘内存 → 通知 POINTING 可以转回了
            log(f"📸 取到帧并压缩({len(buf.getvalue()) / 1024:.0f}KB),送看图…")
            try:
                comp = oai.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{jpg_b64}"}},
                        {"type": "text", "text": SNAP_PROMPTS[mode]},
                    ]}],
                    stream=True,  # omni 必须流式
                    stream_options={"include_usage": True},
                    extra_body={"modalities": ["text"]},
                )
                parts = []
                for chunk in comp:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                desc = "".join(parts).strip()
                log(f"🖼 图像理解({(time.monotonic() - t0) * 1000:.0f}ms 全程):「{desc}」")
            except Exception as e:
                ok = False
                desc = f"看图服务调用失败:{type(e).__name__}"
                log(f"❌ chat.completions 失败:{type(e).__name__}: {e}")

        # ── judge 分流(两段式指向):确认在指且目标不在画面 → 升级转头,本轮不回话 ──
        if mode == "judge" and ok:
            jd = parse_judge(desc)
            if jd is None:
                # 解析失败:别把 JSON 念给用户 → 退化为普通场景描述语句
                if desc.lstrip().startswith("{"):
                    desc = "我看了一眼,不过没看太清,你可以再问我一次。"
            else:
                pointing = bool(jd.get("pointing"))
                visible = bool(jd.get("target_visible"))
                direction = str(jd.get("direction") or "无")
                if pointing and not visible and direction in _DIR_MAP:
                    with st.lock:
                        st.point_request = {"call_id": call_id, "gen": gen0, "dir": direction}
                    log(f"👉 VLM 确认在指、目标不在画面(方向:{direction})→ 升级转头重取景")
                    continue  # 不回 output、不减 pending;转头后第二轮(mode=point)收尾
                desc = str(jd.get("desc") or "").strip() or "我看了一眼,没发现你在指什么特别的东西。"
                log(f"👉 指向判断:pointing={pointing} visible={visible} → 原地回答")
        fire_rc = False
        with st.lock:
            st.snapshot_pending = max(0, st.snapshot_pending - 1)
            fire_rc = st.play_gen == gen0  # 期间被打断则不补话
        try:
            _record_snap_result(call_id, mode, desc, ok)  # 对话可视化录制
            cb.conv.create_item({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"success": ok, "scene_description": desc}, ensure_ascii=False),
            })
        except Exception as e:
            log(f"⚠ 回 function_call_output 失败:{e}")
            continue
        if fire_rc:
            try:
                cb.conv.create_response()  # 让模型用语音转述所见
            except Exception as e:
                log(f"⚠ response.create 失败:{e}")


# ───────── ②跟踪+④指向+逗它:视觉(TRACK-FIX)抓帧泵(主进程)+ MediaPipe 子进程 + 结果积分 ─────────
def frame_pump_loop(mini: ReachyMini, st: State, frame_q, stop: threading.Event) -> None:
    """轻量抓帧泵:唯一持续 get_frame 者。最新帧共享给 take_snapshot;
    降采样(numpy 抽样 ~1ms)喂视觉子进程,maxsize=1 背压只留最新帧。
    USE_WEBCAM=1 时用 Mac 摄像头代替 MuJoCo 虚拟摄像头(仿真模式需真实画面做人脸检测)。
    重活(MediaPipe 12ms/帧)在子进程独立 GIL 跑,不再饿主进程。"""
    import cv2 as _cv2
    cap = None
    if USE_WEBCAM:
        cap = _cv2.VideoCapture(0)
        cap.set(_cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, 720)
        log("📷 USE_WEBCAM=1 → 使用 Mac 摄像头(绕过 MuJoCo 虚拟摄像头)")
    t_last = 0.0
    try:
        while not stop.is_set():
            if cap is not None:
                ret, frame = cap.read()  # BGR(与 get_frame() 格式一致)
                if not ret:
                    frame = None
            else:
                frame = mini.media.get_frame()
            now = time.monotonic()
            if frame is None:
                time.sleep(0.005)
                continue
            with st.lock:
                st.latest_frame = frame
                st.latest_frame_t = now
            if now - t_last < 1.0 / VIS_MAX_FPS:
                continue
            t_last = now
            # stride 抽点(近零开销;INTER_AREA 每帧全分辨率缩放会吃满 CPU 拖慢 SCRFD 推理)
            # 识别精度靠 DECIMATE=2(分辨率比 3 高)+ 全分辨率 ROI 重检(后续)保证
            rgb = np.ascontiguousarray(frame[::DECIMATE, ::DECIMATE, ::-1])  # BGR→RGB
            if VIS_DEBUG:
                with st.lock:
                    st.dbg_frame_small = rgb.copy()
            try:
                frame_q.put_nowait((now, rgb))
            except Exception:
                try:  # 队列满:丢旧换新(检测只该吃最新帧)
                    frame_q.get_nowait()
                    frame_q.put_nowait((now, rgb))
                except Exception:
                    pass
    finally:
        if cap is not None:
            cap.release()


def _make_roi_detector():
    """主进程惰性 SCRFD(只用于识别路径的全分辨率 ROI 重检,与子进程跟踪检测解耦)。
    失败返回 None(embedder 退化为用跟踪给的粗 kps)。"""
    try:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name="buffalo_sc", allowed_modules=["detection"],
                            providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.5)  # ROI 小,320 足够且快
        det = _app.models.get("detection") or getattr(_app, "det_model", None)
        log("🔬 识别 ROI 重检器就绪(全分辨率 sharp kps)")
        return det
    except Exception as e:
        log(f"⚠ ROI 重检器初始化失败({type(e).__name__}),识别退化用跟踪粗 kps")
        return None


def _roi_redetect_kps(detector, full_rgb, box_xywh):
    """在全分辨率帧上对人脸 box 的扩展 ROI 重跑 SCRFD,拿亚像素 sharp kps(全分辨率坐标)。
    返回 [(x,y)*5] 或 None。"""
    try:
        H, W = full_rgb.shape[:2]
        x, y, w, h = box_xywh
        if w <= 0 or h <= 0:
            return None
        m = 0.4                                   # ROI 外扩,确保整脸在内
        x0 = max(0, int(x - m * w)); y0 = max(0, int(y - m * h))
        x1 = min(W, int(x + w + m * w)); y1 = min(H, int(y + h + m * h))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        roi_bgr = np.ascontiguousarray(full_rgb[y0:y1, x0:x1, ::-1])  # RGB→BGR
        bboxes, kpss = detector.detect(roi_bgr, max_num=1, metric="default")
        if kpss is None or len(kpss) == 0:
            return None
        kp = kpss[0]                              # ROI 坐标 → 偏回全分辨率
        return [(float(kp[i][0] + x0), float(kp[i][1] + y0)) for i in range(5)]
    except Exception:
        return None


def _make_face_embedder(rec, roi_detector=None):
    """给 FaceReIDPipeline 注入 ArcFace 提特征器:全分辨率帧 → (ROI 重检 sharp kps 优先,
    否则跟踪粗 kps)→ 5点对齐 → 512d L2。方案B:识别精度不受跟踪降采样影响。"""
    from identity.recognizer import _align_face, _crop_face

    def _embed(full_rgb, box_xywh, kps):
        try:
            sharp = _roi_redetect_kps(roi_detector, full_rgb, box_xywh) if roi_detector is not None else None
            use_kps = sharp if sharp is not None else kps
            if use_kps and len(use_kps) == 5:
                aligned = _align_face(full_rgb, use_kps)
            else:
                aligned = _crop_face(full_rgb, box_xywh)
            return rec.arcface.get_embedding(aligned)
        except Exception:
            return None
    return _embed


def vision_result_loop(st: State, result_q, stop: threading.Event,
                       cb_ref: list = None) -> None:
    """消费视觉子进程结果 → 时间常数型积分跟随目标(逻辑同 F-01,数据源改进程队列)。
    丢脸缓冲(1c):连续 VIS_MISS_N 帧漏检才重置滤波/进入丢脸路径,防侧脸闪断。"""
    fx = OneEuroFilter(min_cutoff=0.8, beta=0.08)
    fy = OneEuroFilter(min_cutoff=0.8, beta=0.08)
    hx = OneEuroFilter(min_cutoff=0.8, beta=0.25)  # 跟手对(PLAY-01):beta 高 → 快手低延迟
    hy = OneEuroFilter(min_cutoff=0.8, beta=0.25)
    t_prev_ctrl = time.monotonic()
    t_prev_play = time.monotonic()
    last_hu = last_hv = None       # 最近滤波后手位置 + One Euro 速度(惯性外推用)
    hvel_u = hvel_v = 0.0
    hand_win: collections.deque = collections.deque()  # (t,u,v) 晃动量统计窗
    miss_streak = 0
    hit_window: collections.deque = collections.deque(maxlen=LOCK_WIN)
    locked = False
    # 身份/切换/限频逻辑已迁至 FaceReIDPipeline(ByteTrack + 三区间 + EMA),此处不再维护
    n_det = 0
    n_hit = 0
    infer_acc: list[float] = []
    stat_t = time.monotonic()
    _asd_log_t = time.monotonic()       # ASD 分周期打印节流
    _head_ema: dict = {}               # 头部专用二级平滑分(身份键→EMA),只管"头看谁"
    _head_key = None                   # 头部当前跟随的身份(person_id 或 t{track_id}),黏滞抗 churn
    # DOA 声音转头(glance #2)已移除(2026-06-30):侧方噪声不再引发转头;改纯视觉跟随 + 喊「小艺」转向。
    _last_frame_t = 0.0               # 上一帧到达时刻(算瞬时 fps)
    _fps_ema = 20.0                   # 检测 fps 的 EMA(低 fps 时冻结相机运动,断 churn 死循环)
    _lowfps_log_t = 0.0              # 低 fps 冻结日志节流

    while not stop.is_set():
        try:
            msg = result_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if msg.get("kind") == "ready":
            log("👁 视觉子进程就绪(Face 跟随 + Hand 指向,独立 GIL)")
            with st.lock:
                st.vis_ready = True
            continue
        now = time.monotonic()
        # 瞬时 fps EMA(检测帧到达率):低 fps 时冻结身体跟随/瞟头,阻断"相机甩→churn→fps更低"死循环
        if _last_frame_t > 0.0:
            _dtf = now - _last_frame_t
            if _dtf > 1e-3:
                _fps_ema = 0.3 * (1.0 / _dtf) + 0.7 * _fps_ema
        _last_frame_t = now
        _cam_ok = _fps_ema >= FPS_FREEZE_BELOW   # 相机运动(身体跟随/瞟头)许可:fps 够才动
        if not _cam_ok and now - _lowfps_log_t >= 2.0:
            _lowfps_log_t = now
            log(f"🧊 低 fps({_fps_ema:.1f})→ 冻结身体跟随/瞟头,等跟踪恢复(断 churn 死循环)")

        # ── UI 注册请求:显式给指定 track 命名(绕过"谁在说话"归属,等价于语音"我叫X")──
        with st.lock:
            _reg = st.register_request
            st.register_request = None
        if _reg and _face_pipeline is not None:
            _rtid, _rname = _reg.get("track_id"), _reg.get("name")
            if _face_pipeline.name_track(_rtid, _rname):
                _rtrk = _face_pipeline.tracker.find_track(_rtid)
                _rpid = _rtrk.identity_id if _rtrk else None
                _face_pipeline.save_gallery()
                if _rpid and _memory_mgr is not None:
                    _memory_mgr.set_name(_rpid, _rname)
                if _rpid and _owner_mgr is not None and not _owner_mgr.has_owner():
                    _owner_mgr.try_claim(_rpid, _rname)
                with st.lock:
                    st.register_result = f"✅ track {_rtid} → {_rname}(已落盘)"
                log(f"🏷 UI 注册: track {_rtid} → {_rname}(gallery 已落盘)")
            else:
                with st.lock:
                    st.register_result = f"⚠ track {_rtid} 注册失败(track 不存在或无特征)"
                log(f"⚠ UI 注册失败: track {_rtid}(track 不存在或无特征)")

        face = msg.get("face")
        u_raw, v_raw = None, None   # 由 FaceReIDPipeline 的 primary 提供(下方)
        infer_ms = msg.get("face_ms", 0.0)
        n_det += 1
        infer_acc.append(infer_ms)
        # 人脸检测结果(供 FaceReIDPipeline)+ 多人脸 DOA 选说话人
        face_box = msg.get("face_box")
        face_kps = msg.get("face_kps")
        all_faces = msg.get("all_faces")
        _doa_selected_idx = None
        _track_views = None        # 每 track 视图(身份+trackid),供 dashboard 每框绘制
        if all_faces and len(all_faces) > 1:
            with st.lock:
                _doa_r = st.doa_resid_stable
                _doa_c = st.doa_confident
                _doa_ty = st.track_yaw
                _doa_by = st.body_yaw_deg
            if _doa_c and _doa_r is not None:
                _doa_selected_idx = select_face_by_doa(
                    all_faces, _doa_r, _doa_ty, _doa_by)
                if _doa_selected_idx is not None:
                    _sel = all_faces[_doa_selected_idx]
                    face_box = _sel["box"]
                    face_kps = _sel.get("kps")

        # ── ByteTrack + 全分辨率 ArcFace + 三区间身份(FaceReIDPipeline)──
        if _face_pipeline is not None and all_faces is not None:
            with st.lock:
                _raw_frame = st.latest_frame
            if _raw_frame is not None:
                try:
                    _full_rgb = np.ascontiguousarray(_raw_frame[:, :, ::-1])  # 全分辨率 BGR→RGB
                    _H0, _W0 = _raw_frame.shape[:2]
                    _dw, _dh = _W0 // DECIMATE, _H0 // DECIMATE   # 与 frame_pump 的 resize 尺寸精确一致
                    _primary, _track_views = _face_pipeline.process(
                        all_faces, (_dw, _dh), _full_rgb, DECIMATE, now, _doa_selected_idx)
                    # ── ASD:按「身份键」聚合喂帧(churn 换 id 也喂同一缓冲,治新人画外)。
                    #    ①只喂最大的 ASD_MAX_TRACKS 个 track(churn 出一堆 ghost 时逐个裁剪+打分会爆 CPU/GPU→fps崩);
                    #    ②fps 低(_cam_ok False)直接跳过 ASD,把资源让给检测先恢复(治长时间低 fps)。
                    if _asd_engine is not None and _asd_engine.available and _track_views and _cam_ok:
                        _feed = sorted(_track_views,
                                       key=lambda v: (v.bbox_px[2] - v.bbox_px[0]) * (v.bbox_px[3] - v.bbox_px[1]),
                                       reverse=True)[:ASD_MAX_TRACKS]
                        for _v in _feed:
                            _b = _v.bbox_px
                            _bbox_full = (_b[0] * DECIMATE, _b[1] * DECIMATE,
                                          _b[2] * DECIMATE, _b[3] * DECIMATE)
                            _k = _v.person_id or f"t{_v.track_id}"
                            _asd_engine.feed_crop(_k, _full_rgb, _bbox_full, now, track_id=_v.track_id)
                        _asd_engine.gc([(_v.person_id or f"t{_v.track_id}") for _v in _feed])

                    # ── 头部目标 + 发言归属:ASD 说话人优先,否则回退 ──
                    _asd_on = _asd_engine is not None and _asd_engine.available
                    _spk = _asd_engine.speaker() if _asd_on else None
                    _scs_all = _asd_engine.scores() if _asd_on else {}
                    if _asd_on and now - _asd_log_t >= 3.0:    # 周期打印 ASD 分(便于无 dashboard 验证)
                        _asd_log_t = now
                        if _scs_all:
                            _spk_k = _spk[0] if _spk else None
                            log("🗣 ASD: " + "  ".join(
                                f"{'▶' if k == _spk_k else ' '}T{_asd_engine.last_track(k) or '?'}={s:+.2f}"
                                for k, s in sorted(_scs_all.items(), key=lambda kv: str(kv[0]))))
                    # 维护"画面内 ASD 说话人"(供 user query 归属;None=画外/无人说话)。_spk[0]=身份键
                    _asp_set = None
                    if _spk is not None and _track_views:
                        _sv = next((v for v in _track_views
                                    if (v.person_id or f"t{v.track_id}") == _spk[0]), None)
                        if _sv is not None:
                            _asp_set = {"pid": _sv.person_id, "name": _sv.person_name,
                                        "track_id": _sv.track_id, "score": float(_spk[1]), "at": now}
                    if _asp_set is not None:    # 有画面内说话人才更新;否则保持(读端按时间窗判 stale)
                        with st.lock:
                            st.asd_speaker = _asp_set
                    # ── 头部"看谁" + 当前人:都按"身份(person_id)"黏滞,抗 track churn ──
                    #    治:①无人说话时头回退最大脸帧间翻=晃;②current_person 随瞬时 ASD 抖动反复
                    #    update_session 重注入→竞态→"归属对但回复叫错人"。归属/记忆保存仍走 speaker_window(敏感)。
                    _key2view = {}                            # 身份键 → 该身份当前 track 视图(首个即可,同身份分数相同)
                    for _v in (_track_views or []):
                        _k = _v.person_id or f"t{_v.track_id}"
                        if _k not in _key2view:
                            _key2view[_k] = _v
                    for _k, _v in _key2view.items():          # 二级重 EMA(更平滑 ASD),按身份累积
                        _inst = _scs_all.get(_k, -1.0)        # 分数已按身份键聚合
                        _pe = _head_ema.get(_k)
                        _head_ema[_k] = _inst if _pe is None else (HEAD_ASD_EMA * _inst + (1.0 - HEAD_ASD_EMA) * _pe)
                    for _gk in [k for k in _head_ema if k not in _key2view]:
                        _head_ema.pop(_gk, None)              # 清掉离场身份
                    _roster = []                              # 在场身份名单 [(pid, name)] → create_item 注入 roster
                    for _rk, _rv in _key2view.items():
                        if _rv.person_id:                     # 只收已识别身份(t{track} 未识别的不进 roster)
                            _rn = ((_memory_mgr.get_name(_rv.person_id) if _memory_mgr is not None else None)
                                   or _rv.person_name)
                            _roster.append((_rv.person_id, _rn))
                    with st.lock:
                        st.present_count = len(_key2view)     # B:供记忆注入判"多人在场"
                        st.roster = _roster
                    if _head_key is not None and _head_key not in _key2view:
                        _head_key = None                      # 跟随对象离场才释放(churn 换 track 不算,身份还在)
                    _cand = max(_head_ema.items(), key=lambda kv: kv[1]) if _head_ema else None
                    if _cand is not None and _cand[1] > HEAD_SPK_ON:       # 有人明确说话才考虑切
                        if _head_key is None or (_cand[0] != _head_key and
                                _cand[1] > _head_ema.get(_head_key, -9.9) + HEAD_SWITCH_MARGIN):
                            _head_key = _cand[0]              # 切到明确说话人(需高出当前余量,防来回甩)
                    if _head_key is None and _primary is not None:
                        _head_key = _primary.person_id or f"t{_primary.track_id}"   # 无人说话且无跟随→锁最大脸(黏滞)
                    _head_view = _key2view.get(_head_key) if _head_key is not None else None
                    if _head_view is None:
                        _head_view = _primary
                    if _head_view is not None:
                        u_raw, v_raw = _head_view.u, _head_view.v
                    # ── 多脸 keep-in-frame:空闲(无人明确说话)+ ≥2 张"够大"的脸 → 瞄准点从"只看主角"
                    #    改成"兜全场",夹住保证主角(_head_view)不出框;兜不住(跨度>FOV)→ 锚主角(远端靠喊「小艺」)。
                    #    有人说话→不进此块,照常看说话人;单人/无合格脸→落回上面的看主角。
                    if (GROUP_FRAMING_ENABLED and _head_view is not None
                            and not (_cand is not None and _cand[1] > HEAD_SPK_ON)):
                        _qual_u = [vw.u for vw in _key2view.values() if vw.h >= GROUP_MIN_H]
                        if len(_qual_u) >= 2:
                            _umin, _umax = min(_qual_u), max(_qual_u)
                            _u_aim = (0.5 * (_umin + _umax) if (_umax - _umin) <= GROUP_FIT_SPAN
                                      else _head_view.u)          # 装得下→居中整组;兜不住→锚主角
                            _keep = 0.5 - GROUP_EDGE_MARGIN
                            u_raw = float(np.clip(_u_aim, _head_view.u - _keep, _head_view.u + _keep))
                            # v 仍跟主角(group framing 只管左右)
                    # 当前人 = 稳定头部焦点(仅供显示/greet/owner)。记忆注入不挂在它身上——
                    # 焦点随对话在多人间切换是正常的,但注入只认「本句说话人」(realtime transcription),
                    # 否则焦点一翻就 update_session 重注入 → 时序竞态 → 回复用错人记忆(治问题2)。
                    if _head_view is not None and _head_view.person_id is not None:
                        _pid_now = _head_view.person_id
                        _pname = _head_view.person_name      # 焦点名每帧从实时库现取(改名立即跟上)
                        if _memory_mgr is not None:
                            _mn = _memory_mgr.get_name(_pid_now)
                            if _mn:
                                _pname = _mn
                        with st.lock:
                            old_pid = st.current_person_id
                            old_name = st.current_person_name
                            st.current_person_id = _pid_now
                            st.current_person_name = _pname
                            st.current_is_owner = (_owner_mgr.is_owner(_pid_now) if _owner_mgr else False)
                        if _pid_now != old_pid or _pname != old_name:   # 切人或改名才记日志,避免刷屏
                            log(f"🆔 {_pname or _pid_now[:12]} (track {_head_view.track_id}/{_head_view.zone}, 焦点)")
                except Exception as e:
                    log(f"⚠ 人脸 pipeline 异常:{e}")

        # ── 安全删除工作流:仅工作流激活时单独识别取 (pid, sim) 驱动高阈值验证 ──
        with st.lock:
            _cwf = st.clear_workflow
        if _cwf is not None and face_box is not None and _face_pipeline is not None:
            with st.lock:
                _raw_frame = st.latest_frame
            if _raw_frame is not None:
                try:
                    _full_rgb2 = np.ascontiguousarray(_raw_frame[:, :, ::-1])
                    _bxf = tuple(int(c * DECIMATE) for c in face_box)
                    _kpf = ([(x * DECIMATE, y * DECIMATE) for x, y in face_kps]
                            if face_kps else None)
                    # 删除验证走与主路径同一身份空间(gallery identity_id)
                    _emb_v = _face_pipeline.embedder(_full_rgb2, _bxf, _kpf)
                    _mr = _face_pipeline.store.match(_emb_v) if _emb_v is not None else None
                    pid = _mr.identity_id if (_mr and _mr.zone == "known") else None
                    sim = _mr.confidence if _mr else 0.0
                    if pid is not None:
                        _cwf_phase = _cwf.get("phase")
                        if _cwf_phase == "verifying":
                            if pid == _cwf["actor_pid"] and sim >= CLEAR_VERIFY_SIM:
                                _cwf["stable_count"] += 1
                            else:
                                _cwf["stable_count"] = 0
                            if _cwf["stable_count"] >= CLEAR_VERIFY_COUNT:
                                # 验证通过 → 权限检查
                                _actor = _cwf["actor_pid"]
                                _target = _cwf["target_pid"]
                                if _actor != _target:
                                    if not (_owner_mgr and _owner_mgr.can_delete_memory(_actor, _target)):
                                        with st.lock:
                                            st.clear_workflow = None
                                            st.clear_lock = False
                                        log("🔒 权限不足,非主人不能删他人记忆")
                                        if cb_ref[0] is not None and cb_ref[0].conv is not None:
                                            inject_clear_msg(cb_ref[0].conv,
                                                "权限不足：只有主人才能删除其他人的记忆。删除流程已取消,请告诉用户。")
                                    else:
                                        _cwf["phase"] = "confirming"
                                        _cwf["verified_at"] = time.monotonic()
                                        _tdesc = _cwf["target_name"] or "你"
                                        log(f"🔒 身份验证通过,进入确认阶段(target={_tdesc})")
                                        if cb_ref[0] is not None and cb_ref[0].conv is not None:
                                            inject_clear_msg(cb_ref[0].conv,
                                                f"身份已验证。请向用户做最后确认：'你确定要我忘掉关于{_tdesc}的所有记忆吗？"
                                                f"包括人脸和所有信息都将被清除,此操作不可恢复。'"
                                                f"等用户明确回答后,调用confirm_clear(confirmed=true或false)。")
                                else:
                                    _cwf["phase"] = "confirming"
                                    _cwf["verified_at"] = time.monotonic()
                                    log("🔒 身份验证通过(删除自己),进入确认阶段")
                                    if cb_ref[0] is not None and cb_ref[0].conv is not None:
                                        inject_clear_msg(cb_ref[0].conv,
                                            "身份已验证。请向用户做最后确认：'你确定要我忘掉关于你的所有记忆吗？"
                                            "包括你的脸和所有信息都将被清除,此操作不可恢复。'"
                                            "等用户明确回答后,调用confirm_clear(confirmed=true或false)。")
                            elif (now - _cwf["started_at"]) > CLEAR_TIMEOUT_S:
                                with st.lock:
                                    st.clear_workflow = None
                                    st.clear_lock = False
                                log("🔒 身份验证超时,取消删除")
                                if cb_ref[0] is not None and cb_ref[0].conv is not None:
                                    inject_clear_msg(cb_ref[0].conv,
                                        "身份验证超时(30秒内未能稳定识别),删除流程已取消。请告诉用户。")
                        elif _cwf_phase == "confirming":
                            if _cwf.get("verified_at") and (now - _cwf["verified_at"]) > CLEAR_TIMEOUT_S:
                                with st.lock:
                                    st.clear_workflow = None
                                    st.clear_lock = False
                                log("🔒 确认超时,取消删除")
                                if cb_ref[0] is not None and cb_ref[0].conv is not None:
                                    inject_clear_msg(cb_ref[0].conv,
                                        "等待确认超时(30秒无回应),删除流程已取消。请告诉用户。")
                except Exception as e:
                    log(f"⚠ 身份识别异常:{e}")
        # 手部结果(平时降频/近手提频):发布食指方向(指向)+ 近手读数(逗它)
        hand = msg.get("hand")
        hand_near = False
        if hand is not None:
            _hv = hand.get("v", 0.5)
            _valid_pos = _hv <= PLAY_HAND_V_MAX   # 底部区域(桌面/衣物)误检过滤
            hand_near = (_valid_pos
                         and hand.get("score", 1.0) >= PLAY_SCORE_MIN
                         and hand.get("size", 0.0) >= PLAY_SIZE_OFF)  # 双门:背景误检不入
            with st.lock:
                if _valid_pos and hand.get("score", 1.0) >= PLAY_SCORE_MIN:  # 低分/底部假手不发布指向
                    st.finger_angle = hand["angle"]
                    st.finger_extended = hand["extended"]
                    st.finger_at = now
                    if hand["extended"]:
                        st.finger_ext_at = now
                if hand_near:
                    st.hand_u = hand["u"]
                    st.hand_v = hand["v"]
                    st.hand_size = hand["size"]
                    st.hand_at = now
                    # 晃动量:窗口内位移极差("晃"才是逗;托下巴的静止手不触发)
                    hand_win.append((now, hand["u"], hand["v"]))
                    while hand_win and now - hand_win[0][0] > PLAY_MOVE_WIN_S:
                        hand_win.popleft()
                    if len(hand_win) >= 3:
                        us = [p[1] for p in hand_win]
                        vs = [p[2] for p in hand_win]
                        st.hand_move = max(max(us) - min(us), max(vs) - min(vs))
                    else:
                        st.hand_move = 0.0
                    # 手势字段（GESTURE-01）：fingers=-1 表示跳帧，不更新
                    fingers = hand.get("fingers", -1)
                    gesture = hand.get("gesture")
                    if fingers >= 0:
                        st.gesture_fingers = fingers
                        if gesture:
                            st.gesture = gesture
                            st.gesture_at = now
        with st.lock:
            # 视觉只在 TRACKING 且无手势时积分头部目标;其余状态只感知(face_seen_at),
            # 头部目标由 behavior_loop 驱动(避免双写 track_yaw)。
            integrate = (st.state == ST_TRACKING) and (not st.action_active)
            play_integrate = (st.state == ST_PLAYING) and (not st.action_active)
            h_at = st.hand_at

        # ── PLAYING:跟手积分(灵敏档 τ/步进/幅度;丢检 ≤0.35s 惯性外推防愣住)──
        def steer_hand(tu: float, tv: float) -> None:
            nonlocal t_prev_play
            dt_p = max(1e-3, now - t_prev_play)
            t_prev_play = now
            ey = YAW_SIGN * (tu - 0.5) * FOV_X_DEG * PLAY_AMP
            ep = PITCH_SIGN * (tv - 0.5) * FOV_Y_DEG * PLAY_AMP
            if abs(ey) < TRACK_DEADBAND:
                ey = 0.0
            if abs(ep) < TRACK_DEADBAND:
                ep = 0.0
            k_p = 1.0 - math.exp(-dt_p / PLAY_TAU)
            with st.lock:
                sy = float(np.clip(k_p * ey, -PLAY_MAX_STEP, PLAY_MAX_STEP))
                sp = float(np.clip(k_p * ep, -PLAY_MAX_STEP, PLAY_MAX_STEP))
                st.track_yaw = float(np.clip(st.track_yaw + sy,
                                             st.body_yaw_deg - PLAY_YAW_LIMIT,
                                             st.body_yaw_deg + PLAY_YAW_LIMIT))
                st.track_pitch = float(np.clip(st.track_pitch + sp,
                                               -PLAY_PITCH_LIMIT, PLAY_PITCH_LIMIT))

        if play_integrate:
            if hand_near:
                hu = hx(hand["u"], now)
                hv = hy(hand["v"], now)
                last_hu, last_hv = hu, hv
                hvel_u, hvel_v = hx.dx_prev, hy.dx_prev
                steer_hand(hu, hv)
            elif last_hu is not None and (now - h_at) <= PLAY_COAST_S:
                age = now - h_at  # 惯性外推:封顶小步,不许飞到画面边(standalone 教训)
                cu = float(np.clip(np.clip(hvel_u, -PLAY_COAST_VEL, PLAY_COAST_VEL) * age,
                                   -PLAY_COAST_DU, PLAY_COAST_DU))
                cv = float(np.clip(np.clip(hvel_v, -PLAY_COAST_VEL, PLAY_COAST_VEL) * age,
                                   -PLAY_COAST_DU, PLAY_COAST_DU))
                steer_hand(min(1.0, max(0.0, last_hu + cu)), min(1.0, max(0.0, last_hv + cv)))
            else:
                t_prev_play = now
        elif last_hu is not None:
            hx.reset()
            hy.reset()
            last_hu = None
            t_prev_play = now
        if u_raw is not None:
            n_hit += 1
            miss_streak = 0
            hit_window.append(1)
            _rate = sum(hit_window) / len(hit_window) if hit_window else 0.0
            if not locked and len(hit_window) >= 3 and _rate >= LOCK_ON_RATE:
                locked = True
                _record_vis_event("vis.face_locked", "🔒 人脸锁定",
                                  {"u": round(u_raw, 3), "v": round(v_raw, 3),
                                   "rate": round(_rate, 2)})
            u = fx(u_raw, now)
            v = fy(v_raw, now)
            with st.lock:
                st.face_seen_at = now
                st.face_locked = locked
            if integrate:
                # 看人说话转过去 = 纯视觉跟随(#3);声音转头(DOA glance #2)已移除(2026-06-30):
                # 侧方噪声不再引发转头。画面外要它注意请喊「小艺」(唤醒 + DOA 转向仍在,见 behavior/二次唤醒)。
                err_yaw = YAW_SIGN * (u - 0.5) * FOV_X_DEG
                err_pitch = PITCH_SIGN * (v - 0.5) * FOV_Y_DEG
                if abs(err_yaw) < TRACK_DEADBAND:
                    err_yaw = 0.0
                if abs(err_pitch) < TRACK_DEADBAND:
                    err_pitch = 0.0
                dt = max(1e-3, now - t_prev_ctrl)
                t_prev_ctrl = now
                k = 1.0 - math.exp(-dt / TRACK_TAU)
                with st.lock:
                    sy = float(np.clip(k * err_yaw, -TRACK_MAX_STEP, TRACK_MAX_STEP))
                    sp = float(np.clip(k * err_pitch, -TRACK_MAX_STEP, TRACK_MAX_STEP))
                    st.track_yaw = float(np.clip(st.track_yaw + sy,
                                                 st.body_yaw_deg - NECK_REL_LIMIT,
                                                 st.body_yaw_deg + NECK_REL_LIMIT))
                    st.track_pitch = float(np.clip(st.track_pitch + sp, -TRACK_PITCH_LIMIT, TRACK_PITCH_LIMIT))
                    # 身体跟随:头偏到颈限阈值时身体跟着转,把人脸保持在中心
                    # 低 fps 时冻结身体(身体甩动=最大相机自运动→运动模糊→track churn→fps 更低,死循环)
                    neck_off = st.track_yaw - st.body_yaw_deg
                    threshold = NECK_REL_LIMIT * BODY_FOLLOW_THRESHOLD
                    if abs(neck_off) > threshold and _cam_ok:
                        body_step = BODY_FOLLOW_SPEED_DPS * dt
                        body_move = math.copysign(min(body_step, abs(neck_off) - threshold), neck_off)
                        st.body_yaw_deg = float(np.clip(st.body_yaw_deg + body_move,
                                                        -BODY_LIMIT_DEG, BODY_LIMIT_DEG))
                        # 身体转了,颈限范围跟着扩,头可以继续追
                        st.track_yaw = float(np.clip(st.track_yaw,
                                                     st.body_yaw_deg - NECK_REL_LIMIT,
                                                     st.body_yaw_deg + NECK_REL_LIMIT))
            else:
                t_prev_ctrl = now
        else:
            miss_streak += 1
            t_prev_ctrl = now
            hit_window.append(0)
            _rate = sum(hit_window) / len(hit_window) if hit_window else 0.0
            if locked and len(hit_window) >= LOCK_WIN and _rate < LOCK_OFF_RATE:
                locked = False
                with st.lock:
                    st.face_locked = False
                _record_vis_event("vis.face_lost", "🔓 人脸丢失",
                                  {"rate": round(_rate, 2)})
            if miss_streak >= VIS_MISS_N:
                fx.reset()
                fy.reset()

        # M3-b 表情:读取 blendshape 微笑/皱眉
        if "smile" in msg:
            with st.lock:
                st.user_smile = msg["smile"]
                st.user_frown = msg.get("frown", 0.0)

        if VIS_DEBUG:
            # 每 track 视图:真实框(降采样像素)+ 身份(Unknown-N/真名)+ trackid + 是否选中
            _tv = []
            if _track_views:
                _asd_on2 = _asd_engine is not None and _asd_engine.available
                _scs2 = _asd_engine.scores() if _asd_on2 else {}
                _spk_ids = _asd_engine.speaking_ids() if _asd_on2 else set()
                for v in _track_views:
                    _vk = v.person_id or f"t{v.track_id}"       # 身份键(与 ASD 聚合一致)
                    _asc = _scs2.get(_vk)
                    # 显示名每帧从实时记忆库现取(改名立即跟上,治缓存滞后);取不到再退回 pipeline 缓存名
                    _live_name = ((_memory_mgr.get_name(v.person_id) if (v.person_id and _memory_mgr is not None) else None)
                                  or v.person_name)
                    _tv.append({
                        "box": v.bbox_px,           # [x1,y1,x2,y2] 降采样像素
                        "track_id": v.track_id,
                        "name": _live_name,         # 实时库名 / Unknown-N / None(未绑定)
                        "zone": v.zone,
                        "confirmed": v.is_confirmed,
                        "selected": (_vk == _head_key),          # = 头部跟随对象(供注册面板高亮)
                        "asd": round(_asc, 2) if _asc is not None else None,
                        "speaking": (_vk in _spk_ids),           # 确信在说话:>阈值且新鲜(治残留绿)
                    })
            with st.lock:
                st.dbg_det = {
                    "face": msg.get("face"),
                    "face_box": face_box,   # DOA 选中后的真实像素框(降采样系),供单脸贴合绘制
                    "hand": msg.get("hand"),
                    "n_faces": msg.get("n_faces", 0),
                    "all_faces": all_faces,
                    "doa_selected_idx": _doa_selected_idx,
                    "track_views": _tv,     # 方案B显示:每框身份+trackid
                }

        if now - stat_t >= 10.0:
            fps = n_det / (now - stat_t)
            avg_inf = float(np.mean(infer_acc)) if infer_acc else 0.0
            hit = 100.0 * n_hit / max(1, n_det)
            with st.lock:
                ty, tp, sname = st.track_yaw, st.track_pitch, st.state
            log(f"👁 视觉:检测 {fps:.1f}fps|推理均值 {avg_inf:.1f}ms|检出率 {hit:.0f}%|"
                f"[{sname}] 头目标 yaw={ty:+.1f}° pitch={tp:+.1f}°")
            stat_t = now
            n_det = 0
            n_hit = 0
            infer_acc = []


from voice.debug_server import vis_debug_server


# ──────────── 状态机(②③④+逗它 的调度大脑;状态图见 ARCHITECTURE.md §3)────────────
def behavior_loop(st: State, snap_q: "queue.Queue", stop: threading.Event,
                  wake_mode: bool = True) -> None:
    """唯一的状态调度者。在非 TRACKING 态驱动 track_yaw/body/pitch(head_control 渲染);
    TRACKING 态把头部目标交给视觉积分,自己只做状态切换。手势(action_active)永远优先,
    behavior 在手势期间暂停驱动。状态:IDLE→ENGAGING→TRACKING↔SEARCHING→RETURNING→IDLE;
    指向请求(POINT-02)→ POINTING(转头朝手指方向)→ snapshot → 回 TRACKING。"""
    dt = 1.0 / FSM_HZ
    step = SND_SPEED_DPS * dt           # 每帧最大转角
    phase_t = time.monotonic()          # 当前状态进入时刻
    scan_dir = 1.0
    pt_yaw_goal = pt_body_goal = pt_pitch_goal = 0.0  # POINTING 目标
    pt_phase = "turn"                                  # POINTING 子阶段
    pt_settle_t = pt_hold_t = 0.0
    _vis_wait_logged = False

    def set_state(s: str, seed_interact: bool = False) -> None:
        nonlocal phase_t
        with st.lock:
            if st.state != s:
                prev = st.state
                log(f"🧭 状态:{prev} → {s}")
                st.state = s
                _record_vis_event("vis.state", f"🧭 {prev} → {s}",
                                  {"from": prev, "to": s})
            # Bug2 修:只在"首次捕获"(IDLE/ENGAGING/RETURNING→TRACKING)播种无互动计时;
            # SEARCHING↔TRACKING 的短暂回切不重置(否则抖动让 15s 永远清零)
            if seed_interact:
                st.last_interaction_at = time.monotonic()
        phase_t = time.monotonic()

    def approach(ty_goal: float, body_goal: float, tp_goal: float) -> bool:
        """把 track/body/pitch 朝目标各走一步(匀速);返回是否已到位。"""
        with st.lock:
            ty, b, tp = st.track_yaw, st.body_yaw_deg, st.track_pitch
            def mv(cur, goal):
                d = goal - cur
                return goal if abs(d) <= step else cur + math.copysign(step, d)
            st.track_yaw = mv(ty, ty_goal)
            st.body_yaw_deg = mv(b, body_goal)
            st.track_pitch = mv(tp, tp_goal)
            done = (abs(st.track_yaw - ty_goal) < 0.5 and
                    abs(st.body_yaw_deg - body_goal) < 0.5 and
                    abs(st.track_pitch - tp_goal) < 0.5)
        return done

    engage_target = 0.0  # 本次 ENGAGING 的世界朝向目标(非 SEEK 的 in-conversation 声源转向用)
    wide_scan = False    # 本次 ENGAGING 是否走 SEEK 宽扫(唤醒寻人=True;in-conversation 声源转=False)
    seek_dir = 1.0       # SEEK 起扫方向(+1 先扫左 / -1 先扫右,由 DOA resid 符号定)
    seek_target = 0.0    # SEEK 两阶段:confident 时的直转目标角(世界系)
    seek_phase = "full"  # SEEK 两阶段:"direct"(直转)→"nearby"(附近扫)→"full"(全场扫)
    seek_nearby_t = 0.0  # "nearby" 阶段起始时刻
    greet_armed = False  # 唤醒应答:本次是"唤醒→SEEK"流程(锁脸时招呼一句);in-conversation 重锁不招呼
    exiting = False      # EXIT-01:正在退出(RETURNING 回中→等告别播完→armed,不被 locked 拉回)
    t_exit = 0.0         # 退出起点时刻(等告别播完的宽限/封顶计时)
    # M1.5-b 切换:switching=正转向新人B;转离开A途中压住认脸(turned_away 才放开),没脸转扫,超时回A
    switching = False
    switch_from = 0.0    # 切换起点(A 的世界朝向)
    switch_target = 0.0  # 切换目标角(confident 时=A方向+resid)
    switch_phase = "turn"  # turn(confident 直转)/ sweep(不确信或转过去没脸→扫)

    def _sync_switch_dbg():
        with st.lock:
            st.dbg_switching = switching
            st.dbg_switch_phase = switch_phase if switching else ""
            st.dbg_switch_target = switch_target
    sw_t = 0.0
    sw_dir = 1.0
    play_big_since = None    # 近处晃动大手持续出现的起点(PLAY-01 进入迟滞)
    play_still_since = None  # 逗它中手开始静止的时刻(静止超时退出)
    while not stop.is_set():
        time.sleep(dt)
        now = time.monotonic()
        with st.lock:
            state = st.state
            action = st.action_active
            locked = st.face_locked                 # Bug1 修:用迟滞锁定判定,不用瞬时 face_fresh
            last_interact = st.last_interaction_at
            speaking = now < st.playback_end_estimate or st.in_flight > 0
        if action:
            phase_t = now  # 手势期间状态计时冻结(手势结束后从当前态继续)
            continue

        # WAKE-01 待命态:只等唤醒(main 连接成功后置 st.wake_ok),其余一律不响应(不跟人/不转声/不逗它)
        if state == ST_ARMED:
            with st.lock:
                woke = st.wake_ok
                # 注意: 不在这里清 wake_ok — 等 set_state(ENGAGING) 后再清,
                # 避免 audio loop 在 wake_ok=False + state=ARMED 窗口误关 WS。
                if woke:
                    sr, sconf, sat = st.doa_resid_stable, st.doa_confident, st.doa_at
                    st.track_yaw = st.track_pitch = 0.0   # armed 居中,从 0 起转
            if woke:
                if not st.vis_ready:
                    if not _vis_wait_logged:
                        log("🔎 唤醒但视觉子进程未就绪,等待…")
                        _vis_wait_logged = True
                    approach(0.0, 0.0, 0.0)
                    continue
                _vis_wait_logged = False
                fresh = sr is not None and (now - sat) < DOA_WAKE_FRESH_S
                if fresh and sconf and sr is not None:
                    # 两阶段 SEEK:confident → 直转到 DOA 角度,到位后附近找脸
                    seek_target = float(np.clip(sr, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                    seek_phase = "direct"
                    seek_dir = 1.0 if sr >= 0 else -1.0
                    hint = f"confident resid {sr:+.0f}° → 直转"
                else:
                    # 不 confident / 不 fresh → 全场扫(=原有行为)
                    seek_target = 0.0
                    seek_phase = "full"
                    seek_dir = -1.0 if (fresh and sr is not None and sr < 0) else 1.0
                    hint = (f"不确信 resid {sr:+.0f}° → 全场扫"
                            if fresh and sr is not None else "无 DOA → 全场扫")
                wide_scan = True
                greet_armed = True
                log(f"🔎 唤醒 → SEEK 寻人({hint})")
                _record_vis_event("vis.seek_start", f"🔎 SEEK 寻人: {hint}", {"hint": hint})
                set_state(ST_ENGAGING, seed_interact=True)
                with st.lock:
                    st.wake_ok = False
                    st.wake_doa = None
            else:
                approach(0.0, 0.0, 0.0)                 # 缓慢保持回正(头控渲染慢呼吸)
            continue

        # EXIT-01:用户结束意图(end_session 工具置 flag)→ 回中 + 告别 cue + 回 armed。
        # 不破坏单写者:ChatCallback 只置 st.exit_request,这里由 behavior 写 st.state。
        with st.lock:
            er = st.exit_request
            if er:
                st.exit_request = False
        if er and state != ST_ARMED:
            exiting = True
            t_exit = now
            with st.lock:
                st.wake_cue = "bye"      # 收束告别动作(天线轻收,复用 cue 渲染)
                st.wake_cue_t = now
            log("👋 结束意图 → 回中 + 告别 → 回待命")
            set_state(ST_RETURNING)
            continue

        # M1.5-b 二次唤醒切换:三档方向(confident→直转 / fresh→粗方向 / 无→反向离A)
        with st.lock:
            swr = st.switch_request
            if swr is not None:
                st.switch_request = None
        if swr is not None and state not in (ST_ARMED, ST_POINTING):
            with st.lock:
                cy = st.track_yaw
            switch_from = cy
            _resid = swr.get("resid")
            _conf = swr.get("confident", False)
            _fresh = swr.get("fresh", False)
            if _conf and _fresh and _resid is not None:
                # 档一:confident → 直转到 DOA 角度(精确)
                switch_target = float(np.clip(cy + _resid, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                sw_dir = 1.0 if _resid >= 0 else -1.0
                _tier = f"confident 直转 {switch_target:+.0f}°"
            elif _fresh and _resid is not None:
                # 档二:不 confident 但有粗方向 → 用 resid 符号(+左/-右)朝"离开A且偏向B"转
                _sign = 1.0 if _resid >= 0 else -1.0
                switch_target = float(np.clip(cy + _sign * SWITCH_COARSE_DEG, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                sw_dir = _sign
                _tier = f"粗方向({'左' if _sign > 0 else '右'}) → {switch_target:+.0f}°"
            else:
                # 档三:无 DOA → 离开A朝中心方向
                _sign = -1.0 if cy >= 0 else 1.0
                switch_target = float(np.clip(cy + _sign * SWITCH_COARSE_DEG, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                sw_dir = _sign
                _tier = f"反向({'左' if _sign > 0 else '右'}) → {switch_target:+.0f}°"
            switch_phase = "turn"
            switching = True
            wide_scan = False
            greet_armed = True
            sw_t = now
            _sync_switch_dbg()
            log(f"🔀 切换({_tier}):从A(at {switch_from:+.0f}°)")
            set_state(ST_ENGAGING, seed_interact=True)
            continue

        # 指向请求(POINT-02-b):从任何态进入 POINTING,计算"手指方向→头部转角"
        with st.lock:
            preq = st.point_request
        if preq is not None and state != ST_POINTING:
            with st.lock:
                fa = st.finger_angle
                fa_fresh = (now - st.finger_at) < POINT_FRESH_S
                cy = st.track_yaw
                cp = st.track_pitch
            pdir = preq.get("dir")
            if pdir in _DIR_MAP:
                # 两段式:方向来自 VLM 看图判断(粗但可信;关键点 2D 角度噪声大会错误抬头)
                dyaw, dpitch = _DIR_MAP[pdir]
                pt_yaw_goal = float(np.clip(cy + dyaw, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                pt_pitch_goal = float(np.clip(dpitch, -TRACK_PITCH_LIMIT, TRACK_PITCH_LIMIT))
                log(f"👉 指向转头(VLM 方向:{pdir})→ yaw{pt_yaw_goal:+.0f}° pitch{pt_pitch_goal:+.0f}°")
            elif fa is not None and fa_fresh:
                ar = math.radians(fa)
                dx, dy = math.cos(ar), math.sin(ar)
                # 画面右(dx>0)= 机器人右 → yaw 负;画面下(dy>0)→ pitch 正(CALIBRATION §11 约定)
                pt_yaw_goal = float(np.clip(cy - dx * POINT_YAW_GAIN, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                pt_pitch_goal = float(np.clip(dy * POINT_PITCH_GAIN, -TRACK_PITCH_LIMIT, TRACK_PITCH_LIMIT))
                log(f"👉 指向转头(关键点兜底):食指 {fa:+.0f}° → yaw{pt_yaw_goal:+.0f}° pitch{pt_pitch_goal:+.0f}°")
            else:
                pt_yaw_goal, pt_pitch_goal = cy, cp  # 没有任何方向线索 → 原地看图(兜底)
                log("👉 未抓到指向方向,原地看图")
            pt_body_goal = float(np.clip(pt_yaw_goal, -BODY_LIMIT_DEG, BODY_LIMIT_DEG))
            pt_phase = "turn"
            set_state(ST_POINTING)
            continue

        # 逗它(PLAY-01-b):近处大手持续出现 → 注意力被手吸引,像猫看逗猫棒。
        # 优先级:手势/指向 > 逗它(上面两个 continue 先吃掉)> 声源/跟脸(下面不再判)
        if state != ST_POINTING and not exiting and not switching:   # 退出/切换途中不被挥手拉进 PLAYING
            with st.lock:
                h_fresh = (now - st.hand_at) < PLAY_FRESH_S
                h_size = st.hand_size
                h_move = st.hand_move
            if state != ST_PLAYING:
                # "晃"才是逗:近 + 大 + 在动(托下巴/扶脸的静止手不触发)
                if h_fresh and h_size >= PLAY_SIZE_ON and h_move >= PLAY_MOVE_MIN:
                    if play_big_since is None:
                        play_big_since = now
                    elif now - play_big_since >= PLAY_ON_S:
                        log(f"🎾 手凑近晃动逗它(size {h_size:.2f} move {h_move:.2f})→ PLAYING")
                        play_big_since = None
                        play_still_since = None
                        set_state(ST_PLAYING, seed_interact=True)  # 逗它也算互动
                        continue
                else:
                    play_big_since = None

        snd = _fresh_sound(st)

        if state == ST_IDLE:
            approach(0.0, 0.0, 0.0)                 # 缓慢归正
            if locked:
                set_state(ST_TRACKING, seed_interact=True)
            elif snd is not None and not speaking:   # not speaking:DOA_DEBUG 常驻时也绝不在机器人说话时转头
                with st.lock:
                    engage_target = float(np.clip(st.track_yaw + snd, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                wide_scan = False    # in-conversation 声源转向走原"转到目标+小扫",非 SEEK 宽扫
                log(f"👂 视场外有人说话(残差 {snd:+.0f}°)→ ENGAGING 朝 {engage_target:+.0f}°")
                _record_vis_event("doa.engage", f"👂 DOA 触发转向 {snd:+.0f}° → {engage_target:+.0f}°",
                                  {"resid": round(snd, 1), "target": round(engage_target, 1)})
                set_state(ST_ENGAGING)
            elif wake_mode and (now - last_interact) > NO_INTERACT_S and not speaking:
                log(f"💤 engaged 无互动 {NO_INTERACT_S:.0f}s → 回 armed 待命")
                set_state(ST_ARMED)

        elif state == ST_ENGAGING:
            if switching:
                # M1.5-b 切换三档:直转目标(turn,全程压锁)→到位附近扫(sweep,认脸)→超时回A。
                # ⭐ turn 阶段完全不认脸(A 的 FOV 在 35° 内还覆盖得到);只有 sweep 阶段 + turned_away 才认脸。
                with st.lock:
                    _sw_ty = st.track_yaw
                turned_away = abs(_sw_ty - switch_from) > SWITCH_AWAY_DEG
                if switch_phase == "sweep" and turned_away and locked:
                    log(f"🔀 锁定 B(at yaw={_sw_ty:+.0f}°, 离A {abs(_sw_ty - switch_from):.0f}°)")
                    if greet_armed:
                        with st.lock:
                            st.greet_now = True
                        greet_armed = False
                    switching = False
                    _sync_switch_dbg()
                    set_state(ST_TRACKING, seed_interact=True)
                    continue
                if (now - phase_t) > SWITCH_TIMEOUT_S:
                    log("🔀 切换没找到人 → 回 A")
                    with st.lock:
                        st.wake_cue = "giveup"
                        st.wake_cue_t = now
                    switching = False
                    _sync_switch_dbg()
                    set_state(ST_RETURNING)
                    continue
                if switch_phase == "turn":
                    bg = float(np.clip(switch_target, -BODY_LIMIT_DEG, BODY_LIMIT_DEG))
                    arrived = approach(switch_target, bg, 0.0)
                    if arrived:
                        switch_phase = "sweep"
                        sw_t = now
                        _sync_switch_dbg()
                        log(f"🔀 到位({switch_target:+.0f}°) → 附近找脸(±{SEEK_NEARBY_DEG:.0f}°)")
                else:
                    tsw = now - sw_t
                    sweep = switch_target + SEEK_NEARBY_DEG * math.sin(2 * math.pi * 0.4 * tsw) * sw_dir
                    sweep = float(np.clip(sweep, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                    bg = float(np.clip(sweep, -BODY_LIMIT_DEG, BODY_LIMIT_DEG))
                    sp = float(np.clip(SEEK_PITCH_UP + SEEK_PITCH_AMP * math.sin(2 * math.pi * SEEK_PITCH_HZ * tsw),
                                       -TRACK_PITCH_LIMIT, TRACK_PITCH_LIMIT))
                    approach(sweep, bg, sp)
                continue
            # SEEK 两阶段认脸:direct 阶段压锁(防途中的脸拽住),但 |resid|<SEEK_SUPPRESS_DEG 的正前方豁免
            seek_suppress = (wide_scan and seek_phase == "direct"
                             and abs(seek_target) >= SEEK_SUPPRESS_DEG)
            if locked and not seek_suppress:
                log("👀 锁定人脸,交给视觉跟随")
                if greet_armed:
                    with st.lock:
                        st.greet_now = True
                    greet_armed = False
                    log("👋 唤醒应答 → 招呼一句")
                set_state(ST_TRACKING, seed_interact=True)
                continue
            if wide_scan:
                if seek_phase == "direct":
                    # 阶段一:直转到 DOA 角度(confident)
                    bg = float(np.clip(seek_target, -BODY_LIMIT_DEG, BODY_LIMIT_DEG))
                    arrived = approach(seek_target, bg, 0.0)
                    if arrived:
                        seek_phase = "nearby"
                        seek_nearby_t = now
                        log(f"🔎 SEEK 到位({seek_target:+.0f}°)→ 附近找脸(±{SEEK_NEARBY_DEG:.0f}°)")
                elif seek_phase == "nearby":
                    # 阶段二:到位附近小范围扫
                    tnb = now - seek_nearby_t
                    sweep = seek_target + SEEK_NEARBY_DEG * math.sin(2 * math.pi * 0.4 * tnb) * seek_dir
                    sweep = float(np.clip(sweep, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                    sweep_pitch = SEEK_PITCH_UP + SEEK_PITCH_AMP * math.sin(2 * math.pi * SEEK_PITCH_HZ * tnb)
                    sweep_pitch = float(np.clip(sweep_pitch, -TRACK_PITCH_LIMIT, TRACK_PITCH_LIMIT))
                    bg = float(np.clip(sweep, -BODY_LIMIT_DEG, BODY_LIMIT_DEG))
                    approach(sweep, bg, sweep_pitch)
                    if tnb > SEEK_NEARBY_TIME_S:
                        seek_phase = "full"
                        phase_t = now
                        log(f"🔎 SEEK 附近没脸 → 退化全场扫")
                else:
                    # 阶段三:全场 sin 扫(=原有行为,兜底)
                    tscan = now - phase_t
                    sweep = WIDE_SCAN_RANGE * math.sin(2 * math.pi * WIDE_SCAN_HZ * tscan) * seek_dir
                    sweep_pitch = SEEK_PITCH_UP + SEEK_PITCH_AMP * math.sin(2 * math.pi * SEEK_PITCH_HZ * tscan)
                    sweep_pitch = float(np.clip(sweep_pitch, -TRACK_PITCH_LIMIT, TRACK_PITCH_LIMIT))
                    body_goal = float(np.clip(sweep, -BODY_LIMIT_DEG, BODY_LIMIT_DEG))
                    approach(sweep, body_goal, sweep_pitch)
                    if tscan > WIDE_SCAN_TIME_S:
                        log("🤷 SEEK 扫遍两侧仍无人 → 回待命")
                        with st.lock:
                            st.wake_cue = "giveup"
                            st.wake_cue_t = now
                        wide_scan = False
                        greet_armed = False
                        set_state(ST_ARMED)
            else:
                body_goal = float(np.clip(engage_target, -BODY_LIMIT_DEG, BODY_LIMIT_DEG))
                arrived = approach(engage_target, body_goal, 0.0)
                if arrived:
                    # 到位后主动扫头找人(±range 正弦),持续 ENGAGE_SCAN_TIME
                    tscan = now - phase_t
                    if tscan < ENGAGE_SCAN_TIME_S:
                        off = ENGAGE_SCAN_RANGE * math.sin(2 * math.pi * 0.5 * tscan) * scan_dir
                        with st.lock:
                            st.track_yaw = float(np.clip(body_goal + off,
                                                         body_goal - NECK_REL_LIMIT,
                                                         body_goal + NECK_REL_LIMIT))
                    else:                            # 转过去没找到脸 → 升级宽扫(统一视觉兜底)
                        log("🛑 转向后扫描无脸 → 升级宽扫找人")
                        wide_scan = True
                        seek_phase = "full"
                        phase_t = now
                if now - phase_t > ENGAGE_TIMEOUT_S and not wide_scan:
                    log("🛑 ENGAGING 超时无脸 → 升级宽扫找人")
                    wide_scan = True
                    seek_phase = "full"
                    phase_t = now

        elif state == ST_TRACKING:
            # 头部目标由视觉积分(behavior 不写);只做转出条件判断
            if not locked:                          # 迟滞已含 1.5s 丢锁 → 不会瞬断空转
                set_state(ST_SEARCHING)
            elif (now - last_interact) > NO_INTERACT_S and not speaking:
                if wake_mode:
                    log(f"💤 {NO_INTERACT_S:.0f}s 无说话互动 → 回 armed 待命")
                    set_state(ST_ARMED)
                else:
                    log(f"💤 {NO_INTERACT_S:.0f}s 无说话互动 → 回中位")
                    set_state(ST_RETURNING)

        elif state == ST_SEARCHING:
            if locked:
                set_state(ST_TRACKING)              # 回切不播种:延续原计时(Bug2)
            elif snd is not None and not speaking:  # 找回阶段有声音 → DOA 辅助再转(机器人说话时不转)
                with st.lock:
                    engage_target = float(np.clip(st.track_yaw + snd, -SND_TARGET_LIMIT, SND_TARGET_LIMIT))
                wide_scan = False
                set_state(ST_ENGAGING)
            elif now - phase_t > SEARCH_TIMEOUT_S:
                set_state(ST_RETURNING)
            # 否则原地保持(不动头)

        elif state == ST_RETURNING:
            done = approach(0.0, 0.0, 0.0)
            if exiting:
                # EXIT-01:回中 + 等告别播完(EXIT_MIN_S 宽限让告别出来 / EXIT_MAX_S 封顶防卡)→ armed;
                # 退出期间【不被 locked 拉回 TRACKING】(说了拜拜不该被人脸又锁回)。
                with st.lock:
                    pbe = st.playback_end_estimate
                if (done and now > pbe and (now - t_exit) > EXIT_MIN_S) or (now - t_exit) > EXIT_MAX_S:
                    exiting = False
                    log("👋 告别播完 → 回 armed 待命")
                    set_state(ST_ARMED)
            elif locked:                            # 回中途中又锁定人 → 重新跟(给足新 15s)
                set_state(ST_TRACKING, seed_interact=True)
            elif done:
                set_state(ST_IDLE)

        elif state == ST_PLAYING:
            # 头部目标由视觉手部积分驱动(vision_result_loop,同 TRACKING 的分工);
            # 开心表达由 head_control 叠加(天线,不打断跟手)。这里只判退出。
            with st.lock:
                h_at = st.hand_at
                h_move = st.hand_move
                gesture = st.gesture
                gesture_age = now - st.gesture_at
            # 手势行为（GESTURE-01）：1s 内的新鲜手势才响应
            if gesture_age < 1.0:
                if gesture == "fist":
                    log("✊ 握拳 → 停止互动")
                    with st.lock:
                        st.gesture = None
                    play_still_since = None
                    set_state(ST_RETURNING)
                elif gesture == "five":
                    log("🖐 张手挥手 → 开心！")
                    with st.lock:
                        st.gesture = None
                elif gesture in ("two", "three", "four"):
                    log(f"✌️ 手势 {gesture}（{st.gesture_fingers}指）")
                    with st.lock:
                        st.gesture = None
            if now - h_at > PLAY_OFF_S:
                log("💤 手离开 → 回到跟脸/待命")
                play_still_since = None
                set_state(ST_RETURNING)             # RETURNING 途中锁定人脸 → 自动回 TRACKING
            elif h_move < PLAY_MOVE_MIN:
                if play_still_since is None:
                    play_still_since = now
                elif now - play_still_since > PLAY_STILL_S:
                    log("🥱 手不动了,没意思 → 回到跟脸/待命")
                    play_still_since = None
                    set_state(ST_RETURNING)
            else:
                play_still_since = None

        elif state == ST_POINTING:
            # 全程冻结人脸跟踪(vision 仅在 TRACKING 态积分,POINTING 天然不积分)
            # 子阶段:turn 转向 → settle 停稳 → hold 抓帧并保持到看图返回 → 转回(RETURNING)
            with st.lock:
                pr = st.point_request
            if pr is None:
                set_state(ST_RETURNING)
            elif pt_phase == "turn":
                arrived = approach(pt_yaw_goal, pt_body_goal, pt_pitch_goal)
                if arrived or (now - phase_t) > POINT_TURN_TIMEOUT_S:
                    pt_settle_t = now
                    pt_phase = "settle"
            elif pt_phase == "settle":
                approach(pt_yaw_goal, pt_body_goal, pt_pitch_goal)  # 保持在目标角
                if now - pt_settle_t >= POINT_SETTLE_S:             # 停稳后再抓帧(目标居中)
                    with st.lock:
                        st.snap_grabbed = False
                    snap_q.put({"call_id": pr["call_id"], "gen": pr["gen"], "mode": "point"})
                    log("📸 转到位并停稳,抓居中目标帧")
                    pt_hold_t = now
                    pt_phase = "hold"
            elif pt_phase == "hold":
                approach(pt_yaw_goal, pt_body_goal, pt_pitch_goal)  # 保持朝向,不被拽回人脸
                with st.lock:
                    pending = st.snapshot_pending
                # 帧抓到 + 看图完成(pending 归零)→ 或封顶 → 转回看人
                if (now - pt_hold_t > POINT_HOLD_MAX_S) or (st.snap_grabbed and pending == 0):
                    with st.lock:
                        st.point_request = None
                    log("↩ 看图完成,转回看人")
                    set_state(ST_RETURNING)


# ───────────────────────── 头部控制线程:渲染层(唯一 set_target 写入口)─────────────────────────
def head_control_loop(mini: ReachyMini, st: State, stop: threading.Event) -> None:
    """25Hz 唯一硬件写入口:头部姿态 = behavior/视觉给的 track 目标 + 微动叠加 + body_yaw。
    M3 新增:全态呼吸(per-state, τ=2s 切换)、cue 缓动曲线(easeOutBack)、cue 微变异(±15%)、
    思考歪头(模型处理中)、表情回应(blendshape → 天线)。"""
    dt = 1.0 / IDLE_HZ
    amp = 0.0
    sway_scale = 1.0
    prev_state = ST_IDLE
    joy_until = 0.0
    next_joy = 0.0
    last_play_exit = -1e9
    ant_parked = True
    # M3-a cue 微变异(新 cue 触发时随机化)
    prev_cue = None
    prev_cue_t = 0.0
    v_dur = v_pitch = v_ant = 0.0
    # M3-b 思考/表情平滑
    think_env = 0.0
    expr_ant_cur = 0.0
    # 读一次 flags(运行时不变)
    _no_easing = st.no_easing
    _no_variation = st.no_variation
    _no_expression = st.no_expression
    # Issue#3: pose change tracer
    _pose_last_ty = 0.0
    _pose_last_body = 0.0
    _pose_log_t = 0.0
    _POSE_THRESH = 3.0
    _POSE_MIN_INT = 0.5
    while not stop.is_set():
        now = time.monotonic()
        with st.lock:
            action = st.action_active
            speaking = now < st.playback_end_estimate
            ty, tp, body = st.track_yaw, st.track_pitch, st.body_yaw_deg
            state = st.state
            tracked = (now - st.face_seen_at) < LOST_HOLD_S
            wake_cue = st.wake_cue
            wake_cue_t = st.wake_cue_t
            thinking = st.thinking
            u_smile = st.user_smile
            u_frown = st.user_frown
        if action:
            amp = 0.0
            prev_state = state
            time.sleep(dt)
            continue

        # ── Cue 渲染:M3-a 缓动(easeOutBack 攻击 + easeInQuad 衰减)+ 微变异(±15%)──
        cue_pitch, cue_ant = 0.0, None
        if wake_cue is not None:
            if wake_cue != prev_cue or wake_cue_t != prev_cue_t:
                _bd = CUE.get(f"{wake_cue}_dur", 0.4)
                _bp_c = CUE.get(f"{wake_cue}_pitch", 3.0)
                _ba = CUE.get(f"{wake_cue}_ant", 0.3)
                if not _no_variation:
                    _rv = lambda b: b * (1.0 + random.uniform(-CUE_VARIATION, CUE_VARIATION))
                    v_dur, v_pitch, v_ant = _rv(_bd), _rv(_bp_c), _rv(_ba)
                else:
                    v_dur, v_pitch, v_ant = _bd, _bp_c, _ba
                prev_cue = wake_cue
                prev_cue_t = wake_cue_t
            el = now - wake_cue_t
            if v_dur > 0 and 0.0 <= el < v_dur:
                t_norm = el / v_dur
                if not _no_easing:
                    if t_norm < EASE_ATTACK_FRAC:
                        env = pytweening.easeOutBack(min(t_norm / EASE_ATTACK_FRAC, 1.0))
                    else:
                        env = 1.0 - pytweening.easeInQuad(
                            min((t_norm - EASE_ATTACK_FRAC) / (1.0 - EASE_ATTACK_FRAC), 1.0))
                    env = max(0.0, env)
                else:
                    env = math.sin(math.pi * t_norm)
                if wake_cue == "heard":
                    cue_pitch = -v_pitch * env
                    da = v_ant * env
                elif wake_cue == "fail":
                    cue_pitch = +v_pitch * env
                    da = -v_ant * env
                elif wake_cue == "giveup":
                    cue_pitch = +v_pitch * env
                    da = -v_ant * env
                elif wake_cue == "barge":
                    cue_pitch = -v_pitch * env
                    da = -v_ant * env
                else:  # bye
                    cue_pitch = +v_pitch * env
                    da = -v_ant * env
                cue_ant = [INIT_ANTENNAS[0] + da, INIT_ANTENNAS[1] + da]

        # ── M3-b 思考微行为(模型处理中歪头 + 天线不对称摆)──
        if not _no_expression and thinking and state in (ST_TRACKING, ST_IDLE, ST_SEARCHING):
            think_env += (1.0 - think_env) * (dt / THINK_BLEND_TAU)
        else:
            think_env += (0.0 - think_env) * (dt / THINK_BLEND_TAU)
        think_roll = think_env * THINK_ROLL_AMP * math.sin(2 * math.pi * THINK_ROLL_F * now)
        think_pitch_off = think_env * THINK_PITCH

        # ── M3-b 表情回应(用户微笑/皱眉 → 天线偏置)──
        if not _no_expression:
            _expr_target = u_smile * EXPR_SMILE_ANT + u_frown * EXPR_FROWN_ANT
            expr_ant_cur += (_expr_target - expr_ant_cur) * (dt / EXPR_BLEND_TAU)
        else:
            expr_ant_cur = 0.0

        # ── ARMED 早返回:慢呼吸(仅 ARMED 有呼吸)+ cue ──
        if state == ST_ARMED:
            br = ARMED_BREATH_PITCH * math.sin(2 * math.pi * ARMED_BREATH_F * now)
            ant = cue_ant if cue_ant is not None else list(INIT_ANTENNAS)
            try:
                mini.set_target(head=head_pose(pitch_deg=tp + br + cue_pitch, yaw_deg=ty),
                                antennas=ant, body_yaw=math.radians(body))
            except Exception:
                time.sleep(1.0)
            prev_state = state
            amp = 0.0
            time.sleep(dt)
            continue

        # ── 开心表达(PLAY-01-b):进入不动天线,持续逗够久才摇(用户反馈调校)──
        if state == ST_PLAYING and prev_state != ST_PLAYING:
            ant_parked = False
            if now - last_play_exit > PLAY_REENTRY_S:
                next_joy = now + PLAY_JOY_DELAY_S
        elif state != ST_PLAYING and prev_state == ST_PLAYING:
            last_play_exit = now
        if state == ST_PLAYING and now >= next_joy:
            joy_until = now + PLAY_JOY_FLICK_S
            next_joy = now + PLAY_JOY_PERIOD_S
        prev_state = state

        antennas = None
        if state == ST_PLAYING:
            if now < joy_until:
                w = 0.3 * math.sin(2 * math.pi * 3.0 * now)
                antennas = [INIT_ANTENNAS[0] - w, INIT_ANTENNAS[1] + w]
            else:
                antennas = list(INIT_ANTENNAS)
        elif not ant_parked:
            antennas = list(INIT_ANTENNAS)
            ant_parked = True

        # M3-b 天线叠加:思考 + 表情(仅 cue 未接管时)
        _need_ant = (cue_ant is None and (think_env > 0.01 or abs(expr_ant_cur) > 0.005))
        if _need_ant and antennas is None:
            antennas = list(INIT_ANTENNAS)
        if antennas is not None and cue_ant is None:
            if think_env > 0.01:
                _ta = think_env * THINK_ANT_AMP * math.sin(2 * math.pi * THINK_ANT_F * now)
                antennas[0] += _ta
                antennas[1] -= _ta
            if abs(expr_ant_cur) > 0.005:
                antennas[0] += expr_ant_cur
                antennas[1] += expr_ant_cur

        sway_ok = state in (ST_IDLE, ST_TRACKING)
        amp += ((1.0 if (speaking and sway_ok) else 0.0) - amp) * (dt / IDLE_TAU)
        target_scale = TRACK_SWAY_SCALE if tracked else 1.0
        sway_scale += (target_scale - sway_scale) * (dt / IDLE_TAU)
        sway_yaw = amp * sway_scale * IDLE_YAW_AMP * math.sin(2 * math.pi * IDLE_YAW_F * now)
        sway_pitch = amp * sway_scale * IDLE_PITCH_AMP * math.sin(2 * math.pi * IDLE_PITCH_F * now + 1.0)
        if cue_ant is not None:
            antennas = cue_ant
        # Issue#3: trace significant head/body movements to conv dashboard
        if now - _pose_log_t >= _POSE_MIN_INT:
            d_ty = abs(ty - _pose_last_ty)
            d_body = abs(body - _pose_last_body)
            if d_ty > _POSE_THRESH or d_body > _POSE_THRESH:
                action_src = "gesture" if action else state
                _record_vis_event(
                    "vis.head_move",
                    f"📐 头 yaw {ty:+.0f}° 身 {body:+.0f}° [{action_src}]",
                    {"track_yaw": round(ty, 1), "body_yaw": round(body, 1),
                     "track_pitch": round(tp, 1), "state": state,
                     "action": bool(action)})
                _pose_last_ty = ty
                _pose_last_body = body
                _pose_log_t = now
        try:
            mini.set_target(
                head=head_pose(pitch_deg=tp + sway_pitch + cue_pitch + think_pitch_off,
                               yaw_deg=ty + sway_yaw,
                               roll_deg=think_roll),
                antennas=antennas,
                body_yaw=math.radians(body))
        except Exception:
            time.sleep(1.0)
        time.sleep(dt)


# ─────────── 说话人切换判断(可替换为声纹等方案) ───────────
def _detect_new_speaker(st: State):
    """音频信号判断唤醒词是否来自不同说话人(非当前跟踪者)。
    返回 (is_new, info)。当前实现: DOA 方向; 后续可替换为声纹比对。
    - DOA 确信 + 偏离 > SWITCH_AWAY_DEG → True(确定是新人)
    - DOA 确信 + 偏离小 → False(确定是同一个人)
    - DOA 不可用/不确信 → None(无法判断,交给调用者决定)"""
    with st.lock:
        sr, sat, sconf = st.doa_resid_stable, st.doa_at, st.doa_confident
    fresh = sr is not None and (time.monotonic() - sat) < DOA_GATE_FRESH_S
    info = {"resid": sr, "confident": sconf, "fresh": fresh}
    if not fresh or not sconf:
        return None, info
    return abs(sr) > SWITCH_AWAY_DEG, info


# ───────────────────────── 主流程 ─────────────────────────
def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        log("❌ 环境变量 DASHSCOPE_API_KEY 未配置,退出。")
        return 1
    dashscope.api_key = api_key
    oai = OpenAI(api_key=api_key, base_url=VISION_BASE_URL)  # take_snapshot 看图

    _args = sys.argv[1:]
    no_wake = "--no-wake" in _args                      # 回退:启动即连即对话(旧行为,排错/对比)
    sim_fail = "--simulate-conn-fail" in _args          # M1c-a 测试:让命中后连接直接失败(测 fail cue,免断网)
    no_gate = "--no-gate" in _args                      # M1.5-a:关方向门控 = 全向(对比/排错)
    no_switch = "--no-switch" in _args                  # M1.5-b:关二次唤醒切换(对比/排错)
    no_sticky = "--no-sticky" in _args                  # M1.5-c:关粘滞选脸 = 每帧 argmax 最大脸(对比/排错)
    no_easing = "--no-easing" in _args                  # M3-a:关缓动曲线,回退 sin 包络
    no_variation = "--no-variation" in _args            # M3-a:关 cue 微变异
    no_expression = "--no-expression" in _args          # M3-b:关表情/思考反应
    no_memory = "--no-memory" in _args                  # M3-c:关记忆系统
    for a in _args:                                      # --cue-heard-pitch=8 等,调确认动作幅度/时长
        if a.startswith("--cue-") and "=" in a:
            _k, _v = a[6:].split("=", 1)
            _ck = _k.replace("-", "_")
            if _ck in CUE:
                try:
                    CUE[_ck] = float(_v)
                except ValueError:
                    log(f"⚠ 忽略非法 --cue 值:{_k}={_v}(需数字)")
    _nums = [a for a in _args if a.replace(".", "", 1).isdigit()]
    run_seconds = float(_nums[0]) if _nums else None    # 编排测试用:到时干净退出

    print("=== 小艺(Reachy Mini)语音对话:可打断 + 动作 + 看图 + 人脸跟随 + 听声转头 ===", flush=True)
    log(f"模型:{MODEL}|semantic_vad|16k上行|24k→16k下行|8 动作 + 看图 + 指向 + 逗它|五层仲裁(手势/指向>逗它>声源>跟随>微动)")

    st = State()
    st.no_easing = no_easing
    st.no_variation = no_variation
    st.no_expression = no_expression
    st.no_memory = no_memory

    global _id_recognizer, _memory_mgr, _owner_mgr, _face_pipeline, _asd_engine
    _owner_mgr = OwnerManager()
    _id_recognizer = IdentityRecognizer()
    _memory_mgr = MemoryManager(owner_mgr=_owner_mgr,
                                 face_db=_id_recognizer.db)
    _roi_detector = _make_roi_detector()   # 方案B:识别走全分辨率 ROI 重检
    _face_pipeline = FaceReIDPipeline(_make_face_embedder(_id_recognizer, _roi_detector),
                                      FaceSystemConfig(), log_fn=log)
    _n_gal = _face_pipeline.load_gallery()
    log(f"🧬 ReID pipeline 就绪(ByteTrack + 三区间, gallery {_n_gal} 人)")
    _asd_engine = AsdEngine()           # LR-ASD 谁在说话(独立评分线程)
    _asd_engine.start()
    log("🗣 ASD 就绪(LR-ASD 谁在说话)" if _asd_engine.available
        else "⚠ ASD 不可用(缺 torch/权重)→ 发言归属回退最大脸")
    if _id_recognizer.startup_merged:
        for drop_pid, keep_pid in _id_recognizer.startup_merged.items():
            _memory_mgr.merge_memories(keep_pid, drop_pid)
        log(f"🧠 记忆合并完成: {len(_id_recognizer.startup_merged)} 对")
    log(f"🆔 身份识别就绪 (特征库 {len(_id_recognizer.db.persons)} 人)")
    play_q: "queue.Queue" = queue.Queue()
    motion_q: "queue.Queue" = queue.Queue()
    snap_q: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    log("连接 Reachy Mini(media_backend=default, automatic_body_yaw=False)…")
    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="default",
        automatic_body_yaw=False,
    ) as mini:
        try:
            if not NO_VOICE:
                audio_ok = mini.media.audio is not None
                camera_ok = mini.media.camera is not None
                log(f"媒体后端: audio={'✅' if audio_ok else '❌ 未初始化'} camera={'✅' if camera_ok else '❌ 未初始化'}")
                if not audio_ok:
                    log("⚠ audio 未初始化 → 麦克风/播放不可用。尝试用 media_backend='local' 重连...")
                mini.media.start_recording()
                # macOS：先让录音管线稳定，再启播放管线，避免 osxaudiosrc 被干扰输出全零
                time.sleep(0.5)
                mini.media.start_playing()
                log("✅ 录音/播放管线已启动")
            else:
                log("🔇 NO_VOICE=1 → 跳过音频管线,仅测试视觉特性(人脸跟踪/指向/逗它)")
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            time.sleep(0.8)
            # 摄像头预热(顺便验证与录音管线并存)
            if USE_WEBCAM:
                import cv2 as _cv2
                _cap_warm = _cv2.VideoCapture(0)
                warm = None
                for _ in range(10):
                    ret, _f = _cap_warm.read()
                    if ret:
                        warm = _f
                        break
                    time.sleep(0.05)
                _cap_warm.release()
            else:
                warm = None
                wdl = time.monotonic() + 10.0
                while warm is None and time.monotonic() < wdl:
                    warm = mini.media.get_frame()
                    if warm is None:
                        time.sleep(0.05)
            log(f"摄像头:{'✅ 出帧 ' + str(warm.shape) if warm is not None else '⚠ 10s 无帧(跟随/take_snapshot 可能失败)'}")

            # M3-c 记忆注入:启动时加载已有画像和记忆,拼入 INSTRUCTIONS
            active_instructions = INSTRUCTIONS
            active_tools = TOOLS
            if not no_memory:
                _profile = load_profile()
                _mems = load_memories()
                if _profile or _mems:
                    _mem_sec = "\n\n--- 你的记忆 ---\n"
                    if _profile and _profile.get("summary"):
                        _mem_sec += f"上次对话:{_profile['summary']}\n"
                    if _mems:
                        _mem_sec += "记住的事:\n" + "\n".join(f"- {m['content']}" for m in _mems[-20:]) + "\n"
                    active_instructions += _mem_sec
                    log(f"📝 已加载记忆({len(_mems)} 条记忆" + (", 有画像" if _profile else "") + ")")
            else:
                active_tools = [t for t in TOOLS if t["name"] not in ("remember_fact", "clear_memory", "confirm_clear", "forget_fact")]

            dialog = RealtimeDialog(st, play_q, motion_q, snap_q, mini,
                                     oai, _memory_mgr, _owner_mgr, _id_recognizer,
                                     active_instructions, active_tools, no_memory,
                                     face_pipeline=_face_pipeline, asd_engine=_asd_engine)

            conv = None
            kws_gate = None
            if no_wake:
                log("连接 Qwen-Omni-Realtime(--no-wake:启动即连)…")
                conv = dialog.open_session(timeout=10.0)
                if conv is None:
                    log("❌ 启动连接失败,中止")
                    return 1
                with st.lock:
                    st.state = ST_IDLE
            else:
                with st.lock:
                    st.state = ST_ARMED
                kws_gate = KwsGate()
                log(f"🌙 待命态(armed):只听唤醒词「小艺」(yī/yìn/yì @{KWS_SINGLE_THR})、未连 Qwen。喊「小艺」唤醒。")

            st.last_interaction_at = time.monotonic()  # 种子:防计时器一上来就误触发
            threading.Thread(target=player_loop, args=(mini, st, play_q, stop), daemon=True).start()
            threading.Thread(target=motion_loop, args=(mini, st, motion_q, stop), daemon=True).start()
            threading.Thread(target=head_control_loop, args=(mini, st, stop), daemon=True).start()
            if not NO_VOICE:
                threading.Thread(target=snapshot_loop, args=(mini, st, dialog.callback, oai, snap_q, stop), daemon=True).start()
            threading.Thread(target=doa_sensor_loop, args=(st, stop), daemon=True).start()
            threading.Thread(target=behavior_loop, args=(st, snap_q, stop, not no_wake), daemon=True).start()
            if VIS_DEBUG:
                threading.Thread(target=vis_debug_server, args=(st, VIS_DEBUG_PORT, stop), daemon=True).start()
            # 视觉(TRACK-FIX):检测在子进程(独立 GIL),主进程只跑抓帧泵+结果积分
            vis_frame_q = None
            _cb_ref = [None]
            _cb_ref[0] = dialog.callback
            _vis_enabled = os.path.exists(VIS_MODEL_PATH)
            if _vis_enabled:
                _fb = os.environ.get("FACE_BACKEND", "scrfd").lower()
                log(f"视觉后端: {_fb}" + (" (sticky OFF)" if no_sticky else ""))
                if no_sticky:
                    os.environ["VISION_NO_STICKY"] = "1"
                elif "VISION_NO_STICKY" in os.environ:
                    del os.environ["VISION_NO_STICKY"]
                vis_frame_q = multiprocessing.Queue(maxsize=1)
                vis_result_q = multiprocessing.Queue(maxsize=64)
                multiprocessing.Process(
                    target=_vision_worker_fn,
                    args=(VIS_MODEL_PATH, HAND_MODEL_PATH, vis_frame_q, vis_result_q),
                    kwargs={"gesture_model": GESTURE_MODEL_PATH},
                    daemon=True,
                ).start()
                threading.Thread(target=frame_pump_loop, args=(mini, st, vis_frame_q, stop), daemon=True).start()
                threading.Thread(target=vision_result_loop, args=(st, vis_result_q, stop, _cb_ref), daemon=True).start()
            else:
                log(f"⚠ 视觉模型不存在({VIS_MODEL_PATH}),本次无人脸跟随(其余功能不受影响)")

            # 排空预热期旧音频(限时 3s,防排空循环被持续来帧拖死)
            drain_dl = time.monotonic() + 3.0
            while time.monotonic() < drain_dl and mini.media.get_audio_sample() is not None:
                pass

            # 主循环:麦克风 → Realtime 上行(每 10s 报一次电平,便于排查"说话没被听见")
            sent_samples = 0
            rms_acc: list[float] = []
            rms_t = time.monotonic()
            vid_send_t = 0.0          # 视频送帧节流(1fps)
            vid_sent = 0              # 已送视频帧计数(周期回显)
            _audio_conv = None        # bug-069:本会话确认送过 audio 的 conv;video 仅在 conv is _audio_conv 时发(防 image-before-audio)
            t_run0 = time.monotonic()
            greet_i = 0   # 唤醒招呼轮换索引
            greet_sent_at = 0.0
            prev_gate_open = True   # M1.5-a 门控状态(只在切换时打日志)
            last_switch = -1e9      # M1.5-b 上次切换时刻(冷却)
            # ── ASR 级联(阶段1 shadow / 阶段2 cascade)──
            #   ASR_SHADOW=1:并行跑 ASR 只打日志(不驱动 S2S,对拍用)。
            #   CASCADE=1   :ASR 轮驱动 Omni(handle_asr_turn:带标签文本 + 记忆 + create_response);
            #                 Omni 侧转写/服务端VAD 已在 open_session 关掉,音频送静音锚点。
            _asr_shadow = os.environ.get("ASR_SHADOW") == "1"
            _cascade = os.environ.get("CASCADE") == "1"
            _cascade_clear_t = time.monotonic()   # cascade:周期清空静音 buffer 防膨胀
            _asr_stream = None
            _asr_agg = None
            if _asr_shadow or _cascade:
                def _asr_resolve_spk(t0):
                    if _asd_engine is not None and _asd_engine.available:
                        sw = _asd_engine.speaker_window(t0)
                        if sw is not None:
                            return sw[0]
                    return None
                def _asr_on_turn(t):
                    if _cascade:                       # 阶段2:ASR 轮驱动 Omni(门控在此:仅 engaged + 非播放才驱动,
                                                       # 滤 idle 环境音/回声;ASR 保活靠恒喂,不靠喂帧门)
                        _c = dialog.callback
                        with st.lock:
                            _eng = st.state != ST_ARMED
                            _playing = time.monotonic() < st.playback_end_estimate
                        if not _eng or _playing:
                            log(f"🧾 ASR轮(忽略·{'idle' if not _eng else '播放中'}):{t.text}")
                            return
                        if _c is not None and _c.conv is not None:
                            try:
                                _c.handle_asr_turn(t.text, t.start_mono)
                            except Exception as _he:
                                log(f"⚠ handle_asr_turn 失败:{type(_he).__name__}: {_he}")
                        else:
                            log(f"🧾 ASR轮(未连 Omni,丢弃):{t.text}")
                        return
                    # 阶段1 shadow:只打日志对拍
                    pid = _asr_resolve_spk(t.start_mono)
                    name = None
                    if pid is not None and _memory_mgr is not None and not str(pid).startswith("t"):
                        try:
                            name = _memory_mgr.get_name(pid)
                        except Exception:
                            name = None
                    who = name or (f"?{str(pid)[-6:]}" if pid else "画外")
                    log(f"🧾 ASR轮[{t.reason}] 归属={who} 句数={t.n_sentences} :: {t.text}")
                _barge_t = [0.0]     # 上次插话打断时刻(去抖)
                _asr_no_barge = os.environ.get("CASCADE_NO_BARGE") == "1"   # =1 关闭自然打断(回声调不好时兜底)
                def _asr_on_partial(_txt):
                    # 级联:用户正在说(ASR 出中间稿)→ ①刷新互动时间防误判冷场断连 ②机器人正说话时=自然打断
                    if not _cascade:
                        return
                    now = time.monotonic()
                    with st.lock:
                        st.last_interaction_at = now
                        _playing = now < st.playback_end_estimate
                        _active = st.active_resp_id is not None and not st.active_resp_done
                    # 自然打断:机器人正说话(播放中/在途回复)时用户开口 → 停播+取消(ASR 基本听不到机器人自己,回声≈无)
                    if (not _asr_no_barge) and (_playing or _active) and (now - _barge_t[0]) > 1.0:
                        _barge_t[0] = now
                        _c = dialog.callback
                        if _c is not None and _c.conv is not None:
                            try:
                                _c._do_barge_in(_active)
                                with st.lock:
                                    st.active_resp_done = True   # 已取消在途回复 → 标记不忙,后续该轮直接答不走接管
                                log(f"⛔ [ASR]插话打断(检测到你开口):{_txt[:12]}…")
                            except Exception as _bge:
                                log(f"⚠ [ASR]打断失败:{type(_bge).__name__}: {_bge}")
                try:
                    _tg = 0.7 if _cascade else 1.2     # cascade turn_gap(真延迟大头是 in_flight 泄漏致每轮 cancel,不是聚轮;调回小值降端点延迟)
                    _asr_agg = TurnAggregator(on_turn=_asr_on_turn, resolve_speaker=_asr_resolve_spk, turn_gap_s=_tg)
                    _asr_stream = AsrStream(on_sentence=_asr_agg.add, on_partial=(_asr_on_partial if _cascade else None))
                    _asr_stream.start()
                    log(f"🎙 ASR 已启动({'CASCADE 驱动 Omni' if _cascade else 'shadow 对拍'};独立 ASR + 轮次聚合 gap={_tg}s)")
                except Exception as _e:
                    log(f"⚠ ASR 启动失败,禁用:{type(_e).__name__}: {_e}")
                    _asr_stream = None
                    _asr_agg = None
            try:
                while True:
                    if run_seconds is not None and time.monotonic() - t_run0 >= run_seconds:
                        log(f"⏱ 到达预设时长 {run_seconds:.0f}s,自动退出")
                        break
                    chunk = mini.media.get_audio_sample()
                    if chunk is None or len(chunk) == 0:
                        time.sleep(0.01)
                        continue
                    mono = chunk[:, 0]
                    if _asd_engine is not None:
                        _asd_engine.feed_audio(mono)   # ASD 同步音频(谁在说话)
                    # WAKE-01:同一份 16k mono 始终喂 KWS(本地);engaged 才扇出给 Qwen
                    wake = kws_gate.feed(mono, chunk) if kws_gate is not None else False
                    with st.lock:
                        state = st.state
                        woke_pending = st.wake_ok
                    if _asr_stream is not None:        # ASR 级联:同一路 mono **恒喂**(保活——ASR 连上后长时间无音频会
                                                       # NO_VALID_AUDIO_ERROR 断连;engaged/播放门控移到 on_turn,不在喂帧处)
                        try:
                            _asr_stream.feed(np.clip(mono * 32767.0, -32768, 32767).astype(np.int16).tobytes())
                        except Exception:
                            pass
                    if state == ST_ARMED:
                        if conv is not None and not woke_pending:
                            dialog.close_session()
                            conv = None
                            log("🌙 已回待命,WS 断开(零连接零计费)")
                        elif wake and conv is None:                    # 命中才连(b)
                            with st.lock:                              # 听到了:0 延迟确认(DOA 分流改由 behavior 连接后实时读)
                                st.wake_cue = "heard"
                                st.wake_cue_t = time.monotonic()
                            log("🔔 听到「小艺」(上扬)→ 连接 Qwen…")
                            _record_vis_event("vis.wake_word", "🔔 唤醒词「小艺」触发", {})
                            tc = time.monotonic()
                            conv = None if sim_fail else dialog.open_session()
                            if conv is not None:
                                log(f"✅ 已连接({(time.monotonic()-tc)*1000:.0f}ms)→ 唤醒,开始对话")
                                with st.lock:
                                    st.wake_ok = True
                            else:
                                with st.lock:                          # 连失败:下垂(后到覆盖 heard)
                                    st.wake_cue = "fail"
                                    st.wake_cue_t = time.monotonic()
                                log("❌ 连接失败/超时 → 失败反馈(天线下垂),留在待命,可重喊"
                                    + ("(--simulate-conn-fail)" if sim_fail else ""))
                        continue                                       # armed:绝不发上行
                    # engaged:扇出上行给 Qwen
                    if conv is None:
                        # WS 已死但 behavior 还停在 ENGAGING/TRACKING:
                        # KWS 命中时直接重连,否则等 behavior 超时回 ARMED
                        if wake:
                            log("🔔 WS 已断但仍在对话态,收到唤醒词 → 重连…")
                            conv = dialog.open_session()
                            if conv is None:
                                log("❌ 重连失败,等 behavior 超时回待命")
                            else:
                                log(f"🔄 重连成功,继续对话")
                        else:
                            time.sleep(0.01)
                        continue
                    # M1.5-b 二次唤醒(A方案:喊"小艺"→ 打断当前回话 + 天线应答 + 转向 DOA 找喊话人,保留会话):
                    if wake and not no_switch and (time.monotonic() - last_switch) > SWITCH_COOLDOWN_S:
                        if st.clear_lock:
                            pass   # 安全删除确认期间不允许切换
                        else:
                            last_switch = time.monotonic()
                            # ① 立刻打断当前回话(停播放 + 取消在途回复)
                            with st.lock:
                                _in_flight = st.in_flight > 0
                            try:
                                dialog.callback._do_barge_in(_in_flight)
                            except Exception as _be:
                                log(f"⚠ 唤醒打断失败:{type(_be).__name__}: {_be}")
                            # ② 天线上扬应答("听到了,我来找你")——覆盖 barge cue
                            with st.lock:
                                st.wake_cue = "heard"
                                st.wake_cue_t = time.monotonic()
                                _sr, _sat, _sconf = st.doa_resid_stable, st.doa_at, st.doa_confident
                            _sfresh = _sr is not None and (time.monotonic() - _sat) < DOA_GATE_FRESH_S
                            # ③ 转向 DOA 方向找喊话人(behavior 三档:confident直转/粗方向/反向),保留会话
                            with st.lock:
                                st.switch_request = {"resid": _sr, "confident": _sconf, "fresh": _sfresh}
                            if vis_frame_q is not None:
                                try:
                                    vis_frame_q.put_nowait("sticky_reset")
                                except Exception:
                                    pass
                            _hint = (f"confident resid {_sr:+.0f}°" if (_sfresh and _sconf)
                                     else f"粗方向 resid {_sr:+.0f}°" if (_sfresh and _sr is not None)
                                     else "无方向")
                            log(f"🔀 二次唤醒 → 打断+转向找喊话人({_hint});保留会话")
                    # 唤醒应答:SEEK 锁脸那刻 behavior 置 greet_now → 让模型招呼一句(走标准 response,可被打断)。
                    # 守卫:仅当无回应在途/不在途才招呼——只喊"小艺"=模型空闲才招呼;带了后续话=模型已在答,不双答。
                    # in_flight>0 漏掉"刚说完、回复还没+1"的竞态窗(治 'Conversation already has an active response'):
                    # 再加 thinking(说完等回复中)+ turn_speaker_at<2s(刚有用户说话→服务端会自动起回复)双兜底。
                    with st.lock:
                        _do_greet = st.greet_now
                        if _do_greet:
                            st.greet_now = False
                        _busy = (st.in_flight > 0 or st.thinking
                                 or (time.monotonic() - st.turn_speaker_at) < 2.0)
                    if _do_greet and not _busy and (time.monotonic() - greet_sent_at) > 1.0:
                        # 身份+记忆注入:首次识别到人后,用 update_session 嵌入记忆(切人时自动替换旧记忆)
                        with st.lock:
                            _g_pid = st.current_person_id
                            _g_pname = st.current_person_name
                        if _g_pid and _memory_mgr is not None:
                            if _cascade:
                                dialog.inject_context(_g_pid, _g_pname)   # cascade:走 create_item,不 update_session(免破坏级联)
                            else:
                                dialog.update_memory(_g_pid, _g_pname)   # 招呼是一次性事件,注入被招呼者记忆
                        _phrase = GREET_PHRASES[greet_i % len(GREET_PHRASES)]
                        greet_i += 1
                        try:
                            conv.create_response(instructions=greet_prompt(_phrase))
                            log(f"👋 唤醒应答:招呼「{_phrase}」")
                        except Exception as e:
                            log(f"⚠ 唤醒招呼发送失败:{e}")
                        greet_sent_at = time.monotonic()
                    elif _do_greet and _busy:
                        log("👋 唤醒应答跳过(模型已在回应后续话,不双答)")
                    # (已移除"按 current_person_id 的延迟补注入":焦点驱动的注入会和「本句说话人」打架→
                    #  回复用错人记忆。记忆注入统一由 realtime transcription 按 turn_speaker 完成。)
                    _rms = float(np.sqrt(np.mean(mono**2)))
                    rms_acc.append(_rms)
                    now_g = time.monotonic()
                    # M1.5-a 方向门控:仅 TRACKING(面前有人在对话)时屏蔽其他方向的声音;
                    # 其他状态(寻人/回中/跟手等)一律放行,避免发静音导致服务端断连。
                    with st.lock:
                        g_resid = st.doa_resid_stable
                        g_fresh = g_resid is not None and (now_g - st.doa_at) < DOA_GATE_FRESH_S
                        g_conf = st.doa_confident
                    gate_open = no_gate or state != ST_TRACKING or not (g_fresh and g_conf and abs(g_resid) > GATE_DEG)
                    if gate_open != prev_gate_open:
                        if gate_open:
                            log("🚪 门控:开 → 正常上行")
                            _record_vis_event("gate.open", "🚪 门控开放(收音)", {"resid": round(g_resid, 1) if g_resid is not None else None})
                        else:
                            log(f"🚪 门控:关(确信范围外 resid {g_resid:+.0f}° >±{GATE_DEG:.0f})→ 发静音,不送/不打断/不计时")
                            _record_vis_event("gate.closed", f"🚪 门控关闭(方向外 {g_resid:+.0f}°)", {"resid": round(g_resid, 1), "gate_deg": GATE_DEG})
                        prev_gate_open = gate_open
                        with st.lock:
                            st.dbg_gate_open = gate_open
                    if gate_open and not _cascade:
                        pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
                    else:
                        pcm16 = np.zeros(len(mono), dtype=np.int16)   # 范围外/级联:静音占位(级联恒送静音锚点,真音频只喂 ASR)
                    _b64_audio = base64.b64encode(pcm16.tobytes()).decode("ascii")
                    # 音频闸门：身份未确认时缓存音频，不送模型(级联恒送静音锚点,不走闸门,免视频失锚)
                    with st.lock:
                        if st.audio_gate_closed and not _cascade:
                            st.audio_gate_buffer.append(_b64_audio)
                            # 超时兜底
                            if (time.monotonic() - st.audio_gate_closed_at) > AUDIO_GATE_TIMEOUT_S:
                                _gate_buf = list(st.audio_gate_buffer)
                                st.audio_gate_buffer.clear()
                                st.audio_gate_closed = False
                            else:
                                continue
                        else:
                            _gate_buf = None
                    if _gate_buf is not None:
                        log(f"⚠ 音频闸门超时开启（身份未确认），flush {len(_gate_buf)} 帧")
                        for _gc in _gate_buf:
                            try:
                                conv.append_audio(_gc)
                            except Exception as _e:
                                log(f"⚠ 闸门 flush 送音频失败,中止剩余:{type(_e).__name__}")
                                break
                    if _cascade and (time.monotonic() - _cascade_clear_t) > 30.0:
                        try:
                            conv.clear_appended_audio()   # 周期清 buffer 防膨胀;紧接着 append 静音重锚(先音频后视频,不失锚)
                        except Exception:
                            pass
                        _cascade_clear_t = time.monotonic()
                    try:
                        conv.append_audio(_b64_audio)
                    except Exception as _ae:
                        # 服务端主动断开（如 InternalError）后 WebSocket 已关闭，
                        # append_audio 会抛 WebSocketConnectionClosedException。
                        # 自动重连，丢弃本帧音频继续。
                        log(f"⚠ 上行音频失败({type(_ae).__name__})，尝试自动重连…")
                        dialog.close_session()
                        conv = dialog.open_session()
                        if conv is None:
                            log("❌ 自动重连失败，等待下次唤醒")
                        else:
                            log("🔄 自动重连成功，继续上行")
                        continue
                    sent_samples += len(mono)
                    _audio_conv = conv        # bug-069:本会话已送过 audio → 允许发 video
                    # ── 视频流送 Omni:每 1s 一帧 720p JPEG(模型实时看见画面,替代 take_snapshot)──
                    # 守门:① 本会话(conv 对象)送过 audio;② update_session(重注入)后抑制一小段,
                    #   让新音频先到服务端,否则 update_session 重置缓冲→图像先于音频→append image before append audio。
                    _now_v = time.monotonic()
                    with st.lock:
                        _vid_suppress = st.video_suppress_until
                    if conv is _audio_conv and _now_v >= _vid_suppress and _now_v - vid_send_t >= 1.0:
                        vid_send_t = _now_v
                        with st.lock:
                            _vframe = st.latest_frame
                        if _vframe is not None:
                            try:
                                _vh, _vw = _vframe.shape[:2]
                                if _vw > 1280:                       # 降到 720p(宽 1280)
                                    _vframe = cv2.resize(_vframe, (1280, int(_vh * 1280.0 / _vw)))
                                _ok, _jpg = cv2.imencode(".jpg", _vframe, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                if _ok:
                                    conv.append_video(base64.b64encode(_jpg.tobytes()).decode("ascii"))
                                    vid_sent += 1
                                    if vid_sent % 30 == 0:        # 每 ~30s 回显一次
                                        log(f"📹 视频流已送 {vid_sent} 帧({_jpg.size // 1024}KB/帧)")
                            except Exception as _ve:
                                log(f"⚠ 送视频帧失败:{type(_ve).__name__}")
                    if time.monotonic() - rms_t >= 10.0:
                        rms = float(np.mean(rms_acc)) if rms_acc else 0.0
                        if rms < 0.005:
                            log(f"🎙 近10s 上行电平偏低(RMS={rms:.4f}),说话请大声靠近")
                        rms_acc = []
                        rms_t = time.monotonic()
            except KeyboardInterrupt:
                print(flush=True)
                log(f"收到 Ctrl+C,退出。本次共上行音频 {sent_samples / 16000:.1f} 秒")
            finally:
                stop.set()
                if vis_frame_q is not None:
                    try:
                        vis_frame_q.put_nowait(None)  # 视觉子进程退出哨兵(daemon 进程兜底)
                    except Exception:
                        pass
                time.sleep(0.15)  # 让 head_control 最后一帧 set_target 落地,避免与回中 goto 抢
                if conv is not None:
                    try:
                        conv.close()
                    except Exception:
                        pass
                try:
                    mini.media.stop_recording()
                    mini.media.stop_playing()
                    # 身体可能转到 ±90°,回正给足时间
                    mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.5, body_yaw=0.0)
                except Exception as _e:
                    log(f"⚠ 退出回正失败(可能未归中):{type(_e).__name__}")
                # M3-c 退出时被动 consolidation:分人做记忆复盘
                if not no_memory and _memory_mgr:
                    with st.lock:
                        _remaining = dict(st.conversation_log)
                    for _exit_pid, _exit_log in _remaining.items():
                        if (_exit_pid not in ("_unknown", "_offscreen")
                                and not _exit_pid.startswith("_track") and len(_exit_log) >= 2):
                            log(f"📝 退出摘要({_exit_pid[:12]})…")
                            dialog.save_summary(_exit_pid, _exit_log)
                log("已释放 Realtime 连接与 Reachy 媒体资源。")
                if _memory_mgr is not None:
                    _memory_mgr.flush()
                    log("💾 记忆已持久化")
                if _face_pipeline is not None:
                    try:
                        _face_pipeline.save_gallery()
                        log(f"💾 gallery 已持久化({len(_face_pipeline.store.identities)} 身份)")
                    except Exception as _e:
                        log(f"⚠ gallery 落盘失败:{type(_e).__name__}")
                if _asr_stream is not None:
                    try:
                        _asr_stream.stop()
                    except Exception:
                        pass
                if _asr_agg is not None:
                    try:
                        _asr_agg.close()
                    except Exception:
                        pass
                if _asd_engine is not None:
                    _asd_engine.stop()
        finally:
            try:
                mini.set_automatic_body_yaw(True)
            except Exception as _e:
                log(f"⚠ 恢复 automatic_body_yaw 失败(身体可能仍为手动):{type(_e).__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    print("hello")
