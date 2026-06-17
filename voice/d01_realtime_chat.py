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
from pathlib import Path
from dotenv import load_dotenv

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
import sys
import threading
import time
import urllib.request
from collections import deque

# VIS_DEBUG 日志环形缓冲区（/state.json 消费）
_vis_log_buf: "deque[tuple[int, str]]" = deque(maxlen=1000)
_vis_log_seq: int = 0

import numpy as np
from PIL import Image
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation as R

import dashscope
from dashscope.audio.qwen_omni import (
    AudioFormat,
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)
from openai import OpenAI
import sherpa_onnx                       # WAKE-01:唤醒词 KWS(本地、离线)
from reachy_mini import ReachyMini

# MediaPipe 不在主进程导入(TRACK-FIX):检测在 vision_worker 子进程跑,独立 GIL。
# mediapipe 可用时用 vision_worker.py(手势+指向);否则降级到 vision_worker_cv.py(HSV后备)。
try:
    import mediapipe as _mp_probe  # noqa: F401 — 仅探测可用性
    del _mp_probe
    from vision_worker import vision_worker as _vision_worker_fn
    _VISION_BACKEND = "mediapipe"
except ImportError:
    from vision_worker_cv import vision_worker as _vision_worker_fn
    _VISION_BACKEND = "opencv"

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

# ───────────────────────── 配置 ─────────────────────────
MODEL = "qwen3.5-omni-flash-realtime-2026-03-15"
VISION_MODEL = "qwen3.5-omni-plus"   # take_snapshot 看图用(chat.completions 回合制)
VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VOICE = "Ethan"
INSTRUCTIONS = (
    "你是桌面机器人,名字叫'小艺'(用户给你起的中文名),有真实的身体(头、天线)和一台摄像头。"
    "别人问你叫什么、喊'小艺'时,你都以小艺自居;不要自称 Reachy Mini(那只是你的硬件型号)。"
    "用简体中文、口语化、简短地回答,一般不超过两三句话。"
    "回答时自然地配合动作工具表达身体语言:打招呼/同意时点头,否定时摇头,"
    "开心/兴奋/被夸时摆天线,好奇/疑惑时歪头。"
    "重要:做动作时必须同时用语音回应,边说边做;绝不要默默做动作不说话。"
    "用户让你看东西时调用 take_snapshot,拿到画面描述后用自己的话自然地告诉用户你看到了什么。"
    "当用户用手指指着某个东西问'这是什么''我指的是什么''这个呢'之类、需要判断他指向哪个物体时,"
    "调用 identify_pointed_object,拿到结果后自然地说出他指的那个东西。"
)

SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")  # 快照存放(已 gitignore)

OUT_SR = 24000   # Realtime 下行采样率
PLAY_SR = 16000  # Reachy 播放管线 appsrc 固定 16kHz
JITTER_S = 0.30
JITTER_WALL_S = 0.50

# idle 微动(说话时的"活着感",O-01a-2)
IDLE_HZ = 25.0
IDLE_YAW_AMP = 2.5    # 度(无人脸时的原幅度)
IDLE_PITCH_AMP = 1.5  # 度
IDLE_YAW_F = 0.20     # Hz
IDLE_PITCH_F = 0.30   # Hz
IDLE_TAU = 0.5        # 包络时间常数(s)
TRACK_SWAY_SCALE = 0.4  # 跟随中微动缩放(小幅叠加,不和跟随打架)

# ── 本地视觉跟随(VIS-01 → F-01 融合,参数与教训见 CALIBRATION §9)──
_VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vision", "models")
VIS_MODEL_PATH = os.path.join(_VIS_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(_VIS_DIR, "hand_landmarker.task")
VIS_MAX_FPS = 40.0   # 检测限频(检测在独立进程,不再让主进程分担,放回 40)
VIS_MISS_N = 5       # 连续 N 帧漏检才算"真丢脸"(单帧漏检不重置滤波,防侧脸闪断)
DECIMATE = 3         # 1920×1080 → ::3 整数抽样 → 640×360
FOV_X_DEG = 65.0
FOV_Y_DEG = 40.0
# ⭐ 铁律:增益必须时间常数型 step = err × (1 − exp(−dt/TAU)),与帧率解耦。
#   按"每帧吃固定比例"在高帧率下等效角速度爆表 + 相机 ~100ms 延迟 → 限位间打摆。
TRACK_TAU = 0.40       # 收敛时间常数(s)
TRACK_DEADBAND = 2.0   # 误差死区(度),防微抖
TRACK_MAX_STEP = 1.5   # 单帧最大步进(度)
TRACK_YAW_LIMIT = 25.0
TRACK_PITCH_LIMIT = 15.0
LOST_HOLD_S = 1.5      # 丢脸后保持朝向时长,超时缓慢回中
RETURN_TAU = 0.8       # 回中时间常数(s)(同样与帧率解耦)
YAW_SIGN = -1.0        # 画面右(u>0.5)= 机器人右边 → yaw 负(摄像头不镜像)
PITCH_SIGN = +1.0      # 画面下 → pitch 正(低头)
# 手势合成安全箱:手势 offset 叠加跟随基准后,yaw 裁剪到身体±箱,pitch 绝对裁剪
GES_YAW_BOX = 25.0   # 相对身体
GES_PITCH_BOX = 16.0

# ── 声源转向(FUSION-02;DOA 要点与参考系教训见 CALIBRATION §11)──
DOA_URL = "http://127.0.0.1:8000/api/state/doa"
DOA_POLL_HZ = 10.0
SND_WIN_S = 1.5            # DOA 中值窗口(VAD 触发率实测仅 11~57%,窗口要够长)
SND_MIN_SAMPLES = 5        # 窗口最少有声样本(压反射双峰)
SND_RESID_MIN = 25.0       # 残差超过此值才算"视场外声源"(触发门槛)
SND_DONE_RESID = 10.0      # 链式跳间:残差小于此值=已对准声源,不再转
DOA_DEBUG = os.environ.get("DOA_DEBUG") == "1"  # 诊断:打印每次 DOA raw/resid/IQR/confident/speaking
# M1.5-a.5:长窗稳健 DOA + 可信度(老 SND_* 窗不动,这是另开的一条)
DOA_WIN_S = 2.0            # 长观测窗(压随机离群,比 SND_WIN_S 长)
DOA_MIN_SAMPLES = 6        # 长窗最少样本才出稳定方向
GATE_SPREAD = 25.0         # 可信阈:长窗 IQR < 此值 = 稳(可用区 3-15° 判 True;镜像翻转 ~90° 判 False)
# M1.5-a 方向门控:engaged 只收头朝向 ±GATE_DEG 内语音;只挡"确信范围外",其余放行
GATE_DEG = 55.0            # 门控半角(±55°=110°宽);可调
DOA_GATE_FRESH_S = 1.5     # 门控用 doa_at 的新鲜窗
# M1.5-b 二次唤醒切换
SWITCH_COOLDOWN_S = 2.0    # 切换冷却:刚切过去这么久内不响应新唤醒(防来回抽搐)
SWITCH_AWAY_DEG = 35.0     # 切换转身:头离开A方向超过此角才放开认脸(途中无视任何脸,防被A拽回)
SWITCH_SETTLE = 8.0        # 到目标角的判定容差
SWITCH_TIMEOUT_S = 8.0     # 切换总超时(转+扫都没锁到B)→ 切换失败,回A(含直转+附近扫)
SWITCH_COARSE_DEG = 70.0   # 三档切换:tier2/3 粗方向转离A的角度(足够远离A、在物理范围内)
SND_FACE_FRESH_S = 1.2     # 最近见脸 < 此值 → 视觉在跟,DOA 不抢
SND_MAX_HOPS = 3           # 一次事件最多链式转几跳(后方镜像角逐跳收缩,可达背后)
SND_WAIT_FACE_S = 2.0      # 每跳后等人脸进视野的时长
SND_COOLDOWN_S = 6.0       # 事件失败后的冷却(防无限转)
SND_SPEED_DPS = 90.0       # 转向角速度(度/秒)
SND_TARGET_LIMIT = 110.0   # 世界系目标限幅(身体 ±90 + 颈 ±23 以内)
BODY_LIMIT_DEG = 90.0      # 身体转动限幅
NECK_REL_LIMIT = 23.0      # 颈(Stewart)相对身体限幅(25° 留 2° 余量)

# ── 行为状态机(FUSION-03;behavior_loop 统一调度,其余线程降级为传感器+执行器)──
ST_ARMED = "ARMED"         # WAKE-01 待命:只听唤醒词"小艺",不连 Qwen,慢呼吸;命中才进 engaged
ST_IDLE = "IDLE_CENTER"    # 中立待命:头回正 + 微动,监听声音/人脸
ST_ENGAGING = "ENGAGING"   # 转向声源 + 主动扫头找人
ST_TRACKING = "TRACKING"   # 视觉稳定跟随 + 对话/动作/微动
ST_SEARCHING = "SEARCHING"  # 短暂找回:原地等,有声音则 DOA 辅助
ST_RETURNING = "RETURNING"  # 平滑回中位
FACE_FRESH_S = 0.4         # 人脸"新鲜"判定(瞬时)
LOCK_ON_S = 0.3            # 迟滞:持续命中多久才算"锁定"(防单帧误触发进 TRACKING)
LOCK_OFF_S = 1.5           # 迟滞:持续丢失多久才算"丢锁"(= LOST_HOLD,防瞬断退出 TRACKING)
ENGAGE_TIMEOUT_S = 6.0     # ENGAGING 总超时(转向+扫描都没找到 → 放弃)
ENGAGE_SCAN_RANGE = 15.0   # 转到声源后,主动扫头找人的幅度(度)
ENGAGE_SCAN_TIME_S = 3.0   # 扫头找人时长
SEARCH_TIMEOUT_S = 4.0     # SEARCHING 超时 → 回中位
NO_INTERACT_S = 15.0       # engaged 无说话互动多久 → 回 armed 待命(WAKE-01 起从"回中"改"回待命")
FSM_HZ = 25.0              # behavior_loop 频率

# ── WAKE-01 唤醒词(详见 CALIBRATION §14;standalone 标定见 tools/wake01_kws_standalone.py)──
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KWS_MODEL_DIR = os.path.join(_REPO, "tools", "_kws_models",
                             "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01")
KWS_KEYWORDS = os.path.join(_REPO, "tools", "_kws_models", "keywords_d01.txt")  # 运行时生成(gitignore)
KWS_FORMS = ["x iǎo y ī", "x iǎo y ìn", "x iǎo y ì"]   # 单字"小艺"三声调形态(真人落点实测)
KWS_SINGLE_THR = 0.17      # 锁定值:召回~9/10,误触地板~0.7/5min(电视同音,留给 M1.5 DOA 门控)
KWS_DEBOUNCE_S = 0.3       # 同一声"小艺"点亮多形态行算一次
KWS_REFRACTORY_S = 2.0     # 唤醒后吞余波,防同次重复
CONNECT_TIMEOUT_S = 3.0    # (b)命中才连:connect+session.updated 超时(实测~400ms),超时→失败反馈回 armed
# armed 慢呼吸(用户定:活着但在休息,比 engaged 微动更轻更慢)
ARMED_BREATH_F = 0.18      # Hz(周期~5.5s)
ARMED_BREATH_PITCH = 2.5   # 度(小幅低头起伏)
# 唤醒确认动作(M1c-a):heard=听到(上扬)、fail=连失败(下垂);一上一下一眼区分。CLI 可调 --cue-*
# 叠加偏置(不抢渲染、可被转向打断);单槽后到覆盖先到。pitch+ =低头,故"抬头"取负。天线 +=上扬 -=下垂。
CUE = {
    "heard_dur": 0.45, "heard_pitch": 7.0, "heard_ant": 0.5,   # 上扬 "嗯?在。"
    "fail_dur": 0.80,  "fail_pitch": 6.0,  "fail_ant": 0.7,    # 下垂 "没连上。"
    "giveup_dur": 0.40, "giveup_pitch": 3.5, "giveup_ant": 0.35,  # 轻微下沉 "咦?没人。"(比 fail 更轻)
    "bye_dur": 0.45,   "bye_pitch": 3.5,   "bye_ant": 0.45,    # 收束告别(EXIT-01;与 heard 上扬首尾呼应)
}
# 命中转向(M1c-b,策略 A;阈值依据见 CALIBRATION §14 DOA 实测)
SPREAD_BAD = 40.0          # DOA 窗 IQR ≥ 此值 = 深后翻转不稳 → 坏区走宽扫(实测:可用≤15° / 深后~90°)
WIDE_SCAN_RANGE = 88.0     # 宽扫可达弧(±度;受物理 ±90° 限,真正后 >90° 够不着)
WIDE_SCAN_HZ = 0.18        # 宽扫正弦频率(慢,配合人脸检测帧率,别扫太快漏脸)
WIDE_SCAN_TIME_S = 7.0     # SEEK 宽扫多久没脸 → 放弃回 armed
DOA_WAKE_FRESH_S = 1.5     # SEEK 起扫方向取 DOA 的新鲜窗(只用符号当弱提示)
# SEEK 的 pitch 覆盖(解决"人比摄像头高、水平扫漏脸";pitch+ 为低头,故抬头取负)
SEEK_PITCH_UP = -6.0       # 抬头偏置(度)
SEEK_PITCH_AMP = 6.0       # pitch 慢摆幅度(覆盖 约 0~-12° 高度范围)
SEEK_PITCH_HZ = 0.30       # pitch 摆动频率(慢,配合人脸检测)
# SEEK 两阶段(confident 直转→附近找脸→全场扫兜底)
SEEK_NEARBY_DEG = 25.0     # 阶段二:到位后附近扫范围(±度)
SEEK_NEARBY_TIME_S = 2.5   # 阶段二:附近扫多久没脸 → 退化全场扫
SEEK_SUPPRESS_DEG = 12.0   # 阶段一:压锁豁免——|resid|<此值视为"正前",不压锁直接秒锁
# 唤醒应答(WAKE-01 后续①):SEEK 锁脸那刻,让模型说一句简短招呼(模型生成,天然轮换;克制)
GREET_PHRASES = ["在呢", "来啦", "你好呀", "我在", "嗨,你好", "诶,在的", "怎么啦"]  # 轮换,避免每次同一句
def greet_prompt(phrase: str) -> str:
    return (f"用户刚出现在你面前(你刚找到他)。用中文口语自然地说一句简短招呼,"
            f"就说「{phrase}」的意思(可带个语气词,保持很短);**别用英文、别解释、别提'找到你了'**。")

# 退出指令(EXIT-01):告别语代码轮换(沿用①教训,变异靠代码不靠 prompt);短语嵌进 function_call_output
BYE_PHRASES = ["好的", "拜拜", "休息啦", "我先歇会儿", "回头见", "去忙啦", "嗯,先这样"]
EXIT_MIN_S = 1.5           # 退出:回 armed 前至少等这么久(让告别语播出来)
EXIT_MAX_S = 6.0           # 退出:封顶,告别再长/卡住也强制回 armed(不挂死)

# ── 指向转头(POINT-02-b):手指 2D 方向 → 头部转角 ──
ST_POINTING = "POINTING"   # 指向转头中(转完 → snapshot → 回 TRACKING)
POINT_FRESH_S = 1.2        # 食指方向"新鲜"判定(behavior 读最近一次手部检测)
POINT_YAW_GAIN = 38.0      # 水平指向 → yaw 转角(度;转头不求精确,把目标转进画面即可)
POINT_PITCH_GAIN = 12.0    # 垂直指向 → pitch 转角(度)
POINT_TURN_TIMEOUT_S = 2.5  # 转头封顶时长
POINT_SETTLE_S = 0.6       # 转到位后停稳多久再抓帧(让电机+相机稳定,目标居中)
POINT_HOLD_MAX_S = 4.0     # 抓帧后最多保持朝向多久(等看图描述返回,期间不被拽回人脸)

# ── 手部互动"逗它"(PLAY-01-b):近处大手吸引注意力 → PLAYING 跟手 + 开心表达 ──
# 跟手参数全部来自 standalone 六轮实测调校(vision/play01_hand_track.py)
ST_PLAYING = "PLAYING"     # 逗它中:头跟手 + 天线开心摆(手势/指向优先级更高)
PLAY_SIZE_ON = 0.30        # 手 bbox 最大边占画面比 ≥ 此值(够近)才可进入
PLAY_SIZE_OFF = 0.22       # 跟踪/保持下限(迟滞;更小的"手"= 背景误检,源头过滤)
PLAY_SCORE_MIN = 0.6       # handedness score 当置信度(真手>0.9,背景误检<0.6)
PLAY_ON_S = 0.3            # 持续够大才进入(防路过挥手误触)
PLAY_OFF_S = 1.5           # 近手消失持续此值才退出(手怼太近检测会连丢 1s+)
PLAY_FRESH_S = 0.4         # 手读数"新鲜"判定
# "晃"才是逗(用户 spec 原文"近处中心晃=逗它";实测教训:托下巴的静止手会误触发→抬头盯手)
PLAY_MOVE_WIN_S = 0.8      # 手部运动量统计窗口
PLAY_MOVE_MIN = 0.08       # 窗口内位移(画面占比)≥ 此值才算"在晃"(托下巴抖动 <0.03)
PLAY_STILL_S = 4.0         # 逗它中手静止超过此值 → 没意思了,回跟脸(猫不盯不动的逗猫棒)
PLAY_TAU = 0.25            # 跟手收敛常数(比跟脸 0.40 灵敏,逗它要跟得上)
PLAY_MAX_STEP = 3.0        # 单帧步进上限(≈90°/s,与听声转头同速)
PLAY_AMP = 0.90            # 幅度系数(用户验收:幅度收 10% 手感最佳)
PLAY_YAW_LIMIT = TRACK_YAW_LIMIT * PLAY_AMP     # 相对身体 ±22.5°
PLAY_PITCH_LIMIT = TRACK_PITCH_LIMIT * PLAY_AMP
PLAY_COAST_S = 0.35        # 快手丢检测时惯性外推时长(像猫预判逗猫棒)
PLAY_COAST_DU = 0.20       # 外推位移封顶(画面占比;不封顶会甩到画面边)
PLAY_COAST_VEL = 2.0       # 外推速度钳位(/s;防检出跳变的速度尖峰)
PLAY_JOY_DELAY_S = 5.0     # 持续逗它这么久后才第一次摇天线(用户反馈:一进就摇很怪)
PLAY_JOY_PERIOD_S = 7.0    # 之后每隔此值小摇一次(有节奏不神经质)
PLAY_JOY_FLICK_S = 0.6     # 小天线摆时长
PLAY_REENTRY_S = 3.0       # 退出后这么久内再进入算"继续逗",不重置节拍(防进出抖动)


def log(msg: str) -> None:
    global _vis_log_seq
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _vis_log_seq += 1
    _vis_log_buf.append((_vis_log_seq, line))


# ───────────────────────── 动作库(CALIBRATION.md §2 标定参数)─────────────────────────
INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]


def head_pose(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


def gpose(yaw: float, pitch: float, body: float, roll: float = 0.0) -> np.ndarray:
    """手势姿态 = 跟随基准 + 手势 offset;yaw 裁剪到身体±箱(颈不顶限),pitch 绝对裁剪。"""
    return head_pose(
        pitch_deg=float(np.clip(pitch, -GES_PITCH_BOX, GES_PITCH_BOX)),
        yaw_deg=float(np.clip(yaw, body - GES_YAW_BOX, body + GES_YAW_BOX)),
        roll_deg=roll,
    )


# 所有手势签名 (mini, base_yaw, base_pitch, body):以当前跟随姿态为基准做、做完回基准;
# ⭐ body_yaw 必须传当前身体朝向(传 0 会把转过去的身体拽回正前)
def act_nod(m: ReachyMini, by: float, bp: float, body: float) -> None:
    brad = math.radians(body)
    for _ in range(2):
        m.goto_target(gpose(by, bp + 15, body), duration=0.35, body_yaw=brad)
        m.goto_target(gpose(by, bp - 10, body), duration=0.35, body_yaw=brad)
    m.goto_target(gpose(by, bp, body), duration=0.35, body_yaw=brad)


def act_shake(m: ReachyMini, by: float, bp: float, body: float) -> None:
    brad = math.radians(body)
    for _ in range(2):
        m.goto_target(gpose(by + 15, bp, body), duration=0.35, body_yaw=brad)
        m.goto_target(gpose(by - 15, bp, body), duration=0.35, body_yaw=brad)
    m.goto_target(gpose(by, bp, body), duration=0.35, body_yaw=brad)


def _look(m: ReachyMini, by: float, bp: float, body: float,
          yaw_off: float = 0.0, pitch_off: float = 0.0) -> None:
    """看向某方向(相对身体正前的偏向),看完回跟随基准。"""
    brad = math.radians(body)
    m.goto_target(gpose(body + yaw_off, pitch_off, body), duration=0.6, body_yaw=brad)
    time.sleep(0.8)
    m.goto_target(gpose(by, bp, body), duration=0.6, body_yaw=brad)


def act_wiggle(m: ReachyMini, by: float, bp: float, body: float) -> None:
    brad = math.radians(body)
    for _ in range(2):
        m.goto_target(antennas=[+0.8, -0.8], duration=0.3, body_yaw=brad)
        m.goto_target(antennas=[-0.8, +0.8], duration=0.3, body_yaw=brad)
    m.goto_target(antennas=INIT_ANTENNAS, duration=0.35, body_yaw=brad)


def act_tilt(m: ReachyMini, by: float, bp: float, body: float) -> None:
    brad = math.radians(body)
    m.goto_target(gpose(by, bp, body, roll=15), duration=0.5, body_yaw=brad)
    time.sleep(0.8)
    m.goto_target(gpose(by, bp, body), duration=0.5, body_yaw=brad)


ACTIONS = {
    "nod": act_nod,
    "shake_head": act_shake,
    "look_left": lambda m, by, bp, body: _look(m, by, bp, body, yaw_off=+16),
    "look_right": lambda m, by, bp, body: _look(m, by, bp, body, yaw_off=-16),
    "look_up": lambda m, by, bp, body: _look(m, by, bp, body, pitch_off=-16),
    "look_down": lambda m, by, bp, body: _look(m, by, bp, body, pitch_off=+16),
    "wiggle_antennas": act_wiggle,
    "tilt_head": act_tilt,
}

_NOPARAM = {"type": "object", "properties": {}}
TOOLS = [
    {"type": "function", "name": "nod", "description": "点头。打招呼、同意、确认、答应请求时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "shake_head", "description": "摇头。否定、拒绝、不同意、说'不'时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_left", "description": "把头转向左边看。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_right", "description": "把头转向右边看。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_up", "description": "抬头看上方。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_down", "description": "低头看下方。", "parameters": _NOPARAM},
    {"type": "function", "name": "wiggle_antennas", "description": "欢快地摆动头顶天线。表达开心、兴奋、被夸奖、热情时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "tilt_head", "description": "歪头。表达好奇、疑惑、思考、没听懂时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "end_session",
     "description": "结束本次对话、让机器人回到待命休息。仅当用户【明确表达要结束对话/让你退下/离开】时才调用,"
                    "例如「走吧」「退下」「你先忙」「没事了」「拜拜」「不聊了」「先这样」「就到这」。"
                    "⚠️ 注意:「再说吧」「这个先放一边」「等会儿」「待会聊」「先放着」「回头说」等只是话题搁置或语气词,"
                    "【不是】结束对话,绝不要因此调用;拿不准时继续对话、不要调。",
     "parameters": _NOPARAM},
    {"type": "function", "name": "take_snapshot",
     "description": "用摄像头拍一张当前画面并理解内容。当用户让你看东西、问'你看到什么''我手里是什么'等需要视觉、但不涉及'指向'的问题时调用。",
     "parameters": _NOPARAM},
    {"type": "function", "name": "identify_pointed_object",
     "description": "当用户用手指指向画面中某个物体、问'这是什么''我指的是什么''这个是啥'等需要判断他指向哪个物体时调用。会拍照并理解用户手指指向的目标。",
     "parameters": _NOPARAM},
]

# 看图 prompt(POINT-01)。两个都做"指向感知":工具路由从音频判断"是否指向"不可靠
# (实测模型对"你看这是什么"会选通用看图),故通用 prompt 也兜底处理指向。
_POINT_GUIDE = ("如果用户正在用手指指向画面中某个物体(看手的朝向、伸出的食指延长线),"
                "请重点判断并明确说出他指的是哪一个物体、那是什么;")
SNAP_PROMPTS = {
    "scene": ("你是机器人的眼睛。用简体中文两三句话回答。" + _POINT_GUIDE +
              "否则描述画面主要内容,特别是人手里举着或拿着的物体(若有)。"),
    "point": ("你是机器人的眼睛。用户正在用手指指向画面中的某个物体。"
              "请仔细观察用户手指的指向(手的朝向、伸出的食指延长线),"
              "判断用户指的是哪一个物体,用简体中文两三句话明确说出那个物体是什么并简要描述。"
              "如果画面里没有看到明显的指向手势,就说你不太确定他指的是哪个,并描述画面里最可能的几个物体。"),
    # 两段式指向(用户定的根治方案):先原地看图,让 VLM 判断"是否真在指/目标是否已在画面",
    # 确认在指且目标不在画面才转头。本地关键点只做廉价提示,不再决定是否转头/往哪转
    # (实测食指延长线 2D 角度噪声大,会算出莫名的"上"分量 → 错误抬头)。
    "judge": ("你是机器人的眼睛。用户刚问了类似'这是什么'的问题。"
              "请只输出一个 JSON 对象,不要输出任何其他文字、不要代码块标记:\n"
              '{"pointing": true或false, "target_visible": true或false, '
              '"direction": "左|右|上|下|左上|左下|右上|右下|无", "desc": "..."}\n'
              "字段含义:pointing=画面中用户是否正在用手指明确指向某个东西;"
              "target_visible=他所指的目标物体是否完整清晰地出现在画面里(目标在画面外、"
              "在边缘被切掉、或顺着手指方向看不到具体目标都算 false);"
              "direction=顺着手指延长线,目标相对画面中心在哪个方向(画面坐标;没在指就填'无');"
              "desc=用简体中文两三句话:若 pointing 为 true 且 target_visible 为 true,"
              "明确说出他指的物体是什么并简述;若 pointing 为 false,正常描述画面主要内容"
              "(特别是人手里拿着的物体);若目标不在画面里,desc 留空字符串。"),
}

# VLM 粗方向 → 头部转角(画面坐标:右 → yaw 负,下 → pitch 正,同关键点映射约定)
_DIR_MAP = {
    "左": (+30.0, 0.0), "右": (-30.0, 0.0),
    "上": (0.0, -10.0), "下": (0.0, +10.0),
    "左上": (+22.0, -8.0), "右上": (-22.0, -8.0),
    "左下": (+22.0, +8.0), "右下": (-22.0, +8.0),
}


def parse_judge(raw: str) -> dict | None:
    """宽容解析 VLM 的 judge JSON(剥代码块/取首尾大括号);失败返回 None。"""
    s = raw.strip()
    if "```" in s:
        s = s.replace("```json", "```").split("```")[1] if s.count("```") >= 2 else s
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(s[i:j + 1])
        return d if isinstance(d, dict) else None
    except Exception:
        return None


# ───────────────────────── One Euro 滤波(VIS-01 验证参数)─────────────────────────
class OneEuroFilter:
    """标准 One Euro:低速强平滑防抖,高速低延迟跟手。丢脸后必须 reset。"""

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.08, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0
        self.t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(1e-3, t - self.t_prev)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


# ───────────────────────── 共享状态 ─────────────────────────
class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_updated = threading.Event()
        # 播放 / 打断
        self.play_gen = 0
        self.drop_audio = False
        self.in_flight = 0
        self.playback_end_estimate = 0.0
        # function calling 协调(O-01a 修复2:即时回 output + 纯动作响应立即补话)
        self.resp_audio_count = 0
        self.fc_seen_this_resp = False
        self.fc_gen = 0
        # 行为状态机(FUSION-03)
        self.state = ST_IDLE           # 当前行为状态(behavior_loop 唯一写;main 见 ST_ARMED 转换管 WS)
        self.wake_ok = False           # WAKE-01:main 连接成功后置一次性信号,behavior 读后清→离开 ARMED
        self.wake_cue = None           # 唤醒确认动作:None/"heard"(上扬)/"fail"(下垂);main 写,head_control 读
        self.wake_cue_t = 0.0          # 上述 cue 的起始时刻(head_control 据此算包络)
        self.greet_now = False         # 唤醒应答:SEEK 锁脸那刻置(behavior 写),main 消费→让模型招呼一句
        self.exit_request = False      # EXIT-01:end_session 工具置(ChatCallback 写 flag),behavior 读→回 armed(不破单写者)
        self.switch_request = None     # M1.5-b:engaged 范围外二次唤醒,main 置 {resid,confident},behavior 读→转向新人 B
        self.action_active = False     # Primary 手势执行中 → 头控让位给 goto
        self.track_yaw = 0.0           # 头的【世界】朝向目标(TRACKING 由视觉积分,其余由 behavior 驱动)
        self.track_pitch = 0.0
        self.body_yaw_deg = 0.0        # 身体当前朝向(度)
        self.face_seen_at = 0.0        # 最近一次检出人脸的时刻(瞬时)
        self.face_locked = False       # 迟滞后的"稳定有脸"判定(behavior 用它做 TRACKING 进出)
        self.last_interaction_at = 0.0  # 最近一次说话互动(用户/机器人)→ RETURNING 计时
        self.sound_resid = None        # DOA 传感器:置信的视场外声源残差(度),无则 None
        self.sound_at = 0.0            # 上述读数的时刻
        self.sound_spread = 0.0        # DOA 窗内 raw 的 IQR(p75−p25):稳定性,坏区(深后翻转)判据
        self.wake_doa = None           # 命中瞬间快照 {resid,spread,fresh}:M1c 转向分流用(main 写,behavior 读)
        # M1.5-a.5:长窗稳健 DOA + 可信度(给 M1.5-b/方向门控用;老 sound_* 不动)
        self.doa_resid_stable = None   # 长窗(DOA_WIN_S)中值残差(度);无则 None
        self.doa_confident = False     # 可信度:长窗 IQR<GATE_SPREAD 且样本够(=稳,非=对)
        self.doa_at = 0.0              # 上述长窗读数时刻(消费方判 fresh)
        # 指向(POINT-02):视觉子进程发布的最近食指方向 + 待处理的指向请求
        self.finger_angle = None       # 食指画面系角度(度),无则 None
        self.finger_at = 0.0
        self.finger_extended = False
        self.finger_ext_at = 0.0       # 最后一次见到"伸出的食指"的时刻(粘滞:检测 ~6Hz 会闪烁)
        # 逗它(PLAY-01):最近一次"近手"读数(已过 score+size 双门,小误检不入)
        self.hand_u = 0.5
        self.hand_v = 0.5
        self.hand_size = 0.0
        self.hand_at = 0.0
        self.hand_move = 0.0           # 近手在 PLAY_MOVE_WIN_S 窗口内的位移(晃动量)
        self.point_request = None      # {"call_id","gen"}:模型调了 identify_pointed_object,待转头看图
        self.snap_grabbed = False      # snapshot_loop 抓到帧的握手(POINTING 等它为真才许转回)
        # 帧共享(视觉线程是唯一持续 get_frame 者;take_snapshot 读这里)
        self.latest_frame = None
        self.latest_frame_t = 0.0
        # take_snapshot:进行中的快照数(挂起时 response.done 不补话,等描述回来)
        self.snapshot_pending = 0
        # VIS_DEBUG 专用：视觉子进程实际看到的降采样帧 + 最新检测结果（VIS_DEBUG=1 时写入）
        self.dbg_frame_small = None   # ndarray RGB H×W×3，与视觉子进程收到的帧相同
        self.dbg_det = None           # dict {"face":(u,v,h)|None, "hand":{...}|None, "n_faces":int}
        # VIS_DEBUG 专用：DOA 可视化字段（DEBUG-02，由 behavior_loop/main loop 写入）
        self.dbg_gate_open = True     # 当前 M1.5-a 门控状态（True=放行，False=静音）
        self.dbg_switching = False    # 当前是否在 M1.5-b 切换中
        self.dbg_switch_phase = ""    # "turn" / "nearby" / "sweep" / ""
        self.dbg_switch_target = 0.0  # 切换目标角（世界坐标，度）
        # 手势状态（GESTURE-01，仅 onnx/mediapipe backend 填充）
        self.gesture = None        # 最新手势 str 或 None
        self.gesture_at = 0.0      # 上次检出有效手势的时刻
        self.gesture_fingers = 0   # 最新手指数


# ───────────────────────── ①对话:回调,收服务端事件(打断/工具分发/计时器喂养)─────────────────────────
class ChatCallback(OmniRealtimeCallback):
    def __init__(self, st: State, play_q: "queue.Queue", motion_q: "queue.Queue",
                 snap_q: "queue.Queue", mini: ReachyMini):
        self.st = st
        self.play_q = play_q
        self.motion_q = motion_q
        self.snap_q = snap_q
        self.mini = mini
        self.conv: OmniRealtimeConversation | None = None
        self.exit_i = 0   # EXIT-01 告别语轮换索引

    def on_open(self) -> None:
        log("✅ WebSocket 已连接 dashscope.aliyuncs.com")

    def on_close(self, close_status_code, close_msg) -> None:
        log(f"🔌 连接关闭:code={close_status_code} msg={close_msg}")

    def _do_barge_in(self, in_flight: bool) -> None:
        """打断:作废队列 → flush 管线残余 → 必要时取消在途回复。动作/跟随不中断。"""
        st = self.st
        with st.lock:
            st.play_gen += 1
            st.drop_audio = True
            st.playback_end_estimate = time.monotonic()
        while True:
            try:
                self.play_q.get_nowait()
            except queue.Empty:
                break
        try:
            self.mini.media.audio.clear_player()
        except Exception as e:
            log(f"⚠ clear_player 失败:{type(e).__name__}: {e}")
        if in_flight and self.conv is not None:
            self.conv.cancel_response()
        log("⛔ 打断:已停止播放" + (",并取消在途回复" if in_flight else ""))

    def on_event(self, event) -> None:  # SDK 实际传入已解析的 dict
        st = self.st
        try:
            etype = event.get("type", "")
            now = time.monotonic()
            if etype == "session.created":
                log(f"✅ 会话已建立 session_id={event['session']['id']}")
            elif etype == "session.updated":
                log("✅ 会话配置生效(semantic_vad / 8 动作 + take_snapshot + identify_pointed_object 已注册)")
                log("▶ 可以对机器人说话了;它说话时可随时插话打断(Ctrl+C 退出)")
                st.session_updated.set()
            elif etype == "input_audio_buffer.speech_started":
                with st.lock:
                    st.last_interaction_at = now  # 用户开口 → 喂 RETURNING 计时器
                    playing = (now < st.playback_end_estimate) or (not self.play_q.empty())
                    in_flight = st.in_flight > 0
                log("🎤 检测到你开始说话…")
                if playing or in_flight:
                    self._do_barge_in(in_flight)
            elif etype == "input_audio_buffer.speech_stopped":
                log("🤫 检测到你说完了,等模型回应…")
            elif etype == "conversation.item.input_audio_transcription.completed":
                log(f"📝 听到的是:「{(event.get('transcript') or '').strip()}」")
            elif etype == "response.created":
                with st.lock:
                    st.in_flight += 1
                    st.drop_audio = False
                    st.resp_audio_count = 0
                    st.fc_seen_this_resp = False
                    st.last_interaction_at = now  # 机器人开口 → 喂 RETURNING 计时器
                log("💭 模型开始生成回复…")
            elif etype == "response.function_call_arguments.done":
                name = event.get("name", "")
                call_id = event.get("call_id", "")
                with st.lock:
                    st.fc_seen_this_resp = True
                    st.fc_gen = st.play_gen
                log(f"🤖 模型调用工具: {name}")
                if name == "take_snapshot":
                    # ⭐ 两段式指向(模型路由 + 关键点方向都不可靠,用户定的根治方案):
                    # 本地 1.2s 内见过伸指(廉价提示)或模型自己选了指向工具 → mode="judge"
                    # 先原地看图,由 VLM 判断"是否真在指/目标是否已在画面/粗方向",
                    # 确认在指且目标不在画面才转头(snapshot_loop 升级 point_request)。
                    with st.lock:
                        # 粘滞窗:1.2s 内见过伸出的食指就算可能在指(检测 ~6Hz 会闪烁)
                        maybe_pointing = (time.monotonic() - st.finger_ext_at) < POINT_FRESH_S
                        st.snapshot_pending += 1
                    mode = "judge" if maybe_pointing else "scene"
                    if maybe_pointing:
                        log("👉 最近见过伸指 → 先原地看图判断是否真在指(两段式)")
                    self.snap_q.put({"call_id": call_id, "gen": st.fc_gen, "mode": mode})
                elif name == "end_session":
                    # EXIT-01:把告别词嵌进 function_call_output,让模型在【当前这个 response】里说出来——
                    # 不再单独 create_response(那会和"调 end_session 的这个 response"撞车:"already active response")。
                    # 仍代码轮换短语(沿用①教训)。状态切换交 behavior_loop(只置 flag,不写 st.state)。
                    phrase = BYE_PHRASES[self.exit_i % len(BYE_PHRASES)]
                    self.exit_i += 1
                    try:
                        self.conv.create_item({
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps(
                                {"success": True,
                                 "say": f"对话结束。用中文只说这一句简短告别:「{phrase}」,别追问、别挽留、别加别的。"},
                                ensure_ascii=False),
                        })
                    except Exception as e:
                        log(f"⚠ end_session 回 output 失败:{e}")
                    with st.lock:
                        st.exit_request = True
                    log(f"👋 收到结束意图 → 告别「{phrase}」+ 回待命")
                elif name == "identify_pointed_object":
                    # 指向工具:同样先 judge(模型以为在指 ≠ 真在指,例如托下巴/无手势)。
                    # snapshot_pending 先占位,防 response.done 抢跑。
                    with st.lock:
                        st.snapshot_pending += 1
                    log("👉 收到指向请求 → 先原地看图判断(两段式)")
                    self.snap_q.put({"call_id": call_id, "gen": st.fc_gen, "mode": "judge"})
                else:
                    # 手势:乐观即时回 output → 说话不等动作做完
                    self.motion_q.put({"name": name, "call_id": call_id})
                    try:
                        self.conv.create_item({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"success": True, "action": name}, ensure_ascii=False),
                        })
                    except Exception as e:
                        log(f"⚠ 回 function_call_output 失败:{e}")
            elif etype == "response.audio_transcript.delta":
                print(event.get("delta", ""), end="", flush=True)
            elif etype == "response.audio_transcript.done":
                print(flush=True)
            elif etype == "response.audio.delta":
                with st.lock:
                    if st.drop_audio:
                        return
                    gen = st.play_gen
                    st.resp_audio_count += 1
                b64 = event.get("delta") or event.get("audio") or ""
                pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                f16k = resample_poly(pcm.astype(np.float32) / 32768.0, PLAY_SR, OUT_SR).astype(np.float32)
                self.play_q.put((gen, f16k))
            elif etype == "response.done":
                fire_rc = False
                with st.lock:
                    st.in_flight = max(0, st.in_flight - 1)
                    # 纯动作响应(无音频且没被打断)→ 马上补话,不等动作做完
                    # 快照挂起时跳过:等图像描述回来再让模型开口
                    if (
                        st.fc_seen_this_resp
                        and st.resp_audio_count == 0
                        and st.fc_gen == st.play_gen
                        and st.snapshot_pending == 0
                    ):
                        fire_rc = True
                d = self.conv.get_last_first_audio_delay() if self.conv else None
                log(f"✅ 本轮回复完成{f'(首音频延迟 {d:.0f}ms)' if d else ''}")
                if fire_rc and self.conv is not None:
                    self.conv.create_response()
            elif etype == "error":
                log(f"❌ 服务端错误事件:{event}")
        except Exception as e:
            log(f"❌ on_event 处理异常:{type(e).__name__}: {e}\n   原始事件:{str(event)[:300]}")


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
def snapshot_loop(mini: ReachyMini, st: State, cb: ChatCallback, oai: OpenAI,
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


def vision_result_loop(st: State, result_q, stop: threading.Event) -> None:
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
    hit_run_start = None       # 连续命中起点(锁定迟滞用)
    miss_run_start = None      # 连续丢失起点(丢锁迟滞用)
    locked = False
    n_det = 0
    n_hit = 0
    infer_acc: list[float] = []
    stat_t = time.monotonic()

    while not stop.is_set():
        try:
            msg = result_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if msg.get("kind") == "ready":
            log("👁 视觉子进程就绪(Face 跟随 + Hand 指向,独立 GIL)")
            continue
        now = time.monotonic()
        face = msg.get("face")
        u_raw, v_raw = (face[0], face[1]) if face else (None, None)
        infer_ms = msg.get("face_ms", 0.0)
        n_det += 1
        infer_acc.append(infer_ms)
        # 手部结果(平时降频/近手提频):发布食指方向(指向)+ 近手读数(逗它)
        hand = msg.get("hand")
        hand_near = False
        if hand is not None:
            hand_near = (hand.get("score", 1.0) >= PLAY_SCORE_MIN
                         and hand.get("size", 0.0) >= PLAY_SIZE_OFF)  # 双门:背景误检不入
            with st.lock:
                if hand.get("score", 1.0) >= PLAY_SCORE_MIN:  # 低分假手不发布指向
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
            # 迟滞锁定:持续命中 LOCK_ON_S → locked=True
            miss_run_start = None
            if hit_run_start is None:
                hit_run_start = now
            if not locked and (now - hit_run_start) >= LOCK_ON_S:
                locked = True
            u = fx(u_raw, now)
            v = fy(v_raw, now)
            with st.lock:
                st.face_seen_at = now  # 始终更新(瞬时);behavior 用 face_locked
                st.face_locked = locked
            if integrate:
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
            else:
                t_prev_ctrl = now  # 非积分态:滤波/计时继续走,但不写头部目标
        else:
            miss_streak += 1
            t_prev_ctrl = now
            # 迟滞丢锁:持续丢失 LOCK_OFF_S → locked=False
            hit_run_start = None
            if miss_run_start is None:
                miss_run_start = now
            if locked and (now - miss_run_start) >= LOCK_OFF_S:
                locked = False
                with st.lock:
                    st.face_locked = False
            if miss_streak >= VIS_MISS_N:  # 1c:连续 N 帧漏检才重置滤波(防侧脸闪断)
                fx.reset()
                fy.reset()

        if VIS_DEBUG:
            with st.lock:
                st.dbg_det = {
                    "face": msg.get("face"),
                    "hand": msg.get("hand"),
                    "n_faces": msg.get("n_faces", 0),
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


# ───────────────────────── ③听声转向:DOA 传感器线程(只感知,不动头)─────────────────────────
def _read_doa(opener) -> tuple[float, bool] | None:
    try:
        with opener.open(DOA_URL, timeout=2.0) as r:
            d = json.loads(r.read().decode("utf-8"))
        return math.degrees(float(d["angle"])), bool(d["speech_detected"])
    except Exception:
        return None


# ───────────────────────── VIS_DEBUG：MJPEG HTTP 调试预览服务 ─────────────────────────
def vis_debug_server(st: State, port: int, stop: threading.Event) -> None:
    """VIS_DEBUG=1 时启动 MJPEG HTTP 服务，浏览器打开 http://localhost:{port} 查看实时标注帧。
    画面 = 视觉子进程实际看到的降采样帧（DECIMATE×），叠加：
      蓝框=人脸(u,v,h)  绿框=有效手(score≥阈值)  黄框=低置信度手(可能误检)
      左上角=状态机/头部目标/face_locked  右上角=帧时间戳"""
    import cv2 as _cv2
    import http.server
    import socketserver

    def _build_frame() -> bytes:
        with st.lock:
            rgb = st.dbg_frame_small
            det = st.dbg_det
            state_name = st.state
            ty = st.track_yaw
            tp = st.track_pitch
            locked = st.face_locked
            hand_at = st.hand_at
            # DOA 字段
            doa_resid = st.doa_resid_stable
            doa_conf = st.doa_confident
            doa_at = st.doa_at
            body_yaw = st.body_yaw_deg
            gate_open = st.dbg_gate_open
            sw_active = st.dbg_switching
            sw_phase = st.dbg_switch_phase
            sw_target = st.dbg_switch_target
            speaking = time.monotonic() < st.playback_end_estimate + 0.1

        if rgb is None:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            _cv2.putText(blank, "Waiting for frame...", (20, 180),
                        _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            _, jpg = _cv2.imencode(".jpg", blank)
            return jpg.tobytes()

        bgr = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]

        # ── 人脸框（蓝色）──
        if det and det.get("face") is not None:
            fu, fv, fh = det["face"]
            fw = fh * 0.85  # 估算宽高比
            fx0 = int((fu - fw / 2) * W)
            fy0 = int((fv - fh / 2) * H)
            fx1 = int((fu + fw / 2) * W)
            fy1 = int((fv + fh / 2) * H)
            _cv2.rectangle(bgr, (fx0, fy0), (fx1, fy1), (255, 80, 0), 2)
            label = f"FACE u={fu:.2f} v={fv:.2f} h={fh:.2f} n={det.get('n_faces',1)}"
            _cv2.rectangle(bgr, (fx0, fy0 - 18), (fx0 + len(label) * 9, fy0), (255, 80, 0), -1)
            _cv2.putText(bgr, label, (fx0 + 2, fy0 - 4),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ── 手部框（绿=有效 / 黄=低置信）──
        if det and det.get("hand") is not None:
            h = det["hand"]
            hu, hv, hsize = h.get("u", 0.5), h.get("v", 0.5), h.get("size", 0.0)
            hscore = h.get("score", 0.0)
            half = hsize * max(W, H) / 2
            hx0, hy0 = int(hu * W - half), int(hv * H - half)
            hx1, hy1 = int(hu * W + half), int(hv * H + half)
            valid = hscore >= PLAY_SCORE_MIN and hsize >= PLAY_SIZE_OFF
            color = (0, 200, 0) if valid else (0, 200, 255)  # 绿 / 黄
            tag = "HAND" if valid else "HAND(LOW)"
            _cv2.rectangle(bgr, (hx0, hy0), (hx1, hy1), color, 2)
            fingers = h.get("fingers", -1)
            gesture = h.get("gesture") or ""
            g_str = f" [{gesture}]" if gesture else (f" {fingers}f" if fingers >= 0 else "")
            hlabel = f"{tag} size={hsize:.2f} score={hscore:.2f}{g_str}"
            _cv2.rectangle(bgr, (hx0, hy1), (hx0 + len(hlabel) * 9, hy1 + 18), color, -1)
            _cv2.putText(bgr, hlabel, (hx0 + 2, hy1 + 14),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # ── 左上角：状态机信息（白字黑底）──
        now_s = time.strftime("%H:%M:%S")
        lines = [
            f"[{state_name}]",
            f"yaw={ty:+.1f}deg  pitch={tp:+.1f}deg",
            f"face_locked={'Y' if locked else 'N'}  hand_age={time.monotonic()-hand_at:.1f}s",
            now_s,
        ]
        for i, line in enumerate(lines):
            y = 18 + i * 20
            _cv2.rectangle(bgr, (0, y - 15), (len(line) * 9 + 4, y + 4), (0, 0, 0), -1)
            _cv2.putText(bgr, line, (2, y),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                         (0, 255, 255) if i == 0 else (255, 255, 255), 1)

        # ── 底部诊断行：区分"没检出"和"管线未写入"──
        if det is None:
            diag = "det=None  vision_result_loop not writing (crashed?)"
            diag_color = (0, 0, 255)   # 红
        else:
            face_s = f"face={det['face']}" if det.get("face") else "face=None"
            hand_s = f"hand=size{det['hand']['size']:.2f} score{det['hand']['score']:.2f}" if det.get("hand") else "hand=None"
            diag = f"{face_s}  {hand_s}  n={det.get('n_faces',0)}"
            diag_color = (0, 255, 0) if det.get("face") or det.get("hand") else (100, 100, 100)
        _cv2.rectangle(bgr, (0, H - 22), (W, H), (0, 0, 0), -1)
        _cv2.putText(bgr, diag, (4, H - 6),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.42, diag_color, 1)

        # ── DOA 弧条（底部第二行，高 38px）──
        _doa_h = 38          # DOA 条高度
        _doa_y0 = H - 22 - _doa_h  # 在底部诊断行上方
        _doa_mx = 12         # 左右边距
        _doa_range = 90.0    # ±90°
        _gate_deg = GATE_DEG  # ±55° 门控范围

        def _deg2x(deg: float) -> int:
            return int(_doa_mx + (deg + _doa_range) / (2 * _doa_range) * (W - 2 * _doa_mx))

        # 背景黑条
        _cv2.rectangle(bgr, (0, _doa_y0), (W, H - 22), (20, 20, 20), -1)

        # 门控范围背景：±GATE_DEG 内绿透明叠，外红透明叠
        _gate_x0 = _deg2x(-_gate_deg)
        _gate_x1 = _deg2x(_gate_deg)
        _overlay = bgr.copy()
        _cv2.rectangle(_overlay, (_doa_mx, _doa_y0 + 2), (_gate_x0, H - 24), (0, 0, 80), -1)   # 左侧超范围=红
        _cv2.rectangle(_overlay, (_gate_x1, _doa_y0 + 2), (W - _doa_mx, H - 24), (0, 0, 80), -1)  # 右侧超范围=红
        _cv2.rectangle(_overlay, (_gate_x0, _doa_y0 + 2), (_gate_x1, H - 24), (0, 60, 0), -1)  # 范围内=绿
        _cv2.addWeighted(_overlay, 0.4, bgr, 0.6, 0, bgr)

        # 刻度线：画面左=机器人左(resid+)，画面右=机器人右(resid-)
        for _d in (-90, -60, -30, 0, 30, 60, 90):
            _tx = _deg2x(float(-_d))
            _cv2.line(bgr, (_tx, _doa_y0 + 2), (_tx, _doa_y0 + 8), (120, 120, 120), 1)
            if _d != 0:
                _lbl = f"{_d:+d}"
                _cv2.putText(bgr, _lbl, (_tx - 10, _doa_y0 + 20),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)
            else:
                _cv2.line(bgr, (_tx, _doa_y0 + 2), (_tx, _doa_y0 + 14), (180, 180, 180), 1)

        # body_yaw 三角标（白色，朝下）
        _bx = _deg2x(float(np.clip(-body_yaw, -_doa_range, _doa_range)))
        _tri = np.array([[_bx, _doa_y0 + 2], [_bx - 5, _doa_y0 + 10], [_bx + 5, _doa_y0 + 10]], np.int32)
        _cv2.fillPoly(bgr, [_tri], (220, 220, 220))
        _cv2.putText(bgr, "H", (_bx - 4, _doa_y0 + 10),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)

        # 切换目标角（橙色三角，朝上，切换中才显示）
        if sw_active:
            _sx = _deg2x(float(np.clip(-sw_target, -_doa_range, _doa_range)))
            _stri = np.array([[_sx, H - 25], [_sx - 5, H - 33], [_sx + 5, H - 33]], np.int32)
            _cv2.fillPoly(bgr, [_stri], (0, 130, 255))  # 橙

        # DOA 方向箭头（主指示器，从中央向外）
        _doa_fresh = doa_resid is not None and (time.monotonic() - doa_at) < DOA_GATE_FRESH_S
        if doa_resid is not None:
            _dx = _deg2x(float(np.clip(-doa_resid, -_doa_range, _doa_range)))
            _cy_bar = (_doa_y0 + H - 22) // 2
            if doa_conf and _doa_fresh:
                _arrow_c = (0, 220, 0)   # 绿：confident + fresh
            elif _doa_fresh:
                _arrow_c = (0, 180, 255)  # 橙：fresh 但不 confident
            else:
                _arrow_c = (80, 80, 80)   # 灰：stale
            _cv2.arrowedLine(bgr, (_deg2x(0.0), _cy_bar), (_dx, _cy_bar),
                             _arrow_c, 2, tipLength=0.2)
            _cv2.circle(bgr, (_dx, _cy_bar), 4, _arrow_c, -1)

        # 右下角 DOA 文字状态
        _now_m = time.monotonic()
        _fresh_s = f"{_now_m - doa_at:.1f}s" if doa_resid is not None else "—"
        _resid_s = f"{doa_resid:+.0f}°" if doa_resid is not None else "—"
        _gate_s = "OPEN" if gate_open else "BLOCK"
        _gate_c = (0, 220, 0) if gate_open else (0, 0, 220)
        _spk_s = "SPK" if speaking else ""
        _sw_s = f"SW:{sw_phase}" if sw_active else ""
        _conf_s = "conf" if doa_conf else "unc"
        _doa_line = f"DOA {_resid_s} {_conf_s} {_fresh_s}  gate:{_gate_s}  {_sw_s}  {_spk_s}"
        _txt_w = len(_doa_line) * 8 + 4
        _cv2.rectangle(bgr, (W - _txt_w - 2, _doa_y0 + 2), (W - 2, _doa_y0 + 18), (0, 0, 0), -1)
        _cv2.putText(bgr, _doa_line, (W - _txt_w, _doa_y0 + 14),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.38, _gate_c, 1)

        _, jpg = _cv2.imencode(".jpg", bgr, [_cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpg.tobytes()

    _VIS_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>小艺 Debug</title>
<style>
:root{--bg:#09090f;--card:#131320;--bdr:#1e1e30;--txt:#dde1ea;--muted:#505877;
     --green:#22d3a0;--red:#f25e6b;--orange:#f5a623;--blue:#38bdf8;--purple:#a78bfa;
     --mono:'SF Mono','Fira Code',Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--txt);font:13px/1.4 system-ui,sans-serif;overflow:hidden}
#hdr{display:flex;align-items:center;gap:10px;padding:5px 14px;background:var(--card);
     border-bottom:1px solid var(--bdr);height:40px;flex-shrink:0}
#hdr h1{font-size:14px;font-weight:600;letter-spacing:-.2px}
.badge{padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.ba{background:#1a1a2e;color:var(--muted)}.be{background:#3d2000;color:#fbbf24}
.bt{background:#003d28;color:var(--green)}.bs{background:#001d3d;color:var(--blue)}
.bp{background:#2a0057;color:var(--purple)}.br{background:#3d0010;color:var(--red)}
#dot{margin-left:auto;width:7px;height:7px;border-radius:50%;background:var(--muted);transition:background .4s}
#dot.ok{background:var(--green)}
#main{display:grid;grid-template-columns:1fr 304px;grid-template-rows:1fr 200px;
      gap:5px;padding:5px;height:calc(100vh - 40px)}
#cv{background:#000;border-radius:8px;overflow:hidden;position:relative;
    display:flex;align-items:center;justify-content:center}
#cv img{max-width:100%;max-height:100%;object-fit:contain;display:block}
#cv-lbl{position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,.65);
        padding:2px 8px;border-radius:4px;font:10px var(--mono);color:var(--muted)}
#side{display:flex;flex-direction:column;gap:5px}
.card{background:var(--card);border:1px solid var(--bdr);border-radius:8px;padding:10px}
.ch{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin-bottom:8px}
#rc{flex:1;display:flex;flex-direction:column;min-height:0;overflow:hidden}
canvas{display:block;margin:0 auto}
.sr{display:flex;justify-content:space-between;align-items:center;
    padding:3px 0;border-bottom:1px solid var(--bdr);gap:8px}
.sr:last-child{border:none}
.sl{color:var(--muted);white-space:nowrap;font-size:12px}
.sv{font:11px var(--mono);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#lw{grid-column:1/3;background:var(--card);border:1px solid var(--bdr);border-radius:8px;
    display:flex;flex-direction:column;overflow:hidden}
#lh{padding:5px 12px;border-bottom:1px solid var(--bdr);flex-shrink:0;display:flex;align-items:center;gap:8px}
#lh span{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.7px;color:var(--muted)}
#lb{flex:1;overflow-y:auto;padding:4px 12px;font:11px/1.75 var(--mono);min-height:0}
.ll{white-space:pre-wrap;word-break:break-all}
.lk{color:var(--green)}.lw2{color:var(--orange)}.le{color:var(--red)}.ld{color:#6366f1}.lm{color:#4b5563}
</style></head>
<body>
<div id="hdr">
  <h1>&#x1F916; 小艺 Reachy Mini &mdash; Debug Dashboard</h1>
  <span id="badge" class="badge ba">—</span>
  <div id="dot"></div>
</div>
<div id="main">
  <div id="cv">
    <img src="/video" alt="">
    <div id="cv-lbl">Camera &middot; VIS_DEBUG annotations</div>
  </div>
  <div id="side">
    <div class="card" id="rc">
      <div class="ch">声源方向 &middot; 世界坐标 (0\xb0=正前)</div>
      <canvas id="radar" width="280" height="206"></canvas>
    </div>
    <div class="card">
      <div class="ch">系统状态</div>
      <div class="sr"><span class="sl">状态机</span><span class="sv" id="ss"></span></div>
      <div class="sr"><span class="sl">收发音</span><span class="sv" id="sp"></span></div>
      <div class="sr"><span class="sl">声源(世界)</span><span class="sv" id="sd"></span></div>
      <div class="sr"><span class="sl">方向门控</span><span class="sv" id="sg"></span></div>
      <div class="sr"><span class="sl">切换</span><span class="sv" id="sw"></span></div>
      <div class="sr"><span class="sl">头/身偏航</span><span class="sv" id="sy"></span></div>
    </div>
  </div>
  <div id="lw">
    <div id="lh"><span>实时日志</span><span id="lc" style="margin-left:auto;color:#374151"></span></div>
    <div id="lb"></div>
  </div>
</div>
<script>
const GATE=55;
const cv=document.getElementById('radar'),cx=cv.getContext('2d');
const W=cv.width,H=cv.height,CX=W/2,CY=H/2,R=Math.min(CX,CY)-16;
const y2r=d=>-(d*Math.PI/180)-Math.PI/2;

function arw(d,len,col,lw,hs=9){
  const r=y2r(d),ex=CX+Math.cos(r)*len,ey=CY+Math.sin(r)*len;
  cx.beginPath();cx.moveTo(CX,CY);cx.lineTo(ex,ey);
  cx.strokeStyle=col;cx.lineWidth=lw;cx.stroke();
  const a=.42;cx.beginPath();
  cx.moveTo(ex,ey);
  cx.lineTo(ex-hs*Math.cos(r-a),ey-hs*Math.sin(r-a));
  cx.lineTo(ex-hs*Math.cos(r+a),ey-hs*Math.sin(r+a));
  cx.closePath();cx.fillStyle=col;cx.fill();
}

function drawRadar(s){
  cx.clearRect(0,0,W,H);
  cx.fillStyle='#0c0c18';cx.beginPath();cx.arc(CX,CY,R+14,0,Math.PI*2);cx.fill();
  cx.lineWidth=1;cx.strokeStyle='#1a1a2a';
  [.35,.7,1].forEach(f=>{cx.beginPath();cx.arc(CX,CY,R*f,0,Math.PI*2);cx.stroke()});
  for(let a=0;a<360;a+=45){const r=y2r(a);cx.beginPath();cx.moveTo(CX,CY);
    cx.lineTo(CX+Math.cos(r)*R,CY+Math.sin(r)*R);cx.stroke()}
  cx.font='bold 11px system-ui';cx.textAlign='center';cx.textBaseline='middle';
  [[0,'前','#4ade80'],[90,'左','#94a3b8'],[-90,'右','#94a3b8'],[180,'后','#374151']].forEach(([d,t,c])=>{
    const r=y2r(d);cx.fillStyle=c;cx.fillText(t,CX+Math.cos(r)*(R+11),CY+Math.sin(r)*(R+11));
  });
  const hy=s.track_yaw||0;
  const r1=y2r(hy-GATE),r2=y2r(hy+GATE);
  cx.beginPath();cx.moveTo(CX,CY);cx.arc(CX,CY,R*.9,r1,r2,true);cx.closePath();
  cx.fillStyle='rgba(34,211,160,.09)';cx.fill();
  cx.strokeStyle='rgba(34,211,160,.22)';cx.lineWidth=1;cx.stroke();
  const by=s.body_yaw_deg||0,rb=y2r(by);
  cx.setLineDash([3,4]);cx.strokeStyle='#4b5563';cx.lineWidth=1.5;
  cx.beginPath();cx.moveTo(CX,CY);cx.lineTo(CX+Math.cos(rb)*R*.75,CY+Math.sin(rb)*R*.75);cx.stroke();
  cx.setLineDash([]);
  arw(hy,R*.72,'#38bdf8',2.5);
  const dr=s.doa_resid_stable;
  if(dr!=null){
    const wd=hy+dr,fr=s.doa_fresh;
    const col=fr?(s.doa_confident?'#22d3a0':'#f5a623'):'#374151';
    arw(wd,R*.88,col,2);
    const rr=y2r(wd);
    cx.beginPath();cx.arc(CX+Math.cos(rr)*R*.88,CY+Math.sin(rr)*R*.88,4,0,Math.PI*2);
    cx.fillStyle=col;cx.fill();
  }
  if(s.switching&&s.switch_target){
    const rs=y2r(s.switch_target);
    cx.beginPath();cx.arc(CX+Math.cos(rs)*R*.72,CY+Math.sin(rs)*R*.72,5,0,Math.PI*2);
    cx.strokeStyle='#f97316';cx.lineWidth=2;cx.stroke();
  }
  cx.beginPath();cx.arc(CX,CY,3,0,Math.PI*2);cx.fillStyle='#fff';cx.fill();
  cx.font='10px monospace';cx.textAlign='left';cx.textBaseline='alphabetic';
  const fr2=s.doa_fresh,co2=s.doa_confident,dr2=s.doa_resid_stable;
  [['‒ ‒','#4b5563','身(body)'],
   ['——','#38bdf8','头(head)'],
   ['——',dr2!=null&&fr2?(co2?'#22d3a0':'#f5a623'):'#374151','声(world)']
  ].forEach(([sym,c,t],i)=>{cx.fillStyle=c;cx.fillText(sym+' '+t,4,H-26+i*13)});
}

const $=id=>document.getElementById(id);
const bmap={ARMED:'ba',ENGAGING:'be',TRACKING:'bt',SEEKING:'bs',SEARCHING:'bs',PLAYING:'bp',RETURNING:'br'};
function sv(id,txt,col){const e=$(id);e.textContent=txt;e.style.color=col||''}

function refresh(s){
  const b=$('badge');b.textContent=s.state||'—';b.className='badge '+(bmap[s.state]||'ba');
  sv('ss',s.state||'—');
  sv('sp',s.speaking?'🔊 说话中':'🎙️ 收听中',s.speaking?'#f97316':'#38bdf8');
  const dr=s.doa_resid_stable,hy=s.track_yaw||0;
  if(dr!=null&&s.doa_fresh){
    const w=hy+dr,dir=w>5?'←左':w<-5?'右→':'↑前';
    sv('sd',(w>=0?'+':'')+w.toFixed(0)+'\xb0 '+dir+' '+(s.doa_confident?'●':'○'),
       s.doa_confident?'#22d3a0':'#f5a623');
  }else{
    sv('sd',dr!=null?(hy+dr).toFixed(0)+'\xb0(旧)':'—','#374151');
  }
  sv('sg',s.gate_open?'✓ 开放 (收音)':'✗ 静音 (门关)',s.gate_open?'#22d3a0':'#f25e6b');
  sv('sw',s.switching?s.switch_phase+' → '+s.switch_target.toFixed(0)+'\xb0':'—',s.switching?'#f97316':'#374151');
  const hv=(s.track_yaw||0).toFixed(1),bv=(s.body_yaw_deg||0).toFixed(1);
  sv('sy','头 '+(hv>=0?'+':'')+hv+'\xb0  身 '+(bv>=0?'+':'')+bv+'\xb0');
  drawRadar(s);
}

let seq=0;const lb=$('lb'),MAX=400;
function lcls(t){
  if(/❌|ERROR|Traceback/.test(t))return 'le';
  if(/⚠|WARN|失败|failed/.test(t))return 'lw2';
  if(/✅|👂|🎙|🤖|就绪|启动|成功/.test(t))return 'lk';
  if(/KWS|raw=|resid=|IQR=|vad=/.test(t))return 'ld';
  return 'lm';
}
function addLog(lines){
  const bot=lb.scrollTop+lb.clientHeight>=lb.scrollHeight-40;
  const f=document.createDocumentFragment();
  lines.forEach(t=>{const d=document.createElement('div');d.className='ll '+lcls(t);d.textContent=t;f.appendChild(d)});
  lb.appendChild(f);
  while(lb.children.length>MAX)lb.removeChild(lb.firstChild);
  if(bot)lb.scrollTop=lb.scrollHeight;
  $('lc').textContent=lb.children.length+' lines';
}

const dot=$('dot');let conn=false;
async function poll(){
  try{
    const r=await fetch('/state.json?after='+seq,{cache:'no-store'});
    if(r.ok){
      const s=await r.json();
      if(!conn){dot.classList.add('ok');conn=true}
      seq=s.log_seq||seq;refresh(s);
      if(s.new_logs&&s.new_logs.length)addLog(s.new_logs);
    }
  }catch(e){dot.classList.remove('ok');conn=false}
  setTimeout(poll,250);
}
poll();
</script></body></html>"""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._html()
            elif path == "/video":
                self._mjpeg()
            elif path == "/state.json":
                self._state()
            else:
                self.send_error(404)

        def _html(self):
            body = _VIS_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not stop.is_set():
                    data = _build_frame()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        + data + b"\r\n"
                    )
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _state(self):
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            after = 0
            for part in qs.split("&"):
                if part.startswith("after="):
                    try:
                        after = int(part[6:])
                    except ValueError:
                        pass
            now = time.monotonic()
            with st.lock:
                data = {
                    "state": st.state,
                    "track_yaw": st.track_yaw,
                    "track_pitch": st.track_pitch,
                    "body_yaw_deg": st.body_yaw_deg,
                    "face_locked": st.face_locked,
                    "speaking": now < st.playback_end_estimate + 0.1,
                    "doa_resid_stable": st.doa_resid_stable,
                    "doa_confident": st.doa_confident,
                    "doa_fresh": (
                        st.doa_resid_stable is not None
                        and (now - st.doa_at) < DOA_GATE_FRESH_S
                    ),
                    "gate_open": st.dbg_gate_open,
                    "switching": st.dbg_switching,
                    "switch_phase": st.dbg_switch_phase,
                    "switch_target": st.dbg_switch_target,
                }
            data["log_seq"] = _vis_log_seq
            data["new_logs"] = [t for s, t in _vis_log_buf if s > after]
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    try:
        server = _Server(("0.0.0.0", port), _Handler)
        log(f"🔍 VIS_DEBUG → Dashboard: http://localhost:{port}  (浏览器打开;/video=MJPEG /state.json=状态)")
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        stop.wait()
        server.shutdown()
    except Exception as e:
        log(f"⚠ VIS_DEBUG 服务启动失败: {e}")


def doa_sensor_loop(st: State, stop: threading.Event) -> None:
    """DOA 纯传感器:10Hz 轮询 → 中值窗口 → 置信的视场外残差发布到 st.sound_resid。
    机器人自己说话期间的读数不入窗(防自声/扬声器反射污染)。behavior_loop 消费,不动头。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if _read_doa(opener) is None:
        log("⚠ DOA 端点不可用,本次无声源转向(其余功能不受影响)")
        return
    log("👂 声源传感器就绪(DOA REST 10Hz)")
    buf: "deque[tuple[float, float]]" = deque()    # 老:1.5s 窗 → sound_resid(SEEK/声源转向消费,不动)
    buf2: "deque[tuple[float, float]]" = deque()   # 新:DOA_WIN_S 窗 → doa_resid_stable/confident(M1.5-b/门控)

    def _med_iqr(samples):
        a = sorted(samples)
        n = len(a)
        return a[n // 2], a[(3 * n) // 4] - a[n // 4]   # (中值, IQR)

    while not stop.is_set():
        time.sleep(1.0 / DOA_POLL_HZ)
        r = _read_doa(opener)
        now = time.monotonic()
        with st.lock:
            robot_speaking = now < st.playback_end_estimate + 0.4
            by = st.body_yaw_deg
        # M1.5-a.5:DOA 常开固化——说话时也采(AEC 已证拾到外部方向,不再自声门控屏蔽)。
        # 只改"算不算",绝不改"转不转"(转向消费在 behavior 有 not speaking 守卫、SEEK 只唤醒消费)。
        if r is not None and r[1]:
            buf.append((now, r[0]))
            buf2.append((now, r[0]))
            if DOA_DEBUG:
                log(f"🎧 raw={r[0]:+6.0f}° vad=1 speaking={int(robot_speaking)} body_yaw={by:+.0f}° (n={len(buf2)})")

        # ── 老窗(SND_WIN_S)→ st.sound_resid(原样,消费方零影响)──
        while buf and now - buf[0][0] > SND_WIN_S:
            buf.popleft()
        if len(buf) >= SND_MIN_SAMPLES:
            med, spread = _med_iqr(a for _, a in buf)
            with st.lock:
                st.sound_resid = 90.0 - med
                st.sound_at = now
                st.sound_spread = spread

        # ── 新长窗(DOA_WIN_S)→ 稳健方向 + 可信度(confident=稳不是对)──
        while buf2 and now - buf2[0][0] > DOA_WIN_S:
            buf2.popleft()
        if len(buf2) >= DOA_MIN_SAMPLES:
            med2, iqr2 = _med_iqr(a for _, a in buf2)
            resid2 = 90.0 - med2
            confident = iqr2 < GATE_SPREAD          # 镜像翻转双簇→IQR~90→False;可用区→低 IQR→True
            with st.lock:
                st.doa_resid_stable = resid2
                st.doa_confident = confident
                st.doa_at = now
            if DOA_DEBUG:
                log(f"🎧→ resid_stable={resid2:+.0f}° IQR={iqr2:.0f}° confident={confident} "
                    f"speaking={int(robot_speaking)} body_yaw={by:+.0f}° n={len(buf2)}")
        else:
            with st.lock:
                st.doa_confident = False             # 样本不够(含静默)→ 不可信


def _fresh_sound(st: State) -> float | None:
    """读取新鲜(<0.6s)且偏离够大(>25°)的声源残差;否则 None。"""
    now = time.monotonic()
    with st.lock:
        if st.sound_resid is None or (now - st.sound_at) > 0.6:
            return None
        return st.sound_resid if abs(st.sound_resid) >= SND_RESID_MIN else None


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

    def set_state(s: str, seed_interact: bool = False) -> None:
        nonlocal phase_t
        with st.lock:
            if st.state != s:
                log(f"🧭 状态:{st.state} → {s}")
                st.state = s
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
            speaking = now < st.playback_end_estimate
        if action:
            phase_t = now  # 手势期间状态计时冻结(手势结束后从当前态继续)
            continue

        # WAKE-01 待命态:只等唤醒(main 连接成功后置 st.wake_ok),其余一律不响应(不跟人/不转声/不逗它)
        if state == ST_ARMED:
            with st.lock:
                woke = st.wake_ok
                if woke:
                    st.wake_ok = False
                    st.wake_doa = None
                    sr, sconf, sat = st.doa_resid_stable, st.doa_confident, st.doa_at
                    st.track_yaw = st.track_pitch = 0.0   # armed 居中,从 0 起转
            if woke:
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
                set_state(ST_ENGAGING, seed_interact=True)
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
    手势执行中(action_active)完全让位(motion goto 独占);微动仅在 IDLE/TRACKING 且说话时叠加。
    PLAYING(逗它)开心表达在此叠加渲染:进入只专注跟手,持续逗 PLAY_JOY_DELAY_S 后
    才第一次摇天线,之后周期小摇;短暂中断(<PLAY_REENTRY_S)再进入算"继续逗"不重置节拍;
    全程不打断跟手(set_target 同帧带 antennas)。"""
    dt = 1.0 / IDLE_HZ
    amp = 0.0
    sway_scale = 1.0
    prev_state = ST_IDLE
    joy_until = 0.0      # 当前天线摆动窗口的截止
    next_joy = 0.0       # 下一次摇天线的时刻
    last_play_exit = -1e9  # 上次退出 PLAYING 的时刻(重入防抖)
    ant_parked = True    # 天线已归位(避免每帧重复发中立位)
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
        if action:
            amp = 0.0  # 硬让位:手势 goto 独占
            prev_state = state
            time.sleep(dt)
            continue

        # 唤醒确认动作(M1c-a):叠加偏置(pitch 偏置 + 天线覆盖),包络 0→1→0;
        # 单槽后到覆盖先到(fail 接管 heard);additive 不抢渲染,转向一来基座照走、cue 自然衰减。
        cue_pitch, cue_ant = 0.0, None
        if wake_cue is not None:
            el = now - wake_cue_t
            dur = CUE[f"{wake_cue}_dur"]
            if 0.0 <= el < dur:
                env = math.sin(math.pi * el / dur)
                if wake_cue == "heard":
                    cue_pitch = -CUE["heard_pitch"] * env          # 抬头(pitch+ 为低头,故取负)
                    da = CUE["heard_ant"] * env                    # 天线上扬
                elif wake_cue == "fail":
                    cue_pitch = +CUE["fail_pitch"] * env           # 低头(连失败)
                    da = -CUE["fail_ant"] * env                    # 天线下垂
                elif wake_cue == "giveup":                         # 宽扫没人,比 fail 更轻
                    cue_pitch = +CUE["giveup_pitch"] * env
                    da = -CUE["giveup_ant"] * env
                else:  # bye(EXIT-01 收束告别,与 heard 上扬首尾呼应)
                    cue_pitch = +CUE["bye_pitch"] * env
                    da = -CUE["bye_ant"] * env
                cue_ant = [INIT_ANTENNAS[0] + da, INIT_ANTENNAS[1] + da]

        # WAKE-01 armed:慢呼吸(小幅低头起伏)+ 唤醒确认 cue 叠加
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
                next_joy = now + PLAY_JOY_DELAY_S  # 新一轮逗它:专注期后才第一次摇
            # 否则:短暂中断后的继续,沿用原节拍(不重置、无任何进场动作)
        elif state != ST_PLAYING and prev_state == ST_PLAYING:
            last_play_exit = now
        if state == ST_PLAYING and now >= next_joy:
            joy_until = now + PLAY_JOY_FLICK_S     # 小摇一下(有节奏,不神经质)
            next_joy = now + PLAY_JOY_PERIOD_S
        prev_state = state

        antennas = None
        if state == ST_PLAYING:
            if now < joy_until:
                w = 0.3 * math.sin(2 * math.pi * 3.0 * now)  # 3Hz 欢快小摆
                antennas = [INIT_ANTENNAS[0] - w, INIT_ANTENNAS[1] + w]
            else:
                antennas = list(INIT_ANTENNAS)   # 非摆动窗口:持续下发中立,姿态稳定
        elif not ant_parked:
            antennas = list(INIT_ANTENNAS)       # 退出逗它:归位一次
            ant_parked = True

        sway_ok = state in (ST_IDLE, ST_TRACKING)   # 转向/搜寻/回中/逗它中不叠微动
        amp += ((1.0 if (speaking and sway_ok) else 0.0) - amp) * (dt / IDLE_TAU)
        target_scale = TRACK_SWAY_SCALE if tracked else 1.0
        sway_scale += (target_scale - sway_scale) * (dt / IDLE_TAU)
        sway_yaw = amp * sway_scale * IDLE_YAW_AMP * math.sin(2 * math.pi * IDLE_YAW_F * now)
        sway_pitch = amp * sway_scale * IDLE_PITCH_AMP * math.sin(2 * math.pi * IDLE_PITCH_F * now + 1.0)
        if cue_ant is not None:          # 唤醒确认动作期间天线由 cue 接管(覆盖微动/逗它天线)
            antennas = cue_ant
        try:
            mini.set_target(head=head_pose(pitch_deg=tp + sway_pitch + cue_pitch, yaw_deg=ty + sway_yaw),
                            antennas=antennas,
                            body_yaw=math.radians(body))
        except Exception:
            time.sleep(1.0)
        time.sleep(dt)


# ───────────────────────── 播放线程:队列 → 扬声器 ─────────────────────────
def player_loop(mini: ReachyMini, st: State, play_q: "queue.Queue", stop: threading.Event) -> None:
    def current_gen() -> int:
        with st.lock:
            return st.play_gen

    def push(chunk: np.ndarray) -> None:
        try:
            mini.media.push_audio_sample(chunk)
        except Exception as e:
            log(f"⚠ push_audio_sample 失败: {type(e).__name__}: {e}")
            return
        with st.lock:
            base = max(st.playback_end_estimate, time.monotonic())
            st.playback_end_estimate = base + len(chunk) / PLAY_SR

    buffering = True
    while not stop.is_set():
        try:
            gen, chunk = play_q.get(timeout=0.1)
        except queue.Empty:
            buffering = True
            continue
        if gen != current_gen():
            continue
        if buffering:
            stash = [(gen, chunk)]
            dur = len(chunk) / PLAY_SR
            t_start = time.monotonic()
            while dur < JITTER_S and time.monotonic() - t_start < JITTER_WALL_S:
                try:
                    g2, c2 = play_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if g2 != current_gen():
                    continue
                stash.append((g2, c2))
                dur += len(c2) / PLAY_SR
            g_now = current_gen()
            valid = [c for g, c in stash if g == g_now]
            if not valid:
                continue
            for c in valid:
                push(c)
            buffering = False
        else:
            push(chunk)


# ───────────────────────── 主流程 ─────────────────────────
class KwsGate:
    """WAKE-01 唤醒门:本地 sherpa-onnx KeywordSpotter(单字"小艺"三形态)+ 去抖/不应期。
    feed(mono_f32_16k) → True 表示一次真唤醒。armed/engaged 都喂(engaged 命中忽略)。"""
    def __init__(self) -> None:
        os.makedirs(os.path.dirname(KWS_KEYWORDS), exist_ok=True)
        with open(KWS_KEYWORDS, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{form} #{KWS_SINGLE_THR:.2f} @小艺" for form in KWS_FORMS) + "\n")
        tag = "epoch-12-avg-2-chunk-16-left-64"
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=os.path.join(KWS_MODEL_DIR, "tokens.txt"),
            encoder=os.path.join(KWS_MODEL_DIR, f"encoder-{tag}.int8.onnx"),
            decoder=os.path.join(KWS_MODEL_DIR, f"decoder-{tag}.int8.onnx"),
            joiner=os.path.join(KWS_MODEL_DIR, f"joiner-{tag}.int8.onnx"),
            num_threads=1, max_active_paths=4,
            keywords_file=KWS_KEYWORDS,
            keywords_score=1.0, keywords_threshold=0.10, num_trailing_blanks=1,
            provider="cpu",
        )
        self.stream = self.kws.create_stream()
        self._last_raw = -1e9
        self._last_wake = -1e9
        self._diag_t = time.monotonic()
        self._diag_chunks = self._diag_dec = 0
        self._diag_rms_sq = self._diag_rms_n = 0
        self._diag_ch_done = False   # 只打一次各通道 RMS 对比

    def feed(self, mono: "np.ndarray", chunk_full: "np.ndarray | None" = None) -> bool:
        # 只打一次各通道 RMS（帮助定位哪个通道有声）
        if not self._diag_ch_done and chunk_full is not None and chunk_full.ndim == 2:
            ch_rms = [(c, float((chunk_full[:, c].astype(np.float64) ** 2).mean()) ** 0.5)
                      for c in range(chunk_full.shape[1])]
            log(f"[KWS通道诊断] shape={chunk_full.shape} dtype={chunk_full.dtype} "
                f"min={float(chunk_full.min()):.5f} max={float(chunk_full.max()):.5f} | "
                + " ".join(f"ch{c}={r:.5f}" for c, r in ch_rms))
            self._diag_ch_done = True
        self.stream.accept_waveform(16000, np.ascontiguousarray(mono, dtype=np.float32))
        hit = False
        n_dec = 0
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            n_dec += 1
        self._diag_chunks += 1
        self._diag_dec += n_dec
        self._diag_rms_sq += float(np.dot(mono, mono))
        self._diag_rms_n += len(mono)
        now = time.monotonic()
        if now - self._diag_t > 3.0:
            rms = (self._diag_rms_sq / max(1, self._diag_rms_n)) ** 0.5
            log(f"[KWS诊断] chunks={self._diag_chunks} dec={self._diag_dec} "
                f"RMS={rms:.4f} {'⚠ 静音?' if rms < 0.001 else '✅ 有声'}")
            self._diag_chunks = self._diag_dec = 0
            self._diag_rms_sq = self._diag_rms_n = 0
            self._diag_t = now
        result = self.kws.get_result(self.stream)
        if result:
            log(f"[KWS] 原始命中: {result!r}")
            self.kws.reset_stream(self.stream)
            hit = True
        if not hit:
            return False
        t = time.monotonic()
        if t - self._last_wake < KWS_REFRACTORY_S or t - self._last_raw < KWS_DEBOUNCE_S:
            self._last_raw = t
            return False
        self._last_raw = t
        self._last_wake = t
        return True


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
    for a in _args:                                      # --cue-heard-pitch=8 等,调确认动作幅度/时长
        if a.startswith("--cue-") and "=" in a:
            _k, _v = a[6:].split("=", 1)
            _ck = _k.replace("-", "_")
            if _ck in CUE:
                try:
                    CUE[_ck] = float(_v)
                except ValueError:
                    pass
    _nums = [a for a in _args if a.replace(".", "", 1).isdigit()]
    run_seconds = float(_nums[0]) if _nums else None    # 编排测试用:到时干净退出

    print("=== 小艺(Reachy Mini)语音对话:可打断 + 动作 + 看图 + 人脸跟随 + 听声转头 ===", flush=True)
    log(f"模型:{MODEL}|semantic_vad|16k上行|24k→16k下行|8 动作 + 看图 + 指向 + 逗它|五层仲裁(手势/指向>逗它>声源>跟随>微动)")

    st = State()
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

            callback = ChatCallback(st, play_q, motion_q, snap_q, mini)

            def open_session(timeout: float = CONNECT_TIMEOUT_S):
                """(b)命中才连:新建 WS + update_session,timeout 内未就绪 → None(超时也不卡死)。"""
                st.session_updated.clear()
                c = OmniRealtimeConversation(model=MODEL, callback=callback)
                holder = {"err": None}
                def _w():
                    try:
                        c.connect()
                        c.update_session(
                            output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                            voice=VOICE,
                            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                            output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                            enable_input_audio_transcription=True,
                            enable_turn_detection=True,
                            turn_detection_type="semantic_vad",
                            instructions=INSTRUCTIONS,
                            tools=TOOLS,
                        )
                    except Exception as e:
                        holder["err"] = e
                threading.Thread(target=_w, daemon=True).start()
                if st.session_updated.wait(timeout):
                    callback.conv = c
                    with st.lock:        # 新会话 = 无在途回复,重置回复状态(防 in_flight 跨会话泄漏→招呼守卫误跳过)
                        st.in_flight = 0
                        st.resp_audio_count = 0
                        st.fc_seen_this_resp = False
                        st.drop_audio = False
                    return c
                log(f"⚠ 连接失败/超时(>{timeout:.1f}s)err={holder['err']}")
                try:
                    c.close()
                except Exception:
                    pass
                return None

            def close_session(c):
                try:
                    c.close()
                except Exception:
                    pass
                callback.conv = None

            conv = None
            kws_gate = None
            if no_wake:
                log("连接 Qwen-Omni-Realtime(--no-wake:启动即连)…")
                conv = open_session(timeout=10.0)
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
                threading.Thread(target=snapshot_loop, args=(mini, st, callback, oai, snap_q, stop), daemon=True).start()
            threading.Thread(target=doa_sensor_loop, args=(st, stop), daemon=True).start()
            threading.Thread(target=behavior_loop, args=(st, snap_q, stop, not no_wake), daemon=True).start()
            if VIS_DEBUG:
                threading.Thread(target=vis_debug_server, args=(st, VIS_DEBUG_PORT, stop), daemon=True).start()
            # 视觉(TRACK-FIX):检测在子进程(独立 GIL),主进程只跑抓帧泵+结果积分
            # mediapipe 后端需 .task 模型文件;opencv 后端无需模型文件,直接启用
            vis_frame_q = None
            _vis_enabled = (
                (_VISION_BACKEND == "mediapipe" and os.path.exists(VIS_MODEL_PATH)) or
                (_VISION_BACKEND == "opencv")
            )
            if _vis_enabled:
                log(f"视觉后端: {_VISION_BACKEND}" + (" (sticky OFF)" if no_sticky else ""))
                if no_sticky:
                    os.environ["VISION_NO_STICKY"] = "1"
                elif "VISION_NO_STICKY" in os.environ:
                    del os.environ["VISION_NO_STICKY"]
                vis_frame_q = multiprocessing.Queue(maxsize=1)
                vis_result_q = multiprocessing.Queue(maxsize=64)
                multiprocessing.Process(
                    target=_vision_worker_fn,
                    args=(VIS_MODEL_PATH, HAND_MODEL_PATH, vis_frame_q, vis_result_q),
                    daemon=True,
                ).start()
                threading.Thread(target=frame_pump_loop, args=(mini, st, vis_frame_q, stop), daemon=True).start()
                threading.Thread(target=vision_result_loop, args=(st, vis_result_q, stop), daemon=True).start()
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
            t_run0 = time.monotonic()
            greet_i = 0   # 唤醒招呼轮换索引
            prev_gate_open = True   # M1.5-a 门控状态(只在切换时打日志)
            last_switch = -1e9      # M1.5-b 上次切换时刻(冷却)
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
                    # WAKE-01:同一份 16k mono 始终喂 KWS(本地);engaged 才扇出给 Qwen
                    wake = kws_gate.feed(mono, chunk) if kws_gate is not None else False
                    with st.lock:
                        state = st.state
                        woke_pending = st.wake_ok
                    if state == ST_ARMED:
                        if conv is not None and not woke_pending:     # engaged→armed:拆 WS(回零连接零计费)
                            close_session(conv)
                            conv = None
                            log("🌙 已回待命,WS 断开(零连接零计费)")
                        elif wake and conv is None:                    # 命中才连(b)
                            with st.lock:                              # 听到了:0 延迟确认(DOA 分流改由 behavior 连接后实时读)
                                st.wake_cue = "heard"
                                st.wake_cue_t = time.monotonic()
                            log("🔔 听到「小艺」(上扬)→ 连接 Qwen…")
                            tc = time.monotonic()
                            conv = None if sim_fail else open_session()
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
                        time.sleep(0.01)
                        continue
                    # M1.5-b 二次唤醒切换:engaged 收到"小艺"且【不是 A 方向】→ 切换转向新人 B。
                    # 不是 A 方向 = 除非"fresh&confident&|resid|≤55°"(确信 A 正前方向=A 自己又喊)。
                    # behavior 负责转向(写 st.state),main 这里只置 flag + 丢弃A重开会话(给B干净对话)。
                    if wake and not no_switch and (time.monotonic() - last_switch) > SWITCH_COOLDOWN_S:
                        nowk = time.monotonic()
                        with st.lock:
                            _sr, _sat, _sconf = st.doa_resid_stable, st.doa_at, st.doa_confident
                        _sfresh = _sr is not None and (nowk - _sat) < DOA_GATE_FRESH_S
                        _is_A = _sfresh and _sconf and abs(_sr) <= GATE_DEG
                        if not _is_A:
                            last_switch = nowk
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
                            log(f"🔀 二次唤醒(范围外)→ 切换转向新人({_hint});丢弃A、重开会话")
                            close_session(conv)
                            conv = open_session()
                            if conv is None:
                                log("⚠ 切换重连失败/超时(留待 behavior 找人;无会话则后续自动回待命)")
                            continue   # 本块跳过 append(切换中)
                        # else:A 自己又喊"小艺"(确信A方向)→ 忽略,继续正常对话
                    # 唤醒应答:SEEK 锁脸那刻 behavior 置 greet_now → 让模型招呼一句(走标准 response,可被打断)。
                    # 守卫:仅当无回应在途(in_flight==0)才招呼——只喊"小艺"=模型空闲才招呼;带了后续话=模型已在答,不双答。
                    with st.lock:
                        _do_greet = st.greet_now
                        if _do_greet:
                            st.greet_now = False
                        _busy = st.in_flight > 0
                    if _do_greet and not _busy:
                        _phrase = GREET_PHRASES[greet_i % len(GREET_PHRASES)]
                        greet_i += 1
                        try:
                            conv.create_response(instructions=greet_prompt(_phrase))
                            log(f"👋 唤醒应答:招呼「{_phrase}」")
                        except Exception as e:
                            log(f"⚠ 唤醒招呼发送失败:{e}")
                    elif _do_greet and _busy:
                        log("👋 唤醒应答跳过(模型已在回应后续话,不双答)")
                    rms_acc.append(float(np.sqrt(np.mean(mono**2))))
                    # M1.5-a 方向门控:只挡"确信范围外"(fresh+confident+|resid|>GATE_DEG),其余一律放行;
                    # 范围外→发静音(服务端听不到→自动不送/不打断/不重置计时,不碰那些代码)。
                    now_g = time.monotonic()
                    with st.lock:
                        g_resid = st.doa_resid_stable
                        g_fresh = g_resid is not None and (now_g - st.doa_at) < DOA_GATE_FRESH_S
                        g_conf = st.doa_confident
                    gate_open = no_gate or not (g_fresh and g_conf and abs(g_resid) > GATE_DEG)
                    if gate_open != prev_gate_open:
                        if gate_open:
                            log("🚪 门控:开 → 正常上行")
                        else:
                            log(f"🚪 门控:关(确信范围外 resid {g_resid:+.0f}° >±{GATE_DEG:.0f})→ 发静音,不送/不打断/不计时")
                        prev_gate_open = gate_open
                        with st.lock:
                            st.dbg_gate_open = gate_open
                    if gate_open:
                        pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
                    else:
                        pcm16 = np.zeros(len(mono), dtype=np.int16)   # 范围外:静音占位
                    conv.append_audio(base64.b64encode(pcm16.tobytes()).decode("ascii"))
                    sent_samples += len(mono)
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
                except Exception:
                    pass
                log("已释放 Realtime 连接与 Reachy 媒体资源。")
        finally:
            try:
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
