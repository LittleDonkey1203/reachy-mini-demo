# -*- coding: utf-8 -*-
"""配置常量、工具元数据、prompt 模板。

纯声明 + 无状态工具函数，无外部库依赖（仅标准库 + memory.manager 的 QWEN_TOOLS）。
"""

from __future__ import annotations

import json
import os

# ── 项目根路径 ──
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 模型 / API ──
MODEL = os.environ.get("REALTIME_MODEL", "qwen3.5-omni-plus-realtime")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3.5-omni-plus")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "qwen-turbo")
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "qwen-plus")  # 每轮工具审视(记忆抽取),FC/判断更稳
VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VOICE = "Ethan"
INSTRUCTIONS = (
    "你是桌面机器人,名字叫'小艺'(用户给你起的中文名),有真实的身体(头、天线)和一台摄像头。"
    "别人问你叫什么、喊'小艺'时,你都以小艺自居;不要自称 Reachy Mini(那只是你的硬件型号)。"
    "用简体中文、口语化、简短地回答,一般不超过两三句话。"
    "回答时自然地配合动作工具表达身体语言:打招呼/同意时点头,否定时摇头,"
    "开心/兴奋/被夸时摆天线,好奇/疑惑时歪头。"
    "重要:做动作时必须同时用语音回应,边说边做;绝不要默默做动作不说话。"
    "你的摄像头画面会持续实时提供给你(每秒约一帧),你一直能看见眼前的场景。"
    "用户让你看东西、问'你看到了什么''我手里拿的是什么''我比的是什么手势''这是什么''那边有什么'等"
    "视觉问题时,直接参考你最近看到的画面,用自己的话自然地说出来,不需要调用任何工具。"
    "若目标在当前画面外(比如让你看某个方向),可先用 look_left/right/up/down 转头,转过去后画面会随之更新,再看再答。"
    "【最重要的规则——你的文字会被语音合成直接朗读给用户听】"
    "你的文字输出只能包含你要说的话。"
    "禁止在文字中写任何动作描述、情绪标注、舞台指示——包括括号、尖括号、星号等任何形式。"
    "它们会被TTS原样念出来,非常奇怪。动作通过工具调用表达,不要写在文字里。"
)

# ── 路径 ──
SNAP_DIR = os.path.join(_REPO, "data", "output")
_MODELS_DIR = os.path.join(_REPO, "models")
_FACE_BACKEND = os.environ.get("FACE_BACKEND", "yunet").lower()
VIS_MODEL_PATH = os.path.join(_MODELS_DIR,
                              "face_detection_yunet_2023mar.onnx" if _FACE_BACKEND != "mediapipe"
                              else "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(_MODELS_DIR, "hand_landmarker.task")
GESTURE_MODEL_PATH = os.path.join(_MODELS_DIR, "gesture_recognizer.task")
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROFILE_PATH = os.path.join(_DATA_DIR, "profile.json")
MEMORY_PATH = os.path.join(_DATA_DIR, "memory.v1.json")

# ── 音频 ──
OUT_SR = 24000
PLAY_SR = 16000
JITTER_S = 0.30
JITTER_WALL_S = 0.50

# ── idle 微动 ──
IDLE_HZ = 25.0
IDLE_YAW_AMP = 2.5
IDLE_PITCH_AMP = 1.5
IDLE_YAW_F = 0.20
IDLE_PITCH_F = 0.30
IDLE_TAU = 0.5
TRACK_SWAY_SCALE = 0.4

# ── 视觉跟随 ──
VIS_MAX_FPS = 40.0
VIS_MISS_N = 5
DECIMATE = int(os.environ.get("DECIMATE", "3"))  # 跟踪用降采样帧(=3 track 稳);识别走全分辨率 ROI 重检(方案B)
FOV_X_DEG = 65.0
FOV_Y_DEG = 40.0
TRACK_TAU = 0.40
TRACK_DEADBAND = 2.0
TRACK_MAX_STEP = 1.5
# 头部"看谁"平滑(只影响头部转向+焦点,不动 ASD 归属判断/参数):引擎 EMA 上再叠重 EMA + 身份黏滞
HEAD_ASD_EMA = 0.18          # 头部专用二级 EMA(越小越平滑,比引擎 0.5 更平滑)
HEAD_SPK_ON = 0.20          # 平滑分 > 此值才算"该看的说话人"(才考虑切头)
HEAD_SWITCH_MARGIN = 0.50   # 切到新人需比当前人平滑分高出此余量(防两人间来回甩);否则黏在当前人直到其离场
# DOA 瞟头(TRACKING 态:侧面有人喊但画面里没人在说话 → 朝声源侧瞟一眼找人;只用 resid 符号,不信角度——见 CALIBRATION §14)
DOA_GLANCE_DEG = 20.0       # 声源偏离(身体系)> 此角度才触发转头找人;转到 ≤此即视为已面向、停转
GLANCE_MAX_DEG = 75.0       # 转头最大偏移:朝 DOA 角度转身找人(resid 是身体系,必须转身体才减小);封顶防转飞
GLANCE_MIN_HOLD_S = 0.3     # DOA 偏离需持续此时长才转(防瞬时误报)
GLANCE_TIMEOUT_S = 5.0      # 转身找人最长时间;超时未找到(可能 DOA 镜像错)→ 停 + 冷却
GLANCE_COOLDOWN_S = 3.0     # 超时后冷却,避免对着错方向反复转
GLANCE_SPEECH_GRACE_S = 1.5 # 转头触发必须"最近有人真说话"在此窗口内;否则纯环境音/DOA幻觉不转(治无声左漂)
GLANCE_MIN_TURN_DEG = 50.0  # F4:DOA 角度不可信(只信符号)→ 朝符号方向至少转这么多,把宽角度的人转进视野(治"转向不够");上限仍是 GLANCE_MAX_DEG
GLANCE_LOCAL_RMS = 0.006    # F1:本地麦克风(门控前)响度超此值即算"有人在说话",喂给瞟头触发,绕开方向门控对>55°声音的静音死锁。麦增益低可调小(正常说话约0.003)
FPS_FREEZE_BELOW = 8.0      # 检测 fps(EMA)低于此值→冻结身体跟随/瞟头,断"相机甩→churn→fps更低"死循环
ASD_MAX_TRACKS = 3          # ASD 每帧最多喂最大的 N 个 track(churn 出一堆 ghost 时逐个裁剪+打分会拖垮 CPU/GPU→fps崩)
TRACK_YAW_LIMIT = 25.0
TRACK_PITCH_LIMIT = 15.0
LOST_HOLD_S = 1.5
RETURN_TAU = 0.8
YAW_SIGN = -1.0
PITCH_SIGN = +1.0
GES_YAW_BOX = 25.0
GES_PITCH_BOX = 16.0

# ── 注视估计 (Gaze) ──
GAZE_MODEL_PATH = os.path.join(_MODELS_DIR, "l2csnet_mobilenetv2.onnx")
GAZE_INPUT_SIZE = 448
GAZE_NUM_BINS = 90
GAZE_BIN_WIDTH = 4.0
GAZE_OFFSET = 180.0
GAZE_MEAN = (0.485, 0.456, 0.406)
GAZE_STD = (0.229, 0.224, 0.225)
GAZE_HEAD_YAW_THRESH = 40.0          # L0 头姿门槛(原45,适当收紧,侧脸>40°淘汰不跑L2)
GAZE_HEAD_PITCH_THRESH = 45.0        # L0 头姿门槛(桌面机器人摄像头偏高,用户脸自然pitch+30~36°)
GAZE_NOT_LOOKING_INTERVAL = 5
GAZE_LOOKING_INTERVAL = 3            # LOOKING 态也降频(每3帧跑一次L2),防L2每帧跑→fps崩→churn
GAZE_MUTUAL_YAW_THRESH = 15.0        # mutual 阈值(收紧:L2CS-Net ~10°误差+桌面俯视偏差,20太宽)
GAZE_MUTUAL_PITCH_THRESH = 13.0      # mutual 阈值(标注数据427张网格搜索最优F1=0.857)
GAZE_DIR_DEADBAND = 8.0              # 方向一致性死区:|head_yaw|<此值时不检查gaze方向(正对相机)
GAZE_L2_EMA_ALPHA = 0.25             # L2 输出 EMA 平滑(越小越平滑,0.25≈4帧有效窗口,压瞬时噪声)
GAZE_MUTUAL_CONFIRM_FRAMES = 10      # mutual_gaze 连续 N 帧L2才确认(10帧×3=30raw帧≈2s,压误检)
GAZE_MUTUAL_DROP_FRAMES = 5          # mutual_gaze 连续 N 帧丢失才确认 NOT_LOOKING(黏性,防闪烁)
GAZE_IDLE_TIMEOUT_S = 2.0
GAZE_SCAN_PERIOD_S = 2.5
GAZE_GLANCE_INTERVAL_S = 4.0
GAZE_MIN_FACE_PX = 40
GAZE_ARMED_TAU = 0.80            # ARMED 注视回看时间常数(s)
GAZE_ARMED_MAX_STEP = 1.2        # 每帧最大转头步进(度)
GAZE_ARMED_DEADBAND = 3.0        # 小于此角度不动(度)
GAZE_ARMED_ENTRY_S = 0.5         # CURIOUS_LOOK 持续这么久才激活(防抖)
GAZE_ARMED_GRACE_S = 1.0         # 注视丢失后保持积分这么久(防闪烁回正)
GAZE_RETURN_DWELL_S = 1.5       # 注视丢失→回正前先停留这么久(自然感)
GAZE_RETURN_SPEED_DPS = 20.0    # 回正速度(°/s),远慢于追踪的90°/s,自然不生硬
# 注视情感反应:长时间对视不说话时随机触发微动作(per-identity冷却)
GAZE_REACT_FIRST_S = 4.0       # 首次微动作需持续注视至少N秒
GAZE_REACT_INTERVAL_S = 8.0    # 之后每隔N秒随机触发一次微动作
GAZE_REACT_MAX_COUNT = 3       # 同一轮最多触发N次微动作,之后不再打扰
GAZE_REACT_COOLDOWN_S = 180.0  # 同一个人一轮反应完后冷却3分钟,不过度打扰

# ── DOA 声源转向 ──
DOA_URL = "http://127.0.0.1:8000/api/state/doa"
DOA_POLL_HZ = 10.0
SND_WIN_S = 1.5
SND_MIN_SAMPLES = 5
SND_RESID_MIN = 25.0
SND_DONE_RESID = 10.0
DOA_DEBUG = os.environ.get("DOA_DEBUG") == "1"
DOA_WIN_S = 2.0
DOA_MIN_SAMPLES = 6
GATE_SPREAD = 25.0
GATE_DEG = 55.0
DOA_GATE_FRESH_S = 1.5
SWITCH_COOLDOWN_S = 2.0
SWITCH_AWAY_DEG = 35.0
SWITCH_SETTLE = 8.0
SWITCH_TIMEOUT_S = 8.0
SWITCH_COARSE_DEG = 70.0
SND_FACE_FRESH_S = 1.2
SND_MAX_HOPS = 3
SND_WAIT_FACE_S = 2.0
SND_COOLDOWN_S = 6.0
SND_SPEED_DPS = 90.0
SND_TARGET_LIMIT = 110.0
BODY_LIMIT_DEG = 90.0
NECK_REL_LIMIT = 23.0
BODY_FOLLOW_THRESHOLD = 0.7   # 头偏到颈限的 70% 时身体开始跟
BODY_FOLLOW_SPEED_DPS = 45.0  # 跟随态身体转速(°/s)

# ── 安全删除工作流 ──
CLEAR_VERIFY_COUNT = 3        # 连续高阈值匹配次数 (×IDENTITY_COOLDOWN_S ≈ 6s)
CLEAR_VERIFY_SIM = 0.80       # 验证阶段身份匹配阈值 (远高于 COSINE_THRESHOLD=0.35)
CLEAR_TIMEOUT_S = 30.0        # 验证/确认各阶段超时(s)

# ── 行为状态机 ──
ST_ARMED = "ARMED"
ST_IDLE = "IDLE_CENTER"
ST_ENGAGING = "ENGAGING"
ST_TRACKING = "TRACKING"
ST_SEARCHING = "SEARCHING"
ST_RETURNING = "RETURNING"
ST_POINTING = "POINTING"
ST_PLAYING = "PLAYING"
FACE_FRESH_S = 0.4
LOCK_WIN = 12
LOCK_ON_RATE = 0.40
LOCK_OFF_RATE = 0.15
ENGAGE_TIMEOUT_S = 6.0
ENGAGE_SCAN_RANGE = 15.0
ENGAGE_SCAN_TIME_S = 3.0
SEARCH_TIMEOUT_S = 4.0
NO_INTERACT_S = 15.0
FSM_HZ = 25.0

# ── 唤醒词 KWS ──
KWS_MODEL_DIR = os.path.join(_REPO, "tools", "_kws_models",
                             "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01")
KWS_KEYWORDS = os.path.join(_REPO, "tools", "_kws_models", "keywords_d01.txt")
KWS_FORMS = ["x iǎo y ī", "x iǎo y ìn", "x iǎo y ì"]
KWS_SINGLE_THR = 0.17
KWS_DEBOUNCE_S = 0.3
KWS_REFRACTORY_S = 2.0
WAKE_BLANK_S = 0.6           # KWS 命中后向 Qwen 发静音的时长(屏蔽"小艺"音频泄漏)
AUDIO_GATE_TIMEOUT_S = 5.0   # 音频闸门超时(s)——身份未确认时最多拦截音频的时间
CONV_SUMMARY_THRESHOLD = 2000  # 估算 token 数超过此值自动触发中途摘要
CONNECT_TIMEOUT_S = 3.0
ARMED_BREATH_F = 0.18
ARMED_BREATH_PITCH = 2.5
CUE = {
    "heard_dur": 0.45, "heard_pitch": 7.0, "heard_ant": 0.5,
    "fail_dur": 0.80,  "fail_pitch": 6.0,  "fail_ant": 0.7,
    "giveup_dur": 0.40, "giveup_pitch": 3.5, "giveup_ant": 0.35,
    "bye_dur": 0.45,   "bye_pitch": 3.5,   "bye_ant": 0.45,
    "barge_dur": 0.25, "barge_pitch": 2.0, "barge_ant": 0.3,
}
SPREAD_BAD = 40.0
WIDE_SCAN_RANGE = 88.0
WIDE_SCAN_HZ = 0.18
WIDE_SCAN_TIME_S = 7.0
DOA_WAKE_FRESH_S = 1.5
SEEK_PITCH_UP = -6.0
SEEK_PITCH_AMP = 6.0
SEEK_PITCH_HZ = 0.30
SEEK_NEARBY_DEG = 25.0
SEEK_NEARBY_TIME_S = 2.5
SEEK_SUPPRESS_DEG = 12.0

# ── 唤醒应答 ──
GREET_PHRASES = ["在呢", "来啦", "你好呀", "我在", "嗨,你好", "诶,在的", "怎么啦"]
BYE_PHRASES = ["好的", "拜拜", "休息啦", "我先歇会儿", "回头见", "去忙啦", "嗯,先这样"]
EXIT_MIN_S = 1.5
EXIT_MAX_S = 6.0

# ── 指向转头 ──
POINT_FRESH_S = 1.2
POINT_YAW_GAIN = 38.0
POINT_PITCH_GAIN = 12.0
POINT_TURN_TIMEOUT_S = 2.5
POINT_SETTLE_S = 0.6
POINT_HOLD_MAX_S = 4.0

# ── 手部互动 "逗它" ──
PLAY_SIZE_ON = 0.30
PLAY_SIZE_OFF = 0.22
PLAY_SCORE_MIN = 0.6
PLAY_HAND_V_MAX = 0.80
PLAY_ON_S = 0.3
PLAY_OFF_S = 1.5
PLAY_FRESH_S = 0.4
PLAY_MOVE_WIN_S = 0.8
PLAY_MOVE_MIN = 0.08
PLAY_STILL_S = 4.0
PLAY_TAU = 0.25
PLAY_MAX_STEP = 3.0
PLAY_AMP = 0.90
PLAY_YAW_LIMIT = TRACK_YAW_LIMIT * PLAY_AMP
PLAY_PITCH_LIMIT = TRACK_PITCH_LIMIT * PLAY_AMP
PLAY_COAST_S = 0.35
PLAY_COAST_DU = 0.20
PLAY_COAST_VEL = 2.0
PLAY_JOY_DELAY_S = 5.0
PLAY_JOY_PERIOD_S = 7.0
PLAY_JOY_FLICK_S = 0.6
PLAY_REENTRY_S = 3.0

# ── M3 运动基础 ──
EASE_ATTACK_FRAC = 0.35
CUE_VARIATION = 0.15

# ── M3 事件反应 ──
THINK_ROLL_AMP = 3.0
THINK_ROLL_F = 0.15
THINK_PITCH = -1.5
THINK_ANT_AMP = 0.15
THINK_ANT_F = 0.25
THINK_BLEND_TAU = 0.5
EXPR_SMILE_ANT = 0.20
EXPR_FROWN_ANT = -0.15
EXPR_BLEND_TAU = 0.8

# ── 工具定义 ──
_NOPARAM = {"type": "object", "properties": {}}
BASE_TOOLS = [
    {"type": "function", "name": "nod",
     "description": "点头。打招呼、同意、确认、答应请求时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "shake_head",
     "description": "摇头。否定、拒绝、不同意、说'不'时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_left",
     "description": "把头转向左边。转过去后摄像头画面会更新,就能看到左边有什么。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_right",
     "description": "把头转向右边。转过去后摄像头画面会更新,就能看到右边有什么。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_up",
     "description": "抬头转向上方。转过去后摄像头画面会更新,就能看到上面有什么。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_down",
     "description": "低头转向下方。转过去后摄像头画面会更新,就能看到下面有什么。", "parameters": _NOPARAM},
    {"type": "function", "name": "wiggle_antennas",
     "description": "欢快地摆动头顶天线。表达开心、兴奋、被夸奖、热情时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "tilt_head",
     "description": "歪头。表达好奇、疑惑、思考、没听懂时使用。", "parameters": _NOPARAM},
    {"type": "function", "name": "end_session",
     "description": "结束本次对话、让机器人回到待命休息。仅当用户【明确表达要结束对话/让你退下/离开】时才调用,"
                    "例如「走吧」「退下」「你先忙」「没事了」「拜拜」「不聊了」「先这样」「就到这」。"
                    "⚠️ 注意:「再说吧」「这个先放一边」「等会儿」「待会聊」「先放着」「回头说」等只是话题搁置或语气词,"
                    "【不是】结束对话,绝不要因此调用;拿不准时继续对话、不要调。",
     "parameters": _NOPARAM},
    # take_snapshot / identify_pointed_object 已移除:改为实时视频流(append_video, 1fps)
    # 直接喂模型,模型一直能看见画面,被问视觉问题直接答,无需工具往返。
]

# ── 看图 prompt ──
_POINT_GUIDE = ("如果用户正在用手指指向画面中某个物体(看手的朝向、伸出的食指延长线),"
                "请重点判断并明确说出他指的是哪一个物体、那是什么;")
SNAP_PROMPTS = {
    "scene": ("你是机器人的眼睛。用简体中文两三句话回答。" + _POINT_GUIDE +
              "否则描述画面主要内容,特别是人手里举着或拿着的物体(若有)。"),
    "point": ("你是机器人的眼睛。用户正在用手指指向画面中的某个物体。"
              "请仔细观察用户手指的指向(手的朝向、伸出的食指延长线),"
              "判断用户指的是哪一个物体,用简体中文两三句话明确说出那个物体是什么并简要描述。"
              "如果画面里没有看到明显的指向手势,就说你不太确定他指的是哪个,并描述画面里最可能的几个物体。"),
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
_DIR_MAP = {
    "左": (+30.0, 0.0), "右": (-30.0, 0.0),
    "上": (0.0, -10.0), "下": (0.0, +10.0),
    "左上": (+22.0, -8.0), "右上": (-22.0, -8.0),
    "左下": (+22.0, +8.0), "右下": (-22.0, +8.0),
}


def greet_prompt(phrase: str) -> str:
    return (f"用户刚出现在你面前(你刚找到他)。用中文口语自然地说一句简短招呼,"
            f"就说「{phrase}」的意思(可带个语气词,保持很短);**别用英文、别解释、别提'找到你了'**。")


def parse_judge(raw: str) -> dict | None:
    """宽容解析 VLM 的 judge JSON（剥代码块/取首尾大括号）；失败返回 None。"""
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
