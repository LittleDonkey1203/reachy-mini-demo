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
    "当用户用手指指着某个方向或物体,说'看看那边''这是什么''我指的是什么''那里有什么'等,"
    "调用 identify_pointed_object——它会自动判断指向、需要时转头去看,拿到结果后自然地说出来。"
    "⚠ 绝不要用 look_left/right 代替'看':那只是肢体动作不会拍照。"
    "要看某个方向有什么,正确做法是调 identify_pointed_object(会自动转头+拍照)或先 look_X 再 take_snapshot。"
    "【最重要的规则——你的文字会被语音合成直接朗读给用户听】"
    "你的每一个字、每一个符号都会被TTS引擎原样朗读出来。"
    "所以绝对不能在文字中写任何标记、标签、动作描述:"
    "禁止<nod> <shake> <smile> <wiggle_antennas> <wave>等任何<xxx>形式;"
    "禁止(点头) (摇头) (微笑)等括号描述;禁止*点头* *摇头*等星号描述。"
    "这些都会被直接念出来,听起来非常奇怪。"
    "要做动作请调用工具函数(nod、shake_head、wiggle_antennas、tilt_head),"
    "工具调用和文字回复是分开的通道,不会被朗读。"
    "正确:'好的!' + 调用nod工具。"
    "错误:'好的!<nod>' '好的!(点头)' '好的!*点头*'。"
    "\n【记忆规则】"
    "当用户提到自己的个人信息(名字、爱好、喜欢/不喜欢的东西、职业、年龄等)时,"
    "必须调用 remember_fact 记住。例如用户说'我喜欢猫'→调用remember_fact(fact='喜欢猫')。"
    "用户告诉你名字时,额外传name参数:remember_fact(fact='叫小明',name='小明')。"
    "用户改变之前说过的信息时,用replaces:'我不喜欢篮球了,喜欢羽毛球'→"
    "remember_fact(fact='喜欢打羽毛球',replaces='篮球')。"
    "用户让你忘记某件事时调用 forget_fact(keyword='猫')。"
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
DECIMATE = 3
FOV_X_DEG = 65.0
FOV_Y_DEG = 40.0
TRACK_TAU = 0.40
TRACK_DEADBAND = 2.0
TRACK_MAX_STEP = 1.5
TRACK_YAW_LIMIT = 25.0
TRACK_PITCH_LIMIT = 15.0
LOST_HOLD_S = 1.5
RETURN_TAU = 0.8
YAW_SIGN = -1.0
PITCH_SIGN = +1.0
GES_YAW_BOX = 25.0
GES_PITCH_BOX = 16.0

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
     "description": "把头转向左边(仅肢体动作,不拍照不看东西)。要看左边是什么请先look_left再take_snapshot。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_right",
     "description": "把头转向右边(仅肢体动作,不拍照不看东西)。要看右边是什么请先look_right再take_snapshot。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_up",
     "description": "抬头看上方(仅肢体动作,不拍照不看东西)。要看上面是什么请先look_up再take_snapshot。", "parameters": _NOPARAM},
    {"type": "function", "name": "look_down",
     "description": "低头看下方(仅肢体动作,不拍照不看东西)。要看下面是什么请先look_down再take_snapshot。", "parameters": _NOPARAM},
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
    {"type": "function", "name": "take_snapshot",
     "description": "用摄像头拍一张当前画面并理解内容。当用户让你看东西、问'你看到什么''我手里是什么'等需要视觉、但不涉及'指向'的问题时调用。",
     "parameters": _NOPARAM},
    {"type": "function", "name": "identify_pointed_object",
     "description": "当用户在用手指指方向或指物体时调用——包括'这是什么''我指的是什么''看看那边''那里有什么'等。"
                    "会自动判断指向方向,需要时转头去看,然后拍照理解目标。优先于 look_left/right + take_snapshot 组合。",
     "parameters": _NOPARAM},
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
