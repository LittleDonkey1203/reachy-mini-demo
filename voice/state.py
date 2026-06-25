# -*- coding: utf-8 -*-
"""共享状态容器、日志、对话事件录制、One Euro 滤波器。

所有跨线程共享数据都在 State 类中；全局缓冲区供对话可视化 Dashboard 使用。
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

from voice.config import ST_IDLE

# ── 全局缓冲区（Dashboard 数据源）──
_vis_log_buf: deque[tuple[int, str]] = deque(maxlen=1000)
_vis_log_seq: int = 0

_conv_events: deque[dict] = deque(maxlen=2000)
_conv_turns: list[dict] = []
_conv_seq: int = 0
_turn_counter: int = 0
_feedback_notes: list[dict] = []
_current_turn: dict | None = None
_feedback_seq: int = 0
_pending_asr: str = ""


# ── 日志 ──
def log(msg: str) -> None:
    global _vis_log_seq
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _vis_log_seq += 1
    _vis_log_buf.append((_vis_log_seq, line))


# ── 对话事件录制（Conversation Dashboard 数据源）──
def _event_label(etype: str, event: dict) -> str | None:
    """为 Realtime API 事件生成人类可读的一行摘要。"""
    if etype == "session.created":
        return "🔗 会话建立"
    if etype == "session.updated":
        return "⚙️ 会话配置生效"
    if etype == "input_audio_buffer.speech_started":
        return "🎤 用户开始说话"
    if etype == "input_audio_buffer.speech_stopped":
        return "🤫 用户说完"
    if etype == "conversation.item.input_audio_transcription.completed":
        t = (event.get("transcript") or "").strip()[:80]
        return f'📝 ASR: 「{t}」'
    if etype == "response.created":
        return "💭 模型开始生成"
    if etype == "response.function_call_arguments.done":
        return f'🤖 工具调用: {event.get("name", "?")}'
    if etype == "response.audio_transcript.done":
        t = (event.get("transcript") or "").strip()[:80]
        return f'🔊 模型输出: 「{t}」'
    if etype == "response.done":
        return "✅ 回复完成"
    if etype == "response.audio_transcript.delta":
        return None
    if etype == "response.audio.delta":
        return None
    return etype


def _record_event(etype: str, event: dict) -> None:
    """录制一条 Realtime API 事件到 _conv_events，并维护高层轮次。"""
    global _conv_seq, _current_turn, _turn_counter, _pending_asr
    label = _event_label(etype, event)
    if label is None:
        return

    _conv_seq += 1
    seq = _conv_seq
    now_wall = time.time()
    now_mono = time.monotonic()
    ts = time.strftime("%H:%M:%S", time.localtime(now_wall)) + f".{int(now_wall * 1000) % 1000:03d}"

    prefix = etype.split(".")[0]
    role = {"input_audio_buffer": "user", "conversation": "user",
            "response": "model", "session": "system"}.get(prefix, "system")

    payload = {k: v for k, v in event.items()
               if k not in ("audio", "delta") and not (isinstance(v, str) and len(v) > 2000)}

    entry = {"seq": seq, "ts": ts, "ts_mono": now_mono,
             "type": etype, "role": role, "label": label, "payload": payload}
    _conv_events.append(entry)

    if etype == "conversation.item.input_audio_transcription.completed":
        asr_text = (event.get("transcript") or "").strip()
        if _current_turn is not None:
            _current_turn["asr"] = asr_text
        else:
            _pending_asr = asr_text
    if etype == "response.created":
        if _current_turn is not None:
            _current_turn["end_ts"] = ts
            _current_turn["end_mono"] = now_mono
        ctx_cutoff = now_mono - 10.0
        ctx = [e["label"] for e in _conv_events
               if e["ts_mono"] > ctx_cutoff and e["role"] == "system"
               and e["type"] not in ("session.created", "session.updated")]
        _turn_counter += 1
        turn = {"turn_id": seq, "turn_num": _turn_counter,
                "start_ts": ts, "start_mono": now_mono,
                "end_ts": None, "end_mono": None,
                "asr": _pending_asr, "tool_calls": [], "transcript": "",
                "snapshot_desc": "", "events": [seq],
                "context": ctx[-5:]}
        _pending_asr = ""
        _current_turn = turn
        if len(_conv_turns) >= 100:
            _conv_turns.pop(0)
        _conv_turns.append(turn)
    elif _current_turn is not None:
        _current_turn["events"].append(seq)
        if etype == "response.function_call_arguments.done":
            _current_turn["tool_calls"].append({
                "name": event.get("name", ""),
                "call_id": event.get("call_id", ""),
                "output_preview": "",
            })
        elif etype == "response.audio_transcript.done":
            _current_turn["transcript"] = (event.get("transcript") or "").strip()
        elif etype == "response.done":
            _current_turn["end_ts"] = ts
            _current_turn["end_mono"] = now_mono
            _current_turn = None


def _record_snap_result(call_id: str, mode: str, desc: str, ok: bool) -> None:
    """把 snapshot_loop 的 VLM 结果写入当前轮次。"""
    global _conv_seq
    _conv_seq += 1
    now_wall = time.time()
    now_mono = time.monotonic()
    ts = time.strftime("%H:%M:%S", time.localtime(now_wall)) + f".{int(now_wall * 1000) % 1000:03d}"
    preview = desc[:120] + ("…" if len(desc) > 120 else "")
    entry = {"seq": _conv_seq, "ts": ts, "ts_mono": now_mono,
             "type": "vlm.result", "role": "tool",
             "label": f'🖼️ VLM[{mode}]: 「{preview}」',
             "payload": {"call_id": call_id, "mode": mode, "ok": ok, "desc": desc}}
    _conv_events.append(entry)
    turn = _current_turn or (_conv_turns[-1] if _conv_turns else None)
    if turn is not None:
        turn["snapshot_desc"] = desc
        turn["events"].append(_conv_seq)
        for tc in turn["tool_calls"]:
            if tc["call_id"] == call_id:
                tc["output_preview"] = preview
                break


def _record_vis_event(etype: str, label: str, payload: dict | None = None) -> None:
    """录制非 Realtime API 事件（状态机、视觉、DOA、门控等）。"""
    global _conv_seq
    _conv_seq += 1
    now_wall = time.time()
    now_mono = time.monotonic()
    ts = time.strftime("%H:%M:%S", time.localtime(now_wall)) + f".{int(now_wall * 1000) % 1000:03d}"
    entry = {"seq": _conv_seq, "ts": ts, "ts_mono": now_mono,
             "type": etype, "role": "system",
             "label": label, "payload": payload or {}}
    _conv_events.append(entry)
    if _current_turn is not None:
        _current_turn["events"].append(_conv_seq)


# ── One Euro 滤波器 ──
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


# ── 共享状态容器 ──
class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_updated = threading.Event()
        self.play_gen = 0
        self.drop_audio = False
        self.in_flight = 0
        self.playback_end_estimate = 0.0
        self.resp_audio_count = 0
        self.fc_seen_this_resp = False
        self.fc_gen = 0
        self.state = ST_IDLE
        self.wake_ok = False
        self.wake_cue = None
        self.wake_cue_t = 0.0
        self.greet_now = False
        self.exit_request = False
        self.switch_request = None
        self.action_active = False
        self.track_yaw = 0.0
        self.track_pitch = 0.0
        self.body_yaw_deg = 0.0
        self.face_seen_at = 0.0
        self.face_locked = False
        self.user_speaking = False
        self.last_interaction_at = 0.0
        self.sound_resid = None
        self.sound_at = 0.0
        self.sound_spread = 0.0
        self.wake_doa = None
        self.doa_resid_stable = None
        self.doa_confident = False
        self.doa_at = 0.0
        self.finger_angle = None
        self.finger_at = 0.0
        self.finger_extended = False
        self.finger_ext_at = 0.0
        self.hand_u = 0.5
        self.hand_v = 0.5
        self.hand_size = 0.0
        self.hand_at = 0.0
        self.hand_move = 0.0
        self.point_request = None
        self.snap_grabbed = False
        self.latest_frame = None
        self.latest_frame_t = 0.0
        self.snapshot_pending = 0
        self.dbg_frame_small = None
        self.dbg_det = None
        self.dbg_gate_open = True
        self.dbg_switching = False
        self.dbg_switch_phase = ""
        self.dbg_switch_target = 0.0
        self.gesture = None
        self.gesture_at = 0.0
        self.gesture_fingers = 0
        self.no_easing = False
        self.no_variation = False
        self.no_expression = False
        self.no_memory = False
        self.thinking = False
        self.user_smile = 0.0
        self.user_frown = 0.0
        self.conversation_log: dict[str, list[tuple[str, str]]] = {}  # pid → [(role, text), ...]
        self.display_transcript: list[dict] = []  # 持久对话记录，不被摘要/close_session 清除
        self.display_transcript_seq: int = 0
        self.current_person_id: str | None = None
        self.current_person_name: str | None = None
        self.current_is_owner: bool = False
        self.identity_injected = False
        self.identity_injected_pid: str | None = None
        self.resp_snapshot_pid: str | None = None
        self.resp_snapshot_name: str | None = None
        self.vis_ready = False
        self.clear_workflow: dict | None = None
        self.clear_lock: bool = False
        self.audio_gate_closed: bool = False
        self.audio_gate_buffer: list[bytes] = []
        self.audio_gate_closed_at: float = 0.0
        self.dbg_memory_prompt: str | None = None
        self.dbg_session_instructions: str | None = None
