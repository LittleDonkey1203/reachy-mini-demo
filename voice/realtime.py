# -*- coding: utf-8 -*-
"""Qwen-Omni-Realtime 对话协议层 — 回调 + 会话生命周期管理。"""

import base64
import json
import os
import queue
import re
import threading
import time

import numpy as np
from scipy.signal import resample_poly

from dashscope.audio.qwen_omni import (
    OmniRealtimeCallback, OmniRealtimeConversation,
    AudioFormat, MultiModality,
)
from reachy_mini import ReachyMini

from voice.config import (
    MODEL, VOICE, SUMMARY_MODEL, EXTRACT_MODEL, CONNECT_TIMEOUT_S,
    POINT_FRESH_S, OUT_SR, PLAY_SR,
    CONV_SUMMARY_THRESHOLD,
)
from voice.state import State, log, _record_event
import voice.state as _st_mod
from tools.base import ToolDeps


def _record_tool_output(st: State, tool_name: str, call_id: str, output: str):
    """把 tool output 写入 display_transcript，保持模型上下文时序完整。"""
    with st.lock:
        st.display_transcript_seq += 1
        st.display_transcript.append({
            "seq": st.display_transcript_seq,
            "ts": time.strftime("%H:%M:%S"),
            "role": "tool_output",
            "text": f"{tool_name} → {output[:120]}",
            "call_id": call_id,
        })
        if len(st.display_transcript) > 100:
            st.display_transcript = st.display_transcript[-80:]


# ── transcript 泄漏标签 → 物理动作兜底 ──
_TAG_TO_ACTION = {
    "nod": "nod", "点头": "nod", "nodding": "nod",
    "shake": "shake_head", "shake_head": "shake_head", "摇头": "shake_head",
    "wiggle": "wiggle_antennas", "摆天线": "wiggle_antennas", "wave": "wiggle_antennas",
    "tilt": "tilt_head", "歪头": "tilt_head", "tilt_head": "tilt_head",
    "smile": "wiggle_antennas", "微笑": "wiggle_antennas",
    "look_left": "look_left", "look_right": "look_right",
    "look_up": "look_up", "look_down": "look_down",
}
_ACTION_TAG_RE = re.compile(
    r"</?(?:" + "|".join(re.escape(k) for k in _TAG_TO_ACTION) + r")[^>]*>"
    r"|[（(](?:" + "|".join(re.escape(k) for k in _TAG_TO_ACTION) + r")[)）]"
    r"|\*(?:" + "|".join(re.escape(k) for k in _TAG_TO_ACTION) + r")\*",
    re.IGNORECASE,
)


def _extract_tag_action(match_str: str) -> str | None:
    s = match_str.strip("<>/()（）* \t").lower()
    return _TAG_TO_ACTION.get(s)


# ── 用户话语→turn_body 兜底:模型应调工具但只说了文本("好嘞，转过去啦")时自动补发 ──
_TURN_CMD_RE = re.compile(
    r"向(左|右)转|转(向|到|去|过去|过来|过身|个身)"
    r"|面(朝|向)(左|右|那边|这边|后面)"
    r"|往(左|右)转",
)


def _parse_turn_direction(text: str) -> dict | None:
    """从用户语音转写中提取 turn_body 方向+角度;无匹配返回 None。"""
    m = _TURN_CMD_RE.search(text)
    if not m:
        return None
    s = m.group()
    if "右" in s:
        direction = "right"
    elif "左" in s:
        direction = "left"
    else:
        direction = "right"    # "转过去""转个身"等无方向指示,默认右转
    return {"direction": direction, "angle": 45}


_BAD_CASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "bad_cases")


def _record_turn_bad_case(user_text: str, parsed_cmd: dict) -> None:
    """记录 turn_body 未调用的 bad case(用户说了转身但模型没调工具),供后续统一优化数据。"""
    try:
        os.makedirs(_BAD_CASE_DIR, exist_ok=True)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type": "turn_body_not_called",
            "user_text": user_text,
            "parsed": parsed_cmd,
        }
        path = os.path.join(_BAD_CASE_DIR, "turn_body.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # best-effort, 不阻塞主流程


# ── 命名 guard:命名是身份关键操作,统一过门(治脑补名/画外命名/反复改名)──
_NAME_OK_RE = re.compile(r"^[一-龥A-Za-z·]{1,8}$")          # 1-8 中/英文字,无数字/标点/空格


def _valid_name(name: str) -> bool:
    if not name:
        return False
    n = name.strip()
    # 机器人自己的名字/别名绝不能当作用户名字
    _BOT_NAMES = {"小艺", "小易", "小意", "小亿", "xiaoyi"}
    return (bool(_NAME_OK_RE.match(n)) and not n.startswith("?T")
            and n not in ("画外", "未知") and n.lower() not in _BOT_NAMES)


def try_name_identity(*, memory_mgr, identity_store, face_pipeline, owner_mgr, st,
                      pid, new_name, transcript, log_fn) -> bool:
    """命名/改名统一 guard。返回是否真正写入了名字。
    门1 名字合法;门2 名字必须出现在当轮转写里(防模型脑补)。
    pid 由调用方保证 = 本句说话人(画外/无归属时为 None,直接拒)。"""
    if not pid or not new_name:
        return False
    n = new_name.strip()
    existing = memory_mgr.get_name(pid) if memory_mgr else None
    if existing and existing == n:
        return False
    if not _valid_name(n):
        log_fn(f"🚫 命名拒绝:名字不合法「{new_name}」")
        return False
    if not (transcript and n in transcript):
        log_fn(f"🚫 命名拒绝:「{n}」不在转写里(防脑补)← 「{(transcript or '')[:30]}」")
        return False
    if existing:
        log_fn(f"✏ 改名:「{existing}」→「{n}」")
    if memory_mgr:
        memory_mgr.set_name(pid, n)
    if identity_store is not None:
        identity_store.set_name(pid, n)
    if face_pipeline is not None:
        try:
            if face_pipeline.store.confirm_identity(pid, n):
                face_pipeline.save_gallery()
                log_fn(f"🏷 gallery 身份已确认并落盘: {n} ({pid[:12]})")
        except Exception as _e:
            log_fn(f"⚠ gallery 命名失败:{type(_e).__name__}: {_e}")
    with st.lock:
        if st.current_person_id == pid:
            st.current_person_name = n
    if owner_mgr is not None and not owner_mgr.has_owner():
        if owner_mgr.try_claim(pid, n):
            log_fn(f"👑 认主成功: {n} ({pid})")
    return True


class ChatCallback(OmniRealtimeCallback):
    """Qwen Omni Realtime 事件回调 — 音频播放、barge-in、工具分发、transcript 解析。"""

    def __init__(self, st: State, play_q: "queue.Queue", motion_q: "queue.Queue",
                 snap_q: "queue.Queue", mini: ReachyMini,
                 memory_mgr, owner_mgr, identity_store=None, registry=None,
                 face_pipeline=None, asd_engine=None):
        self.st = st
        self.play_q = play_q
        self.motion_q = motion_q
        self.snap_q = snap_q
        self.mini = mini
        self.memory_mgr = memory_mgr
        self.owner_mgr = owner_mgr
        self.identity_store = identity_store
        self.registry = registry
        self.face_pipeline = face_pipeline
        self.asd_engine = asd_engine
        self._speech_start_t = 0.0           # 本句说话起点(monotonic),speech_started 时记
        self._pending_turn_cmd: dict | None = None  # 用户说了转身指令但模型未调 turn_body
        self._turn_body_called = False               # 本轮 response 是否调了 turn_body
        self._pending_transcript: str = ""            # 触发 turn_cmd 的用户转写(供 bad case 记录)
        self.conv: OmniRealtimeConversation | None = None
        self.dialog: "RealtimeDialog | None" = None

    def on_open(self) -> None:
        log("✅ WebSocket 已连接 dashscope.aliyuncs.com")

    def on_close(self, close_status_code, close_msg) -> None:
        log(f"🔌 连接关闭:code={close_status_code} msg={close_msg}")

    def _do_barge_in(self, in_flight: bool) -> None:
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
        if not st.no_expression:
            with st.lock:
                st.wake_cue = "barge"
                st.wake_cue_t = time.monotonic()
        log("⛔ 打断:已停止播放" + (",并取消在途回复" if in_flight else ""))

    def on_event(self, event) -> None:
        st = self.st
        try:
            etype = event.get("type", "")
            _record_event(etype, event)
            now = time.monotonic()
            if etype == "session.created":
                log(f"✅ 会话已建立 session_id={event['session']['id']}")
            elif etype == "session.updated":
                if self.conv is None:
                    log("✅ 会话配置生效(semantic_vad / 8 动作 + take_snapshot + identify_pointed_object 已注册)")
                    log("▶ 可以对机器人说话了;它说话时可随时插话打断(Ctrl+C 退出)")
                else:
                    log("✅ 会话 instructions 已更新")
                st.session_updated.set()
            elif etype == "input_audio_buffer.speech_started":
                self._speech_start_t = now           # 本句说话起点(供 ASD speaker_window 归属)
                with st.lock:
                    st.last_interaction_at = now
                    st.user_speaking = True
                    playing = (now < st.playback_end_estimate) or (not self.play_q.empty())
                    in_flight = st.in_flight > 0
                log("🎤 检测到你开始说话…")
                if playing or in_flight:
                    self._do_barge_in(in_flight)
            elif etype == "input_audio_buffer.speech_stopped":
                with st.lock:
                    st.thinking = True
                    st.user_speaking = False
                log("🤫 检测到你说完了,等模型回应…")
            elif etype == "conversation.item.input_audio_transcription.completed":
                _transcript = (event.get("transcript") or "").strip()
                # ── ASD 归属:优先"本句说话期间任意时刻在说话"的 track(speaker_window,
                #    耐 ASD 延迟,治"说完才出分"),否则当前保持的 asd_speaker,再否则画外 ──
                _asp = None
                _sw = (self.asd_engine.speaker_window(self._speech_start_t)
                       if (self.asd_engine is not None and self.asd_engine.available) else None)
                if _sw is not None:
                    _key, _score = _sw                          # key = 身份键(person_id 或 t{track_id})
                    if isinstance(_key, str) and _key.startswith("t"):
                        _asp = {"pid": None, "name": None,         # 画面内但未识别身份
                                "track_id": _key[1:], "score": _score, "at": now}
                    else:                                         # 已识别:key 即 person_id
                        _nm = self.memory_mgr.get_name(_key) if self.memory_mgr else None
                        _asp = {"pid": _key, "name": _nm,
                                "track_id": self.asd_engine.last_track(_key), "score": _score, "at": now}
                if _asp is None:
                    with st.lock:
                        _hold = st.asd_speaker
                    if _hold is not None and (now - _hold.get("at", 0.0)) < 2.0:
                        # 多人场景:ASD 追踪多于 1 个身份时 speaker_window 无结果,
                        # 意味着新人可能还没攒够 ASD 帧(无法出分),此时不 fallback 到旧人
                        # → 走 neutral 路径,让模型回答"不认识"而非张冠李戴(bug-067)
                        _n_tracked = (len(self.asd_engine.tracked_keys())
                                      if (self.asd_engine is not None and self.asd_engine.available)
                                      else 0)
                        if _n_tracked <= 1:
                            _asp = _hold
                        else:
                            log(f"⚠ ASD fallback 拦截:追踪 {_n_tracked} 人但 speaker_window 无结果,"
                                f"不归属到 {_hold.get('name', '?')}(可能是新人说话)")
                if _asp is not None:
                    _tid = _asp.get("track_id")
                    _log_pid = _asp.get("pid") or f"_track{_tid}"      # 在画面但未识别:临时 track 键
                    _real_name = _asp.get("name")
                    _log_name = _real_name or f"?T{_tid}"             # 带 ?T 的占位仅用于日志/dashboard 显示
                    _attr_tag = f"{_log_name} (T{_tid}, ASD{_asp.get('score', 0.0):+.1f})"
                else:
                    _log_pid = "_offscreen"                            # 画外:专门归属标签(不张冠李戴)
                    _real_name = None
                    _log_name = "画外"
                    _attr_tag = "画外(无画面说话人)"
                # ── ① 本句说话人:记忆「存/读」唯一来源(稳),不再用飘的 current_person_id ──
                _tspk_real = (_log_pid not in ("_unknown", "_offscreen")
                              and not _log_pid.startswith("_track"))
                with st.lock:
                    st.turn_speaker_pid = _log_pid if _tspk_real else None
                    st.turn_speaker_name = _real_name if _tspk_real else None   # 占位名 ?T 绝不入 turn_speaker
                    st.turn_speaker_at = now
                log(f"📝 听到的是:「{_transcript}」 → 🗣 归属: {_attr_tag}")
                # ── turn_body 兜底检测:用户说了转身命令,等模型是否调工具 ──
                _tcmd = _parse_turn_direction(_transcript) if _transcript else None
                if _tcmd is not None:
                    self._pending_turn_cmd = _tcmd
                    self._turn_body_called = False
                    self._pending_transcript = _transcript
                if _transcript:
                    with st.lock:
                        st.display_transcript_seq += 1
                        st.display_transcript.append({"seq": st.display_transcript_seq, "ts": time.strftime("%H:%M:%S"), "role": "user", "text": _transcript, "pid": _log_pid, "name": _log_name})
                        if len(st.display_transcript) > 100:
                            st.display_transcript = st.display_transcript[-80:]
                    if not st.no_memory:
                        with st.lock:
                            st.conversation_log.setdefault(_log_pid, []).append(("user", _transcript))
                            _check_log = st.conversation_log.get(_log_pid, [])
                            _est_tok = sum(len(t) * 1.5 for _, t in _check_log)
                        _attributable = (_log_pid not in ("_unknown", "_offscreen")
                                         and not _log_pid.startswith("_track"))
                        if _est_tok > CONV_SUMMARY_THRESHOLD and _attributable and self.memory_mgr:
                            with st.lock:
                                _snap = list(st.conversation_log.get(_log_pid, []))
                                st.conversation_log[_log_pid] = []
                            if self.dialog:
                                threading.Thread(target=self.dialog.save_summary,
                                                 args=(_log_pid, _snap), daemon=True).start()
                            log(f"📝 上下文过长，自动触发 consolidation({_log_pid[:12]}, ~{int(_est_tok)} tok)")
                        # ── ① 身份/记忆注入:探针实测证明 Qwen Omni 忽略会话中途的 create_item system 条目,
                        #    只 honor response.instructions → 身份注入统一由 ③ 的 create_response(instructions=)
                        #    携带(resp_directive),这里不再单独注入(create_item 已砍)。
                        # ── ② 每轮工具审视:无条件用 qwen-plus 抽「本句说话人」的记忆,兜底 realtime 漏调 remember_fact ──
                        if _tspk_real and self.dialog is not None and self.memory_mgr is not None:
                            with st.lock:
                                _recent = [d for d in st.display_transcript
                                           if d.get("role") in ("user", "assistant")][-10:]
                            _ctx = [(d.get("role"), d.get("name"), d.get("text")) for d in _recent]
                            threading.Thread(target=self.dialog.extract_memory_async,
                                             args=(_log_pid, _real_name, _transcript, _ctx),
                                             daemon=True).start()
                    # ── ③ 收回 turn-taking:注入之后我方手动建回复(VAD 只断句、不自动回复),
                    #    保证这轮一定参考到刚注入的当前说话人上下文(根治 semantic_vad 抢跑时序竞态)。
                    #    守卫 in_flight==0:防和唤醒招呼(behavior 侧 create_response)双答;打断已 cancel 旧回复。
                    if self.conv is not None:
                        with st.lock:
                            _busy_r = st.in_flight > 0
                        if not _busy_r:
                            # D:给这次回复带「当前说话人」强指令(response.instructions,直接是本次生成的
                            #    系统指令,比 create_item 的 system 条目强得多)→ 治"模型从历史捞别人名字"
                            _present_r = _tspk_real or _log_pid.startswith("_track")
                            _cur_pid_r = _log_pid if _tspk_real else None
                            # 身份注入统一走 D;顺带刷新显示状态(dashboard MEM / 💭 日志)
                            with st.lock:
                                st.identity_injected = True
                                st.identity_injected_pid = _cur_pid_r or ("_present" if _present_r else "_neutral")
                            _instr_r = (self.dialog.resp_directive(_cur_pid_r, _real_name, _present_r)
                                        if self.dialog is not None else None)
                            try:
                                self.conv.create_response(instructions=_instr_r)
                            except Exception as _e:
                                log(f"⚠ create_response 失败:{type(_e).__name__}: {_e}")
                        else:
                            log(f"⏭ 跳过 create_response(in_flight={st.in_flight},招呼/旧回复在途)")
            elif etype == "response.created":
                with st.lock:
                    st.in_flight += 1
                    st.drop_audio = False
                    st.resp_audio_count = 0
                    st.fc_seen_this_resp = False
                    st.last_interaction_at = now
                    # 记忆保存键到「本句说话人」。本轮有用户说话(turn_speaker_at 新鲜)就用其结果——
                    # 画外/未识别时 turn_speaker_pid=None → resp_snapshot=None → remember_fact 拿到 None 不存,
                    # 绝不把画外的话张冠李戴给在场的人(治"大大被改名坤坤")。仅无近期说话(如招呼)才回退当前人。
                    if (now - st.turn_speaker_at) < 8.0:
                        st.resp_snapshot_pid = st.turn_speaker_pid
                        st.resp_snapshot_name = st.turn_speaker_name
                    else:
                        st.resp_snapshot_pid = st.current_person_id
                        st.resp_snapshot_name = st.current_person_name
                    _dt_seq = st.display_transcript_seq
                    _rc_pid = st.current_person_id
                    _rc_name = st.current_person_name
                    _rc_injected = st.identity_injected
                    _rc_injected_pid = st.identity_injected_pid
                if _st_mod._current_turn is not None:
                    _st_mod._current_turn["dt_seq"] = _dt_seq
                log(f"💭 模型开始生成回复… 当前人={_rc_name}({(_rc_pid or '')[:12]}) injected={_rc_injected}(pid={_rc_injected_pid or '无'})")
            elif etype == "response.function_call_arguments.done":
                name = event.get("name", "")
                call_id = event.get("call_id", "")
                _fc_args = event.get("arguments", "")
                with st.lock:
                    st.fc_seen_this_resp = True
                    st.fc_gen = st.play_gen
                    st.display_transcript_seq += 1
                    st.display_transcript.append({
                        "seq": st.display_transcript_seq,
                        "ts": time.strftime("%H:%M:%S"),
                        "role": "tool_call",
                        "text": f"{name}({_fc_args})",
                        "pid": st.resp_snapshot_pid or st.current_person_id or "_unknown",
                        "name": st.resp_snapshot_name or st.current_person_name,
                    })
                log(f"🤖 模型调用工具: {name}({_fc_args[:200]})")
                if name == "turn_body":
                    self._turn_body_called = True
                try:
                    args_dict = json.loads(_fc_args) if _fc_args else {}
                except (json.JSONDecodeError, TypeError):
                    args_dict = {}
                # ── legacy snapshot 工具（已移除但防残留调用）──
                if name in ("take_snapshot", "identify_pointed_object"):
                    with st.lock:
                        maybe_pointing = (time.monotonic() - st.finger_ext_at) < POINT_FRESH_S if name == "take_snapshot" else True
                        st.snapshot_pending += 1
                    mode = "judge" if maybe_pointing else "scene"
                    self.snap_q.put({"call_id": call_id, "gen": st.fc_gen, "mode": mode})
                # ── registry 统一分发 ──
                elif self.registry is not None and self.registry.get(name) is not None:
                    tool = self.registry.get(name)
                    deps = ToolDeps(
                        st=st, conv=self.conv, motion_q=self.motion_q,
                        memory_mgr=self.memory_mgr, owner_mgr=self.owner_mgr,
                        identity_store=self.identity_store, face_pipeline=self.face_pipeline,
                    )
                    try:
                        output = tool.execute(deps, call_id, args_dict)
                        if output is not None:
                            log(f"📤 create_item(tool_output) call_id={call_id[:8]} tool={name} output={output[:200]}")
                            self.conv.create_item({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": output,
                            })
                            _record_tool_output(st, name, call_id, output)
                    except Exception as e:
                        log(f"⚠ 工具 {name} 执行失败:{type(e).__name__}: {e}")
                        try:
                            self.conv.create_item({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps({"success": False, "error": str(e)}, ensure_ascii=False),
                            })
                        except Exception:
                            pass
                else:
                    log(f"⚠ 未注册工具: {name}")
                    try:
                        self.conv.create_item({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"success": False, "error": f"unknown tool: {name}"}, ensure_ascii=False),
                        })
                    except Exception:
                        pass
            elif etype == "response.audio_transcript.delta":
                print(event.get("delta", ""), end="", flush=True)
            elif etype == "response.audio_transcript.done":
                print(flush=True)
                _atext = (event.get("transcript") or "").strip()
                if _atext:
                    for m in _ACTION_TAG_RE.finditer(_atext):
                        act = _extract_tag_action(m.group())
                        if act:
                            log(f"⚠ 标签泄漏兜底: '{m.group()}' → 触发 {act}")
                            self.motion_q.put({"name": act})
                    _atext = _ACTION_TAG_RE.sub("", _atext).strip()
                if _atext:
                    log(f"💬 小艺:{_atext}")        # 模型回复入 log(网页 log 面板可见)
                    with st.lock:
                        _log_pid = st.resp_snapshot_pid or st.current_person_id or "_unknown"
                        _log_name = st.resp_snapshot_name or st.current_person_name
                        st.display_transcript_seq += 1
                        st.display_transcript.append({"seq": st.display_transcript_seq, "ts": time.strftime("%H:%M:%S"), "role": "assistant", "text": _atext, "pid": _log_pid, "name": _log_name})
                        if len(st.display_transcript) > 100:
                            st.display_transcript = st.display_transcript[-80:]
                    log(f"📝 模型回复:「{_atext[:100]}{'…' if len(_atext)>100 else ''}」 归属={_log_name}({_log_pid[:12]})")
                    if not st.no_memory:
                        with st.lock:
                            st.conversation_log.setdefault(_log_pid, []).append(("assistant", _atext))
            elif etype == "response.audio.delta":
                with st.lock:
                    if st.drop_audio:
                        return
                    gen = st.play_gen
                    st.resp_audio_count += 1
                    if st.thinking:
                        st.thinking = False
                b64 = event.get("delta") or event.get("audio") or ""
                pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                f16k = resample_poly(pcm.astype(np.float32) / 32768.0, PLAY_SR, OUT_SR).astype(np.float32)
                self.play_q.put((gen, f16k))
            elif etype == "response.done":
                fire_rc = False
                with st.lock:
                    st.in_flight = max(0, st.in_flight - 1)
                    st.resp_snapshot_pid = None
                    st.resp_snapshot_name = None
                    st.last_interaction_at = now
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
                    log(f"📤 create_response(仅调工具无语音) 自动触发")
                    # 工具轮后的语音也要带「当前说话人」强指令(否则漏掉 D → 从历史捞别人名字,
                    # 正是"tilt_head 后答你是大大"的漏洞)。说话人取本轮 turn_speaker。
                    with st.lock:
                        _tsp = st.turn_speaker_pid
                        _tsn = st.turn_speaker_name
                    _instr_fc = (self.dialog.resp_directive(_tsp, _tsn, _tsp is not None)
                                 if self.dialog is not None else None)
                    try:
                        self.conv.create_response(instructions=_instr_fc)
                    except Exception as _e:
                        log(f"⚠ create_response(工具轮) 失败:{type(_e).__name__}: {_e}")
                # ── turn_body 兜底:正则预过滤 + qwen-plus 语义判断 → 确认才补发 ──
                if self._pending_turn_cmd is not None and not self._turn_body_called:
                    threading.Thread(
                        target=self._judge_turn_body,
                        args=(self._pending_transcript, self._pending_turn_cmd),
                        daemon=True,
                    ).start()
                self._pending_turn_cmd = None
                self._turn_body_called = False
            elif etype == "error":
                log(f"❌ 服务端错误事件:{event}")
        except Exception as e:
            log(f"❌ on_event 处理异常:{type(e).__name__}: {e}\n   原始事件:{str(event)[:300]}")

    def _judge_turn_body(self, transcript: str, parsed_cmd: dict):
        """qwen-plus 判断用户话语是否是转身指令;确认则补发 motion_q + 标记 fallback。"""
        try:
            if self.dialog is None or self.dialog.oai is None:
                _record_turn_bad_case(transcript, parsed_cmd)
                return
            prompt = (
                "判断下面这句话是不是在**命令**机器人转身/转向。\n"
                "只有「直接命令转身」才算(如'向右转''转过去看看''面朝那边');\n"
                "「问转了多少/讨论转身/提到转向概念/描述过去的动作」不算。\n"
                f"用户说:「{transcript}」\n"
                "严格只输出 JSON,不要解释:\n"
                '{"is_turn": true或false, "direction": "left"或"right"或"center", "angle": 30到90的整数}'
            )
            resp = self.dialog.oai.chat.completions.create(
                model=EXTRACT_MODEL,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": "请输出 JSON。"}],
                temperature=0.1,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            if not result.get("is_turn"):
                log(f"🔍 turn_body 兜底:qwen-plus 判定非转身指令,跳过 (原文:{transcript[:50]})")
                _record_turn_bad_case(transcript, parsed_cmd)
                return
            direction = str(result.get("direction", parsed_cmd["direction"])).lower().strip()
            if direction not in ("left", "right", "center"):
                direction = parsed_cmd["direction"]
            try:
                angle = int(result.get("angle", parsed_cmd["angle"]))
                angle = max(10, min(angle, 90))
            except (TypeError, ValueError):
                angle = parsed_cmd["angle"]
            self.motion_q.put({
                "name": "turn_body",
                "call_id": f"fallback_{int(time.monotonic())}",
                "args": {"direction": direction, "angle": angle},
            })
            log(f"🔄 turn_body 兜底:qwen-plus 确认转身 → 补发 {direction} {angle}°")
            with self.st.lock:
                self.st.turn_body_fallback_fired = True
        except Exception as e:
            log(f"⚠ turn_body 兜底判断失败:{type(e).__name__}: {e}")
            _record_turn_bad_case(transcript, parsed_cmd)


class RealtimeDialog:
    """Qwen-Omni-Realtime 对话协议管理器 — 封装 session 生命周期。"""

    def __init__(self, st: State, play_q, motion_q, snap_q, mini: ReachyMini,
                 oai_client, memory_mgr, owner_mgr, identity_store=None,
                 instructions: str = "", registry=None, no_memory: bool = False,
                 face_pipeline=None, asd_engine=None):
        self.callback = ChatCallback(st, play_q, motion_q, snap_q, mini,
                                     memory_mgr, owner_mgr, identity_store,
                                     registry=registry,
                                     face_pipeline=face_pipeline, asd_engine=asd_engine)
        self.callback.dialog = self
        self.st = st
        self.oai = oai_client
        self.memory_mgr = memory_mgr
        self.instructions = instructions
        self.registry = registry
        self.tools = registry.specs() if registry is not None else []
        self.no_memory = no_memory
        self.conv = None
        self._last_inject_fail = 0.0
        self._last_connect_at = 0.0
        self._min_connect_gap = 1.0

    def open_session(self, timeout: float = CONNECT_TIMEOUT_S):
        """新建 WS + update_session,timeout 内未就绪 → None。"""
        gap = time.monotonic() - self._last_connect_at
        if gap < self._min_connect_gap:
            time.sleep(self._min_connect_gap - gap)
        self._last_connect_at = time.monotonic()
        st = self.st
        st.session_updated.clear()
        c = OmniRealtimeConversation(model=MODEL, callback=self.callback)
        holder = {"err": None}
        def _w():
            try:
                c.connect()
                log(f"📤 update_session(初始化) instructions前80字: {self.instructions[:80]}... tools数: {len(self.tools)}")
                c.update_session(
                    output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                    voice=VOICE,
                    input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                    output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                    enable_input_audio_transcription=True,
                    enable_turn_detection=True,
                    turn_detection_type="semantic_vad",
                    turn_detection_param={"create_response": False},   # 收回 turn-taking:VAD 只断句,回复我方手动建(保注入先于生成)
                    instructions=self.instructions,
                    tools=self.tools,
                )
            except Exception as e:
                holder["err"] = e
        threading.Thread(target=_w, daemon=True).start()
        if st.session_updated.wait(timeout):
            self.callback.conv = c
            self.conv = c
            with st.lock:
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

    def close_session(self):
        """断开 WS、清身份状态、触发 consolidation（遍历所有人的 conv_log）。"""
        st = self.st
        c = self.conv
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
        self.callback.conv = None
        self.conv = None
        with st.lock:
            _all_logs = dict(st.conversation_log)
            st.conversation_log.clear()
            st.identity_injected = False
            st.identity_injected_pid = None
            st.current_person_id = None
            st.current_person_name = None
            st.current_is_owner = False
            st.user_speaking = False
            if st.clear_workflow is not None:
                st.clear_workflow = None
                st.clear_lock = False
        if self.memory_mgr and not self.no_memory:
            for _pid, _log in _all_logs.items():
                if _pid != "_unknown" and len(_log) >= 2:
                    threading.Thread(target=self.save_summary,
                                     args=(_pid, _log), daemon=True).start()
        if _st_mod._current_turn is not None:
            _st_mod._current_turn["end_ts"] = time.strftime("%H:%M:%S")
            _st_mod._current_turn["end_mono"] = time.monotonic()
            _st_mod._current_turn = None
        _st_mod._pending_asr = ""

    def restart_session_for_switch(self, old_pid: str | None, new_pid: str, new_pname: str | None):
        """身份切换时重启 WS 会话，清除旧对话历史防止上下文污染。"""
        st = self.st
        log(f"🔄 身份切换重启: {old_pid and old_pid[:8]}→{new_pid[:8]} ({new_pname})")
        if old_pid and self.memory_mgr and not self.no_memory:
            with st.lock:
                _old_log = list(st.conversation_log.get(old_pid, []))
                st.conversation_log.pop(old_pid, None)
            if len(_old_log) >= 2:
                threading.Thread(target=self.save_summary,
                                 args=(old_pid, _old_log), daemon=True).start()
        c = self.conv
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
            self.callback.conv = None
            self.conv = None
        with st.lock:
            st.in_flight = 0
            st.resp_audio_count = 0
            st.fc_seen_this_resp = False
            st.drop_audio = False
            st.pending_identity_restart = False
        new_c = self.open_session()
        if new_c is not None:
            self.update_memory(new_pid, new_pname)
            log(f"✅ 会话重启完成，已注入 {new_pname or new_pid[:12]} 的记忆")
            return new_c
        else:
            log("⚠ 会话重启失败(open_session 超时)")
            return None

    def update_memory(self, pid: str, pname: str | None) -> bool:
        """用 update_session 将记忆嵌入 session instructions。"""
        if time.monotonic() - self._last_inject_fail < 2.0:
            return False
        st = self.st
        mem_prompt = self.memory_mgr.get_prompt(pid, person_name=pname) if self.memory_mgr else None
        # 已注册但未命名的说话人:绝不给占位名,明确告诉模型「还不知道名字、别编」(治 ?T 被读成名字)
        if not pname and not (self.memory_mgr and self.memory_mgr.get_name(pid)):
            _noname = ("【当前说话人】你还不知道对方叫什么名字,绝不要编名字或套用别人的名字;"
                       "若想知道可以礼貌地问对方怎么称呼。")
            mem_prompt = (mem_prompt + "\n" + _noname) if mem_prompt else _noname
        new_instr = self.instructions + ("\n\n" + mem_prompt if mem_prompt else "")
        c = self.conv
        if c is None:
            return False
        try:
            log(f"📤 update_session(记忆注入) pid={pid} name={pname} 记忆前80字: {(mem_prompt or '')[:80]}...")
            log(f"   完整instructions前200字: {new_instr}...")
            c.update_session(
                output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                voice=VOICE,
                input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                enable_input_audio_transcription=True,
                enable_turn_detection=True,
                turn_detection_type="semantic_vad",
                turn_detection_param={"create_response": False},   # 收回 turn-taking(同 open_session)
                instructions=new_instr,
                tools=self.tools,
            )
            log(f"🧠 记忆已注入 session instructions ({pname or pid[:12]})")
            with st.lock:
                st.identity_injected = True
                st.identity_injected_pid = pid
                st.dbg_memory_prompt = mem_prompt
                st.dbg_session_instructions = new_instr
                _buffered = list(st.audio_gate_buffer)
                st.audio_gate_buffer.clear()
                st.audio_gate_closed = False
            if _buffered:
                for chunk in _buffered:
                    try:
                        c.append_audio(chunk)
                    except Exception:
                        break
                log(f"🔓 音频闸门开启，flush {len(_buffered)} 帧缓存")
            return True
        except Exception as e:
            self._last_inject_fail = time.monotonic()
            log(f"⚠ 记忆 update_session 失败:{e}")
            return False

    def update_memory_neutral(self) -> bool:
        """画外/未识别说话人:注入「看不到对方、不知道是谁」的中性上下文(不带任何人的记忆),
        防止模型用上一个在场人的身份/记忆回答(如被问'我是谁'乱答别人名字)。"""
        if time.monotonic() - self._last_inject_fail < 2.0:
            return False
        st = self.st
        note = ("\n\n【当前说话人】对方不在摄像头画面里(或身份未识别),你看不到对方、不知道对方是谁。"
                "若被问'我是谁/你认识我吗/我叫什么/我喜欢什么'等,如实说你看不到对方、不确定是谁,"
                "绝不要套用其他人的名字或记忆来回答。")
        new_instr = self.instructions + note
        c = self.conv
        if c is None:
            return False
        try:
            c.update_session(
                output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                voice=VOICE,
                input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                enable_input_audio_transcription=True,
                enable_turn_detection=True,
                turn_detection_type="semantic_vad",
                turn_detection_param={"create_response": False},   # 收回 turn-taking(同 open_session)
                instructions=new_instr,
                tools=self.tools,
            )
            log("🫥 画外/未识别说话人 → 注入中性上下文(不认识对方,不套用他人身份)")
            with st.lock:
                st.identity_injected = True
                st.identity_injected_pid = "_neutral"
                st.dbg_memory_prompt = note
                st.dbg_session_instructions = new_instr
            return True
        except Exception as e:
            self._last_inject_fail = time.monotonic()
            log(f"⚠ 中性 update_session 失败:{e}")
            return False

    def resp_directive(self, cur_pid, cur_name, present: bool):
        """构造【单次回复】的强指令(create_response 的 response.instructions):
        基础人设 + 当前说话人身份约束。比 create_item 的 system 条目强得多(直接是这次生成的系统指令),
        治"模型无视注入、从会话历史里捞别人的名字/喜好"。带上 self.instructions 保底防人设被替换。"""
        if cur_name:
            _facts = self.memory_mgr.get_facts(cur_pid) if (cur_pid and self.memory_mgr) else {}
            _kv = [(k, v) for k, v in _facts.items()
                   if k not in ("称呼", "名字", "姓名", "昵称", "name")][:4]
            _fs = ("你记得TA:" + "；".join(f"{k}:{v}" for k, v in _kv) + "。") if _kv else ""
            d = (f"【本次回应对象】现在跟你说话的是「{cur_name}」。{_fs}"
                 f"若TA问「我是谁/我叫什么」就明确回答「{cur_name}」;"
                 "忽略之前对话里提到的其他人,绝不要用别人的名字或记忆来回答。")
        elif present:
            d = ("【本次回应对象】现在跟你说话的是一位你还没记住名字的人。"
                 "你不知道TA叫什么,可以自然地问TA怎么称呼。"
                 "若TA问「我是谁/我叫什么/你认识我吗」,如实说你们好像还没正式认识、你还不知道TA的名字,然后友好地问TA叫什么;"
                 "绝不要编名字、绝不要拿别人的名字或记忆来答。")
        else:
            d = ("【本次回应对象】对方不在画面里、你看不到TA。若被问身份,如实说看不到、不确定是谁;"
                 "别套用其他人的名字或记忆。")
        with self.st.lock:
            if self.st.turn_body_fallback_fired:
                d += ("【重要提醒】上一轮用户让你转身,你没有调用turn_body工具,系统已自动执行。"
                      "以后遇到转身指令请主动调用turn_body工具,不要只说话不调工具。")
                self.st.turn_body_fallback_fired = False
        return self.instructions + "\n\n" + d

    def extract_memory_async(self, pid: str, pname: str | None,
                             current_text: str, context_turns: list) -> None:
        """每轮工具审视:用 EXTRACT_MODEL(qwen-plus)从最近对话抽取「本句说话人」的个人事实,
        兜底 realtime 偶发漏调 remember_fact。context_turns=[(role,name,text),...] 最近 ~5 轮;
        只抽取最后一句(current_text)说话人的信息;save_fact 内置去重,与 realtime 自调不冲突。"""
        try:
            if not current_text or len(current_text.strip()) < 2:
                return
            _ctx_str = "\n".join(
                f"{'小艺' if r == 'assistant' else (nm or '用户')}: {t}"
                for (r, nm, t) in context_turns if t and t.strip()
            )
            existing = self.memory_mgr.get_facts(pid)
            existing_str = json.dumps(existing, ensure_ascii=False) if existing else "[]"
            prompt = (
                "你是记忆抽取助手。下面是最近几轮对话,最后一句是「当前说话人」刚说的。\n"
                f"当前说话人:{pname or '未知'}。只抽取这个人关于他自己的信息,"
                "不要抽别人或小艺(机器人)说的内容。\n"
                f"已有记忆(不要重复):{existing_str}\n\n"
                f"最近对话:\n{_ctx_str}\n\n"
                f"当前这句(重点,可结合上文消解'它/那个/这个'等指代):{current_text}\n\n"
                "任务:判断当前说话人这句有没有透露关于他本人的个人信息"
                "(爱好/喜好/厌恶/职业/年龄/习惯/观点/重要的人或事)或本人姓名。\n"
                "严格只输出 JSON,不要解释:\n"
                '{"name": "本人说自己叫什么则填,否则 null",'
                '"facts": {"字段": "值", ...}}\n'
                "facts 用「字段:值」键值对,字段是简短中文类别(如 爱好/喜欢的食物/讨厌的东西/职业/年龄/习惯/观点/重要的人),值是具体内容;"
                "没有可记信息则 facts 为空对象 {}、name 为 null;"
                "不要把名字写进 facts;不要重复已有记忆。"
            )
            resp = self.oai.chat.completions.create(
                model=EXTRACT_MODEL,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": "请输出 JSON。"}],
                temperature=0.1,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            new_name = result.get("name")
            # facts 是 key-value 字典(与本分支 MemoryManager 的 entity memory 一致)。
            # 兼容模型偶发吐旧格式(list):非 dict 一律当空,避免 save_fact 缺参报错。
            _facts_obj = result.get("facts")
            new_facts = ({str(k).strip(): str(v).strip()
                          for k, v in _facts_obj.items()
                          if str(k).strip() and str(v).strip()}
                         if isinstance(_facts_obj, dict) else {})
            saved = 0
            for k, v in new_facts.items():
                r = self.memory_mgr.save_fact(pid, k, v)   # key-value,同 key 覆盖(LWW)
                if "已记住" in r or "已更新" in r:
                    saved += 1
            _cb = self.callback
            if new_name:
                if try_name_identity(
                        memory_mgr=self.memory_mgr, identity_store=_cb.identity_store,
                        face_pipeline=_cb.face_pipeline, owner_mgr=_cb.owner_mgr, st=self.st,
                        pid=pid, new_name=new_name, transcript=current_text,
                        log_fn=log):
                    log(f"🧠 工具审视:补记名字「{new_name}」({pid[:12]})")
            if saved:
                with self.st.lock:
                    if self.st.current_person_id == pid:
                        self.st.identity_injected = False    # 触发重注入,让模型用上新记忆
                log(f"🧠 工具审视:补存 {saved} 条记忆 → {pname or pid[:12]}")
        except Exception as e:
            log(f"⚠ 工具审视失败:{type(e).__name__}: {e}")

    def save_summary(self, pid: str, conv_log: list):
        """后台线程：会话后 consolidation — 一次 LLM 调用同时生成 entity memory + episodic memory。"""
        try:
            text = "\n".join(f"{'用户' if r == 'user' else '小艺'}: {t}"
                             for r, t in conv_log if t and t.strip())
            if len(text) < 20:
                return
            current_facts = self.memory_mgr.get_facts(pid)
            current_name = self.memory_mgr.get_name(pid)
            facts_str = json.dumps(current_facts, ensure_ascii=False) if current_facts else "{}"
            prompt = (
                "你是记忆管理助手。仔细阅读对话，提取关于用户的所有个人信息。\n\n"
                f"已有记忆：{facts_str}\n\n"
                f"对话内容：\n{text[-4000:]}\n\n"
                "任务：\n"
                "1. facts = KV 格式合并已有记忆 + 对话中发现的新信息。key 是类别，value 是内容。\n"
                "   - 重点提取：爱好、喜好、职业、年龄、习惯、观点、提到的人/事物\n"
                "   - 用户说'我喜欢X' → {\"喜欢的东西\": \"X\"}\n"
                "   - 如果新信息与旧 fact 矛盾，保留新的 value，同 key 覆盖\n"
                "   - 不要把名字放进 facts（名字在独立字段管理）\n"
                "2. summary = 一句话介绍这位用户。\n"
                "   - 体现对用户的整体理解，总结facts和对话而不是列举属性\n"
                "   - 仅做描述，例如：'喜欢打篮球和看科幻小说，最近在关注《黑暗森林》上线'\n"
                "3. episode = 客观总结这次对话的摘要事件\n"
                "   - topic: 具体说聊了什么，不要太笼统\n"
                "   - highlights: 关于用户的关键信息点（每条是完整短句）\n\n"
                "只输出JSON：\n"
                '{"name":"用户的名字(用户自己说出自己叫什么才填,否则为null。注意:小艺是机器人的名字,不是用户的名字)",'
                '"summary":"一句话认知描述",'
                '"facts":{"类别1":"内容1","类别2":"内容2"},'
                '"episode":{"topic":"具体话题","highlights":["要点1"],"mood":"engaged/casual/emotional/tense"}}'
            )
            resp = self.oai.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "请根据上述信息生成记忆JSON。"},
                ],
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            new_facts = result.get("facts", current_facts)
            if isinstance(new_facts, list):
                new_facts = {f"备注{i+1}": f for i, f in enumerate(new_facts)}
            new_name = result.get("name")
            # ── 治本:consolidation 提取的 name 也必须过 _valid_name 校验 ──
            # 根因(bug-075):LLM 把对话中机器人自称"小艺"当用户名字提取了
            if new_name and not _valid_name(new_name):
                log(f"⚠ consolidation 名字被拒:「{new_name}」不合法(bot名/格式)")
                new_name = None
            if new_name is None and current_name:
                new_name = current_name
            new_summary = result.get("summary")
            episode = result.get("episode")
            self.memory_mgr.consolidate_facts(pid, new_facts, new_name, new_summary)
            if episode:
                self.memory_mgr.save_episode(pid, episode)
            log(f"📝 记忆 consolidation 完成 ({pid[:12]}): {len(new_facts)} facts, summary={new_summary[:30] if new_summary else 'none'}, episode={episode.get('topic', '')[:30] if episode else 'none'}")
            with self.st.lock:
                if self.st.current_person_id == pid:
                    self.st.identity_injected = False
        except Exception as e:
            log(f"⚠ 记忆 consolidation 失败({pid[:12]}):{e}")
