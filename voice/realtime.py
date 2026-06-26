# -*- coding: utf-8 -*-
"""Qwen-Omni-Realtime 对话协议层 — 回调 + 会话生命周期管理。"""

import base64
import json
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

from memory.safety import handle_clear_memory_intent, handle_confirm_clear
from voice.config import (
    MODEL, VOICE, SUMMARY_MODEL, CONNECT_TIMEOUT_S,
    BYE_PHRASES, POINT_FRESH_S, OUT_SR, PLAY_SR,
    CONV_SUMMARY_THRESHOLD,
)
from voice.state import State, log, _record_event
import voice.state as _st_mod


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


class ChatCallback(OmniRealtimeCallback):
    """Qwen Omni Realtime 事件回调 — 音频播放、barge-in、工具分发、transcript 解析。"""

    def __init__(self, st: State, play_q: "queue.Queue", motion_q: "queue.Queue",
                 snap_q: "queue.Queue", mini: ReachyMini,
                 memory_mgr, owner_mgr, id_recognizer, face_pipeline=None):
        self.st = st
        self.play_q = play_q
        self.motion_q = motion_q
        self.snap_q = snap_q
        self.mini = mini
        self.memory_mgr = memory_mgr
        self.owner_mgr = owner_mgr
        self.id_recognizer = id_recognizer
        self.face_pipeline = face_pipeline   # 命名时落 gallery(confirm_identity)
        self.conv: OmniRealtimeConversation | None = None
        self.dialog: "RealtimeDialog | None" = None
        self.exit_i = 0

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
                with st.lock:
                    _log_pid = st.current_person_id or "_unknown"
                    _log_name = st.current_person_name
                    _injected_pid = st.identity_injected_pid
                log(f"📝 ASR结果:「{_transcript}」 当前人={_log_name}({_log_pid[:12]}) 已注入记忆pid={_injected_pid or '无'}")
                if _transcript:
                    with st.lock:
                        _log_pid = st.current_person_id or "_unknown"
                        _log_name = st.current_person_name
                        st.display_transcript_seq += 1
                        st.display_transcript.append({"seq": st.display_transcript_seq, "ts": time.strftime("%H:%M:%S"), "role": "user", "text": _transcript, "pid": _log_pid, "name": _log_name})
                        if len(st.display_transcript) > 100:
                            st.display_transcript = st.display_transcript[-80:]
                    if not st.no_memory:
                        with st.lock:
                            st.conversation_log.setdefault(_log_pid, []).append(("user", _transcript))
                            _check_log = st.conversation_log.get(_log_pid, [])
                            _est_tok = sum(len(t) * 1.5 for _, t in _check_log)
                        if _est_tok > CONV_SUMMARY_THRESHOLD and _log_pid != "_unknown" and self.memory_mgr:
                            with st.lock:
                                _snap = list(st.conversation_log.get(_log_pid, []))
                                st.conversation_log[_log_pid] = []
                            if self.dialog:
                                threading.Thread(target=self.dialog.save_summary,
                                                 args=(_log_pid, _snap), daemon=True).start()
                            log(f"📝 上下文过长，自动触发 consolidation({_log_pid[:12]}, ~{int(_est_tok)} tok)")
            elif etype == "response.created":
                with st.lock:
                    st.in_flight += 1
                    st.drop_audio = False
                    st.resp_audio_count = 0
                    st.fc_seen_this_resp = False
                    st.last_interaction_at = now
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
                if name == "take_snapshot":
                    with st.lock:
                        maybe_pointing = (time.monotonic() - st.finger_ext_at) < POINT_FRESH_S
                        st.snapshot_pending += 1
                    mode = "judge" if maybe_pointing else "scene"
                    if maybe_pointing:
                        log("👉 最近见过伸指 → 先原地看图判断是否真在指(两段式)")
                    self.snap_q.put({"call_id": call_id, "gen": st.fc_gen, "mode": mode})
                elif name == "end_session":
                    phrase = BYE_PHRASES[self.exit_i % len(BYE_PHRASES)]
                    self.exit_i += 1
                    output_msg = {"success": True,
                                  "say": f"对话结束。用中文只说这一句简短告别:「{phrase}」,别追问、别挽留、别加别的。"}
                    try:
                        log(f"📤 create_item(tool_output) call_id={call_id[:8]} tool=end_session output={json.dumps(output_msg, ensure_ascii=False)}")
                        self.conv.create_item({
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps(output_msg, ensure_ascii=False),
                        })
                        _record_tool_output(st, "end_session", call_id, json.dumps(output_msg, ensure_ascii=False))
                    except Exception as e:
                        log(f"⚠ end_session 回 output 失败:{e}")
                    with st.lock:
                        st.exit_request = True
                    log(f"👋 收到结束意图 → 告别「{phrase}」+ 回待命")
                elif name == "identify_pointed_object":
                    with st.lock:
                        st.snapshot_pending += 1
                    log("👉 收到指向请求 → 先原地看图判断(两段式)")
                    self.snap_q.put({"call_id": call_id, "gen": st.fc_gen, "mode": "judge"})
                elif name in ("remember_fact", "forget_fact"):
                    with st.lock:
                        pid = st.resp_snapshot_pid or st.current_person_id
                    if pid is None:
                        result = "当前没有识别到用户身份,无法存储记忆。"
                    else:
                        args_str = event.get("arguments", "{}")
                        try:
                            args_dict = json.loads(args_str)
                        except (json.JSONDecodeError, TypeError):
                            args_dict = {}
                        result = self.memory_mgr.handle_tool_call(pid, name, args_dict)
                        with st.lock:
                            st.identity_injected = False
                            st.identity_injected_pid = None
                        if name == "remember_fact":
                            new_name = args_dict.get("name")
                            if new_name:
                                self.memory_mgr.set_name(pid, new_name)
                                if self.id_recognizer is not None:
                                    self.id_recognizer.db.set_name(pid, new_name)
                                # 命名 → gallery 确认(provisional→confirmed,带真名)并落盘
                                if self.face_pipeline is not None:
                                    try:
                                        if self.face_pipeline.store.confirm_identity(pid, new_name):
                                            self.face_pipeline.save_gallery()
                                            log(f"🏷 gallery 身份已确认并落盘: {new_name} ({pid[:12]})")
                                    except Exception as _e:
                                        log(f"⚠ gallery 命名失败:{type(_e).__name__}: {_e}")
                                with st.lock:
                                    st.current_person_name = new_name
                                if self.owner_mgr is not None and not self.owner_mgr.has_owner():
                                    if self.owner_mgr.try_claim(pid, new_name):
                                        log(f"👑 认主成功: {new_name} ({pid})")
                        elif name == "forget_fact":
                            keyword = args_dict.get("keyword", "")
                            if "名" in keyword or "name" in keyword.lower():
                                self.memory_mgr.set_name(pid, None)
                                if self.id_recognizer is not None:
                                    self.id_recognizer.db.set_name(pid, None)
                                with st.lock:
                                    st.current_person_name = None
                    try:
                        result_payload = {"result": result}
                        log(f"📤 create_item(tool_output) call_id={call_id[:8]} tool={name} output={json.dumps(result_payload, ensure_ascii=False)[:200]}")
                        self.conv.create_item({
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps(result_payload, ensure_ascii=False),
                        })
                        _record_tool_output(st, name, call_id, json.dumps(result_payload, ensure_ascii=False))
                    except Exception as e:
                        log(f"⚠ 记忆工具回 output 失败:{e}")
                    log(f"🧠 记忆工具 {name}: {result}")
                elif name == "clear_memory":
                    args_str = event.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}
                    result = handle_clear_memory_intent(st, args_dict, self.conv,
                                                       self.id_recognizer)
                    try:
                        result_payload = {"result": result}
                        log(f"📤 create_item(tool_output) call_id={call_id[:8]} tool=clear_memory output={json.dumps(result_payload, ensure_ascii=False)[:200]}")
                        self.conv.create_item({
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps(result_payload, ensure_ascii=False),
                        })
                        _record_tool_output(st, "clear_memory", call_id, json.dumps(result_payload, ensure_ascii=False))
                    except Exception as e:
                        log(f"⚠ clear_memory 回 output 失败:{e}")
                    log(f"🔒 clear_memory 启动: {result}")
                elif name == "confirm_clear":
                    args_str = event.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}
                    result = handle_confirm_clear(st, args_dict,
                                                  self.memory_mgr, self.id_recognizer)
                    try:
                        result_payload = {"result": result}
                        log(f"📤 create_item(tool_output) call_id={call_id[:8]} tool=confirm_clear output={json.dumps(result_payload, ensure_ascii=False)[:200]}")
                        self.conv.create_item({
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps(result_payload, ensure_ascii=False),
                        })
                        _record_tool_output(st, "confirm_clear", call_id, json.dumps(result_payload, ensure_ascii=False))
                    except Exception as e:
                        log(f"⚠ confirm_clear 回 output 失败:{e}")
                    log(f"🔒 confirm_clear: {result}")
                else:
                    self.motion_q.put({"name": name, "call_id": call_id})
                    output_msg = {"success": True, "action": name}
                    try:
                        log(f"📤 create_item(tool_output) call_id={call_id[:8]} tool={name} output={json.dumps(output_msg, ensure_ascii=False)}")
                        self.conv.create_item({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(output_msg, ensure_ascii=False),
                        })
                        _record_tool_output(st, name, call_id, json.dumps(output_msg, ensure_ascii=False))
                    except Exception as e:
                        log(f"⚠ 回 function_call_output 失败:{e}")
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
                    self.conv.create_response()
            elif etype == "error":
                log(f"❌ 服务端错误事件:{event}")
        except Exception as e:
            log(f"❌ on_event 处理异常:{type(e).__name__}: {e}\n   原始事件:{str(event)[:300]}")


class RealtimeDialog:
    """Qwen-Omni-Realtime 对话协议管理器 — 封装 session 生命周期。"""

    def __init__(self, st: State, play_q, motion_q, snap_q, mini: ReachyMini,
                 oai_client, memory_mgr, owner_mgr, id_recognizer,
                 instructions: str, tools: list, no_memory: bool = False,
                 face_pipeline=None):
        self.callback = ChatCallback(st, play_q, motion_q, snap_q, mini,
                                     memory_mgr, owner_mgr, id_recognizer,
                                     face_pipeline=face_pipeline)
        self.callback.dialog = self
        self.st = st
        self.oai = oai_client
        self.memory_mgr = memory_mgr
        self.instructions = instructions
        self.tools = tools
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
                "2. summary = 基于所有 facts 和对话理解，写一句话描述你对这个人的整体认知。\n"
                "   - 体现对用户的整体理解，不是列举属性\n"
                "   - 用「用户」指代，例如：'用户喜欢打篮球和看科幻小说，最近在关注《黑暗森林》上线'\n"
                "3. episode = 这次对话的结构化事件\n"
                "   - topic: 具体说聊了什么，不要太笼统，用「用户」指代\n"
                "   - highlights: 关键信息点（每条是完整短句），用「用户」指代\n\n"
                "只输出JSON：\n"
                '{"name":"用户名字(对话中提到则更新,否则为null)",'
                '"summary":"一句话认知描述(用\\"用户\\"指代)",'
                '"facts":{"类别1":"内容1","类别2":"内容2"},'
                '"episode":{"topic":"具体话题(用\\"用户\\"指代)","highlights":["要点1(用\\"用户\\"指代)"],"mood":"engaged/casual/emotional/tense"}}'
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
