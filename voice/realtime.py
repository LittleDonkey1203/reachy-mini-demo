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

from memory.safety import handle_clear_memory_intent, handle_confirm_clear
from voice.config import (
    MODEL, VOICE, SUMMARY_MODEL, EXTRACT_MODEL, CONNECT_TIMEOUT_S,
    BYE_PHRASES, POINT_FRESH_S, OUT_SR, PLAY_SR,
    CONV_SUMMARY_THRESHOLD,
)
from voice.state import State, log, _record_event
import voice.state as _st_mod


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


# ── 命名 guard:命名是身份关键操作,统一过门(治脑补名/画外命名/反复改名)──
_NAME_OK_RE = re.compile(r"^[一-龥A-Za-z·]{1,8}$")          # 1-8 中/英文字,无数字/标点/空格
_RENAME_INTENT_RE = re.compile(r"改名|改个名|改成|其实叫|实际叫|叫错|应该叫|重新.{0,4}名|不叫")


def _valid_name(name: str) -> bool:
    if not name:
        return False
    n = name.strip()
    return bool(_NAME_OK_RE.match(n)) and not n.startswith("?T") and n not in ("画外", "未知")


# fact 若是"纯报名字句"(叫X / 名字是X),名字只能走 try_name_identity 过守卫,
# 绝不能作为一条事实漏存(治 remember_fact 把"叫X"当 fact 存,守卫拒名后仍残留假事实)。
_NAME_FACT_RE = re.compile(
    r"^\s*(?:我)?\s*(?:"
    r"(?:就)?(?:叫做|叫作|名叫|叫)\s*(?P<n1>[一-龥A-Za-z·]{1,8}?)"
    r"|(?:的)?(?:名字|全名|姓名)\s*(?:就)?(?:是|叫做|叫|为)?\s*(?P<n2>[一-龥A-Za-z·]{1,8}?)"
    r")\s*[。.!！~呀啦哦呢]*\s*$")


def _name_only_fact(fact: str, name_hint: str | None = None) -> str | None:
    """fact 是"纯报名字句"→ 返回其中名字;否则 None。
    优先正则(叫X / 名字是X);name_hint 已知时兜底:去掉该名字+命名词后无残留也算纯名字句。
    注意只认 叫/名字 类,不认裸"是X"(否则"是医生""是个程序员"会被误判成名字)。"""
    if not fact:
        return None
    m = _NAME_FACT_RE.match(fact.strip())
    if m:
        return m.group("n1") or m.group("n2")
    if name_hint and name_hint in fact:
        residual = re.sub(r"(叫做|叫作|名叫|叫|名字是|名字|称呼|姓名|全名|就是|是|我|的)", "",
                          fact.replace(name_hint, "")).strip(" 。.，,、!！~呀啦哦呢")
        if not residual:
            return name_hint
    return None


def _idtag(pid, name=None):
    """日志用可区分身份标签。同一秒创建的身份 pid 前缀相同(id_<时间戳>...),
    用 pid[:8] 会把两个人打成同一个 → 有名用名,否则用尾 6 位(尾部才区分)。"""
    if name:
        return name
    if not pid:
        return "None"
    if pid in ("_neutral", "_ctx"):
        return pid
    return f"?{str(pid)[-6:]}"


def try_name_identity(*, memory_mgr, id_recognizer, face_pipeline, owner_mgr, st,
                      pid, new_name, transcript, log_fn, allow_rename=True) -> bool:
    """命名/改名统一 guard。返回是否真正写入了名字。
    门1 名字合法;门2 名字必须出现在当轮转写里(防模型脑补);门3 已命名不静默覆盖(仅显式改名意图才改)。
    pid 由调用方保证 = 本句说话人(画外/无归属时为 None,直接拒)。"""
    if not pid or not new_name:
        return False
    n = new_name.strip()
    # 门1 合法性
    if not _valid_name(n):
        log_fn(f"🚫 命名拒绝:名字不合法「{new_name}」")
        return False
    # 门2 必须来自当轮转写(防模型脑补:用户说毕夏、模型却记陛下)
    if not (transcript and n in transcript):
        log_fn(f"🚫 命名拒绝:「{n}」不在转写里(防脑补)← 「{(transcript or '')[:30]}」")
        return False
    # 门2.5 防漏名:名字已被"在场另一个身份"占用 → 拒绝。多人时模型爱把历史里的"我叫大大"
    #   echo 到别人 + ASD 误归 → 两个身份撞同名。真要重名(俩人都叫大大)走 dashboard 手动注册。
    with st.lock:
        _roster_now = list(getattr(st, "roster", []) or [])
    for _rp, _rn in _roster_now:
        if _rp != pid and _rn and _rn == n:
            log_fn(f"🚫 命名拒绝:「{n}」已是在场另一人({_idtag(_rp)})的名字,不重复贴(防漏名/张冠李戴)")
            return False
    # 门3 不静默改名
    existing = memory_mgr.get_name(pid) if memory_mgr else None
    if existing:
        if existing == n:
            return False                                   # 重复声明,no-op
        if not (allow_rename and _RENAME_INTENT_RE.search(transcript)):
            log_fn(f"🚫 改名拒绝:已是「{existing}」,无明确改名意图→不覆盖为「{n}」")
            return False
        log_fn(f"✏ 改名:「{existing}」→「{n}」(检测到改名意图)")
    # 通过 → 写三处库 + gallery 落盘 + 认主
    if memory_mgr:
        memory_mgr.set_name(pid, n)
    if id_recognizer is not None:
        id_recognizer.db.set_name(pid, n)
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
                 memory_mgr, owner_mgr, id_recognizer, face_pipeline=None, asd_engine=None):
        self.st = st
        self.play_q = play_q
        self.motion_q = motion_q
        self.snap_q = snap_q
        self.mini = mini
        self.memory_mgr = memory_mgr
        self.owner_mgr = owner_mgr
        self.id_recognizer = id_recognizer
        self.face_pipeline = face_pipeline   # 命名时落 gallery(confirm_identity)
        self.asd_engine = asd_engine         # 谁在说话:本句归属用 speaker_window
        self._speech_start_t = 0.0           # 本句说话起点(monotonic),speech_started 时记
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
                        _asp = _hold
                if _asp is not None:
                    _tid = _asp.get("track_id")
                    _log_pid = _asp.get("pid") or f"_track{_tid}"      # 在画面但未识别:临时 track 键
                    _real_name = _asp.get("name")                      # 真名(未命名身份=None);注入/turn_speaker 只用它
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
                        # ── ① 记忆注入跟「本句说话人」:回这句话前注入说话人的记忆,让回复称呼对人。
                        #    不写 current_person_id(那是视觉稳定焦点,改它会和视觉循环打架→反复重注入竞态)。
                        #    只在说话人 ≠ 上次已注入者时重注入,避免 update_session 抖动(治"归属对但叫错人")。
                        # 注入走 create_item(通道B):把「在场 roster + 带说话人标签的近史 + 规则」注进对话流,
                        # 不改 system prompt(治多人张冠李戴 + 共享历史串味 + "看不见你")。base 人设 connect 已设。
                        # create_item 不可用(Qwen 拒 system item)才回退老的 update_session,且回退仍保阶段0(有人别翻中性)。
                        if self.dialog is not None and self.memory_mgr is not None:
                            _cur_pid = _log_pid if _tspk_real else None
                            if not self.dialog.inject_context(_cur_pid, _real_name):
                                with st.lock:
                                    _present = st.present_count
                                if _tspk_real:
                                    self.dialog.update_memory(_log_pid, _real_name)
                                elif _present == 0:
                                    self.dialog.update_memory_neutral()
                        # ── 收回 turn-taking:VAD 只断句(turn_detection create_response=false),回复由我方在
                        #    "注入之后"手动建 → 保证这轮一定参考到刚注入的当前说话人上下文(根治时序竞态)。
                        #    守卫 in_flight==0:避免和唤醒招呼(behavior 侧 create_response)双答;打断已 cancel 旧回复。
                        if _transcript and self.conv is not None:
                            with st.lock:
                                _busy = st.in_flight > 0
                            if not _busy:
                                try:
                                    self.conv.create_response()
                                except Exception as _e:
                                    log(f"⚠ create_response 失败:{type(_e).__name__}: {_e}")
                            else:
                                log(f"⏭ 跳过 create_response(in_flight={st.in_flight},招呼/旧回复在途)")
                        # ── ② 每轮工具审视:无条件用 qwen-plus 抽「本句说话人」的记忆,兜底 realtime 漏调 remember_fact ──
                        if _tspk_real and self.dialog is not None and self.memory_mgr is not None:
                            with st.lock:
                                _recent = [d for d in st.display_transcript
                                           if d.get("role") in ("user", "assistant")][-10:]
                            _ctx = [(d.get("role"), d.get("name"), d.get("text")) for d in _recent]
                            threading.Thread(target=self.dialog.extract_memory_async,
                                             args=(_log_pid, _real_name, _transcript, _ctx),
                                             daemon=True).start()
            elif etype == "response.created":
                with st.lock:
                    st.in_flight += 1
                    st.resp_created_at = now          # 级联:记回复起始时刻(判"年龄"→接管 or 排队)
                    st.active_resp_id = event.get("response", {}).get("id")   # 级联忙判据:最新 response 取代前者为 active
                    st.active_resp_done = False
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
                if _st_mod._current_turn is not None:
                    _st_mod._current_turn["dt_seq"] = _dt_seq
                log(f"💭 模型开始生成回复… tsp={_idtag(st.turn_speaker_pid, st.turn_speaker_name)} inj={_idtag(st.identity_injected_pid)}")
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
                log(f"🤖 模型调用工具: {name}")
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
                    with st.lock:
                        st.snapshot_pending += 1
                    log("👉 收到指向请求 → 先原地看图判断(两段式)")
                    self.snap_q.put({"call_id": call_id, "gen": st.fc_gen, "mode": "judge"})
                elif name in ("remember_fact", "forget_fact"):
                    args_str = event.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}
                    with st.lock:
                        pid = st.resp_snapshot_pid     # 存/删 fact 的归属:画外/无归属→None→不存(治画外事实张冠李戴)
                        _focus_pid = st.current_person_id    # 视觉焦点(小艺正看着的那位)
                        _present = st.present_count
                    # 命名兜底:说了名字但本句归属画外/None,而画面里**只有一个人**(present==1,无歧义)→ 名字落到焦点人。
                    #   ASD 常把"我叫X"甩进画外→命名全丢(dashboard 标签不更新);facts 仍严格按 pid 不兜回。
                    #   ⚠ 必须 present==1:多人时模型会多轮调 remember_fact(name=X),焦点在两人间飘→名字漏给非说话人
                    #   (两个身份都被命名成同一个名 → 另一人说话也归到 X)。多人命名只走真实 ASD 归属或 dashboard 注册。
                    _name_arg = args_dict.get("name") if name == "remember_fact" else None
                    _name_pid = pid
                    if _name_arg and pid is None and _focus_pid is not None and _present == 1:
                        _name_pid = _focus_pid
                        log(f"🪪 命名兜底(单人):本句归属画外但画面只有焦点人 → 名字落到 {_idtag(_focus_pid)}")
                    if pid is None and _name_pid is None:
                        result = "当前没有识别到用户身份(可能说话人不在画面里),无法存储记忆。"
                    elif name == "remember_fact":
                        # 名字只走命名 guard,绝不作为 fact 漏存(治"叫X"假事实)
                        _fact = (args_dict.get("fact") or "").strip()
                        _name_in_fact = _name_only_fact(_fact, _name_arg)
                        _eff_name = _name_arg or _name_in_fact
                        _fact_is_name_only = bool(_name_in_fact)   # fact 整句只是报名字 → 不存
                        if _fact_is_name_only or pid is None:
                            result = "好的。"                       # 纯报名字 / fact 无归属 → 只走命名,不存 fact
                        else:
                            result = self.memory_mgr.handle_tool_call(pid, name, args_dict)
                        with st.lock:
                            st.identity_injected = False
                            st.identity_injected_pid = None
                        if _eff_name and _name_pid is not None:
                            with st.lock:                  # 取当轮用户转写,供命名 guard 校验(防脑补)
                                _turn_text = next((d.get("text", "") for d in reversed(st.display_transcript)
                                                   if d.get("role") == "user"), "")
                            _named = try_name_identity(
                                memory_mgr=self.memory_mgr, id_recognizer=self.id_recognizer,
                                face_pipeline=self.face_pipeline, owner_mgr=self.owner_mgr, st=st,
                                pid=_name_pid, new_name=_eff_name, transcript=_turn_text, log_fn=log)
                            if _fact_is_name_only or pid is None:
                                result = "好的,记住你的名字啦。" if _named else "好的。"
                    else:  # forget_fact
                        if pid is None:
                            result = "当前没有识别到用户身份,无法操作。"
                        else:
                            result = self.memory_mgr.handle_tool_call(pid, name, args_dict)
                            with st.lock:
                                st.identity_injected = False
                                st.identity_injected_pid = None
                            keyword = args_dict.get("keyword", "")
                            if "名" in keyword or "name" in keyword.lower():
                                self.memory_mgr.set_name(pid, None)
                                if self.id_recognizer is not None:
                                    self.id_recognizer.db.set_name(pid, None)
                                with st.lock:
                                    st.current_person_name = None
                    try:
                        self.conv.create_item({
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps({"result": result}, ensure_ascii=False),
                        })
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
                        self.conv.create_item({
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps({"result": result}, ensure_ascii=False),
                        })
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
                        self.conv.create_item({
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps({"result": result}, ensure_ascii=False),
                        })
                    except Exception as e:
                        log(f"⚠ confirm_clear 回 output 失败:{e}")
                    log(f"🔒 confirm_clear: {result}")
                else:
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
                _fire_pending = False   # 级联:当前回复结束,补发被"排队"的 ASR 轮(智能接管的排队分支)
                _reinject = None     # A:回复结束、模型空闲 → 补注入(治多人忙时注入被 in_flight 门冻住→身份钉死)
                _done_id = event.get("response", {}).get("id")
                with st.lock:
                    st.in_flight = max(0, st.in_flight - 1)
                    if _done_id is not None and _done_id == st.active_resp_id:
                        st.active_resp_done = True     # 级联:最新 response 完成 → 空闲(孤儿旧 response 的 done 到不到都无所谓)
                    st.resp_snapshot_pid = None
                    st.resp_snapshot_name = None
                    st.last_interaction_at = now
                    if (st.active_resp_done and st.pending_resp_at > 0
                            and self.dialog is not None and self.dialog.cascade):
                        st.pending_resp_at = 0.0
                        _fire_pending = True
                    if (
                        st.fc_seen_this_resp
                        and st.resp_audio_count == 0
                        and st.fc_gen == st.play_gen
                        and st.snapshot_pending == 0
                    ):
                        fire_rc = True
                    # A:空闲(in_flight==0)下若"已注入身份" ≠ "最近说话人" → 补回那次被 _busy 跳过的注入
                    #   cascade:注入只走 handle_asr_turn 的 inject_context(create_item),绝不能在这里回退 update_session
                    #   (会把关掉的 turn_detection/转写又打开 → 破坏级联)。
                    if (st.in_flight == 0 and not st.no_memory
                            and not (self.dialog is not None and self.dialog.cascade)
                            and self.memory_mgr is not None and self.conv is not None):
                        _tsp = st.turn_speaker_pid
                        if _tsp is not None and st.identity_injected_pid != _tsp:
                            _reinject = ("mem", _tsp, st.turn_speaker_name)
                        elif (_tsp is None and st.present_count == 0
                              and st.identity_injected_pid != "_neutral"):
                            # 阶段0:只有画面里真没人才补中性;有人(present>0)时绝不把真身份翻成"看不见"
                            _reinject = ("neu", None, None)
                d = self.conv.get_last_first_audio_delay() if self.conv else None
                log(f"✅ 本轮回复完成{f'(首音频延迟 {d:.0f}ms)' if d else ''}")
                if _reinject is not None and self.dialog is not None:     # 先补注入再可能 fire_rc,让后续回复用对上下文
                    log(f"🔁 补注入(done空闲) {_reinject[0]} → "
                        + (_idtag(_reinject[1], _reinject[2]) if _reinject[0] == 'mem' else 'neutral'))
                    if _reinject[0] == "mem":
                        self.dialog.update_memory(_reinject[1], _reinject[2])
                    else:
                        self.dialog.update_memory_neutral()
                if (fire_rc or _fire_pending) and self.conv is not None:
                    if _fire_pending:
                        log("▶ [ASR]排队回复出:当前回复已完成,答排队的那轮")
                    self.conv.create_response()
            elif etype == "error":
                log(f"❌ 服务端错误事件:{event}")
        except Exception as e:
            log(f"❌ on_event 处理异常:{type(e).__name__}: {e}\n   原始事件:{str(event)[:300]}")

    def handle_asr_turn(self, transcript: str, start_mono: float) -> None:
        """CASCADE(阶段2):本地 ASR 一轮 → 归属 → 带说话人标签 user item + 记忆注入 + 手动 create_response。
        镜像 transcription.completed 的归属/turn_speaker/记忆逻辑,但文本由本地 ASR 给,
        且从源头给 user 轮打「name」标签(根治多人张冠李戴:模型天生区分多人 + 记忆走 inject_context 不串)。"""
        st = self.st
        transcript = (transcript or "").strip()
        if not transcript:
            return
        now = time.monotonic()
        # ── 归属:speaker_window(start_mono),与 S2S 同引擎同逻辑(耐 ASD 延迟)──
        _asp = None
        _sw = (self.asd_engine.speaker_window(start_mono)
               if (self.asd_engine is not None and self.asd_engine.available) else None)
        if _sw is not None:
            _key, _score = _sw
            if isinstance(_key, str) and _key.startswith("t"):
                _asp = {"pid": None, "name": None, "track_id": _key[1:], "score": _score, "at": now}
            else:
                _nm = self.memory_mgr.get_name(_key) if self.memory_mgr else None
                _asp = {"pid": _key, "name": _nm,
                        "track_id": self.asd_engine.last_track(_key), "score": _score, "at": now}
        if _asp is None:
            with st.lock:
                _hold = st.asd_speaker
            if _hold is not None and (now - _hold.get("at", 0.0)) < 2.0:
                _asp = _hold
        if _asp is not None:
            _tid = _asp.get("track_id")
            _log_pid = _asp.get("pid") or f"_track{_tid}"
            _real_name = _asp.get("name")
        else:
            _log_pid = "_offscreen"
            _real_name = None
        if _real_name in ("未知", ""):          # get_name 对无名者返回字面"未知"→当无名(shadow 学习⑤)
            _real_name = None
        _tspk_real = (_log_pid not in ("_unknown", "_offscreen")
                      and not _log_pid.startswith("_track"))
        with st.lock:
            st.turn_speaker_pid = _log_pid if _tspk_real else None
            st.turn_speaker_name = _real_name if _tspk_real else None
            st.turn_speaker_at = now
            st.last_interaction_at = now   # 级联:用户开口即刷新互动时间,防空闲计时器误判冷场断连(bug-075)
        _label = _real_name or "访客"           # 已命名用名;未命名/画外用"访客"(不空缺、绝不套别人名)
        log(f"📝 [ASR]听到:「{transcript}」→ 🗣 归属:{_label}({_log_pid})")
        # conversation_log(consolidation/兜底抽取)+ dashboard 显示,键 = 归属 pid
        if not st.no_memory:
            with st.lock:
                st.conversation_log.setdefault(_log_pid, []).append(("user", transcript))
        with st.lock:
            st.display_transcript_seq += 1
            st.display_transcript.append({"seq": st.display_transcript_seq, "ts": time.strftime("%H:%M:%S"),
                                          "role": "user", "text": transcript, "pid": _log_pid, "name": _label})
            if len(st.display_transcript) > 100:
                st.display_transcript = st.display_transcript[-80:]
        c = self.conv
        if c is None:
            return
        # ① 带说话人标签的 user item 入历史(根治多人区分——update_session 做不到的"重标历史")
        try:
            c.create_item({"type": "message", "role": "user",
                           "content": [{"type": "input_text", "text": f"「{_label}」:{transcript}"}]})
        except Exception as e:
            log(f"⚠ [ASR]create_item(user) 失败:{type(e).__name__}: {e}")
            return
        # ② 易变记忆/消歧:复用 inject_context(system item·简洁,不进持久历史)
        if self.dialog is not None and self.memory_mgr is not None:
            _cur_pid = _log_pid if _tspk_real else None
            self.dialog.inject_context(_cur_pid, _real_name)
        # ③ 手动 create_response。级联智能接管(降延迟):
        #    - 无在途回复 → 直接答;
        #    - 在途回复"刚起"(<TAKEOVER_MIN_AGE)且非卡死 → **排队**:不 cancel(省 3-4s cancel 开销),
        #      待其 response.done 时再答(user item/注入已就位);
        #    - 在途回复"已播一会儿"(真长回复/真打断)或排队卡死太久 → **接管**:cancel + 硬清 in_flight 立即答。
        #    (不能沿用 S2S 的"in_flight>0 就跳过":会因招呼/首轮双回复卡 in_flight 致后续全被跳过。)
        _TAKEOVER_MIN_AGE = 1.5
        with st.lock:
            # 忙判据用 active_resp_id(最新 response 未 done)而非 in_flight 计数器(工具调用致其永久漏+1)
            _busy = (st.active_resp_id is not None and not st.active_resp_done
                     and (now - st.resp_created_at) < 20.0)   # 20s 兜底:再久也不认它还在忙(防卡)
            _age = (now - st.resp_created_at) if st.resp_created_at > 0 else 999.0
            _stale_pending = st.pending_resp_at > 0 and (now - st.pending_resp_at) > 3.0
        _fire = False
        if not _busy:
            _fire = True
        elif _age >= _TAKEOVER_MIN_AGE or _stale_pending:
            try:
                c.cancel_response()
            except Exception as e:
                log(f"⚠ [ASR]cancel_response 失败:{type(e).__name__}: {e}")
            with st.lock:
                st.in_flight = 0
                st.active_resp_done = True
                st.pending_resp_at = 0.0
            log("↩ [ASR]接管:取消在途旧回复,答新一轮")
            _fire = True
        else:
            with st.lock:
                st.pending_resp_at = now
            log("⏳ [ASR]排队:旧回复刚起,待其完成再答(免频繁打断)")
        if _fire:
            try:
                c.create_response()
            except Exception as e:
                log(f"⚠ [ASR]create_response 失败:{type(e).__name__}: {e}")


class RealtimeDialog:
    """Qwen-Omni-Realtime 对话协议管理器 — 封装 session 生命周期。"""

    def __init__(self, st: State, play_q, motion_q, snap_q, mini: ReachyMini,
                 oai_client, memory_mgr, owner_mgr, id_recognizer,
                 instructions: str, tools: list, no_memory: bool = False,
                 face_pipeline=None, asd_engine=None):
        self.callback = ChatCallback(st, play_q, motion_q, snap_q, mini,
                                     memory_mgr, owner_mgr, id_recognizer,
                                     face_pipeline=face_pipeline, asd_engine=asd_engine)
        self.callback.dialog = self
        self.st = st
        self.oai = oai_client
        self.memory_mgr = memory_mgr
        self.instructions = instructions
        self.tools = tools
        self.no_memory = no_memory
        self.conv = None
        self._last_inject_fail = 0.0
        self._ctx_item_id = None     # 上下文条目(create_item)的 id,发新的前删旧的(保洁)
        self._ctx_seq = 0
        self._ctx_use_item = True    # create_item 路是否可用;Qwen 拒 system item 时回退 update_session
        self._last_connect_at = 0.0
        self._min_connect_gap = 1.0
        # 级联(CASCADE=1):Omni 文本入 + 中和音频(关转写/关服务端VAD),回复由本地 ASR 轮驱动。
        # 默认关 = 原 S2S。见 docs/ASR_CASCADE_REDESIGN.md 阶段2。
        self.cascade = os.environ.get("CASCADE") == "1"

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
                c.update_session(
                    output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                    voice=VOICE,
                    input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                    output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                    # 级联:关 Omni 转写(音频只当视频锚点,真转写走本地 ASR)+ 关服务端 VAD(永不据音频自动回复)
                    enable_input_audio_transcription=(not self.cascade),
                    enable_turn_detection=(not self.cascade),
                    turn_detection_type="semantic_vad",
                    turn_detection_param={"create_response": False},   # (非级联)VAD 只断句、不自动回复:回复由我方注入后手动 create_response(保时序)
                    instructions=self.instructions,
                    tools=self.tools,
                )
                if self.cascade:
                    log("🔀 CASCADE=1:Omni 文本入模式(转写关/服务端VAD关;回复由本地 ASR 轮驱动)")
            except Exception as e:
                holder["err"] = e
        threading.Thread(target=_w, daemon=True).start()
        if st.session_updated.wait(timeout):
            self.callback.conv = c
            self.conv = c
            with st.lock:
                st.in_flight = 0
                st.active_resp_id = None      # 新会话:清"忙"状态(级联忙判据)
                st.active_resp_done = True
                st.pending_resp_at = 0.0
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

    def inject_context(self, cur_pid, cur_name) -> bool:
        """把「在场 roster + 带说话人标签的近史 + 规则」作为 system 消息注入对话流(create_item·通道B)。
        base 人设在 connect 已设,这里只补易变上下文——不改 session.instructions,绕开 update_session 重置副作用。
        成功 True;create_item 不可用(Qwen 拒 system item)返回 False → 调用方回退 update_session。"""
        if not self._ctx_use_item:
            return False
        c = self.conv
        if c is None:
            return False
        st = self.st
        with st.lock:
            _present = st.present_count
        # 简洁:只点明"当前正在跟你说话的是谁",不堆整段历史(整段历史反而嘈杂、稀释信号 → 模型照旧串)
        _multi = "画面里还有别人,别把这位和其他人搞混。" if _present > 1 else ""
        if cur_name:                                   # 当前说话人已命名 → 带上TA自己的记忆
            _facts = self.memory_mgr.get_facts(cur_pid) if (cur_pid and self.memory_mgr) else []
            _fs = ("你记得TA:" + "；".join(_facts[:4]) + "。") if _facts else ""
            text = f"【当前说话人】现在正在跟你说话的是「{cur_name}」。{_fs}{_multi}"
        elif _present > 0:                              # 画面有人但当前说话人未命名/未归属
            text = ("【当前说话人】现在跟你说话的是一位你还没记住名字的人。"
                    "若TA问「我是谁/我叫什么/我喜欢什么」,如实说你还不确定TA是谁、还没记住TA,"
                    "绝不要拿别人的名字或记忆来答。" + _multi)
        else:                                          # 画面真没人
            text = ("【当前说话人】对方不在画面里、你看不到TA。若被问身份,如实说看不到、不确定是谁,"
                    "别套用其他人的名字或记忆。")
        if self.cascade:                               # 文本入模式模型少主动做动作 → 每轮 system 条目补一句提示
            text += "。回答时边说边自然配合动作工具(点头/歪头/摆天线/摇头等),别只说话不动。"
        # 保洁:发新条目前先删上一条(客户端指定 id 便于下次删)
        if self._ctx_item_id:
            try:
                c.send_raw(json.dumps({"type": "conversation.item.delete",
                                       "item_id": self._ctx_item_id}))
            except Exception as _e:
                log(f"⚠ 删旧上下文条目失败:{type(_e).__name__}")
        self._ctx_seq += 1
        _iid = f"ctx{self._ctx_seq}"
        try:
            c.create_item({"id": _iid, "type": "message", "role": "system",
                           "content": [{"type": "input_text", "text": text}]})
        except Exception as e:
            log(f"⚠ create_item 注入失败 → 本会话改回退 update_session:{type(e).__name__}: {e}")
            self._ctx_use_item = False
            return False
        self._ctx_item_id = _iid
        with st.lock:
            st.identity_injected = True
            st.identity_injected_pid = cur_pid if cur_pid else "_ctx"
            st.dbg_memory_prompt = text
            st.dbg_session_instructions = self.instructions
        log(f"📌 上下文注入对话(create_item·简洁) present={_present} 当前={_idtag(cur_pid, cur_name)}")
        return True

    def update_memory(self, pid: str, pname: str | None) -> bool:
        """用 update_session 将记忆嵌入 session instructions(create_item 不可用时的回退路)。"""
        if time.monotonic() - self._last_inject_fail < 2.0:
            return False
        st = self.st
        mem_prompt = self.memory_mgr.get_prompt(pid, person_name=pname) if self.memory_mgr else None
        # 已注册但未命名的说话人:绝不给占位名,明确告诉模型「还不知道名字、别编」(治 ?T 被读成名字)
        if not pname and not (self.memory_mgr and self.memory_mgr.get_name(pid)):
            _noname = ("【当前说话人】你还不知道对方叫什么名字,绝不要编名字或套用别人的名字;"
                       "若想知道可以礼貌地问对方怎么称呼。")
            mem_prompt = (mem_prompt + "\n" + _noname) if mem_prompt else _noname
        # B:多人在场时叮嘱模型别假设当前说话人就是这个名字(治"Unknown 被答成吴继豪")
        if pname and st.present_count > 1:
            _multi = ("\n注意:画面里可能不止一个人。以上名字「" + pname + "」和记忆只属于这一个人;"
                      "只有你确认正在跟「" + pname + "」说话时才用这个名字,不确定现在是谁在说话就别直接喊名字,可先礼貌确认。")
            mem_prompt = (mem_prompt + _multi) if mem_prompt else _multi
        new_instr = self.instructions + ("\n\n" + mem_prompt if mem_prompt else "")
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
                turn_detection_param={"create_response": False},   # 与 open_session 一致:VAD 只断句不自动回复
                instructions=new_instr,
                tools=self.tools,
            )
            log(f"🧠 记忆已注入(回退update_session) {_idtag(pid, pname)} present={st.present_count}")
            with st.lock:
                st.identity_injected = True
                st.identity_injected_pid = pid
                st.video_suppress_until = time.monotonic() + 1.2   # bug-069:注入后抑制视频,让新音频先到
                st.dbg_memory_prompt = mem_prompt
                st.dbg_session_instructions = new_instr
                _buffered = list(st.audio_gate_buffer)
                st.audio_gate_buffer.clear()
                st.audio_gate_closed = False
            if _buffered:
                for chunk in _buffered:
                    try:
                        c.append_audio(chunk)
                    except Exception as _e:
                        log(f"⚠ flush 送音频失败,中止剩余:{type(_e).__name__}")
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
                turn_detection_param={"create_response": False},   # 与 open_session 一致:VAD 只断句不自动回复
                instructions=new_instr,
                tools=self.tools,
            )
            log("🫥 画外/未识别说话人 → 注入中性上下文(不认识对方,不套用他人身份)")
            with st.lock:
                st.identity_injected = True
                st.identity_injected_pid = "_neutral"
                st.video_suppress_until = time.monotonic() + 1.2   # bug-069:注入后抑制视频,让新音频先到
                st.dbg_memory_prompt = note
                st.dbg_session_instructions = new_instr
            return True
        except Exception as e:
            self._last_inject_fail = time.monotonic()
            log(f"⚠ 中性 update_session 失败:{e}")
            return False

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
                '"facts": ["中文短句", ...]}\n'
                "没有任何可记的信息则 facts 为空数组、name 为 null;"
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
            new_facts = [f.strip() for f in (result.get("facts") or [])
                         if isinstance(f, str) and f.strip()]
            saved = 0
            for f in new_facts:
                r = self.memory_mgr.save_fact(pid, f)   # 内置去重
                if "已记住" in r or "已更新" in r:
                    saved += 1
            _cb = self.callback
            if new_name:
                # 工具审视只做「首次命名」(allow_rename=False);改名只走模型直调路径。
                # guard 统一过门(合法/来自转写/不覆盖已命名)。
                if try_name_identity(
                        memory_mgr=self.memory_mgr, id_recognizer=_cb.id_recognizer,
                        face_pipeline=_cb.face_pipeline, owner_mgr=_cb.owner_mgr, st=self.st,
                        pid=pid, new_name=new_name, transcript=current_text,
                        log_fn=log, allow_rename=False):
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
            facts_str = json.dumps(current_facts, ensure_ascii=False) if current_facts else "[]"
            prompt = (
                "你是记忆管理助手。仔细阅读对话，提取关于用户的所有个人信息。\n\n"
                f"当前用户名字：{current_name or '未知'}\n"
                f"已有记忆：{facts_str}\n\n"
                f"对话内容：\n{text[-4000:]}\n\n"
                "任务：\n"
                "1. facts = 合并已有记忆 + 对话中发现的新信息。每条是中文短句。\n"
                "   - 重点提取：爱好、喜好、职业、年龄、习惯、观点、提到的人/事物\n"
                "   - 用户说'我喜欢X/我爱X/我常X' → 加入 facts\n"
                "   - 如果新信息与旧 fact 矛盾，保留新的、去掉旧的\n"
                "   - 不要把名字放进 facts（名字在独立字段管理）\n"
                "   - 不要写 '名字是XXX' '叫XXX' 这样的 fact\n"
                "2. episode = 这次对话的结构化事件\n"
                "   - topic: 具体说聊了什么，不要太笼统\n"
                "   - highlights: 关键信息点（每条是完整短句）\n\n"
                "只输出JSON：\n"
                '{"name":"用户名字(对话中提到则更新,否则保留原名,未知则null)",'
                '"facts":["短句1","短句2"],'
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
            new_name = result.get("name")
            episode = result.get("episode")
            self.memory_mgr.consolidate_facts(pid, new_facts, new_name)
            if episode:
                self.memory_mgr.save_episode(pid, episode)
            log(f"📝 记忆 consolidation 完成 ({pid[:12]}): {len(new_facts)} facts, episode={episode.get('topic', '')[:30] if episode else 'none'}")
            with self.st.lock:
                if self.st.current_person_id == pid:
                    self.st.identity_injected = False
        except Exception as e:
            log(f"⚠ 记忆 consolidation 失败({pid[:12]}):{e}")
