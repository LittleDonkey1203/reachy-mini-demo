# -*- coding: utf-8 -*-
"""O-01a-1 编排版:语音对话 + 动作工具(function calling),固定时序测试。

在 D-01b(打断+抖动缓冲)基础上新增,D-01 链路不动:

1. session.update 里声明 8 个动作工具(扁平格式,经 SDK tools= 传入)。
2. 收到 response.function_call_arguments.done → 动作任务入队,
   【独立动作线程】串行执行(goto_target 平滑插值,标定安全幅度),
   绝不阻塞音频收包/播放线程 → 边说边动。
3. 动作完成 → conversation.item.create 回 function_call_output;
   仅当"发起调用的那个响应没出过音频"才补 response.create
   (避免双重说话;纯动作响应后模型能继续开口)。
4. barge-in 时只停音频,在执行的动作让它做完(动作都很短)。

动作参数复用 CALIBRATION.md §2 标定结论:yaw+=左 pitch+=下,
头部 ±10~12°、天线 ±0.5rad,automatic_body_yaw=False,全程 body_yaw=0。
"""

import os

_no_proxy = "localhost,127.0.0.1,::1,.aliyuncs.com,aliyuncs.com"
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

import base64
import json
import queue
import sys
import threading
import time

import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation as R

import dashscope
from dashscope.audio.qwen_omni import (
    AudioFormat,
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)
from reachy_mini import ReachyMini

MODEL = "qwen3.5-omni-plus-realtime"
VOICE = "Ethan"
INSTRUCTIONS = (
    "你是桌面机器人 Reachy Mini,有真实的身体(头、天线)。"
    "用简体中文、口语化、简短地回答。"
    "回答时自然地配合动作工具表达身体语言:打招呼/同意时点头,否定时摇头,"
    "开心/兴奋/被夸时摆天线,好奇/疑惑时歪头。可以边说边做动作。"
)

OUT_SR = 24000
PLAY_SR = 16000
JITTER_S = 0.30
JITTER_WALL_S = 0.50
REC_WINDOW_S = 75.0      # 3 个测试回合
RESPONSE_TIMEOUT_S = 20.0

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[t+{time.monotonic() - T0:7.2f}s] {msg}", flush=True)


# ───────────────────────── 动作库(标定参数,均为短动作)─────────────────────────
INIT_HEAD_POSE = np.eye(4)
INIT_ANTENNAS = [-0.1745, 0.1745]


def head_pose(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return T


def act_nod(m: ReachyMini) -> None:           # 点头:pitch 下/上 ×2
    for _ in range(2):
        m.goto_target(head_pose(pitch_deg=+10), duration=0.35, body_yaw=0.0)
        m.goto_target(head_pose(pitch_deg=-6), duration=0.35, body_yaw=0.0)
    m.goto_target(INIT_HEAD_POSE, duration=0.35, body_yaw=0.0)


def act_shake(m: ReachyMini) -> None:         # 摇头:yaw 左/右 ×2
    for _ in range(2):
        m.goto_target(head_pose(yaw_deg=+10), duration=0.35, body_yaw=0.0)
        m.goto_target(head_pose(yaw_deg=-10), duration=0.35, body_yaw=0.0)
    m.goto_target(INIT_HEAD_POSE, duration=0.35, body_yaw=0.0)


def _look(m: ReachyMini, **kw) -> None:       # 看向某方向,停 0.8s 回中
    m.goto_target(head_pose(**kw), duration=0.6, body_yaw=0.0)
    time.sleep(0.8)
    m.goto_target(INIT_HEAD_POSE, duration=0.6, body_yaw=0.0)


def act_wiggle(m: ReachyMini) -> None:        # 天线摆动:左右交替 ×2
    for _ in range(2):
        m.goto_target(antennas=[+0.5, -0.5], duration=0.3, body_yaw=0.0)
        m.goto_target(antennas=[-0.5, +0.5], duration=0.3, body_yaw=0.0)
    m.goto_target(antennas=INIT_ANTENNAS, duration=0.35, body_yaw=0.0)


def act_tilt(m: ReachyMini) -> None:          # 歪头(roll,幅度同安全标定)
    m.goto_target(head_pose(roll_deg=12), duration=0.5, body_yaw=0.0)
    time.sleep(0.8)
    m.goto_target(INIT_HEAD_POSE, duration=0.5, body_yaw=0.0)


ACTIONS = {
    "nod": act_nod,
    "shake_head": act_shake,
    "look_left": lambda m: _look(m, yaw_deg=+12),
    "look_right": lambda m: _look(m, yaw_deg=-12),
    "look_up": lambda m: _look(m, pitch_deg=-12),
    "look_down": lambda m: _look(m, pitch_deg=+12),
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
]


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session_updated = threading.Event()
        # 播放 / 打断(同 D-01b)
        self.play_gen = 0
        self.drop_audio = False
        self.in_flight = 0
        self.playback_end_estimate = 0.0
        # function calling 协调
        self.resp_audio_count = 0      # 当前响应已出的 audio.delta 数(response.created 时清零)
        self.fc_pending = 0            # 已派发未回 output 的动作数
        self.fc_seen_this_resp = False
        self.fc_response_done = False
        self.fc_needs_rc = False       # 发起调用的响应无音频 → 动作做完后补 response.create
        self.fc_rc_sent = False
        self.fc_gen = 0
        # 留痕
        self.input_transcripts: list[str] = []
        self.reply_parts: list[str] = []
        self.fc_records: list[dict] = []
        self.motion_records: list[dict] = []
        self.barge_ins: list[dict] = []
        self.errors: list[dict] = []
        self.audio_delta_count = 0
        self.first_audio_delay_ms = None


class ChatCallback(OmniRealtimeCallback):
    def __init__(self, st: State, play_q: "queue.Queue", motion_q: "queue.Queue", mini: ReachyMini):
        self.st = st
        self.play_q = play_q
        self.motion_q = motion_q
        self.mini = mini
        self.conv: OmniRealtimeConversation | None = None

    def on_open(self) -> None:
        log("✅ WebSocket 已连接")

    def on_close(self, code, msg) -> None:
        log(f"🔌 连接关闭 code={code} msg={msg}")

    def _do_barge_in(self, in_flight: bool) -> None:
        st = self.st
        with st.lock:
            st.play_gen += 1
            st.drop_audio = True
            residual = max(0.0, st.playback_end_estimate - time.monotonic())
            st.playback_end_estimate = time.monotonic()
            st.barge_ins.append({"t": round(time.monotonic() - T0, 2), "residual_s": round(residual, 2)})
        while True:
            try:
                self.play_q.get_nowait()
            except queue.Empty:
                break
        try:
            self.mini.media.audio.clear_player()
        except Exception as e:
            log(f"⚠ clear_player 失败:{e}")
        if in_flight and self.conv is not None:
            self.conv.cancel_response()
        log(f"⛔ BARGE-IN(残余 {residual:.2f}s 已清,动作不中断)")

    def _maybe_continue_locked(self) -> bool:
        """锁内判断:动作全部回完 + 响应已结束 + 需要补说话 → 该发 response.create 了。"""
        st = self.st
        if (
            st.fc_pending == 0
            and st.fc_response_done
            and st.fc_needs_rc
            and not st.fc_rc_sent
            and st.fc_gen == st.play_gen  # 期间没被打断
        ):
            st.fc_rc_sent = True
            return True
        return False

    def on_event(self, event) -> None:
        st = self.st
        try:
            etype = event.get("type", "")
            now = time.monotonic()
            if etype == "session.created":
                log(f"✅ session.created id={event['session']['id']}")
            elif etype == "session.updated":
                log("✅ session.updated(8 个动作工具已注册)")
                st.session_updated.set()
            elif etype == "input_audio_buffer.speech_started":
                with st.lock:
                    playing = (now < st.playback_end_estimate) or (not self.play_q.empty())
                    in_flight = st.in_flight > 0
                log(f"🎤 speech_started(播放中={playing},生成中={in_flight})")
                if playing or in_flight:
                    self._do_barge_in(in_flight)
            elif etype == "input_audio_buffer.speech_stopped":
                log("🤫 speech_stopped")
            elif etype == "conversation.item.input_audio_transcription.completed":
                t = (event.get("transcript") or "").strip()
                with st.lock:
                    st.input_transcripts.append(t)
                log(f"📝 输入转写:「{t}」")
            elif etype == "response.created":
                with st.lock:
                    st.in_flight += 1
                    st.drop_audio = False
                    st.resp_audio_count = 0
                    if st.fc_pending == 0:  # 新一轮,清上轮 fc 标志
                        st.fc_seen_this_resp = False
                        st.fc_response_done = False
                        st.fc_needs_rc = False
                        st.fc_rc_sent = False
                log("💭 response.created")
            elif etype == "response.function_call_arguments.done":
                name = event.get("name", "")
                call_id = event.get("call_id", "")
                args = event.get("arguments", "")
                with st.lock:
                    st.fc_pending += 1
                    st.fc_seen_this_resp = True
                    st.fc_gen = st.play_gen
                    st.fc_records.append(
                        {"t": round(now - T0, 2), "name": name, "call_id": call_id, "args": args}
                    )
                log(f"🤖 模型调用动作: {name}(call_id={call_id})")
                self.motion_q.put({"name": name, "call_id": call_id})
            elif etype == "response.audio_transcript.delta":
                with st.lock:
                    st.reply_parts.append(event.get("delta", ""))
            elif etype == "response.audio.delta":
                with st.lock:
                    if st.drop_audio:
                        return
                    gen = st.play_gen
                    st.resp_audio_count += 1
                    st.audio_delta_count += 1
                b64 = event.get("delta") or event.get("audio") or ""
                pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                f16k = resample_poly(pcm.astype(np.float32) / 32768.0, PLAY_SR, OUT_SR).astype(np.float32)
                self.play_q.put((gen, f16k))
            elif etype == "response.done":
                fire_rc = False
                with st.lock:
                    st.in_flight = max(0, st.in_flight - 1)
                    if st.fc_seen_this_resp and not st.fc_response_done:
                        st.fc_response_done = True
                        st.fc_needs_rc = st.resp_audio_count == 0
                        fire_rc = self._maybe_continue_locked()
                        mode = "纯动作(做完后补说话)" if st.fc_needs_rc else "音频+动作并发"
                        log(f"✅ response.done(本响应含动作调用,{mode})")
                    else:
                        log("✅ response.done")
                if fire_rc and self.conv is not None:
                    self.conv.create_response()
                    log("▶ 已发 response.create(动作已先完成)")
                if self.conv is not None:
                    st.first_audio_delay_ms = self.conv.get_last_first_audio_delay()
            elif etype == "error":
                with st.lock:
                    st.errors.append(event)
                log(f"❌ error 事件:{event}")
        except Exception as e:
            log(f"❌ on_event 异常 {type(e).__name__}: {e} | 事件:{str(event)[:300]}")


# ───────────────────────── 动作线程:串行执行,完了回 output ─────────────────────────
def motion_loop(mini: ReachyMini, st: State, cb: ChatCallback, motion_q: "queue.Queue", stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            job = motion_q.get(timeout=0.1)
        except queue.Empty:
            continue
        name, call_id = job["name"], job["call_id"]
        fn = ACTIONS.get(name)
        t_start = time.monotonic()
        with st.lock:
            playing_at_start = t_start < st.playback_end_estimate
        log(f"🦾 动作开始: {name}(此刻机器人{'正在' if playing_at_start else '没在'}说话)")
        ok = True
        if fn is None:
            ok = False
            log(f"⚠ 未知动作 {name}")
        else:
            try:
                fn(mini)
            except Exception as e:
                ok = False
                log(f"⚠ 动作 {name} 执行失败:{type(e).__name__}: {e}")
        dur = time.monotonic() - t_start
        with st.lock:
            st.motion_records.append(
                {"name": name, "t_start": round(t_start - T0, 2), "dur_s": round(dur, 2),
                 "speaking_at_start": playing_at_start, "ok": ok}
            )
        log(f"✅ 动作完成: {name}({dur:.1f}s)")
        # 按协议回结果
        output = json.dumps({"success": ok, "action": name}, ensure_ascii=False)
        fire_rc = False
        try:
            cb.conv.create_item({"type": "function_call_output", "call_id": call_id, "output": output})
            with st.lock:
                st.fc_pending = max(0, st.fc_pending - 1)
                fire_rc = cb._maybe_continue_locked()
            log(f"↩ 已回 function_call_output({name})")
        except Exception as e:
            log(f"⚠ 回 function_call_output 失败:{e}")
        if fire_rc:
            try:
                cb.conv.create_response()
                log("▶ 已发 response.create(纯动作响应,让模型继续说)")
            except Exception as e:
                log(f"⚠ response.create 失败:{e}")


# ───────────────────────── 播放线程(同 D-01b)─────────────────────────
def player_loop(mini: ReachyMini, st: State, play_q: "queue.Queue", stop: threading.Event) -> None:
    def current_gen() -> int:
        with st.lock:
            return st.play_gen

    def push(chunk: np.ndarray) -> None:
        mini.media.push_audio_sample(chunk)
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


def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        log("❌ 无 DASHSCOPE_API_KEY")
        return 1
    dashscope.api_key = api_key

    print("=== O-01a-1 编排版:语音对话 + 动作工具 ===", flush=True)
    st = State()
    play_q: "queue.Queue" = queue.Queue()
    motion_q: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    log("连接 Reachy Mini(media_backend=default, automatic_body_yaw=False)…")
    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="default",
        automatic_body_yaw=False,
    ) as mini:
        try:
            mini.media.start_recording()
            mini.media.start_playing()
            log("录音/播放管线已启动;回中立位…")
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
            time.sleep(0.8)

            cb = ChatCallback(st, play_q, motion_q, mini)
            conv = OmniRealtimeConversation(model=MODEL, callback=cb)
            cb.conv = conv
            log("连接 Qwen-Omni-Realtime…")
            conv.connect()
            conv.update_session(
                output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                voice=VOICE,
                input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                enable_input_audio_transcription=True,
                enable_turn_detection=True,
                turn_detection_type="semantic_vad",
                instructions=INSTRUCTIONS,
                tools=TOOLS,  # ★ 动作工具注册(扁平格式,经 kwargs 进 session)
            )
            if not st.session_updated.wait(timeout=10):
                log("❌ 10s 内未收到 session.updated,中止")
                conv.close()
                return 1

            threading.Thread(target=player_loop, args=(mini, st, play_q, stop), daemon=True).start()
            threading.Thread(target=motion_loop, args=(mini, st, cb, motion_q, stop), daemon=True).start()

            while mini.media.get_audio_sample() is not None:
                pass

            log(f"READY_FOR_SPEECH(录音窗口 {REC_WINDOW_S:.0f}s 现在打开)")
            win_end = time.monotonic() + REC_WINDOW_S
            sent = 0
            rms_acc: list[float] = []
            rms_t = time.monotonic()
            while time.monotonic() < win_end:
                chunk = mini.media.get_audio_sample()
                if chunk is None or len(chunk) == 0:
                    time.sleep(0.01)
                    continue
                mono = chunk[:, 0]
                rms_acc.append(float(np.sqrt(np.mean(mono**2))))
                pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
                conv.append_audio(base64.b64encode(pcm16.tobytes()).decode("ascii"))
                sent += len(mono)
                if time.monotonic() - rms_t >= 5.0:  # 每 5s 报一次上行电平,排查"说话没被听见"
                    rms = float(np.mean(rms_acc)) if rms_acc else 0.0
                    log(f"🎙 近5s 上行 RMS={rms:.4f}({'有声' if rms > 0.005 else '偏弱/静音,请大声靠近'})")
                    rms_acc = []
                    rms_t = time.monotonic()
            log(f"录音窗口关闭,共上行 {sent / PLAY_SR:.1f}s 音频;补 1.5s 静音帧")
            silence = base64.b64encode(np.zeros(8000, dtype=np.int16).tobytes()).decode("ascii")
            for _ in range(3):
                conv.append_audio(silence)
                time.sleep(0.5)

            deadline = time.monotonic() + RESPONSE_TIMEOUT_S
            while time.monotonic() < deadline:
                with st.lock:
                    pending = st.in_flight > 0 or st.fc_pending > 0
                if not pending and play_q.empty() and motion_q.empty():
                    break
                time.sleep(0.2)

            with st.lock:
                tail = st.playback_end_estimate - time.monotonic()
            if tail > 0:
                log(f"等待播放余量 {tail:.1f}s…")
                time.sleep(tail + 1.0)

            stop.set()
            try:
                conv.close()
            except Exception:
                pass
            mini.media.stop_recording()
            mini.media.stop_playing()
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS, duration=1.0, body_yaw=0.0)
        finally:
            try:
                mini.set_automatic_body_yaw(True)
            except Exception:
                pass

    # ── 汇总 ──
    print("\n========== 汇总 ==========", flush=True)
    print(f"输入转写:{st.input_transcripts}", flush=True)
    print(f"回复全文:「{''.join(st.reply_parts).strip()}」", flush=True)
    print(f"模型动作调用({len(st.fc_records)} 次):", flush=True)
    for r in st.fc_records:
        print(f"  t+{r['t']}s {r['name']}", flush=True)
    print(f"动作执行记录({len(st.motion_records)} 次):", flush=True)
    for r in st.motion_records:
        concur = "【边说边动】" if r["speaking_at_start"] else ""
        print(f"  t+{r['t_start']}s {r['name']} {r['dur_s']}s ok={r['ok']} {concur}", flush=True)
    print(f"barge-in:{st.barge_ins if st.barge_ins else '无'}", flush=True)
    print(f"audio.delta 总数:{st.audio_delta_count}|error:{st.errors if st.errors else '无'}", flush=True)
    ok = bool(st.fc_records) and all(r["ok"] for r in st.motion_records) and not st.errors
    print(f"=== {'动作工具闭环成功' if ok else '未达验收,看上方记录'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
