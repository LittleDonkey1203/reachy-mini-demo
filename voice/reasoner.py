# -*- coding: utf-8 -*-
"""异步对话策略 Reasoner(Talker-Reasoner 架构,REASONER-01)。

用户聊天时,一个独立 worker 线程在后台用大模型(默认 qwen-max)思考「接下来几轮
用户可能对什么感兴趣 / 怎么聊更开心 / 用什么引子勾话题」,产出简短策略写入
st.reasoner_hint;Talker(Omni)下一轮在 response.instructions(resp_directive)
里带上该策略。

纯文本侧模块:只读 st.display_transcript + memory,只写 st.reasoner_hint,
【绝不】碰 st.state / behavior / motion / head_control。

三红线:
1. 主链路零阻塞:notify() 非阻塞;LLM 调用全在 worker 线程;Reasoner 慢/挂
   不影响对话行为(hint 缺失时 resp_directive 就是现状)。
2. 策略注入只走 resp_directive()(见 hint_gate_ok);本模块不做任何注入。
3. 不碰运动侧。
"""

from __future__ import annotations

import json
import queue
import threading
import time

from voice.config import (
    REASONER_MODEL, REASONER_BASE_URL, REASONER_API_KEY,
    REASONER_DEBOUNCE_S, REASONER_HINT_TTL_S, REASONER_MAX_STALE_TURNS,
    REASONER_TIMEOUT_S, REASONER_PROMPT,
)
from voice.state import log

HINT_LIMIT = 180   # 压缩后策略段字数上限(红线2:≤180 字)


# ────────────────────────── 纯函数(离线可测)──────────────────────────

def parse_reasoner_json(raw):
    """宽容解析 Reasoner 的 JSON(剥 ``` 围栏 / 取首尾大括号);失败返回 None。"""
    if not raw:
        return None
    s = raw.strip()
    if "```" in s:
        s = s.replace("```json", "```")
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(s[i:j + 1])
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def compress_hint(data, limit=HINT_LIMIT):
    """把 Reasoner JSON 压成一段 ≤limit 字的中文策略文本;无有效内容返回 ''。"""
    if not isinstance(data, dict):
        return ""
    parts = []
    topics = data.get("topics")
    if isinstance(topics, list):
        for t in topics[:2]:
            if not isinstance(t, dict):
                continue
            tw = str(t.get("t", "")).strip()
            hook = str(t.get("hook", "")).strip()
            if tw and hook:
                parts.append(f"可聊「{tw}」:{hook}")
            elif hook:
                parts.append(hook)
            elif tw:
                parts.append(f"可聊「{tw}」")
    style = str(data.get("style", "")).strip()
    if style:
        parts.append(f"风格:{style}")
    callback = str(data.get("callback", "")).strip()
    if callback:
        parts.append(f"可回扣:{callback}")
    avoid = str(data.get("avoid", "")).strip()
    if avoid:
        parts.append(f"回避:{avoid}")
    return "；".join(parts)[:limit]


def hint_gate_ok(hint, cur_pid, present, now, cur_seq,
                 ttl=REASONER_HINT_TTL_S, max_stale=REASONER_MAX_STALE_TURNS):
    """三道门:hint 是否可注入当前这轮 resp_directive(供 Batch 1 的 resp_directive 调用)。
    ① 归属门:hint.pid == cur_pid,或 (hint.pid is None 且 present)
    ② 新鲜门:now - hint.ts < ttl
    ③ 轮数门:cur_seq - hint.seq <= max_stale
    """
    if not hint or not hint.get("text"):
        return False
    hp = hint.get("pid")
    if not (hp == cur_pid or (hp is None and present)):
        return False
    if now - hint.get("ts", 0.0) >= ttl:
        return False
    if cur_seq - hint.get("seq", 0) > max_stale:
        return False
    return True


def _default_oai_client_factory():
    """默认客户端:DashScope compatible-mode 的 OpenAI 客户端。
    懒导入 openai —— 便于离线测试(测试用 mock factory,不触发本函数、不装 openai 也能跑)。"""
    from openai import OpenAI
    return OpenAI(api_key=REASONER_API_KEY, base_url=REASONER_BASE_URL)


# ────────────────────────── Reasoner ──────────────────────────

class ConversationReasoner:
    """异步对话策略 Reasoner。用法:construct → start();每轮转写完成后 notify(seq, pid)。"""

    def __init__(self, st, memory_mgr, oai_client_factory=None):
        self.st = st
        self.memory_mgr = memory_mgr
        self._factory = oai_client_factory or _default_oai_client_factory
        self._client = None                       # 懒建(首次调用时)
        self._q = queue.Queue(maxsize=1)          # latest-only:满则丢旧放新
        self._stop = threading.Event()
        self._last_done_at = 0.0
        self._worker = None
        self._started = False
        self._warned_no_start = False
        # 统计
        self.n_calls = 0
        self.n_ok = 0
        self.n_fail = 0
        self.n_dropped = 0
        self.inject_hits = 0        # resp_directive 实际注入策略的次数(批次2;在 _maybe_append_strategy 自增)
        self._total_ms = 0.0

    # ── 生命周期 ──
    def start(self):
        self._started = True
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, daemon=True, name="reasoner")
            self._worker.start()

    def stop(self):
        """优雅退出(~1s 内返回;worker 是 daemon,即便卡在 LLM 调用也不阻塞进程退出)。"""
        self._stop.set()
        try:
            self._q.put_nowait((None, None))      # 唤醒 worker
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=1.0)

    # ── 外部接口(主链路调用,必须非阻塞)──
    def notify(self, seq, pid):
        """非阻塞。latest-only:队满则丢旧放新(红线1:主链路零阻塞)。"""
        if not self._started and not self._warned_no_start:
            self._warned_no_start = True
            log("⚠ Reasoner.notify() 被调用但 worker 未 start() —— 策略将静默失效,检查接线是否漏调 start()。")
        task = (seq, pid)
        try:
            self._q.put_nowait(task)
        except queue.Full:
            try:
                self._q.get_nowait()              # 丢旧
                self.n_dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(task)
            except queue.Full:
                pass

    def invalidate(self):
        """换人 / 会话重启:清空当前 hint。"""
        with self.st.lock:
            self.st.reasoner_hint = None

    def stats(self):
        avg_s = (self._total_ms / self.n_calls / 1000.0) if self.n_calls else 0.0
        rate = (self.inject_hits / self.n_ok) if self.n_ok else 0.0
        return {"calls": self.n_calls, "ok": self.n_ok, "fail": self.n_fail,
                "dropped": self.n_dropped, "avg_latency_s": round(avg_s, 2),
                "inject_hits": self.inject_hits, "inject_rate": round(rate, 2)}

    # ── worker ──
    def _run(self):
        while not self._stop.is_set():
            try:
                seq, pid = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._stop.is_set() or seq is None:
                continue
            try:
                self._process(seq, pid)
            except Exception as e:
                self.n_fail += 1
                log(f"⚠ Reasoner 处理异常:{type(e).__name__}: {e}")

    def _process(self, seq, pid):
        # 去抖:距上次完成 < REASONER_DEBOUNCE_S 则等(等待期间收到 stop 立即退出)
        wait = REASONER_DEBOUNCE_S - (time.monotonic() - self._last_done_at)
        if wait > 0 and self._stop.wait(wait):
            return
        # 去抖结束后 drain 队列取最新快照:绝不处理 sleep 前 pop 的旧任务(latest-only 真语义)
        while True:
            try:
                nseq, npid = self._q.get_nowait()
            except queue.Empty:
                break
            if nseq is None:                      # stop 哨兵
                return
            seq, pid = nseq, npid

        # ── ① 锁内快照拷贝(持 st.lock 只做拷贝:role/name/text 取值拷成 tuple,不留共享引用)──
        with self.st.lock:
            recent = [(d.get("role"), d.get("name"), d.get("text"))
                      for d in self.st.display_transcript
                      if d.get("role") in ("user", "assistant")][-16:]
            prev = self.st.reasoner_hint.get("text") if self.st.reasoner_hint else None

        # ── 锁外:组 prompt + 读 memory + 调 LLM(全程不持 st.lock;memory 走其自有 RLock)──
        ctx = self._build_context(pid, recent, prev)
        raw = self._call_llm(ctx)                 # 可能抛异常(超时/网络)→ 由 _run 兜底
        self._last_done_at = time.monotonic()
        data = parse_reasoner_json(raw)
        if data is None:
            self.n_fail += 1
            log(f"⚠ Reasoner JSON 解析失败,丢弃(raw 前60:{(raw or '')[:60]})")
            return
        text = compress_hint(data)
        if not text:
            self.n_fail += 1
            return
        self.n_ok += 1

        # ── ② 锁内写回(持 st.lock 只做写回)──
        with self.st.lock:
            self.st.reasoner_hint = {"text": text, "pid": pid, "seq": seq,
                                     "ts": time.monotonic(), "raw": data}
        log(f"🧠 Reasoner 生成策略(seq={seq}, pid={pid and pid[:8]}): {text}")

    def _build_context(self, pid, recent, prev):
        """组 prompt(锁外调用,全程不持 st.lock)。recent=[(role,name,text),...] 已在锁内拷好;
        memory_mgr 走 MemoryManager 自己的 RLock,与 st.lock 无关。"""
        lines = [f"{'小艺' if role == 'assistant' else (name or '用户')}: {text or ''}"
                 for (role, name, text) in recent]
        conv = "\n".join(lines) if lines else "(暂无对话)"
        facts_s, eps_s = "", ""
        if pid and self.memory_mgr is not None:
            try:
                facts = self.memory_mgr.get_facts(pid) or {}
                facts_s = "；".join(f"{k}:{v}" for k, v in list(facts.items())[:8])
                eps = (self.memory_mgr.load_memory(pid).get("episodes") or [])[-2:]
                eps_s = "；".join(f"{e.get('topic', '')}({e.get('mood', '')})" for e in eps)
            except Exception:
                pass
        return (f"最近对话:\n{conv}\n\n"
                f"用户画像·记忆事实:{facts_s or '(暂无)'}\n"
                f"用户画像·最近话题:{eps_s or '(暂无)'}\n"
                f"上一份策略(供连续性参考):{prev or '(无)'}\n")

    def _call_llm(self, ctx):
        self.n_calls += 1
        if self._client is None:
            self._client = self._factory()
        t0 = time.monotonic()                     # 于 create 前取
        resp = self._client.chat.completions.create(
            model=REASONER_MODEL,
            messages=[{"role": "system", "content": REASONER_PROMPT},
                      {"role": "user", "content": ctx}],
            temperature=0.7,
            timeout=REASONER_TIMEOUT_S,
        )
        _elapsed_ms = (time.monotonic() - t0) * 1000.0
        with self.st.lock:                        # 锁内累计(create 已返回,不违反"LLM 调用在锁外")
            self._total_ms += _elapsed_ms
        return (resp.choices[0].message.content or "").strip()
