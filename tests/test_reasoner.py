# -*- coding: utf-8 -*-
"""REASONER-01 批次0 离线单测:JSON 解析 / 三道门 / latest-only 丢旧。

mock LLM,不打真网。可 `python -m pytest tests/test_reasoner.py` 或直接
`python tests/test_reasoner.py`(内置兜底 runner)。
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.reasoner import (
    parse_reasoner_json, compress_hint, hint_gate_ok, ConversationReasoner,
)


# ────────── ① JSON 解析(正常 / 带 ``` 围栏 / 畸形)──────────
def test_parse_normal():
    d = parse_reasoner_json('{"topics":[{"t":"露营","hook":"去露营吗?"}],"style":"短"}')
    assert d and d["style"] == "短" and d["topics"][0]["t"] == "露营"


def test_parse_fenced():
    d = parse_reasoner_json('```json\n{"style":"短","avoid":""}\n```')
    assert d and d["style"] == "短"


def test_parse_with_prose():
    d = parse_reasoner_json('好的,结果:{"style":"短"} 以上。')
    assert d and d["style"] == "短"


def test_parse_malformed():
    assert parse_reasoner_json("not json at all") is None
    assert parse_reasoner_json('{"broken": ') is None
    assert parse_reasoner_json("") is None
    assert parse_reasoner_json(None) is None
    assert parse_reasoner_json("[1,2,3]") is None   # 非 dict → None


# ────────── compress_hint ──────────
def test_compress_basic():
    data = {"topics": [{"t": "露营", "hook": "最近去露营了吗?"}, {"t": "美食", "hook": "晚上吃啥?"}],
            "style": "1-2短句+反问", "callback": "喜欢西瓜", "avoid": "别提工作"}
    txt = compress_hint(data)
    assert "露营" in txt and "反问" in txt and "西瓜" in txt and len(txt) <= 180


def test_compress_topics_capped_at_2():
    txt = compress_hint({"topics": [{"t": str(i), "hook": "h"} for i in range(5)]})
    assert txt.count("可聊") == 2


def test_compress_limit_hard():
    long = {"topics": [{"t": "x" * 100, "hook": "y" * 100}], "style": "z" * 100}
    assert len(compress_hint(long)) <= 180


def test_compress_empty():
    assert compress_hint({}) == ""
    assert compress_hint("garbage") == ""


# ────────── ② 三道门(过期 / 换人 / 落后轮数)──────────
def _hint(pid="p1", seq=10, ts=None):
    return {"text": "策略", "pid": pid, "seq": seq,
            "ts": ts if ts is not None else time.monotonic()}


def test_gate_pass():
    now = time.monotonic()
    assert hint_gate_ok(_hint("p1", 10, now), "p1", True, now, 12) is True


def test_gate_person_switch():
    now = time.monotonic()
    assert hint_gate_ok(_hint("p1", 10, now), "p2", True, now, 11) is False


def test_gate_expired():
    now = time.monotonic()
    assert hint_gate_ok(_hint("p1", 10, now - 100.0), "p1", True, now, 11) is False  # TTL 90


def test_gate_stale_turns():
    now = time.monotonic()
    # 显式传 max_stale=6 测边界逻辑本身(不依赖 config 默认)
    assert hint_gate_ok(_hint("p1", 10, now), "p1", True, now, 17, max_stale=6) is False  # 7 > 6
    assert hint_gate_ok(_hint("p1", 10, now), "p1", True, now, 16, max_stale=6) is True   # 6 ≤ 6
    # 默认(config 现为 10)的边界
    assert hint_gate_ok(_hint("p1", 10, now), "p1", True, now, 20) is True    # 10 ≤ 10
    assert hint_gate_ok(_hint("p1", 10, now), "p1", True, now, 21) is False   # 11 > 10


def test_gate_pid_none_present():
    now = time.monotonic()
    assert hint_gate_ok(_hint(None, 10, now), "p1", True, now, 11) is True    # pid None + 在场
    assert hint_gate_ok(_hint(None, 10, now), "p1", False, now, 11) is False  # pid None + 画外
    assert hint_gate_ok(_hint(None, 10, now), None, True, now, 11) is True


def test_gate_empty_or_no_text():
    now = time.monotonic()
    assert hint_gate_ok(None, "p1", True, now, 10) is False
    assert hint_gate_ok({"text": "", "pid": "p1", "seq": 10, "ts": now}, "p1", True, now, 10) is False


# ────────── 测试替身 ──────────
class FakeSt:
    def __init__(self):
        self.lock = threading.Lock()
        self.display_transcript = []
        self.reasoner_hint = None


class _Msg:
    def __init__(self, c): self.content = c


class _Choice:
    def __init__(self, c): self.message = _Msg(c)


class _Resp:
    def __init__(self, c): self.choices = [_Choice(c)]


class _Chat:
    def __init__(self, c):
        self.completions = type("_C", (), {"create": lambda _s, **kw: _Resp(c)})()


class _Client:
    def __init__(self, c): self.chat = _Chat(c)


# ────────── ③ latest-only 丢旧 ──────────
def test_latest_only_drop():
    r = ConversationReasoner(FakeSt(), None, oai_client_factory=lambda: None)
    r.notify(1, "p1")
    r.notify(2, "p1")
    r.notify(3, "p2")               # worker 未启动 → 无人消费,队列只应剩最新
    assert r._q.get_nowait() == (3, "p2")
    assert r._q.empty()
    assert r.n_dropped == 2


# ────────── mock LLM 端到端(_process 首次无去抖)──────────
def test_process_with_mock_llm():
    content = '{"topics":[{"t":"露营","hook":"最近去露营了吗?"}],"style":"1-2短句+反问"}'
    r = ConversationReasoner(FakeSt(), None, oai_client_factory=lambda: _Client(content))
    r._process(7, "p1")             # 首次 last_done_at=0 → 去抖为负 → 不阻塞
    h = r.st.reasoner_hint
    assert h and "露营" in h["text"] and h["seq"] == 7 and h["pid"] == "p1"
    assert r.n_ok == 1 and r.n_calls == 1


def test_process_bad_json_dropped():
    r = ConversationReasoner(FakeSt(), None, oai_client_factory=lambda: _Client("总之聊点开心的吧"))
    r._process(1, None)
    assert r.st.reasoner_hint is None and r.n_fail == 1 and r.n_calls == 1


def test_invalidate():
    st = FakeSt()
    r = ConversationReasoner(st, None, oai_client_factory=lambda: None)
    st.reasoner_hint = {"text": "x"}
    r.invalidate()
    assert st.reasoner_hint is None


# ────────── 锁纪律回归 / drain / start 守卫 ──────────
def test_lock_free_during_llm():
    """锁纪律硬门:LLM create() 被调用时 st.lock 绝不能被持有。"""
    st = FakeSt()
    seen = {}

    class _LockAssertClient:
        def __init__(self):
            def _create(**kw):
                seen["locked"] = st.lock.locked()
                assert not st.lock.locked(), "LLM 调用时不得持有 st.lock"
                return _Resp('{"topics":[{"t":"x","hook":"h"}],"style":"短"}')
            self.chat = type("_Ch", (), {"completions":
                                         type("_Co", (), {"create": staticmethod(_create)})()})()

    r = ConversationReasoner(st, None, oai_client_factory=lambda: _LockAssertClient())
    r._process(3, "p1")
    assert seen["locked"] is False and r.n_ok == 1


def test_drain_uses_latest_snapshot():
    """去抖后 drain:处理队列里最新任务,不是 sleep 前 pop 的旧任务。"""
    r = ConversationReasoner(FakeSt(), None,
                             oai_client_factory=lambda: _Client('{"topics":[{"t":"x","hook":"h"}]}'))
    r._q.put_nowait((9, "p2"))       # 更新的任务已在队列
    r._process(5, "p1")              # 传入旧 (5,p1);drain 应改用 (9,p2)
    h = r.st.reasoner_hint
    assert h["seq"] == 9 and h["pid"] == "p2"


def test_latency_accounted():
    """延迟记账:create() sleep 0.05s → avg_latency_s > 0(排除上批 0.0 = 瞬时 mock 的疑点)。"""
    st = FakeSt()

    def _slow_create(**kw):
        time.sleep(0.05)
        return _Resp('{"topics":[{"t":"x","hook":"h"}],"style":"短"}')

    class _SlowClient:
        def __init__(s):
            s.chat = type("_Ch", (), {"completions":
                                      type("_Co", (), {"create": staticmethod(_slow_create)})()})()

    r = ConversationReasoner(st, None, oai_client_factory=lambda: _SlowClient())
    r._process(1, "p1")
    s = r.stats()
    assert s["avg_latency_s"] > 0, f"latency 未记账: {s}"
    assert s["ok"] == 1


def test_notify_without_start_warns_once():
    """未 start() 就 notify → 打一次告警(log-once),flag 只置一次。"""
    r = ConversationReasoner(FakeSt(), None, oai_client_factory=lambda: None)
    assert r._warned_no_start is False
    r.notify(1, "p1")
    assert r._warned_no_start is True
    r.notify(2, "p1")                # 再调不重复告警
    assert r._warned_no_start is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"  PASS {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
