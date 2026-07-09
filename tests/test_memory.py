#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆管理功能测试。

用例：
  1. 基本读写         — save_fact / get_facts
  2. LWW 覆盖        — 同 key 写两次，后者覆盖前者
  3. 持久化           — 写入后重新加载仍一致
  4. 变更历史         — history 记录 set / clear_all
  5. 清除（需确认）    — confirmed=False 拒绝，confirmed=True 执行
  6. get_prompt 生成  — 有记忆时返回 prompt，无记忆时返回 None
  7. 多人隔离         — 不同 person_id 互不影响
  8. handle_tool_call — 模拟 Qwen function call
  9. flush / unload   — 脏数据持久化、切人卸载

运行:
  cd reachy-mini-demo/voice
  python test_memory.py
"""

import json
import os
import shutil
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from memory.manager import MemoryManager, QWEN_TOOLS

PASS = 0
FAIL = 0


def _check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def test_01_basic_rw():
    print("\n[Test 01] 基本读写")
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)
        result = mm.save_fact("p1", "name", "小明")
        _check("save_fact 返回确认", "已记住" in result)

        facts = mm.get_facts("p1")
        _check("get_facts 包含 name", facts.get("name") == "小明")

        mm.save_fact("p1", "job", "程序员")
        facts = mm.get_facts("p1")
        _check("多条 fact", len(facts) == 2)
        _check("job 正确", facts["job"] == "程序员")
    finally:
        shutil.rmtree(d)


def test_02_lww():
    print("\n[Test 02] LWW 覆盖")
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)
        mm.save_fact("p1", "name", "小A")
        _check("初始 name=小A", mm.get_facts("p1")["name"] == "小A")

        result = mm.save_fact("p1", "name", "小B")
        _check("覆盖返回'已更新'", "已更新" in result)
        _check("覆盖后 name=小B", mm.get_facts("p1")["name"] == "小B")

        mm.save_fact("p1", "name", "小C")
        _check("再覆盖 name=小C", mm.get_facts("p1")["name"] == "小C")
    finally:
        shutil.rmtree(d)


def test_03_persistence():
    print("\n[Test 03] 持久化")
    d = tempfile.mkdtemp()
    try:
        mm1 = MemoryManager(d)
        mm1.save_fact("p1", "name", "持久化")
        mm1.save_fact("p1", "hobby", "画画")

        mm2 = MemoryManager(d)
        facts = mm2.get_facts("p1")
        _check("重新加载后 name 一致", facts.get("name") == "持久化")
        _check("重新加载后 hobby 一致", facts.get("hobby") == "画画")

        fpath = os.path.join(d, "p1.json")
        _check("JSON 文件存在", os.path.exists(fpath))

        with open(fpath) as f:
            raw = json.load(f)
        _check("JSON 包含 facts", "facts" in raw)
        _check("JSON 包含 history", "history" in raw)
    finally:
        shutil.rmtree(d)


def test_04_history():
    print("\n[Test 04] 变更历史")
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)
        mm.save_fact("p1", "name", "小明")
        mm.save_fact("p1", "name", "小红")
        mm.save_fact("p1", "age", "25")

        data = mm.load_memory("p1")
        # 该分支已移除 per-person history 字段(manager.py load_memory 自动删除),改用 episodes 承载
        _check("history 字段已移除", "history" not in data)
        _check("改用 episodes 承载", "episodes" in data)
        _check("facts 为 LWW 最新值", data["facts"].get("name") == "小红")
    finally:
        shutil.rmtree(d)


def test_05_clear():
    print("\n[Test 05] 清除（需确认）")
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)
        mm.save_fact("p1", "name", "小明")
        mm.save_fact("p1", "job", "学生")

        result = mm.clear_all("p1", confirmed=False)
        _check("未确认时拒绝", "确认" in result)
        _check("facts 未清除", len(mm.get_facts("p1")) == 2)

        result = mm.clear_all("p1", confirmed=True)
        _check("确认后返回'已清除'", "已清除" in result)
        _check("facts 为空", len(mm.get_facts("p1")) == 0)

        # history 字段已移除(见 manager.py);clear_all 行为由上面 facts 断言覆盖
        _check("clear 后 history 不残留", "history" not in mm.load_memory("p1"))
    finally:
        shutil.rmtree(d)


def test_06_prompt():
    print("\n[Test 06] get_prompt 生成")
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)

        prompt = mm.get_prompt("p_empty")
        _check("无记忆+无名字 → None", prompt is None)

        prompt = mm.get_prompt("p_empty", person_name="小明")
        _check("有名字 → 含名字", prompt is not None and "小明" in prompt)

        mm.save_fact("p1", "name", "小红")
        mm.save_fact("p1", "hobby", "唱歌")
        mm.save_fact("p1", "job", "老师")
        prompt = mm.get_prompt("p1", person_name="小红")
        _check("prompt 含名字", "小红" in prompt)
        _check("prompt 含 hobby", "唱歌" in prompt)
        _check("prompt 含 job", "老师" in prompt)
        _check("prompt 不重复 name 在 facts 中", prompt.count("小红") <= 2)
    finally:
        shutil.rmtree(d)


def test_07_multi_person():
    print("\n[Test 07] 多人隔离")
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)
        mm.save_fact("alice", "name", "Alice")
        mm.save_fact("alice", "lang", "English")
        mm.save_fact("bob", "name", "Bob")
        mm.save_fact("bob", "hobby", "篮球")

        _check("Alice facts=2", len(mm.get_facts("alice")) == 2)
        _check("Bob facts=2", len(mm.get_facts("bob")) == 2)
        _check("Alice 无 hobby", "hobby" not in mm.get_facts("alice"))
        _check("Bob 无 lang", "lang" not in mm.get_facts("bob"))

        mm.clear_all("alice", confirmed=True)
        _check("清除 Alice 后 Alice 为空", len(mm.get_facts("alice")) == 0)
        _check("清除 Alice 后 Bob 不受影响", len(mm.get_facts("bob")) == 2)
    finally:
        shutil.rmtree(d)


def test_08_tool_call():
    print("\n[Test 08] handle_tool_call")
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)

        result = mm.handle_tool_call("p1", "remember_fact",
                                     {"key": "name", "value": "工具人"})
        _check("remember_fact 返回确认", "已记住" in result)
        _check("fact 写入", mm.get_facts("p1")["name"] == "工具人")

        result = mm.handle_tool_call("p1", "clear_memory",
                                     {"confirmed": False})
        _check("clear 未确认拒绝", "确认" in result)

        result = mm.handle_tool_call("p1", "clear_memory",
                                     {"confirmed": True})
        _check("clear 确认后执行", "已清除" in result)
        _check("facts 为空", len(mm.get_facts("p1")) == 0)

        result = mm.handle_tool_call("p1", "unknown_tool", {})
        _check("未知工具返回错误", "未知" in result)
    finally:
        shutil.rmtree(d)


def test_09_flush_unload():
    print("\n[Test 09] flush / unload")
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)
        mm.save_fact("p1", "name", "测试")

        _check("p1 在 session 中", "p1" in mm._session)

        mm.unload("p1")
        _check("unload 后不在 session", "p1" not in mm._session)

        fpath = os.path.join(d, "p1.json")
        _check("unload 后文件仍在", os.path.exists(fpath))

        mm2 = MemoryManager(d)
        _check("重新加载仍可用", mm2.get_facts("p1")["name"] == "测试")
    finally:
        shutil.rmtree(d)


def test_10_qwen_tools():
    print("\n[Test 10] QWEN_TOOLS 格式")
    _check("QWEN_TOOLS 有 2 个", len(QWEN_TOOLS) == 2)
    names = {t["name"] for t in QWEN_TOOLS}
    _check("包含 remember_fact", "remember_fact" in names)
    _check("包含 clear_memory", "clear_memory" in names)
    for t in QWEN_TOOLS:
        _check(f"{t['name']} 有 parameters", "parameters" in t)
        _check(f"{t['name']} type=function", t["type"] == "function")


def test_11_concurrency():
    print("\n[Test 11] 并发冒烟(MemoryManager RLock:可重入不死锁 + 无丢数据)")
    import threading
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)

        def worker(pid):
            for i in range(50):                      # 每 pid 50 key(避开 MAX_FACTS=50 的 trim)
                mm.save_fact(pid, f"k{i}", f"v{i}")

        t1 = threading.Thread(target=worker, args=("alice",))
        t2 = threading.Thread(target=worker, args=("bob",))
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)
        alive = t1.is_alive() or t2.is_alive()
        na, nb = len(mm.get_facts("alice")), len(mm.get_facts("bob"))
        _check("两线程均完成(无 RLock 死锁)", not alive)
        _check(f"alice 50 facts 完整(实得 {na})", na == 50)
        _check(f"bob 50 facts 完整(实得 {nb})", nb == 50)
        assert not alive, "并发写卡死(RLock 死锁?)"
        assert na == 50 and nb == 50, f"并发丢数据:alice={na}, bob={nb}"
    finally:
        shutil.rmtree(d)


def test_12_concurrency_same_pid():
    print("\n[Test 12] 并发冒烟·同 pid(RLock 序列化读改写,不丢)")
    import threading
    d = tempfile.mkdtemp()
    try:
        mm = MemoryManager(d)

        def worker(base):
            for i in range(20):                  # 2 线程 × 20 key = 40 < MAX_FACTS=50
                mm.save_fact("shared", f"{base}{i}", f"v{i}")

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)
        n = len(mm.get_facts("shared"))
        _check(f"同 pid 40 条完整(实得 {n})", n == 40)
        assert n == 40, f"同 pid 并发丢数据:{n}/40"
    finally:
        shutil.rmtree(d)


def main():
    print("=" * 60)
    print("  记忆管理功能测试 — memory_manager.py")
    print("=" * 60)

    tests = [
        test_01_basic_rw,
        test_02_lww,
        test_03_persistence,
        test_04_history,
        test_05_clear,
        test_06_prompt,
        test_07_multi_person,
        test_08_tool_call,
        test_09_flush_unload,
        test_10_qwen_tools,
        test_11_concurrency,
        test_12_concurrency_same_pid,
    ]

    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  💥 {fn.__name__} 异常: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  结果: ✅ {PASS}  ❌ {FAIL}")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
