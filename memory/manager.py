# -*- coding: utf-8 -*-
"""个人记忆管理 — 会话内短期 + LWW 持久化。

集成到 d01:
  1. 身份匹配后调 load_memory(person_id) 加载记忆
  2. 注入 Qwen session: conv.create_item(role="system", content=get_prompt(pid))
  3. Qwen function_call "remember_fact" → save_fact(pid, key, value)
  4. Qwen function_call "clear_memory" → clear_all(pid, confirmed=True)
  5. 会话结束/切人时自动持久化

用法(独立测试):
  python memory_manager.py --list                 # 列出所有人的记忆
  python memory_manager.py --show <person_id>     # 某人的记忆
  python memory_manager.py --clear <person_id>    # 清除某人记忆
"""

import json
import os
import time
from typing import Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEMORIES_DIR = os.path.join(_REPO, "data", "memories")

QWEN_TOOLS = [
    {
        "type": "function",
        "name": "remember_fact",
        "description": (
            "记住用户告诉你的个人信息。当用户说出自己的名字、喜好、职业、年龄等个人信息时主动调用。"
            "例如用户说'我叫小明'→ key='name' value='小明'；"
            "'我喜欢猫'→ key='likes_cats' value='true'；"
            "'我是程序员'→ key='job' value='程序员'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "信息类别，如 name, age, job, hobby, favorite_color 等"},
                "value": {"type": "string", "description": "具体内容"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "type": "function",
        "name": "clear_memory",
        "description": (
            "清除关于当前用户的所有记忆。仅当用户明确要求'忘掉我''清除我的记忆''删除我的信息'时调用。"
            "调用前必须口头向用户确认：'你确定要我忘掉关于你的所有信息吗？'，用户确认后才调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "confirmed": {
                    "type": "boolean",
                    "description": "用户是否已口头确认要清除",
                },
            },
            "required": ["confirmed"],
        },
    },
]


class MemoryManager:
    """管理每个人的记忆：会话内 dict + 磁盘 JSON 持久化。"""

    def __init__(self, memories_dir: str = _MEMORIES_DIR):
        self.memories_dir = memories_dir
        os.makedirs(self.memories_dir, exist_ok=True)
        self._session: dict[str, dict[str, str]] = {}
        self._dirty: set[str] = set()

    def _path(self, person_id: str) -> str:
        safe_id = person_id.replace("/", "_").replace("..", "_")
        return os.path.join(self.memories_dir, f"{safe_id}.json")

    def load_memory(self, person_id: str) -> dict:
        """加载某人的记忆（磁盘→会话缓存）。"""
        if person_id in self._session:
            return self._session[person_id]
        path = self._path(person_id)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = {"facts": {}, "history": []}
        self._session[person_id] = data
        return data

    def save_fact(self, person_id: str, key: str, value: str) -> str:
        """LWW 写入一条 fact。返回确认消息。"""
        data = self.load_memory(person_id)
        old = data["facts"].get(key)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["facts"][key] = value
        data["history"].append({
            "action": "set",
            "key": key,
            "value": value,
            "old_value": old,
            "at": now,
        })
        if len(data["history"]) > 200:
            data["history"] = data["history"][-100:]
        self._dirty.add(person_id)
        self._persist(person_id)
        if old and old != value:
            return f"已更新：{key} 从 '{old}' 改为 '{value}'"
        return f"已记住：{key} = '{value}'"

    def clear_all(self, person_id: str, confirmed: bool = False) -> str:
        """清除某人所有记忆。需要 confirmed=True。"""
        if not confirmed:
            return "请先向用户确认是否要清除所有记忆。"
        data = self.load_memory(person_id)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        old_facts = dict(data["facts"])
        data["facts"].clear()
        data["history"].append({
            "action": "clear_all",
            "old_facts": old_facts,
            "at": now,
        })
        self._dirty.add(person_id)
        self._persist(person_id)
        return f"已清除所有记忆（共 {len(old_facts)} 条）。"

    def get_facts(self, person_id: str) -> dict[str, str]:
        """获取某人当前所有 facts。"""
        data = self.load_memory(person_id)
        return dict(data["facts"])

    def get_prompt(self, person_id: str, person_name: str = None) -> Optional[str]:
        """生成注入 Qwen session 的记忆 prompt。无记忆则返回 None。"""
        facts = self.get_facts(person_id)
        if not facts and not person_name:
            return None
        parts = []
        display_name = facts.get("name") or person_name
        if display_name:
            parts.append(f"你面前的人叫{display_name}。")
        if facts:
            items = []
            for k, v in facts.items():
                if k == "name":
                    continue
                items.append(f"{k}: {v}")
            if items:
                parts.append("你记得关于ta的信息：" + "；".join(items) + "。")
        parts.append("自然地运用这些记忆，但不要主动背诵。")
        return "".join(parts)

    def _persist(self, person_id: str):
        """写入磁盘。"""
        data = self._session.get(person_id)
        if data is None:
            return
        path = self._path(person_id)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        self._dirty.discard(person_id)

    def flush(self):
        """持久化所有脏数据。"""
        for pid in list(self._dirty):
            self._persist(pid)

    def unload(self, person_id: str):
        """切人时卸载当前人记忆（先持久化）。"""
        if person_id in self._dirty:
            self._persist(person_id)
        self._session.pop(person_id, None)

    def list_all(self) -> list[dict]:
        """列出所有有记忆文件的人。"""
        result = []
        if not os.path.isdir(self.memories_dir):
            return result
        for fname in sorted(os.listdir(self.memories_dir)):
            if not fname.endswith(".json"):
                continue
            pid = fname[:-5]
            path = os.path.join(self.memories_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                result.append({
                    "person_id": pid,
                    "n_facts": len(data.get("facts", {})),
                    "facts": data.get("facts", {}),
                })
            except Exception:
                continue
        return result

    def handle_tool_call(self, person_id: str, tool_name: str, args: dict) -> str:
        """处理 Qwen function call。返回 function_call_output 内容。"""
        if tool_name == "remember_fact":
            key = args.get("key", "")
            value = args.get("value", "")
            if not key or not value:
                return "缺少 key 或 value 参数。"
            return self.save_fact(person_id, key, value)
        elif tool_name == "clear_memory":
            confirmed = args.get("confirmed", False)
            return self.clear_all(person_id, confirmed)
        return f"未知的记忆工具: {tool_name}"


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="记忆管理工具")
    parser.add_argument("--list", action="store_true", help="列出所有人的记忆")
    parser.add_argument("--show", type=str, help="查看某人的记忆 (person_id)")
    parser.add_argument("--clear", type=str, help="清除某人的记忆 (person_id)")
    parser.add_argument("--dir", type=str, default=_MEMORIES_DIR, help="记忆目录")
    args = parser.parse_args()

    mm = MemoryManager(args.dir)

    if args.list:
        persons = mm.list_all()
        if not persons:
            print("无记忆数据。")
            return
        print(f"共 {len(persons)} 人有记忆：")
        for p in persons:
            print(f"\n  {p['person_id']}  ({p['n_facts']} 条)")
            for k, v in p["facts"].items():
                print(f"    {k}: {v}")
        return

    if args.show:
        data = mm.load_memory(args.show)
        facts = data.get("facts", {})
        if not facts:
            print(f"{args.show} 无记忆。")
            return
        print(f"{args.show} 的记忆 ({len(facts)} 条)：")
        for k, v in facts.items():
            print(f"  {k}: {v}")
        prompt = mm.get_prompt(args.show)
        if prompt:
            print(f"\n注入 prompt:\n  {prompt}")
        return

    if args.clear:
        confirm = input(f"确定清除 {args.clear} 的所有记忆? (y/N): ")
        if confirm.lower() == "y":
            result = mm.clear_all(args.clear, confirmed=True)
            print(result)
        return

    print("交互测试模式。输入 '<person_id> <key> <value>' 存记忆，'q' 退出。")
    print("示例: person_abc name 小明")
    while True:
        line = input("> ").strip()
        if line == "q":
            break
        parts = line.split(None, 2)
        if len(parts) < 3:
            print("格式: <person_id> <key> <value>")
            continue
        pid, key, value = parts
        result = mm.save_fact(pid, key, value)
        print(f"  {result}")
        prompt = mm.get_prompt(pid)
        if prompt:
            print(f"  prompt: {prompt}")


if __name__ == "__main__":
    _main()
