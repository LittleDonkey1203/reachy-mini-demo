# -*- coding: utf-8 -*-
"""认知记忆管理 — Entity Memory + Episodic Memory + Working Memory 注入。

架构参考: docs/COGNITIVE_MEMORY_ARCHITECTURE.md

记忆生命周期:
  会话中:
    remember_fact(fact, replaces?) → draft notes 实时存盘
    forget_fact(keyword) → 模糊匹配删除
  会话后 (close_session):
    consolidate_facts(new_facts) → LLM 复盘后整体替换 entity memory
    save_episode(episode) → 结构化事件写入 episodic memory
  下次对话:
    get_prompt(pid) → 从 entity + episodic 组装注入 Working Memory

用法(独立测试):
  python memory/manager.py --list
  python memory/manager.py --show <person_id>
  python memory/manager.py --clear <person_id>
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEMORIES_DIR = os.path.join(_REPO, "data", "memories")

MAX_EPISODES = 10
MAX_FACTS = 50

# ── 旧格式 → 新格式翻译映射 ──
_LEGACY_FACT_MAP = {
    "likes_watermelon": "喜欢吃西瓜",
    "likes_badminton": "喜欢打羽毛球",
    "likes_basketball": "喜欢打篮球",
    "likes_hotpot": "喜欢吃火锅",
    "likes_pineapple": "喜欢吃菠萝",
    "likes_cats": "喜欢猫",
    "likes_dogs": "喜欢狗",
}

_LEGACY_SKIP_KEYS = {"name", "is_owner"}
_LEGACY_SKIP_PREFIXES = ("weather_",)


def _migrate_legacy_facts(old_facts: dict) -> tuple[Optional[str], list[str]]:
    """将旧 {key: value} facts 迁移为 (name, list[str])。"""
    name = old_facts.get("name")
    new_facts = []
    for k, v in old_facts.items():
        if k in _LEGACY_SKIP_KEYS:
            continue
        if any(k.startswith(p) for p in _LEGACY_SKIP_PREFIXES):
            continue
        if k in _LEGACY_FACT_MAP:
            new_facts.append(_LEGACY_FACT_MAP[k])
        elif k.startswith("likes_"):
            thing = k[6:].replace("_", " ")
            new_facts.append(f"喜欢{thing}")
        elif k == "job":
            new_facts.append(f"是{v}")
        elif k == "age":
            new_facts.append(f"{v}岁")
        elif k == "hobby":
            new_facts.append(f"爱好是{v}")
        else:
            new_facts.append(f"{v}")
    return name, new_facts


def _migrate_legacy_summaries(summaries: list[dict]) -> list[dict]:
    """将旧 conversation_summaries 迁移为 episodes。"""
    episodes = []
    for s in summaries:
        episodes.append({
            "ts": s.get("at", datetime.now().isoformat()),
            "topic": s.get("text", ""),
            "highlights": [],
            "mood": "unknown",
        })
    return episodes


# ── Qwen 工具定义 ──

QWEN_TOOLS = [
    {
        "type": "function",
        "name": "remember_fact",
        "description": (
            "记住用户告诉你的个人信息。用一句简短中文描述。"
            "例如：'我喜欢猫'→ fact='喜欢猫'；"
            "'我是做AI的'→ fact='从事AI工作'；"
            "'我叫小明，是程序员'→ name='小明' fact='程序员'。"
            "用户自报姓名时名字只放 name 参数，不要把'叫X'写进 fact。"
            "如果用户改变了之前说的信息（如'我不喜欢篮球了，改喜欢羽毛球'），"
            "用 replaces 指定要替换的关键词（如 replaces='篮球'）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "一句中文短句描述这条信息"},
                "replaces": {
                    "type": "string",
                    "description": "要替换的旧信息关键词（可选，仅在更新已有信息时传）",
                },
                "name": {
                    "type": "string",
                    "description": "用户的名字（仅在用户自报姓名时传）",
                },
            },
            "required": ["fact"],
        },
    },
    {
        "type": "function",
        "name": "clear_memory",
        "description": (
            "当用户表达想要清除/忘掉记忆的意图时调用。系统将自动启动安全验证流程。"
            "你只需要判断用户想删除谁的记忆：不传 target_name 表示删自己的；传名字表示删别人的。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_name": {
                    "type": "string",
                    "description": "要清除记忆的目标人名。不传则清除当前用户自己的记忆。",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "confirm_clear",
        "description": (
            "仅在系统要求你进行二次确认、且用户已口头明确回答后调用。"
            "用户说'是/确认/删吧'→confirmed=true；"
            "用户说'不/算了/取消'→confirmed=false。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "confirmed": {
                    "type": "boolean",
                    "description": "用户是否明确确认要清除",
                },
            },
            "required": ["confirmed"],
        },
    },
    {
        "type": "function",
        "name": "forget_fact",
        "description": (
            "忘掉关于用户的某一条信息。说关键词即可，如'猫''火锅''工作'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "要忘掉的信息关键词"},
            },
            "required": ["keyword"],
        },
    },
]


class MemoryManager:
    """认知记忆管理器：Entity Memory + Episodic Memory。"""

    def __init__(self, memories_dir: str = _MEMORIES_DIR, owner_mgr=None,
                 face_db=None):
        self.memories_dir = memories_dir
        os.makedirs(self.memories_dir, exist_ok=True)
        self._session: dict[str, dict] = {}
        self._dirty: set[str] = set()
        self._owner = owner_mgr
        self._face_db = face_db

    def _path(self, person_id: str) -> str:
        safe_id = person_id.replace("/", "_").replace("..", "_")
        return os.path.join(self.memories_dir, f"{safe_id}.json")

    # ── 加载 + 自动迁移 ──

    def load_memory(self, person_id: str) -> dict:
        """加载某人的记忆（磁盘→会话缓存），自动迁移旧格式。"""
        if person_id in self._session:
            return self._session[person_id]
        path = self._path(person_id)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = {"name": None, "facts": [], "episodes": [], "history": []}

        if isinstance(data.get("facts"), dict):
            migrated_name, migrated_facts = _migrate_legacy_facts(data["facts"])
            data["facts"] = migrated_facts
            if migrated_name and not data.get("name"):
                data["name"] = migrated_name
            if "conversation_summaries" in data:
                data["episodes"] = _migrate_legacy_summaries(
                    data.pop("conversation_summaries"))
            if "name" not in data:
                data["name"] = None
            if "episodes" not in data:
                data["episodes"] = []
            self._dirty.add(person_id)
            self._session[person_id] = data
            self._persist(person_id)
            return data

        if "name" not in data:
            data["name"] = None
        if "episodes" not in data:
            data["episodes"] = []
        if not isinstance(data.get("facts"), list):
            data["facts"] = []
        self._session[person_id] = data
        return data

    # ── Entity Memory: name ──

    def set_name(self, person_id: str, name: Optional[str]):
        data = self.load_memory(person_id)
        data["name"] = name
        self._dirty.add(person_id)
        self._persist(person_id)

    def get_name(self, person_id: str) -> Optional[str]:
        data = self.load_memory(person_id)
        return data.get("name")

    # ── Entity Memory: facts (draft notes / consolidation 输出) ──

    def save_fact(self, person_id: str, fact: str,
                  replaces: str = None) -> str:
        """保存一条 fact。replaces 非空时先删除含该关键词的旧 fact。"""
        data = self.load_memory(person_id)
        facts = data.get("facts", [])
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        removed = []
        if replaces:
            new_facts = []
            for f in facts:
                if replaces in f:
                    removed.append(f)
                else:
                    new_facts.append(f)
            facts = new_facts
        if fact not in facts:
            facts.append(fact)
        if len(facts) > MAX_FACTS:
            facts = facts[-MAX_FACTS:]
        data["facts"] = facts
        history = data.get("history", [])
        history.append({
            "action": "save_fact",
            "fact": fact,
            "replaces": replaces,
            "removed": removed,
            "at": now,
        })
        if len(history) > 200:
            history = history[-100:]
        data["history"] = history
        self._dirty.add(person_id)
        self._persist(person_id)
        if removed:
            return f"已更新：'{removed[0]}' → '{fact}'"
        return f"已记住：{fact}"

    def forget_fact(self, person_id: str, keyword: str,
                    actor_pid: str = None) -> str:
        """删除含 keyword 的 fact。支持模糊匹配。"""
        if actor_pid and self._owner and not self._owner.can_delete_memory(actor_pid, person_id):
            return "只有主人才能删除其他人的记忆哦。"
        data = self.load_memory(person_id)
        facts = data.get("facts", [])
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        matched = [f for f in facts if keyword in f]
        if not matched:
            available = "、".join(facts[:10]) if facts else "无"
            return f"没有找到包含「{keyword}」的记忆。当前记忆: {available}"
        remaining = [f for f in facts if keyword not in f]
        data["facts"] = remaining
        history = data.get("history", [])
        history.append({
            "action": "forget_fact",
            "keyword": keyword,
            "removed": matched,
            "at": now,
        })
        data["history"] = history
        self._dirty.add(person_id)
        self._persist(person_id)
        return f"已忘掉 {len(matched)} 条: {'、'.join(matched)}"

    def get_facts(self, person_id: str) -> list[str]:
        """获取某人当前所有 facts。"""
        data = self.load_memory(person_id)
        return list(data.get("facts", []))

    def consolidate_facts(self, person_id: str, new_facts: list[str],
                          new_name: str = None):
        """会话后 consolidation：整体替换 facts 列表。"""
        data = self.load_memory(person_id)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        old_facts = list(data.get("facts", []))
        data["facts"] = new_facts[:MAX_FACTS]
        if new_name and new_name != data.get("name"):
            data["name"] = new_name
            if self._face_db:
                self._face_db.set_name(person_id, new_name)
        history = data.get("history", [])
        history.append({
            "action": "consolidate",
            "old_facts": old_facts,
            "new_facts": new_facts,
            "at": now,
        })
        if len(history) > 200:
            history = history[-100:]
        data["history"] = history
        self._dirty.add(person_id)
        self._persist(person_id)

    # ── Episodic Memory ──

    def save_episode(self, person_id: str, episode: dict):
        """保存一条结构化事件，保留最近 MAX_EPISODES 条。"""
        data = self.load_memory(person_id)
        episodes = data.get("episodes", [])
        if "ts" not in episode:
            episode["ts"] = datetime.now().isoformat()
        episodes.append(episode)
        data["episodes"] = episodes[-MAX_EPISODES:]
        self._dirty.add(person_id)
        self._persist(person_id)

    # ── Working Memory 注入 ──

    def get_prompt(self, person_id: str, person_name: str = None) -> Optional[str]:
        """从 Entity + Episodic Memory 组装注入 Working Memory 的 prompt。"""
        data = self.load_memory(person_id)
        parts = []
        name = data.get("name") or person_name
        if name:
            parts.append(f"你面前的人叫{name}。")
        facts = data.get("facts", [])
        if facts:
            parts.append("你记得：" + "；".join(facts) + "。")
        episodes = data.get("episodes", [])
        if episodes:
            latest = episodes[-1]
            topic = latest.get("topic", "")
            if topic:
                parts.append(f"你们上次聊过：{topic.rstrip('。')}。")
        if parts:
            parts.append("这些记忆仅作为背景知识，只在用户主动提起相关话题时才使用，绝不要主动提起或背诵。")
        return "".join(parts) if parts else None

    # ── 清除 / 合并 ──

    def clear_all(self, person_id: str, confirmed: bool = False,
                  actor_pid: str = None) -> str:
        """清除某人所有记忆。需要 confirmed=True。"""
        if not confirmed:
            return "请先向用户确认是否要清除所有记忆。"
        if actor_pid and self._owner and not self._owner.can_delete_memory(actor_pid, person_id):
            return "只有主人才能删除其他人的记忆哦。"
        data = self.load_memory(person_id)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        old_facts = list(data.get("facts", []))
        old_name = data.get("name")
        data["name"] = None
        data["facts"] = []
        data["episodes"] = []
        history = data.get("history", [])
        history.append({
            "action": "clear_all",
            "old_name": old_name,
            "old_facts": old_facts,
            "at": now,
        })
        data["history"] = history
        self._dirty.add(person_id)
        self._persist(person_id)
        return f"已清除所有记忆（共 {len(old_facts)} 条事实）。"

    def merge_memories(self, keep_pid: str, drop_pid: str) -> None:
        """合并 drop_pid 的记忆到 keep_pid（facts 去重 + episodes 合并）。"""
        drop_data = self.load_memory(drop_pid)
        drop_facts = drop_data.get("facts", [])
        drop_episodes = drop_data.get("episodes", [])
        drop_name = drop_data.get("name")
        if not drop_facts and not drop_episodes and not drop_name:
            return
        keep_data = self.load_memory(keep_pid)
        keep_facts = keep_data.get("facts", [])
        for f in drop_facts:
            if f not in keep_facts:
                keep_facts.append(f)
        keep_data["facts"] = keep_facts[:MAX_FACTS]
        keep_episodes = keep_data.get("episodes", [])
        keep_episodes.extend(drop_episodes)
        keep_episodes.sort(key=lambda e: e.get("ts", ""))
        keep_data["episodes"] = keep_episodes[-MAX_EPISODES:]
        if not keep_data.get("name") and drop_name:
            keep_data["name"] = drop_name
        self._dirty.add(keep_pid)
        self._persist(keep_pid)
        self.clear_all(drop_pid, confirmed=True)

    # ── 持久化 ──

    def _persist(self, person_id: str):
        data = self._session.get(person_id)
        if data is None:
            return
        path = self._path(person_id)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        self._dirty.discard(person_id)

    def backup_person(self, person_id: str) -> Optional[str]:
        """备份某人的记忆文件到 data/backups/。"""
        src = self._path(person_id)
        if not os.path.exists(src):
            return None
        backup_dir = os.path.join(os.path.dirname(self.memories_dir), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        dst = os.path.join(backup_dir, f"{ts}_{person_id}_memory.json")
        import shutil
        shutil.copy2(src, dst)
        return dst

    def flush(self):
        for pid in list(self._dirty):
            self._persist(pid)

    def unload(self, person_id: str):
        if person_id in self._dirty:
            self._persist(person_id)
        self._session.pop(person_id, None)

    def list_all(self) -> list[dict]:
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
                facts = data.get("facts", [])
                if isinstance(facts, dict):
                    _, facts = _migrate_legacy_facts(facts)
                result.append({
                    "person_id": pid,
                    "name": data.get("name"),
                    "n_facts": len(facts),
                    "facts": facts,
                    "n_episodes": len(data.get("episodes", [])),
                })
            except Exception:
                continue
        return result

    def handle_tool_call(self, person_id: str, tool_name: str, args: dict) -> str:
        """处理 Qwen function call。
        注意: clear_memory 和 confirm_clear 由 d01 工作流直接处理。
        """
        if tool_name == "remember_fact":
            fact = args.get("fact", "")
            if not fact:
                return "缺少 fact 参数。"
            replaces = args.get("replaces")
            return self.save_fact(person_id, fact, replaces=replaces)
        elif tool_name == "forget_fact":
            keyword = args.get("keyword", "")
            if not keyword:
                return "缺少 keyword 参数。"
            return self.forget_fact(person_id, keyword, actor_pid=person_id)
        return f"未知的记忆工具: {tool_name}"


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="认知记忆管理工具")
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
            name_s = p["name"] or "(未命名)"
            print(f"\n  {p['person_id']}  {name_s}  ({p['n_facts']} facts, {p['n_episodes']} episodes)")
            for f in p["facts"]:
                print(f"    · {f}")
        return

    if args.show:
        data = mm.load_memory(args.show)
        name = data.get("name") or "(未命名)"
        facts = data.get("facts", [])
        episodes = data.get("episodes", [])
        print(f"{args.show} ({name})")
        print(f"\n  Entity Memory ({len(facts)} facts):")
        for f in facts:
            print(f"    · {f}")
        print(f"\n  Episodic Memory ({len(episodes)} episodes):")
        for ep in episodes:
            print(f"    [{ep.get('ts', '?')}] {ep.get('topic', '?')} ({ep.get('mood', '?')})")
        prompt = mm.get_prompt(args.show)
        if prompt:
            print(f"\n  Working Memory 注入:\n    {prompt}")
        return

    if args.clear:
        confirm = input(f"确定清除 {args.clear} 的所有记忆? (y/N): ")
        if confirm.lower() == "y":
            result = mm.clear_all(args.clear, confirmed=True)
            print(result)
        return

    print("用法: python memory/manager.py --list | --show <pid> | --clear <pid>")


if __name__ == "__main__":
    _main()
