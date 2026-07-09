# -*- coding: utf-8 -*-
"""认知记忆管理 — Entity Memory + Episodic Memory + Working Memory 注入。

架构参考: docs/COGNITIVE_MEMORY_ARCHITECTURE.md

记忆生命周期:
  会话中:
    remember_fact(key, value) → KV 格式实时存盘，同 key 自动覆盖
    forget_fact(keyword) → 模糊匹配 key 或 value 删除
  会话后 (close_session):
    consolidate_facts(new_facts, summary) → LLM 复盘后整体替换 entity memory
    save_episode(episode) → 结构化事件写入 episodic memory
  下次对话:
    get_prompt(pid) → summary + KV 详情 + episodic 组装注入 Working Memory

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
from pathlib import Path
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


def _migrate_legacy_facts(old_facts) -> tuple[Optional[str], dict[str, str]]:
    """将旧格式 facts 迁移为 (name, dict[str,str])。

    支持两种旧格式:
      1. dict 英文 key (likes_cats: "true") → KV 中文化
      2. list[str] 中文短句 → 用序号 key 兜底
    """
    if isinstance(old_facts, list):
        new_facts = {}
        for i, item in enumerate(old_facts):
            if "喜欢" in item:
                thing = item.replace("喜欢吃", "").replace("喜欢打", "").replace("喜欢", "")
                key = f"喜欢的{'食物' if '吃' in item else '运动' if '打' in item else '东西'}"
                if key in new_facts:
                    key = f"{key}{i}"
                new_facts[key] = item
            elif "是" in item and len(item) < 10:
                new_facts["职业"] = item.lstrip("是")
            elif "岁" in item:
                new_facts["年龄"] = item
            elif "爱好" in item:
                new_facts["爱好"] = item.replace("爱好是", "")
            else:
                new_facts[f"备注{i+1}"] = item
        return None, new_facts

    name = old_facts.get("name")
    new_facts = {}
    for k, v in old_facts.items():
        if k in _LEGACY_SKIP_KEYS:
            continue
        if any(k.startswith(p) for p in _LEGACY_SKIP_PREFIXES):
            continue
        if k in _LEGACY_FACT_MAP:
            mapped = _LEGACY_FACT_MAP[k]
            thing = k[6:].replace("_", " ") if k.startswith("likes_") else k
            key = f"喜欢的{'食物' if 'eat' in k or '吃' in mapped else '运动' if '打' in mapped else '东西'}"
            if key in new_facts:
                key = f"{key}_{thing}"
            new_facts[key] = mapped.replace("喜欢吃", "").replace("喜欢打", "").replace("喜欢", "")
        elif k.startswith("likes_"):
            thing = k[6:].replace("_", " ")
            new_facts[f"喜欢的东西"] = thing
        elif k == "job":
            new_facts["职业"] = v
        elif k == "age":
            new_facts["年龄"] = f"{v}岁"
        elif k == "hobby":
            new_facts["爱好"] = v
        else:
            new_facts[k] = v
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


# ── Qwen 工具定义 (DEPRECATED: 工具规格现由 tools/memory.py 生成，此处保留供参考) ──

QWEN_TOOLS = [
    {
        "type": "function",
        "name": "remember_fact",
        "description": (
            "记住用户告诉你的个人信息。用 key 描述类别，value 描述内容。"
            "例如：'我喜欢猫'→ remember_fact(key='喜欢的动物', value='猫')；"
            "'我是做AI的'→ remember_fact(key='职业', value='AI从业者')；"
            "'我叫小明'→ remember_fact(key='称呼', value='小明', name='小明')。"
            "相同 key 会自动覆盖旧值，不需要额外处理。"
            "注重理解上下文含义，提取有意义的信息分类存储。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "信息类别，如'爱好''职业''喜欢的食物'"},
                "value": {"type": "string", "description": "具体内容，如'打篮球''程序员''火锅'"},
                "name": {
                    "type": "string",
                    "description": "用户的名字（仅在用户自报姓名时传）",
                },
            },
            "required": ["key", "value"],
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
                 identity_store=None, face_db=None):
        self.memories_dir = memories_dir
        os.makedirs(self.memories_dir, exist_ok=True)
        self._session: dict[str, dict] = {}
        self._dirty: set[str] = set()
        self._owner = owner_mgr
        self._identity_store = identity_store

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
            data = {"name": None, "summary": None, "facts": {}, "episodes": []}

        # 自动移除 history 字段
        if "history" in data:
            del data["history"]

        # 迁移旧格式: list[str] 或旧英文 key dict → 新 dict[str,str]
        needs_migrate = False
        if isinstance(data.get("facts"), list):
            migrated_name, migrated_facts = _migrate_legacy_facts(data["facts"])
            data["facts"] = migrated_facts
            if migrated_name and not data.get("name"):
                data["name"] = migrated_name
            needs_migrate = True
        elif isinstance(data.get("facts"), dict):
            sample_keys = list(data["facts"].keys())[:3]
            if sample_keys and all(k.isascii() and "_" in k for k in sample_keys):
                migrated_name, migrated_facts = _migrate_legacy_facts(data["facts"])
                data["facts"] = migrated_facts
                if migrated_name and not data.get("name"):
                    data["name"] = migrated_name
                needs_migrate = True

        if "conversation_summaries" in data:
            data["episodes"] = _migrate_legacy_summaries(
                data.pop("conversation_summaries"))
            needs_migrate = True

        if needs_migrate:
            if "name" not in data:
                data["name"] = None
            if "summary" not in data:
                data["summary"] = None
            if "episodes" not in data:
                data["episodes"] = []
            self._dirty.add(person_id)
            self._session[person_id] = data
            self._persist(person_id)
            return data

        if "name" not in data:
            data["name"] = None
        if "summary" not in data:
            data["summary"] = None
        if "episodes" not in data:
            data["episodes"] = []
        if not isinstance(data.get("facts"), dict):
            data["facts"] = {}
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

    # ── Entity Memory: facts (KV 格式, 同 key 自动覆盖) ──

    def save_fact(self, person_id: str, key: str, value: str) -> str:
        """保存一条 fact。同 key 自动覆盖旧值。"""
        data = self.load_memory(person_id)
        facts = data.get("facts", {})
        old_value = facts.get(key)
        facts[key] = value
        # 超限时删除最老的 key（dict 保序，前面的是旧的）
        if len(facts) > MAX_FACTS:
            oldest_keys = list(facts.keys())[: len(facts) - MAX_FACTS]
            for ok in oldest_keys:
                if ok != key:  # 不删刚加的
                    del facts[ok]
        data["facts"] = facts
        self._dirty.add(person_id)
        self._persist(person_id)
        if old_value is not None:
            return f"已更新：{key}='{old_value}' → '{value}'"
        return f"已记住：{key}={value}"

    def forget_fact(self, person_id: str, keyword: str,
                    actor_pid: str = None) -> str:
        """删除 key 或 value 包含 keyword 的 fact。"""
        if actor_pid and self._owner and not self._owner.can_delete_memory(actor_pid, person_id):
            return "只有主人才能删除其他人的记忆哦。"
        data = self.load_memory(person_id)
        facts = data.get("facts", {})
        matched = {k: v for k, v in facts.items() if keyword in k or keyword in v}
        if not matched:
            available = "、".join(f"{k}={v}" for k, v in list(facts.items())[:10]) if facts else "无"
            return f"没有找到包含「{keyword}」的记忆。当前记忆: {available}"
        for k in matched:
            del facts[k]
        data["facts"] = facts
        self._dirty.add(person_id)
        self._persist(person_id)
        removed_str = "、".join(f"{k}={v}" for k, v in matched.items())
        return f"已忘掉 {len(matched)} 条: {removed_str}"

    def get_facts(self, person_id: str) -> dict[str, str]:
        """获取某人当前所有 facts (KV dict)。"""
        data = self.load_memory(person_id)
        return dict(data.get("facts", {}))

    def consolidate_facts(self, person_id: str, new_facts: dict[str, str],
                          new_name: str = None, new_summary: str = None):
        """会话后 consolidation：整体替换 facts dict + summary。"""
        data = self.load_memory(person_id)
        trimmed = dict(list(new_facts.items())[:MAX_FACTS])
        data["facts"] = trimmed
        if new_summary is not None:
            data["summary"] = new_summary
        if new_name and new_name != data.get("name"):
            data["name"] = new_name
            if self._identity_store:
                self._identity_store.set_name(person_id, new_name)
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
        """从 Entity + Episodic Memory 组装注入 Working Memory 的 prompt。

        格式: [记忆] summary叙事 + KV详情 + episode + 使用指引
        """
        data = self.load_memory(person_id)
        parts = []
        name = data.get("name") or person_name
        if name:
            parts.append(f"你面前的人叫{name}。")
        summary = data.get("summary")
        if summary:
            parts.append(summary)
        facts = data.get("facts", {})
        if facts:
            kv_lines = "\n".join(f"- {k}：{v}" for k, v in facts.items())
            parts.append(kv_lines)
        episodes = data.get("episodes", [])
        if episodes:
            latest = episodes[-1]
            topic = latest.get("topic", "")
            if topic:
                parts.append(f"你们上次聊过：{topic.rstrip('。')}。")
        if parts:
            parts.append("这些是你对这个人的了解，作为背景知识自然运用，不要主动背诵或列举。")
        return "\n".join(parts) if parts else None

    # ── 清除 / 合并 ──

    def clear_all(self, person_id: str, confirmed: bool = False,
                  actor_pid: str = None) -> str:
        """清除某人所有记忆。需要 confirmed=True。操作前自动备份到 backups/。"""
        if not confirmed:
            return "请先向用户确认是否要清除所有记忆。"
        if actor_pid and self._owner and not self._owner.can_delete_memory(actor_pid, person_id):
            return "只有主人才能删除其他人的记忆哦。"
        data = self.load_memory(person_id)
        old_facts = dict(data.get("facts", {}))
        # 备份到 backups/ 目录
        backup_dir = Path(self.memories_dir) / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f"{person_id}_{int(time.time())}.json"
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 清空
        data["name"] = None
        data["summary"] = None
        data["facts"] = {}
        data["episodes"] = []
        self._dirty.add(person_id)
        self._persist(person_id)
        return f"已清除所有记忆（共 {len(old_facts)} 条事实），备份已保存到 {backup_path.name}。"

    def merge_memories(self, keep_pid: str, drop_pid: str) -> None:
        """合并 drop_pid 的记忆到 keep_pid（facts KV 合并 + episodes 合并）。"""
        drop_data = self.load_memory(drop_pid)
        drop_facts = drop_data.get("facts", {})
        drop_episodes = drop_data.get("episodes", [])
        drop_name = drop_data.get("name")
        drop_summary = drop_data.get("summary")
        if not drop_facts and not drop_episodes and not drop_name:
            return
        keep_data = self.load_memory(keep_pid)
        keep_facts = keep_data.get("facts", {})
        for k, v in drop_facts.items():
            if k not in keep_facts:
                keep_facts[k] = v
        trimmed = dict(list(keep_facts.items())[:MAX_FACTS])
        keep_data["facts"] = trimmed
        keep_episodes = keep_data.get("episodes", [])
        keep_episodes.extend(drop_episodes)
        keep_episodes.sort(key=lambda e: e.get("ts", ""))
        keep_data["episodes"] = keep_episodes[-MAX_EPISODES:]
        if not keep_data.get("name") and drop_name:
            keep_data["name"] = drop_name
        if not keep_data.get("summary") and drop_summary:
            keep_data["summary"] = drop_summary
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
                facts = data.get("facts", {})
                if isinstance(facts, list):
                    _, facts = _migrate_legacy_facts(facts)
                elif isinstance(facts, dict):
                    sample_keys = list(facts.keys())[:3]
                    if sample_keys and all(k.isascii() and "_" in k for k in sample_keys):
                        _, facts = _migrate_legacy_facts(facts)
                result.append({
                    "person_id": pid,
                    "name": data.get("name"),
                    "summary": data.get("summary"),
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
            key = args.get("key", "")
            value = args.get("value", "")
            if not key or not value:
                return "缺少 key 或 value 参数。"
            return self.save_fact(person_id, key, value)
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
            summary_s = f"  {p['summary']}" if p.get("summary") else ""
            print(f"\n  {p['person_id']}  {name_s}  ({p['n_facts']} facts, {p['n_episodes']} episodes){summary_s}")
            facts = p["facts"]
            if isinstance(facts, dict):
                for k, v in facts.items():
                    print(f"    · {k}: {v}")
            else:
                for f in facts:
                    print(f"    · {f}")
        return

    if args.show:
        data = mm.load_memory(args.show)
        name = data.get("name") or "(未命名)"
        summary = data.get("summary")
        facts = data.get("facts", {})
        episodes = data.get("episodes", [])
        print(f"{args.show} ({name})")
        if summary:
            print(f"\n  Summary: {summary}")
        print(f"\n  Entity Memory ({len(facts)} facts):")
        if isinstance(facts, dict):
            for k, v in facts.items():
                print(f"    · {k}: {v}")
        else:
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
