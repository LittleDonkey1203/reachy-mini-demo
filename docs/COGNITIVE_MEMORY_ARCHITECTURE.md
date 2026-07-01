# 小艺认知记忆架构重构

## 设计理念

来自人类认知科学的启发：人脑不是统一 memory，而是分层的认知系统。AGI 的 memory 更像一个**持续运行的认知系统**，而不是知识库。核心突破不在更大的模型，而在 **Narrative Memory + Self Model + Consolidation Engine**。

## 完整认知架构（目标蓝图）

```
                     AGENT (小艺)
                        |
          ┌──────── Working Memory ────────┐
          │  当前意识窗口 (4~10 objects)     │
          │  context window + attention     │
          └──────────────┬─────────────────┘
                         │
          ┌──────────────┼──────────────────────┐
          ▼              ▼                      ▼
    Entity Memory   Episodic Memory      Procedural Memory
    结构化事实       事件(发生了什么)       技能/策略
          │              │
          │         Experience Replay
          │              │
          │              ▼
          │      Consolidation Engine ──────────┐
          │              │                      │
          ▼              ▼                      ▼
                  Semantic Memory        Self Model
                  抽象知识(GraphDB)      Agent自我认知
                         │                      │
                         └──────┬───────────────┘
                                ▼
                        Narrative Memory
                        持续更新的人生故事线

                        World Model
                        外部世界状态(独立于个人记忆)
```

| 层 | 内容 | 特点 | 本次 |
|---|------|------|------|
| Working Memory | 当前注入 context 的内容 | 极小、高频更新、token 级 | ✅ 优化注入组装 |
| Entity Memory | 关于人的原子事实 | 结构化、快速查询、直接从交互提取 | ✅ 重构 facts 格式 |
| Episodic Memory | 事件记录 | 存"发生了什么"而非"学到了什么" | ✅ 替代 conversation_summaries |
| Procedural Memory | 可执行技能/策略 | Prompt Programs / Policies | ❌ 已有(hardcoded tools) |
| Semantic Memory | 从 episodes 回放抽象出的知识 | Knowledge Graph / GraphDB | ❌ 需 Consolidation Engine |
| Self Model | Agent 对自己的认知 | strengths/beliefs/personality | ❌ 未来 |
| Narrative Memory | 每个人的持续故事线 | 不是事件列表，是连续叙事 | ❌ 未来 |
| World Model | 外部世界状态 | 独立于个人记忆 | ❌ 未来 |

**关键区分**：
- **Entity Memory ≠ Semantic Memory**。Entity 是直接从交互提取的事实（"喜欢猫"），Semantic 是 Consolidation Engine 从多条 episode 回放抽象生成的知识（"这个人持续在探索 AGI 方向"）
- **Episodic 存事件不存摘要**。"2026-06-25 聊了 memory 架构，用户倾向认知架构方向" 是事件；"用户对AI感兴趣"是摘要（属于 Semantic）

## 当前问题

1. **auto_merge 不同步** — FaceDB 合并碎片人脸后，`merge_memories()` 从未被调用，memory 文件残留
2. **facts 格式死板** — `{likes_cats: "true"}` 英文 key + 布尔 value，注入 prompt 像机器码
3. **无事件记录** — `conversation_summaries` 是摘要而非事件
4. **记忆只靠实时 function call** — 模型漏调就丢信息，没有会话后复盘机制

## 记忆生命周期（核心设计）

```
┌─────────── 会话中 ──────────────┐     ┌─────── 会话后 (close_session) ───────┐
│                                 │     │                                      │
│  用户说话 → 模型回答             │     │  Session-level Consolidation:        │
│       ↓                         │     │                                      │
│  上下文就够用，无需检索          │     │  输入:                               │
│       ↓                         │     │    · 全量对话 transcript              │
│  remember_fact() 随手记要点     │────►│    · 会话中的 draft facts             │
│  (draft notes, 实时存盘)        │     │    · 已有的 entity memory             │
│       ↓                         │     │                                      │
│  identity_injected = False      │     │  LLM 复盘生成:                       │
│  → 下一帧重新注入最新 facts     │     │    · 最终 entity memory (facts)       │
│                                 │     │    · episodic memory (事件)           │
└─────────────────────────────────┘     │                                      │
                                        │  下次对话时注入 Working Memory        │
                                        └──────────────────────────────────────┘
```

**会话中**：`remember_fact(key, value)` 保留，作为实时 KV 写入。同 key 自动覆盖旧值，立即存盘并刷新注入（保持现有 `identity_injected=False` 触发重注入的逻辑）。

**会话后**：close_session 启动后台 consolidation 线程，输入 = 全量对话 + 当前 facts KV + 已有 facts，由 LLM 复盘生成：
1. **最终 entity memory** — 合并现有 facts + 对话中的新信息，KV 格式，去重去过时
2. **summary** — 基于所有 facts 和对话理解的一句话叙事性描述
3. **episodic memory** — 从对话中提取结构化事件（topic + highlights + mood）

这样即使模型会话中漏调 `remember_fact`，consolidation 也能从全量对话中兜底捕获。

## facts 格式设计决策

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| dict 英文 key + 英文 value | upsert 一致 | 注入读起来像机器码 | ❌ 旧格式v1 |
| dict 英文 key + 中文 value | upsert 一致、注入可读 | 更新时模型需完整复述旧值 | ❌ 复述慢且易漏 |
| dict 中文 key | 注入可读 | 中文同义词多导致 key 不一致 | ❌ upsert 命中率低 |
| list[str] + replaces 参数 | 原子操作、注入自然、无需复述 | 需 replaces 关键词匹配，去重不可靠 | ❌ 旧格式v2 |
| **dict[str,str] 中文KV + summary** | **精确更新(同key覆盖) + 叙事理解** | 中文 key 可能不完全一致 | **✅ 采用** |

`dict[str,str]` KV 格式解决精确更新（同 key 覆盖），`summary` 给模型叙事性理解：

- 新增：`remember_fact(key="喜欢的动物", value="猫")`
- 更新（同key覆盖）：`remember_fact(key="喜欢的动物", value="狗")` → 自动覆盖旧值
- 删除：`forget_fact(keyword="猫")` → 模糊匹配 key 或 value

## 数据格式

```json
{
  "name": "坤坤",
  "summary": "坤坤喜欢打篮球和看科幻小说，最近在关注《黑暗森林》上线",
  "facts": {
    "爱好": "打篮球",
    "喜欢的书": "《三体》",
    "近期关注": "《黑暗森林》二要上线"
  },
  "episodes": [
    {
      "ts": "2026-06-25T15:18:00",
      "topic": "介绍自己的功能",
      "highlights": ["用户询问小艺能做什么"],
      "mood": "casual"
    }
  ],
  "history": [...]
}
```

### 注入格式

```
[记忆]
你面前的人叫坤坤。
坤坤喜欢打篮球和看科幻小说，最近在关注《黑暗森林》上线。
- 爱好：打篮球
- 喜欢的书：《三体》
- 近期关注：《黑暗森林》二要上线
你们上次聊过：介绍自己的功能。
这些是你对这个人的了解，作为背景知识自然运用，不要主动背诵或列举。
```

## 实施计划

### Task 1: auto_merge 同步 MemoryManager

**文件**: `identity/recognizer.py`, `voice/d01_realtime_chat.py`

- 删除 `_merged_map` 死字段，auto_merge 结果改为 `startup_merged` 属性
- d01 初始化 (~L2022-2027): 创建 `_memory_mgr` 后遍历 merged map，调用 `merge_memories(keep, drop)`

### Task 2: 重构 memory/manager.py — Entity + Episodic

**文件**: `memory/manager.py`

**API 变更**:

```python
# Entity Memory — 会话中 KV 存储
save_fact(pid, key: str, value: str)
    # 同 key 自动覆盖旧值

forget_fact(pid, keyword: str)
    # 模糊匹配 key 或 value 删除

get_facts(pid) → dict[str, str]  # KV 格式

# Entity Memory — 会话后 consolidation 整体替换
consolidate_facts(pid, new_facts: dict[str,str], new_name: str = None, new_summary: str = None)
    # 整体替换 facts dict + summary
    # 如果 new_name 与当前不同，同步更新 name

# Episodic Memory
save_episode(pid, episode: dict)
    # 保留最近 10 条

# name 独立管理
set_name(pid, name) / get_name(pid)

# Working Memory 注入
get_prompt(pid, person_name=None) → str
    # "你面前的人叫坤坤。"
    # "坤坤喜欢打篮球和看科幻小说..."  (summary)
    # "- 爱好：打篮球"
    # "- 喜欢的书：《三体》"
    # "你们上次聊过：介绍自己的功能。"
    # "这些是你对这个人的了解..."

# 合并（适配 dict facts 合并）
merge_memories(keep_pid, drop_pid)
```

### Task 3: 更新 QWEN_TOOLS

**文件**: `memory/manager.py`

```python
remember_fact:
    key: str       # 信息类别，如"爱好""职业""喜欢的食物"
    value: str     # 具体内容，如"打篮球""程序员""火锅"
    name: str      # 可选，仅用户自报姓名时传

forget_fact:
    keyword: str   # 中文关键词模糊匹配 key 或 value

# clear_memory / confirm_clear 不变
```

### Task 4: Session Consolidation — 会话后复盘

**文件**: `voice/realtime.py` (RealtimeDialog.save_summary)

会话后 consolidation 现在生成 KV facts + summary + episode：

```python
def save_summary(self, pid, conv_log):
    """后台线程：会话后 consolidation — entity memory + episodic memory。"""
    # ... 组装 prompt ...
    # LLM 输出 JSON:
    # {
    #   "name": "用户名字",
    #   "summary": "一句话认知描述",
    #   "facts": {"类别1": "内容1", "类别2": "内容2"},
    #   "episode": {"topic": "...", "highlights": [...], "mood": "..."}
    # }
    self.memory_mgr.consolidate_facts(pid, new_facts, new_name, new_summary)
    self.memory_mgr.save_episode(pid, episode)
```

**关键点**：
- 输入 = 全量对话 transcript + 当前 facts KV（含会话中实时写入）+ 当前 name
- LLM 做合并/去重/去过时，输出干净的 facts dict + summary 叙事
- summary 体现对用户的**整体理解**，不是列举属性
- 同时提取 episodic memory（事件，不是摘要）
- 一次 LLM 调用完成 entity memory + summary + episodic memory

### Task 5: 迁移现有数据

**文件**: `memory/manager.py` (load_memory 中自动迁移)

检测旧格式 → 自动转换为 `dict[str,str]`：

| 旧格式 | 新格式 |
|---|---|
| `list[str]` 中文短句 | 推断 key（"喜欢X"→喜欢的东西），无法推断用"备注N"兜底 |
| `dict` 英文 key (likes_X) | 翻译映射表转中文 KV |
| `name: "大大"` | 顶层 `name` |
| `is_owner: "true"` | 丢弃（在 owner.json） |
| `weather_*` | 丢弃（临时信息） |
| `conversation_summaries` | → `episodes`（text→topic） |

### Task 6: d01 工具调用适配

**文件**: `voice/realtime.py` (ChatCallback.on_event)

- `remember_fact`: args 从 `fact`+`replaces` → `key`+`value`（由 handle_tool_call 处理）
- `name` 参数同步 FaceDB.set_name + OwnerManager.try_claim（不变）
- `forget_fact`: `keyword` 模糊匹配 key 或 value（不变）
- close_session → save_summary 触发 consolidation（不变）
- `identity_injected` 重置逻辑不变

## 不在本次范围

- Semantic Memory（需跨 session 的 Consolidation Engine）
- Self Model / Narrative Memory / World Model / Procedural Memory

## 验证

1. 启动 → auto_merge 后 memory 文件正确合并
2. 现有 memory 文件自动迁移 `dict[str,str]` 格式
3. 对话说"我喜欢猫" → facts 写入 `{"喜欢的动物": "猫"}`
4. 说"我不喜欢猫了喜欢狗" → 同 key 覆盖为 `{"喜欢的动物": "狗"}`
5. 说"忘掉猫" → keyword 匹配删除
6. 结束对话 → consolidation 生成 facts dict + summary + episode
7. 检查 consolidation 后的 facts 是否合并了实时写入 + 对话新信息 + 去掉过时项
8. 下次对话 → 注入格式正确（summary + KV 列表 + episode）
9. 模型不主动背诵记忆，只在用户提起相关话题时自然运用
