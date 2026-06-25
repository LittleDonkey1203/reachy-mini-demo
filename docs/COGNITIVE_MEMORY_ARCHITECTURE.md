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

**会话中**：`remember_fact` 保留，作为 draft notes 实时记录模型认为重要的信息。这些 draft facts 立即存盘并刷新注入（保持现有 `identity_injected=False` 触发重注入的逻辑）。

**会话后**：close_session 启动后台 consolidation 线程，输入 = 全量对话 + draft facts + 已有 facts，由 LLM 复盘生成：
1. **最终 entity memory** — 合并 draft notes + 已有 facts + 对话中的新信息，去重去过时，输出干净的 facts list
2. **episodic memory** — 从对话中提取结构化事件（topic + highlights + mood）

这样即使模型会话中漏调 `remember_fact`，consolidation 也能从全量对话中兜底捕获。

## facts 格式设计决策

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| dict 英文 key + 英文 value | upsert 一致 | 注入读起来像机器码 | ❌ 现状，要改 |
| dict 英文 key + 中文 value | upsert 一致、注入可读 | 更新时模型需完整复述旧值 | ❌ 复述慢且易漏 |
| dict 中文 key | 注入可读 | 中文同义词多导致 key 不一致 | ❌ upsert 命中率低 |
| **list[str] + replaces 参数** | **原子操作、注入自然、无需复述** | 需 replaces 关键词匹配 | **✅ 采用** |

每条 fact 是独立的中文短句，增删改都是单次 function call：
- 新增：`remember_fact(fact="喜欢打羽毛球")`
- 更新：`remember_fact(fact="喜欢打羽毛球", replaces="篮球")` → 找含"篮球"的旧 fact 替换
- 删除：`forget_fact(keyword="篮球")`

## 数据格式

```json
{
  "name": "大大",
  "facts": [
    "喜欢打羽毛球",
    "喜欢吃西瓜",
    "是程序员"
  ],
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

## 实施计划

### Task 1: auto_merge 同步 MemoryManager

**文件**: `identity/recognizer.py`, `voice/d01_realtime_chat.py`

- 删除 `_merged_map` 死字段，auto_merge 结果改为 `startup_merged` 属性
- d01 初始化 (~L2022-2027): 创建 `_memory_mgr` 后遍历 merged map，调用 `merge_memories(keep, drop)`

### Task 2: 重构 memory/manager.py — Entity + Episodic

**文件**: `memory/manager.py`

**API 变更**:

```python
# Entity Memory — 会话中 draft notes
save_fact(pid, fact: str, replaces: str = None)
    # replaces=None → 追加
    # replaces="篮球" → 找含"篮球"的旧 fact 删掉，加入新 fact

forget_fact(pid, keyword: str)
    # 在 facts list 中模糊匹配删除

get_facts(pid) → list[str]  # 原来返回 dict

# Entity Memory — 会话后 consolidation 整体替换
consolidate_facts(pid, new_facts: list[str], new_name: str = None)
    # 整体替换 facts 列表（consolidation 输出）
    # 如果 new_name 与当前不同，同步更新 name

# Episodic Memory
save_episode(pid, episode: dict)
    # 保留最近 10 条

# name 独立管理
set_name(pid, name) / get_name(pid)

# Working Memory 注入
get_prompt(pid, person_name=None) → str
    # "你面前的人叫大大。"
    # "你记得：喜欢打羽毛球；喜欢吃西瓜；是程序员。"
    # "你们上次聊过：介绍自己的功能。"
    # "自然地运用这些记忆，但不要主动背诵。"

# 合并（适配 list facts 去重）
merge_memories(keep_pid, drop_pid)
```

### Task 3: 更新 QWEN_TOOLS

**文件**: `memory/manager.py`

```python
remember_fact:
    fact: str      # 中文短句，如"喜欢猫"
    replaces: str  # 可选，替换含此关键词的旧记忆
    name: str      # 可选，仅用户自报姓名时传

forget_fact:
    keyword: str   # 中文关键词模糊匹配

# clear_memory / confirm_clear 不变
```

### Task 4: Session Consolidation — 会话后复盘

**文件**: `voice/d01_realtime_chat.py`

替代原 `_save_conversation_summary`，改为 `_consolidate_session`：

```python
def _consolidate_session(pid, conv_log, current_facts, current_name):
    """后台线程：会话结束后 LLM 复盘，生成 entity + episodic memory。"""

    transcript = format_transcript(conv_log)

    # 一次 LLM 调用同时生成两层记忆
    prompt = f"""你是记忆管理助手。根据以下对话内容和已有记忆，生成更新后的记忆。

已有记忆：{json.dumps(current_facts, ensure_ascii=False)}
当前用户名字：{current_name or "未知"}

对话内容：
{transcript}

输出JSON：
{{
  "name": "用户名字(如果对话中提到或更正了名字则更新，否则保留原名)",
  "facts": ["关于这个人的事实短句列表，合并已有记忆和对话新信息，去掉过时的"],
  "episode": {{
    "topic": "一句话说这次聊了什么",
    "highlights": ["关键信息点"],
    "mood": "engaged/casual/emotional/tense"
  }}
}}
只输出JSON。"""

    resp = oai.chat.completions.create(model=SUMMARY_MODEL, ...)
    result = json.loads(resp)

    _memory_mgr.consolidate_facts(pid, result["facts"], result.get("name"))
    _memory_mgr.save_episode(pid, result["episode"])
```

**关键点**：
- 输入 = 全量对话 transcript + 当前 facts（含会话中 draft notes）+ 当前 name
- LLM 做合并/去重/去过时，输出干净的 facts list
- 同时提取 episodic memory（事件，不是摘要）
- 一次 LLM 调用完成两层记忆的生成

### Task 5: 迁移现有数据

**文件**: `memory/manager.py` (load_memory 中自动迁移)

检测旧格式（`facts` 是 dict）→ 自动转换：

| 旧 | 新 |
|---|---|
| `name: "大大"` | 顶层 `name` |
| `is_owner: "true"` | 丢弃（在 owner.json） |
| `likes_X: "true"` | 翻译映射表转中文短句 |
| `job: "X"` | `"是X"` |
| `weather_*` | 丢弃（临时信息） |
| 其他 `k: v` | `"k: v"` fallback |
| `conversation_summaries` | → `episodes`（text→topic） |

### Task 6: d01 工具调用适配

**文件**: `voice/d01_realtime_chat.py` (~L442-482)

- `remember_fact`: `(key, value)` → `(fact, replaces?, name?)`
- `name` 参数同步 FaceDB.set_name + OwnerManager.try_claim
- `forget_fact`: `key` → `keyword`
- close_session: 调 `_consolidate_session` 替代 `_save_conversation_summary`
- `identity_injected` 重置逻辑不变

## 不在本次范围

- Semantic Memory（需跨 session 的 Consolidation Engine）
- Self Model / Narrative Memory / World Model / Procedural Memory

## 验证

1. 启动 → auto_merge 后 memory 文件正确合并
2. 现有 4 个 memory 文件自动迁移新格式
3. 对话说"我喜欢吃火锅" → draft fact 立即写入 `"喜欢吃火锅"`
4. 说"我不喜欢火锅了，喜欢烧烤" → replaces 替换生效
5. 说"忘掉烧烤" → keyword 模糊匹配删除
6. 结束对话 → consolidation 生成最终 facts + episode
7. 检查 consolidation 后的 facts 是否合并了 draft notes + 对话新信息 + 去掉过时项
8. 重新对话 → prompt 注入自然语言
