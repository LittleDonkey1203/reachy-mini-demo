# 上下文注入重构方案 —— 从"单一当前说话人"到"在场名单"

> 状态:设计定稿(2026-07-01),待分阶段实现。
> 起因:多人长时聊天后出现"注入身份错乱 / 模型说'看不见你'(实际看得到)"。
> 关联:[[COGNITIVE_MEMORY_ARCHITECTURE]]、bug-069/070、`voice/realtime.py` 注入链。

---

## 1. 现象与真机证据

多人轮流跟小艺聊,一段时间后模型反复回复「哎呀,我看不太清你的脸,所以不太确定你是谁呢」——**尽管人整个在画面里、dashboard 稳定标着正确名字(大大)**。

`log/main.log`(2026-07-01 16:27–16:28)逐行还原:

```
85  🧠 记忆已注入 (Unknown-1 pid=id_17828 present=1)   ← 注入了真身份(此刻名字还是 Unknown-1)
88  💭 tsp=None inj=id_17828                            ← ASD 没把这句归给这个 track → tsp=None
91  🔁 补注入(done空闲) neu → neutral                   ← ★bug-070 A 把真身份翻成"中性"★
92  🫥 注入中性上下文(不认识对方)                        ← 会话从此进入"看不见你"模式
157 「你还记得我…我叫啥」→ 归属 画外 → inj=_neutral
163 💬 哎呀,我看不太清你的脸,不确定你是谁            ← 第一次说"看不见"(中性上下文下)
188 「你记得我」→ 归属 大大 → 189 注入大大 → 191 tsp==inj==大大(注入终于对了)
193 💬 还是"我看不太清你的脸…"                         ← 注入对了也没用:顺着历史惯性继续说
```

关键点:**191 行 `tsp==inj==大大`,注入完全正确,模型照样说"看不见"。** 排除了"注错人 / 归属错 / busy 冻结"三种嫌疑。

---

## 2. 根因(三层,与"边缘脸"无关)

1. **ASD 归属跟不上视觉**:唤醒/短句时 ASD 没给该 track 打分 → `tsp=None` / 归属"画外",**尽管视觉稳稳看着人(present=1、已识别)**。注入/归属**只看 ASD,无视了视觉已经"看见且认得人"**。
2. **bug-070 A 火上浇油**:done-空闲补注入一看 `tsp=None` 就把**刚注入的真身份翻成中性"看不见你"**——在人明明在画面里时。这是把会话推进"失明"模式的**直接扳机**。
3. **保留会话 → 历史污染**:一旦说出"看不见你",`update_session` 只换 system 指令、**换不掉已说出口的历史**;后面即使正确注入,模型顺着历史惯性继续说。

### 更深一层:多人场景下"实时单人归属"本身不成立
多人快速轮流说话时:
- ASD **说完才出分**,而 `semantic_vad` **说停就自动建回复** → 生成早于归属完成。
- "换人"无法可靠检测(轮流说、ASD 滞后)。

⇒ **只要坚持"必须先算出单个当前说话人才能注入",在多人场景就永远追不上。** 换注入通道(system prompt→上下文追加)只能"注得及时些",**治不了"来不及知道注谁"**。

---

## 3. 现有 SDK 的三条注入通道(实测)

`dashscope.audio.qwen_omni.OmniRealtimeConversation`:

| 通道 | 调用 | 本质 | 适合 |
|---|---|---|---|
| **A. 改 system prompt** | `update_session(instructions=...)`(现用) | 重建会话级指令,对**后续所有**回复生效 | **稳定**内容:人设/工具/安全 |
| **B. 追加上下文条目** | `create_item({"type":"message","role":"system","content":[{"type":"input_text","text":...}]})` | 往对话时间线塞一条消息,**不动会话配置** | **易变**内容:在场名单+记忆 |
| **C. 单轮指令** | `create_response(instructions="...")` | 指令**只对这一次回复**生效 | 逐轮身份,最干净 |

- `create_item` 是 `conversation.item.create` 的薄封装(现已用于 `function_call_output`),**换 role 即可塞 system/user 消息**。
- `create_response(instructions=)` 参数存在,但**前提是收回 turn-taking**(否则和 semantic_vad 自动回复撞车 = bug-067)。
- A 的副作用:重置会话 → bug-069(图像先于音频)、reinject 抖动;且**改指令救不回历史**。

---

## 4. 目标架构

### 核心转变
> **别再"实时算出单个当前说话人再喂给模型"。**
> - **READ(注入)= 在场名单(roster)**,交给多模态模型自己归属;
> - **WRITE(存记忆)+ 转头 = 事后 ASD**,容忍延迟。
>
> 归属就此**离开生成的关键路径**,ASD 快不快都不再卡注入。

### 4.1 partner 状态 = 在场名单(集合,非单值)
新增一个感知融合出口(在 `perception/fusion.py` / 新 `partner` 层),输出:

```
roster = [
  { pid, name, presence: 已识别/看得见未识别, memory_brief, since, zone(左/中/右) },
  ...
]
```
- 融合 **视觉(present/identity,已有 IdentityStore/_head_key)** 为主 + ASD/DOA 为辅。
- **presence 以视觉为准**:"看不见对方"**只有 roster 为空时才成立**;ASD 无权判"有没有人",只负责"在场的哪位在说"。
- **迟滞去抖**:人来人走是秒级事件,不随单帧 ASD 抖动。

### 4.2 READ:注入 roster,让模型归属(通道 B)
换人时/名单变化时,用 **通道 B** 追加一条 system 消息:
```
【在场】现在画面里有:大大(记忆:爱吃火锅、程序员)、小坤(记忆:…)。
你能看到画面也能听声音,根据谁在对你说话来称呼;
拿不准就礼貌问一句"是大大还是小坤?",绝不张冠李戴、也绝不说看不见。
```
为什么扛多人:
- 注入**不依赖实时单人归属** → 无竞态(roster 秒级才变)。
- 归属交给**信息最全的一方**(模型有 音频+视频+对话流)。
- 失败优雅:模型拿不准 → 反问,远好过自信答错 / 说"看不见"。
- 今天那个 bug 直接消失:roster 非空,模型不可能说"看不见你"。

### 4.3 WRITE + 转头:事后 ASD(不追实时)
只有两件事真需要"单个人",且**都不在生成关键路径、都容忍延迟**:
1. **记忆写入**("我喜欢X"存对人)→ 用**说完之后**的 `speaker_window`(ASD 此时最准)。
2. **机器人转头看谁** → 行为层,晚几百 ms 无感。

### 4.4 (可选)让回复"等"归属:收回 turn-taking + 通道 C
若某些场景要让**回复本身**等归属确定:说话停了先别自动答 → 等 ASD 对整句出分 → `create_response(instructions=该人记忆)`。**生成等归属,而非抢归属**,"追不上"从定义上消失。代价:多几百 ms 延迟 + 自管 VAD。

### 4.5 system prompt 冻结 + 历史污染治理(端侧账本 + delete 兜底)
- **system prompt 只放不变的**(人设/工具/安全),设一次不再动。
- 历史污染:**源头别让它说错**(4.2 已治大半);需彻底清再叠 `conversation.item.delete`。**不采用 PR#10 的 close+reopen 全量重建**(丢会话连续性)。

**端侧 item 台账(delete 的前置,必须先做)**
- delete 只能按 **server item_id** 删;id 由服务端发。SDK **无 delete 方法**,用 **`send_raw`** 发原始 `conversation.item.delete`(和 `create_item` 同套路)。
- **现状缺口**:代码只记了 `call_id`,`display_transcript` 用自家 `seq` ≠ server id → **现在删不了**。
- **要建的账本**:给 `display_transcript` 每条补 `item_id`,来源事件 = `conversation.item.created`(user/我们注入的项)、`response.output_item.added`/`response.done`(assistant 项)。台账每条记 `{item_id, role, type, text, tag(在哪份 roster 下生成/是否污染)}`。

**何时能调 delete(时机三铁律)**
1. **绝不在生成中删** —— 只在**空闲**(`in_flight==0`,`response.done` 之后)。
2. **要影响下一轮,须在下一个 `response.create` 之前删** —— 但 `semantic_vad` 说停就自动建回复,这个空档极短、和注入同一个 race。**安全窗 = 上轮 `response.done` 之后、下句用户说话之前的真空闲**;**最稳 = 阶段 3 收回 turn-taking 后自己掌握空档**(先删再 `create_response`)。
3. **别过删** —— 优先删两类,真实对话主体留着:
   - **A. roster 保洁**:注入新 roster 时顺手删上一条旧 roster/身份 system 条目(我们自己的项,低风险,治"堆积")。
   - **B. 清污染轮**:`response.done` 后事后检测到污染(如 roster 非空却说"看不见")→ 在下句前空闲窗删那条 **assistant** 轮。

**注意**:①鸡生蛋——先有账本才能删;②best-effort——服务端可能拒删已引用/不存在项,要听 `error` 事件;③需先验证 Omni `conversation.item.delete` 事件 schema + 服务端是否真执行;④delete 不改已生成的回复,只影响之后的条件历史。

---

## 5. 失败模式 → 架构如何根治

| 观察到的 bug | 现补丁 | 目标架构怎么根治 |
|---|---|---|
| 人在画面却说"看不见" | 中性门控 | presence 以视觉为准;roster 非空即不会说看不见 |
| bug-070 A 翻真身份为中性 | 加护栏 | 注入只读稳定 roster,不看瞬时 tsp |
| 历史被"看不见"污染 | 补不了 | 源头不产生 + 可选 item.delete |
| 多人 busy 注入冻结 | bug-070 A | roster 秒级下发,不靠 in_flight 门 |
| ASD 追不上单人归属 | 无 | 注入用 roster,单人归属离开关键路径 |
| update_session 撞自动回复(bug-069) | v2 抑视频 | READ 走通道 B,不重置会话 |
| 模型信视频胜过注入文字 | prompt 兜底 | roster 已识别 → 不注入"看不清",明确以名单为准 |

---

## 6. 分阶段落地(不 big-bang)

- **阶段 0(止血,现在)**:`realtime.py` 三处小改,保证能继续复现/测试:
  - (a) bug-070 A 的 neu 分支加护栏:`present_count>0` 时**绝不**翻中性;
  - (b) 中性注入门控:`present_count>0` 时不走"看不见你",改"看得到人、没听清是哪位";
  - (c) `update_memory_neutral` 文案按 `present_count` 拆分。
- **阶段 1**:落 **roster 状态**(4.1),先**只读+打日志**和现有 ASD 并行对拍,确认它更稳。**并行建 item 台账**(给 display_transcript 补 server item_id,4.5)——delete 的前置,也利于调试。
- **阶段 2**:READ 切到 **通道 B 注入 roster**(4.2),退休 bug-070 A / 中性裸判;记忆写入改**事后** `speaker_window`(4.3)。**用上台账做 roster 保洁**(注入新 roster 删旧条,4.5-A,低风险)。
  - **✅ 已实现(2026-07-01,待真机)**:`RealtimeDialog.inject_context()` 走 `create_item`(role=system)注入「在场 roster(名字+facts)+ 带说话人标签的近史(每条按 pid 现查名)+ 规则」;客户端指定 item id、发新条前 `send_raw` 删旧条(保洁)。转写门先走 create_item,失败(Qwen 拒 system item)`_ctx_use_item=False` 回退 update_session。`st.roster` 由 vision 每帧写(d01)。**未做**:记忆写入改事后 speaker_window;招呼/reinject 仍走 update_session 回退。**风险**:create_item(role=system,含 client id)是否被 Qwen 接受,需真机日志确认(看 `📌` 成功 / `⚠ create_item 注入失败` 回退)。
- **阶段 3(可选)**:收回 turn-taking + 通道 C(4.4);用台账 + `send_raw` 发 `conversation.item.delete` 做**清污染轮**(4.5-B),在空闲窗执行。

---

## 7. 代价与开放问题

- **token**:在场 N 人注 N 份记忆 → 只注**摘要**(名字+≤3 条 fact);N 通常 2-3,可控。
- **模型归属非 100%**:靠"拿不准就问"兜底,地板远高于现状。
- **写入仍需归属对**:但事后算,不赶时间;误写用交叉人物检查(可摘 PR#10 `47b1372`)防漂移。
- **`item.delete`/`truncate` 需验**:确认 Qwen-Omni-Realtime 服务端是否接受该事件。
- **收回 turn-taking 的代价**:自管 VAD/端点检测 + 打断逻辑,复杂度上升;阶段 3 再评估是否值得。
