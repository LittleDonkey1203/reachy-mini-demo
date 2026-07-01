# Qwen-Omni-Realtime 工具调用调研报告

> 调研日期: 2026-06-26

## 一、调研背景

Reachy Mini 使用 Qwen3.5-Omni-Plus-Realtime 模型进行全双工语音对话。发现两个问题:

1. **TTS 标签泄漏**: 模型在文本输出中插入括号动作描述(如 `（点头）`/`(nods)`)，被 TTS 原样朗读
2. **end_session 工具不触发**: 用户明确说"退下"/"拜拜"但模型不调用 end_session 工具

## 二、核心结论

**工具 schema 和事件流均已正确，两个问题的根因在 omni 模型的固有特性 + prompt。**

## 三、工具定义格式

### 3.1 正确格式(扁平风格)

Realtime API 使用 **OpenAI-realtime 扁平风格**，通过 `session.update`(SDK 里 `update_session(tools=...)`) 传入:

```json
{
  "type": "function",
  "name": "end_session",
  "description": "结束本次对话...",
  "parameters": {"type": "object", "properties": {}}
}
```

项目中 `voice/config.py:BASE_TOOLS` 使用的就是这种格式。

### 3.2 注意事项

- 阿里官方文档示例混用了**嵌套 Chat-Completions 风格**(多一层 `"function": {...}`)，那是从非 realtime 的 function-calling 指南复制来的
- Realtime 实测可用的是扁平风格，不要改成嵌套
- `parameters` 即使为空也必须是合法 JSON-Schema 对象

## 四、工具调用事件流

正确的交互流程:

1. 服务端返回 `response.function_call_arguments.done`(含 `name`/`arguments`/`call_id`)
2. 客户端执行工具，通过 `conversation.item.create` 提交 `function_call_output`(`call_id` 必须精确匹配，`output` 为字符串)
3. **必须显式调用 `response.create`** 才能触发后续语音回复

项目中 `voice/realtime.py` 的 `_record_tool_output` 流程完全符合。

## 五、关键限制

### 5.1 无法强制工具调用

- **`tool_choice` 和 `parallel_tool_calls` 均不支持** — 文档明确说明
- 工具调用**完全由模型自主决定**，没有 `required` 这个杠杆
- 这意味着 `end_session` 不被调用时，**无法通过 API 参数强制**，只能靠 prompt 提升触发率

### 5.2 Web Search 与 Tool Calling 互斥

- 用工具时必须确保 `enable_search` 关闭

### 5.3 括号动作描述是固有特性

- 括号动作描述(如 `（点头）`/`(nods)`) 是 **omni 端到端语音模型的已知固有特性**，不是可修复的 bug
- 根因: omni 是端到端语音模型，文本通道训练数据里大量暴露情绪/动作标注
- **没有任何 API flag 能抑制括号**，只能靠 `instructions` system prompt 处理
- 官方设计意图: 情绪应通过**语气/韵律**表达，而非括号文字

## 六、应对方案

### 6.1 Prompt 层(唯一 API 杠杆)

因为没有 `tool_choice: required`，两个问题都只能从 prompt 解决:

#### 防括号泄漏

在 instructions 中明确禁止:
- "需要表达动作时必须调用对应工具（nod/shake_head...），绝不要在文字里输出括号动作描述"
- "你的文字输出只能包含要说的话，不能包含任何动作描述、情绪标注、舞台指示"
- 列出具体的禁止格式: `(点头)` `（微笑）` `<nod>` `*点头*`

#### 工具触发增强

工具描述采用**正例/负例触发词**模式:
- 正例: 具体的用户话术("退下"/"拜拜"/"先这样"/"不聊了")
- 负例: 容易混淆的非触发词("再说吧"/"等会儿"/"先放着")
- "拿不准时继续对话，不要调"

#### 情绪表达

- 引导模型用语气和措辞表达情绪，而非任何标注
- "开心就用活泼的话，难过就用低沉的话"

### 6.2 已知标签 → 物理动作兜底

即使 prompt 禁止，omni 模型仍可能在音频中念出括号动作。由于音频是端到端生成的，**无法在 TTS 前拦截**。但 transcript 文本中仍能捕获这些标签，用于触发物理动作:

```python
# 已知动作标签 → 转为物理动作(至少让机器人做出来)
_ACTION_TAG_RE = re.compile(r"</?(?:nod|shake|...)>|[（(](?:点头|摇头|...)[)）]|...")
```

注意: 这**不能阻止括号被念出来**，只是在检测到已知标签后触发对应的物理动作作为补偿。

### 6.3 不推荐的方案

**宽泛正则删除(BROAD_TAG_RE)** — 尝试清除所有短括号内容:
- 无效: omni 模型是端到端生成音频，括号已在音频流中被念出，文本层删除毫无意义
- 有害: 容易误删正常文本内容(如 `(1)`, `(对)`)
- 已回退

### 6.4 update_session 的局限(身份切换)

- `update_session(instructions=...)` 只替换系统指令，**不清除 conversation items**
- 身份切换后，旧对话提到的爱好/信息仍在模型上下文中 → 记忆污染
- 解决: 身份切换时 close + open_session 重建干净 WS 会话

## 七、情绪/语气控制

### 7.1 结论

**没有独立的情绪/语气 API 参数。** 情绪控制完全靠自然语言 prompt 驱动。

### 7.2 SDK 验证

`update_session` (L227) 只构建: modalities, voice, input/output_audio_format, input_audio_transcription, turn_detection, translation, sample_rate。没有 emotion/style/prosody/speed/volume/pitch 键。

`create_response` (L459) 只接受 `instructions` + `output_modalities`。

### 7.3 两种控制方式

| 方式 | 用法 | 适用场景 |
|------|------|----------|
| 全局语气 | `instructions` 中写 "用开心、活泼的语气回答" | 机器人整体人设 |
| 局部语气 | `create_response(instructions="用兴奋的语气说...")` | 特定回复的情绪 |

模型也支持用户语音指令实时调整：用户说"说快一点"/"大声一点"/"开心一点"，模型会跟随。

### 7.4 voice 参数

- 可用音色: Cherry/Ethan/Serena/Chelsie 等，最新模型支持 55 种(47 多语言 + 8 方言)
- 不支持情绪变体(如 `"Ethan-happy"`)
- 任何音色都能表达情绪，范围由 instructions 决定而非音色 ID

### 7.5 对 TTS 标签泄漏的意义

括号动作描述(如 `（点头）`) 是模型用**文本通道**表达动作/情绪的方式。官方设计意图是用**语气/韵律**替代，即 prompt 中引导"需要表达开心时用活泼的语气说话"而非"在文字里写（开心）"。

当前 `voice/config.py` INSTRUCTIONS 已包含相关指令：
- "需要表达情绪时用语气和措辞，比如开心就用活泼的话，难过就用低沉的话"
- 明确禁止所有括号/星号/XML 动作描述

### 7.6 不推荐的路线

- **Qwen3-TTS-VD** (Voice Design): 独立的 TTS 模型，支持结构化情绪控制(8种情绪+6种风格)，但与 Realtime 模型无关，不能混用
- 第三方声称的 "8 emotions + 6 styles" 参数: 未在官方 SDK 中验证，不可靠

## 八、参考来源

- [实时 Qwen-Omni-Realtime](https://help.aliyun.com/zh/model-studio/realtime) — 官方文档
- [Client/Server Events](https://www.alibabacloud.com/help/en/model-studio/client-events) — 事件详解
- [Function Calling](https://help.aliyun.com/zh/model-studio/qwen-function-calling) — 通用工具调用
- [QwenLM/Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) — GitHub 仓库

## 八、项目相关文件

| 文件 | 用途 |
|------|------|
| `voice/config.py` L243-273 | `BASE_TOOLS` 工具定义(扁平风格) |
| `voice/config.py` L21-52 | `INSTRUCTIONS` 系统指令(含 TTS 禁令) |
| `voice/realtime.py` L55-65 | `_ACTION_TAG_RE` + `_BROAD_TAG_RE` 正则清洗 |
| `voice/realtime.py` L338-350 | 标签泄漏兜底处理(transcript.done 事件) |
| `voice/realtime.py` L413-455 | `open_session` WS 连接 + update_session |
| `voice/realtime.py` L491-535 | `update_memory` 记忆注入(仅改指令，不清历史) |
