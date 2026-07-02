# ASR 级联重构方案 —— Omni 从"语音直入(S2S)"改为"文本入(带说话人)+ 中和音频"

> 状态:设计定稿(2026-07-02),已做真机 spike 去风险,待分阶段实现。
> 起因:多人长时聊天的上下文注入 / 张冠李戴,`update_session` 与 `create_item` 两条注入路线都压不住"历史惯性"和 `semantic_vad` 抢跑时序。
> 关联:[[CONTEXT_INJECTION_REDESIGN]](本方案取代其"注入通道"结论)、bug-067/069/070/071。

---

## 1. 为什么换架构

现状是**纯 S2S**:麦克风 16k PCM → `conv.append_audio()` → Omni 服务端 `semantic_vad` 断句 + 转写 + 生成 + TTS。问题根源有二,都源于"我们无法控制 Omni 看到的文本":

1. **历史惯性**:`update_session` 改的是 system 指令,**改不了已入历史的旧用户/助手轮**。人 A 说过"我叫大大"后,这条留在历史里,人 B 再问"我是谁"时模型顺着历史答"大大"。
2. **时序抢跑**:`semantic_vad` 在 `transcription.completed` 之前就自动建回复,注入(create_item)常常晚于生成 → 模型没参考到。

**根治思路**:让 Omni **只读我们给的文本**——每条用户轮从源头带上说话人标签,易变记忆走单次回复指令。这样多人天生可区分,且无时序竞态。

---

## 2. 真机 Spike 结论(2026-07-02,实测)

用裸 websocket 直连 `wss://dashscope.aliyuncs.com/api-ws/v1/realtime` 验证两点:

| 试验 | 结果 |
|---|---|
| 纯文本 item + `append_video`、**完全不送音频** | ❌ 服务端报 `invalid_request_error: Error append image before append audio` |
| 送**静音音频**锚点 + `turn_detection=null` + 转写关,再 `append_video` + 纯文本 item + `create_response` | ✅ 模型准确描述画面(合成图"绿底紫圆白7"→回复"画面中间是一个紫色的圆形,背景是绿色的,上面有白色数字7") |

**硬约束**:Qwen-Omni-Realtime 的视频帧必须挂在音频 buffer 之后,**不送音频就不能送视频**(与 `d01:2024` 注释一致)。
**结论**:不能"移除音频",但可以"**中和音频**"——音频只当视频载体,Omni 既不转写也不自动回复它;回复完全由我方文本手动驱动,且视频照常被参考。功能上等价于"Omni 只按我们给的文本+看到的画面回答"。

> 复现要点(在 Claude 环境跑 dashscope):SDK 的 `WebSocketApp.run_forever()` 会读 Windows 系统代理(`getproxies()`),被环境注入的代理挡死(5s 超时)。裸连时传 `websocket.create_connection(url, http_proxy_host=None)` 绕过即可;真机启动脚本环境无此代理,故 d01 正常。

---

## 3. 目标架构

```
麦克风16k PCM ─┬─[方向门控+身份闸门](保留)─┬─▶【连接②ASR】Recognition ─▶ sentence_end(+begin/end ms)─┐
               │                            └─▶ ASD.feed_audio(谁在说,不变)                            │
               │                                (回声门:播放中不喂 ASR)                                 │
静音trickle ───┼──append_audio──▶┐                              轮次聚合(更长静音 or 说话人切换才触发) │
视频帧 ────────┴──append_video──▶┤【连接①脑】Omni                                                       │
                                 │ (turn_detection=null,转写关闭)                                      │
                                 └──◀ create_item(role=user, content="「大大」:<transcript>")◀──────────┘
                                    create_response(instructions="现在跟你说话的是大大,你记得TA喜欢西瓜、露营;画面里还有别人别搞混。")
                                 ──▶ response.audio.delta 24k→16k→抖动缓冲→播放(不变)
                                    response.audio_transcript.delta → 显示/conversation_log
```

### 3.1 Omni 会话配置(建连一次)
- `modalities=[text, audio]`(audio 输出 = TTS,不变)
- `input_audio_transcription = None`(**关**:不用 Omni 的 ASR)
- `turn_detection = None`(**关**:Omni 永不自动回复)
- `instructions` = 基础人设(system,稳定不变的部分)
- `tools` = 原 8 动作 + take_snapshot + identify_pointed_object(不变,`create_response` 仍能触发 function_call)

### 3.2 上行:麦克风 → 两个消费者(音频不进 Omni 内容)
- mic 16k PCM →(现有方向门控 + 身份闸门保留)→
  1. **独立 ASR 连接**(见 §3.2.1):喂**裸 PCM 字节**(`pcm16.tobytes()`,即现在 base64 前那份)。
  2. **ASD.feed_audio**(不变)。
- **静音 trickle → 主 Omni `append_audio`**:仅作视频锚点(稳定低速静音 PCM,保证 `append_video` 被接受);此路 Omni 不转写、不据其回复。需周期 `input_audio_buffer.clear` 防 buffer 膨胀(别在 `create_response` 前清掉当前帧锚点)。

> **为什么 ASR 必须是独立的第二条连接**:若把真实音频喂主 Omni 脑连接并开 `enable_input_audio_transcription`,用户音频会变成**无说话人标签的 history user item** → 又回到张冠李戴。所以真实音频只进 ASR 连接;主 Omni 只收静音锚点 + 我方带标签文本。**两条 DashScope 连接**。

### 3.2.1 ASR 选型(两条官方路线,推荐脚本1)

| | **脚本1:`dashscope.audio.asr.Recognition`(推荐)** | **脚本2:`OmniRealtimeConversation` 转写模式** |
|---|---|---|
| 模型 | `fun-asr-realtime` / `paraformer-realtime-v2` | `qwen3-asr-flash-realtime` |
| 喂音频 | `recognition.send_audio_frame(pcm_bytes)`(裸 PCM) | `conv.append_audio(b64)` + `TranscriptionParams` |
| 断句 | `on_event`→`get_sentence()`→`text`,`is_sentence_end(sentence)` | `...transcription.text`(partial:text+stash)/`...completed`(final) + `speech_started/stopped` |
| 时间戳 | ✅ `begin_time/end_time`(词级) | 不直接给 |
| 热词 | ✅ `phrase_id`(`VocabularyService`/`AsrPhraseManager`) | corpus 偏置 |
| 其它 | `disfluency_removal_enabled`、`semantic_punctuation_enabled`、时延指标 | 复用现有 Omni 类 |
| 连接 | 独立 WS(`base_websocket_api_url=.../api-ws/v1/inference`) | 另起一个 Omni 会话 |

**推荐脚本1(`Recognition`)**:①最专、面最小;②有时间戳 → §3.5 `speaker_window` 精确对齐;③成熟热词 → 灌在场人名降识错;④独立生命周期,重启 ASR 不动脑连接。`fun-asr`(更新/中文更准)vs `paraformer-v2`(热词/时间戳成熟)真机 A/B。脚本2 作"复用 Omni 基建、少引依赖"的备选(且自带 speech_started/stopped 作 VAD 边界)。
**热词动态更新**:有人被命名后,用 `VocabularyService` 刷新在场人名词表(`phrase_id`),周期更新。

### 3.3 断句 = ASR 端点检测(这就是"VAD 不变"的落点)
VAD 从 Omni 服务端挪到 ASR 端点检测(`is_sentence_end`),行为同构(静音断句)。灵敏度按所选模型的静音端点参数调。

### 3.4 轮次聚合(触发 create_response 的粒度)
收集连续 `sentence_end` 片段,满足任一即触发一轮:
- 轮末静音 > 阈值(初值 ~1.2s);或
- 说话人切换(ASD 归属变了);或
- 缓冲超过最大时长。
→ 避免一个人中途停顿被切成两轮导致机器人抢答;贴近原 turn 语义。

### 3.5 一轮触发时(核心)
1. `speaker = ASD.speaker_window(turn_start_mono)` → `pid` → `name`/`facts`(memory)。`turn_start_mono` = 本句首个 partial 到达时打的 `monotonic` 戳(简单稳妥);ASR 的 `begin_time` 可作精修对齐。
2. `text = "「大大」:<transcript>"`(未命名 → `"「访客A」:…"`)。
3. `create_item(role="user", content=[{input_text: text}])` —— **稳定说话人标签入历史**。
4. `create_response(instructions="现在跟你说话的是大大,你记得TA…;别和别人搞混")` —— **易变记忆/消歧只作用本轮,不进历史**。
5. barge-in:ASR 侦测到 partial 起始且机器人在说 → 复用现有 `_do_barge_in`(停播 + `cancel_response`)。注意与 §5 回声风险联动:播放期间的 ASR 命中须先过回声门,超阈值才算主动打断。

### 3.6 下行(完全不变)
- `response.audio.delta` 24k → 16k → 抖动缓冲 → 播放。
- `response.audio_transcript.delta` → 显示 + `conversation_log`(用于 consolidation/记忆)。

---

## 4. 为什么这能根治张冠李戴

1. **源头标签**:每条用户轮进历史时就带 `「name」:`,模型天生区分多人;`update_session` 做不到的"重标历史"现在免费获得。
2. **易变记忆隔离**:名字/记忆走 per-response `instructions`,**不持久化进历史** → 无跨轮串味。
3. **无时序竞态**:先集齐文本+说话人,再手动 `create_response`,顺序确定。
4. **归属精确**:ASR 时间戳 + `speaker_window` 对齐同一时间窗。

---

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 视频需音频锚点(已确认) | 静音 trickle 锚点 + 周期 clear(§3.2),已 spike 通过 |
| **回声/AEC(级联新引入)**:独立 ASR 会把机器人自己的 TTS 转写成假轮次(S2S 时 Omni 内部处理了) | 播放期间**门掉 ASR 输入**(仅 `not playing`/`playback_end_estimate` 外喂帧);播放中超阈值说话才当主动打断(§3.5 barge-in)。真机调 |
| 级联延迟(端点等待 + ASR RTT ~几百 ms) | 轮次聚合阈值可调;人名热词减少重试;可接受(换正确性) |
| 人名 ASR 易错 | `phrase_id` 热词灌在场人名(命名后 `VocabularyService` 刷新);必要时对短确认语二次校正 |
| 双人同时说话 ASR 混一条 | 门控已缓解;ASD 取占优说话人,承认为固有限制 |
| 静音 buffer 膨胀 | 周期 `input_audio_buffer.clear`,避开 create_response 时刻 |
| Omni 仍偶发据静音回复 | `turn_detection=null` 已阻断自动回复;create_response 只在我方触发 |

---

## 6. 分阶段实现

- **阶段0**:抽出 `OmniTextDriver`(封装 create_item/create_response/静音锚点/视频),不改现有 S2S 行为,先旁挂验证静音锚点在真机连续帧下稳定。
- **阶段1**:接入 Paraformer 流式 ASR(独立线程,消费与 ASD 同一路 mono),打通 `sentence_end` + 时间戳 + 轮次聚合,先只打日志不驱动回复。
- **阶段2**:切换驱动——停止把真实音频送 Omni(改静音锚点),`turn_detection=null`、转写关;由轮次聚合触发 `create_item(带标签)+create_response(带记忆)`。下行/工具/视觉不变。
- **阶段3**:barge-in(ASR partial 起始)、热词表、buffer clear 节流、参数真机调优。

---

## 7. 保持不变的部分(明确边界)

TTS 下行、视频流(`append_video`,现改为静音锚定)、KWS 唤醒、ASD、ReID、人脸跟随、动作/看图工具、记忆落盘与 consolidation —— **均不变**。变的只有:Omni 的**输入从音频改为文本+说话人**,以及断句/回复由我方驱动。
