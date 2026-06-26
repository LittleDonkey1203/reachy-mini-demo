# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-06-23

## User Preferences

<!-- How the user likes things done. Code style, tools, patterns, communication. -->

- 不要新增 tab/按钮来展示调试信息，优先复用已有的 UI 元素（如 Conversation 面板的 payload modal）
- 闸门/门控机制不能过于敏感：只在明确的切人+声源大幅变化时才生效，避免常规场景误触发
- Dashboard 调试功能要能看到"模型看到什么" — session instructions、memory prompt、conversation log 必须可视化
- 遵循 CLAUDE.md 规则：每次改动后必须更新 wolf 文件（cerebrum/memory/anatomy/buglog）

## Key Learnings

- **Project:** Reachy Mini Lite 语音交互机器人(USB 版)，Qwen3.5-Omni-Realtime 驱动
- **运行环境:** macOS Intel + Python 3.12 + dashscope SDK
- **视觉后端:** SCRFD/InsightFace 默认(2026-06-26 迁移,关键点更稳), YuNet/MediaPipe 可选(FACE_BACKEND 切换), ArcFace 身份识别;新 ReID 链路 = SCRFD检测 + ByteTrack + 三区间 IdentityStore(gallery.json)
- **语音协议:** Qwen-Omni-Realtime WebSocket, update_session 做记忆注入(整体替换), 非 create_item(只增不删)
- **状态机:** 9 态 FSM, TRACKING 态是核心对话状态, 方向门控只在此状态生效
- **记忆存储:** 认知记忆架构(Entity Memory + Episodic Memory + Working Memory 注入)
- **Entity Memory:** per-person JSON facts (`data/memories/<pid>.json`), `dict[str,str]` KV 格式 + `summary` 叙事
- **Session Consolidation:** 会话后 LLM(SUMMARY_MODEL) 从全量对话+当前facts KV 生成最终 entity dict + summary + episodic memory
- **清华镜像:** UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ 所有 pip/uv 安装必用
- **Realtime function-calling:** flash-realtime 触发可靠性差(OmniGAIA flash≈33.9 vs plus≈57.2),会把工具"说成文本"不发 function_call → 记忆/动作丢失;记忆/动作场景必用 plus。Qwen-Omni-Realtime 不支持 tool_choice/parallel_tool_calls,无法强制调用。诊断:日志看 🤖模型调用工具/🧠记忆工具/👑认主成功 三标记;动作全走"标签泄漏兜底"=模型在文本化工具。realtime.py:110 的"已注册"日志是写死文本,不反映真实 tools payload。

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

- [2026-06-24] **clear_memory confirmed 参数不能删**: 用户明确要求保留 confirmed 守卫("如果用户确认了才可以删")。不要因为"简化流程"移除安全确认参数。
- [2026-06-24] **记忆 = 人脸 + 事实**: clear_memory 必须同时清除 face_db 中的 person entry(`clear_person(pid)`) 和 memory facts。不能只清 facts 而留人脸。
- [2026-06-24] **state.json 字段必须完整**: 前端 JS 引用 `s.is_owner` 时，后端 state dict 必须同步添加该字段，否则前端永远显示空。添加 Dashboard 功能后检查数据通路: State class → _build_frame/state dict → JS 渲染。
- [2026-06-24] **不要新增 tab 展示调试信息**: 用户明确要求复用已有的 Conversation 面板 payload modal，不要加新的 tab 或按钮。
- [2026-06-24] **音频闸门不能每次 close_session 都触发**: 只在二次唤醒切人且 DOA 声源方向大幅变化(>SWITCH_AWAY_DEG)时关闸，避免常规断连重连时误拦截音频。
- [2026-06-24] **必须遵循 CLAUDE.md 更新 wolf 文件**: 每次代码改动后必须更新 memory.md(行为日志)、cerebrum.md(学习)、anatomy.md(文件描述)。用户会检查。

- [2026-06-25] **门控逻辑用白名单不用黑名单**: 只在 TRACKING 时关门(有人脸在面前对话)，其他状态一律放行。黑名单逐个豁免状态容易漏，且新状态默认被关门导致静音断连。
- [2026-06-25] **d01 重构要领域驱动**: 移动代码时按领域归属(语音/记忆/感知)分配模块，不按"从哪里提取"机械分配。ChatCallback + 闭包 → 完整对话协议层(realtime.py)，不是"callback.py"。
- [2026-06-25] **多线程 flag 消费顺序**: 清除 flag 必须在写入后续 state 之后(同一锁或之后的锁内)，否则其他线程在 flag=False + state=旧值 窗口误判。不要用超时阈值修补竞态——调整操作顺序消除窗口。
- [2026-06-25] **conv=None 时 KWS 唤醒不能丢弃**: audio loop 在非 ARMED 状态 conv=None 时必须处理 KWS 命中(重连 WS)，否则 WS 意外断连后永远唤不醒。

- [2026-06-26] **诊断"记忆没写入"先看工具有没有被调用**: 不要直接跳到持久化链(save_gallery/flush)。先看日志 🤖模型调用工具/🧠记忆工具/👑认主成功 三标记是否出现——这局根因是 flash 模型根本没发 function call(把"我记住啦"说成文本),持久化 bug 是次级。记忆/动作场景别用 flash-realtime,用 plus。
- [2026-06-26] **realtime.py:110 的"已注册"日志不可信**: 它是写死文本,漏列 end_session 和 4 个记忆工具,不反映真实 update_session(tools=...) payload。别拿它当"工具没注册"的证据。真实注册看 d01:175→1544→1561 + realtime.py:388/463。

## Decision Log

<!-- Significant technical decisions with rationale. Why X was chosen over Y. -->

### YuNet 后端切换 (2026-06-23)
- `FACE_BACKEND` 环境变量控制人脸后端: `yunet`(默认) / `mediapipe`
- YuNet 不提供 blendshapes(smile/frown = 0.0)，MediaPipe 有
- YuNet 用 BGR 输入(`cv2.cvtColor`)，需从 RGB 转换
- YuNet `FaceDetectorYN` 在分辨率变化时需重建实例
- `start_mac.sh --face-mp` 设置 `FACE_BACKEND=mediapipe`
- YuNet 手部检测时间戳独立于人脸(不需要共享单调时钟)

### Face DB 碎片化修复 (2026-06-24)
- 问题: 同一人因角度变化被注册为多个 ID(cross-sim 低至 0.27, 阈值 0.35)
- `match()` 增加质心匹配(avg embedding), 取 max(单embedding, 质心) → 减少大角度漏匹配
- `update_embedding()` 移除 avg_sim 过滤, 改用 max_sim < 0.20 拒收(鼓励 embedding 多样性)
- `auto_merge(threshold=0.50)` 启动时扫描合并重复人(交叉相似度 > 0.50)
- 合并策略: 保留有名字/embedding 多的条目, 合并 embeddings 去重(sim > 0.90)

### 记忆权限 + 认主机制 (2026-06-24)
- `identity/owner.py` — OwnerManager, `data/owner.json` 持久化
- 认主: 第一个被 `remember_fact(name=xxx)` 的人自动成为 owner
- 权限: owner 可删任何人记忆, 非 owner 只能删自己的
- `auto_merge` 增加双命名保护: 两边都有 name 时跳过合并(防止误合并家庭成员泄漏记忆)
- `MemoryManager.__init__` 新增 `owner_mgr` 参数
- `handle_tool_call` 新增 `actor_pid` 透传权限校验
- GestureRecognizer 内含 HandLandmarker, 返回 landmarks + gestures
- 7 种模型手势: Closed_Fist/Open_Palm/Pointing_Up/Victory/Thumb_Up/Thumb_Down/ILoveYou
- 模型手势 score >= 0.6 时优先使用, 否则 fallback 到 _classify_gesture 规则
- 规则覆盖模型不识别的: three, four, ok
- 模型路径: models/gesture_recognizer.task (~8MB float16)

### 记忆注入过时修复 (2026-06-24)
- 问题: create_item(system message) 只增不删, 多人切换时旧记忆污染上下文
- 修复: 用 update_session(instructions=...) 替代, 记忆嵌入 session-level instructions
- update_session 需传完整参数(output_modalities/voice/audio_format/turn_detection/tools), 非增量更新
- session.updated 回调用 self.conv is None 区分初始配置 vs 记忆注入更新
- State 新增 identity_injected_pid 追踪当前已注入记忆的人

### 唤醒优先级修复 (2026-06-24)
- 问题: `_is_A` 门控用 DOA 方向判断"是否 A 自己又喊", B 站在 ±55° 内(常见)被误判为 A
- 修复: 替换为 `a_active = in_flight > 0 or speaking or user_speaking`
- 语义: A 正在说话/robot 在答 A → 屏蔽 B; A 沉默 → 放行 B 的唤醒
- State 新增 `user_speaking`, 在 speech_started/stopped 事件维护
- close_session 重置 user_speaking=False 防止跨会话泄漏

### TRACKING 身体跟随 (2026-06-24)
- 问题: TRACKING 态 body_yaw_deg 完全不更新, 人走到侧面 >23° 头卡住
- 修复: 视觉积分块中检测 neck_off > NECK_REL_LIMIT×0.7 时以 45°/s 转体
- BODY_FOLLOW_THRESHOLD=0.7, BODY_FOLLOW_SPEED_DPS=45.0 (config.py)
- 转体后重新 clamp track_yaw 到新的颈限范围, 头可继续追

### 人脸误识别稳定性 (2026-06-24)
- 问题: 低 sim 匹配(0.45-0.51)立即触发切人+记忆注入, 实际面前人没换
- 修复: 当已跟踪 A 时, B 要 "接管" 需: sim>=0.72 立即; sim<0.72 连续 3 次确认
- 首次识别也需 sim>=0.45(FIRST_DETECT_MIN_SIM), 防止低 sim 误注册
- ID_SWITCH_HIGH_SIM=0.72, ID_SWITCH_CONFIRM_N=3
- _id_switch_candidate / _id_switch_count 做连续匹配计数

### clear_memory 完整清除 + Dashboard 身份信息 (2026-06-24)
- clear_memory 清除链路: confirmed 守卫 → 权限校验(owner) → 清 facts → 清 face_db(clear_person) → 重置 State(pid/name/is_owner/injected)
- Dashboard "身份" 行: 显示 person_name + "✓记忆"(已注入) + "👑"(owner)
- 数据通路: State.current_is_owner → state dict `"is_owner"` → JS `s.is_owner`
- 内存概念: 记忆 = 人脸(identity/recognizer face_db) + 事实(memory/manager facts), 两者必须同步清除

### 安全删除工作流 (2026-06-24)
- 问题: clear_memory 敏感操作完全依赖大模型 confirmed=true, 无后端校验
- 方案: 两个 Tool(`clear_memory` 意图分类 + `confirm_clear` 最终确认) + 后端状态机 `st.clear_workflow`
- 工作流: VERIFYING(5s高阈值匹配) → PERMISSION(权限校验) → CONFIRMING(二次口头确认) → BACKUP → EXECUTE
- CLEAR_VERIFY_SIM=0.80: 远高于普通匹配阈值(0.35), 要求正面清晰人脸
- CLEAR_VERIFY_COUNT=3: 连续 3 次匹配(×2s间隔≈6s)
- clear_lock=True: 验证/确认期间阻止唤醒切换(防他人插嘴)
- 备份: 删除前自动备份到 `data/backups/`(face + memory), 支持手动回滚
- vision_result_loop 通过 cb_ref[0] 引用 ChatCallback 注入系统消息
- close_session 重置 clear_workflow/clear_lock 防跨会话泄漏

### 分人对话摘要 + 音频闸门 (2026-06-24)
- conversation_log 从 `list[tuple]` 改为 `dict[str, list]`，key=pid，按人分桶
- close_session 时提取当前人的对话桶，后台线程做 consolidation
- 音频闸门仅在二次唤醒切人+DOA 声源偏移>SWITCH_AWAY_DEG 时触发，不是每次 close_session
- 闸门关闭期间音频缓存为 b64 字符串，身份确认后 flush 全部缓存帧
- 上下文过长自动 consolidation: 每次 user transcript append 后估算 token(中文字数×1.5)，超过 CONV_SUMMARY_THRESHOLD(2000) 自动触发后台 consolidation+清桶
- consolidation 完成后如果仍在和该人对话，设 identity_injected=False 触发下一帧重新注入
- Dashboard: 事件 payload modal 增加 Session Instructions + Memory Prompt + Conversation Log 显示

### 认知记忆架构重构 (2026-06-25)
- 问题: facts 用 `{key: value}` dict，注入像机器码; conversation_summaries 是摘要不是事件; 记忆只靠实时 function call 无会话后复盘
- facts 格式: `list[str]` 中文短句 + `replaces` 关键词替换 + `keyword` 模糊删除(优于 dict 4 种方案)
- Entity Memory ≠ Semantic Memory: Entity 是直接提取的事实("喜欢猫"), Semantic 是从 episodes 抽象的知识("持续探索 AGI")(未实现)
- Episodic Memory: 结构化事件(topic/highlights/mood), 不是摘要。存"发生了什么"
- Session Consolidation: 会话后一次 LLM 调用同时生成 entity memory(consolidated facts) + episodic memory
  - 输入: 全量对话 transcript + 当前 facts(含 draft notes) + 当前 name
  - LLM 做合并/去重/去过时, 输出干净的 facts list + 结构化 episode
  - 兜底机制: 即使模型会话中漏调 remember_fact, consolidation 从全量对话捕获
- remember_fact 保留作为 draft notes: 会话中实时记录, 实时存盘, identity_injected=False 触发重注入
- 旧数据自动迁移: load_memory 检测 facts 是 dict → _migrate_legacy_facts 翻译映射表
- auto_merge → merge_memories: FaceDB.auto_merge 返回 {drop: keep}, d01 初始化遍历调用 merge_memories

### Facts KV 重构 (2026-06-26)
- 问题: list[str] 去重靠 replaces 子串匹配不可靠; 注入是扁平列举, 模型倾向背诵
- facts 格式: `dict[str,str]` KV, 同 key 自动覆盖, 精确更新
- 新增 summary: LLM consolidation 生成一句话叙事性认知描述, 注入时先放 summary 再列 KV 详情
- remember_fact: `(key, value)` 替代 `(fact, replaces)`, QWEN_TOOLS required=["key","value"]
- 注入格式: summary叙事 + KV详情(\n分隔) + episode topic + 使用指引
- 旧数据兼容: 自动迁移 list[str](推断key) 和英文key dict(翻译映射) 两种旧格式

### 方向门控白名单化 (2026-06-25)
- 问题: 黑名单门控(逐个豁免状态)导致唤醒时 DOA 残留关门 → 纯静音 → 服务端断连
- 修复: 改为白名单: `state != ST_TRACKING` — 仅 TRACKING(面前有人在对话)时屏蔽范围外声音
- 其他状态(ARMED/IDLE/ENGAGING/SEARCHING/RETURNING/POINTING/PLAYING)一律放行
- 原因: 只有锁定人脸且正在对话时才需要过滤其他方向的干扰; 其他状态要么没人、要么在找人, 关门只会送静音导致断连

### 多人脸 DOA 说话人选择 (2026-06-24)
- vision_worker 输出 all_faces: 所有 YuNet/MediaPipe 检测到的脸 [{u,v,h,box,kps}]
- _select_face_by_doa: DOA resid + body_yaw + track_yaw → 预期 u 坐标 → 匹配最近人脸
- 公式: doa_in_camera = (body_yaw + resid) - track_yaw; expected_u = 0.5 - doa_in_camera / FOV_X_DEG
- DOA 选出的脸优先用于 ArcFace 身份识别(替代 FaceSelector 选择)
- 仅在 all_faces > 1 且 doa_confident 时生效，单人脸或无 DOA 时 fallback FaceSelector
- debug overlay: 选中=蓝色粗框+DOA标签, 非选中=灰色细框+序号

### 身份切换会话重启 (2026-06-26)
- 问题: update_session(instructions=...) 只替换系统指令, 不清除 conversation items; 切人后模型仍看到旧对话提到前人的爱好 → 记忆污染
- 修复: 身份切换时先保存旧人 conv_log 做 consolidation, 然后 close + open_session 重建干净 WS
- restart_session_for_switch(old_pid, new_pid, new_pname): 保存旧人日志→关闭WS→新建WS→注入新人记忆
- pending_identity_restart flag 由视觉线程设置, 主循环在 in_flight==0 时执行重启

### Qwen-Omni-Realtime 工具调用特性 (2026-06-26)
- tool_choice 和 parallel_tool_calls 均不支持 — 工具调用完全由模型自主决定
- 括号动作描述(（点头）/(nods))是 omni 端到端语音模型的固有特性, 无 API flag 可关闭
- 工具定义用扁平风格(type/name/description/parameters), 非嵌套 Chat-Completions 风格
- 情绪/语气控制没有 API 参数, 完全靠 instructions prompt 驱动("用开心的语气说")
- voice 不支持情绪变体(如 Ethan-happy), 任何音色都能表达情绪
- 缓解标签泄漏: prompt 禁令 + 引导用语气替代括号 + 已知标签→物理动作兜底
- 正例/负例触发词模式是唯一可靠的工具触发杠杆(end_session 已有)
