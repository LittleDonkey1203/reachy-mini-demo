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
- 记忆抽取兜底要"工具模型每轮都审视"，不要用关键词触发（词表永远漏）。判断交给模型，规则不可靠（2026-06-27 明确）
- 改方案前先"写出来给我审核"，确认后再动手；讨论时给出取舍并附推荐项

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
- **归属有两套,别用混(2026-06-27):** ①转写显示归属 = `transcription.completed` 里 `speaker_window(speech_start)`,看整句谁在说,**稳**;②全局 `current_person_id` = vision loop 用瞬时 `asd.speaker()`+单人 fallback 持续刷,**飘**(会 fallback 切错人,只管头部跟随/显示)。**记忆「存(remember_fact/resp_snapshot)+ 读(update_memory)」必须用①**(新 `st.turn_speaker_pid`,transcription 时设),不能用②。**不需要给 current_person_id 加守卫**:inject 只读不污染数据;d01 late-inject 本就有 `in_flight==0` 守卫(回复在途不注入),fallback 切人只发生在两句之间且下一句自愈。曾加 TURN_LOCK 后判定是伪需求移除。
- **ASD 有三个消费方,需求不同别用混(2026-06-27):** ①**归属/记忆保存**(speaker_window)= 敏感(原参数 ema=0.5/thresh=0,抓任意一瞬说话,不增延迟);②**头部转向 + 当前人焦点**(_head_view/current_person_id)= 稳(引擎 EMA 上叠二级重 EMA `HEAD_ASD_EMA=0.18` + 黏滞 `HEAD_SWITCH_MARGIN` + **按身份 person_id 黏滞、离场才释放**);③**绿框**(speaking_ids)= 敏感但带新鲜度门。
- **track churn 的真正根因 = Stage3 lost 找回纯 embedding,方案B 下失效(2026-06-27,bug-062):** ByteTracker `update` 的 Stage3(lost track 找回)只用 `embedding_distance`(门0.45);但方案B 跟踪检测无 embedding → `embedding_distance` 返回全 1.0 → lost 永远找不回 → **漏检一帧就新建 track**(稳定画面也 ~8个/分钟,崩溃期飙到 84)。**修法:Stage3 按 embedding 有无分两路(都用匈牙利 linear_assignment):全无 embedding→纯 IoU 位置找回(门 1-iou_threshold);有→原 embedding 跨位找回。** 对照 asd-demo `webcam_asd_demo.py` 的朴素 IoU 跟踪(漏检 miss+1、留池、下帧 IoU 重匹配)印证它不 churn。我们的 `linear_assignment` 是真匈牙利(scipy linear_sum_assignment)。**churn 治本后,ASD 限量(ASD_MAX_TRACKS)/fps 冻结(FPS_FREEZE_BELOW)降级为保险。** 小风险:纯 IoU 找回时 position-swap(某人走、另一人 1.5s 内占位)会短暂继承冻结身份误 ID,罕见暂接受。
- **track churn 是多个问题的共同根(2026-06-27):** 相机自运动(头/身大幅转,如 DOA 瞟头+身体跟随追侧面人甩到 -77°)→ 运动模糊+大角度 → SCRFD 检测碎裂 → ByteTrack 疯狂换 track(id 冲到 104)→ 同时存活 track 多 → ASD 每 track crop+打分负载飙 → 抢 CPU/GPU → 子进程 SCRFD wall-time 40→390ms → **fps 崩→跟踪更差→更多 churn 死亡螺旋**。同时 churn 让新人(坤坤)ASD 按 track_id 攒不够帧→没说话分→画外。**断路器**:`FPS_FREEZE_BELOW=8`,vision 循环 fps EMA 低于此→冻结身体跟随+不瞟头(身体甩=最大相机自运动),断螺旋(日志 🧊)。根治待办:ASD/识别按身份(person_id)聚合。
- **记忆注入只认「本句说话人」,绝不挂 current_person_id/焦点(2026-06-27,bug-060):** 焦点(头看谁)在多人交替时本就该翻,若注入跟着焦点翻→每翻一次 update_session 重注入→竞态→回复用错人记忆。唯一注入源 = realtime transcription 按 turn_speaker。视觉循环焦点变化**不重置 identity_injected、不注入**;d01 late-inject 已删。
- **ASD 必须按身份键(person_id 或 t{track_id})聚合,不能按 track_id(2026-06-27,bug-060):** ByteTrack churn 换 id,按 track_id 则每次从空缓冲攒、攒不够 12帧/1.3s→新人一直没说话分→画外。改按身份键后 churn 换 track 喂同一缓冲→能激活。引擎(asd.py)键本就通用,只需调用方传 key;`last_track(key)` 侧表供显示/归属。speaker/speaker_window 返回的是 key:`t` 开头=未识别(画面内未绑定),否则=person_id 直接当 pid。
- **DOA 转头找人 = 状态机(2026-06-27 定稿,bug-060):** 正确逻辑(用户明确):转到 DOA 角度 → **停那等说话** → 锁说话人 → 找不到**不弹回原来的大人脸**。实现:`_glance_phase` 0/1/2。①触发=DOA偏>20°+确信+画面无人说话+fps够+冷却过+持续0.3s。②进入时**锁定一次**目标 `body_yaw+clip(resid,±75°)`(=声源世界角,**不 live 追→平滑不来回转**)+ **清 _head_key**(忘原焦点)。③phase1 转向(头到 DOA;**resid 是身体系,必须转身体才减小**,所以身体跟随必须放开 fps 冻结 `_cam_ok or phase in(1,2)`,否则 churn 掉 fps 身体就停、够不到 45°+ 的人)。④到位(头距目标<8°)→phase2 停那等。⑤phase2 有人说话(head_ema>阈值)→锁定跟随;超 `GLANCE_TIMEOUT_S`(5s)没人→放弃+冷却(保持朝向,不弹回原脸)。**关键认知:转头(颈限±23°)永远够不到侧面 45°+ 的人,必须转身体;身体转会引发 churn,得在"找人"这个有意动作里临时放开 fps 冻结。** **触发必须加"真说话"闸**(`user_speaking` 麦 VAD,最近 `GLANCE_SPEECH_GRACE_S=1.5s` 内):光 DOA 确信会把环境音/反射当声源 → 身体右转时 resid 常报负(左)→ 触发左转 + "放弃不弹回" → **累积无声左漂**;加了说话闸,纯环境音不转。 身份合并只能按 embedding 同脸,绝不按名字(两人可能重名)。
- **画外/未识别说话人要注入「中性上下文」,不能残留上一个在场人的身份(2026-06-27,bug-061):** 否则模型拿上一个在场人(陛下)的记忆回答画外的人,被问"我是谁"会乱答"你是陛下"。`update_memory_neutral()` 注入"看不到对方/不知道是谁/别套用他人名字",`identity_injected_pid='_neutral'` 防抖;在场人再开口自动重注入其记忆。模型回复要 `log('💬 小艺:...')` 才进网页 log 面板(否则只 print 到 stdout)。Dashboard MJPEG 断流在 Windows 是 ConnectionAbortedError(10053),_mjpeg 要 catch 它(+OSError)。
- **画外的话绝不能存给在场的人(2026-06-27,bug-058/059):** resp_snapshot 在"本轮有用户说话(turn_speaker_at 新鲜)"时必须用 turn_speaker_pid(画外=None=不存),**不能回退 current_person_id**——否则画外说"我叫X"会被 remember_fact 张冠李戴给在场人(实测把大大改名坤坤)。仅"无近期用户说话(如系统招呼)"才回退当前人。
- **DOA 角度不可信,只信符号(项目老教训,2026-06-27 再确认):** `doa_resid_stable=90−中值角`(身体系,符号=左/右,`confident=IQR<25°`,10Hz)。ARCHITECTURE/CALIBRATION §14:角度/spread 都不准(镜像错),SEEK/瞟头只用 **resid 符号**定方向、视觉见脸才锁。**DOA 瞟头**(TRACKING 态,侧面有人喊但画面没人说话→朝声源侧瞟 ≤15°找人,`DOA_GLANCE_DEG/GLANCE_MAX_DEG/GLANCE_MIN_HOLD_S`):只在 vision 循环 TRACKING 写头(不和 behavior_loop 抢);GLANCE_MAX_DEG=15 < 颈限23×0.7=16.1 保证只动头不甩身;找到说话人(ASD)→按身份黏滞立即锁(新 key EMA 以瞬时分起步,无弹回)。
- **头部/当前人必须按身份(person_id)黏滞,不能按 track_id 或瞬时 ASD(2026-06-27):** ByteTrack track churn 频繁换 id(同人 T14→T17→…),按 track_id 黏滞会"离场→重选→晃";瞬时 ASD 驱动 current_person_id 会在多人间疯狂切→`update_session` 反复重注入→竞态→**回复称呼错人**(归属对但叫错)。解法:`_head_key = person_id or t{track_id}`,churn 换 track 不算离场;current_person_id 跟稳定焦点走;realtime transcription 只 update_memory(本句说话人)不写 current_person_id。
- **Dashboard 画框配色(2026-06-27 定稿):** 脸:🟩绿=正在说话(speaking_ids,>阈值且新鲜)/⬜灰=跟踪中;手:🟦青=有效/🟧橙=底部过滤/🟨黄=低置信。**绿只给说话脸**(有效手从绿改青,避免手框压脸误认);**蓝色全部去掉**(头部跟谁看机器人朝向/yaw,不用框色重复表达)。ASD 分显示 2 位小数(1 位会把 0.0x 显示成 +0.0)。
- **记忆兜底抽取(2026-06-27):** `RealtimeDialog.extract_memory_async` 每轮 transcription 后无条件用 `EXTRACT_MODEL`(qwen-plus)+最近5轮上下文抽「本句说话人」个人事实,`save_fact` 内置去重 → 兜底 plus 偶发漏调 remember_fact("说了不做")。与 realtime 原生 remember_fact 并存不冲突。
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
- [2026-06-27] **记忆别挂在 current_person_id 上**: 它由 vision loop 瞬时 ASD+fallback 刷,会在说话人没变时 fallback 切错人 → 存错人/读错记忆。记忆存/读统一用 `st.turn_speaker_pid`(speaker_window,per-utterance)。转写归属对≠记忆归属对,两条链当时是分开的。
- [2026-06-27] **"模型说了记住啦"≠真存了**: plus 也会偶发不发 function_call(本轮 0 次 remember_fact),靠模型主动调不可靠。必须有每轮工具模型兜底抽取。诊断"没存"先 grep `🤖 模型调用工具: remember_fact` 看调没调,再看存到谁(磁盘 data/memories/*.json facts)。
- [2026-06-26] **realtime.py:110 的"已注册"日志不可信**: 它是写死文本,漏列 end_session 和 4 个记忆工具,不反映真实 update_session(tools=...) payload。别拿它当"工具没注册"的证据。真实注册看 d01:175→1544→1561 + realtime.py:388/463。

## Decision Log

<!-- Significant technical decisions with rationale. Why X was chosen over Y. -->

### 二次唤醒 A 方案:打断+转向找喊话人,保留会话 (2026-06-29)
- 需求:对话中喊"小艺"→ 打断 + 天线动一下 + 转到 DOA 方向找喊话人。
- 决策(用户选 A):喊"小艺"(KWS)→ `_do_barge_in` 打断当前回话 + `wake_cue="heard"` 天线上扬 + `switch_request` 转向 DOA 找人,**保留会话**(去掉原来无条件 close+reopen);身份仍按本句说话人逐轮注入,上下文不丢。
- 否决 B(丢弃会话重开):上下文丢失、重连有延迟;A 更轻、更连贯。
- **依赖**:必须**不带 `--no-wake`** 启动(否则无 KWS,喊"小艺"无效)。且麦增益要够 KWS 能听到。
- 用唤醒词当触发(KWS 训练模型)比 DOA 响度(RMS 阈值)在低麦增益下可靠得多——这是相对 F1 的更稳路径。

### 命名 guard:命名是身份关键操作,与存事实分离严格 gate (2026-06-29, bug-064)
- 背景:真机测出名字混乱(毕夏被记成陛下;同一身份 1 分钟被改名 坤坤→陛下→大大;画外『我叫大大』落到在场人;碎片幻听→唐林子)。
- 决策:**命名走独立 guard `try_name_identity()`**,三道门:①合法 ②**名字必须出现在当轮转写里**(防模型脑补,名字以 ASR 为准)③**已命名不静默覆盖**(仅显式改名意图『改名/其实叫/叫错/应该叫』才改)。
- **改名策略**:默认拒绝静默覆盖;extract(工具审视)路径 `allow_rename=False` 永不改名,只首次命名;改名只能走模型直调路径且需显式意图。不做确认握手(低麦克风+实时模型下握手脆且加延迟,门2/3 已够)。
- **画外绝不命名**:删 remember_fact 处理器的 `or current_person_id` 兜底(它把画外 None 兜回在场人)。靠 response.created 已有的 turn_speaker gate。
- 取舍:门2 用「子串」最稳——若 ASR 把名字听岔(宫坤↔坤坤),模型传的名不在转写里会被拒,那轮以 ASR 文本为准命名(更对)。用户认可「宁可少记别记错」。

### 显示名实时取,不用缓存 (2026-06-29, bug-065)
- `trk.identity_name`/`FaceResult.person_name` 是识别那刻从 store 缓存的,改名后滞后,且与模型用的 `memory_mgr` 名可能不一致。
- 决策:dashboard 标签 + 焦点名**每帧从 `memory_mgr.get_name(pid)` 现取**(模型用哪个库就显示哪个),便于诊断 store/memory 不一致。

### Track churn 治理:IoU 召回 优先于 ArcFace 重认 (2026-06-29)
- 用户拍板:方案B(检测无 embedding)下治 churn,**保留 Stage3 IoU+匈牙利召回**(lost 池按位置续回),不改成"匹配不上就新建 track + 重提 ArcFace"。
- 理由:IoU 召回时 **track_id 与 identity 都不变** → ASD 时序完全无缝、不提 ArcFace、无"未识别窗口"断点。漏检一帧靠位置秒续。
- 仅当 IoU+匈牙利彻底匹配不上(脸真离开/大跳变)才落到 新建 track → Confirmed 后 ArcFace 按 gallery 身份匹配(罕见路径)。
- 用户三条逻辑映射:① 匹配不上→新建+ArcFace(罕见,已有);② 匹配上→身份不变(Stage1/2 + Stage3 IoU 召回);③ ASD 按身份时序(bug-060 已按 person_id 聚合)。三条全部落地。
- 残留小风险(已知接受):A 走、B 在 max_age(~1.5s)内站到 A 的位置,Stage3 IoU 可能误续成 A 身份。概率低,IoU 召回收益更大。

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
