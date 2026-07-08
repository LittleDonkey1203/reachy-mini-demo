# PROJECT_STATE

## 已完成事项

### ✅ 工具系统重构 — 硬编码 → 插件式 Tool ABC(2026-07-06)

将分散在三个文件的工具系统（config.py BASE_TOOLS + manager.py QWEN_TOOLS + realtime.py 130 行 if/elif 分发链）重构为插件式 Tool 基类模式。

**新建 tools/ 包**（原 tools/ 脚本移至 scripts/）：
- `tools/base.py` — Tool ABC + ToolDeps dataclass
- `tools/registry.py` — ToolRegistry（register/get/specs/exclude）+ build_default_registry()
- `tools/motion.py` — MotionTool 类（8 个实例：nod/shake_head/look_*/wiggle/tilt）
- `tools/session.py` — EndSessionTool（exit_i 计数器从 ChatCallback 移入）
- `tools/memory.py` — RememberFactTool / ForgetFactTool / ClearMemoryTool / ConfirmClearTool

**修改文件**：
- `voice/realtime.py` — ChatCallback/RealtimeDialog 接受 registry 参数；130 行 if/elif 替换为 ~20 行统一分发
- `voice/d01_realtime_chat.py` — TOOLS 列表替换为 build_default_registry()；no_memory 用 registry.exclude()
- `voice/config.py` + `memory/manager.py` — BASE_TOOLS/QWEN_TOOLS 加 DEPRECATED 注释

**验证**：py_compile 10/10 绿；新旧 specs 13/13 完全一致；exclude 过滤正确

**扩展方式**：新增工具 = 创建 Tool 子类 + register()，无需改分发代码
### 🔧 多人张冠李戴修复(2026-07-06,bug-069/070,PR #16,分支 pr14-on-main,⚠️ 待两人硬测)
- **现象**:多人在场,新人/换人问"我是谁/我叫什么/我喜欢吃什么" → 模型答成另一个人(碧霞被答"你叫大大")。
- **诊断 & 探针实测(一锤定音)**:归属层(ASD)+注入内容都对。用随机暗号探针(create_item 埋 A#、response.instructions 埋 B#,看回复带哪个)实测:**A# 从不出现、B# 每轮都带且对得上当轮 → Qwen Omni 完全忽略会话中途的 create_item system 条目,只 honor response.instructions**。所以 v1 的 create_item 注入是废重量。
- **修复(voice/realtime.py,D-only)**:
  - **收回 turn-taking**:3 处会话配置 `turn_detection_param={"create_response":False}` + 注入后我方手动 create_response(in_flight==0 守卫)→ 治 semantic_vad 在 speech_stopped 抢跑。
  - `resp_directive()`(**Option D,唯一有效通道**):`response.instructions` 给单次回复下"当前说话人"强指令(基础人设+身份约束,普通轮 + 工具轮都带)。
  - 过滤 称呼/名字 类 fact 防和身份名打架(治"碧霞→陛下");fix 兜底抽取器 save_fact 缺参 TypeError(bug-069)。
  - **已砍**:create_item 注入(inject_context)整套 + 诊断探针 —— 探针证明无效,身份注入单通道走 D。
- **已验**:D 被 honor(回复带当轮 B#);单人 + 初步多人无张冠李戴(归属对→答对,不确定→中性)。
- **待验**:两人都已识别、都在画面、轮流问身份 → 确认 D 的"忽略历史里的其他人"能压住(真正对抗历史)。
- **后手(若 D 不够)**:Option C —— 身份切换重启会话清历史(`restart_session_for_switch` 现死代码,需接线+去抖)。

### ✅ 注视感知 Phase 2 — ARMED 注视回看 + 时间常数平滑 + Dashboard 可视化(2026-07-02)

**Phase 2a: Dashboard 可视化**
- `debug_server.py`: mutual_gaze 框色(说话绿>注视青>普通灰) + 底部 gaze 标签(LOOK / Y:+5 P:-3) + 注视方向箭头 + 左上角 `gaze=CURIOUS_LOOK →T42` 行
- 注册面板(#reg-panel)改为可关闭(✕)+可拖动(标题栏)+重开按钮(🏷)

**Phase 2b: ARMED 注视回看**
- 有人看机器人(CURIOUS_LOOK/SCANNING)→ 机器人在 ARMED 下缓慢回看对方
- 积分机制: 指数时间常数 τ=0.80s(TRACKING 用 0.40s,这里慢一倍) + OneEuroFilter 平滑
- 防抖: 入场延迟 0.5s + deadband 3° + max_step 1.2°/帧 + 不驱动身体
- 退出: gaze→IDLE/GLANCING 时 behavior_loop 恢复 approach(0,0,0) 回正

**修改文件**:
- `voice/state.py`: +gaze_target_u/v 字段
- `voice/config.py`: +GAZE_ARMED_TAU/MAX_STEP/DEADBAND/ENTRY_S 常量
- `voice/d01_realtime_chat.py`: gaze FSM 存 target u,v + vision_result_loop ARMED 积分分支 + behavior_loop 条件 approach
- `voice/debug_server.py`: 可视化 + 面板交互

### ✅ 注视感知 Phase 1 — 三级级联 Gaze Estimation(2026-06-29，feat/gaze-aware-interaction 分支）

在场人是否在看机器人 + 注视行为状态机，为后续"好奇回看"交互打基础。

**架构**：L0(5点几何头姿,0.02ms) → L1(时间降频,NOT_LOOKING 5帧1次) → L2(L2CS-Net MobileNetV2 ONNX 448×448,~35ms/face macOS Intel CPU)

**新增文件**：
- `perception/gaze.py` — GazeResult/HeadPoseFilter(L0)/GazeEstimator(L2)/GazeModule 级联管理器
- `perception/gaze_behavior.py` — GazeBehaviorFSM(IDLE/CURIOUS_LOOK/SCANNING/GLANCING)
- `scripts/benchmark_gaze.py` — CPU latency benchmark(p50/p95/p99）

**修改文件**：
- `voice/config.py` — 注视估计常量段(阈值/FSM参数)
- `perception/face_pipeline.py` — TrackView 增 gaze 字段 + process() 内级联调用 + _view() gaze 填充
- `voice/state.py` — st.gaze_behavior/gaze_target_id
- `voice/d01_realtime_chat.py` — GazeModule 初始化注入 + FSM 接线 + debug 字段

**关键设计**：
- GazeModule 注入 `_face_pipeline._gaze`（最小侵入，不改 __init__ 签名）
- 模型缺失 → available=False，只跑 L0 头姿，不崩溃
- Phase 1 仅观测：FSM 结果写 st + debug，不覆盖现有 ASD 头部跟随
- 模型权重：`models/l2csnet_mobilenetv2.onnx`（9.3MB,gitignored,需手动下载或 scripts 自动拉取）

**验证**：py_compile 6/6 绿 + 无模型优雅降级测试通过 + benchmark L0=0.02ms L2=35ms p50 + TrackView 向后兼容

### track churn 治本(2026-06-27,bug-062)—— 当前一切乱象的总根
- **根因**:ByteTracker Stage3(lost track 找回)只用 embedding ReID;方案B 跟踪检测无 embedding → `embedding_distance` 全 1.0 → lost 永远找不回 → 漏检一帧就新建 track(稳定画面 ~8个/分钟,崩溃期飙到 84)→ ASD 逐 track 负载爆 → fps 崩 1.0fps。
- **对照** asd-demo `webcam_asd_demo.py`(run_webcam_asd.bat)的朴素 IoU 跟踪:漏检 miss+1、留池、下帧 IoU 重匹配 → 不 churn。印证差在"lost 找回纯 embedding"。
- **修法**:Stage3 按 embedding 有无分两路(都用匈牙利 `linear_assignment`):①全无 embedding→纯 IoU 位置找回(门 1-iou_threshold);②有→原 embedding 跨位找回(门0.45)。新增单测锁定,21 单测全绿。
- **决策已拍板(2026-06-29)**:用户确认**保留 IoU 召回**(优于"匹配不上就新建+重提 ArcFace")——IoU 召回时 track_id+identity 双不变,ASD 时序无缝、不提 ArcFace、无未识别窗口。用户三条逻辑(①匹配不上→新建+ArcFace ②匹配上→身份不变 ③ASD 按身份时序)全部已落地。
- **✅ 真机验证通过(2026-06-29)**:两人+DOA转头场景 **fps 稳 14~18**(修复前同场景崩到 **2.4**),死亡螺旋已断;每次 churn 新建 track 都被 **ArcFace 重认回正确身份**(小一/坤坤 dist 0.085~0.36)→ track_id 可丢、身份稳(逻辑①坐实)。记忆归属(本 session 最初 bug)清晰说话时全链路正确:坤坤报名+喜好 → remember_fact×3 + gallery落盘 + 认主成功。残留:50s 内 track_id 到 13(~12/min,启动预热+DOA整帧位移破IoU),非螺旋、非阻塞。

### ✅ 命名/身份名修复 CP1-5(2026-06-29,bug-063/064/065 已改,待真机验证)
真机测出名字混乱(占位名 ?T 被读出 / 毕夏→陛下脑补 / 同一身份 1 分钟改名 坤坤→陛下→大大 / 画外『我叫大大』落在场人 / 碎片幻听唐林子)。全部 voice/realtime.py + voice/d01_realtime_chat.py:
- **CP1**(bug-063):拆显示名/注入名——`_real_name`(未命名=None,占位 ?T 只用于日志/dashboard),turn_speaker/注入/extract 一律传 _real_name,占位名绝不进模型。
- **CP2**(bug-064):删 remember_fact 的 `or st.current_person_id` → 画外/无归属不存不命名(靠 response.created 的 turn_speaker gate)。
- **CP3**(bug-064):新增 `try_name_identity()` 统一命名 guard,两条命名路径都走——门1 名字合法、门2 **必须在当轮转写里**(防脑补,名以 ASR 为准)、门3 **不静默改名**(仅显式改名意图才覆盖;extract 路径 allow_rename=False 永不改名)。
- **CP4**(bug-063):未命名已注册身份注入时明确"还不知道名字,别编名、别套别人名,可礼貌问"。
- **CP5**(bug-065):dashboard 标签 + 焦点名每帧从 `memory_mgr.get_name` 现取(治改名后显示滞后,可看出 store/memory 不一致)。
- **行为变化**:名字以 ASR 转写为准(模型听岔的 坤坤 vs 实际 宫坤 → 取转写的)。`py_compile` 通过。
- **遗留(非代码)**:麦克风电平 RMS 0.003~0.005 偏低 → ASR 碎片/幻听,需硬件侧改善;gallery 脏数据由用户自行清空重认。

### ✅ DOA 瞟头 F1+F4(2026-06-29,bug-066,待真机验证)
用户反馈"侧边喊很大声头也不转/转不够"。根因:①门控 catch-22(>55° 确信声音被方向门控静音→realtime VAD 收不到→瞟头 _recent_speech 永 False);②触发靠 doa_confident 非音量;③转向用不可信的 resid 角度数值→不够。
- **F1**:本地麦响度(门控前 mono RMS>`GLANCE_LOCAL_RMS`=0.006 且 in_flight==0)stamp `st.local_speech_at`;瞟头 _recent_speech = realtime VAD **或** 本地响度,绕开门控死锁。
- **F4**:转角朝 DOA 符号方向取 `max(|resid|, GLANCE_MIN_TURN_DEG=50)` 封顶 75°,不信角度大小。
- **待调**:`GLANCE_LOCAL_RMS` 估值(正常说话~0.003),真机喊了不转就调小、误触发就调大。未做 F2/F3。

### ✅ 二次唤醒 A 方案(2026-06-29,真机验证通过)
**真机验证(2026-06-29 15:15–15:18)**:不带 --no-wake 启动,KWS RMS 0.03~0.12 健康。共 5 次对话中喊"小艺"→ 全部触发 `🔀 二次唤醒 打断+转向找喊话人`(4 次粗方向 + 1 次 confident)→ 转向 DOA→锁脸→招呼,保留会话。画外问"我是谁"答"看不出你是谁"(中性上下文也对)。说"拜拜"正常回 ARMED。**观察**:5 次里 4 次 DOA 仅"粗方向"(不 confident)→ 按固定大角(~65-70°)转,能找到人但方向近似;想更准需提升 DOA 置信度(F2 类)。

需求:对话中喊"小艺"→ 打断 + 天线动一下 + 转到 DOA 找喊话人。脚手架(KWS/switch_request/天线cue)已有,补两块 + 改保留会话:
- 唤醒块(d01 音频循环):喊"小艺"→ `_do_barge_in` 打断当前回话 + `wake_cue="heard"` 天线上扬应答 + `switch_request` 转向 DOA 找人;**去掉原无条件 close+reopen → 保留会话**(身份按本句说话人逐轮注入)。
- **必须不带 `--no-wake` 启动**(否则无 KWS)。麦增益要够 KWS 能听到"小艺"。
- 上次"崩溃"教训:`--no-wake` 下 SEEK 放弃会进 ARMED+断WS 死胡同(无 KWS 回不来);改用带唤醒词启动可避开,且 DOA 把正前方判 -46° 转走人是 DOA 角度不准的老问题(behavior SoundTurn,非 F1/F4)。
- **连带**:churn 治本后,`ASD_MAX_TRACKS`(ASD每帧只喂最大3个)+ `FPS_FREEZE_BELOW`(低fps冻身体/瞟头)降级为保险网。
- **遗留小风险**:纯 IoU 找回 position-swap(某人走、另一人1.5s内占位)短暂继承冻结身份误 ID,罕见。3D-Speaker 声纹(CAM++)多模态可作后续加强(离线→需改流式)。

### 注入只认说话人 + ASD按身份 + DOA角度转头(2026-06-27,bug-060)
- **#2 记忆注入解耦焦点**:去掉视觉焦点变化的注入触发 + 删 d01 late-inject;注入唯一来源 = realtime transcription 按 `turn_speaker`(治"Unknown-2说→注入毕夏")。
- **#4 ASD 按身份键聚合**(`person_id` 或 `t{track_id}`):feed_crop/scores/speaker/speaker_window/speaking_ids 全改 key,churn 换 track 喂同一缓冲→新人攒够帧能激活(治"大大/坤坤进来一直画外")。引擎加 `last_track(key)`;realtime 归属去 find_track、key 即 pid。
- **#1 DOA 角度转头**:进入时锁 `body_yaw+clip(resid,±40°)` 世界目标,>颈限身体跟随转过去面对;封顶防镜像错转飞;身体转受 fps 断路器约束。
- **遗留 #3 身份碎裂(毕夏×2)**:只能按 embedding 同脸合并(绝不按名字,两人可能重名)——待 #2/#4/#1 验证后做(clustering.merge_identities 按余弦 + merge_memories)。

### fps 螺旋断路 + 画外不串人(2026-06-27,bug-059)
- **fps 崩溃根因 = track churn 死亡螺旋**:相机自运动(DOA瞟头+身体跟随追侧面新人甩到-77°)→运动模糊→ByteTrack狂换track(id到104)→ASD每track负载飙→抢资源→子进程SCRFD 40→390ms→fps崩2.4→跟踪更差→更多churn。
- **断路器**:vision循环算 fps EMA,`< FPS_FREEZE_BELOW(8)` → 冻结身体跟随 + 不瞟头(身体甩=最大相机自运动),断螺旋(日志🧊)。
- **画外不串人**(补全 bug-058):`resp_snapshot` 本轮有用户说话就用 turn_speaker_pid(画外=None=不存),不回退在场人 → 画外的"我叫X"不再被存到在场人头上(治"大大被改名坤坤")。
- 污染数据由用户全清(gallery/memories/face_db 均空),干净重来。
- **遗留**:ASD/识别按身份(person_id)聚合(churn 根治)未做——先验证断路器是否足够(churn 降下来后坤坤这类新人或许就能正常激活)。

### 头部转向平滑 + Dashboard 配色重整(2026-06-27,bug-057/058)
- **头不晃 + 回复不叫错人(按身份黏滞)**:头部"看谁"(`_head_view`)**和当前人 `current_person_id` 都按身份(person_id)黏滞**——引擎 EMA 上叠二级重 EMA(`HEAD_ASD_EMA=0.18`)+ 黏滞(切人需高出 `HEAD_SWITCH_MARGIN=0.5`,否则黏住直到该身份离场;churn 换 track_id 不算离场)。
  - 初版用时间 hold(HEAD_HOLD_S/DOA),实测"无人说话仍晃"——根因=回退最大脸帧间翻 + track churn;改"按身份黏滞、离场才释放"消除。
  - **bug-058**:current_person_id 原由瞬时 ASD 抖动驱动→多人间疯狂切→`update_session` 反复重注入竞态→"归属对但回复叫错人(大大→陛下)"。改由稳定焦点驱动 + realtime transcription 不再写 current_person_id(只按本句说话人 update_memory)。**归属/记忆保存仍走 speaker_window 敏感不变**。
  - **DOA 瞟头(2026-06-27 追加)**:治"我喊他都不转"。TRACKING 态,DOA 确信+偏离>20°+画面里没人在说话 持续≥0.3s → 头朝声源**符号方向**瞟 ≤15°(只用符号不信角度;<颈限×0.7=16.1° 保证只动头不甩身);瞟到说话人→按身份黏滞立即锁。只在 vision 循环 TRACKING 写头,不和 behavior_loop(SEEK/SWITCH 需唤醒词)抢。参数 `DOA_GLANCE_DEG/GLANCE_MAX_DEG/GLANCE_MIN_HOLD_S`。日志 `👀 DOA 瞟头`。
  - 遗留:"出画面瞬间归属仍归到我"(asd_speaker 2s hold)未改,次要。
- **绿框正确**:`asd.speaking_ids()` 加新鲜度门(治"残留正值绿不灭");ASD 分显示改 2 位小数。
- **配色重整**:脸 🟩绿=说话 / ⬜灰=跟踪(**去掉蓝框**);手 🟦青=有效 / 🟧橙=底部 / 🟨黄=低置信(**绿只留给说话脸**,避免手框压脸误认)。
- 验证:py_compile 4 文件绿;**待实机验证头部是否稳 + 配色**。

### 记忆归属统一 + 每轮工具审视兜底(2026-06-27,bug-056)
现象:xx 说"喜欢吃西瓜"没存给 xx,日志还显示"记忆已注入 吴觊豪"。根因 = **两套归属用了不同标准** + **plus 偶发漏调 remember_fact**。
- **根因①(串人)**:转写显示用稳的 `speaker_window`(→xx 对);但记忆"存(resp_snapshot←current_person_id)+读(update_memory←current_person_id)"挂在飘的全局 `current_person_id`(vision loop 瞬时 ASD+单人 fallback 刷),11:04:59 fallback 把 current 切成吴觊豪 → 注入错记忆。
- **根因②(没存)**:plus 本轮 0 次 remember_fact("说了不做")→ 西瓜根本没存到任何人(xx facts 为空)。
- **修①(归属统一)**:新增 `st.turn_speaker_pid/name/at`(transcription 时由 speaker_window 定);记忆 存(`resp_snapshot`/`remember_fact`)+ 读(`update_memory`)统一改用 turn_speaker,与飘的 `current_person_id` 解耦(后者回归焦点/显示本职)。**不需要给 current_person_id 加守卫**:inject 只读不污染数据,且 d01 late-inject 本有 `in_flight==0` 守卫(回复在途不注入),fallback 切人只在两句之间且下一句自愈(曾加 TURN_LOCK 经审视为伪需求,已移除)。
- **修②(每轮兜底)**:新增 `RealtimeDialog.extract_memory_async`,每轮 transcription 后**无条件**用 `EXTRACT_MODEL=qwen-plus`+最近5轮上下文抽"本句说话人"个人事实/姓名,`save_fact` 内置去重 → 兜底 realtime 漏调,与原生 remember_fact 并存。
- **验证**:py_compile 4 文件全绿;**待实机验证**(说个人信息看是否存对人 + 多人/画外不串)。

### 实时视频流送 Omni(1fps/720p,2026-06-27 实机验证通过)
- d01 mic 循环每 1s 取 latest_frame→720p→JPEG q70→base64→`conv.append_video`;加 `📹 视频流已送 N 帧` 计数日志。
- BASE_TOOLS 移除 take_snapshot/identify_pointed_object;INSTRUCTIONS 改"画面持续可见,直接答视觉问题不调工具"。
- **实机验证**:模型能直接答"是一支笔/一部黑色手机/正在说话的是旁边那位女生/画面晃了一下" → 视觉问答与手势识别走通(720p 对小字/远物细节有极限,按用户要求不留高清兜底)。

### 人脸检测/跟踪/ReID 全量迁移(参考 face-tracker-demo,2026-06-26)
完全替换旧 FaceSelector + 零散身份逻辑,落地 5 commit(3781515→9d46898):
- **检测**:vision_worker 默认 InsightFace **SCRFD**(buffalo_sc/det_500m,子进程),输出 all_faces=[{u,v,h,box,kps5,conf}];保留 MediaPipe(手势)与 YuNet 作可选 backend(FACE_BACKEND 切换)。
- **跟踪**:`perception/face_tracker.py` 忠实移植 **ByteTrack**(KalmanBox + 两段 BYTE 关联 + lost-track embedding ReID + Tentative/Confirmed/Lost)。
- **身份**:`identity/identity_store.py` **三区间**(known≤0.65 / unsure / unknown≥0.80,cosine 距离),provisional(Unknown-N 自动)vs confirmed(命名),质量门 min_quality=0.40,distance_log 标定;阈值直接复用 face-tracker-demo(检测+识别全复用故可迁移)。
- **质量/平滑/聚类**:quality.py(FIQA 代理) + clustering.py(EmbeddingSmoother + GalleryClustering 完整移植)。
- **集成层**:`perception/face_pipeline.py`(FaceReIDPipeline)串联 ByteTracker+全分辨率 ArcFace(w600k_mbf,复用既有 recognizer.arcface)+IdentityStore;懒提特征(per-track 限频 + 每帧预算 + DOA 优先);出口仅归一化 u/v/h(铁律:不写 st.state/不调 head_control)。
- **d01 接线**:vision_result_loop 调 pipeline,primary→头部跟随,person_id(=gallery identity_id)→ st.current_person_id → 既有记忆注入/Owner 不变即可工作;安全删除工作流改走同一身份空间;cv2 提前 import 规避 spawn 崩溃。
- **数据**:新开 `data/gallery.json`(旧库已清);记忆 keyed on gallery identity_id。
- **验证**:py_compile 全绿;26 单测绿(test_facereid_port 20 + test_face_pipeline 6);SCRFD 子进程冒烟 6 脸/conf0.88。**待实机全链路验证**。
- **遗留(非阻塞)**:①命名→gallery confirm_identity 钩子;②在线 clustering 维护(周期 find_mergeable_pairs/compact);③Dashboard track_id/zone 叠层;④_vis_enabled 门仍判 face_landmarker.task(机器人上已存在,对 SCRFD 非必需)。

### 人脸 ReID 稳定化(2026-06-26,实机验证"识别很稳定")
迁移后逐项实机调优,识别已稳定。关键修复(commit 8f0e750→a583171):
- **track churn 根因修复**(bug-054):split 路径检测无 embedding,`embedding_distance` 返回全 1.0,按 embedding_weight=0.3 加权把 IoU 门从 0.30 抬到 0.429 → 低 fps 丢轨重建。`face_pipeline` 在 `tracker.update` 前 all-None 时清零 embedding_weight(镜像 face-tracker-demo)。实测最大 track_id 214→6。
- **方案B(跟踪/识别解耦)**:DECIMATE=3 做跟踪(track 稳、fps 23-25);识别走主进程惰性 SCRFD 对选中脸**全分辨率 ROI 重检**拿 sharp kps → 判别力够,异人 dist 0.816 vs 同人 ≤0.58 分开,**误匹配消失**。
- **身份冻结**(Q4):`_needs_embedding` 对已绑定 track 返回 False,只有新 track 才识别(track 在则身份不变)。
- **命名落 gallery**:`realtime`/UI 起名 → `store.confirm_identity` + `save_gallery`;退出也 save_gallery;**跨会话持久化已验证**(重启加载回 confirmed 身份)。
- **每框常驻显示**:`dbg_det.track_views` → debug_server 每框画 身份(Unknown-N/真名)+ T<id>,蓝/灰/绿;右上角毫秒时间戳(对应 log);每 track 识别日志 `🔍 track N → 名 (dist=..)` 供阈值校准。
- **UI 注册功能**:Dashboard 左下角面板,点人脸填 track→命名,绕过"谁在说话"显式命名指定脸。
- **模型档**:记忆/动作场景必用 `qwen3.5-omni-plus-realtime`(flash function-calling 不可靠;plus 需账号开通,已购买,见 buglog-051)。
- **待办**:①阈值校准(0.65 对低质量帧余量紧,靠质量门而非收紧);②残留 track churn(两人动态场景 cosmetic,可上 Kalman 线性噪声);③**"谁在说话"= 移植 asd-demo 的 LR-ASD 音视频同步**(用户已定用 LR-ASD;方案见对话/待实现:perception/asd.py + 音频 ring tap + per-track 灰度累积 + 说话人归属 current_person_id)。

### 架构
- 6 模块拆分 → d01 瘦身(领域驱动): 拆出 kws.py / fusion.py / safety.py / realtime.py
- 方向门控白名单化(仅 TRACKING 关门)

### 核心特性
- YuNet+ArcFace 身份识别 + auto_merge 碎片修复
- GestureRecognizer 手势(模型优先+规则 fallback)
- 认主机制(OwnerManager) + 记忆权限矩阵
- 记忆注入 update_session 替代 create_item
- 多人脸 DOA 说话人选择 + all_faces 输出
- 唤醒优先级(a_active) + TRACKING 身体跟随 + 人脸误识别迟滞
- 安全删除工作流(多步验证+备份)
- display_transcript 持久记录本 + Dashboard 上下文调试
- Intel Mac 兼容(mediapipe<0.10.15 + onnxruntime<1.20)

### 认知记忆架构 (2026-06-25)
- **auto_merge → MemoryManager 同步**: FaceDB 合并碎片人脸后自动调用 merge_memories()
- **Entity Memory**: facts `dict[str,str]` KV 格式，支持同 key 自动覆盖
- **Episodic Memory**: 结构化事件(topic/highlights/mood)，保留最近 10 条
- **Working Memory 注入**: summary 叙事 + KV 详情 + episodic 组装
- **Session Consolidation**: 会话结束后 LLM 复盘(全量对话 + facts KV → entity dict + summary + episode)
- **旧数据自动迁移**: load_memory 自动检测旧 dict/list 格式并转换

### 身份稳定性 + 上下文防污染 (2026-06-26)
- **首次识别最低 sim 阈值**: `FIRST_DETECT_MIN_SIM=0.45`，防止低 sim 误注册(如大哥 sim=0.41 误匹配)
- **身份切换阈值提高**: `ID_SWITCH_HIGH_SIM` 0.65→0.72，`ID_SWITCH_CONFIRM_N=3`
- **身份切换会话重启**: 切人时 close + open_session 重建干净 WS，清除旧对话历史防止记忆串人
- **TTS 标签泄漏防护**: prompt 强化禁止括号/星号/XML动作描述 + 正则兜底清除(双层过滤)
- **Qwen-Omni 工具调用调研**: tool_choice 不支持(模型自主决定)，括号动作是 omni 固有特性(无法 API 关闭)

## 当前架构状态

```
voice/
  config.py        — 常量 + prompt (工具定义已迁移到 tools/)
  state.py         — State 类 + log + OneEuroFilter
  d01_realtime_chat.py — 主程序 (~600 行，已瘦身)
  debug_server.py  — Dashboard
  kws.py           — 唤醒词门控
  realtime.py      — Qwen-Omni-Realtime 协议层 + registry 分发 + Session Consolidation
tools/
  base.py          — Tool ABC + ToolDeps dataclass
  registry.py      — ToolRegistry + build_default_registry()
  motion.py        — MotionTool (8 动作工具)
  session.py       — EndSessionTool
  memory.py        — 4 个记忆工具类
perception/
  vision_worker.py — Face(YuNet/MediaPipe) + Hand(GestureRecognizer)
  fusion.py        — 声源-视觉融合
identity/
  recognizer.py    — ArcFace 身份识别 + auto_merge + startup_merged
  owner.py         — 主人认定
memory/
  manager.py       — 认知记忆管理(Entity + Episodic + Working Memory)
  safety.py        — 安全删除工作流
```

### 身份识别稳定性机制
```
首次识别:
  sim >= FIRST_DETECT_MIN_SIM(0.45) → 接受
  sim < 0.45 → 忽略，日志告警
已有身份切换:
  同人(pid相同) → 直接接受
  sim >= ID_SWITCH_HIGH_SIM(0.72) → 立即切换
  sim < 0.72 → 连续 CONFIRM_N(3) 次才切换
  冷却: 切换后 COOLDOWN_S(6s) 内不再切换
切换时:
  close_session → save_summary(旧人) → open_session → update_memory(新人)
  conversation items 全部清除，防止记忆串人
```

### 记忆生命周期
```
会话中:
  remember_fact(key, value) → KV 实时存盘，同 key 自动覆盖
  forget_fact(keyword) → 模糊匹配 key 或 value 删除
  identity_injected=False → 触发重注入最新 facts
会话后 (close_session):
  save_summary() → LLM consolidation:
    输入: 全量对话 + 当前 facts KV + 已有 facts
    输出: 最终 entity dict + summary 叙事 + episodic memory
下次对话:
  get_prompt(pid) → summary + KV 详情 + episodic 组装注入 Working Memory
```

- 9 状态 FSM: ARMED/IDLE_CENTER/ENGAGING/TRACKING/SEARCHING/RETURNING/POINTING/PLAYING
- 5 层运动仲裁: Primary > Playing > SoundTurn > Tracking > Idle

### 多人"我是谁"误答修复(2026-07-06,bug-068)
- **现象**:已认识 A 在场,新人 B 问"我是谁",模型答"你是 A";但全都认识时接力问则回答正确。
- **诊断**:`test_identity_switch.py` 5 场景全 PASS → 模型能力没问题,能正确处理 `update_session` 中途切换 instructions(含 known→neutral、known→known、session restart)。
- **根因**:ASD fallback(`realtime.py:222-226`)— B 刚出现,ASD 还没攒够帧,`speaker_window()` 返回 None → 2 秒内 fallback 到 `st.asd_speaker`(=A)→ 注入 A 的记忆 → 模型理所当然答"你是 A"。
- **修法**:ASD `tracked_keys()` 暴露当前有 crop 缓冲的 key 集合;fallback 时检查追踪人数——多人(`>1`)不 fallback,走 neutral 路径;单人保持原逻辑。日志 `⚠ ASD fallback 拦截` 可追踪触发情况。
- **待真机验证**:两人场景 + 新人问"我是谁"应答"不认识"。

## 遗留问题

1. **YuNet 无 blendshapes**: smile/frown 恒 0.0, 可用 insightface 2D106 估算
2. **多人同框介绍**: 指着他人说"这是XX" → 关联名字(方案见 docs/MULTI_PERSON_INTRO_PLAN.md)
3. **end_session 触发率低**: Qwen-Omni 不支持 tool_choice=required，只能靠 prompt 正例/负例触发词提升(已优化)
4. **TTS 标签仍可能偶发**: omni 模型固有特性，正则兜底可清除大部分但不保证 100%
5. **Semantic Memory**: 需 Consolidation Engine 从多条 episode 回放抽象知识(未来)

## 下一步建议

1. **真机验证 Phase 2 注视回看**：`VIS_DEBUG=1 bash start_mac.sh`，ARMED 下看机器人→头缓慢转向，看走→回正
2. Phase 3：TRACKING 态注视增强（对话中持续微调头部追踪说话人视线）
3. 真机测试验证身份稳定性修复 + 上下文防污染效果
4. 继续 todo.md 未完成项(#1 DOA / #7 身份优化 / #9 对话质量)
5. Semantic Memory 层 — 从 episodes 抽象知识 + GraphDB
