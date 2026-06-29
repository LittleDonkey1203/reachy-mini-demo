# 小艺(Reachy Mini Lite) — 特性清单 & 测试方案

> 生成日期: 2026-06-24 | 基于 voice/ perception/ identity/ memory/ 全量代码审阅

---

## 一、特性清单

### F1 — 全双工语音对话

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F1.1 Qwen Realtime 全双工 | `d01:main()` L1760-1778 | DashScope WebSocket, `qwen3.5-omni-flash-realtime`, semantic_vad |
| F1.2 打断(barge-in) | `d01:ChatCallback._do_barge_in` L299-322 | speech_started → 代际作废 play_q + clear_player + cancel_response |
| F1.3 打断微反应 | `_do_barge_in` L318-320 | M3-b: 短促后仰 + 天线收缩(barge cue) |
| F1.4 语音下行播放 | `audio:player_loop` | 24k→16k resample + 抖动缓冲 ~300ms + 代际作废 |
| F1.5 上行电平监控 | `d01:main()` L1998-2041 | 每 10s 报 RMS, <0.005 告警 |
| F1.6 ASR transcript 录制 | `d01:ChatCallback.on_event` L348-353 | speech_stopped → conversation_log |
| F1.7 transcript 标签泄漏兜底 | `d01` L184-196, on_event L449-455 | `_ACTION_TAG_RE` 正则捕获 `<nod>` 等 → 物理动作 + 清洗 |
| F1.8 自动断连重连 | `d01:main()` L2022-2034 | append_audio 异常 → close_session + open_session |

### F2 — 唤醒词门控(WAKE-01)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F2.1 KWS 本地唤醒 | `d01:KwsGate` L1581-1647 | sherpa-onnx, "小艺"三声调, threshold=0.17, 去抖+不应期 |
| F2.2 命中才连 | `d01:main()` L1893-1911 | armed + 命中 → open_session(3s 超时); 失败 → fail cue |
| F2.3 ARMED 待命态 | `behavior_loop` L999-1034 | 慢呼吸, 不连 Qwen, 零计费 |
| F2.4 唤醒响应序列 | `behavior_loop + head_control` | heard 上扬(0延迟) → 连接 → SEEK 寻人 → 锁脸招呼 |
| F2.5 唤醒招呼语 | `d01:main()` L1970-1976, `config:GREET_PHRASES` | 7 句轮换: "在呢/来啦/你好呀/..." |
| F2.6 无互动回待命 | `behavior_loop` L1157-1159 | engaged 15s 无互动 → ARMED + 关 WS |
| F2.7 `--no-wake` 回退 | `d01:main()` L1815-1822 | 启动即连, ST_IDLE 初始, 旧行为 |
| F2.8 KWS 诊断日志 | `KwsGate.feed` L1628-1633 | 每 3s: chunks/dec/RMS, 静音告警 |

### F3 — SEEK 寻人(视觉主导)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F3.1 DOA 弱提示起扫方向 | `behavior_loop` L1013-1031 | confident → 直转; 不 confident → resid 符号定扫向; 无 → 全场扫 |
| F3.2 两阶段 SEEK | `behavior_loop` L1217-1253 | direct(直转) → nearby(±25° 找脸) → full(±88° 全场 sin 扫) |
| F3.3 direct 压锁 | `behavior_loop` L1204-1206 | 直转途中不认脸(防途中脸拽住), 正前豁免 |
| F3.4 SEEK 超时放弃 | `behavior_loop` L1247-1253 | 7s 扫遍无人 → giveup cue → ARMED |

### F4 — 人脸跟踪

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F4.1 MediaPipe Face Landmarker | `vision_worker:vision_worker()` L204-211 | VIDEO 模式, num_faces=2, min_confidence=0.3 |
| F4.2 FaceSelector 跨帧粘滞 | `vision_worker:FaceSelector` L65-128 | 匹配距离 <0.18, 切换需另一脸连续 8 帧明显更大 |
| F4.3 One Euro 滤波 | `d01:vision_result_loop` L689-690 | min_cutoff=0.8, beta=0.08, 低速防抖/高速低延迟 |
| F4.4 时间常数型积分 | `d01:vision_result_loop` L860 | `step = err × (1 - exp(-dt/τ))`, τ=0.40 |
| F4.5 face_locked 迟滞 | `vision_result_loop` L838-884 | 0.3s on / 1.5s off, 防空转 |
| F4.6 丢脸缓冲 | `vision_result_loop` L882-884 | 连续 VIS_MISS_N(5) 帧才重置滤波 |
| F4.7 TRACKING→SEARCHING | `behavior_loop` L1277-1280 | !locked → SEARCHING(迟滞已含 1.5s) |
| F4.8 TRACKING 范围限制 | `vision_result_loop` L864-867 | 颈 ±23°(身体相对), pitch ±15° |
| F4.9 `--no-sticky` | `d01:main()` L1663 | 关粘滞, 每帧 argmax 最大脸(对比/排错) |

### F5 — 听声转向(DOA)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F5.1 DOA 传感器轮询 | `audio:doa_sensor_loop` | REST 10Hz, 中值滤波, 自声门控 |
| F5.2 方向门控 | `d01:main()` L1999-2016 | fresh + confident + |resid| > 55° → 静音(不送/不打断/不计时) |
| F5.3 声源转向(ENGAGING) | `behavior_loop` L1145-1156 | IDLE + 视场外声音 + 没说话 → ENGAGING 转向 |
| F5.4 闭环链式转向 | `behavior_loop` L1254-1275 | 转到目标 + 扫头 ±15° 找脸, 超时升级宽扫 |
| F5.5 `--no-gate` | `d01:main()` L1661 | 关方向门控, 全向上行(排错用) |

### F6 — 二次唤醒切换(M1.5-b)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F6.1 engaged 再唤醒(打断+转向找喊话人,A方案) | `d01:main()` 音频循环唤醒块 | 对话中喊"小艺"(KWS命中)→ ①`_do_barge_in` 立刻打断当前回话 ②`wake_cue="heard"` 天线上扬应答 ③`switch_request` 转向 DOA 方向找喊话人(三档),**保留会话**(不再 close/reopen,身份按本句说话人注入)。**用法:必须不带 `--no-wake` 启动**(否则无 KWS) |
| F6.2 三档方向 | `behavior_loop` L1052-1089 | confident 直转 / fresh 粗方向 / 无 → 反向离A |
| F6.3 切换途中找脸 | `behavior_loop` L1161-1203 | turn → sweep(附近找脸), 超时回A |
| F6.4 切换冷却 | `d01:main()` L1919 | SWITCH_COOLDOWN_S=2.0s |
| F6.5 `--no-switch` | `d01:main()` L1662 | 关二次切换(排错用) |

### F7 — 两段式指向理解(POINT-02)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F7.1 judge 路由 | `ChatCallback.on_event` L369-381 | 1.2s 内见伸指 → mode="judge"; 否则 mode="scene" |
| F7.2 judge 轮(VLM) | `snapshot_loop` L593-614 | Qwen-VL 输出 JSON: {pointing, target_visible, direction, desc} |
| F7.3 升级转头 | `snapshot_loop` L609-613 | pointing + !visible + direction → st.point_request → POINTING |
| F7.4 POINTING 状态机 | `behavior_loop` L1355-1385 | turn → settle(0.6s) → hold(抓帧) → RETURNING |
| F7.5 identify_pointed_object | `ChatCallback.on_event` L401-405 | 用户调用 → 恒走 judge 模式 |
| F7.6 VLM 方向映射 | `config:_DIR_MAP` L269-274 | 8 向: 左/右/上/下/左上/右上/左下/右下 → (yaw_offset, pitch_offset) |

### F8 — 看图(视觉理解)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F8.1 take_snapshot 工具 | `ChatCallback.on_event` L369-381 | 共享帧 → 640×360 jpg → Qwen-VL chat.completions |
| F8.2 scene 模式 | `snapshot_loop` L570-586 | 普通场景描述(SNAP_PROMPTS["scene"]) |
| F8.3 point 模式 | `snapshot_loop` L570-586 | 指向理解(SNAP_PROMPTS["point"]), 转头后第二轮 |
| F8.4 共享帧机制 | `frame_pump_loop` L663-664 | st.latest_frame 共享, ≤1.0s 新才用; 否则直接抓帧兜底 |
| F8.5 快照存盘 | `snapshot_loop` L562 | SNAP_DIR 下 `snapshot_01.jpg` ... |

### F9 — 逗它跟手(PLAY-01)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F9.1 进入条件 | `behavior_loop` L1122-1141 | 近手(score≥0.6 + size≥0.30) + 晃动(0.8s 位移≥0.08) + 持续 0.3s |
| F9.2 跟手积分 | `vision_result_loop` L793-828 | τ=0.25, 步进 3.0, 幅度 ×0.9, β=0.25 高速低延迟 |
| F9.3 惯性外推 | `vision_result_loop` L821-827 | 丢检 ≤0.35s 用 One Euro 速度外推(封顶小步) |
| F9.4 退出条件 | `behavior_loop` L1341-1353 | 手离开 1.5s / 手静止 4s → RETURNING |
| F9.5 开心表达 | `head_control_loop` L1509-1525 | 进入不动天线; 持续逗 5s → 小摇天线; 每 ~7s 一次 |
| F9.6 握拳停止 | `behavior_loop` L1327-1331 | GESTURE-01: fist 手势 → 停止互动 |
| F9.7 底部过滤 | `vision_result_loop` L753 | v > 0.80 → 排除桌面/衣物误检 |

### F10 — 手势动作(function calling)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F10.1 nod 点头 | `actions:act_nod` | 2 次上下 ±15°/10°, 350ms |
| F10.2 shake_head 摇头 | `actions:act_shake` | 2 次左右 ±15°, 350ms |
| F10.3 look_left/right/up/down | `actions:_look` | 偏 16°, 停 0.8s, 回基准 |
| F10.4 wiggle_antennas 摆天线 | `actions:act_wiggle` | 2 次 ±0.8rad 交替, 300ms |
| F10.5 tilt_head 歪头 | `actions:act_tilt` | roll 15°, 停 0.8s, 回基准 |
| F10.6 乐观即时回 output | `ChatCallback.on_event` L434-442 | 不等做完就回 output → 边说边动 |
| F10.7 手势基准 = 跟随姿态 | `motion_loop` L506-508 | 进场读 track_yaw/pitch + body_yaw, 做完回基准 |

### F11 — 头部控制渲染(唯一写入口)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F11.1 idle 微动 | `head_control_loop` L1545-1550 | 说话时 sin 微摇: yaw ±2.5°@0.20Hz, pitch ±1.5°@0.30Hz |
| F11.2 跟随时缩微动 | `head_control_loop` L1547 | tracked → scale 40% |
| F11.3 cue 渲染 | `head_control_loop` L1438-1478 | heard/fail/giveup/barge/bye: easeOutBack 攻击 + easeInQuad 衰减 |
| F11.4 cue 微变异 | `head_control_loop` L1440-1448 | ±15% 随机化(M3-a) |
| F11.5 思考歪头 | `head_control_loop` L1481-1487 | M3-b: 模型处理中 → roll 摆 + pitch 偏移 |
| F11.6 表情回应 | `head_control_loop` L1489-1493 | M3-b: 用户微笑 → 天线上扬; 皱眉 → 天线下压 |
| F11.7 ARMED 慢呼吸 | `head_control_loop` L1496-1507 | pitch sin ±2.5°@0.18Hz |
| F11.8 head pose 世界系 | `head_control_loop` L1568-1574 | body_yaw 被 Stewart 补偿, 大转向 head 给完整角 |
| F11.9 `--no-easing` | `d01:main()` L1664 | 关缓动, 回退 sin 包络 |
| F11.10 `--no-variation` | `d01:main()` L1665 | 关 cue 微变异 |
| F11.11 `--no-expression` | `d01:main()` L1666 | 关表情/思考反应 |

### F12 — 身份识别

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F12.1 YuNet 人脸检测 | `identity/recognizer.py:detect_and_recognize_all` L371-387 | OpenCV YuNet, conf=0.65, NMS=0.3 |
| F12.2 ArcFace embedding | `identity/recognizer.py:ArcFaceONNX` L80-98 | 112×112 aligned → 512d L2-norm |
| F12.3 5 点对齐 | `identity/recognizer.py:_align_face` L51-65 | affine warp 到 arcface 标准 |
| F12.4 FaceDB 匹配(质心+单体) | `identity/recognizer.py:FaceDB.match` L122-148 | 单 embedding 最佳 + 质心匹配取 max, cosine_threshold=0.35 |
| F12.5 新人确认帧 | `identity/recognizer.py:IdentityRecognizer.recognize` L307-359 | 3 帧 pending → 平均 embedding → 再匹配/入库 |
| F12.6 embedding 更新(多样性) | `identity/recognizer.py:FaceDB.update_embedding` L164-180 | max_sim>0.85 不加(太近), <0.20 不加(误匹配), 其余追加 |
| F12.7 集成限频 | `vision_result_loop` L727-746 | 有 face_box 且 >2s → 跑一次识别(不阻塞帧率) |
| F12.8 remember_fact → set_name | `ChatCallback.on_event` L418-423 | 用户说名字 → 同步 face_db |
| F12.9 auto_merge 启动自动合并 | `identity/recognizer.py:FaceDB.auto_merge` L254-285 | cross-sim > 0.50 → 合并 embeddings + 迁移名字 |
| F12.10 双命名保护 | `identity/recognizer.py:auto_merge` L271-272 | 两边都有 name 时跳过合并, 防止家庭成员误合并 |
| F12.11 merge_persons | `identity/recognizer.py:FaceDB.merge_persons` L230-252 | 合并 embeddings(去重 sim>0.90), 保留有名字/更早条目 |
| F12.12 _cross_sim | `identity/recognizer.py:FaceDB._cross_sim` L215-228 | 两人间所有 embedding 对的最大 cosine |
| F12.13 _face_key 网格 80px | `identity/recognizer.py:IdentityRecognizer._face_key` L304-305 | 80px 量化格防止边界碎片化 |

### F13 — 认知记忆管理(Cognitive Memory Architecture)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F13.1 Entity Memory (facts) | `memory/manager.py:MemoryManager` | `list[str]` 中文短句 + history + JSON 持久化 |
| F13.2 remember_fact 工具 | `memory/manager.py:QWEN_TOOLS[0]` + `realtime.py:on_event` | `fact`(中文短句) + `replaces`(关键词替换) + `name`(自报姓名) |
| F13.3 clear_memory 工具 | `memory/manager.py:QWEN_TOOLS[1]` + `memory/safety.py` | 安全删除工作流(多步验证+备份) |
| F13.4 forget_fact 工具 | `memory/manager.py:QWEN_TOOLS[3]` + `realtime.py:on_event` | `keyword` 模糊匹配删除 |
| F13.5 Working Memory 注入 | `memory/manager.py:get_prompt` + `realtime.py:update_memory` | Entity + Episodic 组装 → update_session instructions |
| F13.6 延迟注入 | `d01:main()` | 唤醒时识别未出 → 后续补注入 |
| F13.7 Session Consolidation | `realtime.py:save_summary` | 会话后 LLM 复盘: 全量对话 + draft facts → 最终 entity memory + episodic memory |
| F13.8 Episodic Memory | `memory/manager.py:save_episode` | 结构化事件(topic/highlights/mood), 保留最近 10 条 |
| F13.9 `--no-memory` | `d01:main()` | 关记忆系统, 剥离 remember_fact/clear_memory 工具 |
| F13.10 merge_memories | `memory/manager.py:merge_memories` | 人脸合并时迁移 facts(去重) + episodes(合并排序) |
| F13.11 owner 权限校验 | `memory/manager.py:clear_all/forget_fact` | actor_pid + OwnerManager.can_delete_memory 校验 |
| F13.12 consolidate_facts | `memory/manager.py:consolidate_facts` | LLM 复盘后整体替换 facts 列表 |
| F13.13 auto_merge 同步 | `d01:main()` 初始化 + `identity/recognizer.py:startup_merged` | FaceDB 合并碎片后自动调用 merge_memories 同步记忆 |
| F13.14 conversation_log 分桶 | `state.py:State.conversation_log` | `dict[str, list]` 按 pid 分桶，`"_unknown"` 暂存未识别人 |
| F13.15 上下文过长自动 consolidation | `realtime.py:on_event` user transcript | 估算 token > CONV_SUMMARY_THRESHOLD 自动触发后台 consolidation + 清桶 |
| F13.16 退出时遍历 consolidation | `d01:main()` 退出块 | 遍历所有剩余 pid 桶，逐个调 save_summary (consolidation) |
| F13.17 旧数据自动迁移 | `memory/manager.py:load_memory` | 检测旧 dict facts 格式 → 自动转换为 list[str] + episodes |
| F13.18 记忆归属=本句说话人 | `state.py:turn_speaker_pid` + `realtime.py:transcription.completed/response.created` | 记忆存(resp_snapshot/remember_fact)/读(update_memory)统一用 speaker_window 定的本句说话人,与飘的 current_person_id 解耦(后者只管焦点/显示)。用法:无需操作,说个人信息自动存给说话人 |
| F13.19 每轮工具审视兜底 | `realtime.py:RealtimeDialog.extract_memory_async` + `config.py:EXTRACT_MODEL` | 每轮用户说完无条件用 qwen-plus+最近5轮上下文抽"本句说话人"个人事实/姓名,save_fact 去重,兜底 realtime 漏调 remember_fact。用法:自动;`EXTRACT_MODEL` 环境变量可换模型 |
| F13.20 命名 guard(防脑补/画外/乱改名) | `realtime.py:try_name_identity` | 命名走统一 guard 三道门:①名字合法(1-8 中英文字)②**必须出现在当轮转写里**(防模型脑补,名字以 ASR 为准)③**已命名不静默覆盖**。**用法**:首次命名直接说『我叫X』;**改名必须显式说『我改名叫X / 我其实叫X / 叫错了我叫X』**,普通再说『我叫X』不会覆盖已有名字。画外/无归属说话人一律不命名 |
| F13.21 显示名实时取 | `d01:vision_result_loop`(dashboard 标签 + 焦点名) | 人脸名每帧从 `memory_mgr.get_name(pid)` 现取(模型用哪个库就显示哪个),改名后立即跟上,不再用识别那刻的缓存。用法:无需操作,dashboard 框名/🆔 日志即实时库状态 |
| F14.x 头部转向平滑 | `d01:vision_result_loop` 头部块 + `config.py:HEAD_*` | 头"看谁"独立于归属:引擎EMA上叠二级重EMA(`HEAD_ASD_EMA`)+黏滞(`HEAD_SWITCH_MARGIN`)+保持(`HEAD_HOLD_S`/DOA确信`HEAD_DOA_HOLD_S`)。治多人头部来回甩。参数全在 config 可调 |
| F14.y Dashboard 画框配色 | `debug_server.py:_build_frame` + `asd.py:speaking_ids` | 脸:绿=说话(speaking_ids带新鲜度门)/灰=跟踪;手:青=有效/橙=底部/黄=低置信。绿只给说话脸,蓝色已去除,ASD分2位小数 |
| F14.z DOA 瞟头(喊他能找人) | `d01:vision_result_loop` 头部积分块 + `config.py:DOA_GLANCE_DEG/GLANCE_MAX_DEG/GLANCE_MIN_TURN_DEG/GLANCE_LOCAL_RMS` | TRACKING 态,DOA确信+偏离>20°+画面无人说话+最近有人说话→转向声源找人,到位等说话,ASD锁到→按身份黏滞接管。**F1**:"有人说话"= realtime VAD **或** 本地麦响度>`GLANCE_LOCAL_RMS`(门控前,绕开>55°方向门控对侧边声音的静音死锁,喊大声生效)。**F4**:转角朝 DOA **符号**方向取 `max(\|resid\|, GLANCE_MIN_TURN_DEG=50)` 封顶 75°(DOA角度不可信只信符号,治"转向不够")。日志`👀 DOA 转头找人` |

### F21 — 音频闸门(切人身份保护)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F21.1 闸门标记 | `state.py:State.audio_gate_closed/buffer/closed_at` | 闸门关闭时缓存音频帧(b64)，不送模型 |
| F21.2 闸门触发条件 | `d01` 二次唤醒切人块 | 仅在切人+DOA声源偏移>SWITCH_AWAY_DEG时关闸，避免常规重连误触发 |
| F21.3 身份确认开闸 | `d01:_update_memory_instructions` | 记忆注入后 flush 缓存帧 + 开闸 |
| F21.4 超时兜底 | `d01` append_audio 处 | AUDIO_GATE_TIMEOUT_S(5s) 后强制开闸，防永久卡死 |
| F21.5 Dashboard 闸门状态 | `debug_server.py` state dict | audio_gate 字段 + 记忆行颜色指示(橙=闸门关) |

### F22 — Dashboard 上下文调试

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F22.1 payload modal 增强 | `debug_server.py:openModal` JS | 事件 payload 下方追加 Session Instructions + Memory Prompt + Conversation Log |
| F22.2 debug 状态字段 | `state.py:dbg_memory_prompt/dbg_session_instructions` | _update_memory_instructions 时保存注入内容供 dashboard 读取 |
| F22.3 state 端点扩展 | `debug_server.py` /state | 返回 session_instructions、memory_prompt、conversation_log(每人最近20条) |

### F23 — 多人脸 DOA 说话人选择

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F23.1 all_faces 输出 | `perception/vision_worker.py` result dict | YuNet/MediaPipe 检测到的所有脸 [{u,v,h,box,kps},...] 传到主进程 |
| F23.2 _select_face_by_doa | `d01:_select_face_by_doa` | DOA resid→摄像头坐标→匹配最近人脸，返回 all_faces 索引 |
| F23.3 DOA 覆盖身份识别 | `d01:vision_result_loop` 身份识别块 | 多人脸+DOA confident 时用 DOA 选出的脸做 ArcFace 识别，单人脸 fallback FaceSelector |
| F23.4 多人脸框渲染 | `debug_server.py:_build_frame` | 选中=蓝色粗框+DOA标签, 非选中=灰色细框 |

### F20 — 主人认定(认主)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F20.1 首次交互自动绑定 | `identity/owner.py:OwnerManager.try_claim` | 第一个被 remember_fact(name=xxx) 的人成为 owner |
| F20.2 owner 持久化 | `identity/owner.py` + `data/owner.json` | person_id + name + claimed_at |
| F20.3 can_delete_memory 权限 | `identity/owner.py:can_delete_memory` | actor==target 或 actor==owner → 允许 |
| F20.4 转让所有权 | `identity/owner.py:transfer` | owner 说"把你送给xxx" → 更换 owner(需 LLM function_call) |
| F20.5 合并同步 owner pid | `identity/owner.py:update_owner_pid` | 人脸合并后 old_pid→new_pid |
| F20.6 d01 集成认主 | `d01:ChatCallback.on_event` L426-428 | remember_fact(name) 成功 + 无 owner → try_claim |

### F14 — 手势识别

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F14.1 GestureRecognizer 模型 | `vision_worker` L~200 | MediaPipe GestureRecognizer, VIDEO 模式, num_hands=1, gesture_recognizer.task (~8MB float16) |
| F14.2 模型手势映射 | `vision_worker:_MODEL_GESTURE_MAP` | Closed_Fist→fist, Open_Palm→five, Pointing_Up→point, Victory→two, Thumb_Up→thumbup, Thumb_Down→thumbdown, ILoveYou→ily |
| F14.3 混合判定: 模型优先+规则fallback | `vision_worker` 检测循环 | 模型 score >= 0.6 → 用模型结果; 否则 fallback 到 _classify_gesture 规则 |
| F14.4 _classify_gesture 规则覆盖 | `vision_worker:_classify_gesture` | 补充模型不识别的: three, four, ok |
| F14.5 index_dir 食指方向 | `vision_worker:index_dir` L131-151 | 食指角度° + 是否伸出(相对手尺度) |
| F14.6 自适应提频 | `vision_worker` L280-299 | 近手(score≥0.6 + size≥0.22) → 2s 每帧跑手(跟手全帧率) |
| F14.7 晃动量统计 | `vision_result_loop` L769-777 | 0.8s 窗口位移极差(区分"逗"vs"托下巴") |
| F14.8 HandLandmarker fallback | `vision_worker` init | gesture_recognizer.task 不存在时回退到 HandLandmarker |
| F14.9 gesture_model debug 字段 | `vision_worker` 输出 dict | `gesture_model` + `gesture_model_score` 用于 debug |

### F15 — 行为状态机

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F15.1 8 状态 FSM | `behavior_loop` | ARMED/IDLE/ENGAGING/TRACKING/SEARCHING/RETURNING/POINTING/PLAYING |
| F15.2 唯一 state 写者 | `behavior_loop` | behavior_loop 是 st.state 唯一写者 |
| F15.3 五层仲裁 | `head_control_loop` + `motion_loop` | Primary > Playing > SoundTurn > Tracking > Idle |
| F15.4 手势期间暂停 | `behavior_loop` L995-997 | action_active → 状态计时冻结 |
| F15.5 无互动超时 | `behavior_loop` L1157, 1281 | 15s NO_INTERACT_S |

### F16 — 结束对话(EXIT-01)

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F16.1 end_session 工具 | `ChatCallback.on_event` L382-400 | 告别词嵌 function_call_output |
| F16.2 EXIT 流程 | `behavior_loop` L1036-1050 | exit_request → RETURNING + bye cue |
| F16.3 告别语轮换 | `config:BYE_PHRASES` | 7 句: "好的/拜拜/休息啦/..." |
| F16.4 退出不被锁回 | `behavior_loop` L1302-1315 | exiting=True 期间 locked 不拉回 TRACKING |
| F16.5 封顶退出 | `behavior_loop` L1308 | EXIT_MIN_S=1.5 / EXIT_MAX_S=6.0 |

### F17 — 视觉子进程

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F17.1 独立进程 | `d01:main()` L1848-1854 | multiprocessing.Process, 独立 GIL |
| F17.2 抓帧泵 | `d01:frame_pump_loop` L638-683 | 40Hz, BGR→RGB 降采样, maxsize=1 背压 |
| F17.3 USE_WEBCAM | `d01:frame_pump_loop` L643-649 | 仿真模式用 Mac 摄像头替代 MuJoCo |
| F17.4 ready 信号 | `vision_worker` L224 + `vision_result_loop` L714-717 | vis_ready 门控 SEEK |
| F17.5 blendshape 表情提取 | `vision_worker` L251-260 | mouthSmileLeft/Right + mouthFrownLeft/Right |

### F18 — 调试工具

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F18.1 VIS_DEBUG MJPEG | `debug_server:vis_debug_server` | HTTP 实时预览 + 检测标注 |
| F18.2 Conversation Dashboard | `debug_server` | HTML/JS 内联, 对话事件/轮次/状态/DOA 时间线 |
| F18.3 对话事件录制 | `state:_record_event/_record_snap_result/_record_vis_event` | 全量事件 → _conv_events deque |
| F18.4 NO_VOICE 模式 | `d01:main()` L168 + L1705-1717 | 跳过音频管线, 仅跑视觉 + 行为 |
| F18.5 运行时标志 | `d01:main()` L1659-1678 | --no-wake/--no-gate/--no-switch/--no-sticky/--no-easing/--no-variation/--no-expression/--no-memory/--simulate-conn-fail/--cue-xxx=n |
| F18.6 定时自动退出 | `d01:main()` L1875-1877 | 命令行秒数 → 编排测试 |

### F19 — 启动与环境

| 子特性 | 代码位置 | 说明 |
|--------|----------|------|
| F19.1 start_mac.sh | `start_mac.sh` | 找串口 → 启 daemon → 启主程序 → 等 ready → trap 清理 |
| F19.2 代理隔离 | `d01` L64-67 | NO_PROXY=localhost,...,.aliyuncs.com |
| F19.3 摄像头预热 | `d01:main()` L1721-1739 | 10 帧/10s deadline, 确认出帧 |
| F19.4 排空旧音频 | `d01:main()` L1861-1863 | 限时 3s drain |
| F19.5 退出回正 | `d01:main()` L2058-2063 | goto_target 回初始姿态, duration=1.5 |

---

## 二、测试方案

### 测试分层

| 层级 | 说明 | 依赖 |
|------|------|------|
| **L1 单元测试** | 纯逻辑, 无硬件/网络 | pytest |
| **L2 集成测试** | 模块间配合, mock 硬件 | pytest + mock |
| **L3 视觉离线测试** | 静态图片/录制视频 | mediapipe + opencv |
| **L4 硬件在环测试** | 真机 + 摄像头 + 麦克风 | Reachy Mini 实体 |
| **L5 端到端手测** | 人在机器人前完整交互 | 全系统 |

---

### T1 — 全双工语音对话

#### T1.1 打断机制 [L2]
```
前置: mock mini.media, mock OmniRealtimeConversation
步骤:
  1. 模拟 speech_started 事件 → 验证 play_q 被清空
  2. 验证 play_gen 递增(代际作废)
  3. 验证 in_flight>0 时 cancel_response 被调用
  4. 验证 barge cue 被设置(wake_cue="barge")
预期: play_q 清空; drop_audio=True; cancel_response 仅在 in_flight 时调
```

#### T1.2 下行音频链路 [L1]
```
步骤:
  1. 24kHz PCM16 base64 → resample_poly → 16kHz float32
  2. 验证重采样比例正确(24000→16000)
  3. 验证 gen 匹配(被打断的旧 gen 音频不进 play_q)
预期: 输出长度 = 输入长度 × 16000/24000; gen 不匹配时丢弃
```

#### T1.3 标签泄漏清洗 [L1]
```
步骤:
  1. transcript = "好的!<nod>" → _ACTION_TAG_RE 匹配
  2. 验证 motion_q 收到 nod
  3. 验证清洗后 transcript 无标签
  4. 测试多标签: "<shake_head>没有<tilt_head>"
预期: 所有标签被替换; 对应动作被分发; 中文标签("点头/摇头")也被捕获
```

#### T1.4 自动重连 [L2]
```
前置: mock conv.append_audio 抛 WebSocketConnectionClosedException
步骤:
  1. append_audio 异常 → close_session
  2. open_session 成功 → conv 更新
  3. open_session 失败 → conv=None, 等下次唤醒
预期: 不崩溃; 重连成功后继续上行
```

### T2 — 唤醒词门控

#### T2.1 KWS 基本唤醒 [L1]
```
步骤:
  1. 构造含"小艺"的 16kHz mono float32 音频
  2. 分块(256 samples)feed 给 KwsGate
  3. 验证某块返回 True(命中)
  4. 紧接再 feed → 不应期内返回 False
预期: 命中一次; 2s 不应期内不再命中
```

#### T2.2 去抖 [L1]
```
步骤:
  1. 两次 feed 间隔 < KWS_DEBOUNCE_S(0.3s)
  2. 验证第二次被去抖
预期: 仅第一次通过
```

#### T2.3 ARMED→ENGAGING→TRACKING 流程 [L2]
```
前置: mock State, mock open_session
步骤:
  1. st.state=ST_ARMED, KWS 命中 → st.wake_ok=True
  2. behavior_loop tick → ST_ENGAGING
  3. 模拟 face_locked=True → ST_TRACKING
  4. 验证 greet_now=True(招呼)
预期: 状态转换正确; greet_now 仅在首次唤醒锁脸时 True
```

#### T2.4 无互动回待命 [L2]
```
步骤:
  1. st.state=ST_IDLE, last_interaction_at = now - 16s
  2. behavior_loop tick
  3. wake_mode=True → ST_ARMED
预期: 15s 无互动 → ARMED
```

### T3 — SEEK 寻人

#### T3.1 三阶段扫描 [L2]
```
步骤:
  1. confident DOA → seek_phase="direct"
  2. 到位后 → "nearby"(±25° 扫)
  3. 2.5s 无脸 → "full"(±88° 全场扫)
  4. 7s 无脸 → giveup → ARMED
预期: 各阶段切换时刻正确
```

#### T3.2 direct 途中不认脸 [L2]
```
步骤:
  1. seek_phase="direct", |seek_target| ≥ 12°
  2. 模拟 face_locked=True → 应被压住
  3. seek_target < 12° → 正前豁免, 认脸
预期: 大角度直转不被脸拽; 小角度放行
```

### T4 — 人脸跟踪

#### T4.1 FaceSelector 粘滞 [L1]
```
步骤:
  1. 模拟 2 张脸, A 在 (0.3, 0.5, 0.15), B 在 (0.7, 0.5, 0.14)
  2. 首帧选 A(最大); 第 2 帧 A/B 位置不变 → 粘住 A
  3. B 变 0.20(> A×1.20) 连续 8 帧 → 切换到 B
预期: 粘滞不跳; 8 帧后切换
```

#### T4.2 One Euro 滤波 [L1]
```
步骤:
  1. 输入锯齿波(0→1→0), dt=1/30
  2. 输出应平滑(无锯齿)
  3. 输入阶跃(0→1) → 输出快速跟上(低延迟)
  4. reset 后重入 → 无跳变
预期: 低速平滑; 高速低延迟; reset 正确
```

#### T4.3 迟滞锁定 [L1]
```
步骤:
  1. 连续 face=True, 计时 0→0.3s → locked 从 False 变 True
  2. 连续 face=False, 计时 0→1.5s → locked 从 True 变 False
  3. 短暂丢失(<1.5s)再检出 → locked 不变
预期: 0.3s on / 1.5s off 边界正确
```

#### T4.4 时间常数增益 [L1]
```
步骤:
  1. err=10°, dt=0.04s, τ=0.40
  2. step = 10 × (1 - exp(-0.04/0.40)) ≈ 0.95°
  3. 验证在 [-TRACK_MAX_STEP, TRACK_MAX_STEP] 范围内
预期: 计算精度 ±0.01°
```

### T5 — 听声转向

#### T5.1 DOA 门控逻辑 [L1]
```
步骤:
  1. fresh=True, confident=True, |resid|=70° → gate_open=False
  2. |resid|=40° → gate_open=True(范围内)
  3. not fresh → gate_open=True(不新鲜不管)
  4. not confident → gate_open=True
预期: 仅 fresh + confident + |resid| > 55° 才关门
```

#### T5.2 声源转向触发 [L2]
```
步骤:
  1. state=ST_IDLE, face_locked=False, speaking=False
  2. _fresh_sound 返回 resid=40°
  3. → ST_ENGAGING, engage_target = track_yaw + 40°
预期: 转向目标正确; speaking 时不触发
```

### T6 — 二次唤醒切换

#### T6.1 判断"非A方向" [L1]
```
步骤:
  1. fresh=True, confident=True, |resid|=30° → 是 A(≤55°), 不切换
  2. |resid|=60° → 非 A, 切换
  3. not fresh → 切换(不确信 = 假定非 A)
预期: 阈值 GATE_DEG=55° 正确分流
```

### T7 — 两段式指向

#### T7.1 judge JSON 解析 [L1]
```
步骤:
  1. 合法 JSON: '{"pointing": true, "target_visible": false, "direction": "左", "desc": "..."}'
  2. 带代码块: '```json\n{...}\n```'
  3. 前后有文字: '看起来...\n{...}\n综上...'
  4. 非法 JSON: 'I cannot determine...'
预期: 1-3 都能解析; 4 返回 None
```

#### T7.2 指向路由 [L2]
```
步骤:
  1. 1.2s 内有 finger_ext_at → mode="judge"
  2. 超过 1.2s → mode="scene"
  3. identify_pointed_object → 恒 "judge"
预期: 路由正确
```

#### T7.3 升级转头 [L2]
```
步骤:
  1. judge 结果: pointing=True, visible=False, direction="左"
  2. → st.point_request 被设置
  3. behavior_loop → ST_POINTING
  4. turn → settle(0.6s) → hold → snapshot "point" → RETURNING
预期: 完整子阶段流转; point_request 在完成后清除
```

### T8 — 看图

#### T8.1 共享帧读取 [L2]
```
步骤:
  1. st.latest_frame 有帧, latest_frame_t < 1.0s → 使用共享帧
  2. latest_frame_t > 1.0s → 退回直接抓帧(3 帧)
  3. 完全无帧 → "拍照失败"
预期: 优先共享; 超时回退; 无帧报错
```

### T9 — 逗它跟手

#### T9.1 进入条件 [L2]
```
步骤:
  1. hand_size=0.35, score=0.7, hand_move=0.10, 持续 0.4s → 进入 PLAYING
  2. hand_size=0.20(太小) → 不进入
  3. score=0.5(低分) → 不进入
  4. hand_move=0.05(不动, 托下巴) → 不进入
预期: 三门全过 + 持续 0.3s 才进入
```

#### T9.2 退出条件 [L2]
```
步骤:
  1. 手离开 1.5s → RETURNING
  2. 手不动 4s → RETURNING
  3. 握拳(gesture="fist") → RETURNING
预期: 三种退出路径都正确
```

### T10 — 手势动作

#### T10.1 动作基准 [L1]
```
步骤:
  1. base_yaw=10, base_pitch=5, body=20
  2. act_nod → 动作完回到 gpose(10, 5, 20)
  3. body_yaw 传 0 vs 传实际值 → 结果不同
预期: 回到传入基准; body≠0 时不拽回正前
```

#### T10.2 gpose 裁剪 [L1]
```
步骤:
  1. yaw=50, body=20, GES_YAW_BOX=25 → yaw clipped to [20-25, 20+25]=[−5, 45] → 45
  2. pitch=20, GES_PITCH_BOX=16 → 16
预期: 不超出 box
```

### T11 — 头部控制渲染

#### T11.1 Cue 缓动曲线 [L1]
```
步骤:
  1. wake_cue="heard", t_norm=0→1 采样
  2. 攻击段(0→0.35): easeOutBack
  3. 衰减段(0.35→1): easeInQuad
  4. --no-easing → sin 包络
预期: 曲线形状正确; 参数变异在 ±15%
```

#### T11.2 ARMED 呼吸 [L1]
```
步骤:
  1. state=ST_ARMED
  2. pitch 应在 ±2.5° sin 波(0.18Hz)
预期: 波形正确
```

### T12 — 身份识别

#### T12.1 ArcFace embedding [L3]
```
前置: 测试图片(正脸/侧脸)
步骤:
  1. 同一人两张图 → embedding cosine > 0.5
  2. 不同人两张图 → cosine < 0.35
预期: 阈值正确区分
```

#### T12.2 新人确认帧 [L1]
```
前置: mock ArcFaceONNX
步骤:
  1. 未知脸, 第 1-2 帧 → return None(pending)
  2. 第 3 帧 → 平均 embedding 再匹配; 仍未知 → add_person, is_new=True
预期: 3 帧确认; embedding 平均后 L2-norm
```

#### T12.3 embedding 更新策略 [L1]
```
步骤:
  1. 已有 5 个 embedding, 新 embedding max_sim=0.90 → 不加(太近, >0.85)
  2. max_sim=0.65 → 加入(新角度)
  3. max_sim=0.15 → 不加(太远, <0.20, 可能是误匹配)
  4. 超过 max(10) → FIFO 弹出最旧
预期: 策略正确
```

#### T12.4 质心匹配 [L1]
```
步骤:
  1. 3 个 embedding 分布在不同角度(sim 0.40~0.60 之间)
  2. 新 embedding 与任意单体 sim < 0.35, 但与质心 sim > 0.35
  3. match() 返回该 person_id
预期: 质心匹配弥补单体匹配的大角度缺失
```

#### T12.5 auto_merge 基本合并 [L1]
```
步骤:
  1. 两个未命名 person, cross-sim = 0.55 (> 0.50) → 合并
  2. 合并后 person 数减 1, 保留 embedding 多的
  3. 返回 {drop_pid: keep_pid} 映射
预期: 合并正确; embedding 去重(sim>0.90 视为重复)
```

#### T12.6 auto_merge 双命名保护 [L1]
```
步骤:
  1. person_A(name="爸爸") 和 person_B(name="妈妈"), cross-sim = 0.55
  2. auto_merge → 不合并(两边都有名字)
  3. person_C(无名字) 和 person_A(name="爸爸"), cross-sim = 0.55 → 合并(匿名→有名)
预期: 双命名跳过; 匿名→有名正常合并
```

#### T12.7 merge_persons 字段保留 [L1]
```
步骤:
  1. keep 无名字, drop 有名字 → 合并后 keep 继承 drop 的名字
  2. keep 和 drop 都有名字 → 保留 keep 的名字
  3. drop 的 created_at 更早 → 合并后 keep 使用 drop 的 created_at
  4. last_seen_at 取两者最晚
预期: 字段保留策略正确
```

### T13 — 记忆管理

#### T13.1 LWW 读写 [L1]
```
步骤:
  1. save_fact("p1", "name", "小明") → 磁盘 JSON 有 facts.name="小明"
  2. save_fact("p1", "name", "小红") → 覆盖, history 记录 old_value
  3. clear_all("p1", confirmed=True) → facts 清空, history 有 clear_all 记录
  4. clear_all("p1", confirmed=False) → 拒绝
预期: LWW 语义正确; history 封顶 200→100
```

#### T13.2 记忆注入 prompt [L1]
```
步骤:
  1. facts = {"name": "小明", "age": "25"} → prompt 包含 "你面前的人叫小明" + "age: 25"
  2. facts = {} → return None
预期: prompt 格式正确; 空时不注入
```

#### T13.3 forget_fact 模糊匹配 [L1]
```
步骤:
  1. facts = {"name": "小明", "favorite_food": "火锅", "hobby": "游泳"}
  2. forget_fact(pid, "food") → 删除 favorite_food(key 包含 food)
  3. forget_fact(pid, "xyz") → 不匹配, 返回提示
预期: 模糊匹配正确; 无匹配时返回已有 key 列表
```

#### T13.4 merge_memories [L1]
```
步骤:
  1. keep 有 {name: "小明"}, drop 有 {name: "小红", hobby: "游泳"}
  2. merge_memories(keep, drop) → keep 新增 hobby(不覆盖 name)
  3. drop 的记忆被清空
预期: keep 优先不覆盖; drop 清空
```

#### T13.5 owner 权限校验 [L1]
```
前置: OwnerManager 有 owner_pid="p1"
步骤:
  1. actor=p1, target=p2 → clear_all → 允许(owner 删他人)
  2. actor=p2, target=p2 → clear_all → 允许(删自己)
  3. actor=p2, target=p1 → clear_all → 拒绝("只有主人才能删除其他人的记忆")
  4. actor=p2, target=p3 → forget_fact → 拒绝
预期: 权限矩阵正确
```

### T14 — 手势识别

#### T14.1 GestureRecognizer 模型手势 [L3]
```
前置: tests/fixtures/ 手势图片 + gesture_recognizer.task 模型
步骤:
  1. hand_fist.jpg → GestureRecognizer 返回 Closed_Fist → 映射为 "fist"
  2. hand_five.jpg → Open_Palm → "five"
  3. hand_point.jpg → Pointing_Up → "point"
  4. hand_two.jpg → Victory → "two"
  5. 验证 score >= 0.6 时使用模型结果
预期: 模型手势映射正确
```

#### T14.2 混合判定: 模型+规则 [L3]
```
步骤:
  1. 模型 score >= 0.6, gesture=Closed_Fist → 用模型结果 "fist"
  2. 模型 score = 0.3(低) → fallback 到 _classify_gesture 规则
  3. 模型返回 None/Unknown → fallback 到规则
  4. hand_three.jpg → 模型无此手势 → 规则返回 "three"
  5. hand_ok.jpg → 模型无此手势 → 规则返回 "ok"
预期: 模型优先; 低分/未知 fallback; 规则补充 three/four/ok
```

#### T14.3 HandLandmarker fallback [L2]
```
步骤:
  1. gesture_model 路径不存在 → 回退到 HandLandmarker
  2. HandLandmarker 正常检测 → 纯规则 _classify_gesture
预期: 无模型文件时平滑降级
```

#### T14.4 index_dir 角度 [L1]
```
步骤:
  1. 构造 21 点 landmark, 食指水平向右 → angle≈0°
  2. 食指垂直向上 → angle≈-90°
  3. extended 判定: seg > 0.30×hand_size + seg>0.025 + cosang>0.6
预期: 角度计算精度 ±5°; extended 阈值正确
```

### T15 — 行为状态机

#### T15.1 完整状态转换 [L2]
```
步骤: 模拟一次完整生命周期:
  ARMED → (唤醒) → ENGAGING → (锁脸) → TRACKING
  → (丢脸) → SEARCHING → (声音) → ENGAGING
  → (锁脸) → TRACKING → (逗它) → PLAYING
  → (手走) → RETURNING → (回中) → IDLE → (无互动) → ARMED
预期: 每一步状态值正确; set_state 日志正确
```

#### T15.2 指向插入 [L2]
```
步骤:
  1. state=ST_TRACKING, point_request≠None
  2. → ST_POINTING(打断跟踪)
  3. turn → settle → hold → RETURNING
  4. 锁脸 → 回 TRACKING
预期: POINTING 最高优先; 完成后恢复
```

### T16 — 结束对话

#### T16.1 EXIT 流程 [L2]
```
步骤:
  1. ChatCallback 收到 end_session → exit_request=True
  2. behavior_loop → ST_RETURNING + bye cue
  3. 回中 + 告别播完 + >1.5s → ST_ARMED
  4. 退出期间 locked=True → 不被拉回 TRACKING
预期: 流程正确; 封顶 6s
```

### T17 — 视觉子进程

#### T17.1 子进程协议 [L2]
```
步骤:
  1. 启动 vision_worker → 首条消息 {"kind": "ready"}
  2. 发 (t, rgb) → 返回 {"kind": "det", ...}
  3. 发 None → 进程退出
预期: 协议完整
```

#### T17.2 自适应手部检测频率 [L2]
```
步骤:
  1. 无近手 → HAND_EVERY(4) 帧跑一次
  2. 近手(score≥0.6, size≥0.22) → 2s 内每帧跑
  3. 近手消失 2s → 恢复降频
预期: 频率切换正确
```

### T18 — 调试工具

#### T18.1 Dashboard 事件录制 [L2]
```
步骤:
  1. _record_event 各类型 → _conv_events 有对应条目
  2. _conv_turns 有轮次聚合
  3. seq 单调递增
预期: 录制完整; 不影响主逻辑
```

### T19 — 主人认定(认主)

#### T19.1 首次认主 [L1]
```
前置: data/owner.json 不存在或为空
步骤:
  1. try_claim("p1", "小明") → 返回 True, owner_pid="p1"
  2. try_claim("p2", "小红") → 返回 False(已有 owner)
  3. is_owner("p1") → True
  4. is_owner("p2") → False
预期: 先到先得; 不可重复认主
```

#### T19.2 权限判定 [L1]
```
步骤:
  1. can_delete_memory("p1", "p1") → True(删自己)
  2. can_delete_memory("p1", "p2") → True(owner 删他人)
  3. can_delete_memory("p2", "p2") → True(删自己)
  4. can_delete_memory("p2", "p1") → False(非 owner 删他人)
  5. can_delete_memory("p3", "p2") → False
预期: 权限矩阵正确
```

#### T19.3 所有权转让 [L1]
```
步骤:
  1. owner = p1, transfer("p2", "小红")
  2. is_owner("p2") → True
  3. is_owner("p1") → False
  4. owner.json 有 transferred_from 字段
预期: 转让后新 owner 生效; 旧 owner 失权
```

#### T19.4 合并同步 owner [L1]
```
步骤:
  1. owner = p1, update_owner_pid("p1", "p_merged")
  2. is_owner("p_merged") → True
  3. is_owner("p1") → False
预期: 人脸合并后 owner pid 正确更新
```

#### T19.5 d01 认主集成 [L2]
```
前置: mock OwnerManager + MemoryManager + IdentityRecognizer
步骤:
  1. remember_fact(key="name", value="小明") + 无 owner → try_claim 被调用
  2. 再次 remember_fact 另一人 → try_claim 不被调用(已有 owner)
  3. clear_memory 由非 owner 调用 → 返回 "只有主人才能删除其他人的记忆"
预期: 认主触发时机正确; 权限拦截生效
```

---

## 三、测试基础设施建议

### 3.1 目录结构
```
tests/
├── conftest.py              # pytest fixtures: mock State, mock mini, mock conv
├── unit/
│   ├── test_one_euro.py     # T4.2
│   ├── test_face_selector.py# T4.1
│   ├── test_hysteresis.py   # T4.3
│   ├── test_classify_gesture.py  # T14.1, T14.2
│   ├── test_parse_judge.py  # T7.1
│   ├── test_gpose_clip.py   # T10.2
│   ├── test_memory_manager.py # T13.1, T13.2, T13.3, T13.4, T13.5
│   ├── test_owner.py        # T19.1, T19.2, T19.3, T19.4
│   ├── test_gate_logic.py   # T5.1, T6.1
│   ├── test_time_constant.py # T4.4
│   ├── test_face_db.py      # T12.4, T12.5, T12.6, T12.7
│   └── test_tag_cleanup.py  # T1.3
├── integration/
│   ├── test_barge_in.py     # T1.1
│   ├── test_wake_flow.py    # T2.3, T2.4
│   ├── test_seek.py         # T3.1, T3.2
│   ├── test_behavior_fsm.py # T15.1, T15.2
│   ├── test_pointing.py     # T7.2, T7.3
│   ├── test_play.py         # T9.1, T9.2
│   ├── test_exit.py         # T16.1
│   └── test_reconnect.py    # T1.4
├── vision/
│   ├── test_arcface.py      # T12.1
│   ├── test_identity.py     # T12.2, T12.3
│   ├── test_gesture_images.py # T14.1 (静态图片)
│   └── test_gesture_hybrid.py # T14.2, T14.3 (模型+规则混合)
└── fixtures/
    ├── face_front.jpg
    ├── hand_fist.jpg
    └── ...
```

### 3.2 核心 Fixtures

```python
# conftest.py
@pytest.fixture
def state():
    """干净的 State 实例"""
    return State()

@pytest.fixture
def mock_mini():
    """mock ReachyMini (无真实硬件)"""
    m = MagicMock(spec=ReachyMini)
    m.media.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    m.media.get_audio_sample.return_value = np.zeros((256, 1), dtype=np.float32)
    return m

@pytest.fixture
def mock_conv():
    """mock OmniRealtimeConversation"""
    c = MagicMock()
    c.create_item = MagicMock()
    c.create_response = MagicMock()
    c.cancel_response = MagicMock()
    return c
```

### 3.3 测试隔离标志

| 标志 | 测试用途 |
|------|----------|
| `NO_VOICE=1` | 跳过音频, 只测视觉 + 行为 |
| `VIS_DEBUG=1` | 开 MJPEG 预览(手测观察) |
| `--no-wake` | 跳过 KWS, 启动即对话 |
| `--no-gate` | 全向上行(DOA 不挡) |
| `--no-switch` | 关切换(简化场景) |
| `--no-sticky` | 关粘滞(对比基线) |
| `--no-expression` | 关表情(排除干扰) |
| `--no-memory` | 关记忆(排除干扰) |
| `--simulate-conn-fail` | 测连接失败 cue |

### 3.4 优先级

| 优先级 | 测试 | 原因 |
|--------|------|------|
| P0 | T4.1-T4.4(跟踪核心) | 重构最易破坏 |
| P0 | T15.1(状态机全程) | 行为主干 |
| P0 | T1.1(打断) | 对话基本功能 |
| P1 | T7.1-T7.3(指向) | 复杂链路 |
| P1 | T9.1-T9.2(逗它) | 进出条件精细 |
| P1 | T2.1-T2.4(唤醒) | 入口链路 |
| P1 | T12.1-T12.7(身份+合并) | 数据正确性, 防记忆泄漏 |
| P1 | T19.1-T19.5(认主) | 权限安全 |
| P2 | T13.1-T13.5(记忆+权限) | 纯逻辑易测 |
| P2 | T14.1-T14.4(手势混合) | 模型+规则双路径 |
| P2 | T11.1-T11.2(渲染) | 参数验证 |
| P3 | T18.1(Dashboard) | 只读, 低风险 |
