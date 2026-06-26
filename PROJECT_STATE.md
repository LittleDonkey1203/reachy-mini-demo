# PROJECT_STATE

## 已完成事项

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
- **Entity Memory**: facts 从 `{key:value}` dict 改为 `list[str]` 中文短句，支持 `replaces` 关键词替换和 `keyword` 模糊删除
- **Episodic Memory**: 替代 conversation_summaries，存储结构化事件(topic/highlights/mood)
- **Working Memory 注入**: get_prompt() 从 entity + episodic 组装自然语言注入
- **Session Consolidation**: 会话结束后 LLM 复盘，从全量对话 + draft facts 生成最终 entity memory + episodic memory
- **QWEN_TOOLS 更新**: remember_fact(fact, replaces?, name?) / forget_fact(keyword)
- **旧数据自动迁移**: load_memory 自动检测旧 dict 格式并转换

### Bug 修复 (2026-06-25 晚)
- **#22 Consolidation 只跑一个人**: close_session 改为遍历所有 pid 桶，每个有 ≥2 条对话的人都启动独立 consolidation 线程
- **#23 Consolidation 质量差**: 优化 prompt — 明确要求从对话中提取新 facts、排除 name 字段冗余、扩展对话截取到 4000 字符
- **#24 身份切换过于频繁**: CONFIRM_N 2→3，新增 ID_SWITCH_COOLDOWN_S=6s 切人冷却，防止多人坐一起时来回弹
- **#20 身份快照 + response.done 才切人**: response.created 时快照当前 pid/name，回复期间所有 conv_log/display_transcript/function_call 归属用快照值，response.done 后清空快照放行 update_session
- **Dashboard 上下文重建**: modal 里旧的分离视图（Session Instructions / Memory Prompt / Conversation Log）合并为"模型视角完整上下文"——[System] instructions+memory → [User] → [ToolCall] → [Assistant] → [Tools]，function_call 事件也记录到 display_transcript

## 当前架构状态

```
voice/
  config.py        — 常量 + 工具元数据 + prompt
  state.py         — State 类 + log + OneEuroFilter
  d01_realtime_chat.py — 主程序 (~600 行，已瘦身)
  debug_server.py  — Dashboard
  kws.py           — 唤醒词门控
  realtime.py      — Qwen-Omni-Realtime 协议层 + Session Consolidation
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

### 记忆生命周期
```
会话中:
  remember_fact(fact, replaces?) → draft notes 实时存盘
  forget_fact(keyword) → 模糊匹配删除
  identity_injected=False → 触发重注入最新 facts
会话后 (close_session):
  save_summary() → LLM consolidation:
    输入: 全量对话 + draft facts + 已有 facts
    输出: 最终 entity memory + episodic memory
下次对话:
  get_prompt(pid) → 从 entity + episodic 组装注入 Working Memory
```

- 9 状态 FSM: ARMED/IDLE_CENTER/ENGAGING/TRACKING/SEARCHING/RETURNING/POINTING/PLAYING
- 5 层运动仲裁: Primary > Playing > SoundTurn > Tracking > Idle

## 遗留问题

1. **YuNet 无 blendshapes**: smile/frown 恒 0.0, 可用 insightface 2D106 估算
2. **多人同框介绍**: 指着他人说"这是XX" → 关联名字(方案见 docs/MULTI_PERSON_INTRO_PLAN.md)
3. **end_session 乱码**: 模型偶尔把 function_call_output 当文字朗读
4. **Semantic Memory**: 需 Consolidation Engine 从多条 episode 回放抽象知识(未来)

## 下一步建议

1. 真机测试验证认知记忆架构
2. 继续 todo.md 未完成项(#1 DOA / #7 身份优化 / #9 对话质量 / #20 take_snapshot 时延)
3. Semantic Memory 层 — 从 episodes 抽象知识 + GraphDB
