# Reachy Mini Demo — TODO 清单

> 生成日期: 2026-06-22 | 基于上一轮 session 讨论 + 代码现状核对

---

## 状态说明

- ✅ 已完成
- ⚠️ 部分完成
- ❌ 待处理

---

## 1. 唤醒 DOA 能力优化 ❌

**问题**: 唤醒后经常转到相反方向或不动，怀疑多人脸/手跟踪逻辑干扰。

**现状**:
- DOA 已有 IQR 置信度滤波 + 多层回退 (confident→coarse→mirror/stale→visual SEEK)
- `audio/sound_turn.py` 有残差跟踪: `target = current_heading + (90° − DOA_angle)`
- `d01_realtime_chat.py` 中 `doa_sensor_loop` 10Hz REST 轮询 + median filter

**待分析**:
- [ ] 排查唤醒时 DOA 角度是否被视觉跟踪目标覆盖
- [ ] 确认多人脸场景下 FaceSelector sticky 选择是否干扰 DOA 转向
- [ ] 检查 DOA→head_control 的优先级仲裁是否正确

---

## 2. 视觉在 mac 上的能力差 ✅

**已完成**: MediaPipe 后端已集成 (`perception/vision_worker.py`)，FaceLandmarker + HandLandmarker VIDEO 模式，FaceSelector sticky 选择。

---

## 3. 空闲状态呼吸感 + 官方表情库 ⚠️

**已完成部分**:
- M3 (EXP-A): 8 态参数化呼吸 (ARMED/IDLE_CENTER/ENGAGING/TRACKING/SEEKING/SEARCHING/PLAYING/RETURNING)
- 呼吸频率和幅度已按状态区分

**未完成**:
- [ ] 长时间无交互时的随机小动作（偶尔抬头、左右看、耳朵小幅扇动等）
- [ ] 接入并扩展官方表情库（Reachy Mini SDK 的 blendshape 表情系统）
- [ ] 让 idle 动作有变化感而不是单调循环

---

## 5. 视觉搜寻 / 寻物 Observer ❌

**问题**: 需要规划主动寻物能力。

**现状**:
- `take_snapshot` + `identify_pointed_object` 两个 VLM 工具已有
- 两阶段指向: judge round (VLM 判断是否在指东西) → point round (转头确认目标)
- `snapshot_loop` 线程持续读取最新帧

**待规划**:
- [ ] 设计主动寻物 Observer 流程（用户说"帮我找X" → 系统化搜索 → 汇报）
- [ ] 头部扫描策略（系统化扫描路径 vs 随机搜索）
- [ ] VLM 查询优化（针对特定物品的 prompt 模板）

---

## 6. 接入 RAG ❌

**问题**: 问问题时根据 query 触发搜索再回答。

**现状**:
- 当前只有 per-person JSON fact 记忆 (`memory/manager.py`)
- 无 vector store，无检索增强

**待规划**:
- [ ] 确定 RAG 知识源（本地文档? 网络搜索? 产品知识库?）
- [ ] 选型: 本地 embedding + FAISS / 远端向量数据库
- [ ] 设计 query → retrieve → inject prompt 的流程
- [ ] 与现有 Qwen Realtime session 的集成方式

---

## 7. 身份识别方案优化 ⚠️

**已完成部分**:
- YuNet + ArcFace 识别链就位 (`identity/recognizer.py`)
- `face_db.json` 持久化，cosine threshold=0.35，多 embedding per person (max 10)
- 3-frame confirmation for new persons
- `remember_fact(key="name")` 同步 face DB (commit `8b9963b`)

**未完成 — 核心问题: 人脸入库不应默认触发**:
- [ ] **说话人判定机制**: 只有"看着这个人 + 这个人在和你对话"才触发入库
  - 需要: 视觉确认（人脸朝向机器人 / 正在注视）
  - 需要: 语音关联（VAD + DOA 方向与人脸方向一致 → 判断是这个人在说话）
- [ ] **入库时机**: 对话发生后才记忆，而非检测到人脸就入库
- [ ] **多人场景**: 多张人脸时，通过 DOA + 视觉综合判断谁在说话

---

## 8. 渐进式记忆加载 ❌

**问题**: 识别到人后应渐进式加载相关记忆，而非一次性注入。

**现状**:
- 当前: 识别到 person_id → `_memory_mgr.get_prompt(pid)` 一次性生成完整 prompt 注入 Qwen session
- 代码位置: `d01_realtime_chat.py:3729`

**待设计**:
- [ ] 渐进式加载策略（先加载名字 → 再加载关键记忆 → 再加载细节）
- [ ] 记忆权重/优先级排序（最近交互的记忆优先）
- [ ] 与 Qwen Realtime session 的动态 prompt 更新机制
- [ ] 多人说话场景的记忆切换

**前置依赖**: #7 身份识别优化（需要准确判断说话人）

---

## 9. 自然交互对话 / Omni HTML 问题 ✅

**问题**: Qwen Omni 概率返回 HTML/XML 标签导致 TTS 播报；prompt 抑制后回复变成 `"xxx<nod>"`，动作也没做出来。

**已修复**:
- `INSTRUCTIONS` 输出格式铁律大幅强化：4 条具体规则 + 正反示范（voice/config.py）
- Transcript 兜底清洗器：`_ACTION_TAG_RE` 正则捕获泄漏标签 → 触发物理动作 + 清洗日志（voice/d01_realtime_chat.py）
- `remember_fact(key="name")` 同步已修
- [ ] 动作应通过 function call 触发，需要：
  - 检查 function call 中是否有动作触发的 tool 定义
  - 如果没有，需要注册 `perform_action(action_name)` 之类的工具
  - 在 prompt 中明确引导: "要做动作请调用 xxx 工具，不要写在文字里"
- [ ] 测试不同 prompt 策略对 Qwen Omni 输出格式的影响

---

## 其他遗留

### perception/__init__.py 抽象层未创建

计划中有但未实现: `get_vision_worker()` 工厂函数（MediaPipe 优先、OpenCV 兜底）。
当前 `d01_realtime_chat.py` 仍然内联 try/except import。

- [ ] 创建 `perception/__init__.py`
- [ ] 改 `d01_realtime_chat.py` 的 import 为 `from perception import get_vision_worker`

---

## 优先级建议

| 优先级 | 项目 | 原因 |
|--------|------|------|
| P0 | #9 对话质量 | 最基础的交互体验 |
| P0 | #1 DOA 优化 | 唤醒后第一印象 |
| P1 | #7 身份识别优化 | #8 的前置依赖 |
| P1 | #8 渐进式记忆 | 个性化体验核心 |
| P2 | #3 空闲表情 | 体验丰富度 |
| P2 | #5 寻物 Observer | 新能力 |
| P3 | #6 RAG | 知识增强 |


---

## 10. mediapipe 环境修复 ❌

**问题**：当前 `.venv` 中未安装 mediapipe，导致视觉模型测试模块和 vision_worker 无法运行。

**现状**：
- `uv pip install mediapipe` 失败：jaxlib (95MB) 从 PyPI 下载超时
- `uv pip install mediapipe --no-deps` 可以装上 mediapipe 本体，但需要手动补运行时依赖
- mediapipe 0.10.21 的 jaxlib 依赖实际上只有 model-maker 才用到，推理不需要

**修复方案**：
```bash
# 方案 A：--no-deps 装 mediapipe + 手动补推理依赖（推荐）
uv pip install mediapipe --no-deps
uv pip install "opencv-contrib-python>=4.0" "numpy" "flatbuffers>=2.0" \
  "attrs>=21.3.0" "protobuf>=3.11,<5" "absl-py" "Pillow"

# 方案 B：用清华镜像加速（如果 jaxlib 在镜像有）
uv pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方案 C：指定不依赖 jax 的旧版 mediapipe
uv pip install "mediapipe==0.10.14" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验证**：
```bash
python -c "import mediapipe; print('mediapipe', mediapipe.__version__)"
python tests/vision_model_test.py --local-camera --duration 10
```

---

## 11. 视觉模型精度测试模块 ✅

**已完成**：`tests/vision_model_test.py` 已创建

**功能**：
- 三模型测试：人脸(FaceLandmarker) + 手部(HandLandmarker 双参数) + 手势(_classify_gesture)
- 双模式：实时摄像头（Reachy Mini / 本地回退） + 静态图片
- 可视化标注：蓝色人脸框、绿/橙手部框(门控)、红色虚框(仅诊断检出)、手势大字、食指指向线、v-threshold 线
- 终端统计汇总：检出率、门控通过率、手势分布、食指伸出率
- 标注图保存到 `tests/output/`（已 gitignore）

**运行**（需先完成 #10 mediapipe 安装）：
```bash
python tests/vision_model_test.py --local-camera          # 本地摄像头
python tests/vision_model_test.py image1.jpg image2.jpg   # 静态图片
python tests/vision_model_test.py --skip-face             # 只测手部+手势
```

**待更新**：GestureRecognizer 集成后（#12）需同步更新此模块，增加模型手势 vs 规则手势对比。

---

## 12. 手势识别优化：GestureRecognizer + 规则混合方案 ❌

**背景**：当前 `_classify_gesture()` 是纯规则判定（landmark y 坐标比较），one/fist、three 经常误检。原因是 margin=0.02 太小，拇指伸出判定也不够鲁棒。

**方案**：用 MediaPipe GestureRecognizer 替换 HandLandmarker + 规则 fallback

### 改动内容

1. **下载模型**：`gesture_recognizer.task`（~5MB）放到  `models/`）
   - 下载地址：`https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task`

2. **修改 `vision_worker.py`**：
   - `HandLandmarker` → `GestureRecognizer`（后者内部已含 landmark 检测）
   - `GestureRecognizer` 返回：landmarks + handedness + gesture category
   - 输出 dict 新增 `gesture_model` 字段（模型识别的手势）

3. **手势判定优先级**：
   - GestureRecognizer score > 阈值 → 直接用模型结果
   - score 低 或 `Unknown` → fallback 到规则 `_classify_gesture()`
   - 映射表：
     - `Closed_Fist` → fist
     - `Open_Palm` → five
     - `Pointing_Up` → point
     - `Victory` → two
     - `Thumb_Up` / `Thumb_Down` → 保留原标签
     - `ILoveYou` → 保留原标签
   - 规则补充（模型不覆盖的）：
     - three / four → landmark 数手指数
     - ok → thumb-tip 与 index-tip 距离判定

4. **修改 `d01_realtime_chat.py`**：
   - 模型路径新增 `GESTURE_MODEL_PATH`
   - 传递给 `vision_worker()`
   - dashboard overlay 显示模型/规则来源标签

5. **更新测试模块 `tests/vision_model_test.py`**：
   - 新增 GestureRecognizer 测试
   - 对比模型手势 vs 规则手势的一致性

### 参考

- [MediaPipe GestureRecognizer Task Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)
- 默认支持 7 种手势：Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou
- 支持 Model Maker 自定义训练扩展手势

# 13. 当我说”这个人叫xxx”同时有指向手势的时候，记忆也应该更新这个人的信息

## 14. 记忆注入过时 — 切人后旧记忆污染上下文 ✅

**已修复**: 用 `update_session(instructions=...)` 替代 `create_item(system message)`

**改动**:
- `voice/d01_realtime_chat.py`: 新增 `_update_memory_instructions()` helper,调用 `conv.update_session` 将记忆嵌入 session instructions
- 两处记忆注入(greet-time + late injection)从 `conv.create_item(system)` 改为 `_update_memory_instructions()`
- 切人时 session instructions 整体替换,旧记忆自动消失(不再只增不删)
- `voice/state.py`: 新增 `identity_injected_pid` 追踪当前已注入哪个人的记忆
- `session.updated` 回调区分初始配置 vs 记忆更新(不再重复打印冗长日志)

## 15. 记忆权限 + 认主机制 ✅

**问题**：merge 误合并可能泄漏记忆；任何人都能删除他人记忆。

**已完成**：
- `identity/owner.py` — OwnerManager, `data/owner.json` 持久化
- 认主方式: 第一个被 `remember_fact(name=xxx)` 的人自动成为 owner
- `MemoryManager` 增加 `actor_pid` 权限校验：非 owner 只能删除自己的记忆
- `auto_merge` 增加双命名保护：两边都有 name 时跳过合并
- d01 集成: 初始化 OwnerManager, remember_fact 时 try_claim

**权限矩阵**：
| 操作 | Owner | 其他人 |
|------|-------|--------|
| remember_fact(自己) | ✅ | ✅ |
| forget_fact(自己) | ✅ | ✅ |
| clear_memory(自己) | ✅ | ✅ |
| forget_fact(他人) | ✅ | ❌ |
| clear_memory(他人) | ✅ | ❌ |