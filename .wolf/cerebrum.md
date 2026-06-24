# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-06-23

## User Preferences

<!-- How the user likes things done. Code style, tools, patterns, communication. -->

## Key Learnings

- **Project:** shadow-yunwu-claudecode-aibasicalgodevdept-20260529
- **Description:** 把一台 [Reachy Mini Lite](https://www.pollen-robotics.com/reachy-mini/)(USB 版)变成一个

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

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
