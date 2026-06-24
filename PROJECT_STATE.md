# PROJECT_STATE

## 已完成事项

### 重构
- **6 模块拆分完成**: d01_realtime_chat.py 拆出 config.py / state.py / actions.py / audio.py / debug_server.py
- ARCHITECTURE.md 已更新到 6 模块结构

### 依赖 / 环境
- pyproject.toml 移入 reachy-mini-demo/ 项目目录
- Intel Mac (x86_64) 兼容: mediapipe<0.10.15 + onnxruntime<1.20 + jaxlib override
- venv 重建完成 (目录从 Documents/code → code/ 迁移后)
- 始终使用清华镜像 (UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/)

### Bug 修复 (本轮)
- `_current_turn` NameError: close_session 改用 `_st_mod._current_turn`
- daemon 未清理: start_mac.sh 加 EXIT trap
- debug_server SNAP_DIR 缺失 import
- motion_loop crash: tag leak handler 从字符串改 `{"name": act}` dict
- 记忆注入时序: 加延迟注入循环 (greet 后再检查 identity)
- SEEK 失败: 加 `vis_ready` 门控 (vision_worker ready 信号)
- "看看那边"不调 VLM: 更新 look_* tool description 明确是肢体动作
- **"看看那边" + 指向**: 更新 INSTRUCTIONS + identify_pointed_object 描述，引导模型在用户指方向时调用 identify_pointed_object 而非 look_*
- **呼吸感过强**: 全态呼吸回退为仅 ARMED 有呼吸，其他状态无呼吸分量
- **标签泄漏兜底不生效**: 扩展 _TAG_TO_ACTION 覆盖 smile/wave/nodding，正则新增括号形式 `(点头)` 和 markdown 斜体 `*nod*`，修复 strip 提取逻辑
- **人脸锁定太严格**: 从"连续命中 LOCK_ON_S"改为滑动窗口命中率(12帧窗口,40%命中锁定,15%丢锁)，容忍 mediapipe 间歇漏检
- **人脸后端切换 YuNet**: vision_worker 默认用 YuNet (100%检出率 vs MediaPipe 58%)，`FACE_BACKEND=mediapipe` 或 `start_mac.sh --face-mp` 回退 MediaPipe
- **Face DB 碎片化修复**: match() 增加质心匹配, update_embedding() 放宽多样性, 新增 auto_merge() 启动自动合并重复人脸
- **手势识别升级**: GestureRecognizer 替换纯规则 _classify_gesture, 模型优先 + 规则 fallback(three/four/ok)
- **记忆权限 + 认主机制**: OwnerManager 首次交互自动绑定 owner; auto_merge 双命名保护; 非 owner 只能删除自己的记忆
- **记忆注入过时修复**: 用 `update_session(instructions=...)` 替代 `create_item(system message)`,切人时 session instructions 整体替换,旧记忆自动消失
- **分人对话摘要 + 音频闸门**: conversation_log 改为 per-pid dict; close_session 时异步做对话摘要存入 MemoryManager; get_prompt 注入上次对话摘要; 切人+DOA大幅偏移时关闭音频闸门等身份确认; 上下文过长自动触发中途摘要(CONV_SUMMARY_THRESHOLD); Dashboard 显示当前注入的记忆内容
- **多人脸 DOA 说话人选择**: vision_worker 输出 all_faces(所有检测到的脸); d01 中 _select_face_by_doa 根据 DOA 声源方向选出说话人的脸做身份识别; Dashboard 多人脸框渲染(选中=蓝, 非选中=灰)
- **唤醒优先级修复**: 替换 `_is_A` DOA 门控为 `a_active`(A 说话/robot 回应时屏蔽 B;A 沉默时响应 B)
- **TRACKING 身体跟随**: 视觉积分中头偏到颈限 70% 时自动转体,把人脸保持在中心
- **人脸误识别稳定性**: 身份切换增加迟滞(sim>=0.65 立即切;sim<0.65 需连续 2 次确认)
- **安全删除工作流**: clear_memory 改为多步安全流程: 意图分类→高阈值身份验证(sim>=0.80, 6s)→权限校验→二次口头确认→备份→删除; clear_lock 阻止确认期间他人唤醒; data/backups/ 自动备份支持回滚

## 当前架构状态

```
voice/
  config.py        — 常量 + 工具元数据 + prompt
  state.py         — State 类 + log + 对话事件录制 + OneEuroFilter
  actions.py       — act_* 动作函数 + ACTIONS 字典
  audio.py         — doa_sensor_loop + player_loop
  debug_server.py  — vis_debug_server (Dashboard)
  d01_realtime_chat.py — 主程序 (~2100 行)
perception/
  vision_worker.py — 人脸+手部子进程 (YuNet默认/MediaPipe可选, GestureRecognizer手势)
identity/
  recognizer.py    — YuNet + ArcFace 身份识别 + 自动合并碎片
  owner.py         — 主人认定 + 记忆删除权限校验
```

- 9 状态 FSM: ARMED/IDLE_CENTER/ENGAGING/TRACKING/SEARCHING/RETURNING/POINTING/PLAYING
- 5 层运动仲裁: Primary > Playing > SoundTurn > Tracking > Idle
- 两阶段指向: judge round → POINTING state → point round

### 重构准备 (2026-06-23)
- **特性清单完成**: `docs/FEATURE_INVENTORY.md` — 19 大类、80+ 子特性、代码位置、测试方案
- 测试方案覆盖 5 层(单元/集成/视觉离线/硬件在环/端到端)
- 建议目录结构 + conftest fixtures + 优先级排序

## 遗留问题

1. **d01 仍是 god file**: ~2100 行, 包含 ChatCallback + 6 个循环 + KwsGate + main, 需拆分
2. **YuNet 无 blendshapes**: 切换后 smile/frown 恒为 0.0, 表情回应(M3-b)失效; 可后续用 insightface 2D106 landmark 估算
3. **手势识别**: ~~纯规则 _classify_gesture 对 fist/three 误检率高, 待 GestureRecognizer 替换~~ ✅ 已用 GestureRecognizer 替换(模型优先 + 规则 fallback)
4. **Face DB 碎片化**: ~~同一人因角度/光照变化被注册为多个 ID~~ ✅ 已修复: match() 质心匹配 + update_embedding() 放宽 + auto_merge() 启动合并
   - ⚠️ **auto_merge 未同步 MemoryManager**: 启动时 FaceDB 合并碎片人脸后未调用 `merge_memories()` → 被合并的 drop_pid 的 `data/memories/<drop_pid>.json` 残留, keep_pid 缺少 drop_pid 的 facts
5. **多人同框介绍**: 用户指着他人说"这是我朋友XX" → robot 给对应人脸关联名字/关系(方案调研中, 见 docs/MULTI_PERSON_INTRO_PLAN.md)
6. **end_session 乱码**: 模型偶尔把 function_call_output 中的元指令当文字朗读(已简化 output 内容)
7. **记忆注入过时**: ~~切人后旧记忆 system message 仍在 context 中污染回答~~ ✅ 已修复: `_update_memory_instructions()` 用 `update_session(instructions=...)` 整体替换, 切人/更新 fact 时旧记忆自动消失

## 下一步建议

1. **重构拆分**: 在测试覆盖后, 从 d01 拆出 behavior.py / head_control.py / vision_bridge.py / wake.py
2. **记忆格式重构**: 当前 `facts` 是 `{key: value}` 平铺字典, key 命名死板(name/job/hobby 等硬编码风格); 需改为自然语言分类存储, 如: 喜好类("喜欢猫""爱吃火锅")、客观事实类("今天下雨""现在是晚上")、个人信息类("叫小明""是程序员")等, 让模型能更自然地存取和表达记忆
3. 真机测试验证检出率
4. 继续 todo.md 中未完成项(#1 DOA / #7 身份优化 / #12 手势识别)
