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
5. **多人同框介绍**: 用户指着他人说"这是我朋友XX" → robot 给对应人脸关联名字/关系(方案调研中, 见 docs/MULTI_PERSON_INTRO_PLAN.md)
6. **end_session 乱码**: 模型偶尔把 function_call_output 中的元指令当文字朗读(已简化 output 内容)
7. **记忆注入过时**: ~~切人后旧记忆 system message 仍在 context 中污染回答~~ ✅ 已修复: 改用 update_session instructions 替换,切人时旧记忆自动消失

## 下一步建议

1. **写测试(P0)**: 按 FEATURE_INVENTORY.md 优先级, 先写跟踪核心(T4) + 状态机(T15) + 打断(T1) 的单元/集成测试
2. **重构拆分**: 在测试覆盖后, 从 d01 拆出 behavior.py / head_control.py / vision_bridge.py / wake.py
3. 真机测试验证检出率
4. 继续 todo.md 中未完成项(#1 DOA / #7 身份优化 / #12 手势识别)
