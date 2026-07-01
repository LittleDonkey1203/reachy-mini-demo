# anatomy.md

> Auto-maintained by OpenWolf. Last scanned: 2026-06-27T03:28:12.795Z
> Files: 58 tracked | Anatomy hits: 0 | Misses: 0

## ../../../../.claude/plans/

- `rustling-booping-biscuit.md` — Facts 格式重构：list[str] → dict[str,str] + summary (~768 tok)

## ../../../../Library/Application Support/Code/User/

- `settings.json` (~327 tok)

## ./

- `_probe.py` — SDK wake_up 测试脚本 (~302 tok)
- `.gitignore` — Git ignore rules (~147 tok)
- `CALIBRATION.md` — Reachy Mini Lite — 标定与硬件 I/O 特性记录 (~8060 tok)
- `CLAUDE.md` — OpenWolf (~180 tok)
- `connect.py` — 连接 Reachy Mini daemon 的统一封装 (~269 tok)
- `MACOS_SETUP.md` — macOS (Intel) 部署指南 (~1670 tok)
- `PROJECT_STATE.md` — PROJECT_STATE (~814 tok)
- `pyproject.toml` — Python project configuration (~234 tok)
- `README.md` — Project documentation (~1024 tok)
- `start_daemon.sh` — daemon 启动脚本 (~634 tok)
- `start_mac.sh` — macOS 启动脚本(含 --face-mp 选项) (~1618 tok)
- `test_realtime_model.py` — 最小化 Realtime 模型连通性测试 (~1202 tok)
- `todo.md` — Reachy Mini Demo — TODO 清单 (~699 tok)

## .claude/

- `settings.json` (~441 tok)

## .claude/agents/

- `architect.md` — 架构师 Agent (~261 tok)
- `executor.md` — 执行者 Agent (~370 tok)
- `lead.md` — 技术负责人 Agent (~426 tok)

## .claude/rules/

- `openwolf.md` (~313 tok)

## _archive/vision/

- `_diag_face.py` — VIS-01 诊断:MediaPipe 零检出 (~907 tok)
- `_diag_face2.py` — VIS-01 诊断2:标准人脸图检出验证 (~720 tok)
- `_play01_ghost_diag.py` — PLAY-01 假手诊断 (~908 tok)

## _archive/voice/

- `_d01a_orchestrated.py` — D-01a 编排版:固定时序语音闭环 (~2664 tok)
- `_d01b_orchestrated.py` — D-01b 编排版:barge-in 打断 (~3759 tok)
- `_diag_o01.py` — O-01 诊断:上行 75s 无事件 (~1447 tok)
- `_hand_model_diag.py` — 手部模型精度诊断 (~2629 tok)
- `_judge_unit.py` — parse_judge 离线单测 (~256 tok)
- `_motion_amp_test.py` — O-01a 修复:加大动作幅度安全验证 (~1104 tok)
- `_o01a1_orchestrated.py` — O-01a-1 编排版:语音+动作工具 (~5791 tok)
- `_o01a2_orchestrated.py` — O-01a-2 编排版:说话时 idle 微动 (~6116 tok)
- `_o01a3_orchestrated.py` — O-01a 修复2:说话和动作同时出发 (~5744 tok)
- `_point_dir_test.py` — POINT-02-a:手指方向检测 (~1479 tok)
- `_track_verify.py` — TRACK-FIX 跟踪质量验证 (~2634 tok)
- `_v01_orchestrated.py` — V-01-1 编排版:take_snapshot 单帧看图 (~7027 tok)
- `_vis_proc_diag.py` — TRACK-FIX 零检出诊断 (~899 tok)
- `vision_worker_cv.py` — OpenCV 后备视觉子进程 (~1865 tok)

## _experiments/

- `play01_hand_track.py` — PLAY-01-a:手部互动跟手走 (~3555 tok)
- `vis01_face_track.py` — VIS-01:本地视觉看脸+转头 (~2798 tok)

## data/memories/

- `person_f4e8c43f.json` (~20 tok)

## docs/

- `COGNITIVE_MEMORY_ARCHITECTURE.md` — 小艺认知记忆架构重构 (~2175 tok)
- `FEATURE_INVENTORY.md` — 小艺(Reachy Mini Lite) — 特性清单 & 测试方案 (~7692 tok)
- `MULTI_PERSON_INTRO_PLAN.md` — 多人同框介绍朋友方案 (~2783 tok)
- `QWEN_OMNI_TOOL_CALLING.md` — Qwen-Omni-Realtime 工具调用调研报告 (~1151 tok)
- `WAKEWORD_PRIORITY_ANALYSIS.md` — 唤醒词 × 人脸锁定 × DOA 优先级分析 (~2615 tok)

## identity/

- `owner.py` — 主人认定模块(首次交互自动绑定+转让) (~789 tok)
- `recognizer.py` — 身份识别模块 — YuNet 人脸检测 + arcface 特征提取 + 特征库匹配。 (~6249 tok)

## memory/

- `manager.py` — 认知记忆管理 — Entity Memory + Episodic Memory + Working Memory 注入。 (~6068 tok)
- `safety.py` — 记忆安全删除工作流(身份验证+二次确认) (~1023 tok)

## perception/

- `fusion.py` — 声源-视觉感知融合 (~289 tok)
- `vision_worker.py` — 视觉子进程: Face(YuNet/MediaPipe)+Hand(GestureRecognizer) (~5600 tok)

## tests/

- `face_backend_compare.py` — MediaPipe vs YuNet 精度对比 (~1930 tok)

## voice/

- `config.py` — 配置常量、工具元数据、prompt 模板。 (~2775 tok)
- `d01_realtime_chat.py` — Reachy Mini × Qwen3.5-Omni-Realtime 语音对话(D-01+O-01a+V-01+F-01+FUSION-03+PLAY-01:完整体)。 (~25448 tok)
- `debug_server.py` — VIS_DEBUG MJPEG HTTP 调试预览服务 + Conversation Dashboard。 (~17876 tok)
- `kws.py` — WAKE-01 唤醒词门控(sherpa-onnx) (~991 tok)
- `realtime.py` — Qwen-Omni-Realtime 对话协议层 — 回调 + 会话生命周期管理。 (~9050 tok)
- `state.py` — 共享状态容器、日志、对话事件录制、One Euro 滤波器。 (~3110 tok)
