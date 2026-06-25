# Memory

> Chronological action log. Hooks and AI append to this file automatically.
> Old sessions are consolidated by the daemon weekly.

## Session: 2026-06-23 16:02

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 16:03 | Edited start_daemon.sh | inline fix | ~7 |
| 16:03 | Session end: 1 writes across 1 files (start_daemon.sh) | 1 reads | ~692 tok |
| 16:16 | Created tests/face_backend_compare.py | — | ~1929 |
| 16:17 | Edited tests/face_backend_compare.py | 2→2 lines | ~35 |
| 16:18 | Session end: 3 writes across 2 files (start_daemon.sh, face_backend_compare.py) | 1 reads | ~2656 tok |
| 16:20 | Session end: 3 writes across 2 files (start_daemon.sh, face_backend_compare.py) | 1 reads | ~2656 tok |
| 16:20 | Session end: 3 writes across 2 files (start_daemon.sh, face_backend_compare.py) | 1 reads | ~2656 tok |
| 16:20 | Session end: 3 writes across 2 files (start_daemon.sh, face_backend_compare.py) | 1 reads | ~2656 tok |
| 16:20 | Session end: 3 writes across 2 files (start_daemon.sh, face_backend_compare.py) | 1 reads | ~2656 tok |

## Session: 2026-06-23 16:23

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 16:24 | Edited perception/vision_worker.py | modified pick_main_face() | ~268 |
| 16:24 | Edited perception/vision_worker.py | modified __init__() | ~609 |
| 16:27 | Edited perception/vision_worker.py | modified vision_worker() | ~2372 |

## Session: 2026-06-23 16:29

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 16:29 | Edited start_mac.sh | expanded (+12 lines) | ~106 |
| 16:30 | Edited perception/vision_worker.py | added 1 import(s) | ~35 |
| 16:31 | Edited voice/config.py | 2→5 lines | ~88 |
| 14:30 | 重写 vision_worker() 支持双后端(YuNet默认/MediaPipe --face-mp) | perception/vision_worker.py, voice/config.py, start_mac.sh | 编译通过,模型路径正确 | ~3000 |
| 16:32 | Edited PROJECT_STATE.md | 1→2 lines | ~58 |
| 16:32 | Edited PROJECT_STATE.md | 2→2 lines | ~17 |
| 16:33 | Edited PROJECT_STATE.md | 3→3 lines | ~72 |
| 16:33 | Session end: 6 writes across 4 files (start_mac.sh, vision_worker.py, config.py, PROJECT_STATE.md) | 3 reads | ~2513 tok |
| 16:40 | Edited perception/vision_worker.py | modified _pick() | ~61 |
| 16:40 | Edited perception/vision_worker.py | 3→3 lines | ~52 |
| 16:40 | Edited voice/debug_server.py | modified default() | ~108 |
| 16:40 | Edited voice/debug_server.py | inline fix | ~26 |
| 16:42 | Edited voice/d01_realtime_chat.py | 1→2 lines | ~42 |
| 16:45 | 修复 debug_server float32 JSON 序列化崩溃 + _NumpyEncoder 兜底 | perception/vision_worker.py, voice/debug_server.py, voice/d01_realtime_chat.py | select_yunet 返回 Python float,debug_server 加 NumpyEncoder | ~1500 |
| 16:45 | Session end: 11 writes across 6 files (start_mac.sh, vision_worker.py, config.py, PROJECT_STATE.md, debug_server.py) | 5 reads | ~30492 tok |
| 16:54 | Edited memory/manager.py | modified forget_fact() | ~267 |
| 16:55 | Edited memory/manager.py | 4→9 lines | ~109 |
| 16:55 | Edited memory/manager.py | expanded (+15 lines) | ~276 |
| 16:55 | Edited voice/d01_realtime_chat.py | inline fix | ~23 |
| 16:55 | Edited voice/d01_realtime_chat.py | inline fix | ~34 |
| 16:55 | 新增 forget_fact 工具(逐条删记忆,模糊匹配key) | memory/manager.py, voice/d01_realtime_chat.py | 编译通过,end-to-end 验证 OK | ~800 |
| 16:56 | Session end: 16 writes across 7 files (start_mac.sh, vision_worker.py, config.py, PROJECT_STATE.md, debug_server.py) | 6 reads | ~36778 tok |
| 16:57 | Edited memory/manager.py | 14→15 lines | ~168 |
| 16:57 | Session end: 17 writes across 7 files (start_mac.sh, vision_worker.py, config.py, PROJECT_STATE.md, debug_server.py) | 6 reads | ~37128 tok |
| 17:04 | Edited voice/d01_realtime_chat.py | 8→7 lines | ~100 |
| 17:06 | Edited voice/d01_realtime_chat.py | modified lower() | ~317 |
| 17:06 | Session end: 19 writes across 7 files (start_mac.sh, vision_worker.py, config.py, PROJECT_STATE.md, debug_server.py) | 6 reads | ~37533 tok |
| 17:10 | Session end: 19 writes across 7 files (start_mac.sh, vision_worker.py, config.py, PROJECT_STATE.md, debug_server.py) | 6 reads | ~37533 tok |
| 17:15 | Created docs/MULTI_PERSON_INTRO_PLAN.md | — | ~2968 |
| 17:19 | Edited voice/d01_realtime_chat.py | 4→8 lines | ~122 |
| 17:15 | 修复名字回退:vision_result_loop 优先从 MemoryManager 取名字; 发现 face DB 碎片化(同一人3个ID) | voice/d01_realtime_chat.py | 编译通过,根因需 face DB merge 机制 | ~800 |
| 17:21 | Edited PROJECT_STATE.md | 3→6 lines | ~139 |
| 17:21 | Session end: 22 writes across 8 files (start_mac.sh, vision_worker.py, config.py, PROJECT_STATE.md, debug_server.py) | 8 reads | ~45794 tok |

## Session: 2026-06-23 17:23

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|

## Session: 2026-06-24 09:00

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 09:05 | Face DB 碎片化修复: match() 质心匹配 + update_embedding() 放宽 + auto_merge() | identity/recognizer.py | 5→4人, 陛下+未命名合并成功 | ~2000 |
| 09:05 | 新增 merge_memories() | memory/manager.py | 合并时迁移记忆 facts | ~300 |
| 09:06 | _face_key 网格从 40px 放宽到 80px | identity/recognizer.py | 减少网格边界碎片化 | ~50 |
| 09:06 | 下载 gesture_recognizer.task | models/ | 8.3MB float16 模型 | ~0 |
| 09:07 | GestureRecognizer 替换 HandLandmarker | perception/vision_worker.py | 模型优先(score≥0.6)+规则fallback | ~1500 |
| 09:07 | 新增 GESTURE_MODEL_PATH | voice/config.py | 手势模型路径常量 | ~50 |
| 09:07 | 传递 gesture_model 给 vision_worker | voice/d01_realtime_chat.py | kwargs 传参 | ~50 |
| 09:08 | 记忆注入过时方案追加到 todo.md #14 | todo.md | update_session 方案设计 | ~500 |
| 09:08 | 更新 PROJECT_STATE.md | PROJECT_STATE.md | 完成 #3/#4, 新增 #7 | ~200 |
| 18:01 | Created ../../../../.claude/plans/generic-wiggling-sketch.md | — | ~2825 |
| 09:06 | Edited identity/recognizer.py | modified match() | ~320 |
| 09:06 | Edited identity/recognizer.py | modified update_embedding() | ~189 |
| 09:07 | Edited identity/recognizer.py | modified reset() | ~866 |
| 09:07 | Edited identity/recognizer.py | modified __init__() | ~127 |
| 09:07 | Edited memory/manager.py | modified forget_fact() | ~361 |
| 09:09 | Edited todo.md | expanded (+30 lines) | ~323 |
| 09:10 | Edited voice/config.py | 1→2 lines | ~41 |
| 09:11 | Edited perception/vision_worker.py | modified vision_worker() | ~222 |
| 09:11 | Edited perception/vision_worker.py | modified exists() | ~470 |
| 09:12 | Edited perception/vision_worker.py | modified and() | ~852 |
| 09:12 | Edited voice/d01_realtime_chat.py | inline fix | ~23 |
| 09:12 | Edited voice/d01_realtime_chat.py | 5→6 lines | ~86 |
| 09:13 | Edited PROJECT_STATE.md | 1→3 lines | ~85 |
| 09:13 | Edited PROJECT_STATE.md | 3→3 lines | ~36 |
| 09:14 | Edited PROJECT_STATE.md | 2→2 lines | ~63 |
| 09:14 | Edited PROJECT_STATE.md | 1→2 lines | ~48 |
| 09:21 | Session end: 17 writes across 8 files (generic-wiggling-sketch.md, recognizer.py, manager.py, todo.md, config.py) | 10 reads | ~44997 tok |
| 09:21 | Edited identity/recognizer.py | modified _face_key() | ~47 |
| 09:23 | Session end: 18 writes across 8 files (generic-wiggling-sketch.md, recognizer.py, manager.py, todo.md, config.py) | 10 reads | ~50799 tok |
| 09:26 | Session end: 18 writes across 8 files (generic-wiggling-sketch.md, recognizer.py, manager.py, todo.md, config.py) | 10 reads | ~50799 tok |

## Session: 2026-06-24 09:29

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 09:29 | Edited identity/recognizer.py | modified or() | ~174 |
| 09:30 | Created identity/owner.py | — | ~789 |
| 09:30 | Edited memory/manager.py | modified __init__() | ~103 |
| 09:30 | Edited memory/manager.py | modified clear_all() | ~107 |
| 09:31 | Edited memory/manager.py | modified forget_fact() | ~86 |
| 09:31 | Edited memory/manager.py | modified handle_tool_call() | ~251 |
| 09:32 | Edited voice/d01_realtime_chat.py | added 1 import(s) | ~47 |
| 09:32 | Edited voice/d01_realtime_chat.py | 2→3 lines | ~37 |
| 09:32 | Edited voice/d01_realtime_chat.py | 3→4 lines | ~51 |
| 09:33 | Edited voice/d01_realtime_chat.py | modified has_owner() | ~233 |
| 09:33 | Edited todo.md | expanded (+20 lines) | ~186 |
| 09:34 | Edited PROJECT_STATE.md | 1→2 lines | ~47 |
| 09:34 | Edited PROJECT_STATE.md | 2→3 lines | ~26 |
| 09:35 | Session end: 13 writes across 6 files (recognizer.py, owner.py, manager.py, d01_realtime_chat.py, todo.md) | 3 reads | ~35913 tok |
| 09:37 | Edited docs/FEATURE_INVENTORY.md | 12→17 lines | ~396 |
| 09:37 | Edited docs/FEATURE_INVENTORY.md | expanded (+15 lines) | ~494 |
| 09:38 | Edited docs/FEATURE_INVENTORY.md | 9→13 lines | ~284 |
| 09:38 | Edited docs/FEATURE_INVENTORY.md | expanded (+37 lines) | ~281 |
| 09:39 | Edited docs/FEATURE_INVENTORY.md | expanded (+29 lines) | ~241 |
| 09:39 | Edited docs/FEATURE_INVENTORY.md | expanded (+19 lines) | ~276 |
| 09:39 | Edited docs/FEATURE_INVENTORY.md | expanded (+53 lines) | ~361 |
| 09:40 | Edited docs/FEATURE_INVENTORY.md | 11→13 lines | ~144 |
| 09:40 | Edited docs/FEATURE_INVENTORY.md | 4→5 lines | ~52 |
| 09:40 | Edited docs/FEATURE_INVENTORY.md | 12→14 lines | ~118 |
| 09:40 | Edited docs/FEATURE_INVENTORY.md | inline fix | ~18 |
| 09:41 | Session end: 24 writes across 7 files (recognizer.py, owner.py, manager.py, d01_realtime_chat.py, todo.md) | 4 reads | ~38769 tok |
| 09:44 | Session end: 24 writes across 7 files (recognizer.py, owner.py, manager.py, d01_realtime_chat.py, todo.md) | 4 reads | ~38769 tok |

## Session: 2026-06-24 10:11

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 10:15 | Created .gitignore | — | ~149 |
| 10:16 | Session end: 1 writes across 1 files (.gitignore) | 3 reads | ~9320 tok |
| 10:20 | Edited voice/d01_realtime_chat.py | 5→6 lines | ~99 |
| 10:20 | Edited voice/d01_realtime_chat.py | modified lower() | ~65 |
| 10:20 | Edited identity/recognizer.py | modified set_name() | ~50 |
| 10:20 | Session end: 4 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 4 reads | ~37628 tok |
| 10:24 | Session end: 4 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 4 reads | ~37628 tok |
| 10:28 | Edited voice/d01_realtime_chat.py | modified lower() | ~112 |
| 10:28 | Session end: 5 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 5 reads | ~40510 tok |
| 10:32 | Edited voice/d01_realtime_chat.py | modified has_owner() | ~233 |
| 10:32 | Session end: 6 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 5 reads | ~40760 tok |
| 10:33 | Edited voice/d01_realtime_chat.py | modified lower() | ~137 |
| 10:33 | Session end: 7 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 5 reads | ~40944 tok |
| 10:37 | Session end: 7 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 5 reads | ~40969 tok |
| 10:39 | Session end: 7 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 5 reads | ~40969 tok |
| 10:40 | Session end: 7 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 5 reads | ~40969 tok |
| 10:40 | Edited .gitignore | 4→3 lines | ~8 |
| 10:41 | Session end: 8 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 5 reads | ~40978 tok |
| 10:43 | Session end: 8 writes across 3 files (.gitignore, d01_realtime_chat.py, recognizer.py) | 5 reads | ~40978 tok |
| 10:55 | Edited voice/config.py | "qwen3.5-omni-flash-realti" → "qwen3.5-omni-plus-realtim" | ~11 |
| 10:56 | Session end: 9 writes across 4 files (.gitignore, d01_realtime_chat.py, recognizer.py, config.py) | 5 reads | ~40989 tok |
| 10:56 | Edited voice/config.py | 3→4 lines | ~62 |
| 10:57 | Edited voice/config.py | 5→2 lines | ~4 |
| 10:57 | Session end: 11 writes across 4 files (.gitignore, d01_realtime_chat.py, recognizer.py, config.py) | 6 reads | ~41088 tok |
| 10:58 | Session end: 11 writes across 4 files (.gitignore, d01_realtime_chat.py, recognizer.py, config.py) | 7 reads | ~41088 tok |
| 11:01 | Session end: 11 writes across 4 files (.gitignore, d01_realtime_chat.py, recognizer.py, config.py) | 9 reads | ~43151 tok |

## Session: 2026-06-24 11:03

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 11:14 | Edited voice/state.py | 2→3 lines | ~36 |
| 11:14 | Created docs/WAKEWORD_PRIORITY_ANALYSIS.md | — | ~2789 |
| 11:14 | Edited voice/d01_realtime_chat.py | 4→7 lines | ~103 |
| 11:15 | Edited voice/d01_realtime_chat.py | 3→3 lines | ~38 |
| 11:15 | Edited voice/d01_realtime_chat.py | 4→4 lines | ~54 |
| 11:15 | Edited voice/d01_realtime_chat.py | modified _update_memory_instructions() | ~403 |
| 11:16 | Edited voice/d01_realtime_chat.py | reduced (-19 lines) | ~432 |
| 11:16 | Edited voice/d01_realtime_chat.py | 6→3 lines | ~70 |
| 11:17 | Edited voice/d01_realtime_chat.py | 3→4 lines | ~63 |
| 11:17 | Edited voice/d01_realtime_chat.py | 6→8 lines | ~128 |
| 11:17 | Edited voice/d01_realtime_chat.py | 3→3 lines | ~49 |
| 11:18 | Edited voice/d01_realtime_chat.py | 4→4 lines | ~66 |
| 11:18 | Edited voice/d01_realtime_chat.py | 2→3 lines | ~51 |
| 11:19 | Edited PROJECT_STATE.md | 1→2 lines | ~56 |
| 11:20 | Edited PROJECT_STATE.md | inline fix | ~30 |
| 11:20 | Edited todo.md | _update_instructions_with_memory() → _update_memory_instructions() | ~135 |
| 11:22 | Session end: 16 writes across 5 files (state.py, WAKEWORD_PRIORITY_ANALYSIS.md, d01_realtime_chat.py, PROJECT_STATE.md, todo.md) | 10 reads | ~47817 tok |
| 11:23 | Session end: 16 writes across 5 files (state.py, WAKEWORD_PRIORITY_ANALYSIS.md, d01_realtime_chat.py, PROJECT_STATE.md, todo.md) | 10 reads | ~47817 tok |
| 11:30 | Edited voice/d01_realtime_chat.py | 7→8 lines | ~127 |
| 11:31 | Session end: 17 writes across 5 files (state.py, WAKEWORD_PRIORITY_ANALYSIS.md, d01_realtime_chat.py, PROJECT_STATE.md, todo.md) | 10 reads | ~47944 tok |
| 11:35 | Edited todo.md | expanded (+17 lines) | ~185 |
| 11:35 | Session end: 18 writes across 5 files (state.py, WAKEWORD_PRIORITY_ANALYSIS.md, d01_realtime_chat.py, PROJECT_STATE.md, todo.md) | 11 reads | ~50055 tok |
| 11:41 | Session end: 18 writes across 5 files (state.py, WAKEWORD_PRIORITY_ANALYSIS.md, d01_realtime_chat.py, PROJECT_STATE.md, todo.md) | 12 reads | ~55833 tok |

## Session: 2026-06-24 11:45

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 11:50 | 唤醒优先级修复: _is_A→a_active, +user_speaking | voice/state.py, voice/d01_realtime_chat.py | A说话屏蔽B;A沉默响应B | ~500 |
| 11:51 | TRACKING 身体跟随: neck_off>70%时45°/s转体 | voice/config.py, voice/d01_realtime_chat.py | 人走侧面身体跟着转 | ~400 |
| 11:52 | 人脸误识别稳定性: sim<0.65需连续2次确认才切人 | voice/d01_realtime_chat.py | 防低sim抖动误切 | ~300 |
| 11:53 | 更新 todo.md #1/#16/#17/#18, PROJECT_STATE, cerebrum | todo.md, PROJECT_STATE.md, .wolf/cerebrum.md | 文档同步 | ~200 |
| 12:00 | clear_memory修复: 去掉confirmed守卫+不清face_db名字 | memory/manager.py, voice/d01_realtime_chat.py | 调用即生效;名字是身份不是记忆 | ~300 |
| 12:01 | debug_server身份标注: 人脸框下显示人名+MEM状态 | voice/debug_server.py | 视频流+state.json | ~200 |
| 12:02 | 恢复大大的face_db名字(被clear_memory误清) | data/face_db.json | person_5f4ab426→大大 | ~50 |

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 11:49 | Edited voice/state.py | 3→4 lines | ~40 |
| 11:49 | Edited voice/d01_realtime_chat.py | 12→14 lines | ~206 |
| 11:49 | Edited voice/d01_realtime_chat.py | 7→10 lines | ~190 |
| 11:50 | Edited voice/d01_realtime_chat.py | 7→7 lines | ~117 |
| 11:51 | Edited voice/config.py | 4→6 lines | ~53 |
| 11:51 | Edited voice/d01_realtime_chat.py | 1→2 lines | ~34 |
| 11:52 | Edited voice/d01_realtime_chat.py | expanded (+12 lines) | ~404 |
| 11:53 | Edited voice/d01_realtime_chat.py | 1→5 lines | ~76 |
| 11:53 | Edited voice/d01_realtime_chat.py | expanded (+22 lines) | ~583 |
| 11:54 | Edited voice/d01_realtime_chat.py | 3→4 lines | ~50 |
| 11:56 | Edited todo.md | expanded (+6 lines) | ~140 |
| 11:56 | Edited todo.md | NECK_REL_LIMIT() → BODY_LIMIT_DEG() | ~92 |
| 11:57 | Edited todo.md | expanded (+20 lines) | ~152 |
| 11:58 | Edited PROJECT_STATE.md | 1→4 lines | ~83 |
| 11:59 | Session end: 14 writes across 5 files (state.py, d01_realtime_chat.py, config.py, todo.md, PROJECT_STATE.md) | 7 reads | ~48176 tok |
| 12:06 | Edited voice/debug_server.py | 7→10 lines | ~132 |
| 12:06 | Edited voice/debug_server.py | expanded (+8 lines) | ~235 |
| 12:06 | Edited voice/debug_server.py | 3→5 lines | ~73 |
| 12:07 | Edited memory/manager.py | modified clear_all() | ~83 |
| 12:07 | Edited memory/manager.py | reduced (-6 lines) | ~92 |
| 12:07 | Edited memory/manager.py | 12→12 lines | ~95 |
| 12:09 | Edited voice/d01_realtime_chat.py | 7→4 lines | ~63 |
| 12:10 | Session end: 21 writes across 7 files (state.py, d01_realtime_chat.py, config.py, todo.md, PROJECT_STATE.md) | 10 reads | ~67673 tok |
| 14:14 | Edited memory/manager.py | expanded (+6 lines) | ~161 |
| 14:14 | Edited memory/manager.py | modified clear_all() | ~100 |
| 14:15 | Edited voice/d01_realtime_chat.py | expanded (+6 lines) | ~170 |
| 14:16 | Session end: 24 writes across 7 files (state.py, d01_realtime_chat.py, config.py, todo.md, PROJECT_STATE.md) | 10 reads | ~68270 tok |
| 14:16 | Edited voice/debug_server.py | 2→3 lines | ~54 |
| 14:16 | Edited voice/debug_server.py | 2→6 lines | ~79 |
| 14:17 | Edited voice/state.py | 3→4 lines | ~53 |
| 14:18 | Edited voice/d01_realtime_chat.py | 7→8 lines | ~140 |
| 14:18 | Edited PROJECT_STATE.md | 2→3 lines | ~106 |
| 14:18 | Session end: 29 writes across 7 files (state.py, d01_realtime_chat.py, config.py, todo.md, PROJECT_STATE.md) | 10 reads | ~68981 tok |
| 14:18 | Edited voice/d01_realtime_chat.py | 10→11 lines | ~188 |

## Session: 2026-06-24 14:20

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 14:21 | Edited voice/debug_server.py | 2→3 lines | ~49 |
| 14:21 | Edited PROJECT_STATE.md | 1→2 lines | ~74 |
| 14:21 | Edited PROJECT_STATE.md | 2→1 lines | ~32 |
| 14:22 | Session end: 3 writes across 2 files (debug_server.py, PROJECT_STATE.md) | 1 reads | ~15129 tok |

## Session: 2026-06-24 00:00 (continued from context-compacted session)

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 00:01 | Added current_is_owner reset in clear_memory | voice/d01_realtime_chat.py | clear_memory 现在重置 owner 状态 | ~20 |
| 00:01 | Added is_owner to state.json dict | voice/debug_server.py | Dashboard JS 可读取 s.is_owner | ~10 |
| 00:02 | Syntax check all 5 modified files | d01/debug_server/manager/state/config | All OK | ~5 |

## Session: 2026-06-24 14:23

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 14:29 | Edited identity/recognizer.py | modified get_name() | ~134 |
| 14:29 | Edited memory/manager.py | expanded (+6 lines) | ~231 |
| 14:29 | Edited memory/manager.py | expanded (+8 lines) | ~157 |
| 14:30 | Edited memory/manager.py | modified __init__() | ~102 |
| 14:31 | Edited voice/d01_realtime_chat.py | 1→2 lines | ~33 |
| 14:31 | Edited voice/d01_realtime_chat.py | 3→2 lines | ~39 |
| 14:31 | Edited voice/d01_realtime_chat.py | expanded (+6 lines) | ~312 |
| 14:32 | Edited memory/manager.py | modified handle_tool_call() | ~332 |
| 00:15 | Added find_by_name() to FaceDB | identity/recognizer.py | 按名字模糊查找 person_id | ~15 |
| 00:15 | clear_memory 支持 target_name | memory/manager.py | 主人可指定清除他人记忆 | ~40 |
| 00:16 | d01 clear_memory handler 区分 target | voice/d01_realtime_chat.py | 只在清自己时重置 state | ~25 |
| 00:16 | MemoryManager 接受 face_db 参数 | memory/manager.py + d01 | name→pid 解析 | ~10 |
| 14:33 | Session end: 8 writes across 3 files (recognizer.py, manager.py, d01_realtime_chat.py) | 3 reads | ~39844 tok |
| 14:46 | Created .claude/agents/architect.md | — | ~278 |
| 14:46 | Created .claude/agents/executor.md | — | ~274 |

| 14:50 | 创建 architect + executor 自定义 Agent 定义 | .claude/agents/architect.md, .claude/agents/executor.md | 完成 | ~500 |
| 14:47 | Session end: 10 writes across 5 files (recognizer.py, manager.py, d01_realtime_chat.py, architect.md, executor.md) | 7 reads | ~52239 tok |
| 14:47 | Edited .claude/agents/executor.md | expanded (+30 lines) | ~163 |
| 14:47 | Session end: 11 writes across 5 files (recognizer.py, manager.py, d01_realtime_chat.py, architect.md, executor.md) | 8 reads | ~52670 tok |
| 14:50 | Created ../../../../.claude/plans/generic-wiggling-sketch.md | — | ~1039 |
| 14:55 | Created ../../../../.claude/plans/generic-wiggling-sketch.md | — | ~1355 |
| 14:55 | Edited ../../../../.claude/plans/generic-wiggling-sketch.md | inline fix | ~20 |
| 14:55 | Edited ../../../../.claude/plans/generic-wiggling-sketch.md | 0.55 → 0.80 | ~14 |
| 14:58 | Edited ../../../../.claude/plans/generic-wiggling-sketch.md | inline fix | ~16 |
| 14:58 | Edited ../../../../.claude/plans/generic-wiggling-sketch.md | "FaceDB.verify_identity(em" → "FaceDB.verify_identity(em" | ~19 |
| 15:00 | Edited voice/state.py | 2→4 lines | ~49 |
| 15:01 | Created .claude/agents/lead.md | — | ~454 |
| 15:01 | Edited voice/config.py | 3→8 lines | ~76 |
| 15:01 | Session end: 20 writes across 9 files (recognizer.py, manager.py, d01_realtime_chat.py, architect.md, executor.md) | 10 reads | ~56294 tok |
| 15:02 | Edited identity/recognizer.py | modified clear_person() | ~402 |
| 15:03 | Edited memory/manager.py | expanded (+15 lines) | ~329 |
| 15:03 | Edited memory/manager.py | modified _persist() | ~257 |
| 15:04 | Edited memory/manager.py | modified handle_tool_call() | ~207 |

## Session: 2026-06-24 15:05

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 15:05 | Edited voice/d01_realtime_chat.py | 1→2 lines | ~31 |
| 15:05 | Edited voice/d01_realtime_chat.py | modified in() | ~1156 |
| 15:06 | Edited PROJECT_STATE.md | inline fix | ~43 |
| 15:06 | Session end: 3 writes across 2 files (d01_realtime_chat.py, PROJECT_STATE.md) | 1 reads | ~30673 tok |
| 15:06 | Edited voice/d01_realtime_chat.py | modified _handle_clear_memory_intent() | ~1000 |
| 15:08 | Edited voice/d01_realtime_chat.py | added 1 condition(s) | ~1197 |
| 15:08 | Edited voice/d01_realtime_chat.py | modified vision_result_loop() | ~34 |
| 15:09 | Edited voice/d01_realtime_chat.py | 2→3 lines | ~77 |
| 15:10 | Edited voice/d01_realtime_chat.py | modified dir() | ~81 |
| 15:10 | Edited voice/d01_realtime_chat.py | 3→4 lines | ~50 |
| 15:10 | Edited voice/d01_realtime_chat.py | 3→2 lines | ~30 |
| 15:10 | Edited voice/d01_realtime_chat.py | 3→2 lines | ~67 |
| 15:11 | Edited voice/d01_realtime_chat.py | 3→4 lines | ~72 |
| 15:11 | Edited voice/d01_realtime_chat.py | 4→7 lines | ~93 |
| 15:13 | Edited voice/debug_server.py | 2→3 lines | ~52 |
| 15:13 | Edited voice/debug_server.py | 4→5 lines | ~88 |
| 00:30 | Added clear_workflow + clear_lock to State | voice/state.py | 2 fields | ~5 |
| 00:30 | Added CLEAR_VERIFY_COUNT/SIM/TIMEOUT constants | voice/config.py | 3 constants | ~10 |
| 00:31 | Added backup_person + verify_identity to FaceDB | identity/recognizer.py | 2 methods | ~30 |
| 00:31 | Refactored QWEN_TOOLS: clear_memory → intent only + confirm_clear | memory/manager.py | tool defs + backup_person | ~50 |
| 00:35 | Rewrote clear_memory handler + added confirm_clear handler | voice/d01_realtime_chat.py | ~80 lines new | ~100 |
| 00:35 | Added verification logic in vision_result_loop | voice/d01_realtime_chat.py | ~50 lines | ~70 |
| 00:36 | Added clear_lock to wake-word gate + close_session reset | voice/d01_realtime_chat.py | 4 lines | ~10 |
| 00:36 | Added clear_phase to dashboard | voice/debug_server.py | state dict + JS | ~15 |
| 00:37 | All syntax checks pass | 6 files | OK | ~5 |
| 15:15 | Edited PROJECT_STATE.md | 1→2 lines | ~51 |
| 15:15 | Session end: 16 writes across 3 files (d01_realtime_chat.py, PROJECT_STATE.md, debug_server.py) | 3 reads | ~51728 tok |

## Session: 2026-06-24 15:15

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 15:17 | Edited voice/d01_realtime_chat.py | 2→1 lines | ~21 |
| 15:18 | Edited voice/d01_realtime_chat.py | 2→3 lines | ~27 |
| 15:18 | Session end: 2 writes across 1 files (d01_realtime_chat.py) | 1 reads | ~31721 tok |
| 15:22 | Session end: 2 writes across 1 files (d01_realtime_chat.py) | 1 reads | ~31721 tok |
| 15:23 | Session end: 2 writes across 1 files (d01_realtime_chat.py) | 1 reads | ~31721 tok |
| 15:28 | Edited voice/debug_server.py | 4→6 lines | ~227 |

## Session: 2026-06-24 15:30

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 15:31 | Edited voice/debug_server.py | 2→3 lines | ~32 |
| 15:31 | Edited voice/debug_server.py | added 2 condition(s) | ~229 |
| 15:31 | Edited voice/debug_server.py | modified if() | ~68 |
| 11:42 | Dashboard log smart-scroll: CSS + floating button + logAutoScroll + scroll listener | voice/debug_server.py | 默认自动滚到底，向上滑停止+显示↓按钮，点击恢复 | ~200 |
| 15:32 | Session end: 3 writes across 1 files (debug_server.py) | 1 reads | ~15593 tok |
| 15:35 | Session end: 3 writes across 1 files (debug_server.py) | 2 reads | ~47266 tok |
| 15:37 | Session end: 3 writes across 1 files (debug_server.py) | 2 reads | ~47266 tok |
| 15:41 | Created test_realtime_model.py | — | ~1446 |
| 15:41 | Edited test_realtime_model.py | 4→4 lines | ~30 |
| 15:42 | Edited test_realtime_model.py | 4→4 lines | ~36 |
| 15:42 | Edited test_realtime_model.py | inline fix | ~12 |
| 15:44 | Edited test_realtime_model.py | modified in() | ~57 |
| 15:50 | Created test_realtime_model.py | — | ~1202 |
| 15:57 | Session end: 9 writes across 2 files (debug_server.py, test_realtime_model.py) | 3 reads | ~50049 tok |
| 15:59 | Session end: 9 writes across 2 files (debug_server.py, test_realtime_model.py) | 3 reads | ~50049 tok |
| 16:11 | Edited voice/config.py | 2→3 lines | ~31 |
| 16:14 | Edited voice/d01_realtime_chat.py | 32→27 lines | ~487 |
| 16:14 | Session end: 11 writes across 4 files (debug_server.py, test_realtime_model.py, config.py, d01_realtime_chat.py) | 4 reads | ~53424 tok |
| 16:16 | Session end: 11 writes across 4 files (debug_server.py, test_realtime_model.py, config.py, d01_realtime_chat.py) | 4 reads | ~53424 tok |
| 16:23 | Edited identity/recognizer.py | modified __init__() | ~63 |
| 16:23 | Edited identity/recognizer.py | modified _save() | ~70 |
| 16:24 | Edited identity/recognizer.py | added 1 import(s) | ~18 |
| 16:33 | Created ../../../../.claude/plans/generic-wiggling-sketch.md | — | ~1431 |
| 16:35 | Edited ../../../../.claude/plans/generic-wiggling-sketch.md | modified vision_worker() | ~497 |
| 16:48 | Edited ../../../../.claude/plans/generic-wiggling-sketch.md | modified _save_conversation_summary() | ~740 |

## Session: 2026-06-24 16:50

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 16:51 | Edited ../../../../.claude/plans/generic-wiggling-sketch.md | 8→5 lines | ~34 |
| 17:13 | Edited ../../../../.claude/plans/generic-wiggling-sketch.md | modified close_session() | ~573 |
| 17:17 | Edited voice/state.py | inline fix | ~28 |
| 17:17 | Edited voice/state.py | 2→5 lines | ~65 |
| 17:17 | Edited voice/d01_realtime_chat.py | 3→4 lines | ~73 |
| 17:17 | Edited voice/d01_realtime_chat.py | 3→4 lines | ~72 |
| 17:18 | Edited voice/config.py | 1→2 lines | ~35 |
| 17:18 | Edited voice/config.py | 1→2 lines | ~33 |
| 17:18 | Edited voice/d01_realtime_chat.py | modified close_session() | ~358 |
| 17:19 | Edited voice/d01_realtime_chat.py | modified _update_memory_instructions() | ~535 |
| 17:19 | Edited voice/d01_realtime_chat.py | added 1 condition(s) | ~401 |
| 17:20 | Edited voice/d01_realtime_chat.py | 2→3 lines | ~34 |
| 17:20 | Edited memory/manager.py | added 1 import(s) | ~26 |
| 17:21 | Edited memory/manager.py | get_facts() → load_memory() | ~278 |
| 17:21 | Edited memory/manager.py | modified save_conversation_summary() | ~150 |
| 17:22 | Edited voice/d01_realtime_chat.py | modified _save_conversation_summary() | ~422 |
| 17:22 | Edited voice/d01_realtime_chat.py | modified items() | ~134 |
| 17:23 | Edited voice/d01_realtime_chat.py | modified close_session() | ~301 |
| 17:23 | Edited voice/d01_realtime_chat.py | modified log() | ~181 |

## Session: 2026-06-24 17:24

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 17:24 | Edited voice/d01_realtime_chat.py | expanded (+9 lines) | ~268 |
| 17:25 | Edited PROJECT_STATE.md | 6→7 lines | ~112 |
| 17:25 | Edited voice/state.py | 1→2 lines | ~28 |
| 17:25 | Session end: 3 writes across 3 files (d01_realtime_chat.py, PROJECT_STATE.md, state.py) | 3 reads | ~49433 tok |
| 17:26 | Edited voice/d01_realtime_chat.py | 4→5 lines | ~80 |
| 17:26 | Edited voice/debug_server.py | 2→4 lines | ~66 |
| 17:27 | Edited voice/debug_server.py | 2→3 lines | ~76 |
| 17:27 | Edited voice/debug_server.py | 1→2 lines | ~45 |
| 14:30 | 分人对话摘要+音频闸门+自动摘要 完整实现 | state.py, d01, manager.py, config.py, debug_server.py | 完成：per-pid conv_log、close_session异步摘要、音频闸门(仅切人+DOA偏移时)、CONV_SUMMARY_THRESHOLD自动触发、dashboard显示记忆注入 | ~8000 |
| 17:28 | Edited PROJECT_STATE.md | 1→2 lines | ~87 |
| 17:29 | Session end: 8 writes across 4 files (d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 3 reads | ~50063 tok |
| 17:31 | Edited PROJECT_STATE.md | 7→6 lines | ~89 |
| 17:31 | Session end: 9 writes across 4 files (d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 3 reads | ~50159 tok |
| 17:32 | Session end: 9 writes across 4 files (d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 3 reads | ~50176 tok |
| 17:43 | Edited voice/state.py | 1→2 lines | ~31 |
| 17:43 | Edited voice/d01_realtime_chat.py | 1→2 lines | ~35 |
| 17:43 | Edited voice/debug_server.py | 2→6 lines | ~110 |
| 17:43 | Edited voice/debug_server.py | 4→5 lines | ~66 |
| 17:44 | Edited voice/debug_server.py | expanded (+24 lines) | ~366 |
| 17:45 | Edited voice/debug_server.py | 5→4 lines | ~48 |
| 17:45 | Edited voice/debug_server.py | removed 28 lines | ~12 |
| 17:46 | Edited voice/debug_server.py | added 3 condition(s) | ~582 |
| 17:46 | Edited voice/debug_server.py | added 3 condition(s) | ~96 |
| 17:47 | Session end: 18 writes across 4 files (d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 3 reads | ~52158 tok |
| 17:48 | Session end: 18 writes across 4 files (d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 3 reads | ~52158 tok |
| 17:49 | Session end: 18 writes across 4 files (d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 4 reads | ~52158 tok |
| 17:51 | Session end: 18 writes across 4 files (d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 4 reads | ~52158 tok |
| 15:10 | Dashboard context debug: payload modal 增加 session instructions + memory prompt + conv_log 显示 | debug_server.py, state.py, d01 | 完成：点击事件弹窗可看到模型完整上下文 | ~3000 |
| 15:15 | 用户纠正：不要新增 tab，复用已有 payload modal | debug_server.py | 回退 Context tab，改为增强 openModal | ~500 |
| 15:20 | 用户纠正：闸门不能每次 close_session 触发，只在切人+DOA大幅偏移时 | d01 | 闸门移到二次唤醒处+DOA判断 | ~500 |
| 15:25 | 补全 wolf 文件更新(cerebrum/anatomy/memory) | .wolf/* | 用户指出未遵循CLAUDE.md | ~200 |

## Session: 2026-06-24 17:54

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 17:55 | 更新 FEATURE_INVENTORY.md: 新增 F13.13-F13.17(分人对话摘要)、F21(音频闸门)、F22(Dashboard上下文调试) | docs/FEATURE_INVENTORY.md | 完成 | ~300 |
| 18:00 | 多人脸 DOA 选人: vision_worker 输出 all_faces; d01 新增 _select_face_by_doa 函数,多人脸时用 DOA 选脸做身份识别 | perception/vision_worker.py, voice/d01_realtime_chat.py | 完成,语法检查通过 | ~1500 |
| 18:05 | debug_server 多人脸框渲染: 选中=蓝粗框+DOA标签, 非选中=灰细框+序号 | voice/debug_server.py | 完成,语法检查通过 | ~800 |
| 18:08 | 更新 wolf 文件 + PROJECT_STATE + FEATURE_INVENTORY(F23 多人脸DOA) | cerebrum/anatomy/memory/PROJECT_STATE/FEATURE_INVENTORY | 完成 | ~500 |
| 18:00 | Edited perception/vision_worker.py | 2→2 lines | ~49 |
| 18:00 | Edited perception/vision_worker.py | expanded (+14 lines) | ~278 |
| 18:01 | Edited perception/vision_worker.py | expanded (+11 lines) | ~330 |
| 18:03 | Edited voice/d01_realtime_chat.py | modified _select_face_by_doa() | ~286 |
| 18:04 | Edited voice/d01_realtime_chat.py | expanded (+16 lines) | ~261 |
| 18:04 | Edited voice/d01_realtime_chat.py | 7→9 lines | ~98 |
| 18:05 | Edited voice/debug_server.py | modified enumerate() | ~822 |
| 18:06 | Edited docs/FEATURE_INVENTORY.md | expanded (+9 lines) | ~163 |
| 18:07 | Edited PROJECT_STATE.md | 1→2 lines | ~91 |
| 18:09 | Session end: 10 writes across 5 files (FEATURE_INVENTORY.md, vision_worker.py, d01_realtime_chat.py, debug_server.py, PROJECT_STATE.md) | 7 reads | ~68646 tok |
| 18:10 | Edited todo.md | modified A() | ~415 |
| 18:12 | Edited voice/debug_server.py | modified _init_pil_fonts() | ~509 |
| 18:13 | Edited voice/debug_server.py | 2→3 lines | ~26 |
| 18:13 | Edited voice/debug_server.py | modified and() | ~152 |
| 18:14 | Edited voice/debug_server.py | 7→7 lines | ~138 |
| 18:16 | Edited voice/d01_realtime_chat.py | 4→7 lines | ~92 |
| 18:16 | Edited voice/d01_realtime_chat.py | 15→16 lines | ~236 |

## Session: 2026-06-24 18:19

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 18:21 | Edited voice/state.py | 1→2 lines | ~51 |
| 18:21 | Edited voice/d01_realtime_chat.py | expanded (+6 lines) | ~240 |
| 18:22 | Edited voice/d01_realtime_chat.py | expanded (+6 lines) | ~193 |
| 18:22 | Edited voice/debug_server.py | 1→2 lines | ~48 |
| 18:23 | Edited voice/debug_server.py | modified if() | ~273 |
| 18:23 | Edited voice/debug_server.py | added 1 condition(s) | ~60 |
| 18:24 | Edited voice/d01_realtime_chat.py | 13→13 lines | ~270 |
| 18:24 | Edited voice/d01_realtime_chat.py | 7→7 lines | ~160 |
| 18:25 | Edited todo.md | 20→20 lines | ~202 |
| 18:26 | Edited todo.md | 9→7 lines | ~92 |

## Session 2026-06-24 (continued)

| Time | Action | Files | Outcome | ~Tokens |
|------|--------|-------|---------|---------|
| cont | #21 fix: display_transcript 记录本 | state.py, d01_realtime_chat.py, debug_server.py | 新增持久 display_transcript 列表，不被摘要/close_session 清除；Dashboard 展示带时间戳+角色+人名的时间线 | ~800 |
| cont | todo.md 更新 #19/#21/#22 为已完成 | todo.md | 标记三项 bug fix 完成 | ~200 |
| 18:26 | Session end: 10 writes across 4 files (state.py, d01_realtime_chat.py, debug_server.py, todo.md) | 4 reads | ~58079 tok |
| 18:58 | Edited voice/state.py | 1→2 lines | ~36 |
| 18:58 | Edited voice/d01_realtime_chat.py | 7→8 lines | ~168 |
| 18:59 | Edited voice/d01_realtime_chat.py | 7→8 lines | ~166 |
| 18:59 | Edited voice/d01_realtime_chat.py | 8→11 lines | ~145 |
| 19:01 | Edited voice/debug_server.py | added 1 condition(s) | ~145 |
| 19:01 | Edited voice/debug_server.py | added 1 condition(s) | ~262 |
| 19:03 | Edited voice/debug_server.py | modified do_POST() | ~424 |
| cont | #21 refinement: display_transcript 按轮次快照 | state.py, d01_realtime_chat.py, debug_server.py | 每条 display_transcript 加 seq，每轮 turn 记录 dt_seq 快照点，modal 只显示该轮之前的对话 | ~600 |
| cont | #19 离线测试: mock-identity 端点 | debug_server.py | POST /debug/mock-identity {pid, name} 模拟切人，可单人测试记忆注入 | ~300 |
| 19:03 | Session end: 17 writes across 4 files (state.py, d01_realtime_chat.py, debug_server.py, todo.md) | 4 reads | ~59437 tok |
| 19:04 | Session end: 17 writes across 4 files (state.py, d01_realtime_chat.py, debug_server.py, todo.md) | 4 reads | ~59437 tok |
| 19:58 | Edited todo.md | 4→5 lines | ~67 |
| 19:58 | Session end: 18 writes across 4 files (state.py, d01_realtime_chat.py, debug_server.py, todo.md) | 4 reads | ~59477 tok |
| 19:58 | Session end: 18 writes across 4 files (state.py, d01_realtime_chat.py, debug_server.py, todo.md) | 4 reads | ~59477 tok |
| 19:59 | Session end: 18 writes across 4 files (state.py, d01_realtime_chat.py, debug_server.py, todo.md) | 4 reads | ~59477 tok |
| 20:00 | Session end: 18 writes across 4 files (state.py, d01_realtime_chat.py, debug_server.py, todo.md) | 4 reads | ~59508 tok |
| 20:02 | Session end: 18 writes across 4 files (state.py, d01_realtime_chat.py, debug_server.py, todo.md) | 4 reads | ~59508 tok |

## Session: 2026-06-25 09:46

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|

## Session: 2026-06-25 10:21

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 10:29 | Created ../../../../.claude/plans/flickering-brewing-puppy.md | — | ~852 |
| 10:31 | Session end: 1 writes across 1 files (flickering-brewing-puppy.md) | 12 reads | ~52368 tok |
| 10:35 | Session end: 1 writes across 1 files (flickering-brewing-puppy.md) | 12 reads | ~52368 tok |
| 10:43 | Created ../../../../.claude/plans/rustling-booping-biscuit.md | — | ~1629 |
| 10:45 | Edited ../../../../.claude/plans/flickering-brewing-puppy.md | modified _handle_clear() | ~1095 |
| 10:55 | Edited ../../../../.claude/plans/flickering-brewing-puppy.md | modified __init__() | ~564 |
| 10:55 | Edited ../../../../.claude/plans/flickering-brewing-puppy.md | 12→12 lines | ~115 |
| 10:55 | Edited ../../../../.claude/plans/flickering-brewing-puppy.md | 36→36 lines | ~228 |
| 10:56 | Edited ../../../../.claude/plans/flickering-brewing-puppy.md | modified _handle_clear() | ~226 |
| 11:56 | Created voice/kws.py | — | ~991 |
| 11:58 | Created perception/fusion.py | — | ~289 |
| 11:58 | Created memory/safety.py | — | ~1023 |
| 12:01 | Created voice/realtime.py | — | ~6613 |
| 12:02 | Edited voice/d01_realtime_chat.py | reduced (-10 lines) | ~104 |
| 12:03 | Edited voice/d01_realtime_chat.py | 46→44 lines | ~663 |
| 12:03 | Edited voice/d01_realtime_chat.py | added 4 import(s) | ~66 |
| 12:05 | Edited voice/d01_realtime_chat.py | removed 394 lines | ~174 |
| 12:07 | Edited voice/d01_realtime_chat.py | removed 134 lines | ~20 |
| 12:08 | Edited voice/d01_realtime_chat.py | removed 73 lines | ~23 |
| 12:09 | Edited voice/d01_realtime_chat.py | removed 162 lines | ~453 |
| 12:10 | Edited voice/d01_realtime_chat.py | 2→2 lines | ~20 |
| 12:10 | Edited voice/d01_realtime_chat.py | inline fix | ~7 |
| 12:11 | Edited voice/d01_realtime_chat.py | inline fix | ~8 |
| 12:11 | Edited voice/d01_realtime_chat.py | inline fix | ~15 |
| 12:11 | Edited voice/d01_realtime_chat.py | inline fix | ~11 |
| 12:11 | Edited voice/d01_realtime_chat.py | inline fix | ~13 |
| 12:12 | Edited voice/d01_realtime_chat.py | inline fix | ~12 |
| 12:12 | Edited voice/d01_realtime_chat.py | inline fix | ~12 |
| 12:12 | Edited voice/d01_realtime_chat.py | inline fix | ~10 |
| 12:13 | Edited voice/d01_realtime_chat.py | inline fix | ~23 |
| 12:16 | Session end: 28 writes across 7 files (flickering-brewing-puppy.md, rustling-booping-biscuit.md, kws.py, fusion.py, safety.py) | 12 reads | ~57614 tok |
| 12:16 | Session end: 28 writes across 7 files (flickering-brewing-puppy.md, rustling-booping-biscuit.md, kws.py, fusion.py, safety.py) | 12 reads | ~57614 tok |
| 14:06 | Edited voice/d01_realtime_chat.py | added 1 import(s) | ~7 |
| 14:08 | Session end: 29 writes across 7 files (flickering-brewing-puppy.md, rustling-booping-biscuit.md, kws.py, fusion.py, safety.py) | 12 reads | ~57625 tok |
| 14:13 | Edited voice/d01_realtime_chat.py | inline fix | ~34 |
| 14:13 | Session end: 30 writes across 7 files (flickering-brewing-puppy.md, rustling-booping-biscuit.md, kws.py, fusion.py, safety.py) | 12 reads | ~57666 tok |
| 14:15 | Edited voice/d01_realtime_chat.py | 8→8 lines | ~149 |

## Session: 2026-06-25 14:17

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 14:17 | 门控逻辑白名单化: state==ST_ENGAGING → state!=ST_TRACKING | voice/d01_realtime_chat.py | 仅TRACKING关门,其他状态放行 | ~200 |
| 14:18 | 更新 cerebrum.md: 门控白名单决策 + Do-Not-Repeat | .wolf/cerebrum.md | 新增决策日志+错误避免 | ~100 |
