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
