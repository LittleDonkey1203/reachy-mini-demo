# Memory

> Chronological action log. Old sessions archived as one-line summaries.

## Archived Sessions (2026-06-23 ~ 2026-06-24)

| Date | Summary | Key Files | ~Tokens |
|------|---------|-----------|---------|
| 06-23 16:00 | YuNet 后端切换; vision_worker 双后端支持; debug_server float32 JSON 修复; forget_fact 工具; 唤醒方向门控修复 | perception/vision_worker.py, voice/config.py, start_mac.sh, memory/manager.py, voice/d01_realtime_chat.py | ~40k |
| 06-24 09:00 | Face DB 碎片化修复(质心匹配+auto_merge); GestureRecognizer 替换规则; 认主机制; 记忆注入 update_session 修复 | identity/recognizer.py, memory/manager.py, identity/owner.py, voice/d01_realtime_chat.py | ~50k |
| 06-24 11:00 | 唤醒优先级(a_active替代_is_A); TRACKING 身体跟随; 人脸误识别迟滞; clear_memory confirmed 守卫; debug_server 身份标注 | voice/state.py, voice/d01_realtime_chat.py, voice/config.py, voice/debug_server.py, memory/manager.py | ~70k |
| 06-24 15:00 | 安全删除工作流(多步验证+备份); Dashboard log smart-scroll; test_realtime_model.py; face_db 迁移 shutil.copy2 | voice/d01_realtime_chat.py, voice/debug_server.py, test_realtime_model.py, identity/recognizer.py | ~55k |
| 06-24 17:00 | 分人对话摘要(per-pid conv_log); 音频闸门(切人+DOA偏移); CONV_SUMMARY_THRESHOLD 自动摘要; display_transcript 持久记录本; Dashboard 上下文调试 | voice/state.py, voice/d01_realtime_chat.py, memory/manager.py, voice/config.py, voice/debug_server.py | ~60k |
| 06-24 18:00 | 多人脸 DOA 选人(_select_face_by_doa); all_faces 输出; debug overlay 多人框渲染; 中文渲染 PIL 修复 | perception/vision_worker.py, voice/d01_realtime_chat.py, voice/debug_server.py | ~60k |

## Session: 2026-06-25

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 10:30 | d01 瘦身重构方案(领域驱动) — plan 编写 | flickering-brewing-puppy.md | 拆出 kws.py/fusion.py/safety.py/realtime.py | ~3k |
| 11:56 | 创建 4 个新模块 | voice/kws.py, perception/fusion.py, memory/safety.py, voice/realtime.py | 领域模块就位 | ~10k |
| 12:01 | d01 瘦身: 删除已拆出的代码(~763行) | voice/d01_realtime_chat.py | 编译通过 | ~5k |
| 14:17 | 方向门控白名单化: state==TRACKING时才关门 | voice/d01_realtime_chat.py, .wolf/cerebrum.md | 防止新状态默认被关门 | ~2k |
| 14:50 | fix: 唤醒竞态 — wake_ok 延迟到 set_state(ENGAGING) 之后清除 | voice/d01_realtime_chat.py | 消除 audio/behavior 竞态窗口,无需超时阈值 | ~300 |
| 14:50 | fix: conv=None 时 KWS 命中可重连 WS | voice/d01_realtime_chat.py | WS 断连后不再卡死在对话态 | ~200 |
| 14:43 | Edited voice/d01_realtime_chat.py | 33→36 lines | ~530 |
| 14:44 | Created todo.md | — | ~494 |
| 14:45 | Created PROJECT_STATE.md | — | ~426 |
| 14:46 | Session end: 11 writes across 3 files (d01_realtime_chat.py, todo.md, PROJECT_STATE.md) | 6 reads | ~30463 tok |
| 14:47 | Session end: 11 writes across 3 files (d01_realtime_chat.py, todo.md, PROJECT_STATE.md) | 6 reads | ~30463 tok |
| 14:48 | Edited todo.md | expanded (+9 lines) | ~58 |
| 14:48 | Session end: 12 writes across 3 files (d01_realtime_chat.py, todo.md, PROJECT_STATE.md) | 6 reads | ~28419 tok |
| 14:51 | Edited todo.md | — | ~0 |
| 14:51 | Edited todo.md | expanded (+12 lines) | ~139 |
| 14:52 | Session end: 14 writes across 3 files (d01_realtime_chat.py, todo.md, PROJECT_STATE.md) | 6 reads | ~28568 tok |

## Session: 2026-06-25 14:59

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|

## Session: 2026-06-25 15:28

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|

## Session: 2026-06-25 15:29

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 15:30 | Edited todo.md | expanded (+16 lines) | ~146 |
| 15:30 | Session end: 1 writes across 1 files (todo.md) | 2 reads | ~7334 tok |
| 15:39 | Edited voice/realtime.py | modified open_session() | ~146 |
| 15:39 | Edited voice/debug_server.py | 4→4 lines | ~72 |
| 15:39 | Edited voice/d01_realtime_chat.py | 2→3 lines | ~38 |
| 15:39 | Edited voice/d01_realtime_chat.py | 5→8 lines | ~90 |
| 16:00 | fix: vis_ready日志刷屏→仅首次log | voice/d01_realtime_chat.py | 20+行→1行 | ~200 |
| 16:00 | fix: Dashboard角色混淆→assistant不显示人名 | voice/debug_server.py | 🔊无[名]标签 | ~100 |
| 16:00 | fix: WS快速重连死→open_session 1s最小间隔 | voice/realtime.py | 防服务端限流 | ~200 |
| 16:01 | buglog: bug-048/049/050 | .wolf/buglog.json | 3个bug记录 | ~100 |
| 15:41 | Session end: 5 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 5 reads | ~52757 tok |
| 15:48 | Edited voice/debug_server.py | 2→2 lines | ~44 |
| 15:49 | Session end: 6 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 5 reads | ~52801 tok |
| 15:53 | Session end: 6 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 6 reads | ~52801 tok |
| 15:57 | Session end: 6 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 6 reads | ~52849 tok |
| 16:12 | Session end: 6 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 6 reads | ~52849 tok |
| 16:14 | Edited voice/d01_realtime_chat.py | modified _detect_new_speaker() | ~165 |
| 16:14 | Edited voice/d01_realtime_chat.py | modified log() | ~485 |
| 16:15 | Session end: 8 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 6 reads | ~53641 tok |
| 16:35 | Edited voice/d01_realtime_chat.py | modified _detect_new_speaker() | ~164 |
| 16:35 | Edited voice/d01_realtime_chat.py | modified log() | ~574 |
| 16:36 | Session end: 10 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 7 reads | ~54345 tok |
| 16:38 | Edited voice/d01_realtime_chat.py | added 2 condition(s) | ~600 |
| 16:38 | Session end: 11 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 7 reads | ~54945 tok |
| 16:44 | Edited voice/d01_realtime_chat.py | 15→17 lines | ~278 |
| 16:45 | Edited voice/d01_realtime_chat.py | 17→16 lines | ~268 |
| 16:45 | Edited voice/d01_realtime_chat.py | 1→2 lines | ~20 |
| 16:46 | Edited voice/d01_realtime_chat.py | inline fix | ~27 |
| 16:46 | Session end: 15 writes across 4 files (todo.md, realtime.py, debug_server.py, d01_realtime_chat.py) | 7 reads | ~55652 tok |
| 16:59 | Created ../../../../.claude/plans/rustling-booping-biscuit.md | — | ~1828 |

## Session: 2026-06-25 16:59

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 17:50 | Created ../../../../.claude/plans/rustling-booping-biscuit.md | — | ~1382 |
| 17:55 | Created ../../../../.claude/plans/rustling-booping-biscuit.md | — | ~1544 |
| 18:18 | Session end: 2 writes across 1 files (rustling-booping-biscuit.md) | 2 reads | ~28002 tok |
| 18:20 | Created ../../../../.claude/plans/rustling-booping-biscuit.md | — | ~2265 |
| 18:24 | Edited identity/recognizer.py | modified __init__() | ~117 |
| 18:25 | Edited voice/d01_realtime_chat.py | modified items() | ~130 |
| 18:27 | Created memory/manager.py | — | ~5321 |
| 18:28 | Edited memory/manager.py | added 1 import(s) | ~37 |
| 18:28 | Edited memory/manager.py | 2→2 lines | ~24 |

## Session: 2026-06-25 18:31

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 18:34 | Edited voice/realtime.py | modified has_owner() | ~431 |
| 18:34 | Edited voice/realtime.py | modified save_summary() | ~696 |
| 18:35 | Edited voice/realtime.py | "📝 上下文过长，自动触发摘要({_log_pid" → "📝 上下文过长，自动触发 consolidati" | ~30 |
| 18:35 | Edited voice/realtime.py | "断开 WS、清身份状态、触发摘要。" → "断开 WS、清身份状态、触发 consolidat" | ~13 |
| 18:35 | Edited voice/d01_realtime_chat.py | inline fix | ~15 |
| 18:36 | Created PROJECT_STATE.md | — | ~688 |
| 18:37 | Edited docs/FEATURE_INVENTORY.md | 21→21 lines | ~518 |
| 18:39 | Session end: 7 writes across 4 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md, FEATURE_INVENTORY.md) | 4 reads | ~42503 tok |
| 20:28 | Edited memory/manager.py | inline fix | ~18 |
| 20:29 | Session end: 8 writes across 5 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md, FEATURE_INVENTORY.md, manager.py) | 5 reads | ~47855 tok |
| 20:32 | Session end: 8 writes across 5 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md, FEATURE_INVENTORY.md, manager.py) | 5 reads | ~47855 tok |
| 20:35 | Session end: 8 writes across 5 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md, FEATURE_INVENTORY.md, manager.py) | 5 reads | ~47855 tok |
| 20:37 | Edited voice/d01_realtime_chat.py | 5→5 lines | ~71 |
| 20:37 | Session end: 9 writes across 5 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md, FEATURE_INVENTORY.md, manager.py) | 5 reads | ~47930 tok |
| 20:50 | Session end: 9 writes across 5 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md, FEATURE_INVENTORY.md, manager.py) | 6 reads | ~47930 tok |
| 21:02 | Edited voice/d01_realtime_chat.py | 8→11 lines | ~198 |
| 21:04 | Edited voice/config.py | expanded (+7 lines) | ~140 |
| 21:05 | Edited voice/realtime.py | modified close_session() | ~312 |

## Session: 2026-06-25 21:06

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 21:09 | Edited voice/realtime.py | 16→20 lines | ~273 |
| 21:10 | Edited voice/d01_realtime_chat.py | 5→7 lines | ~104 |
| 21:11 | Edited voice/d01_realtime_chat.py | added 1 condition(s) | ~510 |

## Session: 2026-06-25 (continued — context resumed)

| Time | Action | File(s) | Outcome | ~Tokens |
|------|--------|---------|---------|--------|
| 21:20 | Fix #22: close_session 遍历所有 pid | voice/realtime.py | 多人 conv_log 不再丢失 | ~400 |
| 21:22 | Fix #23: consolidation prompt 增强 | voice/realtime.py | 明确提取新 facts + 排除 name + 扩 4000 字符 | ~350 |
| 21:24 | Fix #24: 身份切换加冷却 + 提高确认次数 | voice/d01_realtime_chat.py | CONFIRM_N=3, COOLDOWN=6s | ~300 |
| 21:12 | Edited PROJECT_STATE.md | 1→6 lines | ~79 |
| 21:12 | Session end: 4 writes across 3 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md) | 3 reads | ~33687 tok |
| 21:16 | Session end: 4 writes across 3 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md) | 4 reads | ~51256 tok |
| 21:22 | Session end: 4 writes across 3 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md) | 4 reads | ~51256 tok |
| 21:26 | Edited voice/realtime.py | 11→13 lines | ~182 |
| 21:26 | Edited voice/realtime.py | 11→11 lines | ~232 |
| 21:26 | Edited voice/realtime.py | modified in() | ~49 |
| 21:27 | Edited voice/state.py | 2→4 lines | ~56 |
| 21:28 | Edited voice/realtime.py | 3→5 lines | ~61 |
| 21:29 | Edited voice/debug_server.py | modified if() | ~624 |
| 21:29 | Edited voice/realtime.py | expanded (+10 lines) | ~255 |
| 21:30 | Edited voice/debug_server.py | added 1 condition(s) | ~356 |
| 21:30 | Edited voice/debug_server.py | 50 → 100 | ~21 |
| 21:35 | Fix #20: response.created 身份快照 | voice/realtime.py, voice/state.py | 回复期间 pid 不再跟着人脸线程跳 | ~400 |
| 21:40 | Fix #25: Dashboard 上下文重建视图 | voice/debug_server.py, voice/realtime.py | modal 里展示 [System]+[User]+[ToolCall]+[Assistant]+[Tools] 完整模型视角 | ~500 |
| 21:31 | Edited PROJECT_STATE.md | 1→3 lines | ~124 |
| 21:32 | Edited memory/manager.py | 3→3 lines | ~39 |
| 21:32 | Session end: 15 writes across 6 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 6 reads | ~61951 tok |
| 21:34 | Session end: 15 writes across 6 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 6 reads | ~61951 tok |
| 21:40 | Edited voice/d01_realtime_chat.py | 6→6 lines | ~86 |
| 21:40 | Edited voice/realtime.py | 4→5 lines | ~68 |
| 21:41 | Session end: 17 writes across 6 files (realtime.py, d01_realtime_chat.py, PROJECT_STATE.md, state.py, debug_server.py) | 7 reads | ~62273 tok |
