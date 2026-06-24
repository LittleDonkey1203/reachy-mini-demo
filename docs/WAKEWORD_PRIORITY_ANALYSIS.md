# 唤醒词 × 人脸锁定 × DOA 优先级分析

> 分析日期: 2026-06-24 | 范围: voice/d01_realtime_chat.py, voice/audio.py, voice/state.py, voice/config.py, perception/vision_worker.py
> 问题: 机器人已在和 A 对话/跟踪 A 的脸时,B 喊"小艺"得不到响应

---

## 0. TL;DR(结论先行)

**核心结论:B 喊"小艺"时机器人不响应,根因不是"人脸锁阻止了转头",而是 main 循环里的 `_is_A` 门控逻辑误判**。

二次唤醒切换(M1.5-b)的代码其实**已经存在**(`d01_realtime_chat.py:1942-1969`),理论上 B 喊"小艺"应该触发 `switch_request` 让 behavior 转向 B。它失效的真实原因是判定"这是 A 自己又喊"的门控 `_is_A` 在很多情况下**误判为 True**,于是直接 `忽略` 了 B 的唤醒:

```python
_is_A = _sfresh and _sconf and abs(_sr) <= GATE_DEG   # GATE_DEG = 55°
if not _is_A:
    ...触发切换...
# else: 当作"A 自己又喊" → 忽略
```

只要 DOA 此刻是 fresh + confident 且残差 ≤ 55°(B 站在机器人正前方约 ±55° 锥形内,这是常见情况),B 的唤醒就被当成"A 在正前方又喊了一次"而**静默丢弃**。同时,这个门控**完全没有考虑 A 是否正在说话**——这正是用户想要的语义:A 在说话才该屏蔽 B,A 沉默时应该响应 B。

---

## 1. 当前实现分析:KWS / DOA / 人脸 / FSM 如何交互

### 1.1 四个子系统与共享状态

所有跨线程数据走 `State`(`voice/state.py:212`),`threading.Lock` 保护。关键字段:

| 字段 | 写者 | 含义 |
|---|---|---|
| `state` | behavior_loop(唯一写者) | FSM 当前态 |
| `wake_ok` | main | KWS 命中且连接成功 → 通知 behavior 进入 SEEK |
| `wake_doa` | (置 None,未真正使用) | — |
| `switch_request` | main | 二次唤醒切换请求(dict: resid/confident/fresh) |
| `doa_resid_stable` / `doa_confident` / `doa_at` | doa_sensor_loop | 2s 窗口稳定残差(给切换门控/唤醒寻人用) |
| `sound_resid` / `sound_at` / `sound_spread` | doa_sensor_loop | 1.5s 窗口残差(给 in-conversation 声源转向用) |
| `face_locked` | vision_result_loop | 迟滞锁定(LOCK_ON/OFF_RATE) |
| `track_yaw` / `body_yaw_deg` / `track_pitch` | behavior(非 TRACKING)/ 视觉积分(TRACKING) | 头/体目标角 |
| `playback_end_estimate` | player_loop | 机器人说话结束的预估时刻(=机器人"在说话") |

### 1.2 KWS 唤醒链路(本地,始终运行)

- `KwsGate`(`d01_realtime_chat.py:1604`):sherpa-onnx 本地关键词识别,单字"小艺"三形态(`KWS_FORMS`)。
- main 主循环每个音频块都喂 KWS(`d01_realtime_chat.py:1910`):`wake = kws_gate.feed(mono, chunk)`。**ARMED 和 engaged 状态都喂**——这是支持二次唤醒的基础。
- 去抖/不应期(`d01_realtime_chat.py:1664-1670`):`KWS_REFRACTORY_S = 2.0`(两次真唤醒最小间隔),`KWS_DEBOUNCE_S = 0.3`。

### 1.3 DOA 链路(纯传感器,不动头)

- `doa_sensor_loop`(`voice/audio.py:34`):10Hz 轮询 REST 端点 `http://127.0.0.1:8000/api/state/doa`,返回 `{angle, speech_detected}`。
- **只在 `speech_detected=True`(VAD)时入窗**(`audio.py:57`)。
- 维护两个窗口:
  - `buf`(1.5s,`SND_WIN_S`)→ `sound_resid`(给 IDLE 态 in-conversation 声源转向)。
  - `buf2`(2.0s,`DOA_WIN_S`)→ `doa_resid_stable` + `doa_confident`(IQR < `GATE_SPREAD=25°` 才 confident)。
- 残差定义:`resid = 90.0 - med`,即相对机器人正前方的偏角(符号:+左 / -右,见 config 注释)。
- ⚠ 注意:DOA 读数在机器人自己说话时**也会入窗**——代码计算了 `robot_speaking`(`audio.py:55`)但**只用于 DEBUG 日志,没有用它过滤入窗**。这意味着机器人说话时扬声器/自声反射可能污染 DOA(`DOA_SEEK_DRAFT.md` 提到的镜像翻转问题之外的另一个隐患)。

### 1.4 人脸链路与锁定

- `FaceSelector`(`vision_worker.py:83`):跨帧粘滞选脸。锁住一张脸后跟着它,除非它消失或另一张脸连续 `STICKY_SWITCH_FRAMES=8` 帧 `> 当前×1.20` 才切换。
- `face_locked`(迟滞,`d01_realtime_chat.py:867-902` 附近)由 vision_result_loop 维护:`LOCK_ON_RATE=0.40` / `LOCK_OFF_RATE=0.15`,带 `LOST_HOLD_S=1.5` 丢锁宽限。
- **sticky_reset**:`vision_worker.py:318` 收到 `"sticky_reset"` 消息时 `face_sel.reset()`。当前**只有两处发送**:
  1. 二次唤醒切换时(`d01_realtime_chat.py:1955-1957`)。
  2. (DOA_SEEK_DRAFT 提到首次唤醒未重置,确实如此——ARMED→ENGAGING 唤醒寻人路径没发 sticky_reset。)

### 1.5 FSM 状态机(behavior_loop 是唯一写者)

`behavior_loop`(`d01_realtime_chat.py:941`)25Hz 调度。状态:`ARMED → ENGAGING → TRACKING ↔ SEARCHING → RETURNING → ARMED`,外加 `POINTING / PLAYING`。

- **ARMED**(`:1023`):待命,只等 `wake_ok`。唤醒后读 DOA → 进入 ENGAGING 做 SEEK 寻人(`:1031-1054`)。
- **ENGAGING**(`:1184`):两条子路径:
  - `switching=True`(二次唤醒切换):turn → sweep → 锁 B / 超时回 A(`:1185-1226`)。
  - `wide_scan=True`(唤醒寻人 SEEK)或普通声源转向(`:1227-1298`)。
- **TRACKING**(`:1300`):头部交给视觉积分,behavior 只判转出条件。
- 切换/退出/指向请求在 ENGAGING 之前作为高优先级 flag 处理(`:1061-1143`)。

---

## 2. 问题根因:为什么 B 的唤醒被忽略

### 2.1 主路径:`_is_A` 门控误判(最主要)

`d01_realtime_chat.py:1942-1969`:

```python
if wake and not no_switch and (time.monotonic() - last_switch) > SWITCH_COOLDOWN_S:
    with st.lock:
        _sr, _sat, _sconf = st.doa_resid_stable, st.doa_at, st.doa_confident
    _sfresh = _sr is not None and (nowk - _sat) < DOA_GATE_FRESH_S   # 1.5s
    _is_A = _sfresh and _sconf and abs(_sr) <= GATE_DEG              # 55°
    if not _is_A:
        last_switch = nowk
        st.switch_request = {...}        # ← 触发切换
        vis_frame_q.put_nowait("sticky_reset")
        close_session(conv); conv = open_session()
        continue
    # else: A 自己又喊 → 忽略,继续正常对话
```

**问题:`_is_A` 的语义是"DOA 此刻稳定指向正前方 ±55° 内"。** 它用来判断"是不是 A 自己又喊了一句"。但是:

1. **B 站在机器人前方锥形内(±55°)时,`_is_A` 为 True → B 被当成 A,静默忽略。** 这是最常见的失败场景:两个人都在机器人正面附近时,B 怎么喊都没用。
2. **门控完全没看"A 是否在说话"。** 用户想要的是:A 在说话→屏蔽 B;A 沉默→响应 B。当前逻辑只看 DOA 方向,不看说话与否,语义错位。
3. **DOA 此刻的残差可能就是 B 的方向**(B 刚喊"小艺",VAD 入窗的正是 B 的声音),于是"B 在前方"反而被判成"A 在前方又喊",逻辑自相矛盾。
4. **`doa_confident` 受镜像翻转影响**(`DOA_SEEK_DRAFT.md` 第 1 点):XVF3800 可能稳定指向错误方向,IQR 很小所以 confident=True,残差符号是错的——`_is_A` 的判断基础本就不可靠。

### 2.2 次要路径:人脸锁定确实会"拽住"

即使 `switch_request` 成功置位,切换流程中人脸锁定也设计为**主动压制**,以免转向途中被 A 的脸拽回来:

- 切换 turn 阶段**完全不认脸**(`d01_realtime_chat.py:1187` 注释:"turn 阶段完全不认脸");只有 sweep 阶段 **且 `turned_away`(离 A > `SWITCH_AWAY_DEG=35°`)** 才允许锁 B(`:1191`)。
- 发了 `sticky_reset` 清掉 FaceSelector 的粘滞,但 `face_locked` 迟滞仍有 `LOST_HOLD_S=1.5s` 宽限。

这套压锁逻辑本身是对的(防止刚要转向 B 就被 A 的脸拉回),但它**只在切换被触发后**才起作用。真正的瓶颈在 §2.1 的门控——切换根本没被触发。

### 2.3 还有一个静默杀手:`SWITCH_COOLDOWN_S` + refractory

- `KWS_REFRACTORY_S=2.0`:B 喊"小艺"后 2s 内的重复唤醒被 KwsGate 吞掉。
- `SWITCH_COOLDOWN_S=2.0`:上次切换后 2s 内的唤醒不触发切换(`:1945`)。
- 这两个叠加,在快速连喊时会让部分唤醒"消失"。但这是次要因素,不是主因。

---

## 3. 门控逻辑梳理:当前哪些条件会阻止响应 B

按 main 主循环执行顺序,B 的唤醒要走到"触发切换"需穿过的全部门:

| 门 | 条件 | 位置 | 拦截后果 |
|---|---|---|---|
| KWS 不应期 | `t - last_wake < 2.0s` | `:1665` | 唤醒被吞 |
| 必须 engaged | `state != ARMED` 且 `conv is not None` | `:1914,1939` | ARMED 走唤醒寻人(正常),engaged 才走切换 |
| 切换开关 | `not no_switch` | `:1945` | `--no-switch` 关闭切换 |
| 切换冷却 | `now - last_switch > 2.0s` | `:1945` | 切换被吞 |
| **`_is_A` 门控** | `not (fresh & confident & |resid|≤55°)` | `:1950-1951` | **核心:误判 B=A 则忽略** |
| (behavior)切换态守卫 | `state not in (ARMED, POINTING)` | `:1080` | POINTING 中不切换 |
| (behavior)turned_away | sweep 阶段 + 离 A>35° + locked | `:1191` | 没转够远不认 B |

**关键观察:整条链路没有任何一个门是基于"A 此刻是否在说话"(`speaking = now < playback_end_estimate`,或用户侧 VAD)。** behavior_loop 里 `speaking` 只用于:IDLE 态声源转向抑制(`:1172`)、无互动超时判定(`:1180,1304`),**完全没参与二次唤醒决策**。这正是用户期望语义缺失的地方。

---

## 4. 与现有 DOA 优化方案(DOA_SEEK_DRAFT.md)对比

`docs/DOA_SEEK_DRAFT.md` 关注的是**唤醒寻人(ARMED→SEEK)的转向质量**,不是多人优先级:

| DOA_SEEK_DRAFT 关注点 | 与本问题的关系 |
|---|---|
| DOA confident ≠ correct(镜像翻转) | **直接相关**:`_is_A` 依赖 `doa_confident`,翻转会让门控判断错误 |
| 唤醒时未重置 FaceSelector | 间接相关:本问题里切换路径已发 sticky_reset,但首次唤醒仍未发 |
| "不动"场景(sin(0)=0 起扫位移小) | 不相关 |
| 草案:去掉三阶段,改单一连续弧线扫描 | 不相关(那是寻人动作风格,不涉及优先级仲裁) |

**结论:DOA_SEEK_DRAFT 的"DOA 只用 sign、不信 confident"主张,恰好佐证了本分析 §2.1.4——`_is_A` 不该依赖 confident。** 但 DOA_SEEK_DRAFT 没有触及"A 说话时屏蔽 B / A 沉默时响应 B"这个核心语义,本分析是对它的补充。

---

## 5. 改进方案建议

### 5.1 设计目标(用户语义)

> A 正在说话(VAD active 且 DOA 匹配 A 的方向)→ 屏蔽 B 的唤醒。
> A 沉默(无 VAD,或有 VAD 但 DOA 不指向 A)→ 响应 B,转向 DOA 方向 SEEK B。

### 5.2 核心改动:把 `_is_A` 门控从"方向门"换成"A 在说话门"

当前门控问的是"DOA 是否在正前方"。应改成问"**A 此刻是否正在占用对话(在说话)**"。

**判定"A 正在说话"的信号(任一即可,建议组合):**

1. **机器人正在回应 A**:`in_flight > 0`(模型在生成)或 `speaking = now < playback_end_estimate`(机器人在播音)。机器人在答 A 时,B 不该插队。
2. **用户(A)正在说话**:服务端 VAD 的 `input_audio_buffer.speech_started` 未收到对应 `speech_stopped`。当前 `on_event` 已经在处理这两个事件(`d01_realtime_chat.py:344`),只需在 `State` 增加一个 `user_speaking` flag,在 `speech_started` 置 True、`speech_stopped` 置 False。
3. **(可选)DOA 方向匹配 A**:记录锁定 A 时的 `track_yaw` 作为 A 的方向 `person_a_yaw`,若当前 DOA 残差换算到世界系后接近 `person_a_yaw`(差 < 阈值如 20°),说明说话声来自 A 方向。

**建议的新门控逻辑(替换 `:1950-1951`):**

```python
with st.lock:
    a_busy = st.in_flight > 0 or now < st.playback_end_estimate  # 机器人在答 A
    user_talking = st.user_speaking                              # A 还在说(VAD)
# A 正在占用对话 → 屏蔽 B;否则放行切换
a_active = a_busy or user_talking
if not a_active:
    # A 沉默 → 响应 B:用 DOA 给方向提示,触发切换/SEEK
    last_switch = nowk
    st.switch_request = {"resid": _sr, "confident": _sconf, "fresh": _sfresh}
    ...
# else: A 在说话 → 忽略 B(可选:记一个 pending,A 说完后再问)
```

这样:
- A 在说话 / 机器人在答 A → B 静默(符合"不打断")。
- A 沉默(双方都没在说,或上一轮答完了)→ B 喊"小艺"立刻转向 B 的 DOA 方向。
- 不再依赖不可靠的 `doa_confident` 做"是不是 A"的判断。

### 5.3 DOA 方向用 sign 不用 confident(呼应 DOA_SEEK_DRAFT)

切换转向时,`switch_request` 里仍可带 `resid`,但 behavior 决定转向方向时**优先用残差符号(±=左右)**而非精确角度,降低镜像翻转的伤害。当前 behavior 的三档逻辑(`:1087-1103`)已经有"confident 直转 / fresh 粗方向 / 无 DOA 反向"的降级,可以把档一(confident 直转)的权重调低,默认走粗方向。

### 5.4 需要新增/修改的具体点

1. **`voice/state.py`**:`State.__init__` 增加 `self.user_speaking = False`(以及可选 `self.person_a_yaw = None`)。
2. **`d01_realtime_chat.py` `ChatCallback.on_event`**:
   - `input_audio_buffer.speech_started`(`:344`)→ `st.user_speaking = True`。
   - `input_audio_buffer.speech_stopped` → `st.user_speaking = False`。
3. **`d01_realtime_chat.py:1945-1969`**:把 `_is_A` 门控替换为 §5.2 的 `a_active` 门控。
4. **(可选)记录 A 方向**:behavior 锁定 A 进入 TRACKING 时(`:1199,1237`)记 `st.person_a_yaw = track_yaw`,用于 §5.2 信号 3 的方向匹配。
5. **(可选)A 说话时的 B pending**:若 B 在 A 说话时喊了,存一个 `pending_switch`,A 说完(`speech_stopped` 且 `in_flight==0`)后再触发,避免完全丢弃 B。

### 5.5 注意的副作用

- **机器人自己说话期间 DOA 被污染**(`audio.py:55` 的 `robot_speaking` 没用于过滤入窗)。若放行 B 时 DOA 此刻是机器人自声,方向会错。建议同时修 `audio.py`:`robot_speaking` 时不入窗(或入窗后标记低可信)。
- **`speaking`(机器人在播音)正是 `a_busy` 的一部分**,所以"机器人正给 A 朗读时 B 不插队"自动满足——这通常是用户能接受的:等机器人说完这句再响应 B。

---

## 6. 优先级仲裁矩阵

记 A=当前对话对象,B=新喊"小艺"的人。"A 在说话" = 机器人正在答 A(`in_flight>0` 或 `speaking`)**或** 用户侧 VAD active(A 在讲)。

| 场景 | 当前行为 | 期望行为 | 当前是否正确 |
|---|---|---|---|
| 待命(ARMED)+ B 喊 | 唤醒 → SEEK 寻 B(`:1023-1054`) | 同 | ✅ 正确 |
| 跟踪 A(TRACKING)+ A 沉默 + B 喊,B 在 ±55° 内 | `_is_A`=True → **忽略 B** | 转向 B SEEK | ❌ **核心 bug** |
| 跟踪 A + A 沉默 + B 喊,B 在 55° 外 | `_is_A`=False → 切换转 B | 转向 B SEEK | ✅(碰巧对) |
| 跟踪 A + A 正在说话 + B 喊 | `_is_A` 看方向(可能误切,可能忽略) | **忽略 B**(不打断 A) | ⚠ 不稳定 |
| 机器人正给 A 朗读 + B 喊 | 同上,看 DOA 方向 | **忽略 B**(等机器人说完) | ⚠ 不稳定 |
| 跟踪 A + A 自己又喊"小艺"(A 在正前方) | `_is_A`=True → 忽略(正确) | 忽略(已在对话) | ✅ 正确 |
| 切换冷却 2s 内 B 再喊 | 被冷却吞 | 容忍(防抖) | ✅ 可接受 |

**改进后的目标矩阵(§5.2 落地后):**

| 场景 | 门控 `a_active` | 结果 |
|---|---|---|
| A 沉默 + B 喊(任意方向) | False | 转向 B(DOA sign 给方向)|
| A 正在说话 + B 喊 | True | 忽略 B(可选 pending,A 说完再响应)|
| 机器人正答 A + B 喊 | True | 忽略 B(等本轮答完)|
| A 自己又喊 | A 通常此刻在说话或刚说完 → 多半 True,且方向=A | 忽略(继续对话)|

---

## 7. 改动优先级建议

1. **P0**:`state.py` 加 `user_speaking` + `on_event` 维护它 + 替换 `_is_A` 为 `a_active` 门控(§5.4 #1-3)。这是直接修复用户报告问题的最小改动。
2. **P1**:`audio.py` 修 `robot_speaking` 不入窗,提升放行 B 后 DOA 方向可信度(§5.5)。
3. **P2**:记录 `person_a_yaw` 做方向匹配 + A 说话时 B 的 pending 队列(§5.4 #4-5),让仲裁更精确、不丢 B。
4. **P3**:呼应 DOA_SEEK_DRAFT,切换方向默认用 sign 不用 confident(§5.3)。

> 全部为**只读分析**,未修改任何源文件。
