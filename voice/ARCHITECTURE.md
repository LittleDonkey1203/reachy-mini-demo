# d01_realtime_chat.py 架构说明(MAIN-01 任务1)

> 2026-06-07。对象:`voice/d01_realtime_chat.py` + `voice/vision_worker.py`(视觉子进程)。
> 定位:完整体伴侣机器人 = **传感器层 → 行为状态机(唯一调度)→ 渲染层(唯一硬件写入)**,
> 单进程多线程 + 1 个视觉子进程。本文是"地图";实现细节与踩坑见 `../CALIBRATION.md` §6-§13。

## 0. 一图流

```
                         ┌──────────────────── 传感器层(只感知,不动头)────────────────────┐
  Reachy 摄像头 ─ frame_pump_loop ─ mp.Queue ─►[vision_worker 子进程: Face每帧+Hand自适应]
                         │                                    │ result_q
                         ▼                                    ▼
                  st.latest_frame                      vision_result_loop ──► st.face_*/hand_*/finger_*
                  (take_snapshot 共享帧)                 (仅 TRACKING 积分跟脸;仅 PLAYING 积分跟手)
  XVF3800 DOA ── doa_sensor_loop(REST 10Hz)──────────────────────────────► st.sound_resid
  Reachy 麦克风 ─ main 主循环 16k chunk ──────────────► Qwen Realtime 上行(semantic_vad)
                                                              │
                         ┌────────── 对话/工具事件 ◄───────────┘
                         ▼
                   ChatCallback.on_event ──► play_q(音频)/ motion_q(手势)/ snap_q(看图)
                         │                                    ▲
                         ▼                                    │ 两段式指向升级(judge→point)
                  st.point_request ◄───────────────── snapshot_loop(共享帧→Qwen-VL→回话)
                         │
                         ▼
   behavior_loop(25Hz,状态机唯一调度)──► st.state / st.track_yaw / st.track_pitch / st.body_yaw_deg
                         │
                         ▼
   head_control_loop(25Hz,唯一 set_target 写入口)= track 目标 + 微动叠加 + 天线(逗它)
   motion_loop(手势 goto_target,action_active 期间上面这行整体让位)
   player_loop(play_q → 扬声器,带 ~300ms 抖动缓冲与代际作废)
```

## 1. 四个核心能力 → 代码边界

| 能力 | 传感/输入 | 决策 | 执行 | 配置块 |
|---|---|---|---|---|
| ① 对话(含打断/手势/看图) | main 主循环(麦克风上行)、ChatCallback.on_event(服务端事件) | Qwen Realtime(semantic_vad + function calling) | player_loop(说话)、motion_loop(8 手势)、snapshot_loop(看图) | `MODEL/VOICE/JITTER_*`、`TOOLS/INSTRUCTIONS` |
| ② 头部跟踪(人脸) | vision_worker 子进程(Face 每帧)→ vision_result_loop | behavior_loop:face_locked 迟滞进出 TRACKING | vision_result_loop **仅在 TRACKING 态**积分 st.track_yaw/pitch → head_control 渲染 | `VIS_*/TRACK_*/LOCK_*` |
| ③ 听声转向(DOA) | doa_sensor_loop(REST /api/state/doa,10Hz,中值滤波,机器人自声门控) | behavior_loop:IDLE/SEARCHING 收到视场外残差 → ENGAGING(转向+扫描找人) | behavior 的 approach() 写 track/body 目标 → head_control 渲染 | `DOA_*/SND_*/ENGAGE_*` |
| ④ 指向理解(两段式) | Hand 关键点(伸指=廉价提示)+ 语音工具调用 | snapshot_loop judge 轮:Qwen-VL 判 是否在指/目标是否可见/方向;不可见才升级 st.point_request | behavior 的 POINTING 子阶段 turn→settle→抓帧→hold→RETURNING | `POINT_*`、`SNAP_PROMPTS["judge"/"point"]`、`_DIR_MAP` |
| (附)逗它跟手 PLAY-01 | Hand 中心/大小/置信度(双门)+ 晃动量窗口 | behavior_loop:晃动大手持续 0.3s → PLAYING;手走 1.5s/静止 4s → 退出 | vision_result_loop **仅在 PLAYING 态**积分跟手(灵敏档+惯性外推);head_control 叠加天线开心 | `PLAY_*` |

**铁律(改代码前必读):**
- `head_control_loop` 是**唯一** `set_target` 写入口;手势(action_active)期间它整体让位给 motion_loop 的 goto_target。
- `behavior_loop` 是**唯一**的 st.state 写者;track_yaw/pitch 的写者按状态分工:TRACKING=视觉跟脸积分,PLAYING=视觉跟手积分,其余=behavior.approach()。杜绝双写。
- ⭐ head pose 是**世界系**(body_yaw 被 Stewart 补偿,CALIBRATION §11):大转向 head 给完整目标角,body_yaw 只是分担;手势 goto 必须传当前 body_yaw(传 0 会拽回身体)。
- ⭐ 视觉伺服增益必须时间常数型 `step=err×(1−exp(−dt/τ))`(CALIBRATION §9),禁止按帧固定比例。
- MediaPipe VIDEO 模式时间戳严格递增;Face/Hand 共用单调时钟(vision_worker)。
- 设备独占:全程只有 frame_pump 一个 get_frame 者、main 一个 get_audio_sample 者;take_snapshot 读共享帧。

## 2. 线程/进程清单

| 名字 | 周期 | 职责 | 读 | 写 |
|---|---|---|---|---|
| main 主循环 | 阻塞拉流 | 麦克风 16k chunk → Realtime 上行;到时退出 | mini.media | conv |
| ChatCallback.on_event | 事件驱动 | 服务端事件分发:打断/工具/音频/计时器喂养 | st | play_q/motion_q/snap_q、st.point_request(经 snap judge) |
| player_loop | 队列驱动 | 抖动缓冲 + 代际作废 + 推扬声器 | play_q | st.playback_end_estimate |
| motion_loop | 队列驱动 | Primary 手势串行 goto(以跟随姿态为基准) | motion_q | st.action_active |
| snapshot_loop | 队列驱动 | 共享帧→jpg→Qwen-VL;mode=scene/judge/point;judge 可升级 point_request | snap_q、st.latest_frame | st.snap_grabbed/snapshot_pending/point_request、conv |
| frame_pump_loop | ~40Hz | 唯一抓帧者;共享原帧;降采样喂子进程(drop-old) | mini.media | st.latest_frame、mp.Queue |
| **vision_worker(子进程)** | 每帧 | Face 每帧(pick_main_face 取最大脸)+ Hand 自适应提频 | mp.Queue | result_q |
| vision_result_loop | 队列驱动 | 发布 face/hand/finger;TRACKING 积分跟脸;PLAYING 积分跟手 | result_q、st.state | st.face_*/hand_*/finger_*、st.track_yaw/pitch |
| **KwsGate(在 main 主循环内)** | 每音频块 | 本地唤醒词检测(sherpa-onnx,armed/engaged 都喂) | 16k mono | (返回命中 bool) |
| doa_sensor_loop | 10Hz | DOA 中值滤波 + 自声门控 → 视场外残差 | REST、st.playback_end_estimate | st.sound_resid/sound_at |
| behavior_loop | 25Hz | 状态机唯一调度(见 §3) | st.* | st.state、track/body 目标、point_request 消费 |
| head_control_loop | 25Hz | 唯一 set_target:track 目标+微动+天线 | st.* | 硬件 |

队列:`play_q`(下行音频)、`motion_q`(手势)、`snap_q`(看图任务)、`vis_frame_q`(maxsize=1 背压)、`vis_result_q`。

## 3. 行为状态机(behavior_loop)

```
                 ┌─────────(晃动大手,任何非POINTING态)──────────┐
                 ▼                                              │
IDLE_CENTER ──locked──► TRACKING ◄──locked── SEARCHING      PLAYING
   │  ▲          ▲         │  │                 │  ▲        │ 手走1.5s/静止4s
   │  │          │     !locked 15s无互动        超时│ 声音      ▼
  声音 │       RETURNING ◄──┴──────────────────────┴──── RETURNING
   ▼  │          ▲
ENGAGING ────────┘ (超时/扫完无脸)
   └─locked → TRACKING

POINTING(最高优先,point_request 从任何态进入):turn→settle(0.6s)→抓帧→hold(等看图)→RETURNING
```

- 进出判定用 **face_locked 时间迟滞**(0.3s on / 1.5s off),不用瞬时检出(防空转)。
- 15s 无互动计时器只在**首次捕获**播种(SEARCHING↔TRACKING 回切不重置);说话(双向)与逗它进入都喂计时器。
- 手势(action_active)期间整个状态机暂停计时与驱动。

**WAKE-01(M1)新增 `ST_ARMED` 待命态(behavior_loop 仍是 st.state 唯一写者):**
```
ARMED(待命:慢呼吸,只听唤醒词,不连 Qwen)
  │  KWS 命中"小艺" → main 连接成功置 st.wake_ok
  ▼
ENGAGING(SEEK:从中心慢扫 yaw+pitch,DOA 符号定起扫边)──见脸──► TRACKING
  │  扫两侧 ~7s 无脸 → giveup 小动作                         (= 视觉裁判)
  ▼
ARMED
engaged 任意态 15s 无互动 → ARMED(关 WS,回零连接零计费;wake_mode 才走,--no-wake 仍回中)
```
- 唤醒响应序列:命中 → **heard 上扬(0 延迟)** → 连接 → SEEK 寻人 → 锁脸对话;连接失败 → **fail 下垂**;SEEK 没人 → **giveup 微沉** → 待命。三动作经 head_control 叠加渲染(单槽后到覆盖,additive 可被打断)。
- SEEK = 视觉主导寻人:**DOA 只取 resid 符号当起扫方向弱提示**,不信角度/不信 spread(spread 只测稳不测对,抓不到镜像错);见脸=锁定。详见 CALIBRATION §14。

## 4. 五层动作仲裁(优先级从高到低)

1. **Primary** 手势 goto / POINTING 转头(behavior 驱动)
2. **Playing** 逗它跟手(PLAYING 态,视觉积分)
3. **SoundTurn** 声源转向(ENGAGING,behavior 驱动;有脸在跟绝不抢)
4. **Tracking** 人脸跟随(TRACKING 态,视觉积分)
5. **Idle** 说话微动(仅 IDLE/TRACKING 叠加,跟随时缩 40%)

实现上 2/4 是"同一支笔两种墨水"(vision_result_loop 按 st.state 选积分目标),1/3 由 motion/behavior 驱动,5 在渲染层叠加——所以不存在抢写。

## 5. 两段式指向(POINT-02 v2,2026-06-07 定稿)

```
"这是什么" ─► 模型调 take_snapshot(1.2s内见过伸指→mode=judge)或 identify_pointed_object(恒 judge)
   judge 轮(原地拍):Qwen-VL 输出 JSON {pointing, target_visible, direction, desc}
     ├ 没在指            → desc 当普通看图回答(托下巴/误判不再转头)
     ├ 在指 + 目标可见    → desc 直接回答(不转头,最快路径)
     └ 在指 + 目标不可见  → st.point_request={call_id,gen,dir} → POINTING 转头(_DIR_MAP)
                            → settle → mode=point 第二轮看图 → 回答 → RETURNING
```
- 方向用 VLM 的粗方向(8 向);本地食指延长线 2D 角度只做"要不要走 judge"的提示(其噪声曾致错误抬头)。
- snapshot_pending 在工具调用时 +1,judge 升级时**不**减(continue),由 point 轮收尾减——保证 response.done 不抢跑。

## 6. WAKE-01 唤醒词 + 待命门控(M1,已实现;标定/踩坑见 CALIBRATION §14)

- **唤醒词 = 单字「小艺」**(yī/yìn/yì 三声调形态,sherpa-onnx KWS int8,`--single-thr 0.17`,命中即唤醒)。叠词/双单确认真机实测否决。`KwsGate` 类封装,模型在 `tools/_kws_models/`(gitignore)。
- **上行两路**:main 主循环同一份 16k mono **始终喂 KWS**(本地、与网络无关);**engaged 才** `append_audio` 给 Qwen,armed `continue` 绝不发上行。
- **armed/engaged 门控(命中才连 b)**:armed 不建 WS(零连接零计费、电视声无从触发);KWS 命中 → main `open_session()`(3s 超时 worker 兜底)成功置 `st.wake_ok` → behavior 离开 `ST_ARMED`;engaged 15s 无互动 → `ST_ARMED` 关 WS。`st.state` 仍由 behavior_loop 唯一写;main 经 `st.wake_ok` 握手、绝不写 st.state。`--no-wake` 回退旧"启动即连"。
- **唤醒响应序列**:命中 → heard 上扬(0 延迟)→ 连接 → SEEK 寻人 → 锁脸;失败 → fail 下垂 + 留 armed 可重喊;SEEK 没人 → giveup 微沉 → 待命。
- **SEEK 转向(视觉主导)**:DOA 只取 resid 符号当起扫方向弱提示;从中心慢扫 yaw+pitch,见脸即锁(视觉裁判),镜像错靠"扫两侧+见脸"兜回。**不信 DOA 角度/spread**(详见 §3 + CALIBRATION §14 的两个错误前提推翻)。
- **DOA 可用区/盲区**:前+左右侧+侧后 ~±150° 方位 DOA 大致可用;深后 ~150-210° 是**双重盲区**(唤醒 ~3/10 + 头转不过 ±90°),软件不可救,人需绕到 ±90° 内。

## 7. 已知边界 / 记账

- 背景人声 vs 真人对话:**已由 WAKE-01 待命门控根治**(armed 不连 Qwen,电视声不喂 semantic_vad);engaged 内的电视声区分留 M1.5 方向门控。误触地板 ~0.7 次/5min(电视同音,阈值救不了)。
- 深后方双重盲区(唤醒 ~3/10 + ±90° 物理限);多脸=每帧取最大(无 sticky 迟滞,留 M1.5)。
- MediaPipe 侧脸/移动召回 30-45%(迟滞吸收;备胎:1e InsightFace on RTX5060,已装未集成)。
- 快手跟踪物理上限 ~90°/s;头部跟随范围身体±22.5°(逗它)/±23°(跟脸)。
- 不出声指远处且手够近时,可能先被当逗它跟手(指向靠语音触发,实际影响小)。
