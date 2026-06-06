# Reachy Mini Lite — 标定与硬件 I/O 特性记录

> 设备:Reachy Mini **Lite**(USB 版,VID 38FB Pollen),电机走 COM3。
> daemon:`reachy-mini-daemon.exe`(venv Scripts 下),localhost:8000。
> 本文件记录经**实测验证**的结论(2026-06-03 五项硬件体检 + 2026-06-05 Qwen-Omni-Realtime 语音对话对接),供后续开发(Edge Runtime / 工具调用 / 唤醒词等)直接引用。
> 体检脚本见 `healthcheck/`(各脚本作用见该目录 README)。

---

## 0. 通用:连接与媒体 backend

```python
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"   # 必须在 import reachy_mini 之前!
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
from reachy_mini import ReachyMini
```

- **代理坑:** 本机有 `HTTP(S)_PROXY=127.0.0.1:7897`、`NO_PROXY` 为空,Python websockets 会把 `ws://localhost` 也走代理 → 连接 EOF 失败。**务必在 import 前**把 localhost 加进 `NO_PROXY`(脚本级修,别动全局)。
- **连接模式:** USB Lite,daemon 在本机 → `ReachyMini(connection_mode="localhost_only")`。
- **媒体 backend(`media_backend=`)选用:**
  - `"no_media"` — 只测/用控制平面(连接、关节、pose、动作)。**不碰摄像头/麦克风**,启动快;会让 daemon 释放媒体硬件,退出(用 `with` 上下文管理器)时自动归还。**动作类脚本用这个。**
  - `"default"` — LOCAL backend,经 daemon 本地 IPC 打开摄像头 + GStreamer 音频。**摄像头/麦克风/扬声器类脚本用这个。**
- **务必用 `with ReachyMini(...) as mini:`**,退出时自动释放媒体资源,避免占着设备影响下一个进程。
- 版本:SDK 与 daemon 均 1.7.3,一致。
- 装包统一走清华镜像:`pip install <pkg> -i https://pypi.tuna.tsinghua.edu.cn/simple`。

---

## 1. 连接体检 ✅

- `mini.connection_mode` → `localhost_only`。
- `mini.client.get_status()` → `DaemonStatus`:`state=running`、`wireless_version=False`(Lite)、`camera_specs_name="lite"`。
- **IMU:** `mini.imu` 返回 `None` —— **Lite 无 IMU**,按预期跳过。
- **电池:** Lite USB 供电,**SDK 无电池传感器接口**,跳过。

---

## 2. 动作体检 ✅(方向约定已实物验证)

构造头部 4×4 位姿:`R.from_euler("xyz", [roll, pitch, yaw], degrees=True)`,坐标系 **X 朝前 / Y 朝左 / Z 朝上**。8 个方向已肉眼验证全部正确:

| 轴 | 正方向 | 负方向 |
|---|---|---|
| **yaw**(绕 Z) | + = 看左 | − = 看右 |
| **pitch**(绕 Y) | + = 看下(低头) | − = 看上(抬头) |

- 点头 = pitch 上下摆;摇头 = yaw 左右摆。
- **天线** 关节顺序 `[right, left]`(rad),中立位 `[-0.1745, 0.1745]`;摆动 ±0.5 rad 平滑无异响。
- **body 转动:** `goto_target(head=INIT, body_yaw=弧度)`,`body_yaw +` = 身体向左转。
- 头部动作**不带动身体**时,连接设 `automatic_body_yaw=False`(头纯靠 Stewart 平台)。
- **实测安全平滑幅度:** 头部 ±12°、body ±15°、天线 ±0.5 rad,全程 `goto_target` min-jerk 插值无跳变。

### 9 电机 ID 映射(Dynamixel,ID 10–18)
| ID | 名称 | 作用 |
|---|---|---|
| 10 | `body_rotation` | 身体 yaw |
| 11–16 | `stewart_1` … `stewart_6` | 头部 6 自由度 Stewart 平台 |
| 17 | `right_antenna` | 右天线 |
| 18 | `left_antenna` | 左天线 |

`get_current_joint_positions()` 返回 `(head[7], antennas[2])`:head = [yaw, stewart×6],antennas = [right, left]。`enable_motors/disable_motors(ids=[...])` 用上面的名称。

---

## 3. 摄像头体检 ✅

- 抓帧:**只用 `mini.media.get_frame()`**,**绝不另开 `cv2.VideoCapture`**(会和 SDK 抢设备冲突)。
- 返回:**BGR `numpy.uint8`,shape `(1080, 1920, 3)`** → 分辨率 **1920×1080**。
- **实测帧率 FPS ≈ 49**(60 帧 / 1.20s)。`get_frame()` 内部 `try_pull_sample` 20ms 超时、appsink `max-buffers=1 drop=True` 只保留最新帧;紧循环偶尔返回 `None`(20ms 内无新帧),需重试,不计入帧。
- 存 jpg:SDK 没装 cv2,用 **Pillow**;`get_frame()` 是 BGR,存前要 `frame[:, :, ::-1]` 转 RGB(否则颜色反)。
- **⚠ 喂 MediaPipe / 推理前务必降采样:** 1080p@49fps 直接喂人脸/姿态检测太重。先 resize 到 ~640×480(或更小)再推理,省算力、提帧率。

---

## 4. 麦克风体检 ✅

录音用 SDK:`media.start_recording()` → `media.get_audio_sample()` 循环 → `media.stop_recording()`,走 GStreamer **自动选中 "Reachy Mini Audio" 卡**,**不要另开 sounddevice**。

- 格式:**16000 Hz / 2 声道 / float32**,`get_audio_sample()` 返回 `(N, 2)`。
- 正常说话 RMS ≈ 0.02–0.06,峰值 ~0.76 不削顶;静音底噪 RMS ≈ 0.0008(非零);静音阈值取 0.002。

### A-01 双声道实为单声道复制 ⭐
- 左右两声道**逐样本完全相同**(相关系数 1.000000,差值恒 0)。
- 这是 ReSpeaker **波束成形/降噪后的处理单声道**复制到双声道,不是两路独立原始麦克风。
- **应用:** 喂 OpenAI Realtime 时**只取单声道** `audio[:, 0]` 即可,省一半带宽,信息无损。

### A-02 录音管线 1–2 秒启动延迟 ⭐
- `start_recording()` 后 GStreamer 管线约 **1–2 秒**才稳定出数据;这期间说话会被吞(体检"前面没录到"即此因)。
- **应用(Edge Runtime):** 音频模块要**提前启动录音管线并保持常开(always-on)**,不能即用即开,否则每次唤醒丢开头 1–2 秒。
- 管线稳定后是连续实时交付的,健康。

---

## 5. 扬声器体检 ✅

- 播放用 **`mini.media.play_sound(wav绝对路径)`**(**非阻塞**,返回即开始播,需 `time.sleep(wav时长)` 等播完再退出 `with`,否则被切断)。
- **输出声卡(实测证据):** Windows 上 `play_sound` 经 `get_audio_device("Sink")` 按名字匹配选中 **"Reachy Mini Audio"** 卡(显示名 `回音消除话筒 (Reachy Mini Audio)`),**不是电脑 Realtek 扬声器** → 声音从**机器人自带扬声器**出。已耳朵确认。
- 440Hz 正弦测试音 + 中文 TTS 均清晰出声。
- **中文 TTS:** 用 `pyttsx3`,系统有 **Microsoft Huihui Desktop(zh-CN)** 中文语音;选 voice 时匹配 `languages` 含 `zh` 或 id 含 `ZH-CN`。`save_to_file()` 生成 wav,再用 `play_sound()` 播 → 保证从机器人扬声器出(直接 `runAndWait` 播放会走系统默认设备,不一定是机器人)。
- `gst_monitor_devices("Audio/Sink")` 等 DeviceMonitor 调用**必须在 `ReachyMini` 构造之后**(那时才 `Gst.init()` 过),否则报 "Please call Gst.init(argv)"。

---

## 6. Qwen3.5-Omni-Realtime 语音对话对接 ✅(2026-06-05,D-01)

全双工语音对话已跑通(说话→机器人语音回应→可随时插话打断)。实现:`voice/d01_realtime_chat.py`;编排测试版 `voice/_d01a_orchestrated.py` / `_d01b_orchestrated.py` 留作回归。

### 连接
- 模型 **`qwen3.5-omni-plus-realtime`**,北京端点 `wss://dashscope.aliyuncs.com/api-ws/v1/realtime`(SDK 默认),鉴权 `DASHSCOPE_API_KEY`。
- 用 **dashscope Python SDK**(装的 1.25.21,要求 ≥1.23.9):`dashscope.audio.qwen_omni` 的 `OmniRealtimeConversation` + `OmniRealtimeCallback`。注意 `on_event(event)` 实际收到的是**已解析的 dict**(类型注解写 str,别信)。
- **代理坑扩展:** `NO_PROXY` 除 localhost 外还要加 **`.aliyuncs.com`**(dashscope 走 websocket-client,会读代理环境变量;大陆直连,别让 7897 劫持 wss)。
- VAD 用 **`semantic_vad`**(官方对 qwen3.5 系列的推荐),说话起止/回复触发全部服务端自动;单连接最长 120 分钟。

### 音频管线(实测参数)
- **上行零重采样:** Realtime 要求 16kHz/PCM16/单声道,与 Reachy 麦克风原生格式完全一致 → `get_audio_sample()[:,0]`(A-01)→ `int16` → base64 → `append_audio()`。录音管线常开(A-02)。
- **下行必须重采样:** Realtime 输出 **24kHz** PCM16,而播放管线 appsrc caps 固定 **16kHz**(`audio_gstreamer.py`)→ `scipy.signal.resample_poly(f32, 16000, 24000)` 后 `push_audio_sample()` 流播。
  (若改走"存 wav + `play_sound()`"则任意采样率都行,decodebin 自己转。)
- **抖动缓冲:** 每段回复先攒 **~0.32s**(300ms 目标 + 0.5s 兜底超时)再开播,句中无停顿;收包回调只解码入队,播放由独立线程消费,不阻塞 WebSocket。
- **实测延迟:** 首音频延迟稳定 **~370ms**;回声消除有效——机器人自己说话不会触发 VAD(无自问自答)。

### 三层打断(barge-in,实测打断到静音 ~20ms)
播放中或生成中收到 `input_audio_buffer.speech_started` →
1. **代际计数** `play_gen += 1`:播放队列里旧代际块全部作废;
2. **`mini.media.audio.clear_player()`** ⭐:SDK 原生接口(media_manager **没**透出,要从 `.audio` 后端拿),flush appsrc,已推未播的残余瞬间清空——实测一次清掉 7.1s 残余;
3. 回复仍在生成则 **`cancel_response()`**(此分支因模型生成快于播放,极少触发,尚未实测踩到)。
另:打断后到下一个 `response.created` 之间的 `response.audio.delta` 全部丢弃(在途旧块)。

### 已知待办
- 麦克风常开 → 环境闲聊也会被当输入接话,需要唤醒词/按键门控(后续阶段)。

---

## 7. 动作工具(function calling)✅(2026-06-05,O-01a-1)

模型自主调用动作做身体语言,**边说边动**已实测(语音播放与动作执行并发,互不阻塞)。实现合并在 `voice/d01_realtime_chat.py`;编排测试 `voice/_o01a1_orchestrated.py`;诊断工具 `voice/_diag_o01.py`。

### 协议(实测验证)
- **声明:** 扁平格式 `{"type":"function","name":...,"description":...,"parameters":{...}}`(不是 chat.completions 的嵌套),经 SDK `update_session(..., tools=TOOLS)` 传入(走 kwargs 进 session 配置)。
- **调用事件:** `response.function_call_arguments.done`,字段 `name` / `arguments`(JSON 字符串)/ `call_id`。模型**一个响应可连发多个调用**(实测"你好"一次发了 nod + wiggle_antennas)。
- **回结果:** `create_item({"type":"function_call_output","call_id":...,"output":...})`。
- **协调设计(O-01a 修复2 定稿:说话动作同时出发,实测 4/4 动作均在开口后起手):**
  1. instructions 明确"做动作时必须同时用语音回应,边说边做"——**实测这一句就把模型完全拉到音频+动作同响应路径**(修复后补话兜底 0 次触发);
  2. `function_call_arguments.done` 一到**立即**回 output(动作已派发,乐观上报 success),不等动作做完;
  3. `response.done` 时若该响应纯动作(audio.delta 计数=0 且未被打断)→ **马上**补 `response.create`(兜底);事件同在 ws 线程,output 必先于 response.create,协议安全;
  4. 带音频的响应不补(避免双重说话)。
- 模型可能把音频和工具调用**拆成连续两个响应**下发(一轮两个 response.created),上述逻辑天然兼容。

### 并发与线程模型
- 动作任务入队,**独立动作线程串行执行**(`goto_target` 是阻塞插值,绝不能在音频回调/播放线程里调)。
- 8 个工具:nod / shake_head / look_left|right|up|down / wiggle_antennas / tilt_head(`automatic_body_yaw=False` + 全程 `body_yaw=0`),单个动作 1.6~2.0s。
- **手势幅度(2026-06-05 加大并实测安全,16 次零 IK/限位异常、无异响):** 点头 pitch +15/−10°、摇头 yaw ±15°、看向 ±16°、歪头 roll 15°、天线 ±0.8rad。§2 的 ±12° 是体检时的保守值,手势用这套更大的;验证脚本 `voice/_motion_amp_test.py`。
- **barge-in 时动作不中断**(动作短,让它做完),只停音频。

### 踩坑记录
- **semantic_vad 有音量门槛:** 上行 RMS ≈0.004(说话太小声/离远)**完全不触发** speech_started——表现为"上行了 75s 音频但服务端零事件",像断连但其实是没过门槛。正常说话 RMS ≥0.01 即稳定触发。排查手段:上行循环里按周期打印 RMS(已内置在正式脚本,<0.005 时提示)。
- `update_session` 不传 `tools` 字段时**不会清除**已注册的工具(继承上次配置);要清除需显式传空。
- **⚠ daemon 媒体重取崩溃(7×24 长跑前必须解决):** daemon 连续运行 ~45 分钟、经历多轮媒体 acquire/release 后,`no_media` 客户端退出触发 `Re-acquiring media hardware...` 时 daemon 进程崩溃(exit 116),客户端侧表现为 `/api/media/acquire` ConnectionReset。短期对策:长会话前重启 daemon;根因在 daemon 侧,未修。
- **"No motors detected" ≠ 偶发通信错,先查电源(2026-06-06 实证):** 连续两次启动报 "No motors detected. Check if the power supply is connected and turned on!"(COM3 找得到、9 电机全失联)——实为**机器人电机电源没通**(USB 在 ≠ 电机有电,两路供电独立)。通电后一次启动成功。之前"首启失败重试即好"的经验仅适用于单电机零星通信错;**全员失联直接查电源/线**。
- **左天线 Overload Error 会锁存:** Dynamixel 过载保护触发后持续刷 `Motor 'left_antenna' hardware errors: ['Overload Error']`,重启 daemon 无效——**必须给机器人断电**(检查天线无卡滞、扶正)再上电才清除。实测断电清错后零复发。

---

## 8. take_snapshot 看图(V-01-1)✅(2026-06-05)

机器人"看一眼"能力:Realtime 工具调用 → 抓当前帧 → chat.completions 回合制看图 → 语音转述。实现合并在 `voice/d01_realtime_chat.py`;编排测试 `voice/_v01_orchestrated.py`。

### 链路与实测数据
- **抓帧:** `media.get_frame()` 连抓 3 帧取最新(appsink drop=True 只留最新,多抓防旧帧/黑帧),实测 **47ms**;两次快照画面随用户姿势变化,确证是当前帧。
- **压缩:** 1080p BGR → Pillow `[:, :, ::-1]` 转 RGB → 640×360 jpg q85 ≈ **29KB**(Qwen 建议单张 ≤256KB,余量大)。
- **看图:** chat.completions(`qwen3.5-omni-plus`,北京 compatible-mode),`image_url` 传 `data:image/jpeg;base64,...`,**必须 `stream=True`** + `extra_body={"modalities":["text"]}`(只要文本,省音频生成)。往返 **~3s**,工具调用→描述就绪全程 **~3.1–3.3s**。
- **并存性:** 抓帧/看图期间录音上行与播放零中断(摄像头与音频是独立 GStreamer 管线),等待期插话打断也正常。

### 与手势工具的协调差异(关键设计)
- 手势 = **乐观即时回 output**(不等动作做完);
- take_snapshot = **必须等描述就绪才回 output**(模型要拿着内容才能转述),回完即 `create_response()` 让模型开口;
- `response.done` 的"纯动作立即补话"逻辑在 **快照挂起时跳过**(`snapshot_pending` 计数),否则模型会在没拿到描述时抢跑;
- 期间被打断(代际变化)→ 仍回 output(协议要求)但不补话。

### 体验设计
- ~3s 等待由模型的**口头过渡**填充(调用工具的同响应里自带"让我看看…"音频),不冷场——instructions 引导即可,无需代码。
- 快照存 `voice/output/`(**已 gitignore,可能含人像,严禁推送**)。

---

## 9. 本地视觉:MediaPipe 看脸 + 转头跟随(VIS-01)✅(2026-06-05)

第一次引入本地视觉栈(与音频/Qwen 独立)。实现 `vision/vis01_face_track.py`(独立脚本;**2026-06-06 已经 F-01 融合进正式对话脚本,见 §10**);诊断工具 `vision/_diag_face.py` / `_diag_face2.py`。

### 环境与性能(本机实测)
- **MediaPipe 0.10.35**(清华镜像装,Python 3.12 venv 直接可用;依赖带入 opencv-contrib 与 sounddevice——**照旧禁止开 `cv2.VideoCapture`/sounddevice 流**,只用其库代码)。
- 模型:Face Landmarker `face_landmarker.task`(float16,3.7MB),放 `vision/models/`(gitignore)。下载:`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`(实测大陆直连可下)。
- **性能(640×360 输入,CPU/XNNPACK):端到端 41 FPS,单帧推理均值 12ms / P95 16ms**——这台机器跑实时视觉余量充足,后续可叠加更多模型。
- 降采样:1080p 帧 `frame[::3, ::3, ::-1]` 整数抽样 + BGR→RGB,开销近零,优于 resize。

### 跟随控制(闭环:摄像头在头上,转头即收敛)
- 人脸中心 (u,v)(取最大脸 bbox 中心)→ One Euro 滤波(min_cutoff 0.8, beta 0.08)→ 角度误差 = (u−0.5)×FOV(FOV 估 65°×40°)→ **时间常数型 P 控制** → `set_target` 连续转头。
- **⭐ 核心教训(首轮"一直点头"):** 按"每帧吃固定比例误差"设增益会随帧率缩放——47fps 下等效 ~190°/s,叠加摄像头管线 ~100ms 延迟 → pitch 在 ±15° 限位间持续打摆。**修复:`step = err × (1 − exp(−dt/TAU))`,TAU=0.4s,与帧率解耦**;另 MAX_STEP 1.5°/帧、死区 2°。修后用户确认平滑无抖、方向正确。
- 方向映射(摄像头不镜像):画面右(u>0.5)= 机器人右边 → yaw 取负;画面下 → pitch 取正。跟随限幅 yaw ±25° / pitch ±15°。
- 丢脸策略:保持 1.5s → 每帧 ×0.97 缓慢回中 → 重见即重新锁定;One Euro 丢脸时 reset(防跳变)。

### 踩坑记录
- **MediaPipe VIDEO 模式时间戳必须严格递增**:同一毫秒两帧会抛 "Input timestamp must be monotonically increasing";用 `ts = max(prev+1, 真实ms)` 防撞。
- **零检出先怀疑画面再怀疑代码**:连续两轮 0% 检出,最后发现是机器人对着衣柜/人在画面边缘躺着侧脸。排查路径:存帧人工看 → 标准人脸图(mediapipe-assets portrait.jpg)验证安装 → 全尺寸帧重测。`vision/_diag_face2.py` 即此流程,可复用。

---

## 10. F-01 融合:人脸跟随并进对话完整体(三层动作仲裁)✅(2026-06-06)

VIS-01 跟随与对话栈合体,`voice/d01_realtime_chat.py` 现为最终完整体:对话+打断+动作+微动+看图+**人脸跟随**。用户验收:"跟随平滑、让位恢复无突跳、整体像一个看着我聊天的伙伴"。

### 三层仲裁(优先级从高到低)与结构保证
```
视觉线程(MediaPipe ~19fps)──积分──▶ track_yaw/pitch(跟随目标角)
                                         │
头部控制线程(25Hz set_target)◀──读──────┘   ←── 头部唯一 set_target 写入口
  头部姿态 = 跟随目标 + 微动叠加;action_active 时完全停发(硬让位)
                                         ▲
动作线程(goto_target 手势)──独占时置 action_active
```
- **Primary 手势:** 进场读当前跟随姿态为**基准**,手势相对基准执行(点头=在看着人的朝向上点头,不甩回正中),做完回基准 → 头控线程接管时姿态一致,**零突跳**。手势 offset 叠加基准后裁剪进安全箱(yaw ±25° / pitch ±16°,均在已验证范围)。look_* 是绝对方向指令(本来就要离开人脸),看完回基准。
- **Tracking:** 沿用 §9 全套验证参数(τ=0.4s 时间常数增益、死区 2°、单帧限步 1.5°、One Euro 0.8/0.08);视觉线程**只积分目标角不动头**;手势期间冻结积分,丢脸回中也改成时间常数型(τ=0.8s,等效原 ×0.97/帧@41fps,彻底去帧率化)。
- **Idle 微动:** 跟随中缩到 **40% 幅度**小幅叠加(不打架但保留"活着感");无人脸时平滑恢复原幅度(行为退化为 O-01a-2);幅度切换走包络,无跳变。

### get_frame 协调(单一抓帧者)
- 视觉线程是**唯一持续 get_frame 者**,最新帧(引用)共享在 State;take_snapshot 直接读共享帧(≤25ms 新,优于原"连抓 3 帧",实测当秒完成),视觉线程没帧时才退回直接抓。

### 融合后实测(2026-06-06,120s + 180s 两轮)
- **检测帧率 ~19fps**(独立跑 41fps,限频 30 之下仍被音频线程抢 GIL 压到 19)——**τ 型控制与帧率解耦,19fps 跟随依旧平滑**,这正是 §9 铁律的回报;推理均值:有脸 12~15ms,无脸 2~4ms(landmark 图早退)。
- **对话实时性零退化:** 首音频延迟 35~376ms(与融合前同档),打断 7/7 干净,手势 5 次全部"基准上做、做完恢复跟随"。
- 跟随目标会顶到限位(pitch +15°/yaw −25°):人贴近、坐姿低于摄像头时脸超出头部可达角,物理极限非 bug。
- MediaPipe 遥测上传失败的 E0000 clearcut 日志无害,可忽略。
- 正式脚本支持可选时长参数(`python voice\d01_realtime_chat.py 120` 到时干净退出),编排回归用。

---

## 体检汇总(2026-06-03)

| 项 | 通道 | 结果 | 关键数据 |
|---|---|---|---|
| 1 | 连接 | ✅ | localhost_only,1.7.3,9 电机,IMU/电池 Lite 无 |
| 2 | 动作 | ✅ | 8 方向全对,无异响,±12°/±15° 平滑 |
| 3 | 摄像头 | ✅ | 1920×1080 BGR,FPS≈49 |
| 4 | 麦克风 | ✅ | 16kHz 双声道(实为单声道复制),RMS 正常 |
| 5 | 扬声器 | ✅ | 从 Reachy 声卡出声,440Hz+中文 TTS 清晰 |

## 阶段进展

| 阶段 | 日期 | 结果 |
|---|---|---|
| 五项硬件 I/O 体检 | 2026-06-03 | ✅ 全过(上表) |
| D-01 Realtime 语音对话(含打断) | 2026-06-05 | ✅ 首音频 ~370ms,打断 ~20ms,见 §6 |
| O-01a-1 动作工具(function calling) | 2026-06-05 | ✅ 8 工具,边说边动并发实测,见 §7 |
| O-01a-2 idle 微动 + 幅度加大 + 同时出发 | 2026-06-05 | ✅ 微动让位 30ms,4/4 语音动作重叠,见 §7 |
| V-01-1 take_snapshot 看图 | 2026-06-05 | ✅ 抓帧 47ms,看图 ~3s,口头过渡,见 §8 |
| VIS-01 MediaPipe 看脸+转头跟随 | 2026-06-05 | ✅ 41FPS/12ms,时间常数控制平滑跟随,见 §9 |
| F-01 融合:跟随并进对话(三层仲裁) | 2026-06-06 | ✅ 单一头控写入口,手势基准化零突跳,19fps 够用,见 §10 |
