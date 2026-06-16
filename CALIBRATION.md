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
- **daemon 媒体重取崩溃(exit 116):已于 2026-06-06 定位并缓解,见 §12**(根因:no_media 客户端强制触发媒体释放/重建循环,原生层概率性崩;缓解:release_media 补丁 + daemon_up.py + 长测前 restart)。
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

> **注:§10 是 F-01 三层仲裁的初版结论(线程内视觉、~19fps)。该架构已于 2026-06-06 被
> §13(FUSION-03)取代:视觉挪独立进程(27fps)、四层仲裁、行为状态机、指向转头。
> §10 保留作演进记录,当前实现以 §13 为准。**

## 10. F-01 融合:人脸跟随并进对话完整体(三层动作仲裁)✅(2026-06-06,已被 §13 取代)

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

## 11. 听声辨位(DOA)+ 听声转头 ✅(2026-06-06,DOA-01 / SOUND-TURN-01)

**核心结论:Lite 听声辨位直接可用,不用自己写算法。** 体检发现的"双声道是单声道复制"不影响 DOA——那是波束成形后的 USB 输出;DOA 在 **ReSpeaker XVF3800**(定制 4 麦**线性**阵列,装在**头部**,mic0 靠右天线/mic3 靠左天线)芯片内部用 4 路原始麦计算,板载输出。demo `audio/sound_turn.py`;采样/诊断工具 `audio/doa01_test.py`、`_doa_sample.py`、`_doa_diag.py`、`_doa_frame_test.py`、`_body_yaw_test.py`。

### 获取通道(三层,实测)
- **daemon REST(推荐,唯一可并行通道):** `GET /api/state/doa` → `{"angle": rad, "speech_detected": bool}`。与对话/视觉/动作完全并存(纯 HTTP,不碰媒体管线),10Hz 轮询无压力。
- SDK 客户端 `mini.media.get_DoA()` / 直连 USB(`audio_control_utils.py`,VID 0x38FB PID 0x1001,libusb):**daemon 运行时独占 USB 控制接口,第二进程必被拒**(Access denied)——所以一律走 REST。
- 原始双路麦:`AUDIO_MGR_OP_L/R` 可把 4 路原始麦任选 2 路路由到 USB 立体声(Seeed 官方文档),但会破坏对话用的 AEC 输出且需独占 USB——没必要,有板载 DOA。

### 角度约定与实测精度(2026-06-06,四方位站定验证)
- **0=阵列左,π/2(90°)=正前/正后,π(180°)=右**;⭐ 线性阵列**前后不分**(只有 180° 半平面),正后方说话读数 ≈ 正前(实测 BACK 中位 91°)。
- 实测:左 54° / 前 94° / 右 163° / 后 91°——左/前/右三向区分清晰(`<70 左,≈90 前后,>120 右`),1° 步进连续。
- 静止噪声 std ~10°,**~15% 样本是反射离群点**(双峰,可拖偏 40°+)→ 必须中值滤波。
- **VAD(speech_detected)有效距离 ≤1m**,连续说话触发率仅 **11~57%**(远了完全不触发);窗口设计要按这个触发率算(1.5s 窗口 ≥5 样本 ≈ 33% 率)。

### ⭐⭐ 两个参考系教训(SOUND-TURN 三轮排障的根因,跨任务通用)
1. **DOA 是相对阵列(头)的偏角,不是世界角**:阵列在头上,头转了读数跟着变。目标世界朝向 = 当前头世界朝向 + (90° − DOA)。把 DOA 当世界角 → 转完读数变 → 来回扭振荡。
2. **`goto_target`/`set_target` 的 head pose 是【世界系(底座系)】姿态**:身体(body_yaw)转动会被 Stewart 平台**反向补偿**,头的世界朝向保持不变(实测:身体 +30° 时 DOA 偏移仅 +1°,头 +20° 时偏移 +21°)。**要让头真转过去,head pose 必须直接给完整目标角;body_yaw 只是"分担量"**(保证颈相对量 = 头目标 − 身体 ≤25° 不顶 Stewart 限位)。给"头 25° + 身体余量"的写法 → 身体白转、头永远最多 25° = "转不到位"。

### 听声转头闭环(根因诊断驱动的设计)
- **开环单转必残留误差**(诊断:转一次后残差中位 13°,无核对则永久偏)→ **闭环逐步逼近**:首转(窗口中值,阈值 12°)→ 说话持续期间不断核对残差,≥8° 就补转 → 实测 0~1 次微调即锁定(典型:−53° 大转 + −9° 微调)。
- 中值窗口 1.5s / ≥5 有声样本(压反射双峰 + 匹配 VAD 触发率);转完清窗(旧角度在旧参考系);冷却 1.2s。
- 头身分担:颈 ≤25°,身体 ≤65°,合计 ±90°;时长 0.4s + 0.008s/°。
- 用户验收:"左右都能对准我,收敛也稳"。
- 融合候选(未做):声音触发 → DOA 粗转向 → 人脸跟随接管细定位(互补:DOA 管视场外,视觉管视场内)。

---

## 12. daemon exit 116 崩溃:根因与缓解 ✅(2026-06-06,DAEMON-FIX)

**判定:daemon 原生层(GStreamer/Rust)概率性 bug,源码不可改 → 缓解为主,但触发面可砍到≈0。** 工具 `tools/daemon_up.py`、`tools/_daemon_crash_repro.py`、`tools/_no_release_test.py`。

### 根因(代码级定位)
- **触发路径:** `reachy_mini.py:288-296`——**每个 `media_backend="no_media"` 客户端 connect 时,SDK 主动调 daemon `release_media`**(GStreamer 管线→NULL);`:181-182`——exit 时再调 `acquire_media` → `GstMediaServer.start()` **从零重建**整条管线(相机 + WASAPI + webrtcsink + 内嵌 Rust WebRTC 信令)。每个 motion-only 脚本跑一次 = 一次完整重建循环。
- **崩溃是概率性的,非确定性计数:** 陈年 daemon(2h)上 30 连发 no_media 循环未崩(共 38 循环全部执行);但日志抓到 webrtcsink 重建期 GStreamer 流错误(`net\webrtc\src\webrtcsink\imp.rs`,元素已编号到 `webrtcsink24`——每循环新建一个)。重建路径确实脆弱,崩在哪次看运气(历史:~45min + 多循环后崩)。
- **第二种死法(独立模式):** 强杀旧实例后**立刻**重启,新 daemon "started successfully" 后 ~1 分钟**无客户端静默死**(exit 116,日志无错误尾行)——疑似 WASAPI 设备未释放完的竞态。
- `default`→LOCAL backend(语音完整体用的)**不触发**释放/重取循环,只有 no_media 分支触发。

### 缓解三件套(全部实测)
1. **触发面归零补丁(motion-only 脚本一行):**
   ```python
   from reachy_mini import ReachyMini
   ReachyMini.release_media = lambda self: None  # no_media 时不让 daemon 释放/重建媒体
   ```
   验证:补丁后 connect+动作+exit,daemon 释放计数 38→38 零新增,控制通路正常。
2. **可靠启动器 `tools/daemon_up.py`:** `python tools\daemon_up.py [--restart]`。探活(REST /api/state/full)→ 清残留(强杀后等 4s 让 WASAPI/COM3 释放)→ 启动 → 就绪判定(日志 "started successfully" + REST 探活 + 3s 过载观察)→ **分诊退出码:0=就绪,2=电源没通(9 电机全失联,重试无用,人工查电源),3=过载锁存(断电清),1=其他**;零星通信错自动重试一次。两路径(--restart / 在线 no-op)已实测;daemon 独立于封装进程存活;日志存 `tools/daemon_logs/`(gitignore)。
3. **卫生规则:** 长会话/长测前 `daemon_up.py --restart` 刷新实例;永远不要强杀后立刻裸启(竞态死法②)。

### 上报
- SDK 设计层可改进:no_media 客户端不应强制释放媒体管线;重建路径缺错误恢复。渠道:pollen-robotics/reachy_mini GitHub issues(草稿已起,用户决定是否提交)。

---

## 13. FUSION-03 完整体:四层仲裁 + 视觉进程化 + 行为状态机 + 指向转头 ✅(2026-06-06)

`voice/d01_realtime_chat.py` 的当前架构(取代 §10 的初版)。一个脚本、子进程 + 多线程,集对话/打断/看图/手势/微动/人脸跟随/听声转头/指向理解于一体。子进程 `voice/vision_worker.py`(Face+Hand)。

### 视觉进程化(TRACK-FIX:multiprocessing 逃 GIL)
- **根因:** Windows 上 MediaPipe 只能 CPU;六线程融合后视觉循环被 GIL 饿到 **41→19fps**(推理本身仍 12ms,是被饿不是变慢)。
- **解法:** MediaPipe 检测移入独立子进程(独立 GIL,真并行)。主进程 `frame_pump_loop` 只抓帧+降采样(numpy 抽样 ~1ms)经 `mp.Queue(maxsize=1)` 背压喂子进程(满则丢旧换新,只检测最新帧);`vision_result_loop` 消费结果跑时间常数积分。
- **实测:19→27fps**(用户确认跟踪稳)。检出率:**正脸静止 ~100%,移动/侧脸 30~45%**(MediaPipe 在 640×360 对非正脸召回弱)——靠丢脸缓冲 + 迟滞吸收,肉眼跟踪稳定。
- **丢脸缓冲:** 连续 `VIS_MISS_N=5` 帧漏检才重置 One Euro(防侧脸单帧闪断丢平滑历史)。
- 子进程 Face 每帧(跟随要实时)、Hand 每 4 帧(~7Hz,指向偶发够用),返回 dict 协议。

### 四层动作仲裁(优先级高→低)
**手势(Primary)> 声源转向/指向(事件性)> 人脸跟随(Tracking)> 微动(Idle)。** 头部唯一 set_target 写入口仍是 `head_control_loop`(25Hz):渲染 = behavior/视觉给的 track 目标 + 微动 + body_yaw;手势 `action_active` 时完全让位(goto 独占);微动仅在 IDLE/TRACKING 且说话时叠加。**视觉只在 TRACKING 态积分头部目标**(其余态由 behavior 驱动,杜绝双写)。
- ⭐ 参考系(§11 教训):head pose 是世界系,body_yaw 被 Stewart 反向补偿 → 大角度转向 head 给完整目标角、body_yaw 只是分担量(颈相对 ≤23°,身体 ≤90°)。
- 手势在转过去的身体朝向上做(body_yaw 传当前值,不拽回正中)。

### 行为状态机(behavior_loop 统一调度,25Hz)
`IDLE_CENTER → ENGAGING → TRACKING ↔ SEARCHING → RETURNING → IDLE`,外加 `POINTING`:
- **IDLE:** 头回正+微动,监听。锁定人脸→TRACKING;视场外声源(DOA 残差>25°)→ENGAGING。
- **ENGAGING:** 转向声源 + 到位后主动 ±15° 扫头找人;锁定→TRACKING,超时→RETURNING。
- **TRACKING:** 视觉积分跟随;丢锁→SEARCHING;15s 无说话互动且没在说→RETURNING。
- **SEARCHING:** 原地保持;重新锁定→TRACKING,有声源→ENGAGING,超时(4s)→RETURNING。
- **RETURNING:** 平滑回中位→IDLE。
- **⭐ 两个状态机 bug 的修法(都已实测):**
  1. **状态空转**(TRACKING↔SEARCHING 高频横跳):用**时间迟滞的 face_locked**——持续命中 `LOCK_ON_S=0.3s` 才算锁定、持续丢失 `LOCK_OFF_S=1.5s` 才算丢锁;behavior 用 face_locked(非瞬时 face_seen)做进出。实测 40s 跟踪零空转。
  2. **15s 计时器永不触发**:`last_interaction_at` 只在**首次捕获**(IDLE/ENGAGING/RETURNING→TRACKING)播种,SEARCHING↔TRACKING 回切**不重置**(否则抖动每秒清零)。说话(用户/机器人)也重置。

### POINT-02 指向理解:转头重新取景(关键认知)
**POINT-01 教训:** 拿"物体在边缘的烂图"硬猜不行——Qwen 懂指向语义(会沿食指延长线推理),但 2D 丢深度、延长线撞最显眼大目标。**正解 = 先转头把目标摆进画面中央,再看图**(不追求精确指到点,追求把目标转进视野,看图兜底识别)。
- **检测:** MediaPipe Hand Landmarker(`vision/models/hand_landmarker.task`,7.8MB,google storage 直连)取食指根 MCP(5)+ 指尖 TIP(8)→ 2D 方向角。**左右是镜像物理对应**:用户指自己左边 = 摄像头视角的右边,读"右"正确(摄像头不镜像,无需翻符号)。
- **映射:** 画面右(dx>0)→ yaw 负(机器人右);画面下(dy>0)→ pitch 正。增益 `POINT_YAW_GAIN=38° / POINT_PITCH_GAIN=12°`。
- **POINTING 态四子阶段(关键:冻结跟踪 + 停稳再抓 + hold 到看图完成):**
  `turn` 转向手指方向 → `settle` 停稳 `0.6s`(电机/相机稳定,目标居中)→ 抓帧(`snap_grabbed` 握手)→ `hold` **保持朝向直到 `snap_grabbed && snapshot_pending==0`(帧抓到+看图返回,~1-3s)** → RETURNING 转回看人。
  - **踩坑:** 初版转到位就立刻 `POINTING→TRACKING`,视觉在帧抓到前就把头拽回人脸 → 目标没停稳(5 个只对 2 个)。修法即上面的 hold-until-grabbed。视觉积分在 POINTING 态本就冻结(只 TRACKING 积分),叠加 hold 才彻底不被拽回。
  - 触发:模型调 `identify_pointed_object`(语音"这是什么/我指的是什么")→ callback 置 `point_request` + `snapshot_pending++`(占位防 response.done 抢跑),behavior 接管转头,转完才入 snap_q。
  - **遗留(工具路由模糊):** "你看/看一下这是什么"模型有时选 `take_snapshot`(场景模式,不转头);"这个是什么/这是什么"才稳定走指向转头。看图 prompt 已做"指向感知"兜底。

### ⚠ 已知问题(未来用视觉门控解决)
- **背景人声 vs 真人对话区分不了:** Realtime VAD 会把电视/视频人声当说话 → 反复重置 RETURNING 计时器、甚至误触发 ENGAGING 声源转向。正解:**视觉门控**(检测到有脸 + 嘴在动才算"在跟我说话")。现在记账,未做。

### 1e 底牌(GPU 检测器,已装通不集成)
- 本机 **RTX 5060 Laptop 8GB**;`onnxruntime-gpu` + `insightface` 经清华镜像装通(exit 0)。备用:若将来 MediaPipe 侧脸召回不够,可换 InsightFace/SCRFD on CUDA。**当前不集成**——27fps MediaPipe + 迟滞已满足。

---

## 14. WAKE-01 唤醒词「小艺」standalone 验证 ✅(2026-06-16,M1a)

> 工具:`tools/wake01_kws_standalone.py`(留作标定);引擎:sherpa-onnx KeywordSpotter;
> 全程不碰 d01,d01 没跑时本脚本独占麦(`get_audio_sample`)。模型大文件在 `tools/_kws_models/`(gitignore)。

**最终锁定配置:**
- 唤醒词 = **单字「小艺」**;keywords 留 **yī / yìn / yì 三声调形态**(`x iǎo y ī` / `x iǎo y ìn` / `x iǎo y ì`),**命中即唤醒**,无二次确认、无叠词。
- `--single-thr **0.17**`(per-line `#0.17`);去抖 0.3s(同一声"小艺"会同帧点亮多形态行,算一次)、不应期 2.0s(防同次喊话重复唤醒)。
- 模型:**int8** `sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`(GitHub release 直连可下,32MB);纯 CPU,单 chunk ~2.5ms,与对话/视觉并跑无压力。
- per-line `#threshold` 经离线确定性验证**双向都生效**(`tools/_kws_perline_test.py`),故单字行可独立设阈值(本配置只单字,留此能力给将来 A+C)。

**走过的弯路(防后人重踩):**
- **叠词「小艺小艺」在真人远场音下 token 解码不出**(变体阶梯实测命中 0~1 次),原以为"叠词=两次检出"可做防误触 + 二次确认 —— **该假设在真人语音不成立**(连读「小艺小艺」只产生 ~1 个单字命中,不是两个),双单确认召回被压到 1/10。**叠词 + 双单确认均弃用,退回单字。**
- 调研期合成音(pyttsx3 标准发音)叠词能 4/4,误导了方案;真人远场必须实测。

**真人发音落点(变体阶梯实测,本用户/环境):**
- 发"艺"**系统性偏一声 `yī`(命中最多)与带鼻音 `yìn`**,标准四声 `yì` 只排第三 → keywords 按此配三形态。轻声 `yi`、二/三声 `yí/yǐ` 命中≈0,不挂。

**召回 / 误触实测:**
- 召回:单字「小艺」≈ **9/10**(@0.17;多数命中分 ≥0.20)。
- **误触地板:有电视时 ~0.7 次/5min**(实测 2 次/~15min,且挤在一段大声电视里)。根因:电视出现「小艺/笑意」等**同音**,且命中分 **≥0.20 与真音同强** → **阈值/能量门都分不开**(误触那两次反而更响)。
- ⭐ **正确分工:强误触(电视同音)不靠阈值硬扛**(压到 ≥0.25 会把召回打回 7-8/10),**留给 M1.5 的 DOA 方向门控 + 15s 无互动回待命兜底**。

**麦克风:** ReSpeaker(XVF3800)**F32LE 16kHz mono**(`reachy_mini/audio_base.py` 确认,非 48k 无需重采样);喂 KWS 用 int16 之前的 float32 `chunk[:,0]`。

**⚠️ 仍未查的悬案:** ReSpeaker 的 AGC/降噪/AEC 在芯片固件内,这层 Python SDK **不暴露关 AGC / 取近端 raw 的口子**(全量翻 `reachy_mini` 无此 API)。**若 M1.5 召回/误触吃力,这是头号嫌疑**(处理后的音可能糊了唤醒音)——需碰 XVF3800 固件/控制面,记在案。

**适用范围标注:** 本配置为**当前单用户 / 家庭环境**调校(发音分布、环境噪声特定)。产品化面向他人**须重测**(他人"艺"声调分布、不同环境噪声会变)。

### DOA 方位可用性实测 + M1c 唤醒转向策略(2026-06-16,armed 下 body_yaw=0 纯传感器标定)

> 起因:实机观察"某角度喊会转、换角度不转/转错"。在 `doa_sensor_loop` 加 `DOA_DEBUG=1` 打印 raw/resid/判向,逐方位标定。

**可用区(信 DOA 直接转向)= 前半球 + 左右侧 + 侧后,约 ±150° 内:** raw 随方位平滑变化、方向判定正确(密采样实测:左 30°/60°/90°/120°/150° 的 raw = 75°/61°/13°/26°/47°,全程 <90° 稳定判"左转";右半球同理)。这片唤醒后直接朝 DOA 转 → 人脸跟随收敛锁定。

**坏区(不信 DOA 方向,改转身扫一圈找脸)= 深后方 ~150°–210°(正后及紧贴两侧):**
- **正后**:raw 折成 ~81°≈正前(resid<门)→ 判正前**不转**(="后面喊它不动")。
- **深后翻转**:同一"左后"位置连喊 5-6s,raw **从 54°(左)跳到 147°(右)**、resid 从 +36° 翻到 −58°——**锥面模糊 + 时间不稳定**,同位给左/右两个解(="转错")。

**⚠️ 纠正第一轮粗采样的错结论:** 第一轮 8 点粗采样曾得"左侧/左后 raw=144° → 镜像到右"的结论,并写入旧版本——**密采样证伪**:那其实是站位偏到了深后不稳定区所致,正左到侧后(±150°内)实为稳定正确。**教训:空间边界必须密采样,粗采样会假阳性。**

**性质:** XVF3800 经 REST 暴露的 DOA 在深后方有锥面/前后模糊 + 时间不稳定;**非检测、非门限(SND_RESID_MIN)、非增益**(8 方位全部 vad=1、n≥5 采到声,门限工作正常)——**代码/参数不可治**。

**M1c 转向最终模型 = SEEK「视觉主导寻人,DOA 仅起扫弱提示」(2026-06-16 实机定稿,推翻下面两个早期错误前提):**
- **❌ 错误前提 1「唤醒词那段取 DOA 转向」**:实测"小艺"~0.5s,命中那刻 DOA 只攒到 n=1~3 < SND_MIN_SAMPLES=5 → **算不出方向**(DOA 要 ~1.5s 持续语音)。唤醒词本身给不出方向。
- **❌ 错误前提 2「spread 小 = DOA 可信」**:实测左前命中刻 resid 报 **-57°(右,镜像错)而 spread=0**——**镜像错是"稳定地指向错方向",spread 抓不到、反而判成可信自信转错**。**spread 只测稳不测对。** 那次靠脸在视野内被视觉救;视野外就会背对人停住。
- **✅ DOA 命中场景三失效**:① 唤醒词太短样本不够(算不出);② 稳定地报错方向(镜像);③ 后方 >±90° 物理够不着。三轮实测唯一一直可靠的是**视觉**(每次最终都靠人脸锁回)。
- **✅ SEEK 模型**:唤醒 → 视觉主导寻人。**DOA resid 只取【符号】当"先往哪边扫"的弱提示**(不信角度、不信 spread);从中心 sin 慢扫(正前的人 t≈0 秒锁)+ **pitch 覆盖(抬头+慢摆,防漏高处的脸)**,边扫边跑人脸检测,**见脸即锁=视觉裁判**;扫两侧 ~7s 仍无人 → giveup 小动作回 armed。镜像错被"扫两侧+见脸才算"天然兜回(与 POINT-02 同纪律)。
- **spread 降级**:不进任何决策,**仅留 DOA_DEBUG 日志**看稳定性。`SPREAD_BAD` 常量仅供 debug 标注 zone。

**⭐ 深后方 = 双重硬件盲区(M1c-b 实测,接受为 Reachy Mini 物理边界,软件不可救):**
- ① **唤醒层**:站正后/深后喊"小艺"实测**命中 ~3/10**(原始 KWS 检出 ~5/10)。头载 ReSpeaker 朝前波束,背后人声被衰减/糊化;**KWS 认关键词比 VAD 严得多** → DOA 那轮深后能 `vad=1`(测得出有声/方向)≠ 认得出唤醒词。
- ② **物理层**:头世界朝向 ±90°,**面不过正后**(身体 ±65°+颈 ±25°)。
- 结论:**正后方召唤是硬件盲区**——不为深后调 single-thr(会全局抬误触,得不偿失);**人需绕到 ±90° 内**。
- **宽扫职责正名**:宽扫**不是**"召唤正后方的人"(盲区),而是 **"DOA 镜像错恢复"**——人在 ±90° 内、唤醒正常,但 DOA 指反时(深后翻转映射 raw 横跳→spread 大判坏区),宽扫慢扫 ±90° 弧、用人脸检测把人兜回来。这是高频有用场景,逻辑保留。

**⚠️ 悬案合并:** 深后方根治需查 XVF3800 是否有 raw 音频 / 更多麦 / 更好 DOA 模式 —— **与 §14 的 AGC/降噪 raw 悬案合并**(同一颗 XVF3800 固件/控制面,本 Python SDK 不暴露)。

**多脸选择(现状 + 留项):** 当前多脸 = **每帧取最大脸**(`vision_worker.pick_main_face`,按高度 argmax),**无跨帧迟滞**——两张相近大小的脸理论上会抖动跳脸。单人场景够用。**sticky(相近脸迟滞粘住)+ 多人注意力切换 → 留 M1.5**(其价值只在多人/多相近脸时显现)。

### 唤醒确认动作(M1c-a,2026-06-16,实机验收"成功/失败一眼区分")

> 起因:M1b 成功/失败都是"天线弹",分不清。M1c-a 把三状态(听到/连上/连失败)落成**两个方向相反的动作**。

- **heard(听到了,命中即时 0 延迟,`open_session` 之前触发)= 上扬**:头抬 `pitch −7°` + 双天线 `+0.5 rad`,sin 包络,时长 **0.45s**。"嗯?在。"
- **fail(连失败,3s 超时后触发)= 下垂**:头沉 `pitch +6°` + 双天线 `−0.7 rad`,时长 **0.80s**。"没连上。"
- **success 不做独立动作**(用户拍板):heard 已是注意力信号,连上后直接转向+对话本身就是确认,避免动作过密。三信号实落成 heard/fail 两个,一上一下一眼辨。
- **实现**:cue = 单槽 `st.wake_cue`(kind+起始时刻),**后到覆盖先到**(fail 接管 heard,不叠加抽搐);**additive 叠加偏置**(pitch 偏置 + 天线覆盖),不抢渲染、**可被转向打断**(转向一来基座照走、cue 自然衰减);全部经 head_control(set_target 唯一写入口)渲染,不碰 behavior_loop 状态。CLI 可调 `--cue-heard-pitch/-ant/-dur`、`--cue-fail-*`。
- **⚠️ `--simulate-conn-fail` 回归注意**:该开关让 `open_session` **跳过、立即返回 None** → fail cue **瞬间覆盖 heard**(几乎只见下垂);**真实断网失败是 connect 3s 超时后才触发** → 体感是"上扬 →(等 3s)→ 下垂"两段。回归测失败 cue 时记得两者看到的不一样。

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
| F-01 融合:跟随并进对话(三层仲裁) | 2026-06-06 | ✅ 单一头控写入口,手势基准化零突跳(初版,已被 §13 取代),见 §10 |
| DOA-01 听声辨位调研+实测 | 2026-06-06 | ✅ XVF3800 板载 DOA 经 REST 可用,左/前/右清晰、前后不分,见 §11 |
| SOUND-TURN-01 听声转头 | 2026-06-06 | ✅ 闭环逐步逼近,两条参考系教训 ⭐⭐,左右对准收敛稳,见 §11 |
| DAEMON-FIX exit 116 定位+缓解 | 2026-06-06 | ✅ 根因=no_media 触发媒体重建循环;补丁+daemon_up.py,触发面≈0,见 §12 |
| FUSION-02/03 + TRACK-FIX + POINT-02 | 2026-06-06 | ✅ 四层仲裁+视觉进程化(27fps)+行为状态机+指向转头,见 §13 |
| WAKE-01 唤醒词「小艺」standalone 验证 | 2026-06-16 | ✅ 单字"小艺"yī/yìn/yì @0.17,召回~9/10,误触~0.7次/5min(地板,留给M1.5 DOA门控),见 §14 |
