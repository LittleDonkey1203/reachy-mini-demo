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
- **是否补 `response.create`(关键设计,避免双重说话):**
  - 发起调用的响应**带音频**(边说边动)→ 动作完成只回 output,**不补**;
  - 发起调用的响应**纯动作无音频** → 全部动作完成后**补一次** `response.create`,模型才继续开口;
  - 判据:`response.created` 时清零本响应 audio.delta 计数,`response.done` 时看计数是否为 0;期间被打断(代际变化)则不补。

### 并发与线程模型
- 动作任务入队,**独立动作线程串行执行**(`goto_target` 是阻塞插值,绝不能在音频回调/播放线程里调)。
- 8 个工具:nod / shake_head / look_left|right|up|down / wiggle_antennas / tilt_head,全部复用 §2 标定幅度(头 ±10~12°、天线 ±0.5rad、`automatic_body_yaw=False` + 全程 `body_yaw=0`),单个动作 1.6~1.8s。
- tilt_head 用 roll ±12°(roll 方向当时未标定,幅度在安全范围,实测无异响)。
- **barge-in 时动作不中断**(动作短,让它做完),只停音频。

### 踩坑记录
- **semantic_vad 有音量门槛:** 上行 RMS ≈0.004(说话太小声/离远)**完全不触发** speech_started——表现为"上行了 75s 音频但服务端零事件",像断连但其实是没过门槛。正常说话 RMS ≥0.01 即稳定触发。排查手段:上行循环里按周期打印 RMS(已内置在正式脚本,<0.005 时提示)。
- `update_session` 不传 `tools` 字段时**不会清除**已注册的工具(继承上次配置);要清除需显式传空。

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
