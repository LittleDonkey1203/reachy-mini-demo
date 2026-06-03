# Reachy Mini Lite 硬件 I/O 体检

逐项验证 5 大核心 I/O 通道是否正常的独立脚本。结论已固化到上级目录的 [`../CALIBRATION.md`](../CALIBRATION.md)。

## 前置条件

1. **daemon 必须在跑**(localhost:8000)。启动:
   ```powershell
   & "C:\Users\ldkji\AppData\Local\Reachy Mini Control\.venv\Scripts\reachy-mini-daemon.exe"
   ```
   Lite 自动识别 COM3,启动后机器人会上电唤醒(电机使能、回初始位、播唤醒音)。约 15 秒就绪。
2. 用 venv 的绝对路径 python 跑,并带 `PYTHONUTF8=1` 避免中文乱码:
   ```powershell
   $env:PYTHONUTF8=1
   & "C:\Users\ldkji\AppData\Local\Reachy Mini Control\.venv\Scripts\python.exe" healthcheck\01_connect.py
   ```
3. 依赖:`numpy`、`scipy`、`gi`(GStreamer,SDK 自带)已有;额外装过 `Pillow`(存 jpg)、`pyttsx3`(中文 TTS)。
   清华镜像:`pip install pillow pyttsx3 -i https://pypi.tuna.tsinghua.edu.cn/simple`

每个脚本开头打印 `=== 第N项:XXX 体检 ===`,结尾打印 `=== 通过/失败 ===`,退出码 0=通过 / 1=失败。

## 5 个脚本

| 脚本 | 验证什么 | 怎么跑 / 注意 |
|---|---|---|
| `01_connect.py` | **连接**:连接模式、SDK/daemon 版本、9 电机、当前 pose、IMU(Lite 无)、电池(Lite 无) | 直接跑。用 `no_media`,只测控制平面 |
| `02_motion.py` | **动作**:reset→点头→摇头→看左/右→看上/下→天线摆动→body转动→reset,全程 min-jerk 平滑 | 直接跑。**人在旁边看方向/听异响**。用 `automatic_body_yaw=False` 隔离头部与身体 |
| `03_camera.py` | **视频流**:`media.get_frame()` 抓 60 帧,打印分辨率/dtype/帧间隔,统计 FPS,存第 1/30/60 帧 jpg | 直接跑。**只用 SDK get_frame(),不开 cv2**。看 `output/` 里的 jpg 确认画面 |
| `04_mic.py` | **音频输入**:录 8 秒,打印采样率/声道/总+左右 RMS,存 wav | **建议在自己的 PowerShell 里跑**:能看到实时 3-2-1 倒计时,看到 `▶ 现在开始说话` 再连续说"测试一二三"满 8 秒。回放 `output/mic_recording.wav` 确认 |
| `05_speaker.py` | **音频输出**:生成 440Hz 正弦 + 中文 TTS(Huihui)两个 wav,用 `play_sound()` 播放,打印实际选中的输出声卡 | 直接跑。**坐在机器人旁边听**:确认声音从机器人扬声器出、中文听清 |

## 产物(`output/`)

- `camera_frame_01/30/60.jpg` — 摄像头抽样帧
- `mic_recording.wav` — 麦克风录音(16kHz 双声道)
- `test_tone_440hz.wav` — 440Hz 测试音
- `tts_zh.wav` — 中文 TTS 语音

## 关键约束(都已在脚本中遵守)

- 摄像头只用 `media.get_frame()`,**不另开 `cv2.VideoCapture`**(抢设备冲突)。
- 音频只用 SDK 的 `media` GStreamer 接口,**不另开 `sounddevice`**;SDK 自动按名字选 "Reachy Mini Audio" 卡。
- 动作全程 `goto_target` 平滑插值,幅度保守。
- 每个脚本用 `with ReachyMini(...)`,退出自动释放媒体资源,**不占着设备影响下一项**。
- import `reachy_mini` 前先设 `NO_PROXY`(绕本机代理,否则连接 EOF 失败)。

## 体检结论(2026-06-03)

5 项全部 ✅ 通过。详见 [`../CALIBRATION.md`](../CALIBRATION.md)。
