# macOS (Intel) 部署指南 — 小艺 Reachy Mini Lite

> 原项目在 Windows 11 实测开发。本文档记录在 **macOS Intel (x86_64)** 上完整复现的过程，  
> 包含所有踩坑记录、代码改动和功能验证结果，供后续复现参考。  
> 验证日期：2026-06-09 | SDK 版本：reachy-mini 1.8.0

---

## 一、环境差异总览

| 项目 | Windows（原版） | macOS Intel（本次） |
|---|---|---|
| daemon 启动 | `tools/daemon_up.py`（硬编码 `.exe` 路径） | `reachy-mini-daemon` CLI（SDK 内置） |
| Python | venv 随 Reachy Mini Control 安装 | uv 管理，Python 3.12 |
| mediapipe | ✅ 支持 | ❌ 无 x86_64 wheel，改用 OpenCV 替代 |
| 串口 | COM3 | `/dev/cu.usbmodem5B7B0104271`（自动检测） |
| GStreamer | 随 daemon 安装 | 随 reachy-mini PyPI 包附带 |

---

## 二、前置依赖安装

### 1. 安装系统依赖

```bash
# Homebrew（若未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.12 + uv
brew install python@3.12 uv

# pygobject 编译依赖（reachy-mini 依赖链需要）
brew install pkg-config cairo
```

> **踩坑**：系统自带 Python 3.7 太老；pygobject 编译需要 `cairo` 和 `pkg-config`，缺少时 `uv sync` 报错。

### 2. 初始化 uv 项目

```bash
cd <项目根目录>
uv init   # 若已有 pyproject.toml 则跳过
```

### 3. 配置 pyproject.toml

`mediapipe` 在 macOS x86_64 无 wheel，需加平台限定：

```toml
[project]
name = "shadow-yunwu-claudecode-aibasicalgodevdept-20260529"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "reachy-mini",
    "openai",
    "dashscope",
    "mediapipe; sys_platform == 'win32' or (sys_platform == 'linux') or (sys_platform == 'darwin' and platform_machine == 'arm64')",
    "opencv-python",
    "numpy",
    "scipy",
    "Pillow",
    "pyttsx3",
]
```

### 4. 同步依赖

```bash
# 使用清华镜像加速（国内网络）
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple/
```

> **踩坑**：直连 pypi.org 超时，需加镜像参数。

---

## 三、代码改动

macOS Intel 上需要修改两处文件（已提交到代码库）：

### 改动 1：`voice/vision_worker_cv.py`（新增）

mediapipe 在 Intel Mac 不可用，新增基于 OpenCV Haar Cascade 的人脸检测后备进程，协议与原 `vision_worker.py` 完全兼容（`hand` 字段恒为 `None`）：

```
voice/vision_worker_cv.py   ← 新增文件
```

功能：
- 使用 `haarcascade_frontalface_default.xml`（cv2 内置，无需额外下载）
- 输出协议与 mediapipe 版本相同，主程序无感切换
- 不支持手部检测（hand 恒 None），影响"逗它互动"功能

### 改动 2：`voice/d01_realtime_chat.py`（两处修改）

**① import 段（第 87 行附近）**：自动探测 mediapipe 可用性，不可用时降级到 opencv 后端：

```python
# 原代码
from vision_worker import vision_worker

# 改为
try:
    import mediapipe as _mp_probe
    del _mp_probe
    from vision_worker import vision_worker as _vision_worker_fn
    _VISION_BACKEND = "mediapipe"
except ImportError:
    from vision_worker_cv import vision_worker as _vision_worker_fn
    _VISION_BACKEND = "opencv"
```

**② 视觉子进程启动段（第 1388 行附近）**：opencv 后端无需 `.task` 模型文件即可启动：

```python
# 原代码：只在 .task 文件存在时启动
if os.path.exists(VIS_MODEL_PATH):
    ...

# 改为：opencv 后端直接启动，mediapipe 才检查文件
_vis_enabled = (
    (_VISION_BACKEND == "mediapipe" and os.path.exists(VIS_MODEL_PATH)) or
    (_VISION_BACKEND == "opencv")
)
if _vis_enabled:
    log(f"视觉后端: {_VISION_BACKEND}")
    ...
    multiprocessing.Process(
        target=_vision_worker_fn,  # 用统一变量
        ...
    )
```

> **注**：若平台支持 mediapipe（Windows / Linux x86_64 / macOS arm64），建议仍下载原版模型文件以获得手部检测能力：
> - `face_landmarker.task`：https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
> - `hand_landmarker.task`：https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
> - 下载后放入 `vision/models/`

---

## 四、启动步骤

### 1. 确认串口

```bash
ls /dev/cu.usbmodem* /dev/cu.usb*
# 预期看到类似：/dev/cu.usbmodem5B7B0104271
```

### 2. 启动 daemon

```bash
cd <项目根目录>
mkdir -p log

nohup .venv/bin/reachy-mini-daemon \
  -p /dev/cu.usbmodem5B7B0104271 \
  --localhost-only \
  --log-level INFO \
  >> log/daemon.log 2>&1 &

echo $! > .server.pid

# 等待约 15 秒后验证
sleep 15
curl -s http://127.0.0.1:8000/api/state/full | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('control_mode:', d['control_mode'])"
# 期望输出：control_mode: enabled
```

> **踩坑 1**：加 `--no-wake-up-on-start` 会导致 `control_mode: disabled`，电机无法动作，必须去掉。  
> **踩坑 2**：用普通 `&` 启动，shell 退出时 daemon 被 SIGHUP 杀死。必须用 `nohup` + 后台运行。

### 3. 设置 API Key

```bash
export DASHSCOPE_API_KEY=<你的阿里云百炼 API Key>
```

DashScope 统一平台，一个 Key 覆盖对话（qwen3.5-omni-plus-realtime）和 VLM（qwen3.5-omni-plus），无需分开申请。  
申请地址：https://bailian.console.aliyun.com/ → API-KEY 管理

### 4. 启动主程序

```bash
export NO_PROXY=localhost,127.0.0.1,::1
export PYTHONUNBUFFERED=1

nohup .venv/bin/python -u reachy-mini-demo/voice/d01_realtime_chat.py \
  >> log/main.log 2>&1 &

echo $! > .main.pid
```

观察日志确认启动成功：

```bash
tail -f log/main.log
```

期望看到：
```
✅ 录音/播放管线已启动;回中立位…
摄像头:✅ 出帧 (1080, 1920, 3)
✅ 会话已建立 session_id=...
✅ WebSocket 已连接 dashscope.aliyuncs.com
✅ 会话配置生效(semantic_vad / 8 动作 + take_snapshot + identify_pointed_object 已注册)
▶ 可以对机器人说话了
视觉后端: opencv
👂 声源传感器就绪(DOA REST 10Hz)
👁 视觉子进程就绪(Face 跟随 + Hand 指向,独立 GIL)
```

### 5. 停止程序

```bash
# 停主程序
kill $(cat .main.pid)

# 停 daemon（daemon 会自动把机器人放到睡眠位）
kill $(cat .server.pid)
```

---

## 五、功能验证结果（2026-06-09）

### Healthcheck 套件

```bash
# 逐项运行（需 daemon 已启动）
.venv/bin/python reachy-mini-demo/healthcheck/01_connect.py
.venv/bin/python reachy-mini-demo/healthcheck/02_motion.py
```

| 脚本 | 结果 | 关键指标 |
|---|---|---|
| 01_connect.py | ✅ 通过 | 9 个电机（头部 7 + 天线 2），SDK/daemon 均 1.8.0 |
| 02_motion.py | ✅ 通过 | 点头/摇头/看四方/天线/身体转动全部执行 |

### 主程序功能验证

| 功能 | 结果 | 实测数据 |
|---|---|---|
| 🗣 全双工对话 | ✅ | 首音频延迟 329-371ms（目标 <400ms） |
| 🦾 身体语言 | ✅ | `wiggle_antennas`、`tilt_head` 等工具自动调用 |
| 👀 人脸跟随 | ✅ | OpenCV 后端 ~29fps，检出率 25-67% |
| 👂 听声转向 | ✅ | DOA 残差 +42° → ENGAGING → 锁定人脸全流程 |
| 👉 指向理解 | ✅ | 两段式：VLM 判断在指→转头重取景→描述目标 |
| 🎾 逗它互动 | ⚠️ 不可用 | 依赖手部检测，Intel Mac mediapipe 不支持 |
| 📷 看图描述 | ✅ | VLM 精确描述画面内容，延迟 1.4-3.3s |
| 🧭 行为状态机 | ✅ | IDLE/ENGAGING/TRACKING/SEARCHING/RETURNING/POINTING 正常切换 |

**7/8 功能验证通过**。

---

## 六、已知限制与后续

### Intel Mac 特有限制

- **mediapipe 不可用**：macOS x86_64 无预编译 wheel（官方仅提供 arm64 / Linux / Windows）。
  已用 OpenCV Haar Cascade 替代人脸检测；手部检测无替代，"逗它互动"功能不可用。
- **人脸检出率偏低**：OpenCV Haar Cascade 约 25-67%（mediapipe 正脸接近 100%）。
  侧脸/光线不足时跟随会有中断，影响不大但有感知。

### 如需完整功能（含手部检测）

在以下平台运行可恢复全部 8 项功能：
- Windows 11（原版环境）
- Linux x86_64
- macOS Apple Silicon (M1/M2/M3)

恢复步骤：删除 `vision/models/` 下的占位目录，下载两个 `.task` 文件，`pyproject.toml` 中
mediapipe 的平台限定条件自动命中，`uv sync` 后即可。

### GStreamer 警告（无害）

启动时有大量 `libgstpython.dylib` 找不到 `libpython3.12.dylib` 的警告，原因是 GStreamer
Python 插件的 rpath 写死了非 venv 路径。不影响任何功能，忽略即可。

### daemon 掉线问题

原 CALIBRATION.md §12 记录的 exit 116 崩溃在 Windows 版偶发。macOS 上暂未复现，
但建议长时间运行前检查 `log/daemon.log`，发现异常重启 daemon 即可。

---

## 七、目录结构（macOS 新增文件）

```
voice/
  vision_worker_cv.py    ← 新增：OpenCV Haar Cascade 人脸检测（mediapipe 后备）
  d01_realtime_chat.py   ← 修改：自动切换视觉后端，增加 opencv 启动分支
pyproject.toml           ← 修改：增加平台限定 mediapipe 条件 + pyttsx3
MACOS_SETUP.md           ← 新增：本文档
```
