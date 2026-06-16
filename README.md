# 小艺(Xiaoyi)— 基于 Reachy Mini Lite 的桌面对话机器人

把一台 [Reachy Mini Lite](https://www.pollen-robotics.com/reachy-mini/)(USB 版)变成一个
**能看、能听、能聊、能玩**的桌面伙伴:全双工语音对话 + 人脸跟随 + 听声转向 + 指向理解 + 逗它互动,
行为由统一状态机调度,像一只好奇的小猫待在桌上。

> 详细技术标定、参数与踩坑沿革:[CALIBRATION.md](CALIBRATION.md)
> 主程序架构(线程/状态机/仲裁/改码铁律):[voice/ARCHITECTURE.md](voice/ARCHITECTURE.md)

## 当前能力(均已实机验收)

| 能力 | 说明 |
|---|---|
| 🗣 全双工对话 | Qwen3.5-Omni-Realtime(阿里云百炼),semantic_vad,首音频 ~300-400ms,随时插话打断(~20ms 闭嘴);自称"小艺" |
| 🦾 身体语言 | 8 个动作工具由模型自主调用(点头/摇头/看向/摆天线/歪头),说话时叠加 idle 微动 |
| 👀 人脸跟随 | MediaPipe 子进程 ~27fps,时间常数型平滑控制,迟滞防抖;聊天时头温和跟着人转 |
| 👂 听声转向 | XVF3800 板载 DOA,视场外有人说话 → 闭环转向找人 → 人脸进视野交还视觉 |
| 👉 指向理解 | 两段式:先原地看图由 VLM 判断"是否真在指/目标是否在画面",目标在画面外才转头重取景再回答 |
| 🎾 逗它互动 | 近处晃动的手 → 像猫被逗猫棒吸引地跟手(灵敏档+惯性外推),持续逗会开心摇天线;手停/离开则失去兴趣回去跟脸 |
| 📷 看图 | "你看到什么" → 抓帧 → Qwen-VL 描述 → 语音转述 |
| 🌙 唤醒待命 | 待命态只听唤醒词「小艺」(本地 sherpa-onnx KWS,不连云、零计费);喊「小艺」→ 上扬确认 → 视觉主导寻人(SEEK)→ 锁脸对话;长静默回待命 |
| 🧭 行为状态机 | ARMED / IDLE / ENGAGING / TRACKING / SEARCHING / RETURNING / POINTING / PLAYING,五层动作仲裁(手势/指向 > 逗它 > 声源 > 跟脸 > 微动),头部唯一硬件写入口 |

## 环境要求

- Windows 11(本项目在 Win11 + PowerShell 实测)
- Reachy Mini Lite,USB 连接(实测 COM3),**确认机器人电源开启**(USB 在 ≠ 电机有电)
- Python 3.12 venv:随 Reachy Mini Control 安装,路径形如
  `C:\Users\<你>\AppData\Local\Reachy Mini Control\.venv`(SDK/daemon 1.7.3)
- 环境变量 `DASHSCOPE_API_KEY`(阿里云百炼,对话+看图都用它)
- MediaPipe 模型(gitignore,不随库;下载地址见 CALIBRATION.md §9/§13):
  `vision/models/face_landmarker.task`(3.7MB)、`vision/models/hand_landmarker.task`(7.8MB)
- GPU 可选:纯 CPU 即可跑全部能力(RTX 仅作为将来检测器升级的备胎)
- 如本机有代理:脚本已内置 `NO_PROXY=localhost,127.0.0.1,.aliyuncs.com`,无需手工设置
- pip 装包建议清华镜像:`-i https://pypi.tuna.tsinghua.edu.cn/simple`

## 启动

```powershell
$py = "C:\Users\<你>\AppData\Local\Reachy Mini Control\.venv\Scripts\python.exe"
$env:PYTHONUTF8 = 1

# 1. 启动 daemon(探活/清残留/就绪判定/电源-过载分诊;退出码 2=查电源 3=断电清过载)
& $py tools\daemon_up.py            # 已在线则跳过;--restart 强制重启(长跑前建议)

# 2. 启动小艺完整体(默认待命态:喊"小艺"唤醒;Ctrl+C 退出;带秒数到时自动干净退出)
& $py voice\d01_realtime_chat.py
& $py voice\d01_realtime_chat.py 180        # 跑 180 秒
& $py voice\d01_realtime_chat.py --no-wake  # 回退:启动即连即对话(跳过待命门控,排错/对比)
```

**唯一主入口就是 `voice/d01_realtime_chat.py`**(自动拉起 `vision_worker.py` 视觉子进程);
其余可执行脚本均为独立 demo / 调参 / 诊断工具,不参与主程序。

## 项目结构

```
voice/
  d01_realtime_chat.py   ⭐ 主程序:对话+动作+看图+跟脸+听声转向+指向+逗它(完整体)
  vision_worker.py        视觉子进程:Face 每帧 + Hand 自适应提频(独立 GIL)
  ARCHITECTURE.md         主程序架构说明(四能力边界/线程/状态机/仲裁/铁律)
  _judge_unit.py          指向 judge JSON 解析的离线单测
vision/
  vis01_face_track.py     人脸跟随独立 demo(VIS-01)
  play01_hand_track.py    逗它跟手独立调参脚本(PLAY-01-a,六轮实测调校)
  _play01_ghost_diag.py   手部误检诊断(存帧画框人工看)
  models/                 MediaPipe 模型(gitignore,需自行下载)
audio/
  sound_turn.py           听声转头闭环 demo(SOUND-TURN-01)
  doa01_test.py / _doa_*  DOA 调研与诊断脚本
tools/
  daemon_up.py            ⭐ daemon 可靠启动器(标准启动方式)
  ISSUE_DRAFT_daemon_116.md  daemon 崩溃问题的上游 issue 草稿
healthcheck/              五项硬件 I/O 体检脚本
docs/
  KWS_RESEARCH.md         唤醒词"小艺"可行性调研报告(sherpa-onnx,已最小验证)
CALIBRATION.md            ⭐ 全部实测标定/参数/踩坑(§1-§13)
connect.py                最小连接冒烟
```

## 已知限制

- **背景人声**:已由 **WAKE-01 待命门控根治**——待命态根本不连 Qwen,电视声无从触发(M1 完成)。
  残留:engaged(对话中)的电视声区分留 M1.5 方向门控;单字"小艺"误触地板 ~0.7 次/5min(电视同音,阈值救不了,靠方向门控压)。
- **深后方双重盲区**:正后/深后(~150-210°)唤醒命中 ~3/10(头载麦朝前)+ 头转不过 ±90°,**正后召唤是硬件盲区**,人需绕到 ±90° 内。
- **多脸**:每帧取最大脸,无跨帧迟滞(相近脸会跳);sticky + 多人注意力切换留 M1.5。
- 指向理解的 2D 限制:目标在画面内时由 VLM 直接判断;画面外靠粗方向转头,不保证一次精确对准。
- MediaPipe 侧脸/移动中检出率 30-45%(正脸 ~100%),由丢脸缓冲+迟滞吸收;极端场景备胎为
  InsightFace/SCRFD on GPU(已装通,未集成)。
- 快手跟踪物理上限 ~90°/s,头部跟随范围身体 ±22.5°;手扫出扇区只能贴边追。
- 安静坐着不说话时,15s 无互动计时器会触发回中→看见脸又回跟踪的小循环(主线整合时重新设计待命逻辑)。
- daemon 偶发崩溃(exit 116)已定位缓解未根治(CALIBRATION §12);用 `daemon_up.py` 启动即可。

## 主线剧本进度

> 待命(只听唤醒词"小艺")→ 喊它 → 上扬确认 → 视觉主导寻人(SEEK)→ 锁脸+对话
> → 对话中手部互动(逗它/指向)→ 长时间无互动 → 回待命

- ✅ **M1 完成(单人闭环跑通)**:唤醒词「小艺」(单字,sherpa-onnx,叠词经实测否决)+ armed/engaged 待命门控(命中才连,待命零计费)+ 上行两路 + 唤醒确认动作(heard 上扬/fail 下垂/giveup 微沉)+ SEEK 视觉主导寻人(DOA 仅起扫弱提示)。详见 CALIBRATION §14、voice/ARCHITECTURE.md §6。
- ⏭ **M1.5**:engaged 内电视声区分(DOA 方向门控,110°)、多人唤醒切换 + 多脸 sticky 迟滞、深后盲区评估(查 XVF3800 raw/固件)。
- ⏭ **后续**:睡眠词、情绪化动作变异等。
