# 注视感知交互方案 (Gaze-Aware Interaction)

> 目标：让 Reachy Mini Lite 感知"谁在看我"，像小动物一样好奇地回看、扫视、或自主探索。

## 1. 调研结论

### 1.1 候选模型对比

| 模型 | 参数量 | 精度 (MAE) | CPU 速度 | 体积 | 开源 | 评价 |
|------|--------|------------|----------|------|------|------|
| L2CS-Net + MobileNetV2 | ~3.5M | 13.07° (Gaze360) | 待测 | 9.6 MB | [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) | **选定方案**，精度-性能平衡最优 |
| L2CS-Net + MobileOne S0 | ~2M | 12.58° (Gaze360) | 待测 | 4.8 MB | 同上 | 更小但 MobileOne 算子兼容性待验证 |
| FGI-Net | 1.51M | 3.74° (MPIIFaceGaze) | 未知 | ~6 MB | [CZ178/FGI-Net](https://github.com/CZ178/FGI-Net) | 精度最高但 CPU 未验证，arXiv 预印本 |
| OpenFace 3.0 | 29.4M | 2.56° (MPIIGaze) | ~26fps (Threadripper) | ~120 MB | 2025.6 preprint | 多任务一体，但太重 |
| FR-Net | 0.67M | 3.86° (MPIIGaze) | 23ms CPU | 极小 | 无代码 | 不可用 |
| GazeCapsNet | 11.7M | 竞争力 | 20ms (V100) | ~45 MB | 有代码 | GPU benchmark，CPU 未知 |

> **注意**：Gaze360 和 MPIIFaceGaze 精度不可直比。Gaze360 含极端头姿(全方位)，MPIIFaceGaze 仅正面笔记本场景。

### 1.2 Mutual Gaze 检测方案

| 方案 | 精度 | 方法 | 适用场景 |
|------|------|------|----------|
| VGG16 3-class (Frontiers 2024) | 94.3% guided | ETH-XGaze 预训练→微调 | 精度高但 VGG16 太重 |
| 头姿阈值法 | ~85-90% | pitch/yaw 几何估计 | 最轻量，作 L0 预过滤 |
| Gaze vector 阈值法 | 取决于 gaze 模型 | pitch≈0, yaw≈0 | L2CS-Net 输出直接判定 |

### 1.3 社交注视行为框架

无现成开源状态机可用，需自建。参考文献：
- RASA Robot (Springer 2020): Elicited Attention 竞争网络，多线索加权选注视目标
- simple_robot_gaze (GitHub): ROS 优先级仲裁，架构可参考
- Hanifi et al. (Frontiers 2024): iCub 8fps 多级管线验证了可行性

### 1.4 关键调研来源

- Cheng et al., IEEE TPAMI 2024 — gaze estimation 综述 + GazeHub 标准化 benchmark
- yakhyo/gaze-estimation — L2CS-Net 轻量 backbone 扩展 + ONNX 导出
- Prajod et al., Frontiers Robotics AI 2024 — 协作机器人 3-class 注视分类
- OpenFace 3.0, arXiv 2025.6 — 多任务不确定性加权(对比参考)

## 2. 架构设计

### 2.1 三级级联架构

```
┌──────────────────────────────────────────────────────┐
│  SCRFD Face Detection (已有，子进程 DECIMATE=3)       │
│  输出: face_crop, 5-point landmarks, track_id        │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  L0: 头姿预过滤 (≈0.05ms/face)                       │
│  5-point landmarks → 几何头姿估计 (yaw, pitch)        │
│  |yaw| > 45° 或 |pitch| > 35° → NOT_LOOKING          │
│  预计过滤 40-60% 无效帧                               │
└──────────────────┬───────────────────────────────────┘
                   ▼ (仅候选帧)
┌──────────────────────────────────────────────────────┐
│  L1: 时间降频                                         │
│  NOT_LOOKING track: 每 5 帧重检一次                    │
│  LOOKING track: 每帧推理 (保持响应性)                  │
│  新 track: 立即推理一次                                │
└──────────────────┬───────────────────────────────────┘
                   ▼ (需推理的帧)
┌──────────────────────────────────────────────────────┐
│  L2: L2CS-Net MobileNetV2 ONNX (~10-15ms/face)       │
│  输入: 224×224 face crop (从 SCRFD bbox resize)        │
│  输出: gaze_pitch, gaze_yaw (弧度)                    │
│  Mutual gaze: |pitch| < 15° AND |yaw| < 12°          │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  注视行为状态机 (GazeBehaviorFSM)                      │
│  输入: per-track {track_id, mutual_gaze, gaze_dir}    │
│  输出: robot_gaze_target, robot_behavior              │
└──────────────────────────────────────────────────────┘
```

### 2.2 L0 头姿估计算法

利用 SCRFD 已有的 5 点关键点 (left_eye, right_eye, nose, left_mouth, right_mouth)：

```python
def estimate_head_pose_from_5pts(kps5):
    """从 5 点关键点几何估计头姿，无需 solvePnP"""
    le, re, nose, lm, rm = kps5
    
    eye_center = (le + re) / 2
    inter_eye = np.linalg.norm(re - le)
    
    # yaw: 鼻尖相对双眼中心的水平偏移
    yaw_rad = np.arctan2(nose[0] - eye_center[0], inter_eye)
    
    # pitch: 鼻尖相对双眼中心的垂直偏移  
    pitch_rad = np.arctan2(nose[1] - eye_center[1], inter_eye)
    
    return np.degrees(yaw_rad), np.degrees(pitch_rad)
```

### 2.3 L1 时间降频策略

```python
class GazeTrackState:
    last_gaze_result: str  # "LOOKING" / "NOT_LOOKING" / "UNKNOWN"
    frames_since_check: int
    gaze_pitch: float
    gaze_yaw: float
    
    def needs_gaze_inference(self) -> bool:
        if self.last_gaze_result == "UNKNOWN":
            return True  # 新 track，立即推理
        if self.last_gaze_result == "LOOKING":
            return True  # 正在看我，每帧跑
        # NOT_LOOKING: 降频
        return self.frames_since_check >= 5
```

### 2.4 注视行为状态机

```
状态:
  IDLE          — 无人/无人看我，自主探索动画
  CURIOUS_LOOK  — 1人看我，好奇地注视对方
  SCANNING      — 多人看我，视线缓慢扫过每个人
  GLANCING      — 有人在场但没人看我，偶尔瞥一下

转移:
  任意 → CURIOUS_LOOK : len(looking_at_me) == 1
  任意 → SCANNING      : len(looking_at_me) > 1
  任意 → GLANCING      : len(faces) > 0 AND len(looking_at_me) == 0
  任意 → IDLE          : len(faces) == 0, 持续 2s

行为映射:
  CURIOUS_LOOK → 头部持续追踪看我的人(复用已有 TRACKING 头部跟随)
  SCANNING     → 头部在看我的人之间匀速扫视(周期 2-3s/人)
  GLANCING     → 周期性(每 3-5s)瞥一下最近的人脸，其余时间看前方
  IDLE         → 缓慢左右微摇 + 偶尔低头(好奇/无聊动画)
```

### 2.5 与现有管线集成点

```
face_pipeline.py (FaceReIDPipeline)
  └── tracker.update() → tracked faces
  └── per-track identity (已有)
  └── per-track gaze (新增) ← gaze.py
       ├── L0 头姿预过滤 (kps5 → head_yaw/pitch)
       ├── L1 降频控制 (GazeTrackState)
       └── L2 ONNX 推理 (需要时)

d01_realtime_chat.py
  └── vision_result_loop
       └── primary face → 头部跟随 (已有)
       └── gaze_behavior → 头部行为 (新增) ← gaze_behavior.py
```

## 3. 文件规划

| 文件 | 职责 | 新建/修改 |
|------|------|-----------|
| `perception/gaze.py` | L0 头姿 + L1 降频 + L2 ONNX 推理 | 新建 |
| `perception/gaze_behavior.py` | 注视行为状态机 (FSM) | 新建 |
| `perception/face_pipeline.py` | 在 track loop 中调用 gaze 模块 | 修改 |
| `voice/d01_realtime_chat.py` | vision_result_loop 接入 gaze behavior | 修改 |
| `voice/config.py` | 新增 gaze 相关常量 | 修改 |
| `models/l2csnet_mobilenetv2.onnx` | MobileNetV2 ONNX 权重 | 下载 |
| `scripts/benchmark_gaze.py` | CPU benchmark 脚本 | 新建 |

## 4. 实施计划

### Phase 1: 感知层 (1-2 天)

1. 下载 yakhyo/gaze-estimation MobileNetV2 ONNX 权重
2. 实现 `perception/gaze.py`:
   - `HeadPoseFilter`: 5-point → yaw/pitch 几何估计 + 阈值过滤
   - `GazeEstimator`: ONNX Runtime 推理 + per-track 降频状态
   - `GazeResult`: dataclass (track_id, head_yaw, head_pitch, gaze_yaw, gaze_pitch, mutual_gaze)
3. `scripts/benchmark_gaze.py`: macOS Intel CPU 实测 latency
4. 集成到 `face_pipeline.py` track loop

### Phase 2: 决策层 (1 天)

1. 实现 `perception/gaze_behavior.py`:
   - `GazeBehaviorFSM`: 4 态状态机
   - 输出 `GazeCommand`: target_track_id / scan_targets / idle_animation
2. 接入 `d01_realtime_chat.py` vision_result_loop
3. 映射到头部运动(复用已有 head_control)

### Phase 3: 调参验真 (1 天)

1. L0 阈值调参 (yaw/pitch 门限)
2. Mutual gaze 阈值调参 (gaze pitch/yaw 门限)
3. 状态机转移延迟 / 扫视速度
4. 降频步长 vs 响应延迟 tradeoff

### Phase 4: 可选增强

- 微调 MobileNetV2 做 looking-at-me 2-class 分类(需采集数据)
- 结合 DOA 声源方向: 有人说话+看我 → 优先级更高
- 动态降频: 根据 CPU 负载自动调整 L1 步长

## 5. 依赖

```bash
# 仅需 onnxruntime (项目已有)
# 无新增依赖
pip install onnxruntime  # 已安装 (insightface 依赖)
```

ONNX 权重下载:
```bash
# 从 yakhyo/gaze-estimation release 或 Zenodo (DOI: 10.5281/zenodo.14257640)
wget -O models/l2csnet_mobilenetv2.onnx <release_url>
```

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| MobileNetV2 ONNX macOS Intel 推理 >20ms | 总帧率下降 | L1 降频兜底；退守 MobileOne S0 (4.8MB) |
| 5-point 头姿估计精度不够 | L0 漏过/误杀 | 放宽阈值(yaw>55°)，让 L2 兜底 |
| Gaze360 训练集偏差 | 桌面近距离精度差 | Phase 4 微调或换 MPIIFaceGaze 权重 |
| 状态机切换抖动 | 头部运动不自然 | 转移加迟滞(hysteresis) + OneEuroFilter 平滑 |
