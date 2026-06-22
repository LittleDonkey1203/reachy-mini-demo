# tests — 视觉模型测试

验证人脸 / 手 / 手势三个检测模型，并量化**跨平台精度差异**（Windows·macOS-arm64 走
MediaPipe；macOS Intel 退回 OpenCV Haar，手部不可用、人脸检出率偏低，见
`../MACOS_SETUP.md`）。诊断脚本风格：`print` + `exit(0/1)`，不依赖 pytest。

模型由生产模块 `voice/vision_worker.py` 提供（本测试复刻其**生产阈值**，测的是线上精度），
可视化复用 `voice/_hand_model_diag.py` 的标注样式（底部过滤线 + bbox + score/size + 门控）。

## 1. 采集夹具（需连接 Reachy Mini）

```bash
python tests/_vision_capture_fixtures.py          # 交互式拍全套（人脸/各手势/负样本）
python tests/_vision_capture_fixtures.py five ok  # 只补拍指定项
```

按提示摆好姿势回车拍摄，图存入 `tests/fixtures/`，期望标注写入
`tests/fixtures/manifest.json`（已预置全套期望，采集时按文件名合并）。

## 2. 运行测试

```bash
python tests/_vision_model_test.py                       # 夹具模式：断言 + 存标注图 + 汇总，exit 0/1
python tests/_vision_model_test.py --live 20             # 实时摄像头模式：滚动统计 + 标注图（人工查看）
python tests/_vision_model_test.py tests/fixtures/hand_five.jpg   # 临时图：单/多图检测 + 标注
```

标注图输出到 `tests/output/`，文件名含判定结果（`fixture_03_pass_hand_five.jpg`）。

## 3. 跨平台对比

在 Windows / macOS-arm64 / macOS-Intel 各跑一次 `python tests/_vision_model_test.py`
（用**同一批**夹具图），对比末尾汇总行的 `通过率` 与 `tests/output/` 标注图：

- MediaPipe 后端：人脸/手/手势应全 `✅`。
- OpenCV 后端（Intel Mac）：手/手势用例显示 `⏭️ 跳过`（模型不可用），人脸通过率偏低
  （符合 Haar ~25-67% 预期）—— 据此量化差异。

## 隐私与提交范围

夹具图与标注输出含人脸，**不入库**（与 `voice/output/`、`vision/debug/` 同规）。
`.gitignore` 已忽略 `tests/output/` 和 `tests/fixtures/*.jpg`，仅提交脚本、
`manifest.json` 与本说明。每位使用者在本机自行采集夹具。

旧的 `tests/output/vtest_*.jpg`（来自已丢失的旧测试、全黑无效）可直接删除。
