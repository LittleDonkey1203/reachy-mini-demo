#!/usr/bin/env bash
# ============================================================
#  小艺 Reachy Mini Lite — macOS 一键启动脚本
#  用法: bash start_mac.sh
#       bash start_mac.sh stop    # 停止所有进程
# ============================================================
set -euo pipefail

# ── 路径 ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"          # reachy-mini-demo/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"         # 项目根目录（含 .venv）
PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="$PROJECT_ROOT/log"
DAEMON_PID="$PROJECT_ROOT/.server.pid"
MAIN_PID="$PROJECT_ROOT/.main.pid"
MAIN_SCRIPT="$SCRIPT_DIR/voice/d01_realtime_chat.py"

# ── 颜色 ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── stop 子命令 ──────────────────────────────────────────────
stop_all() {
  info "停止主程序..."
  [ -f "$MAIN_PID" ] && kill "$(cat "$MAIN_PID")" 2>/dev/null && info "主程序已停止" || warn "主程序未在运行"
  sleep 1
  info "停止 daemon (机器人将进入睡眠)..."
  [ -f "$DAEMON_PID" ] && kill "$(cat "$DAEMON_PID")" 2>/dev/null && info "Daemon 已停止" || warn "Daemon 未在运行"
  info "完成。"
  exit 0
}

[ "${1:-}" = "stop" ] && stop_all

# ════════════════════════════════════════════════════════════
echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   小艺 Reachy Mini Lite — macOS 启动  ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# ── 1. 检查 Python venv ──────────────────────────────────────
info "检查 Python 环境..."
[ -f "$PYTHON" ] || error "找不到 .venv，请先在项目根目录执行: uv sync"

# # ── 2. 检查 DASHSCOPE_API_KEY（NO_VOICE=1 时跳过）───────────────
# if [ "${NO_VOICE:-0}" != "1" ]; then
#   if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
#     echo ""
#     warn "未检测到 DASHSCOPE_API_KEY"
#     echo -n "  请输入阿里云百炼 API Key（sk-...）: "
#     read -r DASHSCOPE_API_KEY
#     [ -z "$DASHSCOPE_API_KEY" ] && error "API Key 不能为空"
#     export DASHSCOPE_API_KEY
#   fi
#   info "DASHSCOPE_API_KEY 已就绪 (${#DASHSCOPE_API_KEY} 位)"
# else
#   info "NO_VOICE=1 — 跳过 API Key 检查（仅视觉测试模式）"
# fi

# ── 3. 找串口 ────────────────────────────────────────────────
info "扫描 USB 串口..."
SERIAL_PORT=""
for port in /dev/cu.usbmodem* /dev/cu.usb*; do
  [ -e "$port" ] && SERIAL_PORT="$port" && break
done

if [ -z "$SERIAL_PORT" ]; then
  error "未找到 USB 串口，请确认 Reachy Mini 已通过 USB 连接并开机"
fi
info "找到串口: $SERIAL_PORT"

# ── 4. 停止残留进程 ──────────────────────────────────────────
if [ -f "$DAEMON_PID" ] && kill -0 "$(cat "$DAEMON_PID")" 2>/dev/null; then
  warn "发现旧 daemon 进程，先停止..."
  kill "$(cat "$DAEMON_PID")" 2>/dev/null; sleep 2
fi
if [ -f "$MAIN_PID" ] && kill -0 "$(cat "$MAIN_PID")" 2>/dev/null; then
  warn "发现旧主程序进程，先停止..."
  kill "$(cat "$MAIN_PID")" 2>/dev/null; sleep 1
fi
# 清理占用调试端口的残留进程（防止 VIS_DEBUG [Errno 48]）
_VIS_PORT="${VIS_DEBUG_PORT:-7654}"
_OLD_PID=$(lsof -ti tcp:"$_VIS_PORT" 2>/dev/null || true)
if [ -n "$_OLD_PID" ]; then
  warn "端口 $_VIS_PORT 被 PID $_OLD_PID 占用，先杀掉..."
  kill "$_OLD_PID" 2>/dev/null || true
  sleep 0.5
fi

# ── 5. 启动 daemon ───────────────────────────────────────────
mkdir -p "$LOG_DIR"
info "启动 daemon (串口: ${SERIAL_PORT})..."
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1          # 禁止 daemon 访问 HuggingFace 网络
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

nohup "$PROJECT_ROOT/.venv/bin/reachy-mini-daemon" \
  -p "$SERIAL_PORT" \
  --localhost-only \
  --log-level INFO \
  >> "$LOG_DIR/daemon.log" 2>&1 &
echo $! > "$DAEMON_PID"
info "Daemon PID: $(cat "${DAEMON_PID}"), 等待上电就绪..."

# 轮询 API，最多等 30 秒
READY=0
for i in $(seq 1 30); do
  sleep 1
  MODE=$(curl -s --max-time 1 http://127.0.0.1:8000/api/state/full 2>/dev/null \
         | python3 -c "import sys,json; print(json.load(sys.stdin)['control_mode'])" 2>/dev/null || true)
  if [ "$MODE" = "enabled" ]; then
    READY=1; break
  fi
  printf "."
done
echo ""

if [ "$READY" -eq 0 ]; then
  error "Daemon 启动超时或 control_mode 未 enabled，请查看 $LOG_DIR/daemon.log"
fi
info "Daemon 就绪，control_mode=enabled ✅"

# ── 6. 启动主程序 ────────────────────────────────────────────
info "启动小艺主程序..."
> "$LOG_DIR/main.log"   # 清空旧日志，防止 grep 命中上次运行的内容
cd "$SCRIPT_DIR/voice"
nohup "$PYTHON" -u d01_realtime_chat.py \
  >> "$LOG_DIR/main.log" 2>&1 &
echo $! > "$MAIN_PID"
cd "$SCRIPT_DIR"
info "主程序 PID: $(cat "$MAIN_PID")"

# 等首条就绪日志（NO_VOICE 模式等视觉就绪，否则等语音就绪）
info "等待程序就绪..."
if [ "${NO_VOICE:-0}" = "1" ]; then
  READY_STR="vision_worker ready"
else
  READY_STR="可以对机器人说话了"
fi
for i in $(seq 1 20); do
  sleep 1
  if grep -q "$READY_STR" "$LOG_DIR/main.log" 2>/dev/null; then
    break
  fi
  printf "."
done
echo ""

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║           小艺已就绪，开始说话吧！      ║"
echo "  ╚══════════════════════════════════════╝"
echo ""
info "实时日志: tail -f $LOG_DIR/main.log"
info "停止程序: bash $0 stop"
if [ "${VIS_DEBUG:-0}" = "1" ]; then
  VIS_PORT="${VIS_DEBUG_PORT:-7654}"
  info "视觉调试流: http://localhost:${VIS_PORT}  (蓝框=人脸 绿/黄框=手 左上=状态)"
fi
echo ""

# ── 7. 实时跟进日志（Ctrl+C 停主程序 + daemon）──────────────────
trap '
  echo ""
  info "收到 Ctrl+C，正在停止..."
  [ -f "$MAIN_PID" ] && kill "$(cat "$MAIN_PID")" 2>/dev/null && info "主程序已停止"
  sleep 1
  [ -f "$DAEMON_PID" ] && kill "$(cat "$DAEMON_PID")" 2>/dev/null && info "Daemon 已停止 (机器人将进入睡眠)"
  exit 0
' INT

# VIS_DEBUG: 后台等日志中出现 Dashboard 行再 open
if [ "${VIS_DEBUG:-0}" = "1" ]; then
  VIS_URL="http://localhost:${VIS_PORT}"
  (
    for _ in $(seq 1 60); do
      sleep 1
      if grep -q "视觉子进程就绪" "$LOG_DIR/main.log" 2>/dev/null; then
        open "$VIS_URL" 2>/dev/null || true
        exit 0
      fi
    done
    # 60s 内未出现就绪日志，兜底等 10s 再打开
    sleep 10
    open "$VIS_URL" 2>/dev/null || true
  ) &
fi

tail -f "$LOG_DIR/main.log"
