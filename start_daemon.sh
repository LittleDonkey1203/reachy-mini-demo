#!/usr/bin/env bash
# ============================================================
#  小艺 Reachy Mini Lite — daemon 启动脚本
#  用法: bash start_daemon.sh          # 启动
#       bash start_daemon.sh stop     # 停止
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/log"
DAEMON_PID="$PROJECT_ROOT/.server.pid"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── stop ────────────────────────────────────────────────────
if [ "${1:-}" = "stop" ]; then
  if [ -f "$DAEMON_PID" ] && kill -0 "$(cat "$DAEMON_PID")" 2>/dev/null; then
    kill "$(cat "$DAEMON_PID")"
    info "Daemon 已停止 (机器人将进入睡眠)"
  else
    warn "Daemon 未在运行"
  fi
  exit 0
fi

# ── 找串口 ──────────────────────────────────────────────────
info "扫描 USB 串口..."
SERIAL_PORT=""
for port in /dev/cu.usbmodem* /dev/cu.usb*; do
  [ -e "$port" ] && SERIAL_PORT="$port" && break
done
[ -z "$SERIAL_PORT" ] && error "未找到 USB 串口，请确认 Reachy Mini 已连接并开机"
info "串口: ${SERIAL_PORT}"

# ── 清理旧进程 ──────────────────────────────────────────────
if [ -f "$DAEMON_PID" ] && kill -0 "$(cat "$DAEMON_PID")" 2>/dev/null; then
  warn "发现旧 daemon，先停止..."
  kill "$(cat "$DAEMON_PID")" 2>/dev/null; sleep 2
fi

# ── 启动 ────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

info "启动 daemon..."
"$PROJECT_ROOT/.venv/bin/reachy-mini-daemon" \
  -p "$SERIAL_PORT" \
  --fastapi-host 127.0.0.1 \
  --log-level INFO \
  >> "$LOG_DIR/daemon.log" 2>&1 &
DAEMON_CHILD=$!
echo $DAEMON_CHILD > "$DAEMON_PID"
info "PID: ${DAEMON_CHILD}"

# ── 等待就绪 ────────────────────────────────────────────────
info "等待 control_mode=enabled..."
for i in $(seq 1 30); do
  sleep 1
  MODE=$(curl -s --max-time 1 http://127.0.0.1:8000/api/state/full 2>/dev/null \
         | python3 -c "import sys,json; print(json.load(sys.stdin)['control_mode'])" 2>/dev/null || true)
  if [ "$MODE" = "enabled" ]; then
    echo ""
    info "Daemon 就绪 ✅  (http://127.0.0.1:8000)"
    break
  fi
  printf "."
  if [ "$i" -eq 30 ]; then
    echo ""
    error "启动超时，请查看: $LOG_DIR/daemon.log"
  fi
done

# ── hold: 前台等待，Ctrl+C 时 kill daemon ──────────────────
_cleanup() {
  echo ""
  info "正在停止 daemon (机器人将进入睡眠)..."
  kill "$DAEMON_CHILD" 2>/dev/null && wait "$DAEMON_CHILD" 2>/dev/null
  rm -f "$DAEMON_PID"
  info "Daemon 已停止"
}
trap '_cleanup; exit 0' INT TERM
trap '_cleanup' EXIT

info "Daemon 运行中，Ctrl+C 停止"
wait "$DAEMON_CHILD"
