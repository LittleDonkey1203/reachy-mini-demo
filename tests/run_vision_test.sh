#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 视觉模型一键测试：交互式选择模式 → 采集/断言/实时验证

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/../.venv/bin/python"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
DIM='\033[2m'; BOLD='\033[1m'; REV='\033[7m'; NC='\033[0m'
info()  { echo -e "${CYAN}▶ $*${NC}"; }
ok()    { echo -e "${GREEN}✅ $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠️  $*${NC}"; }
error() { echo -e "${RED}❌ $*${NC}"; }

VENV_DIR="$PROJECT_ROOT/../.venv"
LOG_DIR="$PROJECT_ROOT/tools/daemon_logs"

[ -f "$PYTHON" ] || { error "找不到 Python: $PYTHON"; echo "请先在项目根目录执行: uv sync"; exit 1; }

# ══════════════════════════════════════════════════════════
#  交互式菜单：方向键上下选择，回车确认
# ══════════════════════════════════════════════════════════
MENU_LABELS=(
    "全流程        采集夹具 → 断言 → 实时 20s"
    "跳过采集      用已有夹具跑断言 + 实时"
    "仅夹具断言    不需要摄像头/daemon"
    "仅实时验证    实时摄像头 20s"
)
MENU_COUNT=${#MENU_LABELS[@]}

select_mode() {
    local sel=0
    local key

    # 隐藏光标
    tput civis 2>/dev/null || true

    # 绘制菜单
    draw_menu() {
        # 移到菜单起始位置（先清除旧的）
        for ((i=0; i<MENU_COUNT; i++)); do
            tput cuu1 2>/dev/null || printf '\033[1A'
        done
        tput cr 2>/dev/null || printf '\r'

        for ((i=0; i<MENU_COUNT; i++)); do
            tput el 2>/dev/null || printf '\033[2K'
            if [ $i -eq $sel ]; then
                echo -e "  ${REV}${BOLD} ▸ ${MENU_LABELS[$i]} ${NC}"
            else
                echo -e "  ${DIM}   ${MENU_LABELS[$i]} ${NC}"
            fi
        done
    }

    # 先打印空行占位，再画一次
    for ((i=0; i<MENU_COUNT; i++)); do echo ""; done
    draw_menu

    # 读键循环
    while true; do
        IFS= read -rsn1 key
        case "$key" in
            $'\x1b')  # ESC 序列（方向键）
                read -rsn2 key
                case "$key" in
                    '[A') [ $sel -gt 0 ] && sel=$((sel - 1)) ;;           # 上
                    '[B') [ $sel -lt $((MENU_COUNT - 1)) ] && sel=$((sel + 1)) ;;  # 下
                esac
                draw_menu
                ;;
            '')  # 回车
                break
                ;;
        esac
    done

    # 恢复光标
    tput cnorm 2>/dev/null || true
    SELECTED=$sel
}

# ── 如果命令行有参数就直接用，没有就交互选择 ──
SKIP_CAPTURE=false
FIXTURE_ONLY=false
LIVE_ONLY=false
LIVE_SECS=20

if [ $# -gt 0 ]; then
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-capture) SKIP_CAPTURE=true; shift ;;
            --fixture-only) FIXTURE_ONLY=true; shift ;;
            --live-only)    LIVE_ONLY=true; [[ ${2:-} =~ ^[0-9]+$ ]] && { LIVE_SECS=$2; shift; }; shift ;;
            -h|--help)
                echo "用法: $0 [--skip-capture] [--fixture-only] [--live-only [秒]]"
                echo "  无参数时进入交互选择菜单"
                exit 0 ;;
            *) error "未知参数: $1"; exit 1 ;;
        esac
    done
else
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║          视觉模型测试 · 选择运行模式               ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${DIM}使用 ↑↓ 方向键切换，回车确认${NC}"
    echo ""

    select_mode

    case $SELECTED in
        0) ;;                          # 全流程，默认值即可
        1) SKIP_CAPTURE=true ;;
        2) FIXTURE_ONLY=true ;;
        3) LIVE_ONLY=true ;;
    esac
    echo ""
    ok "已选择: ${MENU_LABELS[$SELECTED]}"
    echo ""
fi

# ── 环境检查 ──
info "检查环境..."
BACKEND=$("$PYTHON" -c "
try:
    import mediapipe; print('mediapipe')
except Exception:
    print('opencv')
" 2>/dev/null)
ok "Python: $($PYTHON --version 2>&1)  后端: $BACKEND"

FACE_MODEL="$PROJECT_ROOT/vision/models/face_landmarker.task"
HAND_MODEL="$PROJECT_ROOT/vision/models/hand_landmarker.task"
[ -f "$FACE_MODEL" ] && ok "人脸模型: $(du -h "$FACE_MODEL" | cut -f1)" || warn "人脸模型缺失: $FACE_MODEL"
[ -f "$HAND_MODEL" ] && ok "手部模型: $(du -h "$HAND_MODEL" | cut -f1)" || warn "手部模型缺失: $HAND_MODEL"
echo ""

# ── daemon 检查 + 自动启动（需要摄像头的模式才检查）──
NEED_DAEMON=true
$FIXTURE_ONLY && NEED_DAEMON=false  # --fixture-only 不需要硬件

if $NEED_DAEMON; then
    info "检查 daemon..."
    export NO_PROXY="localhost,127.0.0.1,::1"
    export no_proxy="localhost,127.0.0.1,::1"

    DAEMON_OK=false
    MODE=$(curl -s --max-time 2 http://127.0.0.1:8000/api/state/full 2>/dev/null \
           | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['control_mode'])" 2>/dev/null || true)
    if [ "$MODE" = "enabled" ]; then
        ok "daemon 已在线 (control_mode=enabled)"
        DAEMON_OK=true
    else
        warn "daemon 未就绪，正在启动..."

        # 查找串口 (macOS: /dev/cu.usbserial-*)
        SERIAL_PORT=$(ls /dev/cu.usbserial-* 2>/dev/null | head -1 || true)
        if [ -z "$SERIAL_PORT" ]; then
            SERIAL_PORT=$(ls /dev/cu.usbmodem* 2>/dev/null | head -1 || true)
        fi
        if [ -z "$SERIAL_PORT" ]; then
            error "未找到 USB 串口，请确认 Reachy Mini 已通过 USB 连接并开机"
            exit 1
        fi
        info "找到串口: $SERIAL_PORT"

        # 杀残留
        pkill -f reachy-mini-daemon 2>/dev/null || true
        sleep 2

        # 启动 daemon
        DAEMON_BIN="$VENV_DIR/bin/reachy-mini-daemon"
        if [ ! -f "$DAEMON_BIN" ]; then
            error "找不到 daemon: $DAEMON_BIN"
            echo "  请先 uv sync 或手动启动 daemon 后再跑本脚本"
            exit 1
        fi

        mkdir -p "$LOG_DIR"
        export PYTHONUNBUFFERED=1
        export HF_HUB_OFFLINE=1
        nohup "$DAEMON_BIN" \
            -p "$SERIAL_PORT" \
            --localhost-only \
            --log-level INFO \
            >> "$LOG_DIR/daemon.log" 2>&1 &
        DAEMON_PID=$!
        info "daemon 已启动 (PID=$DAEMON_PID), 等待就绪..."

        # 轮询等待，最多 30s
        READY=0
        for i in $(seq 1 30); do
            sleep 1
            MODE=$(curl -s --max-time 1 http://127.0.0.1:8000/api/state/full 2>/dev/null \
                   | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['control_mode'])" 2>/dev/null || true)
            if [ "$MODE" = "enabled" ]; then
                READY=1; break
            fi
            printf "."
        done
        echo ""

        if [ $READY -eq 1 ]; then
            ok "daemon 就绪 (control_mode=enabled)"
            DAEMON_OK=true
        else
            error "daemon 启动超时，请查看日志: $LOG_DIR/daemon.log"
            exit 1
        fi
    fi
    echo ""
fi

# ════════════════════════════════════════════════════════
#  步骤 1：采集夹具
# ════════════════════════════════════════════════════════
if ! $FIXTURE_ONLY && ! $LIVE_ONLY && ! $SKIP_CAPTURE; then
    FIXTURES_DIR="$SCRIPT_DIR/fixtures"
    MANIFEST="$FIXTURES_DIR/manifest.json"
    EXISTING=0
    if [ -f "$MANIFEST" ]; then
        EXISTING=$("$PYTHON" -c "import json; print(sum(1 for e in json.load(open('$MANIFEST')) if __import__('os').path.isfile('$FIXTURES_DIR/'+e['file'])))" 2>/dev/null || echo 0)
    fi

    if [ "$EXISTING" -ge 9 ]; then
        ok "夹具已齐全（${EXISTING} 张），跳过采集"
    else
        info "步骤 1/3：采集夹具图（已有 ${EXISTING}/9 张）"
        echo "  需要连接 Reachy Mini。按提示摆姿势回车拍摄，s 跳过，q 退出。"
        echo ""
        "$PYTHON" "$SCRIPT_DIR/_vision_capture_fixtures.py" || {
            warn "采集中断，继续后续步骤（已采集的夹具仍可用）"
        }
        echo ""
    fi
fi

# ════════════════════════════════════════════════════════
#  步骤 2：夹具断言
# ════════════════════════════════════════════════════════
FIXTURE_EXIT=0
if ! $LIVE_ONLY; then
    info "步骤 2/3：夹具自动断言"
    echo ""
    "$PYTHON" "$SCRIPT_DIR/_vision_model_test.py" && FIXTURE_EXIT=0 || FIXTURE_EXIT=$?
    echo ""
    if [ $FIXTURE_EXIT -eq 0 ]; then
        ok "夹具断言全部通过"
    else
        warn "有用例未通过 (exit=${FIXTURE_EXIT}), 查看 tests/output/ 标注图"
    fi
    echo ""
fi

# ════════════════════════════════════════════════════════
#  步骤 3：实时摄像头验证
# ════════════════════════════════════════════════════════
if ! $FIXTURE_ONLY; then
    info "步骤 3/3：实时摄像头验证（${LIVE_SECS}s，Ctrl+C 提前停止）"
    echo ""
    "$PYTHON" "$SCRIPT_DIR/_vision_model_test.py" --live "$LIVE_SECS" || true
    echo ""
fi

# ════════════════════════════════════════════════════════
#  汇总
# ════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║                    完成                         ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  后端: $(printf '%-40s' "$BACKEND")║"
echo "║  标注图: tests/output/                          ║"
if ! $LIVE_ONLY; then
    if [ $FIXTURE_EXIT -eq 0 ]; then
        echo "║  夹具断言: ✅ 全部通过                         ║"
    else
        echo "║  夹具断言: ❌ 有失败，查看标注图                ║"
    fi
fi
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "跨平台对比：把 tests/fixtures/ 拷到 Windows，跑同样的命令对比通过率。"

exit $FIXTURE_EXIT
