# -*- coding: utf-8 -*-
"""daemon_up:可靠的 Reachy Mini daemon 启动/重启封装(DAEMON-FIX 交付物)。

解决的痛点(见 CALIBRATION §7/§12 坑列表):
  - daemon 不常驻,开工前状态未知(活着?僵死?没起?)
  - "No motors detected" = 电源没通(USB 在 ≠ 电机有电),重试无用,要人工查电源
  - 单电机零星通信错 → 重试一次通常就好
  - 电机 Overload Error 锁存 → 必须断电清,重启 daemon 无用
  - 强杀旧实例后 WASAPI/COM3 释放需要时间,立刻重启易失败

用法:
  python tools\\daemon_up.py            # 确保 daemon 就绪(已活着则什么都不做)
  python tools\\daemon_up.py --restart  # 强制重启(长会话前刷新,避开媒体重取坑)

退出码:0=就绪;2=电源问题(需人工);3=过载锁存(需断电);1=其他失败。
日志存 tools/daemon_logs/(gitignore)。
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# 启动器要自洽:不依赖外部 PYTHONUTF8,否则 emoji 在 GBK 控制台崩(实测踩过)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DAEMON_EXE = r"C:\Users\ldkji\AppData\Local\Reachy Mini Control\.venv\Scripts\reachy-mini-daemon.exe"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daemon_logs")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
READY_TIMEOUT_S = 45.0
KILL_SETTLE_S = 4.0


def log(m: str) -> None:
    print(f"[daemon_up] {m}", flush=True)


def rest_alive(timeout: float = 3.0) -> bool:
    """daemon REST 是否在响应(/api/state/full 不依赖媒体)。"""
    try:
        with OPENER.open("http://127.0.0.1:8000/api/state/full", timeout=timeout) as r:
            json.loads(r.read().decode("utf-8"))
        return True
    except Exception:
        return False


def kill_existing() -> bool:
    """杀掉所有 reachy daemon 进程;返回是否真的杀了。

    坑:daemon 常由 python 托管启动(镜像名=python.exe,不是 reachy-mini-daemon.exe),
    `taskkill /IM reachy-mini-daemon.exe` 杀不到 → 僵尸 daemon 累积、多实例抢相机
    (media_server: gst 'Internal data stream error -5' → 无帧)。故再按命令行清扫一遍。
    """
    r = subprocess.run(
        ["taskkill", "/F", "/IM", "reachy-mini-daemon.exe"],
        capture_output=True, text=True,
    )
    killed = r.returncode == 0
    # 按命令行清扫 python 托管的 daemon(排除 powershell 启动窗口与本进程,避免误伤)
    ps_cmd = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -and $_.CommandLine -match 'reachy-mini-daemon' "
        "-and $_.Name -ne 'powershell.exe' -and $_.ProcessId -ne $PID "
        "} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; 'killed ' + $_.ProcessId }"
    )
    r2 = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True, text=True,
    )
    swept = bool((r2.stdout or "").strip())
    if killed or swept:
        log(f"已强杀旧实例,等 {KILL_SETTLE_S:.0f}s 让 WASAPI/COM3/相机 释放…")
        time.sleep(KILL_SETTLE_S)
    return killed or swept


def start_once() -> tuple[str, str]:
    """启动一次 daemon,等到就绪或失败。

    Returns:
        (verdict, log_path)
        verdict ∈ {"ready", "no_power", "motor_comm", "overload", "timeout", "died"}
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"daemon_{time.strftime('%Y%m%d_%H%M%S')}.log")
    lf = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [DAEMON_EXE],
        stdout=lf, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # 独立于本脚本存活
    )
    log(f"daemon 已启动(pid={proc.pid}),日志 {log_path}")

    deadline = time.monotonic() + READY_TIMEOUT_S
    started = False
    while time.monotonic() < deadline:
        time.sleep(1.5)
        if proc.poll() is not None:
            lf.flush()
            return ("died", log_path)
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            content = ""
        if "No motors detected" in content:
            return ("no_power", log_path)
        if "Daemon started successfully" in content:
            started = True
        if started and rest_alive():
            # 就绪后再看 3s 有没有过载刷屏
            time.sleep(3.0)
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()
            if tail.count("Overload Error") >= 3:
                return ("overload", log_path)
            if "Failed to start daemon" in tail:
                return ("motor_comm", log_path)
            return ("ready", log_path)
        if "Failed to start daemon" in content:
            return ("motor_comm", log_path)
    return ("timeout", log_path)


def ensure_daemon(force_restart: bool = False) -> int:
    if rest_alive():
        if not force_restart:
            log("✅ daemon 已在线(--restart 可强制刷新)")
            return 0
        log("daemon 在线,按要求强制重启…")
        kill_existing()
    else:
        log("daemon 不在线,清理残留后启动…")
        kill_existing()

    for attempt in (1, 2):
        verdict, log_path = start_once()
        if verdict == "ready":
            log("✅ daemon 就绪(REST 探活通过,无电机错误)")
            return 0
        if verdict == "no_power":
            log("❌ 9 电机全失联 = 机器人电源没通(USB 在 ≠ 电机有电)。"
                "请检查电源适配器/开关后重跑。重试无用,不重试。")
            return 2
        if verdict == "overload":
            log("❌ 电机 Overload Error 锁存(常见左天线)。"
                "请断电 → 检查天线无卡滞 → 上电 → 重跑。重启 daemon 无用。")
            kill_existing()
            return 3
        log(f"⚠ 第 {attempt} 次启动失败(verdict={verdict},日志 {log_path}),"
            + ("重试一次…" if attempt == 1 else "放弃"))
        kill_existing()
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="确保 Reachy Mini daemon 就绪")
    ap.add_argument("--restart", action="store_true", help="即使在线也强制重启")
    args = ap.parse_args()
    sys.exit(ensure_daemon(force_restart=args.restart))
