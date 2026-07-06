# -*- coding: utf-8 -*-
"""DAEMON-FIX 复现:no_media 客户端连接/退出循环 → 触发 daemon 媒体释放/重取循环。

假设(reachy_mini.py:288-296 + 181-182):
  no_media connect → daemon release_media(GStreamer→NULL)
  no_media exit    → daemon acquire_media(GstMediaServer.start() 全重建)
多轮循环 + daemon 长时运行 → 原生层崩溃 exit 116。

本脚本:循环 connect(no_media)/exit,每轮后探测 daemon 是否存活;
daemon 死亡即停,报告崩溃轮数。
用法:python tools\\_daemon_crash_repro.py [最大轮数=30] [可选 backend=no_media|local]
"""

import json
import os
import sys
import time
import urllib.request

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
T0 = time.monotonic()


def log(m: str) -> None:
    print(f"[t+{time.monotonic() - T0:7.2f}s] {m}", flush=True)


def daemon_alive() -> bool:
    try:
        with OPENER.open("http://127.0.0.1:8000/api/state/doa", timeout=3.0) as r:
            json.loads(r.read().decode("utf-8"))
        return True
    except Exception:
        return False


def main() -> int:
    max_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    backend = sys.argv[2] if len(sys.argv) > 2 else "no_media"

    from reachy_mini import ReachyMini  # noqa: E402(env 先于 import)

    print(f"=== daemon 崩溃复现:{backend} 客户端循环 ×{max_cycles} ===", flush=True)
    if not daemon_alive():
        log("❌ daemon 不在线,中止")
        return 1

    for i in range(1, max_cycles + 1):
        log(f"—— 第 {i} 轮:connect({backend}) ——")
        try:
            with ReachyMini(
                connection_mode="localhost_only",
                media_backend=backend,
            ) as _mini:
                time.sleep(0.5)  # 保持连接片刻
            log(f"第 {i} 轮:客户端正常退出")
        except Exception as e:
            log(f"⚠ 第 {i} 轮客户端异常:{type(e).__name__}: {e}")
        # 给 daemon 重取留时间,再探活
        time.sleep(2.0)
        if not daemon_alive():
            # 再确认一次(防瞬时忙)
            time.sleep(3.0)
            if not daemon_alive():
                log(f"💥 daemon 在第 {i} 轮后死亡(复现成功)")
                print(f"=== 复现成功:{backend} 循环 {i} 轮后 daemon 崩溃 ===", flush=True)
                return 0
        log(f"第 {i} 轮:daemon 存活 ✓")

    print(f"=== {max_cycles} 轮跑完 daemon 未崩(本轮未复现)===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
