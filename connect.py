"""连接 Reachy Mini daemon 的统一封装。

约定:
- 本机有 HTTP(S)_PROXY=127.0.0.1:7897 的代理,Python websockets 会跟着走,导致
  ws://localhost 连不上。必须在 import reachy_mini 之前把 localhost 加进 NO_PROXY。
- 使用 USB Lite 版,daemon 跑在本机 localhost:8000,所以 connection_mode='localhost_only'。
"""

import os

# 必须在 import reachy_mini 之前设置,绕过本机代理
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

from reachy_mini import ReachyMini  # noqa: E402


def get_robot(automatic_body_yaw: bool = True) -> ReachyMini:
    """获取已连接的 ReachyMini 实例。

    Args:
        automatic_body_yaw: 是否让 SDK 自动根据头姿态调整身体 yaw。默认 True,
            转头幅度大时身体会跟着转一点;关掉则身体完全不动,只用头部的 Stewart
            平台。

    Returns:
        已连接 daemon、可立即调用的 ReachyMini 实例。

    Raises:
        ConnectionError: daemon 未启动或代理设置不对(检查 NO_PROXY)。
    """
    return ReachyMini(
        connection_mode="localhost_only",
        automatic_body_yaw=automatic_body_yaw,
    )
