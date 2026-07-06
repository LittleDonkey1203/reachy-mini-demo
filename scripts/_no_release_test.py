# -*- coding: utf-8 -*-
"""验证缓解补丁:短路 release_media 后,no_media 客户端不再触发 daemon 媒体循环。

补丁(供所有 motion-only 脚本复用):
    from reachy_mini import ReachyMini
    ReachyMini.release_media = lambda self: None   # no_media 时不让 daemon 释放/重建媒体

验证:连接 no_media → 动一下头 → 退出;前后对比 daemon 日志的
"Releasing media" 行数,应不变;机器人动作应正常。
"""

import os
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import numpy as np
from reachy_mini import ReachyMini

# ★ 缓解补丁:no_media 客户端不再触发 daemon release/acquire 循环
ReachyMini.release_media = lambda self: None  # type: ignore[method-assign]

print("=== 补丁验证:no_media 连接(不应触发 daemon 媒体释放)===", flush=True)
with ReachyMini(connection_mode="localhost_only", media_backend="no_media") as mini:
    print("已连接;动头验证控制通路…", flush=True)
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [0, 0, 15], degrees=True).as_matrix()
    mini.goto_target(T, duration=0.6, body_yaw=0.0)
    time.sleep(0.3)
    mini.goto_target(np.eye(4), duration=0.6, body_yaw=0.0)
    print("动作完成,退出(不应触发 Re-acquiring)", flush=True)
print("=== 完成 ===", flush=True)
