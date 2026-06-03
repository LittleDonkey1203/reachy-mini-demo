# -*- coding: utf-8 -*-
"""第 1 项:连接体检 — 验证与 daemon 的控制通道是否正常。

只测控制平面(连接 / 版本 / 电机 / pose / IMU),不初始化摄像头和麦克风,
那些通道由第 3/4/5 项单独验证。
"""

# 必须在 import reachy_mini 之前修好本地代理穿透问题(见 env-local-proxy 记录)
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

from importlib.metadata import version, PackageNotFoundError
import numpy as np
from reachy_mini import ReachyMini


def main() -> bool:
    print("=== 第1项:连接体检 ===")
    ok = True

    # 用 no_media:只验证控制通道,不抢摄像头/麦克风。退出时自动把媒体硬件还给 daemon。
    with ReachyMini(connection_mode="localhost_only", media_backend="no_media") as mini:
        # 1) 连接模式
        print(f"连接模式            : {mini.connection_mode}")

        # 2) SDK 版本 + daemon 版本(确认是否一致)
        try:
            sdk_ver = version("reachy_mini")
        except PackageNotFoundError:
            sdk_ver = "未知"
        status = mini.client.get_status()
        print(f"SDK 版本            : {sdk_ver}")
        print(f"daemon 版本         : {status.version}")
        print(f"机器人名称          : {status.robot_name}")
        print(f"daemon 状态         : {status.state.value if hasattr(status.state, 'value') else status.state}")
        print(f"无线版?             : {status.wireless_version}  (Lite 应为 False)")
        print(f"硬件 ID             : {status.hardware_id}")
        print(f"摄像头规格名        : {status.camera_specs_name or '(未报告)'}")
        if sdk_ver != status.version:
            print(f"⚠ 注意:SDK 与 daemon 版本不一致")

        # 3) 电机数量(7 头部 stewart/yaw + 2 天线 = 9,对应 Dynamixel ID 10-18)
        head_joints, antenna_joints = mini.get_current_joint_positions()
        n_motors = len(head_joints) + len(antenna_joints)
        print(f"检测到电机数量      : {n_motors}  (头部 {len(head_joints)} + 天线 {len(antenna_joints)})")
        np.set_printoptions(precision=4, suppress=True)
        print(f"当前头部关节(rad)  : {np.array(head_joints)}")
        print(f"当前天线关节(rad)  : {np.array(antenna_joints)}")

        # 4) 当前头部 pose(4x4 齐次矩阵)
        pose = mini.get_current_head_pose()
        print("当前头部 pose (4x4) :")
        print(np.array2string(np.array(pose), precision=4, suppress_small=True, prefix="                      "))

        # 5) IMU(Lite 无 IMU,预期返回 None)
        imu = mini.imu
        if imu is None:
            print("IMU                 : None — Lite 版无 IMU,按预期跳过")
        else:
            print(f"IMU                 : {imu}")

        # 6) 电池:Lite 为 USB 供电,SDK 无电池传感器接口 → 跳过
        print("电池                : Lite 为 USB 供电,无电池传感器,跳过")

        # 基本健全性检查
        if n_motors != 9:
            print(f"⚠ 电机数量异常:期望 9,实际 {n_motors}")
            ok = False
        if str(status.state).lower().endswith("running") is False and str(status.state) != "DaemonState.RUNNING":
            # 仅提示,不一定算失败
            pass

    print("=== 通过 ===" if ok else "=== 失败 ===")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
