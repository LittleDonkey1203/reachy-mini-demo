"""跑一次 SDK 自带的 wake_up,看头能否站正。"""

import sys
import time

import numpy as np

from connect import get_robot

NEUTRAL_HEAD_JOINTS = [0.0, 0.525, -0.669, 0.607, -0.607, 0.669, -0.525]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    mini = get_robot()
    print(">>> 调用 SDK 自带的 wake_up()", flush=True)
    mini.wake_up()

    print("    保持 3 秒,看头有没有站正", flush=True)
    time.sleep(3)

    head_joints, antenna_joints = mini.get_current_joint_positions()
    head_pose = mini.get_current_head_pose()
    print(f"\n    wake_up 完成后 head_joints: {[round(x, 3) for x in head_joints]}", flush=True)
    print(f"    与理想中立差(rad): {np.round(np.array(head_joints) - np.array(NEUTRAL_HEAD_JOINTS), 3).tolist()}", flush=True)
    print(f"    与理想中立差(度): {np.round(np.rad2deg(np.array(head_joints) - np.array(NEUTRAL_HEAD_JOINTS)), 1).tolist()}", flush=True)
    print(f"    天线 (rad): {[round(x, 3) for x in antenna_joints]}", flush=True)
    print(f"\n    head_pose:\n{np.round(head_pose, 3)}", flush=True)


if __name__ == "__main__":
    main()
