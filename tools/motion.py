"""动作工具 — 8 个 MotionTool 实例（nod / shake_head / look_* / wiggle / tilt）。"""
from __future__ import annotations

import json

from tools.base import Tool, ToolDeps

_MOTION_DEFS = [
    ("nod", "点头。打招呼、同意、确认、答应请求时使用。"),
    ("shake_head", "摇头。否定、拒绝、不同意、说'不'时使用。"),
    ("look_left", "把头转向左边。转过去后摄像头画面会更新,就能看到左边有什么。"),
    ("look_right", "把头转向右边。转过去后摄像头画面会更新,就能看到右边有什么。"),
    ("look_up", "抬头转向上方。转过去后摄像头画面会更新,就能看到上面有什么。"),
    ("look_down", "低头转向下方。转过去后摄像头画面会更新,就能看到下面有什么。"),
    ("wiggle_antennas", "欢快地摆动头顶天线。表达开心、兴奋、被夸奖、热情时使用。"),
    ("tilt_head", "歪头。表达好奇、疑惑、思考、没听懂时使用。"),
]


class MotionTool(Tool):
    """动作工具 — 放入 motion_q 后立即返回成功（fire-and-forget）。"""

    def __init__(self, tool_name: str, tool_description: str):
        self._name = tool_name
        self._description = tool_description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, deps: ToolDeps, call_id: str, args: dict) -> str:
        deps.motion_q.put({"name": self._name, "call_id": call_id})
        return json.dumps({"success": True, "action": self._name}, ensure_ascii=False)


def make_motion_tools() -> list[MotionTool]:
    """构造全部 8 个动作工具实例。"""
    return [MotionTool(n, d) for n, d in _MOTION_DEFS]
