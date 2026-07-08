"""动作工具 — 8 个 MotionTool 实例（nod / shake_head / look_* / wiggle / tilt）+ TurnBodyTool。"""
from __future__ import annotations

import json
from typing import Any, Dict

from tools.base import Tool, ToolDeps

_MOTION_DEFS = [
    ("nod", "点头。打招呼、同意、确认、答应请求时使用。"),
    ("shake_head", "摇头。否定、拒绝、不同意、说'不'时使用。"),
    ("look_left", "看向左边——小幅偏头扫视左侧,摄像头画面随之更新。仅用于想看某个方向有什么,不是转身。"),
    ("look_right", "看向右边——小幅偏头扫视右侧,摄像头画面随之更新。仅用于想看某个方向有什么,不是转身。"),
    ("look_up", "看向上方——小幅抬头扫视上方,摄像头画面随之更新。仅用于想看某个方向有什么。"),
    ("look_down", "看向下方——小幅低头扫视下方,摄像头画面随之更新。仅用于想看某个方向有什么。"),
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


# ── turn_body 工具 ──

_TURN_BODY_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["left", "right", "center"],
            "description": "转向方向:left=左转,right=右转,center=回正面对用户",
        },
        "angle": {
            "type": "number",
            "description": "转动角度(度),10~90,默认45。direction=center时忽略",
        },
    },
    "required": ["direction"],
}

_DEFAULT_ANGLE = 45.0
_MIN_ANGLE = 10.0


class TurnBodyTool(Tool):
    """转动身体朝向 — 修改底盘 yaw，带方向+角度参数。"""

    @property
    def name(self) -> str:
        return "turn_body"

    @property
    def description(self) -> str:
        return (
            "转动身体(底盘)朝向指定方向。"
            "用户说'向左转''向右转''转过去''面朝那边''转个身'等要求改变朝向时使用。"
            "用户说'转回来''面对我''转过来'时用 direction=center 回正并恢复面对用户。"
            "注意:这是转身,不是看某个方向——看某方向用 look_left/right。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return _TURN_BODY_PARAMS

    def execute(self, deps: ToolDeps, call_id: str, args: dict) -> str:
        """放入 motion_q，附带 args 供 motion_loop 解析执行。"""
        # 参数兜底：direction
        direction = str(args.get("direction", "left")).lower().strip()
        if direction not in ("left", "right", "center"):
            direction = "left"
        # center 不需要 angle 处理
        if direction == "center":
            deps.motion_q.put({
                "name": "turn_body",
                "call_id": call_id,
                "args": {"direction": "center"},
            })
            return json.dumps(
                {"success": True, "action": "turn_body", "direction": "center"},
                ensure_ascii=False,
            )
        # 参数兜底：angle
        try:
            angle = float(args.get("angle", _DEFAULT_ANGLE))
        except (TypeError, ValueError):
            angle = _DEFAULT_ANGLE
        if angle <= 0:
            angle = _DEFAULT_ANGLE
        angle = max(_MIN_ANGLE, min(angle, 90.0))  # clamp [10, 90]

        deps.motion_q.put({
            "name": "turn_body",
            "call_id": call_id,
            "args": {"direction": direction, "angle": angle},
        })
        return json.dumps(
            {"success": True, "action": "turn_body",
             "direction": direction, "angle": angle},
            ensure_ascii=False,
        )
