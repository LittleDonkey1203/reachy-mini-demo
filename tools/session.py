"""会话控制工具 — EndSessionTool。"""
from __future__ import annotations

import json

from tools.base import Tool, ToolDeps
from voice.config import BYE_PHRASES


class EndSessionTool(Tool):
    """结束对话，让机器人回到待命。"""

    def __init__(self):
        self._exit_i = 0

    @property
    def name(self) -> str:
        return "end_session"

    @property
    def description(self) -> str:
        return (
            "结束本次对话、让机器人回到待命休息。仅当用户【明确表达要结束对话/让你退下/离开】时才调用,"
            "例如「走吧」「退下」「你先忙」「没事了」「拜拜」「不聊了」「先这样」「就到这」。"
            "⚠️ 注意:「再说吧」「这个先放一边」「等会儿」「待会聊」「先放着」「回头说」等只是话题搁置或语气词,"
            "【不是】结束对话,绝不要因此调用;拿不准时继续对话、不要调。"
        )

    def execute(self, deps: ToolDeps, call_id: str, args: dict) -> str:
        phrase = BYE_PHRASES[self._exit_i % len(BYE_PHRASES)]
        self._exit_i += 1
        with deps.st.lock:
            deps.st.exit_request = True
        return json.dumps(
            {
                "success": True,
                "say": f"对话结束。用中文只说这一句简短告别:「{phrase}」,别追问、别挽留、别加别的。",
            },
            ensure_ascii=False,
        )
