"""ToolRegistry — 工具注册表 + 默认注册工厂。"""
from __future__ import annotations

from tools.base import Tool


class ToolRegistry:
    """按名称注册、查找工具，生成 Qwen function-calling specs。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        """返回所有工具的 spec dict 列表，保持注册顺序。"""
        return [t.spec() for t in self._tools.values()]

    def exclude(self, *names: str) -> ToolRegistry:
        """返回新 registry，排除指定名称的工具。"""
        reg = ToolRegistry()
        for n, t in self._tools.items():
            if n not in names:
                reg._tools[n] = t
        return reg

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def build_default_registry() -> ToolRegistry:
    """构造默认工具注册表（13 个工具，按原 BASE_TOOLS + QWEN_TOOLS 顺序）。"""
    from tools.memory import (
        ClearMemoryTool,
        ConfirmClearTool,
        ForgetFactTool,
        RememberFactTool,
    )
    from tools.motion import TurnBodyTool, make_motion_tools
    from tools.seek import FindPersonTool
    from tools.session import EndSessionTool

    reg = ToolRegistry()
    for t in make_motion_tools():
        reg.register(t)
    reg.register(TurnBodyTool())
    reg.register(FindPersonTool())
    reg.register(EndSessionTool())
    reg.register(RememberFactTool())
    reg.register(ClearMemoryTool())
    reg.register(ConfirmClearTool())
    reg.register(ForgetFactTool())
    return reg
