"""tools — 插件式工具系统。

用法:
    from tools import build_default_registry
    registry = build_default_registry()       # 13 个工具
    specs = registry.specs()                   # list[dict] for update_session
    tool = registry.get("nod")                 # Tool | None
"""
from tools.base import Tool, ToolDeps
from tools.registry import ToolRegistry, build_default_registry

__all__ = ["Tool", "ToolDeps", "ToolRegistry", "build_default_registry"]
