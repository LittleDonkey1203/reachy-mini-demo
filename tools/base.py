"""Tool ABC 和 ToolDeps — 所有 LLM 可调用工具的基类。"""
from __future__ import annotations

import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from memory.manager import MemoryManager
    from voice.state import State

_NOPARAM: Dict[str, Any] = {"type": "object", "properties": {}}


@dataclass
class ToolDeps:
    """运行时依赖，每次工具调用时由分发层构建。"""

    st: State
    conv: Any  # OmniRealtimeConversation (duck-typed)
    motion_q: queue.Queue
    memory_mgr: Any | None = None
    owner_mgr: Any | None = None
    id_recognizer: Any | None = None
    face_pipeline: Any | None = None


class Tool(ABC):
    """LLM 可调用工具的基类。

    子类必须实现 name / description / execute。
    parameters 默认无参数，有参数的工具覆盖即可。
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def parameters(self) -> Dict[str, Any]:
        return _NOPARAM

    def spec(self) -> Dict[str, Any]:
        """生成 Qwen function-calling 所需的工具规格 dict。"""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @abstractmethod
    def execute(self, deps: ToolDeps, call_id: str, args: dict) -> str | None:
        """执行工具。

        返回值:
            str — JSON 字符串，调用方统一 create_item + _record_tool_output
            None — 工具自行处理输出（预留给异步工具如 snapshot）
        """
        ...
