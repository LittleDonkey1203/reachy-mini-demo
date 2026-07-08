"""寻人工具 — FindPersonTool:转头搜索认识的人,报告方位。"""
from __future__ import annotations

import json
from typing import Any, Dict

from tools.base import Tool, ToolDeps
from voice.state import log


class FindPersonTool(Tool):
    """寻找认识的人——机器人转头环顾搜索,找到后告知方位。"""

    @property
    def name(self) -> str:
        return "find_person"

    @property
    def description(self) -> str:
        return (
            "寻找认识的人。机器人会转头环顾四周搜索,找到后告诉用户那人在哪个方向。"
            "当用户想知道某个人在哪、是否在附近时调用。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "要找的人的名字"},
            },
            "required": ["name"],
        }

    def execute(self, deps: ToolDeps, call_id: str, args: dict) -> str | None:
        target_name = (args.get("name") or "").strip()

        # ── 前置校验 ──
        if not target_name:
            return json.dumps({"result": "请告诉我你要找谁。"}, ensure_ascii=False)

        if deps.face_pipeline is None:
            return json.dumps({"result": "视觉系统未就绪,暂时无法找人。"}, ensure_ascii=False)

        # ── 查 gallery ──
        ident = deps.face_pipeline.store.find_by_name(target_name)

        # ── fallback: 查 memory manager(名字可能在记忆里但 gallery 名不同步) ──
        if ident is None and deps.memory_mgr is not None:
            for mi in deps.memory_mgr.list_all():
                if mi.get("name") and target_name in mi["name"]:
                    found_pid = mi["person_id"]
                    ident = deps.face_pipeline.store.identities.get(found_pid)
                    break

        if ident is None or not ident.embeddings:
            return json.dumps(
                {"result": f"我不认识叫{target_name}的人,没法帮你找。"},
                ensure_ascii=False,
            )

        # ── 异步:交给 behavior_loop 驱动搜索 ──
        with deps.st.lock:
            deps.st.seek_person_request = {
                "pid": ident.identity_id,
                "name": ident.name,
                "call_id": call_id,
            }
        log(f"🔍 寻人启动: 找「{target_name}」(gallery: {ident.name}, pid={ident.identity_id[:12]})")
        return None  # 不立即回 tool output;behavior_loop 搜完后写 st.seek_person_result
