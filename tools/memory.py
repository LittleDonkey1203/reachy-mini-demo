"""记忆工具 — RememberFactTool / ForgetFactTool / ClearMemoryTool / ConfirmClearTool。"""
from __future__ import annotations

import json

from tools.base import Tool, ToolDeps
from voice.state import log


class RememberFactTool(Tool):
    """记住用户告诉你的个人信息。"""

    @property
    def name(self) -> str:
        return "remember_fact"

    @property
    def description(self) -> str:
        return (
            "记住用户告诉你的个人信息。用 key 描述类别，value 描述内容。"
            "例如：'我喜欢猫'→ remember_fact(key='喜欢的动物', value='猫')；"
            "'我是做AI的'→ remember_fact(key='职业', value='AI从业者')；"
            "'我叫小明'→ remember_fact(key='称呼', value='小明', name='小明')。"
            "相同 key 会自动覆盖旧值，不需要额外处理。"
            "注重理解上下文含义，提取有意义的信息分类存储。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "信息类别，如'爱好''职业''喜欢的食物'"},
                "value": {"type": "string", "description": "具体内容，如'打篮球''程序员''火锅'"},
                "name": {"type": "string", "description": "用户的名字（仅在用户自报姓名时传）"},
            },
            "required": ["key", "value"],
        }

    def execute(self, deps: ToolDeps, call_id: str, args: dict) -> str:
        st = deps.st
        with st.lock:
            pid = st.resp_snapshot_pid
        if pid is None:
            return json.dumps(
                {"result": "当前没有识别到用户身份(可能说话人不在画面里),无法存储记忆。"},
                ensure_ascii=False,
            )

        new_name = args.get("name")
        name_accepted = False
        if new_name:
            with st.lock:
                _turn_text = next(
                    (d.get("text", "") for d in reversed(st.display_transcript) if d.get("role") == "user"),
                    "",
                )
            from voice.realtime import try_name_identity

            name_accepted = try_name_identity(
                memory_mgr=deps.memory_mgr,
                identity_store=deps.identity_store,
                face_pipeline=deps.face_pipeline,
                owner_mgr=deps.owner_mgr,
                st=st,
                pid=pid,
                new_name=new_name,
                transcript=_turn_text,
                log_fn=log,
            )
            if not name_accepted:
                _NAME_KEYS = {"称呼", "名字", "姓名", "昵称", "name"}
                k = args.get("key", "").strip()
                if k in _NAME_KEYS:
                    return json.dumps(
                        {"result": f"命名未通过验证,「{new_name}」未被记录。"},
                        ensure_ascii=False,
                    )

        result = deps.memory_mgr.handle_tool_call(pid, "remember_fact", args)
        with st.lock:
            st.identity_injected = False
            st.identity_injected_pid = None

        return json.dumps({"result": result}, ensure_ascii=False)


class ForgetFactTool(Tool):
    """忘掉关于用户的某一条信息。"""

    @property
    def name(self) -> str:
        return "forget_fact"

    @property
    def description(self) -> str:
        return "忘掉关于用户的某一条信息。说关键词即可，如'猫''火锅''工作'。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "要忘掉的信息关键词"},
            },
            "required": ["keyword"],
        }

    def execute(self, deps: ToolDeps, call_id: str, args: dict) -> str:
        st = deps.st
        with st.lock:
            pid = st.resp_snapshot_pid
        if pid is None:
            return json.dumps(
                {"result": "当前没有识别到用户身份(可能说话人不在画面里),无法存储记忆。"},
                ensure_ascii=False,
            )

        result = deps.memory_mgr.handle_tool_call(pid, "forget_fact", args)
        with st.lock:
            st.identity_injected = False
            st.identity_injected_pid = None

        keyword = args.get("keyword", "")
        if "名" in keyword or "name" in keyword.lower():
            deps.memory_mgr.set_name(pid, None)
            if deps.identity_store is not None:
                deps.identity_store.set_name(pid, None)
            with st.lock:
                st.current_person_name = None

        return json.dumps({"result": result}, ensure_ascii=False)


class ClearMemoryTool(Tool):
    """启动安全记忆清除流程。"""

    @property
    def name(self) -> str:
        return "clear_memory"

    @property
    def description(self) -> str:
        return (
            "当用户表达想要清除/忘掉记忆的意图时调用。系统将自动启动安全验证流程。"
            "你只需要判断用户想删除谁的记忆：不传 target_name 表示删自己的；传名字表示删别人的。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target_name": {
                    "type": "string",
                    "description": "要清除记忆的目标人名。不传则清除当前用户自己的记忆。",
                },
            },
            "required": [],
        }

    def execute(self, deps: ToolDeps, call_id: str, args: dict) -> str:
        from memory.safety import handle_clear_memory_intent

        result = handle_clear_memory_intent(
            deps.st, args, deps.conv, deps.identity_store
        )
        return json.dumps({"result": result}, ensure_ascii=False)


class ConfirmClearTool(Tool):
    """二次确认记忆清除。"""

    @property
    def name(self) -> str:
        return "confirm_clear"

    @property
    def description(self) -> str:
        return (
            "仅在系统要求你进行二次确认、且用户已口头明确回答后调用。"
            "用户说'是/确认/删吧'→confirmed=true；"
            "用户说'不/算了/取消'→confirmed=false。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "confirmed": {
                    "type": "boolean",
                    "description": "用户是否明确确认要清除",
                },
            },
            "required": ["confirmed"],
        }

    def execute(self, deps: ToolDeps, call_id: str, args: dict) -> str:
        from memory.safety import handle_confirm_clear

        result = handle_confirm_clear(
            deps.st, args, deps.memory_mgr, deps.identity_store
        )
        return json.dumps({"result": result}, ensure_ascii=False)
