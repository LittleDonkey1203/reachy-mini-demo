# -*- coding: utf-8 -*-
"""记忆安全删除工作流 — 删除前的身份验证 + 二次确认。"""

import time

from voice.state import State, log


def handle_clear_memory_intent(st: State, args: dict, conv,
                               id_recognizer) -> str:
    """模型调用 clear_memory → 仅做意图分类，启动验证工作流。"""
    with st.lock:
        pid = st.current_person_id
        existing_wf = st.clear_workflow
    if pid is None:
        return "当前没有识别到用户身份,无法操作。"
    if existing_wf is not None:
        return "已有一个删除流程在进行中,请等待完成或超时。"
    target_name = args.get("target_name")
    target_pid = pid
    if target_name and id_recognizer is not None:
        found = id_recognizer.db.find_by_name(target_name)
        if found is None:
            return f"没有找到叫「{target_name}」的人。"
        target_pid = found
    with st.lock:
        st.clear_workflow = {
            "phase": "verifying",
            "actor_pid": pid,
            "target_pid": target_pid,
            "target_name": target_name,
            "started_at": time.monotonic(),
            "verified_at": None,
            "stable_count": 0,
        }
        st.clear_lock = True
    target_desc = target_name or "你"
    try:
        conv.create_item({
            "type": "message", "role": "system",
            "content": [{"type": "input_text",
                         "text": f"安全验证流程已启动(目标:{target_desc})。请告诉用户：'为了安全,请正面看着我保持几秒,我需要确认你的身份。'然后等待系统下一步指示。"}],
        })
        conv.create_response()
    except Exception as e:
        log(f"⚠ 注入验证提示失败: {e}")
    return "⏳ 安全验证已启动,正在确认身份。"


def handle_confirm_clear(st: State, args: dict,
                         memory_mgr, id_recognizer) -> str:
    """模型调用 confirm_clear → 验证工作流状态后执行备份+删除。"""
    confirmed = args.get("confirmed", False)
    with st.lock:
        wf = st.clear_workflow
        cur_pid = st.current_person_id
    if wf is None or wf.get("phase") != "confirming":
        return "当前没有待确认的删除流程。"
    if not confirmed:
        with st.lock:
            st.clear_workflow = None
            st.clear_lock = False
        log("🔒 用户取消删除")
        return "好的,已取消删除。"
    if cur_pid != wf["actor_pid"]:
        with st.lock:
            st.clear_workflow = None
            st.clear_lock = False
        log("🔒 身份变化,取消删除")
        return "身份验证失败(面前的人已变化),已取消。"
    actor_pid = wf["actor_pid"]
    target_pid = wf["target_pid"]
    backup_face = None
    backup_mem = None
    if id_recognizer is not None:
        backup_face = id_recognizer.db.backup_person(target_pid)
    if memory_mgr is not None:
        backup_mem = memory_mgr.backup_person(target_pid)
    log(f"🔒 备份完成: face={backup_face}, mem={backup_mem}")
    result = "删除失败。"
    if memory_mgr is not None:
        result = memory_mgr.clear_all(target_pid, confirmed=True,
                                      actor_pid=actor_pid)
    if id_recognizer is not None:
        id_recognizer.db.clear_person(target_pid)
    if target_pid == cur_pid:
        with st.lock:
            st.current_person_id = None
            st.current_person_name = None
            st.current_is_owner = False
            st.identity_injected = False
            st.identity_injected_pid = None
    with st.lock:
        st.clear_workflow = None
        st.clear_lock = False
    log(f"🔒 删除完成: {result}")
    return result


def inject_clear_msg(conv, text: str):
    """注入系统消息并触发模型回应。"""
    try:
        conv.create_item({
            "type": "message", "role": "system",
            "content": [{"type": "input_text", "text": text}],
        })
        conv.create_response()
    except Exception as e:
        log(f"⚠ 注入 clear msg 失败: {e}")
