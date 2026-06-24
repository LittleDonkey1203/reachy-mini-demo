# OpenWolf

@.wolf/OPENWOLF.md

This project uses OpenWolf for context management. Read and follow .wolf/OPENWOLF.md every session. Check .wolf/cerebrum.md before generating code. Check .wolf/anatomy.md before reading files.


# 基本原则

1. 安装依赖时优先用国内镜像
2. Agent 不应该依赖上下文窗口记忆，而应该把记忆写入文件系统。Memory = Files
3. 阅读项目代码时优先用codegraph、graphify之类的工具而不是grep
4. 当用户和agent在讨论方案时，永远期待agent能回答更好的方案是...
5. todo就是todo，定期归档一下已完成的todo，尽量只留待办，不要让todo文档越来越长


## Persistent Project State

每次完成任务后必须更新 PROJECT_STATE.md。

更新内容包括：

1. 已完成事项
2. 当前架构状态
3. 遗留问题
4. 下一步建议

在开始新的规划前：

1. 阅读 PROJECT_STATE.md
2. 理解历史修改
3. 基于当前状态生成计划

不要仅依赖当前对话上下文。PROJECT_STATE.md 是项目真实状态来源, 文件状态优先于对话状态。

如果实现了项目新特性，则需要更新docs/FEATURE_INVENTORY.md，介绍特性如何使用