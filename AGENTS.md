# AGENTS.md

## 开发环境

* Conda 环境：`PCB_Agent`
* 所有 Python、模型、API、RAG、KG、Agent 相关操作均使用该环境。
* 不随意创建或切换其他环境。

## 开始任务前

优先读取：

1. `docs/PROJECT.md`：项目目标、边界和整体安排
2. `docs/STATUS.md`：当前总体状态和下一步
3. `docs/TASKS.yaml`：具体任务、状态和依赖
4. `docs/DECISIONS.md`：已确定的重要技术决策

不要重复询问文档中已经明确的信息。

## 开发规则

* 按 `STATUS.md` 和 `TASKS.yaml` 推进当前任务。
* 不随意扩大项目范围。
* 已确定的 API、JSON 协议和 Tool 名称不要随意修改。
* Agent 通过 Tool/API 调用模型，不直接耦合模型内部代码。
* 修改代码后进行最小可运行验证。
* 不做与当前任务无关的大规模重构。

## 任务结束后

检查并同步：

* `STATUS.md`：更新总体进度、当前重点、下一步和阻塞问题。
* `TASKS.yaml`：更新任务状态；新增任务时记录任务、状态和依赖。
* `DECISIONS.md`：仅记录新的长期技术决策及原因。
* `PROJECT.md`：仅在项目目标、边界或整体安排变化时修改。

文档保持简洁，只记录后续开发真正需要的信息。
