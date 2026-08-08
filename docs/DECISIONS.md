# DECISIONS.md

## D001 Tool 正式目录

- `tool/services/aoi`、`tool/services/spi`、`tool/services/reflow` 是三套模型服务的唯一正式实现。
- `tool/common` 保存跨服务公共契约，`tool/agent_tools` 保存 Agent 侧 HTTP Tool。
- 旧重复实现移入 `archive/legacy_tools`，`数据资料`目录保持不变。

原因：消除重复实现和文档漂移，同时保留可恢复副本。

## D002 服务地址

- AOI：`http://127.0.0.1:8000`
- 回流焊：`http://127.0.0.1:8001`
- SPI/VTE：`http://127.0.0.1:8002`
- 保留现有业务路由和 5 个 Tool 名称，通过 Agent Tool 层隐藏服务路由差异。
- 启动脚本可用 `-Port` 临时覆盖端口，Agent Tool 可用 `PCBA_AOI_BASE_URL`、`PCBA_REFLOW_BASE_URL`、`PCBA_SPI_BASE_URL` 覆盖地址。

原因：避免破坏已确定的 API 路径，并解决端口冲突。

## D003 Agent Tool 形式

5 个 Agent Tool 首先实现为框架无关的 Python HTTP 客户端，使用 Pydantic 输入模型并提供统一超时、连接错误和 API 错误处理；LangGraph 注册适配留到 Agent 编排阶段。

原因：保持模型服务与 Agent 框架解耦。

## D004 统一错误契约

三套服务的所有错误响应统一包含：`success`、`request_id`、API/Tool/模型版本信息、`execution_time_ms`、`data`、`warnings`，以及含 `code`、`message`、`details` 的 `error` 对象。

原因：让 Agent Tool 可以使用同一套错误解析和审计逻辑。

## D005 统一运行环境

三套服务、Agent Tool 和测试全部使用 Conda 环境 `PCB_Agent`；`tool/requirements.txt` 记录通过三模型加载与端到端验收的统一依赖版本，不再为单项服务创建独立虚拟环境。

原因：避免模型服务之间的依赖漂移，并使验收环境可复现。

## D006 GitHub 发布范围

- 正式远端为 `https://github.com/jixinzhou/PCBA_Agent.git`，默认分支为 `main`。
- `数据资料/` 不进入 Git，Python 缓存、运行日志、本地环境和临时文件同样排除。
- 模型、项目文档、Tool 代码、归档内容和 `txt_source/` 纳入版本管理。

原因：按项目发布要求保留可运行代码和知识来源，同时避免上传体量巨大的原始训练数据及本地生成物。
