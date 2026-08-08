# Agent HTTP Tools

本包提供以下稳定名称：

- `pcba_defect_classification`
- `spi_vte_prediction`
- `spi_parameter_optimization`
- `reflow_profile_prediction`
- `reflow_parameter_optimization`

每个 Tool 提供：

- `name` 和 `description`；
- Pydantic `input_model`；
- 可交给函数调用系统的 `input_schema`；
- 同步 `invoke(arguments)`；
- 统一的 `ToolTransportError`、`ToolAPIError` 和 `ToolContractError`。

当前实现不绑定 LangChain 或 LangGraph。Agent 编排阶段只需把名称、描述、Schema 和 `invoke` 注册到目标框架，无需访问服务内部模型。
