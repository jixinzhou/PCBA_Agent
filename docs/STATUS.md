# STATUS.md

## 当前任务

5 个核心 Tool 的正式封装、统一和端到端验收已完成。

## 已完成

- 建立唯一正式目录：`tool/services/{aoi,spi,reflow}`。
- 将旧重复实现归档到 `archive/legacy_tools`，未修改 `数据资料`。
- 统一 Conda 环境、启动脚本、基础 URL、健康检查和完整错误 Schema。
- 补齐 SPI `/health`，并修正文档中的路径、端口和环境说明。
- 实现 5 个框架无关的 Agent HTTP Tool、Pydantic 输入 Schema 和统一异常类型。
- 固化 `PCB_Agent` 已验证依赖版本到 `tool/requirements.txt`。
- 完成真实 HTTP 端到端验收：3 个健康检查、5 个成功调用、5 个 422 错误契约，以及两个优化结果的再次预测均通过。
- 验收结果位于 `tool/tests/e2e/VALIDATION_REPORT.md` 和 `tool/tests/e2e/artifacts/latest_report.json`。

## 当前状态

三套模型服务和 5 个 Agent Tool 可用于后续 Agent 编排。正式端口为 AOI `8000`、回流焊 `8001`、SPI `8002`，并支持启动参数和环境变量覆盖。

## 下一步

按 `TASKS.yaml` 推进 T09：定义缺陷致因本体与术语体系，为后续 RAG、知识图谱和 LangGraph 编排准备统一概念模型。

## 阻塞问题

- 本机 `8000` 当前被 NeatReader 占用；可关闭该应用，或为 AOI 使用 `-Port` 和 `PCBA_AOI_BASE_URL` 覆盖地址。
- `archive/legacy_tools` 是否永久删除，待用户在后续验收后决定；当前不影响正式实现。
