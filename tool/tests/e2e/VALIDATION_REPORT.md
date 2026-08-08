# 5 Tool 端到端验收报告

- 日期：2026-08-08
- 环境：Conda `PCB_Agent`
- 方式：三个真实 Uvicorn 服务 + Agent HTTP Tool
- 机器可读结果：`artifacts/latest_report.json`

## 环境版本

`FastAPI 0.141.1`、`Pydantic 2.13.4`、`httpx 0.28.1`、`scikit-learn 1.9.0`、`PyTorch 2.5.1+cu121`、`LightGBM 4.6.0`。

## 验收结果

| 检查项 | 结果 | HTTP 墙钟时间 |
|---|---:|---:|
| AOI 健康检查与模型加载 | 通过 | — |
| 回流焊健康检查与模型加载 | 通过 | — |
| SPI 健康检查与模型加载 | 通过 | — |
| `pcba_defect_classification` | 通过 | 795 ms |
| `spi_vte_prediction` | 通过 | 80 ms |
| `spi_parameter_optimization` | 通过 | 8198 ms |
| `reflow_profile_prediction` | 通过 | 70 ms |
| `reflow_parameter_optimization` | 通过 | 1278 ms |
| 5 个 Tool 的 HTTP 422 统一错误 Schema | 全部通过 | — |
| 3 个服务的 HTTP 404 统一错误 Schema | 全部通过 | — |
| SPI 优化后重新预测 | 通过 | — |
| 回流焊优化后重新预测 | 通过 | — |

AOI 使用 `数据资料/AOI图像识别数据/API_test/short (1).png`。预测与优化参数采用各服务 README 中的正式示例。

## 端口说明

正式端口约定为 AOI `8000`、回流焊 `8001`、SPI `8002`。本机 `8000` 已由 NeatReader 占用，因此本次验收仅将 AOI 临时启动在 `18000`；测试的是同一应用和接口实现。启动脚本和 Agent Tool 均支持地址覆盖。

## 结论

三套服务和 5 个 Agent Tool 满足当前接口契约，可进入知识图谱、RAG 和 Agent 编排阶段。测试结束后，本次启动的三个服务进程均已停止。
