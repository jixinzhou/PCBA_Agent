# PCBA LangGraph Agent 与本地 Web

本模块把已冻结的 RAG、KG 和五个 Tool 组合为可中断、可恢复的确定性诊断图。Qwen 仅负责输入抽取和结果表述；候选保留、Tool 选择、异常判断、优化触发和复验均由代码策略决定。

## 运行

在 `PCB_Agent` 环境中配置 `agent/.env`，然后执行：

```powershell
conda run -n PCB_Agent python agent/scripts/run_agent.py --request request.json
conda run -n PCB_Agent python agent/scripts/run_agent.py --resume resume.json --thread-id THREAD_ID
```

请求和结果契约位于 `agent/schemas/`。缺少输入时返回 `needs_input`；补充数据或将无法提供的字段加入 `unavailable_inputs` 后，使用同一 `thread_id` 恢复。

## 本地 Web 体验

先配置 `agent/.env` 与 `kg/.env`，确认 Docker Desktop 已启动，然后在项目根目录执行：

```powershell
.\agent\scripts\start_web.ps1
```

脚本使用 Conda 环境 `PCB_Agent`，启动 Neo4j、Qdrant、AOI、SPI、Reflow 和 Agent Web，并自动打开浏览器。默认地址为 `http://127.0.0.1:8080`；端口被占用时会从 `18080` 开始自动寻找下一个可用端口。按 Enter 停止本轮 Web 与模型服务。

页面只保留聊天与图片附件。用户可用自然语言提出识别、原因诊断、工艺优化或解释请求，也可直接补充 SPI/Reflow 参数；Qwen负责语义理解，程序决定Tool路由和缺失项。会话、消息和诊断快照持久化到SQLite，解释型追问复用上一轮证据，不重新执行整条诊断。最终回复为LLM+RAG五段式中文报告，精确推荐参数与复验结果由程序附加。

附图后消息为空或明确询问“什么缺陷”时，程序绕过Qwen直接调用AOI，返回缺陷、置信度和Top-3候选；上传新图片会开启隔离的案例上下文，不继承上一张图的工艺参数。

Web启动时预热BGE与Reranker，因此冷启动约需几十秒；页面发送诊断后使用后台任务和阶段进度轮询，不会因长请求冻结。候选致因、知识证据和执行过程可在Agent回复中展开。

HTTP 接口保持现有 Agent JSON 契约：

```text
POST /api/v1/agent/images
POST /api/v1/agent/invoke
POST /api/v1/agent/resume/{thread_id}
POST /api/v1/conversations
GET  /api/v1/conversations/{conversation_id}
POST /api/v1/conversations/{conversation_id}/message-jobs
GET  /api/v1/conversation-jobs/{job_id}
GET  /health
```

Swagger 位于 `/docs`。上传图片只保存在被 Git 忽略的 `agent/storage/uploads`，支持 PNG/JPEG，默认最大 10 MB。

## 决策边界

- 只支持四类冻结缺陷；未知缺陷先补问，不自由扩展本体。
- Reranker 失败显式降级为 RRF Top-5；整个 Retriever 失败时继续 KG 分支。
- Tool 失败只令对应候选保持 `inconclusive`，不阻断其他候选。
- SPI VTE 小于95%时支持少锡印刷候选，大于105%时支持多锡印刷候选，95%至105%保持`inconclusive`。
- 回流优化只在预测明确不合格且用户目标为 `diagnose_and_optimize` 时触发；推荐参数必须重新调用预测 Tool，复验不合格即拒绝。
- 任何推荐都只供人工审核，不写入设备。
