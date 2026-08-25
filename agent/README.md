# PCBA LangGraph Agent（T13）

本模块把已冻结的 RAG、KG 和五个 Tool 组合为可中断、可恢复的确定性诊断图。Qwen 仅负责输入抽取和结果表述；候选保留、Tool 选择、异常判断、优化触发和复验均由代码策略决定。

## 运行

在 `PCB_Agent` 环境中配置 `agent/.env`，然后执行：

```powershell
conda run -n PCB_Agent python agent/scripts/run_agent.py --request request.json
conda run -n PCB_Agent python agent/scripts/run_agent.py --resume resume.json --thread-id THREAD_ID
```

请求和结果契约位于 `agent/schemas/`。缺少输入时返回 `needs_input`；补充数据或将无法提供的字段加入 `unavailable_inputs` 后，使用同一 `thread_id` 恢复。

## 决策边界

- 只支持四类冻结缺陷；未知缺陷先补问，不自由扩展本体。
- Reranker 失败显式降级为 RRF Top-5；整个 Retriever 失败时继续 KG 分支。
- Tool 失败只令对应候选保持 `inconclusive`，不阻断其他候选。
- SPI VTE 尚无批准阈值，因此预测结果保持 `inconclusive`，不会自动优化。
- 回流优化只在预测明确不合格且用户目标为 `diagnose_and_optimize` 时触发；推荐参数必须重新调用预测 Tool，复验不合格即拒绝。
- 任何推荐都只供人工审核，不写入设备。
