# T13 LangGraph 编排验证

验证日期：2026-08-25

## 结论

T13 通过。已实现结构化请求/结果契约、Qwen OpenAI 兼容适配器、确定性条件图、RAG/KG/Tool 适配、SQLite checkpoint、`thread_id` 中断恢复、CLI 和诊断轨迹。

## 自动化验证

- Agent 单元/分支测试：11/11 通过。
- Agent JSON Schema：2/2 为合法 Draft 2020-12 Schema。
- KG 回归测试：32/32 通过。
- Reranker 失败可显式降级到 RRF Top-5；Retriever 全失败可继续 KG。
- SPI VTE 无批准阈值时固定为 `inconclusive`。
- 回流优化建议只有在再次预测合格后才 `accepted`，复验不合格时为 `rejected`。

## 真实组件验证

- 当前 Qdrant、本地 BGE-M3 和 BGE Reranker 完成一次真实查询：RRF Top-20 → Reranker → `w=0.73` 排名融合 → Top-5，未降级。
- 临时启动 AOI、回流和 SPI 服务后，3 项健康检查、5 个 Tool 正常调用、5 个参数错误契约、3 个 HTTP 错误契约以及 2 个优化后预测复验均通过，随后已停止临时服务。
- 使用真实 RAG、真实 Neo4j KG、真实回流预测 Tool 和 SQLite checkpoint 跑通 `shifted_component`：首轮因人工检查缺失返回 `needs_input`，同一线程恢复后返回 `completed`；保留两条候选路径，回流路径被 Tool 判为 `contradicted`，未触发无关优化。

## 运行限制

- T13 初次验收时尚未配置 Qwen；后续已通过真实API验证结构化抽取，并在T14样例中完成真实补问与报告调用。固定模型名、严格JSON Schema和关闭thinking的请求契约均保持不变。
- Agent 不写设备；SPI 阈值仍待项目方批准，当前不会自动优化 SPI。
