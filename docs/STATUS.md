# STATUS.md

## 当前任务

T14 已用桥连诊断和元件偏移优化两个真实样例跑通核心闭环并获用户确认；当前进入 T15 系统评测阶段。

## 当前可用能力

- 三套正式模型服务位于 `tool/services/{aoi,spi,reflow}`，端口默认为 AOI `8000`、回流焊 `8001`、SPI `8002`。
- 5 个框架无关 Agent HTTP Tool 已实现：`pcba_defect_classification`、`spi_vte_prediction`、`spi_parameter_optimization`、`reflow_profile_prediction`、`reflow_parameter_optimization`。
- T09 V1.1 缺陷致因本体已冻结，权威文件为 `ontology/pcba_defect_causality.v1.1.yaml`。
- RAG 已完成 PDF 解析、Chunk、Metadata、Embedding、Qdrant、Dense/Sparse Retriever、Hybrid RRF 与 Retrieval Evaluation 基线。
- 当前最新知识库为 V0.3：7 个来源、216 页、245 条 Chunk，Qdrant Collection 为 `pcba_industrial_knowledge_v0_3`。
- T11.1 KG 静态契约已建立：`CausalHypothesis` 图模型、T09确定性映射、查询Schema、Neo4j约束、三个示例和13项契约测试均已完成。
- T11.2 运行基线为 Neo4j Community `2026.07.1` 与 Driver `6.2.0`；核心图为28节点、40关系，两轮导入图指纹一致，容器`pcba-neo4j`当前健康运行。
- T11.3 可按四类缺陷或单个`relationship_id`返回排序后的候选路径、缺失观测量与下一验证动作；32项KG测试通过，五条路径已从真实Neo4j查询验证。
- T13 LangGraph Agent 已实现结构化输入/输出、Qwen适配、条件路由、RAG/KG/Tool接入、SQLite中断恢复和诊断轨迹；11项Agent测试与真实`shifted_component`链路通过。
- Qwen真实API已通过结构化抽取与闭环报告调用；T14回流异常样例PWI由`105.46`降至`24.23`并经独立预测复验合格，精确参数由程序生成而非LLM复述。

## T10 最终结论

- 权威说明为 `rag/evaluation/README.md`，冻结参数及11个权威文件哈希为 `rag/evaluation/FROZEN_MANIFEST.json`。
- 正式范围为Retriever主评测16题、多证据3题、Agent/KG 2题和No-answer 3题；主评测候选池共320条且全部完成标注。
- 16题中14题含43条rel=2；Q019、Q024无直接Gold并单列，不强行计入排序主指标。
- 冻结排名方式为RRF Top-20→BGE Reranker→固定排名融合`w=0.73`→Top-5；Macro/Micro Recall@5为`0.762188/0.627907`（27/43），直接答案覆盖14/14。
- Recall@5 `0.762188`、MRR@10 `0.585714` 未达到原定 `0.85/0.70` 门槛；在当前小规模资料与Gold质量下继续调参收益有限，已由用户接受指标例外，不再阻塞后续系统集成。
- Agent 阶段默认采用冻结的 `w=0.73` RRF+Reranker 证据排序方案；现有框架无关 Retriever 保留等权 RRF 基线，接入与降级策略在 T13 完成。

## 下一步

1. T15建立覆盖四类缺陷、图片输入、Tool失败和证据不足的系统评测集。
2. 评测Tool选择、缺失数据识别、引用完整性、因果一致性、复验率和拒答正确性。

## 阻塞与注意事项

- 本机 `8000` 可能被 NeatReader 占用；可关闭该应用，或为 AOI 使用 `-Port` 和 `PCBA_AOI_BASE_URL` 覆盖地址。
- `archive/legacy_tools` 是否永久删除，待用户后续决定；当前不影响正式实现。
- `txt_source/` 包含 IPC/GJB 等参考 PDF；公开仓库分发前需要项目方确认授权。
- T10.10冻结文件哈希变化时必须显式重冻结并全量重算，不能直接覆盖权威数据。
- Neo4j 可视化或静态节点展示不算 T11 完成；必须能被后续 Agent 查询并驱动条件分支。
- 本地Neo4j凭据只保存在被Git忽略的`kg/.env`，仓库只保存占位模板；图数据卷可由T09本体重建。
- Qwen凭据保存在被Git忽略的`agent/.env`；真实调用已验证，未配置时确定性路由仍可运行但文本抽取和报告生成会降级。
- 启动任务时只读取本文件的当前摘要；需要追溯阶段细节时再读取 `docs/archive/`。
