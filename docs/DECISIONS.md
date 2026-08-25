# DECISIONS.md

本文件只保留仍约束后续开发的长期决策；历史展开说明已归档到 `docs/archive/DECISIONS_HISTORY_2026-08-19.md`。

## 长期决策

- D001：`tool/services/aoi`、`tool/services/spi`、`tool/services/reflow` 是三套模型服务的唯一正式实现，旧重复实现只作为 `archive/legacy_tools` 归档。
- D002：服务默认地址为 AOI `127.0.0.1:8000`、回流焊 `127.0.0.1:8001`、SPI `127.0.0.1:8002`，可通过启动参数和 `PCBA_*_BASE_URL` 环境变量覆盖。
- D003：Agent Tool 保持框架无关 Python HTTP 客户端形态，LangGraph 注册适配留到 Agent 编排阶段。
- D004：三套服务错误响应统一使用包含 `success`、`request_id`、版本信息、耗时、`data`、`warnings` 和结构化 `error` 的契约。
- D005：三套服务、Agent Tool、RAG、KG 和测试统一使用 Conda 环境 `PCB_Agent`，依赖版本以仓库 requirements 文件和已验收环境为准。
- D006：`数据资料/`、缓存、日志和本地环境文件不进入 Git；正式代码、项目文档、模型服务、RAG资料和 `txt_source/` 按发布策略纳入版本管理。
- D007：首期正式缺陷范围为 `insufficient_solder`、`excessive_solder`、`short`、`shifted_component`，候选致因只表示待验证路径，不声明唯一真实根因。
- D008：T09 V1.1 权威本体为 `ontology/pcba_defect_causality.v1.1.yaml`，`rag/schemas/entity_dictionary.json` 必须与其保持一致。
- D009：短路→湿焊膏桥连、元件偏移→贴装位置偏差在无 Tool 时仍以 `strong + unverified` 保留；缺少 Tool 不得导致候选原因被删除。
- D010：`tool_supported` 关系在优化后必须再次调用预测 Tool 验证。
- D011：Page/Block/Chunk/Metadata/Retrieval/Embedding 等 JSON Schema 是正式数据契约，新增版本不得破坏已验收旧版本。
- D012：Chunk ID 必须基于来源、章节、页、Block 边界和文本哈希稳定生成，不使用会因前序插入而漂移的顺序编号。
- D013：Metadata 使用 T09 V1.1 本体驱动的确定性规则，不调用 LLM；英文 canonical name 是程序过滤和关系映射的权威值。
- D014：Embedding 固定使用 BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`，输入为 `section_path + 正文`，不注入规则 Metadata。
- D015：Qdrant Collection 中 Point ID 使用 `UUIDv5(chunk_id)`；Collection 和 named volume 是可重建派生产物，不是唯一数据源。
- D016：Retriever 保持框架无关，查询规范化只做 NFKC、首尾空白清理和连续空白合并，不自动翻译、扩展术语或放宽过滤。
- D017：Hybrid 检索固定 Dense Top-20 与 Sparse Top-20，等权 RRF(k=60)，融合 Top-10，默认最终 Top-5；不强制来源多样性。
- D018：Retrieval Evaluation 的 Gold 来自冻结候选池标注，指标需记录知识库、索引、Embedding、Retriever 和 Fusion 版本，出现未标注返回项时拒绝生成指标。
- D019：V0.1、V0.2、V0.3 知识库使用独立 Qdrant Collection，旧版本保留用于追溯和对比。
- D020：三篇补充双栏论文使用经核对的显式小标题与原生逻辑阅读顺序切分，不改变原四份核心资料的已验收切分。
- D021：失败归因只读取冻结标签池和生产 Top-10 结果，不修改资料、Chunk、Embedding、索引或 Retriever。
- D022：33 题 Reranker 全量实验不自动改变 T10.9 生产 Retriever 契约，是否接入生产需用户审核后决定。
- D023：`STATUS.md` 和 `DECISIONS.md` 只保留启动必读信息，历史过程和展开说明归档到 `docs/archive/` 并按需读取。
- D024：Retriever正式主评测目标固定为`rag/evaluation/question/retriever_main.json`中的16题，但Recall/MRR/nDCG只对冻结候选池中存在rel=2的题计算，无直接Gold题必须单列。
- D025：T10.10冻结链路为V0.3等权RRF Top-20、固定revision的BGE Reranker及`w=0.73`排名融合Top-5，权威文件必须通过`rag/evaluation/FROZEN_MANIFEST.json`哈希校验。
- D026：T10以指标例外结项，Agent默认采用D025冻结排序方案，现有等权RRF作为Reranker不可用时的显式降级路径。
- D027：T11只有在图谱能返回带来源与验证状态的候选因果路径和下一验证动作时才算完成，Neo4j静态展示不作为验收结果。
- D028：T13必须由缺失数据、证据和Tool结果驱动条件状态转移，固定顺序调用全部模块不作为智能体实现。
- D029：候选致因在KG中使用`CausalHypothesis`节点建模并以`relationship_id`幂等标识，不使用暗示确定根因的`CAUSED_BY`关系。
- D030：T11核心因果事实只来自T09 V1.1，查询契约允许多路径和跨工序指标汇总，但未审核RAG证据、未确认阈值及本体外实体不得进入有效决策。
- D031：Neo4j运行基线固定为Community`2026.07.1`与官方Python Driver`6.2.0`，凭据使用本地环境变量，named volume只作可由T09重建的派生数据存储。
- D032：KG查询按`strong > medium_strong > conditional`稳定返回全部权威候选，案例数据只决定验证输入是否齐全，候选支持或否定必须由T13结合Tool结果评估。
- D033：T13固定使用`qwen3.7-flash-2026-07-15`的OpenAI兼容API，Qwen只负责输入抽取、补问和报告生成，Tool选择与状态路由保持确定性。
- D034：仅当用户目标为`diagnose_and_optimize`且预测支持异常时自动计算优化建议，建议不写入设备并必须再次调用预测Tool复验。
- D035：T13使用原生LangGraph StateGraph与SQLite checkpoint保存纯JSON状态，缺失输入通过同一`thread_id`中断恢复，外部Tool调用不交给LLM自由路由。
- D036：Agent最终报告中的Tool数值、推荐参数和复验结论必须由程序从结构化响应确定性生成，Qwen只生成不含具体参数数值的诊断概括。
