# STATUS.md

## 当前任务

T10.1 至 T10.9 和 T09 V1.1 已通过用户审核。T10.10 已完成33题范围收敛与Reranker全量重评，等待用户审核。T10.11 未开始。

## 已完成

- 建立唯一正式目录：`tool/services/{aoi,spi,reflow}`。
- 将旧重复实现归档到 `archive/legacy_tools`，未修改 `数据资料`。
- 统一 Conda 环境、启动脚本、基础 URL、健康检查和完整错误 Schema。
- 补齐 SPI `/health`，并修正文档中的路径、端口和环境说明。
- 实现 5 个框架无关的 Agent HTTP Tool、Pydantic 输入 Schema 和统一异常类型。
- 固化 `PCB_Agent` 已验证依赖版本到 `tool/requirements.txt`。
- 完成真实 HTTP 端到端验收：3 个健康检查、5 个成功调用、5 个 422 错误契约，以及两个优化结果的再次预测均通过。
- 验收结果位于 `tool/tests/e2e/VALIDATION_REPORT.md` 和 `tool/tests/e2e/artifacts/latest_report.json`。
- 已初始化 Git 仓库并发布到 `https://github.com/jixinzhou/PCBA_Agent.git` 的 `main` 分支。
- `数据资料/`、Python 缓存、运行日志和本地环境文件已通过 `.gitignore` 排除。
- 完成四类缺陷的致因本体与术语体系，冻结缺陷、候选原因、工序、验证指标、验证 Tool、优化 Tool 和关系强度映射。
- 明确 `shifted_component` 的贴装位置偏差为强候选关系；回流热不均/润湿不同步仅作为条件性候选原因，不表述为必然根因。
- 明确短路与贴装位置偏差等当前无 Tool 的因素保留为 `unverified`，不因缺少工具而从候选原因中删除。
- 完成四份资料的章节感知 Chunk，共生成 183 条 Chunk V1 记录；1,467 个可检索 Block 全部覆盖，无空 Chunk、跨文档 Chunk 或章节边界违规。
- 固定 BGE-M3 Tokenizer revision，按 300～600 tokens 组合，普通文本超过 800 tokens 递归拆分；GJB 第 8 页一张 944-token 原子表格按结构保护规则完整保留并显式告警。
- 稳定 Chunk ID 使用来源、章节、页、Block 边界和文本哈希生成；相同输入重复构建的 183 个 ID 完全一致。
- 建立 T09 V1.1 唯一权威本体，冻结四类缺陷、五个候选原因和五条致因关系；关系强度与 Tool 验证状态使用独立字段。
- 修正旧实体词典，删除已移出正式范围的 `cold_solder_joint`、`tombstoning` 和 `insufficient_reflow_heat_input`，补齐多锡、短路、元件偏移及对应候选原因。
- 固化两条无 Tool 关系：短路→湿焊膏桥连、元件偏移→贴装位置偏差，均以 `strong + unverified` 保留。
- 新增 Chunk/Metadata V1.1 和映射追踪 V1，保留 T10.4 的 Chunk V1、稳定 ID、正文、章节和页码不变。
- 完成 183 条 Chunk 的确定性 T09 术语映射和 183 条逐 Chunk 追踪；同义词、相关术语、候选原因、工序词、证据角色规则和人工配置来源可区分。
- 允许单 Chunk 多工序与多缺陷；最终生成 42 条多工序 Chunk 和 2 条多缺陷 Chunk，四类缺陷均有检索标签覆盖。
- 对 `short`、`bridge`、`profile`、`placement`、`wetting` 和 `润湿` 禁用裸词匹配，仅允许明确短语；封面、目录、广告、参考文献、作者和后置内容不加语义标签，结构区段误标为 0。
- 将统一环境升级为 `torch 2.6.0+cu118` 和 `torchvision 0.21.0+cu118`，解决 BGE-M3 PyTorch 权重的安全加载要求；AOI 模型已完成最小 CUDA 加载与单次四分类推理验证。
- 使用固定 revision 的本地 BGE-M3 为 183 条 Chunk 生成 1024 维 Dense 和 Sparse lexical weights，全部记录 Schema 有效、Sparse 非空且无截断。
- Embedding 输入固定为 `section_path + 正文`，不注入规则 Metadata；20 条结构性 Chunk 保留向量并携带排除标记，供后续 Retriever 过滤。
- 两次全量重建的四个 Embedding 文件 SHA256 完全一致；5 条分层样本的 Dense 与 Sparse 重复编码检查通过。
- 使用 Docker Compose 运行固定版本 Qdrant Server 1.18.2，并在统一环境安装 Qdrant Client 1.19.0；服务仅绑定本机 `127.0.0.1:6333/6334`，named volume 保存可重建索引。
- 建立 `pcba_industrial_knowledge_v0_1` Collection，将 183 条 Chunk 一一映射为稳定 UUIDv5 Point，同时保存 1024 维 Dense、BGE-M3 Sparse 与完整 Chunk/Metadata Payload。
- 183 个 Point 全量读回、7 类 Payload 过滤、Dense/Sparse 自查询和内容指纹检查通过；幂等更新与显式完整重建的指纹完全一致。
- 新增 Retrieval V1.1，在保留 Retrieval V1 不变的前提下返回 Metadata V1.1、Dense/Sparse 检索通道、查询 token 数、系统过滤和完整索引追踪。
- 实现框架无关 `Retriever` Python 接口，使用固定 revision BGE-M3 `encode_queries` 和 Qdrant `query_points` 分别完成 Dense、Sparse 原始候选检索，不依赖 LangGraph 或模型服务内部代码。
- 查询规范化只执行 NFKC、首尾空白清理和连续空白合并；支持来源、工序、缺陷、证据角色、语言和文档类型过滤，并默认排除20条结构性 Chunk。
- 三组真实中英文检索、过滤合规和 Schema 验证通过；中文查询对英文正文的 Sparse 合法空结果被原样保留，不自动翻译或放宽过滤。
- 新增 Retrieval V1.2 和框架无关 `retrieve_hybrid`，固定双路Top-20、等权RRF(k=60)、融合Top-10和默认最终Top-5；不启用来源多样性或Reranker。
- Dense/Sparse候选按稳定`chunk_id`严格合并并保留两路排名、原始分数、RRF贡献和完整引用；同一ID的正文、引用或Metadata不一致时明确报错。
- 三组真实Hybrid查询覆盖双路各20条、双路完全重合和Sparse空结果；最终结果均无重复，Sparse为空时不填充不相关候选。
- T10.10已校验48题原始评测集，其中40题标为可回答、8题标为无答案；候选生成不使用缺陷、工序或答案标签作为Retriever过滤条件。
- 每题从Dense Top-20与Sparse Top-20按当前T10.9等权RRF(k=60)计算评测专用Top-20，共生成960行候选；生产Retriever的融合Top-10配置保持不变。
- `rag/evaluation/evaluation_top20.csv`是唯一权威标注文件，旧Markdown、旧480行清单和既有人工标签全部作废；CSV保留稳定候选ID、双路排名、分数、完整正文、来源、页码和索引追踪。
- 已实现`qwen3.8-max` OpenAI兼容API标注脚本，每题一次请求且发出前隐藏排名、分数和原始答案标签；固定输出`final_answerable`及0/1/2相关性，支持JSON校验、三次重试、失败留空和逐题原子写回。
- `qwen3.8-max`已完成全部48题、960条候选标注，无遗漏或失败；相关性0、1、2分别为668、236和56条，22题存在直接答案，22题仅有辅助证据，4题全部无关。
- 已生成`rag/evaluation/gold_dataset.json`，通过CSV SHA256、标注模型、Prompt、知识库、索引、Embedding、Retriever和Fusion版本冻结Chunk、Source与Page级Gold。
- 已实现`rag/evaluation/evaluate_retriever.py`，使用无业务Metadata过滤的生产Hybrid Top-10重新检索48题；全部480条返回均位于已标注池内，无未标注结果。
- T10.10主指标覆盖22个`candidate_answerable=true`查询：Recall@5为0.510606、MRR@10为0.679924、nDCG@10为0.645312、Source Hit@5为0.863636、Page Hit@5为0.818182；26个候选池不可回答查询单独统计。
- 四份核心资料均进入候选池且至少有一条`relevance>=1`证据；IPC仅有14条候选、1条辅助证据，没有`relevance=2`直接Gold，因此T10.10的四来源直接Gold覆盖验收项尚未满足。
- 评测结果位于`rag/evaluation/evaluation_results.json`，最简报告位于`rag/reports/T10.10_RETRIEVAL_EVALUATION.md`；55项RAG、20项评测和8项本体测试、Python编译及`pip check`全部通过。
- Top-50隔离实验生成26题×50候选共1300行：按Chunk ID复用520条Top-20标签，只对780条新增候选调用Qwen；新增相关性0、1、2分别为674、93和13条，全部标注完成。
- 13条初判直接证据经独立严格Prompt复核后仅确认3条，分别恢复Q012、Q014和Q029共3/26题，首次命中RRF排名为47、31和33；IPC确认直接证据仍为0。
- 实验结论为不建议将正式候选池全局扩展到Top-50，优先检查知识覆盖、问题设计和标注判断。最终报告位于`rag/reports/T10.10_TOP50_EXPANSION_EXPERIMENT.md`；实验未修改正式Top-20 CSV、冻结Gold或基线指标。
- 已生成`rag/evaluation/T10.10_UNRESOLVED_QUESTION_REVIEW.md`，列出Top-50后仍未解决的23题：17题原设计为业务可回答问题，6题原设计为无答案或越界负样本；当前不提前执行知识检索或删除问题。
- 已按用户审查结果保留15个正样本与全部6个负样本，删除Q027和Q037；负样本不参与知识补充核查。
- 已实现独立于Retriever排序的关键词／同义词知识审计：对183条Chunk和136个非空Page检索完整与部分要点，共生成60条候选。9题检出候选，Q007、Q010、Q017、Q030、Q033、Q034未检出核心关键词候选。
- 人工核对报告位于`rag/evaluation/T10.10_KNOWLEDGE_COVERAGE_REVIEW.md`，候选明细位于`rag/evaluation/knowledge_audit_candidates.csv`；自动初筛不替代PDF原文裁决。
- 已删除评测问题 Q010、Q027、Q030、Q037，保留 Q020；最终评测集为 44 题。
- 已将 `txt_source/补充` 的 3 份 PDF 纳入知识库 V0.2，新增 34 页、32 条 Chunk；知识库合计 7 份资料、216 页、215 条 Chunk。
- 仅对 32 条新增 Chunk 生成 BGE-M3 Dense/Sparse Embedding，并建立 `pcba_industrial_knowledge_v0_2` Collection；V0.1 Collection 保留不变。
- 44 题 Top-20 共 880 条候选中精确复用 769 条旧标签，仅将 111 条新增候选交给 Qwen 标注；最终无空标签或错误。
- 已冻结 `gold_dataset_final.json` 并完成生产 Hybrid Top-10 重检索；Top-10 未标注返回项为 0。
- V0.2 最终指标为 Recall@5 0.451449、MRR@10 0.577899、nDCG@10 0.600400、Source Hit@5 0.826087、Page Hit@5 0.782609；Recall 和 MRR 尚未达到 T10.11 门槛。
- 三篇补充论文已改用经核对的显式小标题与原生逻辑阅读顺序，章节数分别由1恢复为27、10、14；补充Chunk由32条重切为62条，不再跨真实小节拼接。
- V0.3共7来源、216页、245条Chunk；245条Metadata、62条变化Embedding和`pcba_industrial_knowledge_v0_3` Collection均已构建并通过校验，旧128个评测候选Chunk正文与页码零变化。
- V0.3的44题Top-20候选池共880条，712条旧标签精确复用；168条变化候选均由Qwen完成，标签无缺失或失败。
- 已冻结`gold_dataset_final_v3.json`并重跑生产Hybrid Top-10，未标注返回项为0；27个可回答问题的Recall@5为0.415226、MRR@10为0.554733、nDCG@10为0.592216、Source Hit@5为0.814815、Page Hit@5为0.777778。
- 与V0.2共同的23个可回答问题相比，V0.3的MRR@10由0.577899升至0.597464，Source/Page Hit@5升至0.869565/0.826087；Recall@5小幅降至0.443961，nDCG@10小幅降至0.594675。
- Q011、Q012、Q014、Q048新增直接Gold，首次候选排名分别为8、9、20、1；新增知识已生效，但Q014仍未进入生产Top-10，Recall与MRR仍未达到T10.11门槛。
- 已生成44题失败归因表`rag/evaluation/T10.10_FAILURE_ATTRIBUTION_V0.3.csv`；18题成功、8题为`ranking_failure`、12题为`knowledge_or_question_gap`，已确认负样本中3题为`knowledge_gap`、3题为`rewrite`，没有需要优先修Dense/Sparse召回的`recall_failure`。
- 8个排序失败问题为Q011、Q012、Q014、Q018、Q021、Q022、Q024、Q025，首次直接证据排名为8、9、20、14、20、8、13、17；下一轮Retriever优化应优先使用这组问题。
- 已生成`rag/evaluation/T10.10_KNOWLEDGE_QUESTION_GAP_REVIEW.md`，列出12个`knowledge_or_question_gap`问题、rel=1数量及代表性来源/页码，等待人工填写`多证据问题/改写/删除/知识缺口`。
- 已使用固定revision的`BAAI/bge-reranker-v2-m3`对8题现有Top-20执行隔离重排：Top-5直接答案命中由0/8提升至6/8，MRR@5由0提升至0.5104；Q024重排至第6，Q025重排至第12。
- Reranker缓存后模型加载约8.10秒、160对打分约7.51秒，CUDA峰值显存约1109.79MB；最长输入806 tokens且无截断。实验报告位于`rag/reports/T10.10_RERANKER_EXPERIMENT.md`，生产Retriever保持不变。
- 用户最终删除Q001、Q004、Q007、Q008、Q009、Q013、Q017、Q020、Q029、Q033、Q034共11题；Q019保留并改写为“为什么过大的焊膏沉积量可能在印刷和贴装后未出现桥连，却在回流后形成桥连？”，最终实验范围为33题。
- 对33题共660个原RRF Top-20候选执行全量Reranker并复用原V0.3 Gold：27个可回答问题的Recall@5由0.415226升至0.641534、MRR@10由0.554733升至0.595062、nDCG@10由0.592216升至0.732433、Source/Page Hit@5均升至0.962963。
- Reranker使直接答案Top-5覆盖由19/27升至25/27，Top-10覆盖由22/27升至26/27；Q024仍未进入Top-5，Q025仍未进入Top-10。Recall与MRR仍未达到T10.11门槛。
- 全量重评缓存后模型加载约8.03秒、660对打分约26.48秒，峰值显存约1113.06MB；最长输入974 tokens且无截断。报告位于`rag/reports/T10.10_RERANKER_FULL_EVALUATION.md`，生产Retriever仍未修改。

## 当前状态

三套模型服务、5 个 Agent Tool 和缺陷致因本体可用于后续 RAG、知识图谱和 Agent 编排。正式端口为 AOI `8000`、回流焊 `8001`、SPI `8002`，并支持启动参数和环境变量覆盖。

GitHub 远端 `origin/main` 已建立并与本地 `main` 跟踪同步。

T10 父任务已进入规划状态，并按以下技术路线拆分为 T10.1 至 T10.11：

```text
PDF
→ Chunk
→ Metadata
→ Embedding
→ Vector DB
→ Retriever
→ Top-K
→ Retrieval Evaluation
```

T10 V0.1 仅使用以下四份核心资料：

- `indium-guide-to-minimizing-solder-defects-2021-tpcag2sd.pdf`
- `Solder-Paste-Print-Inspection-Defect-Guide.pdf`
- `IPC-7530+焊工艺温度曲线指南(回流焊和波峰焊)+中文版.pdf`
- `GJB_3243A-2021_电子元器件表面安装要求.pdf`

其他 PDF 不进入 V0.1 的解析、索引和检索评测。

T10.10 知识补充后使用 V0.2：在上述四份核心资料之外增加 `txt_source/补充` 中三份工艺资料，形成 7 来源、215 Chunk 的独立 V0.2 索引；V0.1 数据与 Collection 保留用于追溯。

T10 严格采用串行用户验收：每次只实施一个子任务；完成后提交产物和验证结果给用户检查；只有用户明确通过，才开始下一个子任务。

T10.1 已创建并冻结四份核心 PDF 白名单、Page/Chunk/Metadata/Retrieval V1 JSON Schema、最小示例和 RAG 目录说明。Schema 仅包含已确认的必填字段，Metadata 不包含别名、相关术语或候选致因。

T10.2 已完成四份 PDF 共 20 个代表页的原生文本提取、AES 读取、本地中英文 OCR、表格按行文本提取、页面质量判断和 Page Schema 验证。20 条样本记录全部通过 Schema；Indium AES 文档可读取；GJB 5 个扫描样本页均完成 OCR；3 个低内容页被明确标记为 `partial`，没有静默失败。

T10.2 采用 pypdf + cryptography、pypdfium2、RapidOCR + ONNX Runtime 和 jsonschema。未采用 PaddleOCR，因为其依赖解析会降级现有 NumPy 和 PyYAML。安装后 `pip check`、7 项自动化测试和既有核心依赖版本回归均通过；该阶段尚未执行四份 PDF 全量解析，也未创建 Chunk、Embedding 或向量索引。

T10.3 已完成四份白名单 PDF 全量页面解析，共生成 182 条 Page V1.1 记录：130 页 `success`、10 页 `partial`、42 页已确认 `blank`、0 页 `failed`。182 条记录全部通过 Schema，不存在静默丢页。页面数据增加印刷页码、标题/段落/列表/表格/公式等 Blocks、归一化坐标、阅读顺序和断点续跑状态。

IPC PDF 第 7 至 48 页经人工确认属于空白/无效页，显式配置为 `blank`，固定阅读器水印不进入正文或印刷页码。Indium 的 4 个无文字图片页保留为 `partial + non_text_content`，不误判为空白页。全量干净重跑耗时 141.419 秒，17 项自动化测试、13 个代表页抽检和 `pip check` 全部通过。

T10.4 使用 Page V1.1 Blocks 构建 title、text、list、table 和 formula 语义原子单元，并在同一章节内组合。目录页、封面、广告页和 GJB 附录使用显式章节覆盖；GJB 与 IPC 禁止无编号 OCR 短句自动升级为章节标题。四份资料分别生成 32、64、2、85 条 Chunk，共 183 条，全部通过 Chunk/Metadata Schema。

183 条 Chunk 中 66 条位于 300～600 tokens，12 条位于 601～800 tokens，104 条因短章节或前后置内容低于 300 tokens，1 条不可拆分原子表格超过 800 tokens。25 项全量自动化测试、Python 编译检查、稳定 ID 双构建验证和 `pip check` 全部通过。验收报告位于 `rag/reports/T10.4_CHUNK_VALIDATION.md`。

T09 V1.1 的权威文件为 `ontology/pcba_defect_causality.v1.1.yaml`，由 `ontology/schemas/pcba_defect_causality.v1.schema.json` 约束。`rag/schemas/entity_dictionary.json` 是与该本体对齐的程序词典，不再独立定义缺陷范围。原始 `txt_source/致因本体与术语体系` 仅保留为历史讨论来源。

T09 V1.1 共有 3 条 `tool_supported` 关系和 2 条 `unverified` 关系。8 项专项测试已验证本体 Schema、四类缺陷、五条关系、跨引用、工序、Tool、Metadata 枚举和实体词典一致性；按用户要求不生成独立验收报告。

T10.5 使用 `metadata_mapping.v0.1.yaml`、T09 V1.1 本体和实体词典对 Chunk 进行确定性映射，不调用 LLM。183 条 Chunk V1.1 与 183 条映射追踪全部通过 Schema，T10.4 不可变字段全部保持一致。114 条 Chunk 有工序标签、21 条有缺陷标签、161 条有证据角色；工序计数为印刷 83、贴装 30、回流 52，缺陷计数为少锡 9、多锡 7、短路 2、元件偏移 5。

33 项 RAG 回归测试、8 项 T09 本体测试、Python 编译检查和 `pip check` 均通过。T10.5 验证摘要位于 `rag/reports/t10.5_summary.json`，简要报告位于 `rag/reports/T10.5_METADATA_VALIDATION.md`。

T10.6 使用 `FlagEmbedding 1.4.0` 和 BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`。组合输入最大 958 tokens，冻结 `max_length=1024`、`batch_size=1`、CUDA FP16 推理和 float32 保存；不生成 ColBERT 多向量。183 条 Dense 均为 1024 维且归一化，183 条 Sparse 均非空，NNZ 范围为 4～379。

RTX 3050 Laptop GPU 全量 Embedding 推理约 6.743 秒，峰值已分配显存约 1113.89 MB。37 项 RAG 测试、8 项本体测试、Python 编译和 `pip check` 全部通过。T10.6 摘要位于 `rag/reports/t10.6_summary.json`。

T10.7 使用固定命名空间的 UUIDv5 将 Chunk ID 转换为 Qdrant Point ID。Collection 同时配置 `dense` 1024 维 Cosine 向量和无 IDF modifier 的 `sparse` 向量；Cosine 存储前显式执行与 Qdrant 一致的 L2 归一化。完整正文、引用、Metadata、Embedding 哈希与版本均保存在 Payload。

Qdrant 当前状态为 green，共 183 个唯一 Point；20 条结构性 Chunk 保留并可通过 `semantic_tag_excluded` 过滤。来源、工序、缺陷、证据角色、语言、文档类型和结构排除共7个 Payload 索引的实际过滤计数均与输入一致，Dense/Sparse 自查询、5 条分层读回和完整内容指纹通过。最简摘要位于 `rag/reports/t10.7_summary.json`。

T10.8 保留原 Retrieval V1，并新增 Retrieval V1.1 作为 Retriever 正式契约。请求继续包含 query、top_k 和六类可选过滤数组；响应增加 `retrieval_mode`，返回 Chunk 正文、来源、页码、章节、Metadata V1.1、通道分数和索引/模型追踪。Dense 通道只填写 dense_score，Sparse 通道只填写 sparse_score；fusion_score 与 rerank_score 保持空值供后续任务使用。

三组真实查询的 Dense 结果数为5、2、5，Sparse 结果数为0、2、3，所有非空结果均满足请求过滤。首个 Dense 调用包含约18.7秒模型冷加载，后续 Dense 约59～65毫秒、Sparse 约7～19毫秒。48项 RAG 测试、8项本体测试、Python 编译和 `pip check` 全部通过；最简摘要位于 `rag/reports/t10.8_summary.json`。

T10.9 新增 Retrieval V1.2 和 `fusion.v0.1.yaml`。同一规范化查询先分别获取Dense/Sparse Top-20，再按`chunk_id`合并，使用`1/(60+rank)`等权RRF融合；排序并列时依次使用最优通道排名和Chunk ID，保证结果稳定。融合池限制Top-10，默认最终Top-5，可请求1～10条。

三组真实验证中：无过滤查询Dense/Sparse各20条、跨通道重复11条、合并后29条；短路过滤查询两路各2条且完全重合；中文少锡查询Dense 9条、Sparse 0条并自然退化为Dense。55项RAG测试、8项本体测试、Python编译和`pip check`全部通过。按用户要求仅生成最简报告`rag/reports/T10.9_FUSION_VALIDATION.md`，未生成T10.9 JSON摘要。

## 下一步

等待用户审核33题Reranker全量重评结果；未获得确认前不把Reranker接入生产Retriever。

## 阻塞问题

- 本机 `8000` 当前被 NeatReader 占用；可关闭该应用，或为 AOI 使用 `-Port` 和 `PCBA_AOI_BASE_URL` 覆盖地址。
- `archive/legacy_tools` 是否永久删除，待用户在后续验收后决定；当前不影响正式实现。
- `txt_source/` 中包含 IPC/GJB 等参考 PDF；若仓库保持公开，需要由项目方确认其公开分发授权。
