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

## D007 缺陷致因本体范围

- 首期冻结四类缺陷术语：`insufficient_solder`、`excessive_solder`、`short`、`shifted_component`。
- 少锡和多锡关联印刷工序，通过 `spi_vte_prediction` 验证，并可由 `spi_parameter_optimization` 优化后再次预测。
- 短路关联湿焊膏桥连，关系强，但当前无直接验证 Tool，保留为不可工具验证的候选原因。
- 元件偏移同时保留贴装位置偏差和回流热不均/润湿不同步两个候选方向；前者为强关系但当前无 Tool，后者通过回流预测与优化 Tool 检查，证据等级为条件性。
- 候选原因只表示需要检查和验证的致因路径，不声明唯一真实根因。

原因：使印刷、贴装和回流三道工序进入统一诊断模型，同时避免把间接或条件性工艺关系表述为确定因果。

## D008 T10.3 页面结构契约

- 保留已验收的 Page V1，不覆盖 T10.2 样本；T10.3 全量数据使用 Page V1.1 和 Block V1。
- Page V1.1 明确区分 `success`、`partial`、`blank` 和 `failed`，并增加印刷页码、页面 Blocks、归一化坐标和阅读顺序。
- 只有人工确认并写入配置的页面可以标记为 `blank`；无文字图片页标记为 `partial + non_text_content`。
- T10.3 只生成页面和结构数据，不生成 Chunk 或术语 Metadata。

原因：在不破坏既有契约的前提下，为后续章节切分保留标题、表格、公式和版面边界，并避免把图片内容误判为空白页。

## D009 T10.5 双语术语 Metadata

- 缺陷类型以 T09 冻结的 `insufficient_solder`、`excessive_solder`、`short`、`shifted_component` 为准。
- 工序和缺陷 Metadata 同时保留稳定实体 ID、英文 canonical name 和中文展示名，并记录词典版本。
- 英文 canonical name 是程序过滤和关系映射的权威值，中文名称用于展示与中文检索。

原因：避免中英文独立数组错位，同时保证程序接口稳定和中文检索可读性。该映射在 T10.5 实施，不扩大 T10.3 范围。

## D010 T10.4 章节切分与稳定 Chunk ID

- Chunk 以 Page V1.1 Blocks 为输入，按原始阅读顺序构建 title、text、list、table 和 formula 语义原子单元；页眉、页脚和页码不进入 Chunk。
- 表题与表格、列表项与续接段落保持为不可拆分结构。目标长度为 300～600 个 BGE-M3 tokens；普通文本超过 800 tokens 递归拆分，不可拆分原子结构超限时完整保留并告警。
- Tokenizer 固定为 `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`，不使用与后续 Embedding 模型不一致的近似 Token 计数。
- 标题纠错采用通用过滤规则和显式页面章节覆盖；GJB 与 IPC 不允许无编号 OCR 短句自动升级为章节标题，GJB 附录 A/B/C 使用固定页范围恢复章节。
- Chunk ID 为 `source_id:c_<24位SHA256>`，哈希输入包含 Chunker 版本、来源、章节路径、页、Block 边界和文本哈希；不使用会因前序插入而整体漂移的顺序编号。
- T10.4 只填写来源级 Metadata，工序、缺陷和 T09 术语映射留到 T10.5。

原因：在保留工业文档结构的同时获得可复现、可增量重建的 Chunk，并避免 OCR 标题误判和顺序编号漂移影响后续 Embedding、索引与引用追踪。

## D011 T09 V1.1 唯一权威本体

- `ontology/pcba_defect_causality.v1.1.yaml` 是缺陷致因本体的唯一机器可读权威文件；`rag/schemas/entity_dictionary.json` 必须与其保持一致，`txt_source/致因本体与术语体系` 仅作为历史讨论来源。
- 正式缺陷范围仅包含 `insufficient_solder`、`excessive_solder`、`short`、`shifted_component`；`cold_solder_joint` 和 `tombstoning` 不属于当前正式缺陷实体。
- 冻结五条关系：少锡→焊膏沉积或转移不足、多锡→焊膏沉积量过大、短路→湿焊膏桥连、元件偏移→贴装位置偏差、元件偏移→回流热不均或润湿不同步。
- 关系强度与 Tool 验证状态分离。短路的湿焊膏桥连和元件偏移的贴装位置偏差均为 `strong + unverified`；缺少 Tool 不得导致候选原因被删除。
- 元件偏移的回流热不均或润湿不同步为 `conditional + tool_supported`，只能表示需要通过回流指标检查的候选方向，不得表述为必然根因。
- `tool_supported` 关系在优化后必须再次调用预测 Tool 验证；所有关系仍遵守“候选原因不等于已确认唯一根因”。

原因：消除旧实体词典与 T09 最终范围之间的漂移，并为 T10.5 Metadata、T11 知识图谱和后续 Agent 编排提供同一套稳定实体与关系定义。本决策是 D007 的机器可读 V1.1 修订。

## D012 T10.5 Metadata V1.1 与可追溯术语映射

- 保留 T10.4 Chunk V1 不变，T10.5 生成 Chunk/Metadata V1.1；Chunk ID、正文、章节、页码和文本哈希不得因 Metadata 映射改变。
- Metadata 同时保存稳定实体 ID、英文 canonical name、中文展示名、词典版本、本体版本和映射版本；程序过滤仍以英文 canonical name 为权威。
- 映射采用 T09 V1.1 本体驱动的确定性规则与受控配置，不调用 LLM；映射追踪区分同义词、相关术语、候选原因、工序词、证据角色规则和人工配置默认值。
- `short`、`bridge`、`profile`、`placement`、`wetting` 和 `润湿` 不允许裸词命中，只接受上下文明确的长短语；结构性区段不添加工序、缺陷或证据角色标签。
- 单个 Chunk 可以同时属于多个工序或多个缺陷。候选原因命中仅用于检索标签并保留关系 ID，不表示已确认根因。

原因：兼顾跨工序检索召回率与误匹配控制，同时使每一个最终 Metadata 标签都可回溯到明确规则和 T09 权威实体。

## D013 T10.6 BGE-M3 Dense 与 Sparse Embedding

- Embedding 模型固定为 `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`，使用 `FlagEmbedding 1.4.0` 从本地缓存加载。
- 模型输入固定为 `section_path + 两个换行 + Chunk text`；规则生成的工序、缺陷和证据角色 Metadata 不注入模型文本。
- V0.1 同时生成归一化 1024 维 Dense 与 Sparse lexical weights，不生成 ColBERT 多向量；Sparse 使用排序且唯一的 token ID 和正权重保存。
- 冻结 `max_length=1024`、`batch_size=1`、CUDA FP16 推理和 float32 向量保存。所有输入必须预先计数并保证零截断。
- 183 条 Chunk 全部生成向量；结构性 Chunk 不静默删除，而是保留 `semantic_tag_excluded` 标记供后续检索过滤。
- 每条记录保存 Chunk ID、原始文本哈希、Embedding 输入哈希、模型 revision、配置版本、设备和数值精度。相同输入、权重和配置必须可重建。
- 统一环境使用 `torch 2.6.0+cu118` 与 `torchvision 0.21.0+cu118`，满足 PyTorch 权重安全加载要求；AOI 模型升级后必须至少通过一次模型加载和推理验证。

原因：为 T10.7 Qdrant 同时提供语义召回和词法召回向量，并避免 Metadata 规则标签污染向量语义，同时保持向量与原始 Chunk 的可追溯和可重建性。

## D014 T10.7 Docker Qdrant 与稳定 Point

- RAG V0.1 使用 Docker Compose 运行 Qdrant Server 1.18.2，Python Client 固定为 1.19.0；开发服务仅绑定 `127.0.0.1:6333/6334`，索引保存在 named volume。
- Collection 固定为 `pcba_industrial_knowledge_v0_1`，每个 Chunk 对应一个 Point；Point ID 使用固定命名空间的 `UUIDv5(chunk_id)`，原始 Chunk ID 继续保存在 Payload。
- Collection 同时保存命名 `dense` 与 `sparse` 向量。Dense 为 1024 维 Cosine float32，写入前显式执行与 Qdrant 一致的 L2 归一化；Sparse 直接保存 BGE-M3 lexical weights，不启用 IDF modifier。
- Payload 保存 Chunk 正文、章节、来源页、完整 Metadata、结构排除标记、文本与 Embedding 哈希及版本。来源、工序、缺陷、证据角色、语言、文档类型和结构排除字段建立过滤索引。
- 默认构建对兼容 Collection 幂等 upsert，并删除不属于当前权威输入的旧 Point；只有显式 `--recreate` 才删除并完整重建 Collection。每次构建必须执行精确数量、全集 ID、向量和 Payload 读回、过滤查询及内容指纹检查。
- Qdrant named volume 和 Collection 是可重建派生产物，不是唯一数据源；T10.5 Chunk V1.1 与 T10.6 Embedding JSONL 仍是索引输入的权威记录。

原因：为后续框架无关 Retriever 提供稳定的 Dense、Sparse 和 Metadata 过滤服务，同时保证 Point 可追溯、索引可重复构建，并降低 Windows bind mount 和误覆盖风险。

## D015 T10.8 框架无关 Dense/Sparse Retriever

- 保留 T10.1 已冻结的 Retrieval V1 不变，新增 Retrieval V1.1；V1.1 返回完整 Metadata V1.1，并显式记录 Dense/Sparse 通道、查询 token 数、索引版本、模型 revision 和固定系统过滤。
- Retriever 实现为独立 Python 接口 `retrieve_dense` 与 `retrieve_sparse`，只依赖本地 BGE-M3 和 Qdrant API，不依赖 LangGraph、FastAPI 或模型服务内部代码。T10.8 不执行融合、去重、RRF或重排。
- 查询使用与文档相同固定 revision 的 BGE-M3 `encode_queries`，模型延迟加载并缓存最近一次查询向量，CUDA优先且允许CPU回退；查询最大512 tokens，超限明确报错，不静默截断。
- 查询规范化仅执行 Unicode NFKC、去除首尾空白和合并连续空白，不翻译、不扩展工业术语、不调用LLM，也不添加查询指令。
- 同一过滤字段的多个值使用 OR，不同字段之间使用 AND；来源、工序、缺陷、证据角色、语言和文档类型均可过滤。未知来源直接报错，结构性 Chunk 固定使用 `semantic_tag_excluded=false` 排除并写入追踪。
- Dense Cosine 分数与 Sparse lexical 分数保留为各自原始值，不直接比较。合法过滤后的空结果原样返回，不自动放宽过滤；跨语言查询出现 Sparse 空结果不视为执行失败，后续由 T10.9 Hybrid 融合和 T10.10 评测判断整体召回效果。

原因：在不提前耦合 Agent 或融合策略的前提下提供可测试、可追溯的双路候选检索，并避免跨语言词法空结果、隐藏查询改写或自动放宽过滤造成不可解释行为。

## D016 T10.9 等权RRF融合与稳定去重

- 保留Retrieval V1.1不变，新增Retrieval V1.2作为Hybrid正式契约；结果记录Dense/Sparse原始排名与分数、各通道RRF贡献、fusion_score、完整引用和候选数量轨迹。
- Hybrid固定获取Dense Top-20与Sparse Top-20，使用等权RRF且`k=60`；每个通道贡献为`1/(60+rank)`，缺失通道贡献为0。融合池保留Top-10，默认最终Top-5，并允许请求1～10条。
- 跨通道只按稳定`chunk_id`去重。当前Chunker没有滑动重叠，不执行模糊文本或页面级去重，以免删除同页不同条款；同一Chunk ID的正文、引用或Metadata不一致时直接报错。
- 融合分数相同时依次使用最优通道排名和Chunk ID排序，保证相同输入得到稳定结果。不为来源多样性强制替换结果，不自动放宽过滤，也不填充不相关候选。
- Sparse返回空列表时Hybrid自然退化为Dense排名，不视为执行失败。RRF分数只表示当前候选集合内的相对排序，不作为概率或置信度解释。
- T10.9不启用Reranker，`rerank_score`保持空值。按用户要求验证产物只生成最简Markdown报告，不生成T10.9 JSON摘要。

原因：在不直接比较Dense与Sparse异构原始分数的前提下获得可复现的双路融合结果，同时完整保留每条证据的来源、通道贡献和退化行为，供后续检索评测调参。

## D017 T10.10候选池与盲标规则

- Retrieval Evaluation基线对48个问题执行无业务Metadata过滤的检索；数据集中的缺陷、工序、语言、答案状态和评测重点只用于后续分组分析，不作为Retriever请求过滤条件。
- 每题实际取得Dense Top-20与Sparse Top-20，并按D016相同的等权RRF(k=60)计算评测专用Top-20；该扩展只用于Gold候选池，不修改生产Retriever的融合Top-10和最终Top-5配置。
- `rag/evaluation/evaluation_top20.csv`是候选及标签的唯一权威文件，固定48题×20候选共960行；既有Markdown、480行清单和人工标签全部作废。
- 相关性固定为`2=直接回答`、`1=部分或辅助证据`、`0=无关`；`final_answerable`表示当前20个候选整体是否足以回答问题。全部48题使用`qwen3.8-max`重新标注。
- 模型输入稳定打乱候选并隐藏RRF排名、通道分数和原始answerable标签；API结果必须通过候选全集、布尔值、枚举及语义一致性校验，失败重试3次后留空，不制造标签。
- 来源和页码由Retriever复制且不交给模型修改；Gold Chunk、Source和Page在模型标注完成并复核后由程序统一派生。无答案问题后续与可回答问题的Recall、MRR和nDCG分开统计。

原因：用更宽的候选池降低当前Top-10对Gold构建的自证偏差，以国内可用模型减少人工负担，同时让模型判断、来源页码和冻结检索配置保持可追溯。

## D018 T10.10 Gold冻结与指标口径

- `evaluation_top20.csv`继续作为唯一标签源；派生的`gold_dataset.json`记录源CSV SHA256及标注、知识库、索引、Embedding、Retriever和Fusion版本，CSV变化后必须显式重新冻结。
- `relevance=2`作为Recall@5与MRR@10的直接命中；nDCG@10使用0、1、2三级相关性和`2^relevance-1`增益。主指标只统计`candidate_answerable=true`查询，候选池不可回答查询单独报告。
- Source Hit@5与Page Hit@5使用生产最终Top-5；Source要求来源ID相同，Page要求来源相同且PDF页码范围相交。
- 评测重新运行无业务Metadata过滤的生产Hybrid Top-10。返回结果必须与冻结Gold的版本追踪一致，且全部位于已标注Top-20池内；出现未标注Chunk时拒绝生成指标。
- 当前Gold属于候选池Gold，不是知识库全部Chunk的穷尽标注；T10.10结果只作为当前配置的可复现基线和T10.11对比依据。

原因：明确直接答案、辅助证据、无答案问题和页码范围的统一计算口径，同时防止标签源、索引版本或未标注返回项变化后产生不可追溯的指标。

## D019 T10.10 知识库 V0.2 与增量 Gold

- 最终评测删除 Q010、Q027、Q030、Q037，保留 Q020，共 44 题；删除项不进入最终 Gold 或指标。
- 将 `txt_source/补充` 中三份资料纳入知识库，新增 34 页和 32 条 Chunk；V0.2 共 7 来源、216 页、215 条 Chunk。
- V0.2 使用独立 Collection `pcba_industrial_knowledge_v0_2` 和索引版本 `0.2.0`；V0.1 Collection 不覆盖、不删除。
- 旧候选只有在 query ID、Chunk ID、正文、来源和页码完全一致时复用原标签；其他候选全部交给 Qwen 增量标注，不以相似文本自动迁移标签。
- 最终 Gold、检索结果和指标使用 V0.2 版本追踪；最终报告是 T10.10 的唯一用户验收报告。

原因：在知识补充后避免重复标注未变化候选，同时保证新增或变化证据经过独立判断，并使 V0.1 基线与 V0.2 最终结果均可追溯。

## D020 补充论文的章节感知切分与 V0.3

- 原四份资料继续使用Page Blocks主线且183条Chunk不变；三篇双栏补充论文使用其原生逻辑文本顺序和逐篇核对的显式小标题表。
- 编号论文保留章、节、小节层级；无编号论文使用明确的版面标题。Chunk只能在同一真实小标题内组合，小节不足300 tokens时允许保留短Chunk。
- 三篇论文分别形成27、10、14个章节及33、13、16条Chunk，共62条；V0.3知识库合计245条Chunk。
- V0.3使用独立Collection `pcba_industrial_knowledge_v0_3`和索引版本`0.3.0`，V0.1/V0.2均保留。
- V0.3候选仍只复用query、Chunk ID、正文、来源及页码完全一致的标签；变化候选必须重新标注。

原因：双栏论文的通用Blocks会合并不同栏位，且过度标题保护会把整篇论文压成单一章节；来源级显式标题与逻辑顺序可恢复完整意思群，同时不改变旧资料的已验收切分。

---

## D021 T10.10 失败归因口径

- 失败归因只读取V0.3冻结Top-20标签池和生产Top-10结果，不修改资料、Chunk、Embedding、索引或Retriever。
- 有`relevance=2`时，Source/Page Hit@5以直接Gold为目标；无`relevance=2`时，以`relevance=1`辅助证据作为正确区域。若Top-20连辅助证据也没有，则Source/Page Hit留空，不臆造正确来源。
- 首个直接证据位于1～5名为`success`，位于6～20名为`ranking_failure`；无直接证据但正确Source/Page进入Top-5为`knowledge_or_question_gap`，存在正确区域但Top-5未命中为`recall_failure`。
- 已人工确认的6个负样本不触发Retriever优化：Q042、Q045、Q046记为`knowledge_gap`，Q044、Q047、Q048记为`rewrite`。

原因：把“知识存在但排序靠后”与“知识／问题边界不足”分开，避免将已确认的负样本或没有Gold区域的问题错误归咎于Retriever召回。

---

## D022 T10.10 评测范围收敛与Reranker全量重评

- 根据人工审查删除Q001、Q004、Q007、Q008、Q009、Q013、Q017、Q020、Q029、Q033、Q034共11题；保留Q019并改写为“为什么过大的焊膏沉积量可能在印刷和贴装后未出现桥连，却在回流后形成桥连？”。重评范围由44题收敛为33题。
- 全量实验固定使用原V0.3 RRF Top-20候选与Gold相关性标签，经`BAAI/bge-reranker-v2-m3`重排后取Top-10/Top-5；不修改Chunk、Embedding、Qdrant或生产Retriever。
- Q019的新文本仅用于Reranker打分。因遵循复用原Gold的要求，不重新生成其RRF候选或标签；原Q019无`relevance=2`，不进入27个可回答问题的主指标。
- Reranker是否进入生产配置必须在用户审核全量指标后另行决定，隔离实验结果不自动改变T10.9契约。

原因：在保留原Gold可比性的同时验证Reranker对排序的实际增益，并明确区分问题改写实验与重新检索、重新标注的完整评测。
