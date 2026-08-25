# RAG V0.1

本目录保存 T10 工业知识检索层的可版本化配置与数据契约。T10 V0.1 只允许 `config/sources.v0.1.yaml` 中启用的四份 PDF 进入后续处理链路。

## 目录约定

```text
rag/
├── config/                 # 受版本控制的来源与流水线配置
├── schemas/                # 受版本控制的数据契约
│   └── examples/           # 不含真实文档内容的最小合同示例
├── src/                    # 框架无关的RAG实现
├── scripts/                # 命令行入口
├── tests/                  # 自动化测试
├── reports/                # 不含全文的验收报告
├── data/processed/         # PDF 页面、Chunk 等派生产物，不进入 Git
├── storage/                # 向量数据库文件，不进入 Git
├── cache/                  # OCR、模型等缓存，不进入 Git
└── logs/                   # 运行日志，不进入 Git
```

原始 PDF 继续位于 `txt_source/`，本目录不复制或修改原文件。

## 合同边界

- `page.v1.schema.json`：页面来源、文本、解析方式和质量状态。
- `chunk.v1.schema.json`：内容边界、章节、页码、稳定 ID 和 Metadata。
- `metadata.v1.schema.json`：来源属性、检索标签和处理追踪。
- `chunk.v1.1.schema.json` / `metadata.v1.1.schema.json`：T10.5 双语实体、映射版本与完整检索标签。
- `mapping_trace.v1.schema.json`：逐 Chunk 术语命中类别、规则和最终标签追踪。
- `retrieval.v1.schema.json`：检索请求、结果、引用、评分和检索轨迹。
- `retrieval.v1.1.schema.json`：T10.8 Dense/Sparse 检索通道、Metadata V1.1 和系统过滤追踪。
- `retrieval.v1.2.schema.json`：T10.9 Hybrid结果、两路排名、RRF贡献和融合轨迹。

Schema V1 仅定义当前确认的必填字段。缺陷别名、相关术语和候选致因不属于 Metadata；后续直接复用 T09 本体，不在 T10 中重复定义。

## 数据权威关系

- `sources.v0.1.yaml` 是 V0.1 来源白名单和来源属性的权威记录。
- Page 是 PDF 页码和页面处理状态的权威记录。
- Chunk 是章节路径和内容页码范围的权威记录。
- Retriever Result 中的 Citation 由来源清单和 Chunk 生成，不独立维护。
- Vector DB 是可重建派生产物，不是唯一数据源。

运行时还需验证 `pdf_page_end >= pdf_page_start`，以及 Chunk 的 `page_ids` 与页码范围一致；这些跨字段关系不由 JSON Schema 单独表达。

## T10.2 样本验证

T10.2 使用 pypdf 进行原生文本提取，使用 pypdfium2 渲染低文本页面，并通过 RapidOCR + ONNX Runtime 在本地执行中英文 OCR。表格页按照 OCR 文本框坐标进行行列排序，仍输出 Page 文本，不生成 Chunk。

在 `PCB_Agent` 环境运行：

```powershell
python rag/scripts/run_pdf_sample.py
python -m unittest discover -s rag/tests -p "test_*.py"
```

## T10.3 全量页面与结构数据

T10.3 使用 Page V1.1 和 Block V1 保存印刷页码、空白页状态、页面结构块、
归一化坐标和阅读顺序。全量派生数据写入 `rag/data/processed/pages/`，断点状态写入
`rag/data/processed/checkpoints/`，两者均不进入 Git。T10.3 不生成 Chunk 或术语 Metadata。

在 `PCB_Agent` 环境运行：

```powershell
python rag/scripts/run_full_page_pipeline.py
python -m unittest discover -s rag/tests -p "test_*.py"
```

默认启用断点续跑。只有需要丢弃所选来源的既有 T10.3 派生数据并重新处理时，才使用
`--restart`；可通过重复传入 `--source-id` 限定来源。

## T10.4 结构化 Chunk 与稳定 ID

T10.4 按 Page V1.1 的 Block 阅读顺序构建章节感知 Chunk。页眉、页脚和页码不进入
Chunk；表题与表格、列表项与续接段落作为语义原子单元。目标长度为 300～600 个
BGE-M3 tokens，普通文本超过 800 tokens 时递归拆分；不可拆分的表格、列表和公式保留
完整并在报告中告警。

BGE-M3 Tokenizer 固定为 `BAAI/bge-m3` revision
`5617a9f61b028005a4858fdac845db406aefb181`。首次运行若本地不存在，仅下载 Tokenizer
文件到 `rag/cache/bge-m3-tokenizer/`。Chunk 数据写入
`rag/data/processed/chunks/`，不进入 Git。

```powershell
python rag/scripts/run_chunk_pipeline.py
python -m unittest discover -s rag/tests -p "test_*.py"
```

相同 Page、配置和 Chunker 版本会生成相同内容哈希 ID。T10.4 仅填写来源级 Metadata；
`process_ids`、`defect_ids` 和 T09 术语映射由 T10.5 完成。

## T10.5 Chunk Metadata 与 T09 术语映射

T10.5 读取 T10.4 Chunk V1，通过 `config/metadata_mapping.v0.1.yaml`、T09 V1.1
权威本体和实体词典执行确定性映射。输出 Chunk V1.1 到
`rag/data/processed/chunks_enriched/`，逐 Chunk 映射追踪写入
`rag/data/processed/metadata_traces/`；两类派生数据均不进入 Git。

同义词、相关术语和候选原因在追踪记录中保持不同类别。英文短语采用词边界，歧义裸词
被禁用，结构性区段不添加语义标签；单个 Chunk 可以具有多个工序或缺陷标签。

```powershell
python rag/scripts/run_metadata_pipeline.py
python -m unittest discover -s rag/tests -p "test_*.py"
```

## T10.6 BGE-M3 Dense 与 Sparse Embedding

T10.6 使用本地 `BAAI/bge-m3` 固定 revision，通过 `FlagEmbedding` 同时生成
1024 维归一化 Dense 和 Sparse lexical weights。输入为章节路径与 Chunk 正文，
不注入规则 Metadata；V0.1 不生成 ColBERT 多向量。

模型缓存位于 `rag/cache/bge-m3-model/`，Embedding 派生数据写入
`rag/data/processed/embeddings/`，均不进入 Git。每条记录保存 Chunk ID、文本哈希、
输入哈希、模型 revision、配置版本和设备精度。

统一环境使用 Torch 2.6.0+cu118。首次重建环境时应使用 `requirements.txt` 中配置的
PyTorch CUDA 11.8 附加索引。

```powershell
python rag/scripts/run_embedding_pipeline.py
python -m unittest discover -s rag/tests -p "test_*.py"
```

## T10.7 Qdrant V0.1 向量索引

T10.7 通过 Docker Compose 运行固定版本的 Qdrant 服务，将每个 Chunk 保存为一个
Point。Point 使用由 Chunk ID 确定生成的 UUIDv5，并同时保存命名 Dense、Sparse 向量
和完整 Chunk Payload。Dense 在写入 Cosine Collection 前显式执行与 Qdrant 一致的 L2
归一化。Qdrant named volume 是可重建派生产物，不进入 Git。

```powershell
docker compose -f rag/docker-compose.qdrant.yml up -d
python rag/scripts/run_qdrant_pipeline.py
python -m unittest discover -s rag/tests -p "test_*.py"
```

默认运行对兼容 Collection 执行幂等 upsert 并清理不属于当前输入的旧 Point。只有明确
需要删除并完整重建派生 Collection 时才使用 `--recreate`。服务仅绑定本机
`127.0.0.1:6333/6334`；未来对外部署时必须另行配置认证与 TLS。

## T10.8 框架无关 Retriever

T10.8 使用固定 revision 的本地 BGE-M3 `encode_queries` 生成查询 Dense 与 Sparse，
并通过 Qdrant 分别返回原始候选。查询只执行 NFKC、首尾空白清理和连续空白合并；不翻译、
不改写，也不调用 LLM。同一过滤字段内为 OR，不同字段间为 AND；结构性 Chunk 固定排除。

```powershell
python rag/scripts/run_retriever.py "焊膏印刷少锡的常见原因" --mode both --top-k 5
python rag/scripts/validate_retriever.py
python -m unittest discover -s rag/tests -p "test_*.py"
```

`retrieve_dense` 与 `retrieve_sparse` 返回 Retrieval V1.1；Dense 与 Sparse 原始分数不可
直接比较。合法过滤后的空结果原样返回，不自动翻译查询或放宽过滤条件。融合、去重、RRF
和最终 Top-K 留到 T10.9。

## T10.9 Hybrid RRF融合

T10.9 复用同一次查询向量，分别获取Dense Top-20和Sparse Top-20，以固定`k=60`、
等权重RRF进行融合。跨通道候选只按稳定`chunk_id`合并；当前Chunk没有滑动重叠，
不做可能误删不同规范条款的近似文本去重。融合Top-10后默认返回Top-5。

```powershell
python rag/scripts/run_retriever.py "What causes solder paste bridging?" --mode hybrid
python rag/scripts/validate_fusion.py
python -m unittest discover -s rag/tests -p "test_*.py"
```

Hybrid响应使用Retrieval V1.2，保留两路原始排名、分数、各自RRF贡献、最终融合分数和
完整引用轨迹。Sparse为空时自然退化为Dense结果，不填充不相关候选；不启用来源多样性
或Reranker。T10.9只生成最简Markdown验证报告，不生成JSON摘要。

## T10.10 Retrieval Evaluation

正式评测已冻结为16道Retriever主评测题。权威标注池、Gold、指标、固定参数、文件哈希和
完整复现说明统一位于`rag/evaluation/`，详见`rag/evaluation/README.md`与
`rag/evaluation/FROZEN_MANIFEST.json`。旧轮次已标注数据仅在
`rag/evaluation/archive/legacy_labeled/`中保留用于追溯，不再参与正式指标。
