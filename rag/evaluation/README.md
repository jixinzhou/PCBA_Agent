# PCBA RAG 检索评测

## 冻结状态

本目录保存已经冻结的 PCBA Retriever 主评测。正式评测目标是 16 道题，固定使用知识库 V0.3、RRF Top-20、`BAAI/bge-reranker-v2-m3` 和排名融合 `w=0.73`。冻结参数与文件哈希见 `FROZEN_MANIFEST.json`。

这里冻结的是评测数据、算法和计算口径，不代表已经修改生产 Retriever；生产配置变更仍需单独审核。

## 题集划分

| 题集 | 数量 | 文件 | 用途 |
|---|---:|---|---|
| Retriever 主评测 | 16 | `question/retriever_main.json` | 计算检索与重排指标 |
| 多证据评测 | 3 | `question/multi_evidence.json` | 后续评估多段证据组合 |
| Agent/KG 评测 | 2 | `question/agent_kg.json` | 后续评估 Agent 与知识图谱协同 |
| No-answer / Abstention | 3 | `question/no_answer_abstention.json` | 后续评估拒答能力 |

Retriever 主评测题号为：Q002、Q003、Q005、Q011、Q012、Q015、Q018、Q019、Q021、Q022、Q024、Q026、Q028、Q031、Q035、Q036。

三道题使用冻结改写：

- Q019：为什么过大的焊膏沉积量可能在印刷和贴装后未出现桥连，却在回流后形成桥连？
- Q024：为什么回流过程中总焊料量相对于焊点间距过大会导致桥连？
- Q036：Can an initial component placement offset change during reflow due to solder self-alignment

## 候选池与标注

候选池 `question/retriever_main_top20_v0.3.csv` 共 16×20=320 行：

1. Dense 与 Sparse 各检索 Top-20。
2. 使用等权 RRF，`k=60`。
3. 截取融合后的 Top-20 作为冻结候选池。
4. 13 道未改写题的 260 条候选，在问题文本、Chunk ID 和正文完全一致时复用 V0.3 标签。
5. Q019、Q024、Q036 的 60 条候选按改写后的问题重新调用 Qwen 标注。

320 条候选均为 `annotation_status=completed`，标签分布为：rel=0 共 185 条、rel=1 共 92 条、rel=2 共 43 条。

相关性定义：

- `rel=2`：可以直接回答问题的证据；用于 Recall、MRR、Source Hit 和 Page Hit。
- `rel=1`：相关但不能独立直接回答的辅助证据；参与 nDCG。
- `rel=0`：无关证据。

Gold 文件为 `question/retriever_main_gold_v0.3.json`，记录候选池 SHA256、知识库、索引、Embedding、Retriever、Fusion 和标注模型版本。

## 排名方法

评测比较三种顺序：

1. RRF Retriever：直接使用原 RRF 排名。
2. 纯 Reranker：使用 BGE Reranker 分数排序。
3. 固定排名融合：

```text
fusion_score = 0.27 / (60 + retrieval_rank)
             + 0.73 / (60 + reranker_rank)
```

最终取融合 Top-5。`w=0.73` 位于本次 Macro Recall@5 最优区间 `0.72–0.74`，因此冻结为 Recall 优先方案。

## 指标口径

- Macro Recall@5：每题先计算 `Top-5命中rel=2数 / 该题rel=2总数`，再对可回答题取平均。
- Micro rel=2 Recall@5：全部题 Top-5 命中的 rel=2 总数除以 rel=2 总数。
- MRR@10：首条 rel=2 的倒数排名，对可回答题取平均。
- nDCG@10：使用 0/1/2 分级标签，增益为 `2^rel-1`。
- Source Hit@5 / Page Hit@5：Top-5 是否命中直接 Gold 的来源或相交页码。

16 题中有 14 题包含 rel=2，共 43 条直接 Gold。Q019、Q024 在当前 Top-20 中没有 rel=2，被单列为候选池不可回答题，不强行按 Recall=0 计入排序主指标。正式题集的直接 Gold 覆盖率为 14/16=87.5%。

## 冻结结果

| 方法 | Macro Recall@5 | Micro rel=2 Recall@5 | rel=2命中 | Top-5命中题 | MRR@10 | nDCG@10 | Source Hit@5 | Page Hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RRF Retriever | 0.3625 | 0.3953 | 17/43 | 9/14 | 0.4806 | 0.5821 | 0.7857 | 0.7143 |
| 纯 Reranker | 0.7364 | 0.5814 | 25/43 | 14/14 | 0.6333 | 0.7891 | 1.0000 | 1.0000 |
| 排名融合 `w=0.73` | **0.7622** | **0.6279** | **27/43** | **14/14** | 0.5857 | 0.7725 | 1.0000 | 1.0000 |

结论：Reranker 明显有效；`w=0.73` 在 Recall 上最好，纯 Reranker 的 MRR 和 nDCG 更高。当前冻结方案以 Recall 优先，采用 `w=0.73`。

简表保存在 `question/retriever_main_metrics_summary.json`，完整 0.00–1.00 权重网格保存在 `question/retriever_main_rank_fusion_metrics.json`。

## 复现方式

所有命令均在项目根目录、Conda 环境 `PCB_Agent` 中执行。冻结文件不得直接覆盖；需要更新时先复制到新版本文件，并完成全量重标注和重新冻结。

```powershell
conda run -n PCB_Agent python rag/evaluation/build_top20_dataset.py `
  --input rag/evaluation/question/retriever_main.json `
  --output <new_candidate_pool.csv>

conda run -n PCB_Agent python rag/evaluation/annotate_with_qwen.py `
  --csv <new_candidate_pool.csv>

conda run -n PCB_Agent python rag/evaluation/evaluate_retriever.py `
  --csv <new_candidate_pool.csv> `
  --gold <new_gold.json> `
  --freeze-gold --gold-only

# 复制两份正式 YAML 为新版本并修改候选池、Gold 和输出路径后运行：
conda run -n PCB_Agent python rag/evaluation/run_reranker_rank_fusion_experiment.py `
  --config <new_rank_fusion_config.yaml>
```

测试命令：

```powershell
conda run -n PCB_Agent python -m unittest discover -s rag/evaluation -p "test_*.py"
conda run -n PCB_Agent python rag/evaluation/verify_frozen_evaluation.py
```

## 目录规则

- `question/`：唯一正式题集、标注候选池、Gold 和指标。
- `archive/legacy_labeled/`：旧轮次已标注数据与对应 Gold，仅供追溯，不参与正式指标。
- 根目录 Python 文件：生成候选、标注、冻结 Gold、Reranker 与排名融合的最小可复现链路。
- `.env`：本地标注服务凭据，不纳入评测数据，也不得公开。

任何正式题目、标签、候选池、Gold、模型 revision、RRF 参数或融合权重发生变化，都必须更新版本号、重新计算全部指标，并重建 `FROZEN_MANIFEST.json`；否则不得称为同一冻结评测。
