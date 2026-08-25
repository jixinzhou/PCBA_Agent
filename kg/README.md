# PCBA 缺陷致因知识图谱

本目录保存 T11 知识图谱的框架无关实现。T11.1 冻结图模型与查询契约，T11.2 完成真实 Neo4j 幂等导入，T11.3 提供不依赖 LangGraph 的只读查询接口。

## 权威来源

- 因果事实唯一来源：`ontology/pcba_defect_causality.v1.1.yaml`
- 术语辅助来源：`rag/schemas/entity_dictionary.json`
- Tool 输入契约来源：`tool/agent_tools/models.py`
- T11.1 不使用 LLM 创建实体或关系，也不把 RAG 关键词匹配结果自动提升为图谱证据。

## 核心图模型

候选致因关系被建模为 `CausalHypothesis` 节点，而不是确定性的 `CAUSED_BY` 边：

```text
(Defect)-[:HAS_HYPOTHESIS]->(CausalHypothesis)
(CausalHypothesis)-[:PROPOSES_CAUSE]->(CandidateCause)
(CandidateCause)-[:BELONGS_TO]->(Process)
(CausalHypothesis)-[:REQUIRES_METRIC]->(QualityMetric)
(CausalHypothesis)-[:VALIDATED_BY]->(Tool)
(CausalHypothesis)-[:OPTIMIZED_BY]->(Tool)
```

所有本体实体同时带有 `KnowledgeEntity` 标签，并使用 `entity_id` 作为幂等键；`CausalHypothesis` 使用 `relationship_id` 作为幂等键。

当前有效范围为 3 个工序、4 个缺陷、5 个候选原因、7 个验证指标、4 个验证/优化 Tool 和 5 个候选假设。缺陷分类 Tool 属于诊断入口，不属于致因验证图。

## 查询语义

查询结果允许一个缺陷返回多条 `candidates`，并用 `required_controls` 汇总所有候选路径涉及的工序指标。当前 `shifted_component` 必须同时返回：

1. `placement_offset`：`strong + unverified + manual_inspection`；
2. `reflow_thermal_imbalance`：`conditional + tool_supported`，根据输入完整性返回 `request_missing_data` 或 `invoke_tool`。

`verification_capability=tool_supported` 只表示存在可用于验证的 Tool，不表示当前案例已经被验证。T11.1 示例的 `assessment_status` 均为 `not_evaluated`。

`required_controls` 的数组结构允许未来同一缺陷同时汇总 VTE 和 PWI，但 T11.1 不新增 T09 中不存在的具体因果事实。

## 证据边界

每条候选路径记录本体路径、版本和 SHA256。`evidence_refs` 只接受经过审核的 Chunk 引用；当前没有权威 `relationship_id -> chunk_id` 映射，因此正式示例保持空数组。

## T11.1 文件

- `config/neo4j_mapping.v1.yaml`：本体到属性图的确定性映射。
- `schemas/causal_query.v1.schema.json`：Agent 可消费的查询响应 Schema。
- `cypher/schema.v1.cypher`：唯一约束与查询索引。
- `examples/`：SPI Tool、人工检查和多候选路径示例。
- `tests/test_kg_contract.py`：本体、映射、Schema、示例与 Tool 名称一致性测试。

## 当前不做

- 不由 KG 调用预测或优化 Tool，不根据工艺数据值自行确认根因。
- 不实现 LangGraph 节点或诊断状态机。
- 不导入本体外的 `vte_variance`，不启用待确认的 VTE/PWI 阈值。
- 不自动绑定 RAG Chunk。

## T11.2 本地运行

运行基线固定为 Neo4j Community `2026.07.1` 和官方 Python Driver `6.2.0`。本地凭据位于被 Git 忽略的 `kg/.env`，仓库仅保存 `kg/.env.example`；图数据位于可重建 named volume `pcba_neo4j_data`。

```powershell
conda run -n PCB_Agent python -m pip install -r kg/requirements.txt
docker compose --env-file kg/.env -f kg/docker-compose.neo4j.yml up -d
conda run -n PCB_Agent python kg/scripts/import_ontology.py --runs 2
```

`--runs 2` 会在每轮 `MERGE` 后验证节点、关系、五条候选路径、两条无 Tool 路径和 `shifted_component` 双路径，并比较完整图指纹。停止容器但保留数据可运行：

```powershell
docker compose --env-file kg/.env -f kg/docker-compose.neo4j.yml stop
```

Neo4j Browser 默认地址为 `http://127.0.0.1:7474`，Bolt 默认地址为 `bolt://127.0.0.1:7687`；端口均可在 `kg/.env` 覆盖。

T11.2 已依据本目录契约完成真实 Neo4j 幂等导入。

## T11.3 查询接口

按缺陷查询全部候选路径：

```powershell
conda run -n PCB_Agent python kg/scripts/query_causal_paths.py --defect shifted_component
```

按关系 ID 查看单条路径：

```powershell
conda run -n PCB_Agent python kg/scripts/query_causal_paths.py --defect shifted_component --relationship-id REL-SHIFTED-COMPONENT-REFLOW
```

可通过 `--observations-file` 传入 UTF-8 JSON 对象。接口支持嵌套字段或点路径字段，例如：

```json
{
  "input": {
    "points": [{"point_id": "P1"}],
    "zone_means_c": [150, 160, 170, 180, 190, 200, 210, 220, 230, 235, 240, 240, 235],
    "belt_speed_cm_min": 85
  }
}
```

Python 调用入口为 `pcba_kg.query_causal_paths()`；返回严格遵循 `schemas/causal_query.v1.schema.json`。KG 只根据必填路径是否存在返回 `request_missing_data`、`invoke_tool` 或 `manual_inspection`，不执行 Tool，也不解释数值是否异常。
