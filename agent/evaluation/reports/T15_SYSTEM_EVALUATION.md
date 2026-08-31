# T15 智能体系统评测报告

## 结论

T15 冻结集修复后复评**全部通过**：6 个基础 Case 严格通过 `6/6`，1 个 Tool 失败变体通过全部安全降级检查。

本轮证明当前 Agent 可以完成最小系统闭环：图片缺陷识别、RAG 检索、KG 候选路径生成、工艺 Tool 选择、缺失输入补问、候选状态判定、参数优化后复验，以及 Tool 失败时保守降级。T15 可以结项并进入 T16。

## 评测范围与运行配置

| 项目 | 值 |
|---|---:|
| 基础 Case | 6 |
| Tool 失败变体 | 1 |
| 覆盖缺陷 | 少锡、多锡、短路、元件偏移 |
| 数据集状态 | `frozen_by_user_approval` |
| 数据集 SHA256 | `86445bba9e6969c10d3b4350c7aca5410e2f567a668d16f4ac9e5dd6dea0e655` |
| Evaluation ID | `t15-1788142079` |
| 执行时间 | 2026-08-31 10:07:59—10:09:23（Asia/Shanghai） |
| Qwen | `qwen3.7-flash-2026-07-15` |
| RAG 排名 | RRF Top-20 → BGE Reranker → `w=0.73` 排名融合 → Top-5 |
| VTE 判定 | `<95%`支持少锡；`>105%`支持多锡；中间区间保持不确定 |

原始结果：[`t15_evaluation_results.v0.1.json`](../results/t15_evaluation_results.v0.1.json)

## 核心指标

| 指标 | 结果 |
|---|---:|
| 严格 Case 通过率 | `6/6 = 100%` |
| 全部 Tool Call Recall | `11/11 = 100%` |
| 全部 Tool Call Precision | `11/11 = 100%` |
| Tool Call F1 | `100%` |
| 不必要 Tool 调用率 | `0/11 = 0%` |
| 完整 Tool 序列一致率 | `6/6 = 100%` |
| 工艺 Tool Recall（排除 AOI） | `5/5 = 100%` |
| 工艺 Tool 序列一致率（排除 AOI） | `6/6 = 100%` |
| AOI 分类调用 Recall | `6/6 = 100%` |
| 缺陷名称识别准确率 | `6/6 = 100%` |
| 缺失输入精确匹配率 | `4/4 = 100%` |
| 候选因果状态准确率 | `4/4 = 100%` |
| 优化后复验率 | `1/1 = 100%` |
| RAG 引用完整率 | `30/30 = 100%` |
| Tool 失败安全降级率 | `1/1 = 100%` |

说明：本轮 T15 检查 RAG 是否返回证据及引用字段是否完整，不重新评估证据语义相关性；语义检索质量仍以冻结的 T10 指标为准。

## 逐 Case 结果

| Case | 实际 Tool 路径 | 关键结果 | 严格通过 |
|---|---|---|---:|
| C01 少锡 + SPI 低 VTE | `AOI → SPI预测` | AOI确认少锡；VTE `93.0103% < 95%`，少锡印刷候选为 `supported` | 是 |
| C02 多锡 + SPI 高 VTE | `AOI → SPI预测` | AOI确认多锡；VTE `107.0934% > 105%`，多锡印刷候选为 `supported` | 是 |
| C03 短路 + 无 Tool 路径 | `AOI` | AOI确认短路；正确不调用工艺 Tool；人工观测路径保持未验证 | 是 |
| C04 偏移 + PWI 105.46 | `AOI → Reflow预测 → 优化 → Reflow复验` | AOI确认偏移；异常支持回流候选；优化后复验完成 | 是 |
| C05 多锡 + 缺 SPI 数据 | `AOI` | AOI确认多锡；精确补问 4 个 SPI 字段；未误调工艺 Tool | 是 |
| C06 偏移 + 缺 Reflow 数据 | `AOI` | AOI确认偏移；精确补问 Reflow 和人工贴装观测；未误调工艺 Tool | 是 |
| V01 Reflow Tool 超时 | `AOI → Reflow预测失败` | 候选保持 `inconclusive`；未继续优化；未生成虚假确认结论 | 是 |

6 个基础 Case 的图像结果均与文本缺陷一致，运行状态中的缺陷来源均为 `aoi_tool_confirmed`。每个基础 Case 返回 5 条 RAG 证据，且每条均含来源和页码。

## 本轮修复

### AOI 图文校验

图片存在时始终调用 AOI，不再因为问题文字已包含缺陷名称而跳过图像分类：

- 图文一致：记录为 `aoi_tool_confirmed`；
- 普通问题文字与 AOI 冲突：采用 AOI 结果并记录 `defect_evidence_conflict`；
- 用户显式提供 `provided_defect` 与 AOI 冲突：保留显式缺陷并记录冲突；
- AOI 失败或返回本体外类别：保留可用文本信息并显式记录降级。

### VTE 候选路径判定

SPI 判定同时读取 `data.vte_mean`、`within_training_domain` 和稳定的 `relationship_id`：

- 少锡印刷候选且 `VTE < 95%`：`supported`；
- 多锡印刷候选且 `VTE > 105%`：`supported`；
- `95% ≤ VTE ≤ 105%`、方向不匹配、训练域外或数值无效：`inconclusive`；
- VTE 只支持候选路径，不声明唯一真实根因。

修复相关的 14 项 Agent 单元测试全部通过，覆盖 VTE 两侧阈值、等于阈值、方向不匹配、训练域外、无效值、图片强制调用和图文冲突。

## 验收判断

T15 的既定验收范围已满足：四类缺陷、无 Tool 关系、缺失数据、工艺验证、优化复验、RAG 引用、Tool 失败和证据不足场景均已覆盖并通过。冻结数据集未修改，复评结果可复现。
