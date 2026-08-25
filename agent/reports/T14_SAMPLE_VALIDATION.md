# T14 真实样例闭环验证

验证日期：2026-08-25

## 结论

两条真实样例闭环均完成，Qwen、RAG、Neo4j KG、LangGraph checkpoint 和回流 Tool 全部使用正式实现，未出现错误或降级；结果已获用户确认，T14 结项。

## 桥连诊断样例

- 输入：`回流焊后发现焊锡桥连，请帮我诊断可能原因。`
- 缺陷：`short`。
- RAG：返回融合 Top-5，未降级。
- KG：返回 `wet_solder_paste_bridging` 强候选路径。
- 路由：因缺少 `manual_observation.paste_bridge` 返回 `needs_input`；声明无法提供后，同一线程恢复为 `completed`。
- 结论：该路径保持 `unverified/not_evaluated`，没有 Tool 调用，也没有伪造已确认根因。

## 元件偏移优化样例

- Qwen 从英文问题抽取缺陷：`shifted_component`。
- KG 同时保留 `placement_offset` 人工路径和 `reflow_thermal_imbalance` Tool 路径。
- 初始回流预测：PWI `105.46`，`qualified=false`，且处于训练域内。
- 优化建议：温区 `[135,145,155,165,173,180,190,220,230,230,249,270,265]`，链速 `91 cm/min`。
- 强制复验：PWI `24.23`，`qualified=true`，且处于训练域内。
- PWI 降幅：`77.02%`；推荐状态：`accepted`。
- Tool 轨迹严格为 `validation → optimization → revalidation`，未写入设备。
- 贴装路径仍为 `unverified/not_evaluated`，没有被回流 Tool 路径覆盖。

## 发现并修正的问题

首次运行中，Tool 的结构化推荐参数正确，但 Qwen 对温区变化的自然语言概括不准确。现已将具体数值、温区、链速和复验结果改为程序从 Tool 响应确定性生成，Qwen 只生成不含参数数值的诊断概括；修正后重新运行并通过一致性断言。

原始运行结果保存在被 Git 忽略的 `agent/storage/t14_sample_results.json`。
