from __future__ import annotations

from copy import deepcopy
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .models import AgentState
from .policies import (
    assess_prediction, deep_merge, infer_defect_from_text, spi_vte_in_target,
    unique_missing,
)
from .qwen_client import QwenUnavailableError


class PCBAAgentGraph:
    def __init__(self, *, settings: Any, qwen: Any, rag: Any, kg: Any, tools: Any) -> None:
        self.settings = settings
        self.qwen = qwen
        self.rag = rag
        self.kg = kg
        self.tools = tools

    def build(self, checkpointer: Any) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("prepare", self.prepare)
        graph.add_node("request_defect", self.request_defect)
        graph.add_node("finalize_unknown", self.finalize_unknown)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("query_kg", self.query_kg)
        graph.add_node("execute", self.execute)
        graph.add_node("request_inputs", self.request_inputs)
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "prepare")
        graph.add_conditional_edges(
            "prepare", lambda state: "retrieve" if state.get("defect_name") else "request_defect"
        )
        graph.add_conditional_edges(
            "request_defect", lambda state: "retrieve" if state.get("defect_name") else "finalize_unknown"
        )
        graph.add_edge("retrieve", "query_kg")
        graph.add_edge("query_kg", "execute")
        graph.add_conditional_edges(
            "execute", lambda state: "request_inputs" if state.get("pending_inputs") else "finalize"
        )
        graph.add_edge("request_inputs", "query_kg")
        graph.add_edge("finalize", END)
        graph.add_edge("finalize_unknown", END)
        return graph.compile(checkpointer=checkpointer)

    def prepare(self, state: AgentState) -> dict[str, Any]:
        request = state["request"]
        observations = deep_merge(request.get("observations", {}), state.get("observations", {}))
        errors = list(state.get("errors", []))
        degraded = list(state.get("degradation_trace", []))
        trace = list(state.get("tool_trace", []))
        defect = request.get("provided_defect") or state.get("defect_name")
        source = "provided" if request.get("provided_defect") else state.get("defect_source")
        question = request.get("user_question")
        if not defect:
            defect = infer_defect_from_text(question)
            source = "deterministic_text" if defect else None
        if question:
            try:
                extracted = self.qwen.extract(question)
                defect = defect or extracted.get("defect")
                source = source or ("qwen_text" if defect else None)
                observations = deep_merge(extracted.get("observations", {}), observations)
                if request.get("goal") == "diagnose" and extracted.get("goal"):
                    request = {**request, "goal": extracted["goal"]}
            except QwenUnavailableError as exc:
                degraded.append({"stage": "qwen_extraction", "reason": str(exc)})
        image_path = request.get("image_path")
        if image_path:
            try:
                response = self.tools.classify(image_path, request["request_id"])
                data = response.get("data") or {}
                predicted = data.get("predicted_class") or {}
                raw = predicted.get("class_name_en")
                image_defect = infer_defect_from_text(raw) or raw
                if image_defect not in self.settings.raw["routing"]["supported_defects"]:
                    image_defect = None
                if image_defect:
                    if defect and defect != image_defect:
                        keep_provided = bool(request.get("provided_defect"))
                        degraded.append({
                            "stage": "defect_evidence_conflict",
                            "reason": (
                                f"文本/显式缺陷={defect}，AOI缺陷={image_defect}。"
                            ),
                            "resolution": "provided_defect" if keep_provided else "aoi_tool",
                        })
                        if keep_provided:
                            source = "provided_aoi_conflict"
                        else:
                            defect = image_defect
                            source = "aoi_tool_text_conflict"
                    else:
                        defect = image_defect
                        source = "aoi_tool_confirmed" if source else "aoi_tool"
                    trace.append({
                        "phase": "defect_classification",
                        "tool_name": "pcba_defect_classification",
                        "success": True,
                        "predicted_defect": image_defect,
                    })
                else:
                    degraded.append({
                        "stage": "aoi_classification",
                        "reason": "AOI未返回冻结本体支持的缺陷类别。",
                        "fallback": source or "request_defect",
                    })
                    trace.append({
                        "phase": "defect_classification",
                        "tool_name": "pcba_defect_classification",
                        "success": True,
                        "predicted_defect": None,
                    })
            except Exception as exc:
                errors.append({"stage": "aoi_classification", "error": str(exc)})
                trace.append({
                    "phase": "defect_classification",
                    "tool_name": "pcba_defect_classification",
                    "success": False,
                    "error": str(exc),
                })
        return {
            "request": request,
            "observations": observations,
            "defect_name": defect,
            "defect_source": source,
            "unavailable_inputs": list(state.get("unavailable_inputs", [])),
            "tool_trace": trace,
            "degradation_trace": degraded,
            "errors": errors,
            "validation_records": dict(state.get("validation_records", {})),
            "optimization_records": dict(state.get("optimization_records", {})),
        }

    def request_defect(self, state: AgentState) -> dict[str, Any]:
        fallback = "无法确定缺陷类型，请补充 insufficient_solder、excessive_solder、short 或 shifted_component。"
        try:
            prompt = self.qwen.clarify(["provided_defect"], {"supported_defects": self.settings.raw["routing"]["supported_defects"]})
        except (QwenUnavailableError, AttributeError):
            prompt = fallback
        resume = interrupt({
            "reason": "unknown_defect",
            "missing_inputs": ["provided_defect"],
            "prompt": prompt,
        })
        defect = resume.get("provided_defect") if isinstance(resume, dict) else None
        observations = resume.get("observations", {}) if isinstance(resume, dict) else {}
        unavailable = resume.get("unavailable_inputs", []) if isinstance(resume, dict) else []
        return {
            "defect_name": defect,
            "defect_source": "resume",
            "observations": deep_merge(state.get("observations", {}), observations),
            "unavailable_inputs": sorted(set(state.get("unavailable_inputs", []) + unavailable)),
        }

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        request = state["request"]
        query = request.get("user_question") or f"PCBA {state['defect_name']} causes and process controls"
        result = self.rag.retrieve(query)
        degraded = list(state.get("degradation_trace", []))
        if result.get("degraded"):
            degraded.append({
                "stage": result.get("stage"), "reason": result.get("error"),
                "fallback": "rrf_top5" if result.get("stage") == "reranker" else "kg_only",
            })
        return {"rag_evidence": result.get("evidence", []), "degradation_trace": degraded}

    def query_kg(self, state: AgentState) -> dict[str, Any]:
        try:
            response = self.kg.query(state["defect_name"], state.get("observations", {}))
            return {"kg_response": response, "candidates": response["candidates"]}
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append({"stage": "kg_query", "error": str(exc)})
            return {"kg_response": {}, "candidates": [], "errors": errors}

    def execute(self, state: AgentState) -> dict[str, Any]:
        observations = state.get("observations", {})
        unavailable = state.get("unavailable_inputs", [])
        validations = deepcopy(state.get("validation_records", {}))
        optimizations = deepcopy(state.get("optimization_records", {}))
        trace = list(state.get("tool_trace", []))
        errors = list(state.get("errors", []))
        missing: list[str] = []
        for candidate in state.get("candidates", []):
            relationship_id = candidate["relationship_id"]
            action = candidate["validation_action"]
            candidate_missing = unique_missing(action.get("missing_inputs", []), unavailable)
            if candidate_missing:
                missing.extend(candidate_missing)
                continue
            tool_name = action.get("tool_name")
            if action.get("action_type") == "invoke_tool" and relationship_id not in validations:
                arguments = {"request_id": state["request"]["request_id"], "input": observations["input"]}
                if tool_name == "reflow_profile_prediction":
                    arguments["options"] = {"return_temperature_curve": False}
                try:
                    response = self.tools.invoke(tool_name, arguments)
                    status, reason = assess_prediction(
                        tool_name,
                        response,
                        relationship_id=relationship_id,
                    )
                    validations[relationship_id] = {
                        "tool_name": tool_name, "assessment_status": status,
                        "reason": reason, "response": response,
                    }
                    trace.append({"phase": "validation", "relationship_id": relationship_id, "tool_name": tool_name, "success": True})
                except Exception as exc:
                    validations[relationship_id] = {
                        "tool_name": tool_name, "assessment_status": "inconclusive",
                        "reason": "Tool调用失败，候选路径保持不确定。", "error": str(exc),
                    }
                    trace.append({"phase": "validation", "relationship_id": relationship_id, "tool_name": tool_name, "success": False, "error": str(exc)})
                    errors.append({"stage": "validation_tool", "relationship_id": relationship_id, "error": str(exc)})

            record = validations.get(relationship_id, {})
            opt_action = candidate.get("optimization_action") or {}
            if (
                state["request"].get("goal") == "diagnose_and_optimize"
                and record.get("assessment_status") == "supported"
                and opt_action.get("tool_name")
                and relationship_id not in optimizations
            ):
                optimization_fields = (
                    () if opt_action.get("tool_name") == "spi_parameter_optimization"
                    else ("optimization_target", "adjustable_parameters")
                )
                for path in optimization_fields:
                    if not observations.get(path) and path not in unavailable:
                        missing.append(path)
                if (
                    not any(path in missing for path in optimization_fields)
                    and not any(path in unavailable for path in optimization_fields)
                ):
                    optimizer = (
                        self._optimize_spi
                        if opt_action.get("tool_name") == "spi_parameter_optimization"
                        else self._optimize_reflow
                    )
                    optimizer(state, candidate, observations, record, optimizations, trace, errors)
        return {
            "validation_records": validations,
            "optimization_records": optimizations,
            "pending_inputs": unique_missing(missing, unavailable),
            "pending_reason": "candidate_validation_or_optimization",
            "tool_trace": trace,
            "errors": errors,
        }

    def _optimize_spi(
        self, state: AgentState, candidate: dict[str, Any], observations: dict[str, Any],
        validation: dict[str, Any], optimizations: dict[str, dict[str, Any]],
        trace: list[dict[str, Any]], errors: list[dict[str, Any]],
    ) -> None:
        rid = candidate["relationship_id"]
        tool_name = candidate["optimization_action"]["tool_name"]
        try:
            optimized = self.tools.invoke(tool_name, {
                "request_id": state["request"]["request_id"],
                "input": {"current_parameters": observations["input"]},
            })
            trace.append({"phase": "optimization", "relationship_id": rid,
                          "tool_name": tool_name, "success": True})
            recommendation = ((optimized.get("data") or {}).get("recommended_parameters") or {})
            revalidated = self.tools.invoke("spi_vte_prediction", {
                "request_id": state["request"]["request_id"], "input": recommendation,
            })
            accepted = spi_vte_in_target(revalidated)
            optimizations[rid] = {
                "recommendation_status": "accepted" if accepted else "rejected",
                "optimization_response": optimized,
                "revalidation_response": revalidated,
                "reason": "复验回到目标区间，参数建议可供人工审核。" if accepted else "复验仍异常，参数建议已拒绝。",
            }
            trace.append({"phase": "revalidation", "relationship_id": rid,
                          "tool_name": "spi_vte_prediction", "success": True,
                          "qualified": accepted})
        except Exception as exc:
            optimizations[rid] = {"recommendation_status": "failed", "error": str(exc)}
            trace.append({"phase": "optimization_or_revalidation", "relationship_id": rid,
                          "tool_name": tool_name, "success": False, "error": str(exc)})
            errors.append({"stage": "optimization_or_revalidation",
                           "relationship_id": rid, "error": str(exc)})

    def _optimize_reflow(
        self, state: AgentState, candidate: dict[str, Any], observations: dict[str, Any],
        validation: dict[str, Any], optimizations: dict[str, dict[str, Any]],
        trace: list[dict[str, Any]], errors: list[dict[str, Any]],
    ) -> None:
        rid = candidate["relationship_id"]
        base_input = observations["input"]
        points = (base_input.get("points") or [])[:1]
        if not points:
            raise ValueError("回流优化需要一个固定测温点")
        arguments = {
            "request_id": state["request"]["request_id"],
            "input": {
                "points": points,
                "current_parameters": {
                    "zone_means_c": base_input["zone_means_c"],
                    "belt_speed_cm_min": base_input["belt_speed_cm_min"],
                },
                "optimization_target": observations["optimization_target"],
                "adjustable_parameters": observations["adjustable_parameters"],
            },
        }
        tool_name = candidate["optimization_action"]["tool_name"]
        try:
            optimized = self.tools.invoke(tool_name, arguments)
            trace.append({"phase": "optimization", "relationship_id": rid, "tool_name": tool_name, "success": True})
            data = optimized.get("data") or {}
            recommendation = data.get("recommended_parameters") or {}
            revalidation_args = {
                "request_id": state["request"]["request_id"],
                "input": {
                    "points": points,
                    "zone_means_c": recommendation["zone_means_c"],
                    "belt_speed_cm_min": recommendation["belt_speed_cm_min"],
                },
                "options": {"return_temperature_curve": False},
            }
            revalidated = self.tools.invoke("reflow_profile_prediction", revalidation_args)
            accepted = bool(((revalidated.get("data") or {}).get("overall") or {}).get("qualified"))
            optimizations[rid] = {
                "recommendation_status": "accepted" if accepted else "rejected",
                "optimization_response": optimized,
                "revalidation_response": revalidated,
                "reason": "复验合格，参数建议可供人工审核。" if accepted else "复验未合格，参数建议已拒绝。",
            }
            trace.append({"phase": "revalidation", "relationship_id": rid, "tool_name": "reflow_profile_prediction", "success": True, "qualified": accepted})
        except Exception as exc:
            optimizations[rid] = {"recommendation_status": "failed", "error": str(exc)}
            trace.append({"phase": "optimization_or_revalidation", "relationship_id": rid, "tool_name": tool_name, "success": False, "error": str(exc)})
            errors.append({"stage": "optimization_or_revalidation", "relationship_id": rid, "error": str(exc)})

    def request_inputs(self, state: AgentState) -> dict[str, Any]:
        pending = state.get("pending_inputs", [])
        fallback = "请补充以下数据；若无法提供，请将字段放入 unavailable_inputs：" + "、".join(pending)
        try:
            prompt = self.qwen.clarify(
                pending,
                {"defect": state.get("defect_name"), "reason": state.get("pending_reason")},
            )
        except (QwenUnavailableError, AttributeError):
            prompt = fallback
        resume = interrupt({
            "reason": state.get("pending_reason"),
            "missing_inputs": pending,
            "prompt": prompt,
            "candidates": [item.get("relationship_id") for item in state.get("candidates", [])],
        })
        resume = resume if isinstance(resume, dict) else {}
        return {
            "observations": deep_merge(state.get("observations", {}), resume.get("observations", {})),
            "unavailable_inputs": sorted(set(state.get("unavailable_inputs", []) + resume.get("unavailable_inputs", []))),
            "pending_inputs": [],
        }

    def finalize(self, state: AgentState) -> dict[str, Any]:
        validations = state.get("validation_records", {})
        optimizations = state.get("optimization_records", {})
        candidates = []
        limitations = list((state.get("kg_response") or {}).get("warnings", []))
        for raw in state.get("candidates", []):
            item = deepcopy(raw)
            record = validations.get(item["relationship_id"])
            item["knowledge_status"] = "knowledge_supported"
            if record:
                item["assessment_status"] = record["assessment_status"]
                item["validation_result"] = record
            elif item["verification_capability"] == "unverified":
                item["assessment_status"] = "not_evaluated"
            item["optimization_result"] = optimizations.get(item["relationship_id"], {
                "recommendation_status": "not_requested" if state["request"].get("goal") == "diagnose" else "not_applicable"
            })
            candidates.append(item)
            limitations.extend(item.get("limitations", []))
        defect_entity = (state.get("kg_response") or {}).get("defect")
        fallback = self._fallback_text(state["defect_name"], candidates)
        evidence_context = []
        for index, evidence in enumerate(state.get("rag_evidence", [])[:5], start=1):
            citation = evidence.get("citation") or {}
            evidence_context.append({
                "evidence_id": f"证据{index}",
                "text": str(evidence.get("text") or "")[:900],
                "source_id": citation.get("source_id"),
                "section_path": citation.get("section_path") or [],
                "pdf_page_start": citation.get("pdf_page_start"),
                "pdf_page_end": citation.get("pdf_page_end"),
            })
        context = {
            "report_language": "zh-CN",
            "defect": None if defect_entity is None else {
                "canonical_name": defect_entity.get("canonical_name"),
                "display_name_zh": defect_entity.get("display_name_zh"),
            },
            "candidates": [{
                "relationship_id": item["relationship_id"],
                "cause": {
                    "canonical_name": item["candidate_cause"]["canonical_name"],
                    "display_name_zh": item["candidate_cause"].get("display_name_zh"),
                },
                "verification_capability": item["verification_capability"],
                "assessment_status": item["assessment_status"],
                "recommendation_status": (item.get("optimization_result") or {}).get("recommendation_status"),
                "tool_conclusion": (item.get("validation_result") or {}).get("reason"),
            } for item in candidates],
            "rag_evidence": evidence_context,
            "limitations": sorted(set(limitations)),
        }
        try:
            narrative = self.qwen.synthesize(context)
        except QwenUnavailableError:
            narrative = fallback
        response_text = narrative + "\n\n" + self._deterministic_details(candidates)
        result = {
            "schema_version": "1.0.0", "status": "completed",
            "request_id": state["request"]["request_id"], "thread_id": state["request"]["thread_id"],
            "defect": defect_entity, "candidates": candidates,
            "rag_evidence": state.get("rag_evidence", []), "pending_inputs": [],
            "pending_prompt": None, "tool_trace": state.get("tool_trace", []),
            "degradation_trace": state.get("degradation_trace", []), "errors": state.get("errors", []),
            "response_text": response_text, "limitations": sorted(set(limitations)),
        }
        return {"result": result}

    def finalize_unknown(self, state: AgentState) -> dict[str, Any]:
        request = state["request"]
        return {"result": {
            "schema_version": "1.0.0", "status": "completed",
            "request_id": request["request_id"], "thread_id": request["thread_id"],
            "defect": None, "candidates": [], "rag_evidence": [], "pending_inputs": [],
            "pending_prompt": None, "tool_trace": state.get("tool_trace", []),
            "degradation_trace": state.get("degradation_trace", []), "errors": state.get("errors", []),
            "response_text": "缺陷类型无法确认，已停止自动诊断；请在获得可识别的缺陷信息后重试。",
            "limitations": ["未知缺陷不映射到冻结本体，也不调用诊断Tool。"],
        }}

    @staticmethod
    def _fallback_text(defect: str, candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return f"诊断结论\n未能为缺陷 {defect} 获得可追溯的候选致因路径。"
        parts = ["诊断结论", f"当前识别为 {defect}，共获得 {len(candidates)} 条候选致因路径。", "", "候选原因分析"]
        for item in candidates:
            cause = item["candidate_cause"].get("display_name_zh") or item["candidate_cause"]["canonical_name"]
            parts.append(f"- {cause}：{item['assessment_status']}")
        parts.extend(["", "限制说明", "候选路径不等于已确认的唯一根因；当前文本为LLM不可用时的程序降级报告。"])
        return "\n".join(parts)

    @staticmethod
    def _deterministic_details(candidates: list[dict[str, Any]]) -> str:
        lines = ["工艺验证与推荐参数（程序生成）"]
        for item in candidates:
            cause = item["candidate_cause"]["canonical_name"]
            lines.append(
                f"- {cause}: assessment_status={item['assessment_status']}, "
                f"verification_capability={item['verification_capability']}"
            )
            validation = item.get("validation_result") or {}
            validation_data = (validation.get("response") or {}).get("data") or {}
            if validation_data.get("vte_mean") is not None:
                lines.append(
                    f"  初始预测: vte_mean={validation_data.get('vte_mean')}, "
                    f"vte_unit={validation_data.get('vte_unit')}, "
                    f"within_training_domain={validation_data.get('within_training_domain')}"
                )
            overall = validation_data.get("overall") or {}
            if overall:
                lines.append(
                    f"  初始预测: max_pwi={overall.get('max_pwi')}, "
                    f"qualified={overall.get('qualified')}, "
                    f"within_training_domain={validation_data.get('within_training_domain')}"
                )
            optimization = item.get("optimization_result") or {}
            recommendation_status = optimization.get("recommendation_status")
            if recommendation_status not in (None, "not_requested", "not_applicable"):
                optimization_data = (optimization.get("optimization_response") or {}).get("data") or {}
                recommendation = optimization_data.get("recommended_parameters") or {}
                revalidation_data = (optimization.get("revalidation_response") or {}).get("data") or {}
                revalidation_overall = revalidation_data.get("overall") or {}
                lines.append(f"  优化建议状态: {recommendation_status}")
                if recommendation:
                    if recommendation.get("zone_means_c") is not None:
                        lines.append(
                            "  推荐参数: zone_means_c="
                            f"{recommendation.get('zone_means_c')}, "
                            f"belt_speed_cm_min={recommendation.get('belt_speed_cm_min')}"
                        )
                    else:
                        lines.append(f"  推荐印刷参数: {recommendation}")
                if revalidation_overall:
                    lines.append(
                        f"  复验: max_pwi={revalidation_overall.get('max_pwi')}, "
                        f"qualified={revalidation_overall.get('qualified')}, "
                        f"within_training_domain={revalidation_data.get('within_training_domain')}"
                    )
                elif revalidation_data.get("vte_mean") is not None:
                    lines.append(
                        f"  复验: vte_mean={revalidation_data.get('vte_mean')}, "
                        f"within_training_domain={revalidation_data.get('within_training_domain')}"
                    )
        lines.append("所有参数建议仅供人工审核，本次未写入设备。")
        return "\n".join(lines)
