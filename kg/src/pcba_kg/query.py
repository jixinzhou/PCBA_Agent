from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

from .core import (
    DEFAULT_ENV_FILE,
    DEFAULT_RUNTIME_CONFIG,
    RuntimeSettings,
    load_runtime_settings,
)


class KGQueryError(RuntimeError):
    """The requested causal path cannot be returned under the frozen KG contract."""


class InvalidQueryInputError(KGQueryError):
    """A query argument does not satisfy the framework-neutral input contract."""


class CausalPathNotFoundError(KGQueryError):
    """No authoritative causal path matched the requested defect or relationship."""


class CausalPathStore(Protocol):
    def fetch_causal_paths(
        self, defect: str, relationship_id: str | None = None
    ) -> list[dict[str, Any]]: ...


def _entity(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": properties["entity_id"],
        "canonical_name": properties["canonical_name"],
        "display_name_zh": properties["display_name_zh"],
    }


def _metric(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_entity(properties),
        "tool_observable": properties["tool_observable"],
        "api_field": properties.get("api_field"),
    }


def _meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return len(value) > 0
    return True


def _has_observation(observations: Mapping[str, Any], path: str) -> bool:
    if path in observations:
        return _meaningful(observations[path])
    current: Any = observations
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return False
        current = current[segment]
    return _meaningful(current)


def _tool_name(tool: Mapping[str, Any] | None) -> str | None:
    return None if tool is None else str(tool["canonical_name"])


class CausalQueryService:
    """Build Agent-ready causal candidates from read-only Neo4j path rows."""

    def __init__(self, store: CausalPathStore, mapping_path: Path) -> None:
        self.store = store
        self.mapping_path = mapping_path
        mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
        self.tool_contracts: dict[str, dict[str, Any]] = mapping["tool_contracts"]
        query_rules = mapping["query_rules"]
        self.strength_order = {
            strength: index
            for index, strength in enumerate(query_rules["candidate_order"])
        }

    def query_causal_paths(
        self,
        defect: str,
        observations: Mapping[str, Any] | None = None,
        relationship_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(defect, str) or not defect.strip():
            raise InvalidQueryInputError("defect must be a non-empty canonical name or entity_id")
        if observations is None:
            observations = {}
        if not isinstance(observations, Mapping):
            raise InvalidQueryInputError("observations must be a JSON-like mapping")
        if relationship_id is not None and (
            not isinstance(relationship_id, str) or not relationship_id.strip()
        ):
            raise InvalidQueryInputError("relationship_id must be a non-empty string")

        defect = defect.strip()
        relationship_id = relationship_id.strip() if relationship_id is not None else None
        rows = self.store.fetch_causal_paths(defect, relationship_id)
        if not rows:
            suffix = f" and relationship_id={relationship_id}" if relationship_id else ""
            raise CausalPathNotFoundError(
                f"No authoritative causal path for defect={defect}{suffix}"
            )

        candidates = [self._candidate(row, observations) for row in rows]
        candidates.sort(
            key=lambda item: (
                self.strength_order.get(item["relation_strength"], 99),
                item["relationship_id"],
            )
        )
        defect_entity = _entity(rows[0]["defect"])
        self._validate_rows(defect_entity, candidates)
        return {
            "schema_version": "1.0.0",
            "ontology": self._ontology(rows[0]["hypothesis"]),
            "defect": defect_entity,
            "candidates": candidates,
            "required_controls": self._required_controls(candidates),
            "warnings": self._warnings(candidates),
        }

    def _candidate(
        self, row: Mapping[str, Any], observations: Mapping[str, Any]
    ) -> dict[str, Any]:
        hypothesis = row["hypothesis"]
        verification_capability = hypothesis["verification_status"]
        validation_tool = _tool_name(row.get("validation_tool"))
        optimization_tool = _tool_name(row.get("optimization_tool"))
        metrics = self._ordered_metrics(hypothesis, row["metrics"])

        if verification_capability == "tool_supported":
            if validation_tool is None or validation_tool not in self.tool_contracts:
                raise KGQueryError(
                    f"Tool-supported path {hypothesis['relationship_id']} has no known validation Tool"
                )
            required_inputs = list(
                self.tool_contracts[validation_tool].get("required_input_paths", [])
            )
            missing_inputs = [
                path for path in required_inputs if not _has_observation(observations, path)
            ]
            action_type = "request_missing_data" if missing_inputs else "invoke_tool"
            reason = (
                f"缺少执行{validation_tool}所需输入。"
                if missing_inputs
                else f"所需输入已提供，可以调用{validation_tool}验证该候选路径。"
            )
        elif verification_capability == "unverified":
            if validation_tool is not None or optimization_tool is not None:
                raise KGQueryError(
                    f"Unverified path {hypothesis['relationship_id']} must not expose a Tool"
                )
            required_inputs = [
                f"manual_observation.{item['canonical_name']}" for item in metrics
            ]
            missing_inputs = [
                path for path in required_inputs if not _has_observation(observations, path)
            ]
            action_type = "manual_inspection"
            reason = (
                "当前候选路径无验证Tool，需要补充人工检查结果。"
                if missing_inputs
                else "人工检查数据已提供，仍需由后续Agent评估，KG不直接确认根因。"
            )
        else:
            raise KGQueryError(
                f"Unsupported verification capability: {verification_capability}"
            )

        return {
            "relationship_id": hypothesis["relationship_id"],
            "candidate_cause": _entity(row["cause"]),
            "process": _entity(row["process"]),
            "relation_strength": hypothesis["relation_strength"],
            "verification_capability": verification_capability,
            "assessment_status": "not_evaluated",
            "interpretation": hypothesis["interpretation"],
            "validation_metrics": metrics,
            "validation_action": {
                "action_type": action_type,
                "tool_name": validation_tool,
                "required_inputs": required_inputs,
                "missing_inputs": missing_inputs,
                "reason": reason,
            },
            "optimization_action": {
                "action_type": "not_available",
                "tool_name": optimization_tool,
                "required_inputs": [],
                "missing_inputs": [],
                "reason": (
                    "必须先完成预测并确认异常，才能进入参数优化。"
                    if optimization_tool is not None
                    else "当前候选路径没有正式优化Tool。"
                ),
            },
            "revalidation_required": verification_capability == "tool_supported",
            "provenance": {
                "source_type": "ontology",
                "source_path": hypothesis["ontology_source"],
                "ontology_version": hypothesis["ontology_version"],
                "source_sha256": hypothesis["ontology_sha256"],
                "evidence_status": hypothesis["evidence_status"],
            },
            "evidence_refs": [],
            "limitations": self._limitations(
                verification_capability, hypothesis["relation_strength"]
            ),
        }

    @staticmethod
    def _ontology(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "ontology_id": hypothesis["ontology_id"],
            "ontology_version": hypothesis["ontology_version"],
            "source_path": hypothesis["ontology_source"],
            "source_sha256": hypothesis["ontology_sha256"],
        }

    @staticmethod
    def _ordered_metrics(
        hypothesis: Mapping[str, Any], metric_rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        by_id = {item["entity_id"]: item for item in metric_rows}
        metric_ids = hypothesis["validation_metric_ids"]
        if set(metric_ids) != set(by_id):
            raise KGQueryError(
                f"Metric path mismatch for {hypothesis['relationship_id']}"
            )
        return [_metric(by_id[metric_id]) for metric_id in metric_ids]

    @staticmethod
    def _limitations(capability: str, strength: str) -> list[str]:
        if capability == "unverified":
            limitations = [
                "无Tool关系必须保留为unverified。",
                "人工检查完成前不能宣称该候选原因已被验证。",
            ]
        else:
            limitations = [
                "存在验证Tool不表示当前案例已经得到Tool支持。",
                "当前没有经过审核的关系到RAG Chunk证据映射。",
            ]
        if strength == "conditional":
            limitations.append("conditional关系不能表述为缺陷的必然原因。")
        return limitations

    @staticmethod
    def _required_controls(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        controls: dict[tuple[str, str, str | None], dict[str, Any]] = {}
        for candidate in candidates:
            process_id = candidate["process"]["canonical_name"]
            validation_tool = candidate["validation_action"]["tool_name"]
            for metric in candidate["validation_metrics"]:
                key = (metric["entity_id"], process_id, validation_tool)
                if key not in controls:
                    controls[key] = {
                        "metric_id": metric["entity_id"],
                        "canonical_name": metric["canonical_name"],
                        "process_id": process_id,
                        "validation_tool": validation_tool,
                        "source_relationship_ids": [],
                    }
                controls[key]["source_relationship_ids"].append(
                    candidate["relationship_id"]
                )
        for control in controls.values():
            control["source_relationship_ids"].sort()
        return list(controls.values())

    @staticmethod
    def _warnings(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
        warnings = ["候选致因不代表已确认的唯一真实根因。"]
        if any(item["verification_capability"] == "unverified" for item in candidates):
            warnings.append("无Tool候选路径已保留，需人工检查后再评估。")
        if {item["relationship_id"] for item in candidates} == {
            "REL-SHIFTED-COMPONENT-PLACEMENT",
            "REL-SHIFTED-COMPONENT-REFLOW",
        }:
            warnings.append(
                "元件偏移返回两条独立候选路径，不能因Tool可用性删除人工检查路径。"
            )
        return warnings

    @staticmethod
    def _validate_rows(
        defect: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
    ) -> None:
        if not candidates:
            raise KGQueryError("A causal query must return at least one candidate")
        relationship_ids = [item["relationship_id"] for item in candidates]
        if len(relationship_ids) != len(set(relationship_ids)):
            raise KGQueryError(
                f"Duplicate causal paths returned for {defect['canonical_name']}"
            )


def query_causal_paths(
    defect: str,
    observations: Mapping[str, Any] | None = None,
    relationship_id: str | None = None,
    *,
    settings: RuntimeSettings | None = None,
    config_path: Path = DEFAULT_RUNTIME_CONFIG,
    env_file: Path = DEFAULT_ENV_FILE,
) -> dict[str, Any]:
    """Open the configured Neo4j store and return one framework-neutral response."""
    from .neo4j_store import Neo4jGraphStore

    resolved_settings = settings or load_runtime_settings(
        config_path=config_path,
        env_file=env_file,
    )
    with Neo4jGraphStore(resolved_settings) as store:
        service = CausalQueryService(store, resolved_settings.mapping_path)
        return service.query_causal_paths(defect, observations, relationship_id)
