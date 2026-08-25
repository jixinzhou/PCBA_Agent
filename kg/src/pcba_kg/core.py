from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG = PROJECT_ROOT / "kg/config/runtime.v1.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / "kg/.env"


class KGConfigurationError(RuntimeError):
    """The local KG runtime configuration is missing or inconsistent."""


class KGSourceError(RuntimeError):
    """The authoritative ontology no longer matches the frozen mapping."""


class KGValidationError(RuntimeError):
    """The generated or persisted graph violates the T11 contract."""


@dataclass(frozen=True)
class RuntimeSettings:
    uri: str
    user: str
    password: str
    database: str
    ontology_path: Path
    mapping_path: Path
    schema_path: Path
    expected_graph: dict[str, Any]


@dataclass(frozen=True)
class GraphPlan:
    nodes: dict[str, list[dict[str, Any]]]
    relationships: dict[str, list[dict[str, Any]]]
    ontology_id: str
    ontology_version: str
    ontology_sha256: str

    @property
    def total_nodes(self) -> int:
        return sum(len(rows) for rows in self.nodes.values())

    @property
    def total_relationships(self) -> int:
        return sum(len(rows) for rows in self.relationships.values())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise KGConfigurationError(
            f"Missing {path.relative_to(PROJECT_ROOT)}; copy kg/.env.example to kg/.env"
        )
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise KGConfigurationError(f"Invalid env line {line_number} in {path}")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise KGConfigurationError(f"Empty env name at line {line_number} in {path}")
        os.environ.setdefault(name, value)


def load_runtime_settings(
    config_path: Path = DEFAULT_RUNTIME_CONFIG,
    env_file: Path = DEFAULT_ENV_FILE,
) -> RuntimeSettings:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    load_env_file(env_file)
    connection = config["neo4j"]["connection"]
    password = os.getenv(connection["password_env"])
    if not password or password == "change_me_before_use":
        raise KGConfigurationError("PCBA_NEO4J_PASSWORD must be set to a local non-placeholder value")
    uri = os.getenv(connection["uri_env"], connection["default_uri"])
    user = os.getenv(connection["user_env"], connection["default_user"])
    database = os.getenv(connection["database_env"], connection["default_database"])
    import_config = config["import"]
    return RuntimeSettings(
        uri=uri,
        user=user,
        password=password,
        database=database,
        ontology_path=PROJECT_ROOT / import_config["ontology_path"],
        mapping_path=PROJECT_ROOT / import_config["mapping_path"],
        schema_path=PROJECT_ROOT / import_config["schema_path"],
        expected_graph=config["expected_graph"],
    )


def _with_provenance(
    properties: dict[str, Any], ontology: dict[str, Any], ontology_path: Path, digest: str
) -> dict[str, Any]:
    return {
        **properties,
        "ontology_id": ontology["ontology_id"],
        "ontology_version": ontology["ontology_version"],
        "ontology_source": ontology_path.relative_to(PROJECT_ROOT).as_posix(),
        "ontology_sha256": digest,
    }


def _entity_rows(
    ontology: dict[str, Any], ontology_path: Path, digest: str
) -> dict[str, list[dict[str, Any]]]:
    collection_labels = {
        "processes": "Process",
        "defects": "Defect",
        "candidate_causes": "CandidateCause",
        "validation_metrics": "QualityMetric",
        "tools": "Tool",
    }
    nodes: dict[str, list[dict[str, Any]]] = {}
    for collection, label in collection_labels.items():
        rows = []
        for entity in ontology[collection]:
            properties = _with_provenance(dict(entity), ontology, ontology_path, digest)
            rows.append({"identity": entity["entity_id"], "properties": properties})
        nodes[label] = rows
    hypothesis_rows = []
    for relationship in ontology["relationships"]:
        properties = _with_provenance(
            {
                **relationship,
                "evidence_status": "ontology_only",
            },
            ontology,
            ontology_path,
            digest,
        )
        hypothesis_rows.append(
            {"identity": relationship["relationship_id"], "properties": properties}
        )
    nodes["CausalHypothesis"] = hypothesis_rows
    return nodes


def _edge(
    edge_type: str,
    from_label: str,
    from_id: str,
    to_label: str,
    to_id: str,
    source_relationship_id: str | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    if source_relationship_id is not None:
        properties["source_relationship_id"] = source_relationship_id
    return {
        "type": edge_type,
        "from_label": from_label,
        "from_id": from_id,
        "to_label": to_label,
        "to_id": to_id,
        "properties": properties,
    }


def _relationship_rows(ontology: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    defects = {item["canonical_name"]: item["entity_id"] for item in ontology["defects"]}
    causes = {item["entity_id"]: item for item in ontology["candidate_causes"]}
    processes = {
        item["canonical_name"]: item["entity_id"] for item in ontology["processes"]
    }
    metrics = {item["entity_id"]: item for item in ontology["validation_metrics"]}
    tools = {item["canonical_name"]: item for item in ontology["tools"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for cause in causes.values():
        grouped["BELONGS_TO"].append(
            _edge(
                "BELONGS_TO",
                "CandidateCause",
                cause["entity_id"],
                "Process",
                processes[cause["process_id"]],
            )
        )
    for metric in metrics.values():
        grouped["BELONGS_TO"].append(
            _edge(
                "BELONGS_TO",
                "QualityMetric",
                metric["entity_id"],
                "Process",
                processes[metric["process_id"]],
            )
        )
    for tool in tools.values():
        grouped["BELONGS_TO"].append(
            _edge(
                "BELONGS_TO",
                "Tool",
                tool["entity_id"],
                "Process",
                processes[tool["process_id"]],
            )
        )

    for relationship in ontology["relationships"]:
        relationship_id = relationship["relationship_id"]
        grouped["HAS_HYPOTHESIS"].append(
            _edge(
                "HAS_HYPOTHESIS",
                "Defect",
                defects[relationship["defect_id"]],
                "CausalHypothesis",
                relationship_id,
                relationship_id,
            )
        )
        grouped["PROPOSES_CAUSE"].append(
            _edge(
                "PROPOSES_CAUSE",
                "CausalHypothesis",
                relationship_id,
                "CandidateCause",
                relationship["candidate_cause_id"],
                relationship_id,
            )
        )
        for metric_id in relationship["validation_metric_ids"]:
            grouped["REQUIRES_METRIC"].append(
                _edge(
                    "REQUIRES_METRIC",
                    "CausalHypothesis",
                    relationship_id,
                    "QualityMetric",
                    metric_id,
                    relationship_id,
                )
            )
        if relationship["validation_tool"] is not None:
            grouped["VALIDATED_BY"].append(
                _edge(
                    "VALIDATED_BY",
                    "CausalHypothesis",
                    relationship_id,
                    "Tool",
                    tools[relationship["validation_tool"]]["entity_id"],
                    relationship_id,
                )
            )
        if relationship["optimization_tool"] is not None:
            grouped["OPTIMIZED_BY"].append(
                _edge(
                    "OPTIMIZED_BY",
                    "CausalHypothesis",
                    relationship_id,
                    "Tool",
                    tools[relationship["optimization_tool"]]["entity_id"],
                    relationship_id,
                )
            )
    return dict(grouped)


def _validate_plan(plan: GraphPlan, expected: dict[str, Any]) -> None:
    actual_nodes = {label: len(rows) for label, rows in plan.nodes.items()}
    actual_relationships = {
        edge_type: len(rows) for edge_type, rows in plan.relationships.items()
    }
    if actual_nodes != expected["nodes"]:
        raise KGValidationError(f"Node plan mismatch: {actual_nodes} != {expected['nodes']}")
    if actual_relationships != expected["relationships"]:
        raise KGValidationError(
            f"Relationship plan mismatch: {actual_relationships} != {expected['relationships']}"
        )
    if plan.total_nodes != expected["total_nodes"]:
        raise KGValidationError(f"Expected {expected['total_nodes']} nodes, got {plan.total_nodes}")
    if plan.total_relationships != expected["total_relationships"]:
        raise KGValidationError(
            f"Expected {expected['total_relationships']} relationships, got {plan.total_relationships}"
        )
    node_keys = {
        (label, row["identity"]) for label, rows in plan.nodes.items() for row in rows
    }
    if len(node_keys) != plan.total_nodes:
        raise KGValidationError("Duplicate node identity in import plan")
    edge_keys = {
        (row["type"], row["from_label"], row["from_id"], row["to_label"], row["to_id"])
        for rows in plan.relationships.values()
        for row in rows
    }
    if len(edge_keys) != plan.total_relationships:
        raise KGValidationError("Duplicate relationship identity in import plan")


def build_graph_plan(settings: RuntimeSettings) -> GraphPlan:
    ontology = yaml.safe_load(settings.ontology_path.read_text(encoding="utf-8"))
    mapping = yaml.safe_load(settings.mapping_path.read_text(encoding="utf-8"))
    digest = file_sha256(settings.ontology_path)
    frozen = mapping["source_contracts"]["ontology"]
    if digest != frozen["sha256"]:
        raise KGSourceError(f"Ontology SHA256 changed: {digest} != {frozen['sha256']}")
    if ontology["ontology_id"] != frozen["ontology_id"]:
        raise KGSourceError("Ontology ID no longer matches the frozen mapping")
    if ontology["ontology_version"] != frozen["ontology_version"]:
        raise KGSourceError("Ontology version no longer matches the frozen mapping")
    plan = GraphPlan(
        nodes=_entity_rows(ontology, settings.ontology_path, digest),
        relationships=_relationship_rows(ontology),
        ontology_id=ontology["ontology_id"],
        ontology_version=ontology["ontology_version"],
        ontology_sha256=digest,
    )
    _validate_plan(plan, settings.expected_graph)
    return plan


def plan_summary(plan: GraphPlan) -> dict[str, Any]:
    return {
        "ontology_id": plan.ontology_id,
        "ontology_version": plan.ontology_version,
        "ontology_sha256": plan.ontology_sha256,
        "total_nodes": plan.total_nodes,
        "nodes": {label: len(rows) for label, rows in plan.nodes.items()},
        "total_relationships": plan.total_relationships,
        "relationships": {
            edge_type: len(rows) for edge_type, rows in plan.relationships.items()
        },
    }


def json_output(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
