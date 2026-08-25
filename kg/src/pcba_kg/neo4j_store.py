from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase, ManagedTransaction

from .core import GraphPlan, KGValidationError, RuntimeSettings


ENTITY_LABELS = {"Process", "Defect", "CandidateCause", "QualityMetric", "Tool"}
ALL_NODE_LABELS = ENTITY_LABELS | {"CausalHypothesis"}
RELATIONSHIP_TYPES = {
    "HAS_HYPOTHESIS",
    "PROPOSES_CAUSE",
    "BELONGS_TO",
    "REQUIRES_METRIC",
    "VALIDATED_BY",
    "OPTIMIZED_BY",
}


CAUSAL_PATH_QUERY = """
MATCH (defect:Defect)
WHERE defect.canonical_name = $defect OR defect.entity_id = $defect
MATCH (defect)-[:HAS_HYPOTHESIS]->(hypothesis:CausalHypothesis)
      -[:PROPOSES_CAUSE]->(cause:CandidateCause)-[:BELONGS_TO]->(process:Process)
WHERE $relationship_id IS NULL OR hypothesis.relationship_id = $relationship_id
MATCH (hypothesis)-[:REQUIRES_METRIC]->(metric:QualityMetric)
OPTIONAL MATCH (hypothesis)-[:VALIDATED_BY]->(validation_tool:Tool)
OPTIONAL MATCH (hypothesis)-[:OPTIMIZED_BY]->(optimization_tool:Tool)
RETURN properties(defect) AS defect,
       properties(hypothesis) AS hypothesis,
       properties(cause) AS cause,
       properties(process) AS process,
       collect(DISTINCT properties(metric)) AS metrics,
       head(collect(DISTINCT properties(validation_tool))) AS validation_tool,
       head(collect(DISTINCT properties(optimization_tool))) AS optimization_tool
ORDER BY CASE hypothesis.relation_strength
           WHEN 'strong' THEN 0
           WHEN 'medium_strong' THEN 1
           WHEN 'conditional' THEN 2
           ELSE 99
         END,
         hypothesis.relationship_id
"""


def _schema_statements(path: Path) -> list[str]:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("//")
    ]
    return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]


def _write_rows(tx: ManagedTransaction, query: str, rows: list[dict[str, Any]]) -> None:
    tx.run(query, rows=rows).consume()


def _node_query(label: str) -> str:
    if label not in ALL_NODE_LABELS:
        raise KGValidationError(f"Unsupported node label: {label}")
    if label == "CausalHypothesis":
        return """
        UNWIND $rows AS row
        MERGE (node:CausalHypothesis {relationship_id: row.identity})
        SET node = row.properties
        """
    return f"""
    UNWIND $rows AS row
    MERGE (node:KnowledgeEntity:{label} {{entity_id: row.identity}})
    SET node = row.properties
    """


def _relationship_query(edge_type: str, from_label: str, to_label: str) -> str:
    if edge_type not in RELATIONSHIP_TYPES:
        raise KGValidationError(f"Unsupported relationship type: {edge_type}")
    if from_label not in ALL_NODE_LABELS or to_label not in ALL_NODE_LABELS:
        raise KGValidationError(f"Unsupported relationship labels: {from_label} -> {to_label}")
    from_key = "relationship_id" if from_label == "CausalHypothesis" else "entity_id"
    to_key = "relationship_id" if to_label == "CausalHypothesis" else "entity_id"
    return f"""
    UNWIND $rows AS row
    MATCH (source:{from_label} {{{from_key}: row.from_id}})
    MATCH (target:{to_label} {{{to_key}: row.to_id}})
    MERGE (source)-[edge:{edge_type}]->(target)
    SET edge = row.properties
    """


class Neo4jGraphStore:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.user, settings.password),
        )

    def __enter__(self) -> "Neo4jGraphStore":
        self.driver.verify_connectivity()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.driver.close()

    def apply_schema(self) -> None:
        with self.driver.session(database=self.settings.database) as session:
            for statement in _schema_statements(self.settings.schema_path):
                session.run(statement).consume()

    def import_plan(self, plan: GraphPlan) -> None:
        self.apply_schema()
        with self.driver.session(database=self.settings.database) as session:
            for label, rows in plan.nodes.items():
                session.execute_write(_write_rows, _node_query(label), rows)
            grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
            for rows in plan.relationships.values():
                for row in rows:
                    grouped[(row["type"], row["from_label"], row["to_label"])].append(row)
            for (edge_type, from_label, to_label), rows in grouped.items():
                query = _relationship_query(edge_type, from_label, to_label)
                session.execute_write(_write_rows, query, rows)

    def _data(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.settings.database) as session:
            return [record.data() for record in session.run(query, **parameters)]

    def fetch_causal_paths(
        self, defect: str, relationship_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the frozen semantic paths for one defect without mutating the graph."""
        return self._data(
            CAUSAL_PATH_QUERY,
            defect=defect,
            relationship_id=relationship_id,
        )

    def _graph_fingerprint(self) -> str:
        nodes = self._data(
            """
            MATCH (node)
            RETURN labels(node) AS labels,
                   coalesce(node.entity_id, node.relationship_id) AS identity,
                   properties(node) AS properties
            ORDER BY identity
            """
        )
        relationships = self._data(
            """
            MATCH (source)-[edge]->(target)
            RETURN coalesce(source.entity_id, source.relationship_id) AS source,
                   type(edge) AS type,
                   coalesce(target.entity_id, target.relationship_id) AS target,
                   properties(edge) AS properties
            ORDER BY source, type, target
            """
        )
        payload = json.dumps(
            {"nodes": nodes, "relationships": relationships},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate(self, plan: GraphPlan) -> dict[str, Any]:
        expected = self.settings.expected_graph
        node_counts = {
            row["label"]: row["count"]
            for row in self._data(
                """
                UNWIND $labels AS label
                CALL (label) {
                  MATCH (node)
                  WHERE label IN labels(node)
                  RETURN count(node) AS count
                }
                RETURN label, count
                ORDER BY label
                """,
                labels=list(expected["nodes"]),
            )
        }
        relationship_counts = {
            row["type"]: row["count"]
            for row in self._data(
                """
                MATCH ()-[edge]->()
                RETURN type(edge) AS type, count(edge) AS count
                ORDER BY type
                """
            )
        }
        total_nodes = self._data("MATCH (node) RETURN count(node) AS count")[0]["count"]
        total_relationships = self._data(
            "MATCH ()-[edge]->() RETURN count(edge) AS count"
        )[0]["count"]
        if node_counts != expected["nodes"]:
            raise KGValidationError(f"Persisted node mismatch: {node_counts} != {expected['nodes']}")
        if relationship_counts != expected["relationships"]:
            raise KGValidationError(
                f"Persisted relationship mismatch: {relationship_counts} != {expected['relationships']}"
            )
        if total_nodes != expected["total_nodes"]:
            raise KGValidationError(f"Persisted total nodes: {total_nodes}")
        if total_relationships != expected["total_relationships"]:
            raise KGValidationError(f"Persisted total relationships: {total_relationships}")

        paths = self._data(
            """
            MATCH (defect:Defect)-[:HAS_HYPOTHESIS]->(hypothesis:CausalHypothesis)
                  -[:PROPOSES_CAUSE]->(cause:CandidateCause)-[:BELONGS_TO]->(process:Process)
            MATCH (hypothesis)-[:REQUIRES_METRIC]->(metric:QualityMetric)
            OPTIONAL MATCH (hypothesis)-[:VALIDATED_BY]->(validation_tool:Tool)
            OPTIONAL MATCH (hypothesis)-[:OPTIMIZED_BY]->(optimization_tool:Tool)
            RETURN hypothesis.relationship_id AS relationship_id,
                   defect.canonical_name AS defect_id,
                   cause.entity_id AS candidate_cause_id,
                   process.canonical_name AS process_id,
                   hypothesis.relation_strength AS relation_strength,
                   hypothesis.verification_status AS verification_status,
                   collect(DISTINCT metric.entity_id) AS metric_ids,
                   collect(DISTINCT validation_tool.canonical_name) AS validation_tools,
                   collect(DISTINCT optimization_tool.canonical_name) AS optimization_tools
            ORDER BY relationship_id
            """
        )
        expected_paths = {
            row["properties"]["relationship_id"]: row["properties"]
            for row in plan.nodes["CausalHypothesis"]
        }
        if len(paths) != len(expected_paths):
            raise KGValidationError(f"Expected {len(expected_paths)} causal paths, got {len(paths)}")
        for path in paths:
            source = expected_paths[path["relationship_id"]]
            for field in (
                "defect_id",
                "candidate_cause_id",
                "process_id",
                "relation_strength",
                "verification_status",
            ):
                if path[field] != source[field]:
                    raise KGValidationError(
                        f"Path {path['relationship_id']} field {field}: {path[field]} != {source[field]}"
                    )
            if set(path["metric_ids"]) != set(source["validation_metric_ids"]):
                raise KGValidationError(f"Metric mismatch for {path['relationship_id']}")
            expected_validation = {source["validation_tool"]} - {None}
            expected_optimization = {source["optimization_tool"]} - {None}
            if set(path["validation_tools"]) != expected_validation:
                raise KGValidationError(f"Validation Tool mismatch for {path['relationship_id']}")
            if set(path["optimization_tools"]) != expected_optimization:
                raise KGValidationError(f"Optimization Tool mismatch for {path['relationship_id']}")

        unverified = self._data(
            """
            MATCH (hypothesis:CausalHypothesis {verification_status: 'unverified'})
            OPTIONAL MATCH (hypothesis)-[:VALIDATED_BY]->(validation_tool:Tool)
            OPTIONAL MATCH (hypothesis)-[:OPTIMIZED_BY]->(optimization_tool:Tool)
            RETURN count(DISTINCT hypothesis) AS hypotheses,
                   count(DISTINCT validation_tool) AS validation_tools,
                   count(DISTINCT optimization_tool) AS optimization_tools
            """
        )[0]
        expected_unverified = expected["unverified_hypotheses"]
        if unverified != {
            "hypotheses": expected_unverified,
            "validation_tools": 0,
            "optimization_tools": 0,
        }:
            raise KGValidationError(f"Unverified-path mismatch: {unverified}")

        shifted = self._data(
            """
            MATCH (:Defect {canonical_name: 'shifted_component'})
                  -[:HAS_HYPOTHESIS]->(hypothesis:CausalHypothesis)
            RETURN count(hypothesis) AS count,
                   collect(hypothesis.relationship_id) AS relationship_ids
            """
        )[0]
        if shifted["count"] != expected["shifted_component_hypotheses"]:
            raise KGValidationError(f"shifted_component path mismatch: {shifted}")

        constraints = self._data(
            "SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names"
        )[0]["names"]
        components = self._data(
            "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
        )
        return {
            "database": self.settings.database,
            "components": components,
            "total_nodes": total_nodes,
            "nodes": node_counts,
            "total_relationships": total_relationships,
            "relationships": relationship_counts,
            "causal_paths": paths,
            "unverified": unverified,
            "shifted_component": shifted,
            "constraints": sorted(constraints),
            "graph_fingerprint": self._graph_fingerprint(),
        }
