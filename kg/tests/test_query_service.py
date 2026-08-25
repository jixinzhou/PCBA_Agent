from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "kg/src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pcba_kg.core import RuntimeSettings, build_graph_plan
from pcba_kg.neo4j_store import CAUSAL_PATH_QUERY
from pcba_kg.query import (
    CausalPathNotFoundError,
    CausalQueryService,
    InvalidQueryInputError,
)


class PlanPathStore:
    """Expose GraphPlan as query rows so service tests do not need a database."""

    def __init__(self, plan: Any) -> None:
        self.nodes = {
            label: {row["identity"]: row["properties"] for row in rows}
            for label, rows in plan.nodes.items()
        }
        self.relationships = plan.relationships

    def _targets(self, edge_type: str, source_id: str) -> list[str]:
        return [
            row["to_id"]
            for row in self.relationships[edge_type]
            if row["from_id"] == source_id
        ]

    def fetch_causal_paths(
        self, defect: str, relationship_id: str | None = None
    ) -> list[dict[str, Any]]:
        defect_rows = [
            item
            for item in self.nodes["Defect"].values()
            if defect in {item["canonical_name"], item["entity_id"]}
        ]
        if not defect_rows:
            return []
        defect_row = defect_rows[0]
        rows = []
        for hypothesis_id in self._targets(
            "HAS_HYPOTHESIS", defect_row["entity_id"]
        ):
            if relationship_id is not None and hypothesis_id != relationship_id:
                continue
            hypothesis = self.nodes["CausalHypothesis"][hypothesis_id]
            cause_id = self._targets("PROPOSES_CAUSE", hypothesis_id)[0]
            process_id = self._targets("BELONGS_TO", cause_id)[0]
            metric_ids = self._targets("REQUIRES_METRIC", hypothesis_id)
            validation_tool_ids = self._targets("VALIDATED_BY", hypothesis_id)
            optimization_tool_ids = self._targets("OPTIMIZED_BY", hypothesis_id)
            rows.append(
                {
                    "defect": defect_row,
                    "hypothesis": hypothesis,
                    "cause": self.nodes["CandidateCause"][cause_id],
                    "process": self.nodes["Process"][process_id],
                    "metrics": [
                        self.nodes["QualityMetric"][metric_id]
                        for metric_id in metric_ids
                    ],
                    "validation_tool": (
                        self.nodes["Tool"][validation_tool_ids[0]]
                        if validation_tool_ids
                        else None
                    ),
                    "optimization_tool": (
                        self.nodes["Tool"][optimization_tool_ids[0]]
                        if optimization_tool_ids
                        else None
                    ),
                }
            )
        return rows


class T113QueryServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        runtime_path = PROJECT_ROOT / "kg/config/runtime.v1.yaml"
        runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
        import_config = runtime["import"]
        settings = RuntimeSettings(
            uri="bolt://127.0.0.1:7687",
            user="neo4j",
            password="unit-test-only",
            database="neo4j",
            ontology_path=PROJECT_ROOT / import_config["ontology_path"],
            mapping_path=PROJECT_ROOT / import_config["mapping_path"],
            schema_path=PROJECT_ROOT / import_config["schema_path"],
            expected_graph=runtime["expected_graph"],
        )
        cls.plan = build_graph_plan(settings)
        cls.service = CausalQueryService(PlanPathStore(cls.plan), settings.mapping_path)
        schema = json.loads(
            (PROJECT_ROOT / "kg/schemas/causal_query.v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.validator = Draft202012Validator(schema)

    def assert_schema_valid(self, response: dict[str, Any]) -> None:
        errors = sorted(
            self.validator.iter_errors(response), key=lambda item: list(item.path)
        )
        self.assertEqual([], errors, msg=[error.message for error in errors])

    def test_all_four_defects_return_schema_valid_authoritative_paths(self) -> None:
        expected_counts = {
            "insufficient_solder": 1,
            "excessive_solder": 1,
            "short": 1,
            "shifted_component": 2,
        }
        for defect, expected_count in expected_counts.items():
            with self.subTest(defect=defect):
                response = self.service.query_causal_paths(defect)
                self.assert_schema_valid(response)
                self.assertEqual(expected_count, len(response["candidates"]))
                self.assertTrue(
                    all(
                        item["assessment_status"] == "not_evaluated"
                        for item in response["candidates"]
                    )
                )

    def test_shifted_component_keeps_strong_manual_path_before_conditional_tool_path(self) -> None:
        response = self.service.query_causal_paths("shifted_component")
        self.assertEqual(
            [
                "REL-SHIFTED-COMPONENT-PLACEMENT",
                "REL-SHIFTED-COMPONENT-REFLOW",
            ],
            [item["relationship_id"] for item in response["candidates"]],
        )
        placement, reflow = response["candidates"]
        self.assertEqual("manual_inspection", placement["validation_action"]["action_type"])
        self.assertEqual("unverified", placement["verification_capability"])
        self.assertEqual("request_missing_data", reflow["validation_action"]["action_type"])
        self.assertEqual("tool_supported", reflow["verification_capability"])

    def test_missing_spi_inputs_are_reported_exactly(self) -> None:
        response = self.service.query_causal_paths(
            "insufficient_solder",
            {"input": {"squeegee_pressure_kgf": 4.0}},
        )
        action = response["candidates"][0]["validation_action"]
        self.assertEqual("request_missing_data", action["action_type"])
        self.assertEqual(
            [
                "input.squeegee_speed_m_s",
                "input.separation_speed_m_s",
                "input.separation_distance_mm",
            ],
            action["missing_inputs"],
        )

    def test_complete_nested_spi_inputs_enable_prediction_without_calling_it(self) -> None:
        response = self.service.query_causal_paths(
            "insufficient_solder",
            {
                "input": {
                    "squeegee_pressure_kgf": 4.0,
                    "squeegee_speed_m_s": 0.1,
                    "separation_speed_m_s": 0.001,
                    "separation_distance_mm": 1.0,
                }
            },
        )
        action = response["candidates"][0]["validation_action"]
        self.assertEqual("invoke_tool", action["action_type"])
        self.assertEqual("spi_vte_prediction", action["tool_name"])
        self.assertEqual([], action["missing_inputs"])
        self.assertEqual("not_evaluated", response["candidates"][0]["assessment_status"])
        self.assert_schema_valid(response)

    def test_complete_flat_reflow_inputs_enable_only_reflow_path(self) -> None:
        response = self.service.query_causal_paths(
            "shifted_component",
            {
                "input.points": [{"point_id": "P1"}],
                "input.zone_means_c": [150.0] * 13,
                "input.belt_speed_cm_min": 85.0,
            },
        )
        placement, reflow = response["candidates"]
        self.assertEqual("manual_inspection", placement["validation_action"]["action_type"])
        self.assertEqual("invoke_tool", reflow["validation_action"]["action_type"])
        self.assertEqual([], reflow["validation_action"]["missing_inputs"])

    def test_false_manual_observation_is_present_but_does_not_confirm_path(self) -> None:
        response = self.service.query_causal_paths(
            "short", {"manual_observation": {"paste_bridge": False}}
        )
        candidate = response["candidates"][0]
        self.assertEqual([], candidate["validation_action"]["missing_inputs"])
        self.assertEqual("manual_inspection", candidate["validation_action"]["action_type"])
        self.assertEqual("not_evaluated", candidate["assessment_status"])

    def test_empty_collection_is_still_missing(self) -> None:
        response = self.service.query_causal_paths(
            "shifted_component",
            {
                "input": {
                    "points": [],
                    "zone_means_c": [150.0] * 13,
                    "belt_speed_cm_min": 85.0,
                }
            },
        )
        action = response["candidates"][1]["validation_action"]
        self.assertEqual(["input.points"], action["missing_inputs"])

    def test_entity_id_and_relationship_filter_are_supported(self) -> None:
        response = self.service.query_causal_paths(
            "DEFECT-SHIFTED-COMPONENT",
            relationship_id="REL-SHIFTED-COMPONENT-REFLOW",
        )
        self.assertEqual(1, len(response["candidates"]))
        self.assertEqual(
            "REL-SHIFTED-COMPONENT-REFLOW",
            response["candidates"][0]["relationship_id"],
        )
        self.assert_schema_valid(response)

    def test_required_controls_cover_both_shifted_component_processes(self) -> None:
        controls = self.service.query_causal_paths("shifted_component")["required_controls"]
        self.assertEqual(5, len(controls))
        self.assertEqual({"placement", "reflow"}, {item["process_id"] for item in controls})

    def test_unknown_defect_and_relationship_fail_closed(self) -> None:
        with self.assertRaises(CausalPathNotFoundError):
            self.service.query_causal_paths("not_a_defect")
        with self.assertRaises(CausalPathNotFoundError):
            self.service.query_causal_paths(
                "short", relationship_id="REL-NOT-AUTHORITATIVE"
            )

    def test_invalid_query_inputs_are_rejected(self) -> None:
        for defect in ("", "   ", None):
            with self.subTest(defect=defect), self.assertRaises(InvalidQueryInputError):
                self.service.query_causal_paths(defect)  # type: ignore[arg-type]
        with self.assertRaises(InvalidQueryInputError):
            self.service.query_causal_paths("short", observations=[])  # type: ignore[arg-type]

    def test_neo4j_query_is_parameterized_and_read_only(self) -> None:
        self.assertIn("$defect", CAUSAL_PATH_QUERY)
        self.assertIn("$relationship_id", CAUSAL_PATH_QUERY)
        self.assertIsNone(
            re.search(
                r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b",
                CAUSAL_PATH_QUERY.upper(),
            )
        )


if __name__ == "__main__":
    unittest.main()
