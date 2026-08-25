from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from tool.agent_tools.registry import TOOL_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = PROJECT_ROOT / "ontology/pcba_defect_causality.v1.1.yaml"
DICTIONARY_PATH = PROJECT_ROOT / "rag/schemas/entity_dictionary.json"
TOOL_MODELS_PATH = PROJECT_ROOT / "tool/agent_tools/models.py"
MAPPING_PATH = PROJECT_ROOT / "kg/config/neo4j_mapping.v1.yaml"
SCHEMA_PATH = PROJECT_ROOT / "kg/schemas/causal_query.v1.schema.json"
CYPHER_PATH = PROJECT_ROOT / "kg/cypher/schema.v1.cypher"
EXAMPLES_DIR = PROJECT_ROOT / "kg/examples"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class T111KGContractTests(unittest.TestCase):
    EXPECTED_RELATIONSHIPS = {
        "REL-INSUFFICIENT-SOLDER-PRINTING": (
            "insufficient_solder",
            "CAUSE-INSUFFICIENT-SOLDER-DEPOSITION",
            "printing",
            "strong",
            "tool_supported",
        ),
        "REL-EXCESSIVE-SOLDER-PRINTING": (
            "excessive_solder",
            "CAUSE-EXCESSIVE-SOLDER-DEPOSITION",
            "printing",
            "medium_strong",
            "tool_supported",
        ),
        "REL-SHORT-WET-PASTE-BRIDGING": (
            "short",
            "CAUSE-WET-PASTE-BRIDGING",
            "printing",
            "strong",
            "unverified",
        ),
        "REL-SHIFTED-COMPONENT-PLACEMENT": (
            "shifted_component",
            "CAUSE-PLACEMENT-OFFSET",
            "placement",
            "strong",
            "unverified",
        ),
        "REL-SHIFTED-COMPONENT-REFLOW": (
            "shifted_component",
            "CAUSE-REFLOW-THERMAL-IMBALANCE",
            "reflow",
            "conditional",
            "tool_supported",
        ),
    }
    EXPECTED_EXAMPLES = {
        "insufficient_solder.example.json": {
            "REL-INSUFFICIENT-SOLDER-PRINTING"
        },
        "short.example.json": {"REL-SHORT-WET-PASTE-BRIDGING"},
        "shifted_component.example.json": {
            "REL-SHIFTED-COMPONENT-PLACEMENT",
            "REL-SHIFTED-COMPONENT-REFLOW",
        },
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
        cls.dictionary = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        cls.mapping = yaml.safe_load(MAPPING_PATH.read_text(encoding="utf-8"))
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.validator = Draft202012Validator(cls.schema)
        cls.examples = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(EXAMPLES_DIR.glob("*.example.json"))
        }
        cls.relationships = {
            item["relationship_id"]: item for item in cls.ontology["relationships"]
        }
        cls.causes = {
            item["entity_id"]: item for item in cls.ontology["candidate_causes"]
        }
        cls.processes = {
            item["canonical_name"]: item for item in cls.ontology["processes"]
        }
        cls.metrics = {
            item["entity_id"]: item for item in cls.ontology["validation_metrics"]
        }

    def test_source_contract_hashes_are_frozen_and_current(self) -> None:
        contracts = self.mapping["source_contracts"]
        self.assertEqual(sha256(ONTOLOGY_PATH), contracts["ontology"]["sha256"])
        self.assertEqual(sha256(DICTIONARY_PATH), contracts["entity_dictionary"]["sha256"])
        self.assertEqual(sha256(TOOL_MODELS_PATH), contracts["tool_models"]["sha256"])
        self.assertEqual("causal_fact_source", contracts["ontology"]["authority"])

    def test_mapping_scope_matches_frozen_ontology_exactly(self) -> None:
        expected_counts = self.mapping["active_graph"]["expected_counts"]
        self.assertEqual(len(self.ontology["processes"]), expected_counts["Process"])
        self.assertEqual(len(self.ontology["defects"]), expected_counts["Defect"])
        self.assertEqual(
            len(self.ontology["candidate_causes"]), expected_counts["CandidateCause"]
        )
        self.assertEqual(
            len(self.ontology["validation_metrics"]), expected_counts["QualityMetric"]
        )
        self.assertEqual(len(self.ontology["tools"]), expected_counts["Tool"])
        self.assertEqual(
            len(self.ontology["relationships"]), expected_counts["CausalHypothesis"]
        )

    def test_all_five_authoritative_causal_paths_are_unchanged(self) -> None:
        actual = {
            relationship_id: (
                item["defect_id"],
                item["candidate_cause_id"],
                item["process_id"],
                item["relation_strength"],
                item["verification_status"],
            )
            for relationship_id, item in self.relationships.items()
        }
        self.assertEqual(self.EXPECTED_RELATIONSHIPS, actual)

    def test_mapping_has_operational_topology_without_caused_by(self) -> None:
        expected_types = {
            "HAS_HYPOTHESIS",
            "PROPOSES_CAUSE",
            "BELONGS_TO",
            "REQUIRES_METRIC",
            "VALIDATED_BY",
            "OPTIMIZED_BY",
        }
        self.assertEqual(
            expected_types, set(self.mapping["active_graph"]["relationship_types"])
        )
        mapped_types = {
            item["type"] for item in self.mapping["relationship_mappings"]
        }
        self.assertEqual(expected_types, mapped_types)
        self.assertNotIn("CAUSED_BY", CYPHER_PATH.read_text(encoding="utf-8"))

    def test_query_schema_and_all_examples_are_valid(self) -> None:
        Draft202012Validator.check_schema(self.schema)
        self.assertEqual(set(self.EXPECTED_EXAMPLES), set(self.examples))
        for name, example in self.examples.items():
            errors = sorted(self.validator.iter_errors(example), key=lambda item: list(item.path))
            self.assertEqual([], errors, msg=f"{name}: {[error.message for error in errors]}")

    def test_examples_mirror_ontology_relationships_and_metrics(self) -> None:
        ontology_sha = sha256(ONTOLOGY_PATH)
        for name, expected_relationship_ids in self.EXPECTED_EXAMPLES.items():
            example = self.examples[name]
            self.assertEqual(ontology_sha, example["ontology"]["source_sha256"])
            candidates = {
                item["relationship_id"]: item for item in example["candidates"]
            }
            self.assertEqual(expected_relationship_ids, set(candidates))
            for relationship_id, candidate in candidates.items():
                relationship = self.relationships[relationship_id]
                cause = self.causes[relationship["candidate_cause_id"]]
                process = self.processes[relationship["process_id"]]
                self.assertEqual(cause["entity_id"], candidate["candidate_cause"]["entity_id"])
                self.assertEqual(
                    process["canonical_name"], candidate["process"]["canonical_name"]
                )
                self.assertEqual(
                    relationship["relation_strength"], candidate["relation_strength"]
                )
                self.assertEqual(
                    relationship["verification_status"],
                    candidate["verification_capability"],
                )
                self.assertEqual("not_evaluated", candidate["assessment_status"])
                self.assertEqual(relationship["interpretation"], candidate["interpretation"])
                self.assertEqual(
                    set(relationship["validation_metric_ids"]),
                    {item["entity_id"] for item in candidate["validation_metrics"]},
                )
                for metric in candidate["validation_metrics"]:
                    source_metric = self.metrics[metric["entity_id"]]
                    self.assertEqual(
                        source_metric["tool_observable"], metric["tool_observable"]
                    )
                    self.assertEqual(source_metric["api_field"], metric["api_field"])

    def test_shifted_component_returns_manual_and_tool_paths_together(self) -> None:
        candidates = self.examples["shifted_component.example.json"]["candidates"]
        self.assertEqual(2, len(candidates))
        placement, reflow = candidates
        self.assertEqual("REL-SHIFTED-COMPONENT-PLACEMENT", placement["relationship_id"])
        self.assertEqual("manual_inspection", placement["validation_action"]["action_type"])
        self.assertEqual("unverified", placement["verification_capability"])
        self.assertEqual("REL-SHIFTED-COMPONENT-REFLOW", reflow["relationship_id"])
        self.assertEqual(
            "request_missing_data", reflow["validation_action"]["action_type"]
        )
        self.assertEqual(
            "reflow_profile_prediction", reflow["validation_action"]["tool_name"]
        )
        self.assertEqual("conditional", reflow["relation_strength"])

    def test_unverified_relations_are_manual_and_never_gain_tools(self) -> None:
        candidates = [
            candidate
            for example in self.examples.values()
            for candidate in example["candidates"]
            if candidate["verification_capability"] == "unverified"
        ]
        self.assertEqual(2, len(candidates))
        for candidate in candidates:
            self.assertEqual(
                "manual_inspection", candidate["validation_action"]["action_type"]
            )
            self.assertIsNone(candidate["validation_action"]["tool_name"])
            self.assertEqual(
                "not_available", candidate["optimization_action"]["action_type"]
            )
            self.assertIsNone(candidate["optimization_action"]["tool_name"])
            self.assertFalse(candidate["revalidation_required"])

    def test_tool_names_match_registry_and_tool_capability_is_not_result(self) -> None:
        ontology_tools = {item["canonical_name"] for item in self.ontology["tools"]}
        self.assertEqual(ontology_tools, set(TOOL_REGISTRY) - {"pcba_defect_classification"})
        configured_tools = set(self.mapping["tool_contracts"])
        self.assertEqual(ontology_tools, configured_tools)
        self.assertTrue(
            self.mapping["query_rules"]["tool_supported_means_capability_not_result"]
        )
        for tool_name, contract in self.mapping["tool_contracts"].items():
            tool = TOOL_REGISTRY[tool_name]
            self.assertEqual(tool.input_model.__name__, contract["model"])
            schema_text = json.dumps(tool.input_schema)
            for path in contract.get("required_input_paths", []):
                self.assertIn(path.split(".")[-1], schema_text)

    def test_required_controls_are_derived_from_candidate_metrics(self) -> None:
        for example in self.examples.values():
            expected = set()
            for candidate in example["candidates"]:
                tool_name = candidate["validation_action"]["tool_name"]
                for metric in candidate["validation_metrics"]:
                    expected.add(
                        (
                            metric["entity_id"],
                            candidate["process"]["canonical_name"],
                            tool_name,
                            candidate["relationship_id"],
                        )
                    )
            actual = {
                (
                    item["metric_id"],
                    item["process_id"],
                    item["validation_tool"],
                    relationship_id,
                )
                for item in example["required_controls"]
                for relationship_id in item["source_relationship_ids"]
            }
            self.assertEqual(expected, actual)

    def test_schema_supports_cross_process_control_aggregation_without_new_fact(self) -> None:
        controls = self.schema["properties"]["required_controls"]
        self.assertNotIn("maxItems", controls)
        process_enum = set(
            self.schema["$defs"]["required_control"]["properties"]["process_id"]["enum"]
        )
        self.assertEqual({"printing", "placement", "reflow"}, process_enum)
        all_example_relationships = {
            candidate["relationship_id"]
            for example in self.examples.values()
            for candidate in example["candidates"]
        }
        self.assertTrue(all_example_relationships <= set(self.relationships))

    def test_pending_dictionary_entities_are_explicitly_excluded(self) -> None:
        dictionary_ids = {item["entity_id"] for item in self.dictionary["entities"]}
        excluded = {
            item["entity_id"] for item in self.mapping["excluded_from_active_graph"]
        }
        self.assertEqual(
            {
                "METRIC-SPI-VTE-VARIANCE",
                "CRITERION-VTE-TARGET",
                "CRITERION-PWI-UPPER",
            },
            excluded,
        )
        self.assertTrue(excluded <= dictionary_ids)

    def test_cypher_defines_stable_uniqueness_and_query_indexes(self) -> None:
        cypher = CYPHER_PATH.read_text(encoding="utf-8")
        required_fragments = {
            "CREATE CONSTRAINT knowledge_entity_id_unique IF NOT EXISTS",
            "REQUIRE node.entity_id IS UNIQUE",
            "CREATE CONSTRAINT causal_hypothesis_relationship_id_unique IF NOT EXISTS",
            "REQUIRE hypothesis.relationship_id IS UNIQUE",
            "CREATE INDEX knowledge_entity_canonical_name IF NOT EXISTS",
            "CREATE INDEX causal_hypothesis_defect_id IF NOT EXISTS",
        }
        for fragment in required_fragments:
            self.assertIn(fragment, cypher)


if __name__ == "__main__":
    unittest.main()
