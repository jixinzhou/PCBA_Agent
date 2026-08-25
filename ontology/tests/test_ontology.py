from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = PROJECT_ROOT / "ontology/pcba_defect_causality.v1.1.yaml"
SCHEMA_PATH = PROJECT_ROOT / "ontology/schemas/pcba_defect_causality.v1.schema.json"
DICTIONARY_PATH = PROJECT_ROOT / "rag/schemas/entity_dictionary.json"


class T09V11OntologyTests(unittest.TestCase):
    EXPECTED_DEFECTS = {
        "insufficient_solder",
        "excessive_solder",
        "short",
        "shifted_component",
    }
    EXPECTED_RELATIONSHIPS = {
        (
            "insufficient_solder",
            "CAUSE-INSUFFICIENT-SOLDER-DEPOSITION",
            "printing",
            "strong",
            "tool_supported",
        ),
        (
            "excessive_solder",
            "CAUSE-EXCESSIVE-SOLDER-DEPOSITION",
            "printing",
            "medium_strong",
            "tool_supported",
        ),
        (
            "short",
            "CAUSE-WET-PASTE-BRIDGING",
            "printing",
            "strong",
            "unverified",
        ),
        (
            "shifted_component",
            "CAUSE-PLACEMENT-OFFSET",
            "placement",
            "strong",
            "unverified",
        ),
        (
            "shifted_component",
            "CAUSE-REFLOW-THERMAL-IMBALANCE",
            "reflow",
            "conditional",
            "tool_supported",
        ),
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.dictionary = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))

    def test_ontology_matches_schema(self) -> None:
        Draft202012Validator.check_schema(self.schema)
        Draft202012Validator(self.schema).validate(self.ontology)

    def test_exact_frozen_defect_and_relationship_scope(self) -> None:
        defects = {item["canonical_name"] for item in self.ontology["defects"]}
        self.assertEqual(self.EXPECTED_DEFECTS, defects)
        self.assertEqual(self.EXPECTED_DEFECTS, set(self.ontology["scope"]["defect_ids"]))
        relationships = {
            (
                item["defect_id"],
                item["candidate_cause_id"],
                item["process_id"],
                item["relation_strength"],
                item["verification_status"],
            )
            for item in self.ontology["relationships"]
        }
        self.assertEqual(self.EXPECTED_RELATIONSHIPS, relationships)
        self.assertEqual(5, self.ontology["scope"]["relationship_count"])

    def test_references_and_processes_are_consistent(self) -> None:
        defects = {item["canonical_name"] for item in self.ontology["defects"]}
        causes = {item["entity_id"]: item for item in self.ontology["candidate_causes"]}
        metrics = {item["entity_id"]: item for item in self.ontology["validation_metrics"]}
        tools = {item["canonical_name"]: item for item in self.ontology["tools"]}
        process_ids = {item["canonical_name"] for item in self.ontology["processes"]}
        all_entity_ids = [
            item["entity_id"]
            for group in (
                "processes",
                "defects",
                "candidate_causes",
                "validation_metrics",
                "tools",
            )
            for item in self.ontology[group]
        ]
        self.assertEqual(len(all_entity_ids), len(set(all_entity_ids)))
        for relation in self.ontology["relationships"]:
            self.assertIn(relation["defect_id"], defects)
            self.assertIn(relation["process_id"], process_ids)
            cause = causes[relation["candidate_cause_id"]]
            self.assertEqual(relation["process_id"], cause["process_id"])
            for metric_id in relation["validation_metric_ids"]:
                metric = metrics[metric_id]
                self.assertEqual(relation["process_id"], metric["process_id"])
            if relation["validation_tool"] is not None:
                self.assertEqual(
                    relation["process_id"], tools[relation["validation_tool"]]["process_id"]
                )
            if relation["optimization_tool"] is not None:
                self.assertEqual(
                    relation["process_id"], tools[relation["optimization_tool"]]["process_id"]
                )

    def test_no_tool_relationships_are_retained_as_unverified(self) -> None:
        unverified = {
            item["relationship_id"]: item
            for item in self.ontology["relationships"]
            if item["verification_status"] == "unverified"
        }
        self.assertEqual(
            {
                "REL-SHORT-WET-PASTE-BRIDGING",
                "REL-SHIFTED-COMPONENT-PLACEMENT",
            },
            set(unverified),
        )
        for relation in unverified.values():
            self.assertIsNone(relation["validation_tool"])
            self.assertIsNone(relation["optimization_tool"])
            self.assertEqual("strong", relation["relation_strength"])
        self.assertTrue(self.ontology["system_principles"]["unverified_relations_are_retained"])

    def test_tool_supported_relationships_use_observable_metrics(self) -> None:
        metrics = {item["entity_id"]: item for item in self.ontology["validation_metrics"]}
        for relation in self.ontology["relationships"]:
            if relation["verification_status"] != "tool_supported":
                continue
            self.assertIsNotNone(relation["validation_tool"])
            self.assertIsNotNone(relation["optimization_tool"])
            for metric_id in relation["validation_metric_ids"]:
                self.assertTrue(metrics[metric_id]["tool_observable"])

    def test_dictionary_is_aligned_with_authoritative_ontology(self) -> None:
        self.assertEqual("1.1.0", self.dictionary["dictionary_version"])
        self.assertEqual(
            "ontology/pcba_defect_causality.v1.1.yaml",
            self.dictionary["ontology_source"],
        )
        self.assertEqual(self.ontology["ontology_version"], self.dictionary["ontology_version"])
        dictionary_entities = {
            item["entity_id"]: item for item in self.dictionary["entities"]
        }
        ontology_groups = {
            "processes": "Process",
            "defects": "Defect",
            "candidate_causes": "CandidateCause",
        }
        for group, entity_type in ontology_groups.items():
            expected_ids = {item["entity_id"] for item in self.ontology[group]}
            actual_ids = {
                item["entity_id"]
                for item in self.dictionary["entities"]
                if item["entity_type"] == entity_type
            }
            self.assertEqual(expected_ids, actual_ids)
            for ontology_entity in self.ontology[group]:
                dictionary_entity = dictionary_entities[ontology_entity["entity_id"]]
                for field in ("canonical_name", "display_name_zh", "aliases"):
                    self.assertEqual(ontology_entity[field], dictionary_entity[field])
                if group == "candidate_causes":
                    self.assertEqual(
                        ontology_entity["process_id"],
                        dictionary_entity["properties"]["process"],
                    )

        for metric in self.ontology["validation_metrics"]:
            dictionary_metric = dictionary_entities[metric["entity_id"]]
            self.assertEqual("QualityMetric", dictionary_metric["entity_type"])
            for field in ("canonical_name", "display_name_zh", "aliases"):
                self.assertEqual(metric[field], dictionary_metric[field])
            self.assertEqual(metric["process_id"], dictionary_metric["properties"]["process"])
            self.assertEqual(
                metric["tool_observable"],
                dictionary_metric["properties"]["tool_observable"],
            )
            self.assertEqual(metric["api_field"], dictionary_metric["properties"]["api_field"])

        ontology_tools = {item["entity_id"]: item for item in self.ontology["tools"]}
        dictionary_tool_ids = {
            item["entity_id"]
            for item in self.dictionary["entities"]
            if item["entity_type"] == "ValidationTool"
        }
        self.assertEqual(set(ontology_tools), dictionary_tool_ids)
        for entity_id, ontology_tool in ontology_tools.items():
            dictionary_tool = dictionary_entities[entity_id]
            self.assertEqual(ontology_tool["canonical_name"], dictionary_tool["canonical_name"])
            self.assertEqual(ontology_tool["display_name_zh"], dictionary_tool["display_name_zh"])
            self.assertEqual(ontology_tool["process_id"], dictionary_tool["properties"]["process"])
            self.assertEqual(ontology_tool["tool_role"], dictionary_tool["properties"]["tool_type"])

    def test_removed_legacy_defects_do_not_reenter_dictionary(self) -> None:
        canonical_names = {
            item["canonical_name"] for item in self.dictionary["entities"]
        }
        self.assertNotIn("cold_solder_joint", canonical_names)
        self.assertNotIn("tombstoning", canonical_names)
        self.assertNotIn("insufficient_reflow_heat_input", canonical_names)

    def test_rag_metadata_schema_uses_same_four_defects(self) -> None:
        metadata_schema = json.loads(
            (PROJECT_ROOT / "rag/schemas/metadata.v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        defect_enum = set(metadata_schema["properties"]["defect_ids"]["items"]["enum"])
        self.assertEqual(self.EXPECTED_DEFECTS, defect_enum)


if __name__ == "__main__":
    unittest.main()
