from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "kg/src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pcba_kg.core import RuntimeSettings, build_graph_plan, plan_summary


class T112ImportPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = yaml.safe_load(
            (PROJECT_ROOT / "kg/config/runtime.v1.yaml").read_text(encoding="utf-8")
        )
        import_config = cls.runtime["import"]
        cls.settings = RuntimeSettings(
            uri="bolt://127.0.0.1:7687",
            user="neo4j",
            password="unit-test-only",
            database="neo4j",
            ontology_path=PROJECT_ROOT / import_config["ontology_path"],
            mapping_path=PROJECT_ROOT / import_config["mapping_path"],
            schema_path=PROJECT_ROOT / import_config["schema_path"],
            expected_graph=cls.runtime["expected_graph"],
        )
        cls.plan = build_graph_plan(cls.settings)

    def test_frozen_plan_has_exactly_28_nodes_and_40_relationships(self) -> None:
        summary = plan_summary(self.plan)
        self.assertEqual(28, summary["total_nodes"])
        self.assertEqual(40, summary["total_relationships"])
        self.assertEqual(self.runtime["expected_graph"]["nodes"], summary["nodes"])
        self.assertEqual(
            self.runtime["expected_graph"]["relationships"], summary["relationships"]
        )

    def test_shifted_component_has_two_hypothesis_edges(self) -> None:
        defects = {
            row["properties"]["canonical_name"]: row["identity"]
            for row in self.plan.nodes["Defect"]
        }
        shifted_edges = [
            row
            for row in self.plan.relationships["HAS_HYPOTHESIS"]
            if row["from_id"] == defects["shifted_component"]
        ]
        self.assertEqual(2, len(shifted_edges))
        self.assertEqual(
            {
                "REL-SHIFTED-COMPONENT-PLACEMENT",
                "REL-SHIFTED-COMPONENT-REFLOW",
            },
            {row["to_id"] for row in shifted_edges},
        )

    def test_unverified_hypotheses_have_no_tool_edges(self) -> None:
        unverified = {
            row["identity"]
            for row in self.plan.nodes["CausalHypothesis"]
            if row["properties"]["verification_status"] == "unverified"
        }
        tool_sources = {
            row["from_id"]
            for edge_type in ("VALIDATED_BY", "OPTIMIZED_BY")
            for row in self.plan.relationships[edge_type]
        }
        self.assertEqual(2, len(unverified))
        self.assertTrue(unverified.isdisjoint(tool_sources))

    def test_compose_pins_server_and_named_volume(self) -> None:
        compose = yaml.safe_load(
            (PROJECT_ROOT / "kg/docker-compose.neo4j.yml").read_text(encoding="utf-8")
        )
        service = compose["services"]["neo4j"]
        self.assertEqual("neo4j:2026.07.1", service["image"])
        self.assertEqual("pcba-neo4j", service["container_name"])
        self.assertIn("pcba_neo4j_data:/data", service["volumes"])
        self.assertEqual("pcba_neo4j_data", compose["volumes"]["pcba_neo4j_data"]["name"])

    def test_driver_is_exactly_pinned_and_official_package_name_is_used(self) -> None:
        requirements = (PROJECT_ROOT / "kg/requirements.txt").read_text(encoding="utf-8")
        self.assertEqual("neo4j==6.2.0", requirements.strip())
        self.assertNotIn("neo4j-driver", requirements)

    def test_local_secret_is_ignored_and_example_is_safe(self) -> None:
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        env_example = (PROJECT_ROOT / "kg/.env.example").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore)
        self.assertIn("!/kg/.env.example", gitignore)
        self.assertIn("change_me_before_use", env_example)

    def test_neo4j_store_module_is_available_after_driver_install(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("pcba_kg.neo4j_store"))


if __name__ == "__main__":
    unittest.main()
