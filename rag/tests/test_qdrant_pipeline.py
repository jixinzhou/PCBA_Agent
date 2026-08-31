from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag/src"))

from pcba_rag.qdrant_pipeline import (
    build_payload,
    build_point,
    create_qdrant_client,
    load_index_inputs,
    load_qdrant_config,
    point_id_for_chunk,
    validate_collection,
    wait_for_qdrant,
)


class QdrantPointContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_qdrant_config(PROJECT_ROOT)
        cls.items = load_index_inputs(PROJECT_ROOT, cls.config)

    def test_all_inputs_join_by_chunk_id_and_text_hash(self) -> None:
        self.assertEqual(245, len(self.items))
        self.assertEqual(
            len(self.items), len({item.chunk["chunk_id"] for item in self.items})
        )
        for item in self.items:
            self.assertEqual(item.chunk["chunk_id"], item.embedding["chunk_id"])
            self.assertEqual(item.chunk["text_hash"], item.embedding["text_hash"])

    def test_uuid_v5_point_id_is_stable(self) -> None:
        chunk_id = "gjb_3243a_2021:c_26bfc87be68bb9e47aa76e51"
        point_id = point_id_for_chunk(
            chunk_id, self.config["point"]["uuid_namespace"]
        )
        self.assertEqual("9c4e2fa0-147f-5f9e-ae66-82286469bba7", point_id)
        self.assertEqual(
            point_id,
            point_id_for_chunk(chunk_id, self.config["point"]["uuid_namespace"]),
        )

    def test_point_has_two_named_vectors_and_complete_payload(self) -> None:
        item = self.items[0]
        point = build_point(item, self.config)
        self.assertEqual({"dense", "sparse"}, set(point.vector))
        self.assertEqual(1024, len(point.vector["dense"]))
        self.assertAlmostEqual(
            1.0, math.sqrt(sum(value * value for value in point.vector["dense"])), 6
        )
        self.assertGreater(len(point.vector["sparse"].indices), 0)
        payload = build_payload(item, self.config)
        self.assertEqual(item.chunk["text"], payload["text"])
        self.assertEqual(item.chunk["metadata"], payload["metadata"])
        self.assertEqual(item.chunk["page_ids"], payload["page_ids"])
        self.assertEqual(item.embedding["model"]["revision"], payload["embedding_model_revision"])

    def test_filter_indexes_cover_retriever_fields(self) -> None:
        self.assertEqual(
            {
                "source_id",
                "metadata.process_ids",
                "metadata.defect_ids",
                "metadata.evidence_roles",
                "metadata.language",
                "metadata.document_type",
                "semantic_tag_excluded",
            },
            set(self.config["payload_indexes"]),
        )

    def test_local_runtime_can_override_qdrant_url_without_editing_config(self) -> None:
        with patch.dict("os.environ", {"PCBA_QDRANT_URL": "http://127.0.0.1:16333"}):
            config = load_qdrant_config(PROJECT_ROOT)
        self.assertEqual("http://127.0.0.1:16333", config["qdrant"]["url"])


class GeneratedQdrantIndexTests(unittest.TestCase):
    def test_live_collection_if_available(self) -> None:
        summary_path = PROJECT_ROOT / "rag/reports/t10.7_v0.3_summary.json"
        if not summary_path.exists():
            self.skipTest("T10.7 index has not been built yet")
        config = load_qdrant_config(PROJECT_ROOT)
        client = create_qdrant_client(config)
        try:
            try:
                wait_for_qdrant(client, config)
            except (ConnectionError, RuntimeError):
                self.skipTest("Pinned Qdrant service is not available")
            if not client.collection_exists(config["collection"]["name"]):
                self.fail("T10.7 summary exists but Collection is missing")
            items = load_index_inputs(PROJECT_ROOT, config)
            validation = validate_collection(client, items, config)
        finally:
            client.close()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(len(items), validation["exact_point_count"])
        self.assertTrue(
            all(
                check["passed"]
                for check in validation["payload_filter_checks"].values()
            )
        )
        self.assertEqual(
            summary["validation"]["actual_fingerprint"],
            validation["actual_fingerprint"],
        )
        self.assertTrue(summary["acceptance_checks_passed"])


if __name__ == "__main__":
    unittest.main()
