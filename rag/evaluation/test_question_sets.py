from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.evaluation.prepare_question_sets import (
    QUERY_OVERRIDES,
    QUESTION_GROUPS,
    build_question_sets,
)


class QuestionSetTests(unittest.TestCase):
    def test_approved_groups_and_rewrites(self) -> None:
        source = (
            Path(__file__).resolve().parent
            / "archive/legacy_labeled/evaluation_final.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            manifest = build_question_sets(source, output_dir)
            self.assertEqual(manifest["formal_retriever_goal_query_count"], 16)
            self.assertEqual(manifest["selected_query_count"], 24)
            all_ids = [query_id for ids in QUESTION_GROUPS.values() for query_id in ids]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            main = json.loads(
                (output_dir / "retriever_main.json").read_text(encoding="utf-8")
            )
            by_id = {item["query_id"]: item for item in main["items"]}
            for query_id, query in QUERY_OVERRIDES.items():
                self.assertEqual(by_id[query_id]["query"], query)
                self.assertEqual(by_id[query_id]["query_version"], "rewritten_v1")


if __name__ == "__main__":
    unittest.main()
