from __future__ import annotations

import unittest

from rag.evaluation.prepare_retriever_main_annotations import prepare_rows


class PrepareRetrieverMainAnnotationsTest(unittest.TestCase):
    def test_reuses_only_exact_unchanged_query(self) -> None:
        old = [{
            "query_id": "Q002", "query": "same", "chunk_id": "C1", "text": "t",
            "final_answerable": "true", "relevance": "2", "model_id": "m",
            "prompt_sha256": "p", "annotation_status": "completed", "annotation_error": "",
        }]
        new = [
            {"query_id": "Q002", "query": "same", "chunk_id": "C1", "text": "t", "candidate_id": "A", "label_origin": "", "relevance": "", "final_answerable": "", "model_id": "", "prompt_sha256": "", "annotation_status": "pending", "annotation_error": ""},
            {"query_id": "Q019", "query": "rewrite", "chunk_id": "C2", "text": "u", "candidate_id": "B", "label_origin": "", "relevance": "", "final_answerable": "", "model_id": "", "prompt_sha256": "", "annotation_status": "pending", "annotation_error": ""},
        ]
        rows, counts = prepare_rows(new, old)
        self.assertEqual(counts, {"reused": 1, "pending": 1})
        self.assertEqual(rows[0]["relevance"], "2")
        self.assertEqual(rows[1]["annotation_status"], "pending")


if __name__ == "__main__":
    unittest.main()
