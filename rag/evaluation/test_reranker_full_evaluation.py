from __future__ import annotations

import sys
import unittest
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_DIR))

from run_reranker_full_evaluation import aggregate_scores, score_order


def gold_query() -> dict:
    graded = [
        {
            "chunk_id": f"chunk-{rank}",
            "relevance": 2 if rank in {1, 6} else (1 if rank == 2 else 0),
            "source_id": "gold" if rank in {1, 6} else "other",
            "pdf_page_start": rank,
            "pdf_page_end": rank,
        }
        for rank in range(1, 21)
    ]
    return {
        "query_id": "Q001",
        "direct_gold_chunks": [graded[0], graded[5]],
        "supporting_chunks": [graded[1]],
        "graded_relevance": graded,
    }


def rows(order: list[int]) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk-{rank}",
            "source_id": "gold" if rank in {1, 6} else "other",
            "pdf_page_start": str(rank),
            "pdf_page_end": str(rank),
        }
        for rank in order
    ]


class FullRerankerEvaluationTests(unittest.TestCase):
    def test_scores_top5_top10_and_page_hits(self) -> None:
        scored = score_order(gold_query(), rows(list(range(1, 21))))
        self.assertEqual(scored["recall_at_5"], 0.5)
        self.assertEqual(scored["direct_gold_count"], 2)
        self.assertEqual(scored["direct_hit_count_at_5"], 1)
        self.assertEqual(scored["mrr_at_10"], 1.0)
        self.assertTrue(scored["source_hit_at_5"])
        self.assertTrue(scored["page_hit_at_5"])

    def test_reranked_direct_chunks_improve_recall(self) -> None:
        order = [6, 1] + [rank for rank in range(2, 21) if rank != 6]
        scored = score_order(gold_query(), rows(order))
        self.assertEqual(scored["recall_at_5"], 1.0)
        summary = aggregate_scores([scored])
        self.assertEqual(summary["direct_hit_at_5_count"], 1)
        self.assertEqual(summary["macro_recall_at_5"], 1.0)
        self.assertEqual(summary["micro_rel2_recall_at_5"], 1.0)

    def test_macro_and_micro_recall_are_reported_separately(self) -> None:
        common = {
            "candidate_answerable": True,
            "mrr_at_10": 1.0,
            "ndcg_at_10": 1.0,
            "source_hit_at_5": True,
            "page_hit_at_5": True,
            "first_rel2_rank": 1,
        }
        summary = aggregate_scores(
            [
                {
                    **common,
                    "query_id": "Q001",
                    "recall_at_5": 1.0,
                    "direct_gold_count": 1,
                    "direct_hit_count_at_5": 1,
                },
                {
                    **common,
                    "query_id": "Q002",
                    "recall_at_5": 0.4,
                    "direct_gold_count": 5,
                    "direct_hit_count_at_5": 2,
                },
            ]
        )
        self.assertAlmostEqual(summary["macro_recall_at_5"], 0.7)
        self.assertAlmostEqual(summary["micro_rel2_recall_at_5"], 0.5)
        self.assertEqual(summary["rel2_gold_count"], 6)
        self.assertEqual(summary["rel2_hit_at_5_count"], 3)

    def test_unanswerable_query_is_excluded_from_primary_metrics(self) -> None:
        query = gold_query()
        query["direct_gold_chunks"] = []
        scored = score_order(query, rows(list(range(1, 21))))
        self.assertFalse(scored["candidate_answerable"])


if __name__ == "__main__":
    unittest.main()
