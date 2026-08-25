from __future__ import annotations

import sys
import unittest
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_DIR))

from run_reranker_experiment import aggregate, query_metrics, rerank_rows


def rows() -> list[dict[str, str]]:
    return [
        {
            "chunk_id": f"chunk-{rank}",
            "rrf_rank": str(rank),
            "relevance": "2" if rank == 8 else "0",
            "source_id": "source",
            "pdf_page_start": str(rank),
            "pdf_page_end": str(rank),
        }
        for rank in range(1, 21)
    ]


class RerankerExperimentTests(unittest.TestCase):
    def test_direct_gold_can_move_into_top5(self) -> None:
        candidates = rows()
        scores = [1.0 if row["relevance"] == "2" else 0.0 for row in candidates]
        reranked = rerank_rows(candidates, scores)
        metrics = query_metrics("Q001", candidates, reranked)
        self.assertEqual(metrics["before_first_rel2_rank"], 8)
        self.assertEqual(metrics["after_first_rel2_rank"], 1)
        self.assertTrue(metrics["after_top5_hit"])

    def test_stable_tie_break_uses_original_rank(self) -> None:
        candidates = rows()
        reranked = rerank_rows(candidates, [0.0] * 20)
        self.assertEqual([row["rrf_rank"] for row in reranked], [str(i) for i in range(1, 21)])

    def test_aggregate_reports_hit_and_mrr(self) -> None:
        metrics = [
            {
                "before_top5_hit": False,
                "after_top5_hit": True,
                "before_mrr_at_5": 0.0,
                "after_mrr_at_5": 0.5,
            },
            {
                "before_top5_hit": False,
                "after_top5_hit": False,
                "before_mrr_at_5": 0.0,
                "after_mrr_at_5": 0.0,
            },
        ]
        summary = aggregate(metrics)
        self.assertEqual(summary["after_top5_hit_count"], 1)
        self.assertEqual(summary["after_mrr_at_5"], 0.25)


if __name__ == "__main__":
    unittest.main()
