from __future__ import annotations

import sys
import unittest
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_DIR))

from run_reranker_rank_fusion_experiment import (
    RankFusionExperimentError,
    fuse_rows,
    weight_grid,
)


def candidate(rank: int) -> dict:
    return {
        "chunk_id": f"chunk-{rank}",
        "rrf_rank": str(rank),
        "relevance": "0",
    }


class RankFusionExperimentTests(unittest.TestCase):
    def test_weight_endpoints_reproduce_input_orders(self) -> None:
        original = [candidate(rank) for rank in range(1, 6)]
        reranked = list(reversed(original))
        rrf_order = [row["chunk_id"] for row in fuse_rows(original, reranked, 0.0, 60)]
        reranker_order = [
            row["chunk_id"] for row in fuse_rows(original, reranked, 1.0, 60)
        ]
        self.assertEqual(rrf_order, [row["chunk_id"] for row in original])
        self.assertEqual(reranker_order, [row["chunk_id"] for row in reranked])

    def test_weight_grid_includes_both_endpoints(self) -> None:
        values = weight_grid(0.0, 1.0, 0.01)
        self.assertEqual(len(values), 101)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)

    def test_candidate_mismatch_is_rejected(self) -> None:
        with self.assertRaises(RankFusionExperimentError):
            fuse_rows([candidate(1)], [candidate(2)], 0.5, 60)


if __name__ == "__main__":
    unittest.main()
