from __future__ import annotations

import unittest

from rag.evaluation.verify_frozen_evaluation import verify_frozen_evaluation


class FrozenEvaluationTest(unittest.TestCase):
    def test_manifest_and_annotated_pool(self) -> None:
        result = verify_frozen_evaluation()
        self.assertEqual("valid", result["status"])
        self.assertEqual(320, result["candidate_count"])
        self.assertEqual({"0": 185, "1": 92, "2": 43}, result["relevance_counts"])


if __name__ == "__main__":
    unittest.main()
