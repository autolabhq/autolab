from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_actual_s6_fermat_tensor_train_probe_r78.py"
SPEC = importlib.util.spec_from_file_location("p1553_r78", MODULE_PATH)
assert SPEC and SPEC.loader
R78 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R78)


class ActualS6FermatTensorTrainProbeTests(unittest.TestCase):
    def test_tt_ranks_detect_rank_one_and_zero_tensors(self) -> None:
        size = 2
        rank_one = [
            (indices[0] + 1)
            * (indices[1] + 1)
            * (indices[2] + 1)
            * (indices[3] + 1)
            * (indices[4] + 1)
            for indices in __import__("itertools").product(
                range(size), repeat=5
            )
        ]
        self.assertEqual(R78.tt_ranks(rank_one, size, 101), [1, 1, 1, 1])
        self.assertEqual(
            R78.tt_ranks([0] * (size**5), size, 101),
            [0, 0, 0, 0],
        )

    def test_fermat_control_preserves_duplicate_occurrences(self) -> None:
        control = R78.fermat_occurrence_control()
        self.assertEqual(control["zero_mask"], [1, 1, 0])
        self.assertTrue(control["duplicate_occurrences_preserved"])

    def test_small_standard_curve_replays_source_and_nonclaim(self) -> None:
        curve = dict(R78.R72.CURVES[0])
        result = R78.probe_curve(curve, prefix_sizes=(3,))
        blind = next(
            row
            for row in result["targets"]
            if row["target_id"] == "blind_hash_target"
        )
        forced = next(
            row
            for row in result["targets"]
            if row["target_id"] == "forced_positive_target"
        )
        self.assertEqual(blind["raw_s6_zero_occurrence_count"], 0)
        self.assertEqual(forced["raw_s6_zero_occurrence_count"], 1)
        self.assertTrue(forced["source_relation_verified"])
        self.assertTrue(
            forced["dyadic_child_replay"][
                "all_parent_counts_equal_child_sums"
            ]
        )

    def test_small_report_keeps_cost_boundary(self) -> None:
        report = R78.build_report(prefix_sizes=(3,))
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertEqual(
            report["cost_ledger"]["raw_value_tensor_entry_exponent_B"],
            5.0,
        )


if __name__ == "__main__":
    unittest.main()
