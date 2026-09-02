from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_batched_nested_norm_node_compiler_probe_r80.py"
SPEC = importlib.util.spec_from_file_location("p1553_r80", MODULE_PATH)
assert SPEC and SPEC.loader
R80 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R80)


class BatchedNestedNormNodeCompilerProbeTests(unittest.TestCase):
    def test_product_tree_multiplies_all_factor_roots(self) -> None:
        factors = [
            {
                "indices": (index,),
                "endpoint": (False, (index + 1,)),
                "polynomial": R80.R73.polynomial_from_roots(
                    [index + 1],
                    101,
                ),
                "identity_count": 0,
            }
            for index in range(4)
        ]
        tree, receipts = R80.build_product_tree(factors, 101)
        expected = R80.R73.polynomial_from_roots([1, 2, 3, 4], 101)
        self.assertEqual(tree["polynomial"], expected)
        self.assertEqual(receipts["leaf_count"], 4)
        self.assertEqual(receipts["internal_node_count"], 3)

    def test_gcd_degree_is_not_tuple_count_control(self) -> None:
        control = R80.synthetic_gcd_count_control()
        self.assertEqual(control["mobius_tuple_count"], 4)
        self.assertEqual(control["product_polynomial_gcd_degree"], 3)
        self.assertTrue(control["gcd_degree_is_not_tuple_count"])

    def test_small_curve_replays_gcd_source_and_nonclaim(self) -> None:
        curve = dict(R80.R76.R74.R72.CURVES[0])
        result = R80.probe_curve(curve, prefix_sizes=(3,))
        targets = result["prefixes"][0]["targets"]
        blind = next(
            row for row in targets if row["target_id"] == "blind_hash_target"
        )
        forced = next(
            row
            for row in targets
            if row["target_id"] == "forced_positive_target"
        )
        self.assertTrue(blind["product_polynomial_detects_existence"])
        self.assertEqual(blind["mobius_exact_tuple_count"], 0)
        self.assertTrue(forced["product_polynomial_detects_existence"])
        self.assertEqual(forced["mobius_exact_tuple_count"], 1)
        self.assertTrue(forced["product_tree_source_relation_verified"])

    def test_small_report_records_strict_improvement_but_fails_caps(self) -> None:
        report = R80.build_report(prefix_sizes=(3,))
        compiled = report["cost_ledger"]["compiled_fast_arithmetic"]
        integrated = report["cost_ledger"]["best_exact_integrated_path"]
        self.assertTrue(compiled["strict_improvement_over_leaf_B5"])
        self.assertFalse(compiled["lane_inside_caps"])
        self.assertEqual(integrated["fresh_target_work_exponent_B"], 3.0)
        self.assertEqual(integrated["fresh_target_workspace_exponent_B"], 2.0)
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
