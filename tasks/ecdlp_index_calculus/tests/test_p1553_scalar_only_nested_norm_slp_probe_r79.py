from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_scalar_only_nested_norm_slp_probe_r79.py"
SPEC = importlib.util.spec_from_file_location("p1553_r79", MODULE_PATH)
assert SPEC and SPEC.loader
R79 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R79)


class ScalarOnlyNestedNormSlpProbeTests(unittest.TestCase):
    def test_scalar_product_and_fermat_count(self) -> None:
        values = [3, 5, 0, 7]
        self.assertEqual(R79.scalar_product(values, 101), 0)
        self.assertEqual(
            R79.fermat_zero_indicators(values, 101),
            [0, 0, 1, 0],
        )

    def test_occurrence_control_separates_tuple_and_root_counts(self) -> None:
        control = R79.occurrence_semantics_control()
        self.assertEqual(control["tuple_occurrence_count"], 2)
        self.assertEqual(control["gcd_degree_sum"], 3)
        self.assertTrue(control["duplicates_preserved"])
        self.assertTrue(control["multiple_common_roots_count_tuple_once"])

    def test_scalar_tree_counts_charge_every_leaf(self) -> None:
        counts = R79.scalar_tree_counts(4)
        self.assertEqual(
            counts["leaf_s6_resultant_evaluation_count"],
            4**5,
        )
        self.assertEqual(
            counts["scalar_product_multiplication_count"],
            4**5 - 1,
        )
        self.assertEqual(
            counts["cached_scalar_node_count"],
            sum(4**level for level in range(6)),
        )

    def test_direct_curve_control_and_nonclaim(self) -> None:
        curve = dict(R79.R78.R72.CURVES[0])
        control = R79.direct_scalar_control(curve)
        self.assertTrue(control["all_root_products_match_counts"])
        report = R79.build_report()
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertEqual(
            report["cost_ledger"]["asymptotic"][
                "blind_failed_zero_certificate_work_exponent_B"
            ],
            5.0,
        )


if __name__ == "__main__":
    unittest.main()
