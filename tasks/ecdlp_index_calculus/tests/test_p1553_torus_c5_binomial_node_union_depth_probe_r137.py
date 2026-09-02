from __future__ import annotations

import importlib.util
import pathlib
import unittest
from fractions import Fraction


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_binomial_node_union_depth_probe_r137.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r137", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R137 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R137)


class TorusC5BinomialNodeUnionDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R137.build_bundle()
        cls.report = cls.bundle["report"]
        cls.theorem = cls.report["theorem"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.bundle["cost"]["routes"]
        }

    def test_source_bindings_and_primary_pdf_are_exact(self) -> None:
        self.assertEqual(len(R137.verify_source_bindings()), 11)
        self.assertEqual(
            R137.sha256_file(R137.DEVOS_PDF),
            R137.DEVOS_PDF_SHA256,
        )

    def test_iterated_cauchy_davenport_formula(self) -> None:
        self.assertEqual(
            R137.cauchy_davenport_iterated_bound(101, 7, 5),
            31,
        )
        self.assertEqual(
            R137.cauchy_davenport_iterated_bound(11, 7, 5),
            11,
        )
        self.assertIn("5|A|-4", self.theorem["iterated_product_set_bound"])

    def test_random_depth_exponents_are_exact(self) -> None:
        self.assertEqual(
            [R137.random_depth_exponent_q(modes) for modes in (2, 3, 4)],
            [Fraction(3, 4), Fraction(1, 4), Fraction(1, 12)],
        )
        self.assertEqual(
            [
                row["minimum_depth_exponent_B"]["exact"]
                for row in self.theorem["random_depth_table"]
            ],
            ["15/4", "5/4", "5/12"],
        )

    def test_structured_binomial_depth_is_B_three_quarters(self) -> None:
        statement = self.theorem[
            "deterministic_binomial_depth_lower_bound"
        ]
        self.assertIn("5m-4", statement)
        self.assertIn("B^(3/4+o(1))", statement)
        self.assertEqual(
            self.theorem["maximum_deterministic_node_mode_count_closed"],
            2,
        )

    def test_all_actual_color_product_sets_meet_exact_bound(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertEqual(self.controls["active_color_control_count"], 30)
        self.assertTrue(
            self.controls[
                "all_color_product_sets_meet_cauchy_davenport"
            ]
        )
        self.assertTrue(
            self.controls["all_color_controls_are_bound_tight"]
        )

    def test_finite_product_set_sizes_are_one_or_six(self) -> None:
        for control in self.controls["controls"]:
            for color in control["color_controls"]:
                expected = 1 if color["part_size"] == 1 else 6
                self.assertEqual(color["product_set_size"], expected)
                self.assertEqual(
                    color["cauchy_davenport_lower_bound"],
                    expected,
                )

    def test_positive_and_inverse_empty_share_path(self) -> None:
        self.assertTrue(
            self.controls["all_selected_positives_in_global_c5"]
        )
        self.assertTrue(
            self.controls["all_inverse_empties_outside_global_c5"]
        )
        self.assertTrue(
            self.controls[
                "all_positive_inverse_pairs_share_all_nonzero_path"
            ]
        )

    def test_random_four_mode_result_receives_no_structured_credit(
        self,
    ) -> None:
        self.assertEqual(
            self.theorem["maximum_random_model_node_mode_count_closed"],
            4,
        )
        self.assertFalse(
            self.theorem[
                "random_support_model_transferred_to_structured_factor_base"
            ]
        )

    def test_surviving_selector_classes_remain_open(self) -> None:
        for route_id in (
            "structured_three_plus_mode_zero_test_tree",
            "five_plus_mode_low_slp_zero_test_tree",
            "nonzero_value_frobenius_coordinate_dag",
        ):
            self.assertEqual(self.routes[route_id]["status"], "open")

    def test_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 23)
        self.assertTrue(
            admission["structured_binomial_tree_negative_admitted"]
        )
        self.assertTrue(
            admission["random_model_four_mode_tree_negative_admitted"]
        )
        self.assertFalse(
            admission[
                "structured_three_plus_mode_or_nonzero_selector_admitted"
            ]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
