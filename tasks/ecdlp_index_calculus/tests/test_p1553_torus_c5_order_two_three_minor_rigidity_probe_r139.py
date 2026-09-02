from __future__ import annotations

import importlib.util
import pathlib
import unittest
from fractions import Fraction


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_order_two_three_minor_rigidity_probe_r139.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r139", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R139 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R139)


class TorusC5OrderTwoThreeMinorRigidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R139.build_bundle()
        cls.report = cls.bundle["report"]
        cls.theorem = cls.report["theorem"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.bundle["cost"]["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R139.verify_source_bindings()), 10)

    def test_affine_normalization_and_determinant_equation_are_explicit(
        self,
    ) -> None:
        self.assertIn("rows {0,1,r}", self.theorem["affine_normalization"])
        self.assertIn("modes {0,1,s}", self.theorem["affine_normalization"])
        self.assertIn(
            "(zeta-1)(zeta^(r*s)-1)",
            self.theorem["normalized_determinant_equation"],
        )

    def test_order_two_frobenius_forces_contradiction(self) -> None:
        self.assertIn(
            "zeta^(-1)",
            self.theorem["order_two_frobenius_step"],
        )
        self.assertIn("(r-1)(s-1)=0", self.theorem["contradiction"])
        self.assertEqual(
            self.theorem[
                "maximum_structured_node_mode_count_closed_unconditionally"
            ],
            3,
        )

    def test_one_normalized_certificate_replays_every_identity(self) -> None:
        field = R139.Field(29)
        root = R139.find_order_q_root(field, 5)
        row = R139.normalized_minor_certificate(
            root,
            5,
            2,
            3,
            field,
        )
        self.assertTrue(row["left_nonzero"])
        self.assertTrue(row["right_nonzero"])
        self.assertTrue(row["determinant_nonzero"])
        self.assertEqual(row["matrix_rank"], 3)
        self.assertTrue(row["left_frobenius_identity"])
        self.assertTrue(row["right_frobenius_identity"])
        self.assertFalse(row["left_right_multiplier_equal"])
        self.assertTrue(row["contradiction_exponent_nonzero"])

    def test_all_actual_normalized_controls_pass(self) -> None:
        self.assertEqual(self.controls["actual_control_count"], 8)
        self.assertEqual(
            self.controls["actual_normalized_minor_count"],
            2048,
        )
        for control in self.controls["actual_controls"]:
            self.assertEqual(
                control["field_prime_mod_subgroup_order"],
                control["subgroup_order"] - 1,
            )
            self.assertTrue(control["root_has_exact_prime_order"])

    def test_synthetic_normalized_sweeps_are_exhaustive(self) -> None:
        self.assertEqual(self.controls["synthetic_control_count"], 9)
        expected = sum(
            (order - 2) ** 2 for order in R139.SYNTHETIC_SUBGROUP_ORDERS
        )
        self.assertEqual(
            self.controls["synthetic_normalized_minor_count"],
            expected,
        )
        for control in self.controls["synthetic_controls"]:
            self.assertEqual(
                control["sweep"]["normalized_minor_count"],
                control["normalized_parameter_pair_count"],
            )

    def test_every_finite_certificate_passes_without_credit(self) -> None:
        self.assertTrue(self.controls["all_determinants_nonzero"])
        self.assertTrue(self.controls["all_matrix_ranks_three"])
        self.assertTrue(self.controls["all_frobenius_identities_exact"])
        self.assertTrue(
            self.controls["all_contradiction_exponents_nonzero"]
        )
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_credit"]
        )

    def test_structured_trinomial_tree_is_closed(self) -> None:
        route = self.routes["structured_trinomial_zero_test_tree"]
        self.assertEqual(
            route["minimum_rejecting_path_depth_exponent_B"]["exact"],
            "3/4",
        )
        self.assertEqual(
            route["minimum_accepting_path_depth_exponent_B"]["exact"],
            "5/2",
        )
        self.assertIn("rejected_deterministically", route["status"])

    def test_surviving_routes_remain_open(self) -> None:
        for route_id in (
            "structured_four_plus_mode_zero_test_tree",
            "five_plus_mode_low_slp_zero_test_tree",
            "nonzero_value_frobenius_coordinate_dag",
        ):
            self.assertEqual(self.routes[route_id]["status"], "open")

    def test_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 23)
        self.assertTrue(
            admission["order_two_three_minor_theorem_admitted"]
        )
        self.assertTrue(
            admission["structured_trinomial_tree_negative_admitted"]
        )
        self.assertFalse(
            admission[
                "structured_four_plus_or_nonzero_selector_admitted"
            ]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
