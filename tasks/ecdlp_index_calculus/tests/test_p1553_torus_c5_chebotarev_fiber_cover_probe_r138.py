from __future__ import annotations

import importlib.util
import pathlib
import unittest
from fractions import Fraction


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_chebotarev_fiber_cover_probe_r138.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r138", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R138 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R138)


class TorusC5ChebotarevFiberCoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R138.build_bundle()
        cls.report = cls.bundle["report"]
        cls.theorem = cls.report["theorem"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.bundle["cost"]["routes"]
        }

    def test_source_bindings_and_primary_pdf_are_exact(self) -> None:
        self.assertEqual(len(R138.verify_source_bindings()), 11)
        self.assertEqual(
            R138.sha256_file(R138.CHEBOTAREV_PDF),
            R138.CHEBOTAREV_PDF_SHA256,
        )

    def test_multiplicative_order_and_fiber_depth_are_exact(self) -> None:
        self.assertEqual(R138.multiplicative_order_mod_prime(10, 11), 2)
        self.assertEqual(R138.multiplicative_order_mod_prime(2, 11), 10)
        self.assertEqual(R138.sparse_fiber_depth_lower_bound(17, 3), 9)
        self.assertEqual(R138.sparse_fiber_depth_lower_bound(17, 4), 6)

    def test_tuple_fiber_lemma_preserves_product_collisions(self) -> None:
        statement = self.theorem["tuple_fiber_cover_lemma"]
        self.assertIn("(t-1)*m^4", statement)
        self.assertIn("ceil(m/(t-1))", statement)
        self.assertIn("Product collisions do not weaken", statement)

    def test_conditional_trinomial_depth_is_B_three_quarters(self) -> None:
        statement = self.theorem["conditional_trinomial_tree_bound"]
        self.assertIn("ceil(m/2)", statement)
        self.assertIn("B^(3/4+o(1))", statement)
        self.assertEqual(
            self.theorem[
                "maximum_structured_node_mode_count_closed_conditionally"
            ],
            3,
        )
        self.assertEqual(
            self.routes[
                "conditional_atom_full_spark_trinomial_tree"
            ]["minimum_rejecting_path_depth_exponent_B"]["exact"],
            "3/4",
        )

    def test_corrected_chebotarev_hypothesis_is_not_transferred(self) -> None:
        source = self.theorem["primary_source"]
        self.assertIn("primitive modulo", source["version_boundary"])
        self.assertFalse(
            self.theorem[
                "chebotarev_transfer_to_actual_norm_one_families"
            ]
        )
        self.assertEqual(
            self.routes["finite_field_chebotarev_generic_transfer"][
                "required_primitive_order_extension_exponent_B"
            ]["exact"],
            "5",
        )

    def test_all_actual_characteristic_orders_are_two(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertTrue(
            self.controls["all_characteristic_orders_equal_two"]
        )
        self.assertFalse(
            self.controls["any_chebotarev_order_condition_satisfied"]
        )
        for control in self.controls["controls"]:
            self.assertEqual(
                control["field_prime_mod_subgroup_order"],
                control["subgroup_order"] - 1,
            )
            self.assertEqual(
                control["characteristic_order_mod_subgroup_order"],
                2,
            )

    def test_exact_all_mode_three_atom_controls_are_full_spark(self) -> None:
        self.assertEqual(
            self.controls["all_mode_three_atom_control_count"],
            2,
        )
        self.assertTrue(
            self.controls["all_exact_three_atom_controls_full_spark"]
        )
        exact = [
            row["all_mode_three_atom_control"]
            for row in self.controls["controls"]
            if row["all_mode_three_atom_control"] is not None
        ]
        for row in exact:
            self.assertEqual(
                row["normalized_projective_ratios_checked"],
                row["subgroup_order"] - 1,
            )
            self.assertIsNone(row["collision_modes"])

    def test_bounded_window_controls_are_exact_but_uncredited(self) -> None:
        self.assertEqual(self.controls["window_minor_count_checked"], 73920)
        self.assertTrue(self.controls["all_window_minors_nonsingular"])
        self.assertEqual(
            self.controls["eligible_colored_atom_triple_count"],
            0,
        )
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_credit"]
        )

    def test_surviving_routes_remain_open(self) -> None:
        for route_id in (
            "actual_atom_characteristic_specific_spark",
            "five_plus_mode_low_slp_zero_test_tree",
            "nonzero_value_frobenius_coordinate_dag",
        ):
            self.assertEqual(self.routes[route_id]["status"], "open")

    def test_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 23)
        self.assertTrue(
            admission["conditional_trinomial_tree_negative_admitted"]
        )
        self.assertTrue(
            admission["generic_chebotarev_transfer_negative_admitted"]
        )
        self.assertFalse(
            admission["actual_atom_restricted_spark_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
