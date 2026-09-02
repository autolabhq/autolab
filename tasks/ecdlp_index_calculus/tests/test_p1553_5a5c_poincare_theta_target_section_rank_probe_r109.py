from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "p1553_5a5c_poincare_theta_target_section_"
    "rank_probe_r109.py"
)


def load_producer():
    spec = importlib.util.spec_from_file_location("p1553_r109_test", PRODUCER)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R109 producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R109 = load_producer()


class PoincareThetaTargetSectionRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R109.build_bundle()

    def test_all_unary_pairwise_cylinders_escape_zero_fiber(self) -> None:
        controls = R109.actual_and_matched_controls()
        self.assertEqual(controls["instance_count"], 16)
        self.assertTrue(controls["all_cylinder_controls_exact"])
        for row in [
            *controls["cylinder_actual"],
            *controls["cylinder_matched_random"],
        ]:
            self.assertEqual(row["tested_cylinder_count"], 56)
            self.assertTrue(
                row["all_size_at_most_two_cylinders_escape_zero_fiber"]
            )

    def test_regular_pure_pairwise_product_is_refuted(self) -> None:
        theorem = R109.pure_pairwise_zero_theorem()
        self.assertTrue(
            theorem["regular_finite_valued_pure_product_refuted"]
        )
        self.assertFalse(theorem["rational_pole_cancellation_covered"])
        self.assertFalse(theorem["bounded_sum_of_products_covered"])

    def test_balanced_images_and_sample_section_ranks_are_full(self) -> None:
        controls = R109.actual_and_matched_controls()
        self.assertTrue(controls["all_balanced_side_images_injective"])
        self.assertTrue(controls["all_sample_section_ranks_full"])
        self.assertEqual(
            controls["minimum_sample_rank"],
            controls["minimum_sample_dimension"],
        )

    def test_unique_poles_force_b12o5_uniform_rank(self) -> None:
        theorem = R109.translated_section_rank_theorem()
        self.assertEqual(
            theorem["uniform_two_block_section_rank_lower_exponent_B"][
                "exact"
            ],
            "12/5",
        )
        self.assertTrue(theorem["rank_lower_bound_above_setup_cap"])
        self.assertTrue(theorem["rank_lower_bound_above_online_cap"])
        self.assertFalse(theorem["ffe_scalar_extension_changes_rank"])

    def test_pair_tables_fit_but_do_not_supply_scalar_product(self) -> None:
        ledger = R109.cost_ledger()
        self.assertTrue(
            ledger["theorem_of_cube_pair_tables"][
                "all_pair_tables_inside_online_cap"
            ]
        )
        self.assertFalse(
            ledger["regular_pure_pairwise_scalar_product"][
                "constructor_valid"
            ]
        )
        self.assertFalse(
            ledger["regular_pure_pairwise_scalar_product"][
                "pairwise_norm_product_cost_credit"
            ]
        )

    def test_scope_preserves_implicit_high_rank_network(self) -> None:
        scope = R109.cost_ledger()["scope_boundary"]
        self.assertTrue(scope["pure_regular_pairwise_section_product_closed"])
        self.assertTrue(scope["uniform_separated_rank_below_B12O5_closed"])
        self.assertTrue(scope["rational_pole_cancellation_network_open"])
        self.assertTrue(scope["actual_deck_specific_high_rank_circuit_open"])
        self.assertFalse(
            scope["general_arithmetic_circuit_lower_bound_claimed"]
        )

    def test_bundle_preserves_nonclaim_and_routes_theta_network(self) -> None:
        report = self.bundle["report"]
        admission = report["admission"]
        self.assertFalse(admission["lane_admitted"])
        self.assertEqual(admission["passed_obligation_count"], 13)
        self.assertEqual(admission["obligation_count"], 26)
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])
        self.assertIn("theta-addition", report["next_action"])
        self.assertIn(
            "No lower bound for arbitrary arithmetic circuits is proved.",
            report["non_claims"],
        )


if __name__ == "__main__":
    unittest.main()
