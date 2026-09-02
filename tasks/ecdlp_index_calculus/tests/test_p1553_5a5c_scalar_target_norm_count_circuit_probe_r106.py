from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "p1553_5a5c_scalar_target_norm_count_circuit_probe_r106.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r106_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load R106 probe")
R106 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R106)


class ScalarTargetNormCountCircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R106.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["actual_and_matched_controls"]
        cls.theorem = cls.report["character_density_theorem"]

    def test_actual_and_matched_uncertainty_bounds(self) -> None:
        self.assertEqual(self.controls["actual_count"], 8)
        self.assertEqual(self.controls["matched_random_deck_count"], 8)
        self.assertTrue(self.controls["all_subgroup_orders_prime"])
        self.assertTrue(self.controls["all_uncertainty_bounds_positive"])
        self.assertTrue(
            self.controls["all_live_mode_fractions_above_99_percent"]
        )

    def test_finite_bound_matches_side_support_formula(self) -> None:
        for row in [
            *self.controls["actual"],
            *self.controls["matched_random_decks"],
        ]:
            expected = (
                row["subgroup_order_q"]
                - row["a5_endpoint_support"]
                - row["c5_endpoint_support"]
                + 2
            )
            self.assertEqual(
                row["full_count_fourier_support_lower_bound"],
                expected,
            )
            self.assertGreater(expected, 0)

    def test_character_mode_exponent_is_full_group(self) -> None:
        self.assertEqual(
            self.theorem["live_mode_exponent_B"]["exact"],
            "5",
        )
        self.assertFalse(self.theorem["character_state_inside_setup_cap"])
        self.assertFalse(
            self.theorem["dense_character_sum_inside_online_cap"]
        )

    def test_composite_order_positive_control_is_sparse(self) -> None:
        control = self.report["composite_order_positive_control"]
        self.assertEqual(control["primal_support_size"], 8)
        self.assertEqual(control["fourier_support"], [0, 8])
        self.assertEqual(control["fourier_support_size"], 2)
        self.assertTrue(control["violates_prime_order_additive_bound"])

    def test_scope_keeps_noncharacter_circuits_open(self) -> None:
        self.assertIn(
            "not an arithmetic-circuit lower bound",
            self.theorem["scope"],
        )
        self.assertTrue(self.report["character_diagonal_constructor_closed"])
        self.assertTrue(self.report["noncharacter_scalar_constructor_open"])

    def test_bundle_has_no_algorithm_claim(self) -> None:
        self.assertFalse(self.report["admission"]["lane_admitted"])
        self.assertFalse(self.report["factor_log_solve_complete"])
        self.assertFalse(self.report["fresh_target_descent_complete"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
