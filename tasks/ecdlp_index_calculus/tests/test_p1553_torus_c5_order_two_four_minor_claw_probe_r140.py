import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "p1553_torus_c5_order_two_four_minor_claw_probe_r140.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r140_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R140 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R140)


class TorusC5OrderTwoFourMinorClawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R140.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R140.verify_source_bindings()), 11)

    def test_all_actual_progression_controls_complete(self):
        self.assertEqual(self.controls["control_count"], 12)
        self.assertTrue(self.controls["all_bounded_mode_claws_found"])

    def test_mobius_factorization_and_involution_are_exact(self):
        self.assertTrue(
            self.controls["all_factorization_identities_exact"]
        )
        self.assertTrue(self.controls["all_mobius_linear_factors_zero"])
        self.assertTrue(self.controls["all_mobius_involutions_exact"])

    def test_order_two_images_retain_norm_one(self):
        self.assertTrue(self.controls["all_mode_values_norm_one"])

    def test_four_by_four_minors_have_rank_three(self):
        self.assertTrue(self.controls["all_matrices_rank_three"])
        self.assertTrue(
            self.controls["all_kernel_coefficients_nonzero"]
        )
        self.assertTrue(self.controls["all_kernels_exact"])

    def test_progression_root_sets_are_exactly_forced_four(self):
        self.assertTrue(
            self.controls[
                "all_progression_root_sets_exactly_forced_four"
            ]
        )
        for control in self.controls["controls"]:
            self.assertEqual(
                control["roots_in_six_point_progression"],
                [0, 1, 3, 4],
            )

    def test_finite_scans_do_not_consume_dlp_or_get_credit(self):
        self.assertFalse(
            self.controls["candidate_discrete_log_oracle_consumed"]
        )
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_credit"]
        )
        self.assertEqual(
            self.controls["maximum_scanned_mode_count"],
            18_377,
        )

    def test_generic_baseline_misses_setup_cap(self):
        cost = self.bundle["cost"]
        self.assertEqual(
            cost["generic_birthday_or_dlp_exponent_B"]["exact"],
            "5/2",
        )
        self.assertEqual(cost["setup_cap_exponent_B"]["exact"], "9/4")
        self.assertFalse(cost["generic_baseline_inside_setup_cap"])

    def test_gate_rejects_selector_and_algorithm_promotion(self):
        admission = self.report["admission"]
        self.assertTrue(
            admission["four_column_full_spark_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(
            admission["obligations"]["inside_cap_four_mode_selector"]
        )
        self.assertFalse(self.report["breakthrough"])

    def test_structured_claw_and_nonzero_routes_remain_open(self):
        next_action = self.report["next_action"]
        self.assertIn("sub-q^(9/20)", next_action)
        self.assertIn("nonzero-value", next_action)
        self.assertFalse(self.report["shoup_bound_improvement"])


if __name__ == "__main__":
    unittest.main()
