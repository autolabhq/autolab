import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "p1553_torus_c5_sextic_mobius_character_router_probe_r141.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r141_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R141 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R141)


class TorusC5SexticMobiusCharacterRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R141.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R141.verify_source_bindings()), 10)

    def test_cayley_claw_form_replays(self):
        self.assertTrue(
            self.controls["all_actual_cayley_claw_identities_exact"]
        )

    def test_all_actual_positive_inverse_controls_are_exact(self):
        self.assertEqual(self.controls["actual_control_count"], 8)
        self.assertTrue(self.controls["all_actual_c5_supports_injective"])
        self.assertTrue(
            self.controls["all_actual_positive_inverses_empty"]
        )

    def test_greedy_character_signatures_separate_all_inverses(self):
        self.assertTrue(
            self.controls["all_actual_inverse_pairs_separated"]
        )
        self.assertEqual(
            self.controls["maximum_selected_parameter_count"],
            4,
        )

    def test_actual_product_character_matrices_have_full_row_rank(self):
        self.assertTrue(
            self.controls[
                "all_actual_c2_by_c3_matrices_full_row_rank"
            ]
        )

    def test_multiplicative_defect_attains_every_coset(self):
        self.assertTrue(
            self.controls[
                "all_actual_multiplicative_defects_attain_six_cosets"
            ]
        )

    def test_synthetic_fourier_support_is_q_minus_one_or_more(self):
        self.assertEqual(self.controls["synthetic_control_count"], 8)
        self.assertTrue(
            self.controls[
                "all_synthetic_fourier_supports_at_least_q_minus_one"
            ]
        )

    def test_character_cost_is_polylog_but_linear_orbit_is_B5(self):
        cost = self.bundle["cost"]
        self.assertEqual(
            cost["inverse_separator_exponent_B"]["exact"],
            "0",
        )
        self.assertEqual(
            cost["linear_translation_rank_exponent_B"]["exact"],
            "5",
        )
        self.assertFalse(cost["linear_translation_inside_setup_cap"])

    def test_weil_parseval_scope_is_explicit(self):
        theorem = self.report["theorem"]
        self.assertEqual(
            theorem["primary_source"]["doi"],
            "10.1073/pnas.34.5.204",
        )
        self.assertIn("translation-invariant linear", theorem["scope"])
        self.assertIn("not a lower bound", theorem["scope"])

    def test_gate_does_not_promote_source_router_or_algorithm(self):
        admission = self.report["admission"]
        self.assertTrue(admission["polylog_inverse_separator_admitted"])
        self.assertTrue(
            admission["linear_translation_rank_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(
            admission["obligations"][
                "nonlinear_character_source_router_complete"
            ]
        )
        self.assertFalse(self.report["breakthrough"])
        self.assertFalse(self.report["shoup_bound_improvement"])


if __name__ == "__main__":
    unittest.main()
