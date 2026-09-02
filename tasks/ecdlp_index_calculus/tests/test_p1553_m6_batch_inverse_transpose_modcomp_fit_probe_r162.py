from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_batch_inverse_transpose_modcomp_fit_probe_r162.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r162", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R162 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R162)


class M6BatchInverseTransposeModcompFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R162.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R162.verify_source_bindings()
        self.assertEqual(len(actual), 12)
        self.assertEqual(
            actual["modcomp_precomputation_2020"],
            "9fdce743a5183f544df2e2d640e30e6eb2cf2c647b6d89b04aea5faa21cdb488",
        )
        self.assertEqual(
            actual["two_relation_modcomp_2026"],
            "bfa0a9fb8f3ec6cd1d2aa95907a03df131d6a4ffb3abac56983bfee42c235866",
        )

    def test_prior_pair_sum_and_divisor_lanes_are_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("translated-divisor", dedup["r117"])
        self.assertIn("static pair-sum", dedup["r148"])
        self.assertIn("only compresses", dedup["r161"])

    def test_preregistered_control_grid_is_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)

    def test_c3_divisor_degree_is_inherited_exactly(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["c3_divisor_degree"],
                math.comb(row["factor_base_dimension"] + 2, 3),
            )

    def test_batch_u_evaluation_matches_direct(self) -> None:
        self.assertEqual(
            self.controls["exact_batch_u_evaluation_control_count"], 6
        )
        self.assertTrue(
            all(row["batch_u_evaluation_matches_direct"] for row in self.rows)
        )

    def test_exceptional_roots_are_detected_exactly(self) -> None:
        self.assertEqual(
            self.controls["exact_exceptional_detection_control_count"], 6
        )
        for row in self.rows:
            self.assertTrue(row["exceptional_detection_exact"])
            self.assertEqual(
                row["expected_exceptional_flags"],
                row["detected_exceptional_flags"],
            )
            self.assertTrue(any(row["detected_exceptional_flags"]))

    def test_quotient_and_inverse_identities_are_exact(self) -> None:
        self.assertEqual(
            self.controls["exact_quotient_inverse_control_count"], 6
        )
        for row in self.rows:
            self.assertTrue(row["all_quotient_identities_exact"])
            self.assertTrue(row["all_regular_inverse_identities_exact"])

    def test_three_transposed_functionals_are_exact(self) -> None:
        self.assertEqual(
            self.controls["exact_transposed_functional_control_count"], 6
        )
        for row in self.rows:
            self.assertEqual(row["functional_count"], 3)
            self.assertTrue(row["all_quotient_functionals_exact"])
            self.assertTrue(row["all_inverse_functionals_exact"])
            self.assertTrue(
                all(
                    functional["quotient_functional_values_exact"]
                    and functional[
                        "inverse_functional_values_exact_off_exceptional_roots"
                    ]
                    for functional in row["functional_rows"]
                )
            )

    def test_published_precomputation_orientation_does_not_fit(self) -> None:
        fit = self.report["literature"]["generic_precomputation"]["fit"]
        self.assertIn("precomputes both M and A", fit)
        self.assertIn("varies A=phi_j", fit)
        self.assertIn("does not share", fit)

    def test_published_single_composition_result_is_not_a_batch(self) -> None:
        fit = self.report["literature"]["two_relation_matrices"]["fit"]
        self.assertIn("one generic modular composition", fit)
        self.assertIn("does not state", fit)
        self.assertIn("many-varying-inner", fit)

    def test_strict_batch_and_global_rho_caps_are_distinguished(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["strict_r159_batch_cap_exponent_B"]["exact"], "5/4")
        self.assertEqual(cost["global_pollard_rho_exponent_B"]["exact"], "5/2")
        self.assertEqual(
            cost["target_dependent_soft_linear_n_pass_exponent_B"]["exact"],
            "9/4",
        )
        self.assertFalse(cost["soft_linear_n_pass_meets_strict_batch_cap"])
        self.assertTrue(cost["soft_linear_n_pass_is_below_pollard_rho"])

    def test_monomial_cost_fit_conditions_are_exact(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["strict_batch_fit_condition"], "9 alpha + 5 beta <= 5")
        self.assertEqual(cost["global_below_rho_fit_condition"], "9 alpha + 5 beta < 10")
        self.assertTrue(cost["one_soft_linear_n_pass_global_fit_pair"]["fits_below_rho"])
        self.assertFalse(cost["independent_near_linear_pair"]["fits_below_rho"])

    def test_nonlinear_composition_and_source_layer_remain_open(self) -> None:
        cost = self.report["cost"]
        self.assertTrue(cost["batch_denominator_scalar_evaluation_supplied"])
        self.assertTrue(cost["batch_fixed_linear_functional_access_supplied"])
        self.assertFalse(cost["batch_nonlinear_composition_supplied"])
        self.assertFalse(cost["batch_source_gcd_and_backpointer_supplied"])
        self.assertIn("aggregate nonlinear", self.report["next_action"])

    def test_candidate_oracles_are_not_consumed(self) -> None:
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        for row in self.rows:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_finite_checks_receive_no_attack_credit(self) -> None:
        self.assertFalse(self.controls["finite_controls_receive_attack_credit"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(self.report["cost"]["finite_checks_receive_attack_credit"])

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        admission = self.report["admission"]
        obligations = admission["obligations"]
        self.assertTrue(admission["batch_inverse_linear_layer_admitted"])
        self.assertTrue(admission["global_below_rho_n_pass_window_admitted"])
        self.assertFalse(admission["target_batched_nonlinear_layer_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(obligations["unconditional_total_attack_cost_complete"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
