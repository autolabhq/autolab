from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_kummer_x_translate_signed_verification_probe_r166.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r166", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R166 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R166)


class M6KummerXTranslateSignedVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R166.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R166.verify_source_bindings()
        self.assertEqual(len(actual), 18)
        self.assertEqual(
            actual["r165_producer"],
            "32b21c70e961b92a54d975d7294fb13403106e2c8e82f289e19458798d9d615b",
        )

    def test_neighboring_lanes_are_semantically_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("sets r=0", dedup["r165"])
        self.assertIn("signed forms v-p+q", dedup["r158"])
        self.assertIn("x-only Semaev", dedup["r152"])
        self.assertIn("signed point dictionary", dedup["r161"])
        self.assertIn("static 3SUM", dedup["r148"])

    def test_preregistered_control_grids_are_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)
        self.assertEqual(self.controls["matched_random_control_count"], 6)
        self.assertEqual(
            self.controls["matched_random_target_count_per_control"], 4096
        )

    def test_kummer_function_has_s_plus_negative_s_divisor(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["fixed_kummer_function_pole_order"],
                2 * row["c3_divisor_degree"],
            )
            self.assertEqual(
                row["fixed_kummer_function_zero_divisor_degree"],
                2 * row["c3_divisor_degree"],
            )

    def test_x_only_zero_is_exactly_true_or_opposite_sign(self) -> None:
        self.assertTrue(self.controls["all_kummer_zero_biconditionals_exact"])
        self.assertTrue(self.controls["all_candidate_factors_exact"])
        for row in self.rows:
            self.assertTrue(row["all_kummer_zero_biconditionals_exact"])
            self.assertTrue(
                row["candidate_factor_matches_true_or_opposite_union"]
            )
            self.assertEqual(
                row["candidate_roots"], row["expected_candidate_roots"]
            )

    def test_true_signed_roots_are_never_lost(self) -> None:
        self.assertTrue(self.controls["all_true_roots_retained"])
        self.assertTrue(all(row["all_true_roots_retained"] for row in self.rows))

    def test_signed_verifier_recovers_r163_union(self) -> None:
        self.assertTrue(self.controls["all_signed_verified_unions_exact"])
        for row in self.rows:
            self.assertTrue(
                row["signed_verification_removes_all_opposite_only_roots"]
            )
            self.assertTrue(row["verified_union_matches_r163_exactly"])
            self.assertEqual(
                row["verified_factor_sha256"],
                row["r163_expected_union_factor_sha256"],
            )

    def test_positive_signed_forms_are_nonzero_with_bounded_l1(self) -> None:
        self.assertTrue(
            self.controls[
                "all_positive_coefficient_forms_have_sum_six_and_are_nonzero"
            ]
        )
        for row in self.rows:
            self.assertTrue(
                row[
                    "all_positive_coefficient_forms_have_sum_six_and_are_nonzero"
                ]
            )
            self.assertLessEqual(row["positive_coefficient_form_max_l1"], 12)
            self.assertGreater(row["positive_coefficient_form_count"], 0)

    def test_actual_batches_record_zero_opposite_sign_pairs_without_credit(self) -> None:
        self.assertEqual(self.controls["actual_batch_true_signed_pair_count"], 241)
        self.assertEqual(self.controls["actual_batch_opposite_sign_pair_count"], 0)
        self.assertEqual(
            self.controls["actual_batch_opposite_sign_only_root_count"], 0
        )
        self.assertFalse(self.controls["finite_controls_receive_attack_credit"])

    def test_matched_random_controls_exercise_both_orientations(self) -> None:
        self.assertGreater(self.controls["matched_random_true_pair_count"], 0)
        self.assertGreater(
            self.controls["matched_random_opposite_pair_count"], 0
        )
        for row in self.controls["matched_random_controls"]:
            self.assertEqual(row["target_count"], 4096)
            self.assertTrue(
                row[
                    "deterministic_scalar_sampler_is_not_hash_to_curve_transfer_proof"
                ]
            )
            self.assertFalse(row["finite_control_receives_attack_credit"])

    def test_forced_opposite_sign_branch_is_exercised_and_removed(self) -> None:
        control = self.controls["forced_opposite_sign_control"]
        self.assertTrue(control["translated_is_negative_opposite"])
        self.assertTrue(control["forced_left_appears_in_candidate"])
        self.assertEqual(control["true_signed_pair_count"], 0)
        self.assertGreater(control["opposite_sign_pair_count"], 0)
        self.assertTrue(control["signed_verification_returns_empty_union"])
        self.assertTrue(
            control["candidate_factor_matches_true_or_opposite_union"]
        )

    def test_expected_candidate_and_verification_costs_are_below_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["expected_total_candidate_exponent_B"]["exact"], "3/4"
        )
        self.assertEqual(cost["signed_verification_exponent_B"]["exact"], "2")
        self.assertEqual(cost["global_pollard_rho_exponent_B"]["exact"], "5/2")
        self.assertTrue(cost["expected_signed_verification_inside_rho"])

    def test_constructor_is_simpler_but_still_above_rho_when_explicit(self) -> None:
        cost = self.report["cost"]
        self.assertFalse(cost["global_randomizer_required"])
        self.assertTrue(cost["signed_v_interpolant_required_by_constructor"])
        self.assertTrue(cost["signed_y_side_table_required_by_verifier"])
        self.assertEqual(cost["signed_divisor_v_state_exponent_B"]["exact"], "9/4")
        self.assertEqual(
            cost["explicit_translate_product_divisor_exponent_B"]["exact"],
            "7/2",
        )
        self.assertFalse(cost["standard_explicit_translate_product_inside_rho"])
        self.assertFalse(cost["output_sensitive_kummer_translate_product_supplied"])

    def test_hash_to_curve_transfer_remains_open(self) -> None:
        cost = self.report["cost"]
        obligations = self.report["admission"]["obligations"]
        self.assertFalse(
            cost["iid_label_theorem_receives_hash_to_curve_transfer_credit"]
        )
        self.assertFalse(
            obligations["deterministic_hash_to_curve_transfer_complete"]
        )

    def test_candidate_oracles_are_not_consumed(self) -> None:
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        for row in self.rows:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        admission = self.report["admission"]
        obligations = admission["obligations"]
        self.assertTrue(admission["kummer_x_relaxation_admitted"])
        self.assertTrue(admission["signed_false_branch_bound_admitted_in_iid_model"])
        self.assertFalse(admission["output_sensitive_translate_constructor_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(
            obligations["output_sensitive_kummer_translate_product_complete"]
        )
        self.assertFalse(obligations["unconditional_total_attack_cost_complete"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
