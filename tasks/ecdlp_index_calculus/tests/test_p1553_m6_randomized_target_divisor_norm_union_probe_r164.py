from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_randomized_target_divisor_norm_union_probe_r164.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r164", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R164 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R164)


class M6RandomizedTargetDivisorNormUnionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R164.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R164.verify_source_bindings()
        self.assertEqual(len(actual), 13)
        self.assertEqual(
            actual["dahan_dynamic_evaluation"],
            "e17f13261cab77b08313c5524764e2a7b1030dfc83b56afc1a88c954034c667f",
        )

    def test_prior_norm_and_union_lanes_are_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("B^2 label/backpointer", dedup["r163"])
        self.assertIn("B^(13/5)", dedup["r107"])
        self.assertIn("subresultant", dedup["dahan_dynamic_evaluation"])

    def test_preregistered_control_grid_is_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)

    def test_target_label_algebras_are_collision_safe(self) -> None:
        for row in self.rows:
            self.assertTrue(row["target_labels_distinct"])
            self.assertTrue(row["target_labels_fit_base_field"])
            self.assertTrue(row["label_modulus_sha256"])
            self.assertTrue(row["randomizer_interpolant_sha256"])

    def test_no_true_regular_root_is_lost(self) -> None:
        self.assertTrue(self.controls["all_true_regular_roots_retained"])
        self.assertTrue(all(row["true_regular_roots_never_lost"] for row in self.rows))

    def test_direct_verification_recovers_r163_union(self) -> None:
        self.assertTrue(self.controls["all_verified_unions_exact"])
        for row in self.rows:
            self.assertTrue(row["verification_removes_all_false_positives"])
            self.assertTrue(row["verified_union_matches_r163_exactly"])
            self.assertEqual(
                row["verified_union_factor_sha256"],
                row["r163_expected_union_factor_sha256"],
            )

    def test_incidence_pair_bound_holds(self) -> None:
        self.assertTrue(self.controls["all_incidence_pair_bounds_hold"])
        for row in self.rows:
            self.assertLessEqual(row["incidence_pair_count"], row["target_count"])
            self.assertTrue(row["incidence_pair_count_at_most_target_count"])

    def test_forced_false_positive_is_exercised_and_removed(self) -> None:
        control = self.controls["forced_false_positive_control"]
        self.assertEqual(control["combined_residual"], 0)
        self.assertTrue(control["forced_root_is_not_true_union_root"])
        self.assertTrue(control["forced_root_appears_as_randomized_false_positive"])
        self.assertTrue(control["verification_removes_forced_false_positive"])
        self.assertTrue(control["verified_union_matches_r163_exactly"])

    def test_positive_incidence_branch_is_exercised(self) -> None:
        control = self.controls["positive_incidence_control"]
        self.assertTrue(control["target_x_equals_p_x"])
        self.assertEqual(control["incidence_factor_degree"], 1)
        self.assertEqual(control["regular_norm_factor_degree"], 1)
        self.assertTrue(control["incidence_root_is_true_exceptional_match"])
        self.assertTrue(control["opposite_orientation_is_regular_norm_root"])
        self.assertTrue(control["combined_verified_union_is_exact"])

    def test_false_positive_bound_is_strictly_one_sided(self) -> None:
        theorem = self.report["theorem"]
        cost = self.report["cost"]
        self.assertIn("no false negatives", theorem["false_positive_bound"])
        self.assertEqual(cost["false_union_probability_exponent_B"]["exact"], "-3/2")
        self.assertTrue(cost["regular_true_roots_have_zero_false_negative_probability"])
        self.assertTrue(cost["direct_verification_makes_output_exact"])

    def test_post_norm_work_is_below_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["expected_regular_verification_exponent_B"]["exact"], "2")
        self.assertEqual(cost["incidence_factor_and_split_exponent_B"]["exact"], "9/4")
        self.assertEqual(cost["global_pollard_rho_exponent_B"]["exact"], "5/2")
        self.assertTrue(cost["exceptional_incidence_constructor_inside_rho"])

    def test_standard_norm_remains_above_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["standard_coefficient_ring_norm_exponent_B"]["exact"], "7/2")
        self.assertEqual(cost["standard_norm_rho_excess_exponent_B"]["exact"], "1")
        self.assertFalse(cost["standard_represented_norm_inside_rho"])
        self.assertFalse(cost["output_sensitive_elliptic_translation_norm_supplied"])

    def test_candidate_oracles_are_not_consumed(self) -> None:
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        for row in self.rows:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_finite_pair_table_receives_no_attack_credit(self) -> None:
        self.assertFalse(self.controls["finite_controls_receive_attack_credit"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(self.report["cost"]["finite_pair_enumeration_receives_attack_credit"])

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        admission = self.report["admission"]
        obligations = admission["obligations"]
        self.assertTrue(admission["randomized_norm_reduction_admitted"])
        self.assertTrue(admission["incidence_constructor_admitted"])
        self.assertFalse(admission["regular_norm_constructor_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(obligations["regular_target_norm_below_rho_constructed"])
        self.assertFalse(obligations["unconditional_total_attack_cost_complete"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
