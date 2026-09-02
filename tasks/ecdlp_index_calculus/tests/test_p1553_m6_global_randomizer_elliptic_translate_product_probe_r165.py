from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_global_randomizer_elliptic_translate_product_probe_r165.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r165", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R165 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R165)


class M6GlobalRandomizerEllipticTranslateProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R165.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R165.verify_source_bindings()
        self.assertEqual(len(actual), 15)
        self.assertEqual(
            actual["miller_paper"],
            "39c76c7643278b87b3d8c24b9a07d0b4cbfb561cd13735548990848e0f0bd166",
        )

    def test_prior_product_and_indexing_lanes_are_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("independence is unnecessary", dedup["r164"])
        self.assertIn("fixed translation orbit", dedup["r113"])
        self.assertIn("static 3SUM-indexing", dedup["r148"])
        self.assertIn("scalar-multiple divisor chains", dedup["miller"])

    def test_preregistered_control_grid_is_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)

    def test_fixed_function_has_all_prescribed_c3_zeros(self) -> None:
        self.assertTrue(self.controls["all_fixed_function_prescribed_zeros_exact"])
        for row in self.rows:
            self.assertTrue(row["fixed_function_prescribed_c3_zeros_exact"])
            self.assertEqual(
                row["fixed_function_pole_order"],
                2 * row["c3_divisor_degree"],
            )
            self.assertEqual(
                row["fixed_function_zero_divisor_degree"],
                2 * row["c3_divisor_degree"],
            )

    def test_all_factors_are_translates_of_one_function(self) -> None:
        self.assertTrue(self.controls["all_translation_identities_exact"])
        for row in self.rows:
            self.assertTrue(row["all_translation_identities_exact"])
            self.assertGreater(row["regular_translation_identity_count"], 0)
            self.assertGreater(row["pole_equality_pair_count"], 0)

    def test_global_randomizer_loses_no_true_roots(self) -> None:
        self.assertTrue(self.controls["all_true_roots_retained"])
        self.assertTrue(all(row["all_true_roots_retained"] for row in self.rows))

    def test_direct_verification_recovers_r163_union(self) -> None:
        self.assertTrue(self.controls["all_verified_unions_exact"])
        for row in self.rows:
            self.assertTrue(row["verification_removes_all_false_positives"])
            self.assertTrue(row["verified_union_matches_r163_exactly"])
            self.assertEqual(
                row["verified_factor_sha256"],
                row["r163_expected_union_factor_sha256"],
            )

    def test_forced_global_correlation_is_exercised_and_removed(self) -> None:
        control = self.controls["forced_correlated_false_positive_control"]
        self.assertEqual(control["combined_residual"], 0)
        self.assertTrue(control["forced_root_is_not_true_union_root"])
        self.assertTrue(control["forced_root_appears_as_false_positive"])
        self.assertTrue(control["verification_removes_all_correlated_false_positives"])
        self.assertTrue(control["verified_union_matches_r163_exactly"])

    def test_tangent_is_global_and_only_equality_is_a_pole(self) -> None:
        control = self.controls["tangent_and_pole_control"]
        self.assertTrue(control["tangent_x_incidence"])
        self.assertTrue(control["tangent_left_is_negative_target"])
        self.assertTrue(control["tangent_translate_is_point_q"])
        self.assertTrue(control["tangent_fixed_function_zero"])
        self.assertTrue(control["opposite_translate_is_point_p"])
        self.assertTrue(control["opposite_fixed_function_zero"])
        self.assertTrue(control["pole_translate_is_infinity"])
        self.assertEqual(control["pole_semantic_product_factor"], 1)

    def test_chart_exception_pair_bound_holds(self) -> None:
        for row in self.rows:
            self.assertLessEqual(
                row["pole_equality_pair_count"] + row["tangent_pair_count"],
                row["target_count"],
            )
            self.assertTrue(row["chart_exception_pair_count_at_most_target_count"])

    def test_one_global_randomizer_preserves_probability_bound(self) -> None:
        theorem = self.report["theorem"]
        cost = self.report["cost"]
        self.assertIn("Independence across targets is unnecessary", theorem["one_global_randomizer"])
        self.assertEqual(cost["false_union_probability_exponent_B"]["exact"], "-3/2")
        self.assertFalse(cost["per_target_independent_randomizers_required"])
        self.assertFalse(cost["target_label_randomizer_interpolation_required"])

    def test_standard_translate_product_remains_above_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["standard_translate_product_exponent_B"]["exact"], "7/2")
        self.assertEqual(cost["global_pollard_rho_exponent_B"]["exact"], "5/2")
        self.assertEqual(cost["standard_product_rho_excess_exponent_B"]["exact"], "1")
        self.assertFalse(cost["standard_explicit_translate_product_inside_rho"])
        self.assertFalse(cost["output_sensitive_translate_product_supplied"])

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
        self.assertTrue(admission["global_randomizer_reduction_admitted"])
        self.assertTrue(admission["fixed_function_translate_identity_admitted"])
        self.assertFalse(admission["translate_product_constructor_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(obligations["translate_product_below_rho_constructed"])
        self.assertFalse(obligations["unconditional_total_attack_cost_complete"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
