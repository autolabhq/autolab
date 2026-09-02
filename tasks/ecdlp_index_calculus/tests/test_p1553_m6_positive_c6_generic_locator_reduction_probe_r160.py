from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_positive_c6_generic_locator_reduction_probe_r160.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r160", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R160 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R160)


class M6PositiveC6GenericLocatorReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R160.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R160.verify_source_bindings()
        self.assertEqual(len(actual), 19)
        self.assertEqual(
            actual["shoup_generic_lower_bound"],
            "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3",
        )

    def test_prior_pair_sum_and_batch_cost_lanes_are_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("C3+C3 source interface", dedup["r116"])
        self.assertIn("N=B^(5/4)", dedup["r148"])
        self.assertIn("R159", dedup["r160_delta"])

    def test_preregistered_control_grid_is_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)

    def test_c3_occurrence_counts_are_exact(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["c3_occurrence_count"],
                math.comb(row["factor_base_dimension"] + 2, 3),
            )
            self.assertTrue(row["c3_occurrence_count_formula_exact"])

    def test_generic_embedding_coefficients_have_nonzero_b(self) -> None:
        for row in self.rows:
            self.assertTrue(row["all_b_coefficients_nonzero"])
            self.assertTrue(row["verifier_secret_not_used_by_candidate"])

    def test_every_factor_base_column_is_covered(self) -> None:
        self.assertEqual(self.controls["all_columns_covered_control_count"], 6)
        for row in self.rows:
            self.assertEqual(row["uncovered_column_count"], 0)
            self.assertEqual(
                row["covered_column_count"], row["factor_base_dimension"]
            )

    def test_every_relation_identity_is_publicly_exact(self) -> None:
        for row in self.rows:
            self.assertTrue(row["all_public_relation_identities_exact"])
            self.assertTrue(
                all(
                    relation["public_relation_identity_exact"]
                    for relation in row["relation_rows"]
                )
            )

    def test_every_relation_matrix_is_full_rank(self) -> None:
        self.assertEqual(self.controls["full_rank_control_count"], 6)
        for row in self.rows:
            self.assertTrue(row["full_rank"])
            self.assertEqual(
                row["relation_rank_mod_subgroup_order"],
                row["factor_base_dimension"],
            )

    def test_factor_logs_are_publicly_verified(self) -> None:
        self.assertEqual(
            self.controls["publicly_verified_factor_log_control_count"], 6
        )
        for row in self.rows:
            self.assertTrue(row["factor_logs_publicly_verified"])

    def test_embedded_dlps_are_publicly_recovered(self) -> None:
        self.assertEqual(self.controls["publicly_verified_dlp_control_count"], 6)
        for row in self.rows:
            self.assertTrue(row["recovered_dlp_candidates_agree"])
            self.assertTrue(row["candidate_equals_verifier_secret"])
            self.assertTrue(row["public_dlp_verification"])

    def test_identical_positive_c6_descent_also_succeeds(self) -> None:
        self.assertEqual(
            self.controls["successful_identical_descent_control_count"], 6
        )
        for row in self.rows:
            descent = row["identical_positive_c6_target_descent"]
            self.assertTrue(descent["success"])
            self.assertTrue(descent["public_scalar_verification"])
            self.assertTrue(descent["verifier_secret_not_used_by_candidate"])

    def test_candidate_oracles_are_not_consumed(self) -> None:
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        for row in self.rows:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_finite_c3_scan_is_fully_charged(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["standard_c3_scan_full_batch_exponent_B"]["exact"], "7/2"
        )
        self.assertEqual(
            cost["materialized_c3_pair_table_exponent_B"]["exact"], "9/2"
        )
        self.assertFalse(cost["finite_c3_scan_receives_asymptotic_credit"])

    def test_generic_reduction_exponent_is_below_shoup(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["reduced_generic_dlp_exponent_q"]["exact"], "9/20")
        self.assertEqual(
            cost["shoup_generic_lower_bound_exponent_q"]["exact"], "1/2"
        )
        self.assertEqual(cost["contradiction_gap_exponent_q"]["exact"], "1/20")
        self.assertFalse(
            cost["generic_locator_at_requested_caps_compatible_with_shoup"]
        )

    def test_scope_preserves_coordinate_specific_algorithms(self) -> None:
        theorem = self.report["generic_reduction"]
        self.assertIn("opaque generic encodings", theorem["scope"])
        self.assertIn("summation-polynomial", theorem["scope"])
        self.assertFalse(self.report["cost"]["coordinate_specific_locator_excluded"])
        self.assertTrue(
            self.report["admission"]["coordinate_specific_escape_preserved"]
        )

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        admission = self.report["admission"]
        obligations = admission["obligations"]
        self.assertTrue(admission["generic_locator_reduction_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(obligations["generic_prime_coordinate_family_algorithm"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
