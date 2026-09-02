from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_random_diagonal_known_target_rank_probe_r159.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r159", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R159 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R159)


class M6RandomDiagonalKnownTargetRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R159.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R159.verify_source_bindings()
        self.assertEqual(len(actual), 10)
        self.assertEqual(
            actual["r158_parent"],
            "6a88c2a881948d474248f15e91f826109bf97c46fc7c2cfe35f044f300530f89",
        )

    def test_preregistered_public_curve_grid_is_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["offsets"], [0, 1])
        self.assertEqual(self.controls["seeds"], [15901, 15902])
        self.assertEqual(self.controls["control_count"], 12)

    def test_positive_c6_source_count_formula_is_exact(self) -> None:
        for row in self.rows:
            dimension = row["factor_base_dimension"]
            self.assertEqual(
                row["positive_c6_source_count"],
                math.comb(dimension + 5, 6),
            )
            self.assertTrue(
                row["positive_c6_source_count_formula_exact"]
            )

    def test_finite_positive_c6_maps_are_injective(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["positive_c6_unique_endpoint_count"],
                row["positive_c6_source_count"],
            )
            self.assertEqual(row["positive_c6_collision_pair_count"], 0)
            self.assertEqual(row["positive_c6_bad_source_count"], 0)

    def test_query_caps_match_the_frozen_formula(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["known_target_query_cap_per_column"],
                R159.target_query_cap(
                    row["subgroup_order"],
                    row["positive_c6_source_count"],
                    row["factor_base_dimension"],
                ),
            )
            self.assertEqual(row["query_constant"], 8)

    def test_every_factor_base_column_is_covered(self) -> None:
        self.assertEqual(
            self.controls["all_columns_covered_control_count"], 12
        )
        for row in self.rows:
            self.assertEqual(row["uncovered_column_count"], 0)
            self.assertEqual(
                row["covered_column_count"],
                row["factor_base_dimension"],
            )

    def test_public_relation_identities_are_exact(self) -> None:
        self.assertTrue(
            self.controls["all_public_relation_identities_exact"]
        )
        for row in self.rows:
            self.assertTrue(row["all_public_relation_identities_exact"])
            self.assertTrue(
                all(
                    relation["public_relation_identity_exact"]
                    for relation in row["relation_rows"]
                )
            )

    def test_random_diagonal_relation_matrices_are_full_rank(self) -> None:
        self.assertEqual(self.controls["full_rank_control_count"], 12)
        for row in self.rows:
            self.assertTrue(row["full_rank"])
            self.assertEqual(
                row["relation_rank_mod_subgroup_order"],
                row["factor_base_dimension"],
            )

    def test_every_recovered_factor_log_is_publicly_verified(self) -> None:
        self.assertEqual(
            self.controls[
                "publicly_verified_factor_log_control_count"
            ],
            12,
        )
        for row in self.rows:
            self.assertTrue(row["factor_logs_publicly_verified"])
            self.assertEqual(
                len(row["recovered_factor_logs"]),
                row["factor_base_dimension"],
            )

    def test_identical_positive_c6_descent_succeeds(self) -> None:
        self.assertEqual(
            self.controls["successful_identical_descent_control_count"],
            12,
        )
        for row in self.rows:
            descent = row["identical_positive_c6_target_descent"]
            self.assertTrue(descent["success"])
            self.assertTrue(descent["public_scalar_verification"])
            self.assertTrue(descent["candidate_equals_verifier_secret"])
            self.assertTrue(
                descent["verifier_secret_not_used_by_candidate"]
            )

    def test_candidate_oracles_are_not_consumed(self) -> None:
        self.assertFalse(
            self.report["candidate_discrete_log_oracle_consumed"]
        )
        for row in self.rows:
            self.assertFalse(
                row["candidate_discrete_log_oracle_consumed"]
            )
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_coverage_and_rank_theorem_is_admitted(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(
            admission["zero_coverage_and_full_rank_theorem_admitted"]
        )
        self.assertTrue(
            admission["ideal_conditioned_sampler_transfer_admitted"]
        )
        self.assertTrue(admission["identical_descent_reduction_admitted"])

    def test_deterministic_hash_transfer_remains_open(self) -> None:
        obligations = self.report["admission"]["obligations"]
        self.assertFalse(
            obligations["deterministic_hash_to_curve_transfer_complete"]
        )
        self.assertIn(
            "Deterministic hash-to-curve pseudorandomness",
            self.report["theorem"]["scope"],
        )

    def test_costs_charge_targets_and_explicit_endpoints(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["all_column_known_target_count_exponent_B"]["exact"],
            "5/4",
        )
        self.assertEqual(
            cost["explicit_positive_c6_enumeration_exponent_B"]["exact"],
            "9/2",
        )
        self.assertTrue(cost["explicit_control_exceeds_pollard_rho"])
        self.assertFalse(
            cost["finite_explicit_controls_receive_attack_credit"]
        )

    def test_batched_reverse_ffe_locator_remains_open(self) -> None:
        admission = self.report["admission"]
        self.assertFalse(
            admission[
                "batched_positive_c6_unique_source_locator_admitted"
            ]
        )
        self.assertFalse(
            admission["reverse_only_signed_marker_operator_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        obligations = self.report["admission"]["obligations"]
        self.assertFalse(obligations["generic_prime_family_algorithm"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
