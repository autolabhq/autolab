from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_aggregate_union_factor_label_recovery_probe_r163.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r163", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R163 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R163)


class M6AggregateUnionFactorLabelRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R163.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R163.verify_source_bindings()
        self.assertEqual(len(actual), 12)
        self.assertEqual(
            actual["r88_report"],
            "d73e017c731a54c6913aeaa94e6b5c6d54ca3757f56be14e8ca8ee5524031de1",
        )

    def test_prior_localizer_and_divisor_lanes_are_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("zero oracle", dedup["r88"])
        self.assertIn("per-target signed membership", dedup["r161"])
        self.assertIn("removes target labeling", dedup["r162"])

    def test_preregistered_control_grid_is_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)

    def test_c3_divisor_degree_is_exact(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["c3_divisor_degree"],
                math.comb(row["factor_base_dimension"] + 2, 3),
            )

    def test_aggregate_union_factor_is_exact(self) -> None:
        self.assertEqual(self.controls["exact_union_factor_control_count"], 6)
        for row in self.rows:
            self.assertTrue(row["aggregate_union_factor_exact"])
            self.assertEqual(
                row["union_factor_sha256"], row["expected_union_sha256"]
            )

    def test_union_degree_bound_holds(self) -> None:
        for row in self.rows:
            self.assertLessEqual(
                row["union_factor_degree"], row["union_degree_bound"]
            )
            self.assertTrue(row["union_degree_bound_holds"])

    def test_target_labels_and_sources_are_recovered(self) -> None:
        self.assertEqual(self.controls["exact_source_recovery_control_count"], 6)
        for row in self.rows:
            self.assertTrue(row["all_expected_sources_recovered"])
            self.assertTrue(row["all_label_source_identities_exact"])

    def test_empty_targets_are_rejected(self) -> None:
        self.assertEqual(self.controls["exact_empty_target_control_count"], 6)
        self.assertTrue(all(row["empty_target_rejected"] for row in self.rows))

    def test_inherited_exceptional_semantics_are_preserved(self) -> None:
        self.assertEqual(self.controls["exact_exceptional_control_count"], 6)
        self.assertTrue(
            all(row["exceptional_matches_preserved"] for row in self.rows)
        )

    def test_positive_exceptional_match_is_exercised(self) -> None:
        control = self.controls["synthetic_positive_exceptional_control"]
        self.assertTrue(control["target_x_equals_left_x"])
        self.assertGreaterEqual(control["positive_exceptional_match_count"], 1)
        self.assertTrue(control["all_point_identities_exact"])
        self.assertTrue(control["union_factor_exact"])

    def test_cross_target_uncoupled_product_is_rejected(self) -> None:
        theorem = self.report["theorem"]
        self.assertIn("different targets", theorem["cross_target_false_positive_guard"])
        self.assertFalse(self.report["cost"]["cross_target_uncoupled_product_valid"])

    def test_post_union_label_scan_is_below_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["target_label_scan_exponent_B"]["exact"], "2")
        self.assertEqual(cost["global_pollard_rho_exponent_B"]["exact"], "5/2")
        self.assertEqual(cost["post_union_rho_gap_exponent_B"]["exact"], "1/2")
        self.assertTrue(cost["post_union_target_labels_and_backpointers_inside_rho"])

    def test_aggregate_union_constructor_remains_open(self) -> None:
        cost = self.report["cost"]
        admission = self.report["admission"]
        self.assertFalse(cost["aggregate_union_factor_algorithm_supplied"])
        self.assertFalse(admission["aggregate_union_constructor_admitted"])
        self.assertIn("aggregate union factor itself", self.report["next_action"])

    def test_candidate_oracles_are_not_consumed(self) -> None:
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        for row in self.rows:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_finite_enumeration_receives_no_attack_credit(self) -> None:
        self.assertFalse(self.controls["finite_controls_receive_attack_credit"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(
            self.report["cost"]["finite_union_factor_enumeration_receives_attack_credit"]
        )

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        admission = self.report["admission"]
        obligations = admission["obligations"]
        self.assertTrue(admission["aggregate_union_semantics_admitted"])
        self.assertTrue(admission["post_union_label_recovery_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(obligations["aggregate_union_factor_below_rho_constructed"])
        self.assertFalse(obligations["unconditional_total_attack_cost_complete"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
