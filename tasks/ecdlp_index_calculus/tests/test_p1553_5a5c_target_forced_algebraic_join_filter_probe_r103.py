from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "p1553_5a5c_target_forced_algebraic_join_filter_probe_r103.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r103_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load R103 probe")
R103 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R103)


class TargetForcedAlgebraicJoinFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R103.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["actual_controls"]
        cls.costs = cls.report["asymptotic_control"]

    def test_actual_true_pairs_survive_exact_s3_filter(self) -> None:
        self.assertEqual(self.controls["instance_count"], 8)
        self.assertEqual(self.controls["target_control_count"], 16)
        self.assertTrue(self.controls["all_true_pairs_survive_s3"])
        self.assertTrue(
            self.controls["all_s3_pairs_equal_translation_x_branches"]
        )
        self.assertTrue(self.controls["all_positive_sources_replay"])
        self.assertTrue(
            self.controls["all_blind_targets_have_no_true_join"]
        )

    def test_regular_s3_quadratic_has_two_translation_roots(self) -> None:
        self.assertTrue(
            self.controls["all_regular_translation_roots_vanish"]
        )
        self.assertTrue(
            self.controls["all_regular_quadratic_factorizations_exact"]
        )

    def test_synthetic_control_exposes_all_sign_branches(self) -> None:
        control = self.controls["synthetic_sign_complete_control"]
        self.assertTrue(control["all_four_signed_points_pass_s3"])
        self.assertTrue(control["exactly_one_signed_point_is_true_join"])
        self.assertTrue(control["two_distinct_x_roots"])
        self.assertTrue(control["root_factorization_exact"])
        self.assertFalse(control["scalar_labels_consumed"])

    def test_pointwise_s3_does_not_reduce_local_oracle_exponent(
        self,
    ) -> None:
        pointwise = self.costs["pointwise_local_oracle_composition"]
        self.assertEqual(
            pointwise["enumerate_left_then_query_right_exponent_B"]["exact"],
            "19/5",
        )
        self.assertEqual(
            pointwise["enumerate_right_then_query_left_exponent_B"]["exact"],
            "16/5",
        )
        self.assertEqual(pointwise["best_exponent_B"]["exact"], "16/5")
        self.assertFalse(pointwise["inside_online_cap"])
        self.assertFalse(
            pointwise["constant_sign_branches_change_exponent"]
        )

    def test_canonical_and_aggregate_s3_bodies_miss_caps(self) -> None:
        canonical = self.costs["canonical_prefix_suffix"]
        aggregate = self.costs["standard_aggregate_representations"]
        self.assertEqual(
            canonical["prefix_state_exponent_B"]["exact"],
            "11/5",
        )
        self.assertEqual(
            canonical["suffix_query_exponent_B"]["exact"],
            "14/5",
        )
        self.assertTrue(canonical["inside_setup_cap"])
        self.assertFalse(canonical["inside_online_cap"])
        self.assertFalse(aggregate["right_body_inside_setup_cap"])
        self.assertFalse(aggregate["left_body_inside_setup_cap"])
        self.assertFalse(aggregate["fresh_suffix_body_inside_online_cap"])

    def test_materialized_ffe_factorization_preserves_total_degree(
        self,
    ) -> None:
        ffe = self.costs["ffe_factorization"]
        self.assertTrue(ffe["endpoint_x_roots_lie_in_base_field"])
        self.assertTrue(ffe["materialized_polynomial_splits_into_linear_factors"])
        self.assertEqual(
            ffe["total_linear_factor_count_exponent_B"]["exact"],
            "12/5",
        )
        self.assertFalse(
            ffe["factorization_reduces_total_degree_or_source_payload"]
        )
        self.assertTrue(ffe["nonstandard_compact_preendpoint_pushdown_open"])

    def test_bundle_admits_identity_but_not_full_join(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 21)
        self.assertEqual(admission["obligation_count"], 32)
        self.assertFalse(admission["lane_admitted"])
        self.assertTrue(self.report["target_forced_s3_identity_admitted"])
        self.assertFalse(self.report["factor_log_solve_complete"])
        self.assertFalse(self.report["fresh_target_descent_complete"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "nonstandard target-specialized S3/FFE pushdown",
            self.report["scope_boundary"],
        )


if __name__ == "__main__":
    unittest.main()
