from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_gcd_equivalent_target_subresultant_probe_r182.py"


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r182_tested", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R182 = load_module()


class GcdEquivalentTargetSubresultantProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R182.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["controls"]
        cls.cost = cls.report["cost"]

    def test_source_bindings_are_complete(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 15)
        self.assertEqual(
            set(self.report["source_bindings"]),
            {name for name, _, _ in R182.SOURCE_BINDINGS},
        )

    def test_replay_and_held_out_controls_are_complete(self) -> None:
        self.assertEqual(self.controls["control_count"], 12)
        self.assertEqual(self.controls["replay_control_count"], 9)
        self.assertEqual(self.controls["held_out_control_count"], 3)
        self.assertEqual(self.controls["replay_seeds"], [16001, 16002, 18104])
        self.assertEqual(self.controls["held_out_seeds"], [18206])

    def test_public_target_deduplication_and_charts_are_exact(self) -> None:
        self.assertEqual(self.controls["raw_retained_target_count_sum"], 80)
        self.assertEqual(self.controls["unique_retained_target_count_sum"], 79)
        self.assertEqual(self.controls["duplicate_target_count"], 1)
        self.assertTrue(self.controls["all_target_charts_x_injective"])
        self.assertTrue(self.controls["all_target_charts_cover_unique_targets"])

    def test_opposite_point_control_forces_two_charts(self) -> None:
        control = self.controls["opposite_point_two_chart_control"]
        self.assertTrue(control["opposite_points_share_x"])
        self.assertTrue(control["opposite_points_have_distinct_y"])
        self.assertEqual(control["raw_target_count"], 3)
        self.assertEqual(control["unique_target_count"], 2)
        self.assertEqual(control["target_chart_count"], 2)
        self.assertTrue(control["all_target_charts_x_injective"])
        self.assertTrue(control["target_charts_cover_unique_targets"])
        self.assertTrue(control["all_remainder_evaluations_exact"])
        self.assertTrue(
            control["all_norm_zero_iff_subresultant_gcd_nontrivial"]
        )
        self.assertTrue(control["candidate_roots_equal_gcd_roots"])

    def test_target_remainders_are_dense_and_full_rank(self) -> None:
        self.assertEqual(self.controls["selected_divisor_degree_sum"], 404)
        self.assertEqual(self.controls["remainder_coefficient_slot_count"], 2962)
        self.assertEqual(
            self.controls["remainder_nonzero_coefficient_count"], 2962
        )
        self.assertTrue(self.controls["all_remainder_bodies_fully_dense"])
        self.assertTrue(
            self.controls["all_remainder_matrices_full_target_rank"]
        )

    def test_remainders_norms_and_gcds_are_exact(self) -> None:
        self.assertTrue(self.controls["all_remainder_evaluations_exact"])
        self.assertTrue(
            self.controls["all_norm_zero_iff_subresultant_gcd_nontrivial"]
        )
        self.assertTrue(self.controls["all_newton_power_sum_norms_exact"])
        self.assertEqual(self.controls["source_target_gcd_degree_sum"], 409)

    def test_unit_inverse_certificates_are_exact_and_charged(self) -> None:
        self.assertTrue(self.controls["all_unit_inverse_certificates_exact"])
        self.assertEqual(self.controls["candidate_source_count"], 253)
        self.assertEqual(self.controls["noncandidate_source_count"], 151)
        self.assertEqual(self.controls["unit_inverse_certificate_count"], 151)
        self.assertEqual(
            self.controls["unit_inverse_certificate_coefficient_slot_count"],
            1130,
        )

    def test_newton_trace_body_is_explicitly_charged(self) -> None:
        self.assertEqual(self.controls["trace_power_projection_count"], 2962)
        self.assertEqual(
            self.controls["trace_output_coefficient_slot_count"], 2962
        )
        self.assertEqual(
            self.cost["direct_newton_trace_body_exponent_B"]["exact"], "7/2"
        )
        self.assertFalse(
            self.cost["direct_newton_trace_body_strictly_below_rho"]
        )

    def test_represented_routes_remain_above_rho(self) -> None:
        for field in (
            "represented_remainder_body_strictly_below_rho",
            "noncandidate_inverse_certificate_body_strictly_below_rho",
            "triangular_norm_strictly_below_rho",
            "standard_target_half_gcd_strictly_below_rho",
            "generic_bivariate_resultant_input_strictly_below_rho",
        ):
            self.assertFalse(self.cost[field])

    def test_candidate_factors_replay_without_an_oracle(self) -> None:
        self.assertTrue(
            self.controls["all_candidate_factors_equal_signed_replay"]
        )
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        sources = [
            row["candidate_replay_source"] for row in self.controls["controls"]
        ]
        self.assertEqual(sources.count("r181_frozen_control"), 9)
        self.assertEqual(sources.count("fresh_r174_held_out_replay"), 3)

    def test_slp_direct_resultant_remains_open(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 24)
        self.assertEqual(admission["obligation_count"], 35)
        self.assertTrue(admission["target_subresultant_identity_admitted"])
        self.assertTrue(admission["represented_standard_routes_closed"])
        self.assertFalse(
            admission["slp_direct_output_sensitive_resultant_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])

    def test_no_general_lower_bound_or_attack_credit(self) -> None:
        self.assertFalse(
            self.cost["finite_density_claimed_as_general_lower_bound"]
        )
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R182.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "applicability"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
