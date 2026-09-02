from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_sparse_projector_prony_locator_probe_r183.py"


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r183_tested", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R183 = load_module()


class SparseProjectorPronyLocatorProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R183.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["controls"]
        cls.cost = cls.report["cost"]

    def test_source_bindings_are_complete(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 16)
        self.assertEqual(
            set(self.report["source_bindings"]),
            {name for name, _, _ in R183.SOURCE_BINDINGS},
        )
        self.assertEqual(
            self.report["classification"], self.report["disposition"]
        )

    def test_r182_replays_and_new_divisor_family_are_complete(self) -> None:
        self.assertEqual(self.controls["control_count"], 13)
        self.assertEqual(self.controls["r182_replay_control_count"], 12)
        self.assertEqual(self.controls["new_divisor_family_control_count"], 1)
        self.assertEqual(self.controls["replay_seeds"], [16001, 16002, 18104, 18206])
        self.assertEqual(self.controls["new_divisor_seed"], 18307)

    def test_nested_projector_trace_equals_target_gcd_degree(self) -> None:
        self.assertTrue(
            self.controls["all_fermat_projector_traces_equal_gcd_degrees"]
        )
        self.assertTrue(
            self.controls[
                "all_gcd_degrees_equal_direct_group_law_multiplicities"
            ]
        )
        self.assertTrue(
            self.controls["all_target_multiplicities_below_characteristic"]
        )
        self.assertTrue(
            self.controls[
                "all_candidate_supports_equal_nonzero_multiplicity_support"
            ]
        )

    def test_weighted_prony_recovery_is_exact(self) -> None:
        self.assertTrue(
            self.controls["all_weighted_hankel_ranks_equal_candidate_degrees"]
        )
        self.assertTrue(
            self.controls[
                "all_berlekamp_massey_orders_equal_candidate_degrees"
            ]
        )
        self.assertTrue(
            self.controls["all_prony_locators_equal_candidate_factors"]
        )
        self.assertTrue(self.controls["all_r182_candidate_factors_replayed"])

    def test_control_counts_are_fully_charged(self) -> None:
        self.assertEqual(self.controls["selected_divisor_degree_sum"], 411)
        self.assertEqual(self.controls["unique_target_count_sum"], 84)
        self.assertEqual(self.controls["candidate_degree_sum"], 256)
        self.assertEqual(self.controls["noncandidate_degree_sum"], 155)
        self.assertEqual(self.controls["target_gcd_multiplicity_sum"], 414)
        self.assertEqual(self.controls["weighted_moment_count_sum"], 512)
        self.assertEqual(
            self.controls["represented_source_target_value_slot_count"], 2997
        )
        self.assertEqual(
            self.controls["total_distinct_target_value_count"], 2839
        )
        self.assertEqual(self.controls["full_target_spectrum_source_count"], 298)
        self.assertEqual(self.controls["source_count"], 411)

    def test_new_divisor_family_has_positive_and_negative_sources(self) -> None:
        row = self.controls["controls"][-1]
        self.assertEqual(row["control_role"], "new_strided_source_divisor")
        self.assertEqual(row["selected_divisor_degree"], 7)
        self.assertEqual(row["target_chart_count"], 2)
        self.assertEqual(row["candidate_degree"], 3)
        self.assertEqual(row["noncandidate_degree"], 4)
        self.assertEqual(row["maximum_target_gcd_multiplicity"], 2)
        self.assertTrue(row["prony_locator_equals_candidate_factor"])
        self.assertEqual(
            row["replay_source"], "independent_group_law_strided_divisor"
        )

    def test_source_blind_scalar_trace_loses_candidate_labels(self) -> None:
        collision = self.controls["source_blind_trace_collision"]
        self.assertTrue(collision["all_source_blind_power_sums_equal"])
        self.assertTrue(collision["candidate_locators_differ"])
        self.assertEqual(
            collision["global_power_sums_sha256"],
            collision["row_swapped_power_sums_sha256"],
        )
        self.assertNotEqual(
            collision["original_candidate_locator"],
            collision["row_swapped_candidate_locator"],
        )

    def test_output_stage_is_below_rho_but_constructor_is_missing(self) -> None:
        for field in (
            "projector_moment_output_strictly_below_rho",
            "prony_locator_recovery_strictly_below_rho",
            "candidate_target_verification_strictly_below_rho",
        ):
            self.assertTrue(self.cost[field])
        for field in (
            "represented_tensor_projector_strictly_below_rho",
            "a_linear_target_krylov_output_strictly_below_rho",
        ):
            self.assertFalse(self.cost[field])
        self.assertFalse(
            self.cost["slp_direct_nested_projector_moment_constructor_supplied"]
        )

    def test_admission_preserves_the_open_constructor(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 19)
        self.assertEqual(admission["obligation_count"], 28)
        self.assertTrue(admission["nested_projector_identity_admitted"])
        self.assertTrue(admission["weighted_prony_output_stage_admitted"])
        self.assertFalse(
            admission["slp_direct_nested_moment_constructor_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])

    def test_no_attack_or_asymptotic_credit(self) -> None:
        self.assertFalse(
            self.controls[
                "finite_pair_or_projector_enumeration_receives_asymptotic_credit"
            ]
        )
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R183.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "applicability"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
