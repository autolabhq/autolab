from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_marked_fitting_signed_norm_dedup_probe_r178.py"


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r178_tested", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R178 = load_module()


class MarkedFittingSignedNormDedupProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R178.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["controls"]
        cls.cost = cls.report["cost"]

    def test_source_bindings_are_complete(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 20)
        self.assertEqual(
            set(self.report["source_bindings"]),
            {name for name, _, _ in R178.SOURCE_BINDINGS},
        )

    def test_signed_row_norm_support_is_the_candidate_set(self) -> None:
        self.assertIn("C_h(P)=product_(Q in D) h(P+Q)", self.report["theorem"]["signed_row_norm"])
        self.assertTrue(self.controls["all_row_norm_zero_sets_match_r177"])
        self.assertEqual(self.controls["candidate_root_count"], 140)

    def test_aggregate_interpolations_are_exact_and_fully_dense(self) -> None:
        self.assertTrue(self.controls["all_aggregate_interpolations_exact"])
        self.assertEqual(self.controls["row_norm_count"], 202)
        self.assertEqual(self.controls["aggregate_output_slot_count"], 202)
        self.assertEqual(self.controls["aggregate_nonzero_coefficient_count"], 202)

    def test_candidate_factor_is_identical_to_r177(self) -> None:
        self.assertTrue(self.controls["all_candidate_factors_equal_r177"])
        self.assertEqual(self.controls["candidate_factor_degree_sum"], 140)

    def test_candidate_factor_is_identical_to_r174_signed_norm(self) -> None:
        self.assertTrue(
            self.controls["all_candidate_factors_equal_r174_signed_norm"]
        )
        for row in self.controls["controls"]:
            self.assertTrue(row["candidate_factor_equals_r174_signed_norm"])

    def test_fitting_filtration_reconstructs_the_marker(self) -> None:
        self.assertIn("L(A)=product_(r>=1) G_r(A)", self.report["theorem"]["fitting_filtration"])
        self.assertTrue(
            self.controls["all_threshold_products_equal_r177_markers"]
        )
        self.assertEqual(self.controls["threshold_degree_sum"], 241)
        self.assertEqual(self.controls["incidence_multiplicity_sum"], 241)

    def test_first_fitting_layer_is_the_candidate_factor(self) -> None:
        self.assertTrue(
            self.controls["all_first_thresholds_equal_candidate_factors"]
        )
        self.assertEqual(self.controls["threshold_degree_sums"], [140, 71, 23, 7])

    def test_higher_multiplicity_layers_add_no_candidate_roots(self) -> None:
        self.assertIn("only G_1", self.report["theorem"]["locator_deduplication"])
        self.assertFalse(self.cost["full_marker_is_distinct_ecdlp_primitive"])

    def test_pair_work_is_finite_only(self) -> None:
        self.assertEqual(self.controls["pair_evaluation_count"], 8922)
        self.assertFalse(
            self.controls[
                "finite_pair_scan_and_interpolation_receive_asymptotic_credit"
            ]
        )

    def test_standard_explicit_routes_remain_above_rho(self) -> None:
        self.assertEqual(self.cost["explicit_signed_target_grid_exponent_B"]["exact"], "7/2")
        self.assertEqual(self.cost["r177_pair_algebra_exponent_B"]["exact"], "9/2")
        self.assertEqual(
            self.cost["r177_explicit_marker_interpolation_exponent_B"]["exact"],
            "6",
        )
        self.assertFalse(self.cost["standard_explicit_routes_inside_rho"])

    def test_conditional_unified_interface_fits_below_rho(self) -> None:
        self.assertEqual(
            self.cost["conditional_nonlocal_signed_norm_total_exponent_B"]["exact"],
            "9/4",
        )
        self.assertTrue(
            self.cost["conditional_nonlocal_signed_norm_strictly_inside_rho"]
        )
        self.assertFalse(self.cost["nonlocal_signed_norm_constructor_supplied"])

    def test_mechanism_level_deduplication_is_admitted(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 22)
        self.assertTrue(admission["marked_fitting_filtration_admitted"])
        self.assertTrue(admission["mechanism_level_deduplication_admitted"])
        self.assertFalse(admission["nonlocal_signed_norm_constructor_admitted"])
        self.assertFalse(admission["lane_admitted"])

    def test_no_oracle_or_attack_credit_is_consumed(self) -> None:
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])

    def test_no_breakthrough_flags(self) -> None:
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("NOT_DISTINCT_ALGORITHMIC_LANE", self.report["classification"])

    def test_standard_negative_is_not_a_circuit_lower_bound(self) -> None:
        self.assertFalse(
            self.cost["standard_route_negative_claimed_as_circuit_lower_bound"]
        )
        self.assertIn("not a lower bound", self.report["theorem"]["scope"])

    def test_next_action_is_the_unified_nonlocal_primitive(self) -> None:
        action = self.report["next_action"]
        self.assertIn("unified nonlocal signed elliptic translate-product", action)
        self.assertIn("G_1=gcd(U,C_h)", action)
        self.assertIn("softly O(n+N)", action)
        self.assertIn("Reject nN target grids", action)

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R178.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "equivalence"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
