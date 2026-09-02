import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PRODUCER_PATH = ROOT / "p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_r170.py"
REPORT_PATH = ROOT / "p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_report_r170.json"


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r170_test", PRODUCER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load R170 producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R170 = load_module()


class LambdaZeroFittingTargetNormDedupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT_PATH.read_text())
        cls.bundle = R170.build_bundle()

    def test_source_bindings_are_exact(self) -> None:
        actual = R170.verify_source_bindings()
        self.assertEqual(len(actual), 15)
        for name, _, expected in R170.SOURCE_BINDINGS:
            self.assertEqual(actual[name], expected)

    def test_schema_and_control_count(self) -> None:
        self.assertEqual(self.report["schema"], R170.SCHEMA)
        self.assertEqual(self.report["controls"]["control_count"], 6)
        self.assertEqual(self.report["controls"]["family_count"], 3)

    def test_all_coefficient_ring_replays_are_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_coefficient_ring_replays_exact"])
        for row in controls["controls"]:
            self.assertTrue(row["all_target_factor_interpolants_exact"])
            self.assertTrue(row["aggregate_matches_direct_target_norm"])
            self.assertTrue(
                row["corrected_lambda_zero_fitting_matches_direct_norm"]
            )
            self.assertTrue(row["r167_all_reciprocity_identities_exact"])

    def test_candidate_factors_equal_r167(self) -> None:
        for row in self.report["controls"]["controls"]:
            self.assertTrue(row["candidate_factor_matches_r167"])
            self.assertEqual(row["candidate_roots"], row["r167_candidate_roots"])
        self.assertEqual(
            self.report["controls"]["candidate_factor_degree_sum"], 140
        )

    def test_target_factor_inputs_are_full_degree_and_dense(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_target_factor_interpolants_full_degree"])
        self.assertEqual(
            controls["minimum_observed_factor_coefficient_density"], 1.0
        )
        for row in controls["controls"]:
            self.assertTrue(row["all_target_factor_interpolants_full_degree"])
            self.assertEqual(row["minimum_factor_coefficient_density"], 1.0)
            self.assertEqual(row["mean_factor_coefficient_density"], 1.0)

    def test_aggregate_elements_are_full_degree_and_dense(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_aggregate_polynomials_full_degree"])
        self.assertEqual(
            controls["minimum_observed_aggregate_coefficient_density"], 1.0
        )
        for row in controls["controls"]:
            aggregate = row["aggregate_polynomial"]
            self.assertTrue(row["aggregate_is_full_degree"])
            self.assertEqual(aggregate["degree"], row["c3_divisor_degree"] - 1)
            self.assertEqual(aggregate["density"], 1.0)

    def test_represented_slot_count_is_charged(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["represented_factor_slot_count"], 1486)
        for row in controls["controls"]:
            self.assertEqual(
                row["represented_factor_slot_count"],
                row["c3_divisor_degree"] * row["retained_target_count"],
            )

    def test_lambda_zero_identity_is_not_promoted_to_a_constructor(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(admission["lambda_zero_target_norm_equivalence_admitted"])
        self.assertTrue(admission["standard_fraction_free_route_closed_by_cost"])
        self.assertFalse(admission["slp_streaming_norm_constructor_admitted"])
        self.assertFalse(admission["lane_admitted"])

    def test_cost_boundary(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["represented_target_factor_table_exponent_B"]["exact"], "7/2"
        )
        self.assertEqual(
            cost["swapped_fitting_matrix_exponent_B"]["exact"], "9/2"
        )
        self.assertEqual(
            cost["represented_aggregate_element_exponent_B"]["exact"], "9/4"
        )
        self.assertEqual(cost["global_pollard_rho_exponent_B"]["exact"], "5/2")
        self.assertFalse(cost["standard_represented_route_inside_rho"])
        self.assertFalse(cost["aggregate_element_constructor_inside_rho_supplied"])

    def test_elliptic_cauchy_scope_is_narrow(self) -> None:
        fit = self.report["literature"]["elliptic_cauchy_matrices"]["fit"]
        self.assertIn("complex sigma-function", fit)
        self.assertIn("does not give", fit)
        self.assertFalse(
            self.report["cost"][
                "elliptic_cauchy_paper_supplies_finite_field_constructor"
            ]
        )

    def test_finite_density_has_no_lower_bound_credit(self) -> None:
        self.assertFalse(
            self.report["cost"][
                "finite_density_receives_asymptotic_lower_bound_credit"
            ]
        )
        self.assertFalse(
            self.report["finite_controls_receive_asymptotic_credit"]
        )

    def test_no_oracle_is_consumed(self) -> None:
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        for row in self.report["controls"]["controls"]:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_no_breakthrough_flags(self) -> None:
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_next_action_forbids_represented_bodies(self) -> None:
        action = self.report["next_action"]
        self.assertIn("SLP-streaming", action)
        self.assertIn("nN coefficient body", action)
        self.assertIn("n^2 Fitting matrix", action)

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R170.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(set(self.bundle), {
            "report", "frozen", "cost", "replay", "controls", "fitting"
        })
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
