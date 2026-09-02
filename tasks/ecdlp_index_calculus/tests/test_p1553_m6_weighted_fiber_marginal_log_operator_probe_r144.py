import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT
    / "p1553_m6_weighted_fiber_marginal_log_operator_probe_r144.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r144_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R144 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R144)


class M6WeightedFiberMarginalLogOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R144.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R144.verify_source_bindings()), 12)

    def test_all_actual_controls_are_present(self):
        self.assertEqual(self.controls["actual_control_count"], 8)

    def test_weight_derivative_marginal_sums_are_exact(self):
        self.assertTrue(
            self.controls[
                "all_weight_derivative_marginal_sums_exact"
            ]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(control["all_a_marginal_sums_exact"])
            self.assertTrue(control["all_c_marginal_sums_exact"])

    def test_every_known_rhs_fiber_identity_is_exact(self):
        self.assertTrue(
            self.controls["all_known_rhs_fiber_identities_exact"]
        )

    def test_aggregate_rank_dimensions_are_stable(self):
        dimensions = sorted(
            control["meaningful_log_dimension"]
            for control in self.controls["actual_controls"]
        )
        ranks = sorted(
            control["aggregate_marginal_rank"]
            for control in self.controls["actual_controls"]
        )
        self.assertEqual(dimensions, [4, 4, 7, 7, 8, 8, 10, 10])
        self.assertEqual(ranks, dimensions)
        self.assertTrue(
            self.controls[
                "all_aggregate_marginal_matrices_full_meaningful_rank"
            ]
        )

    def test_finite_known_rhs_solves_recover_all_factor_logs(self):
        self.assertTrue(
            self.controls["all_finite_factor_log_solves_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(control["meaningful_factor_logs_recovered"])
            self.assertTrue(control["all_cartesian_factor_logs_replay"])

    def test_positive_shifted_and_empty_descent_semantics_are_exact(self):
        self.assertTrue(
            self.controls[
                "all_positive_and_shifted_target_descents_exact"
            ]
        )
        self.assertTrue(self.controls["all_empty_targets_rejected"])
        self.assertTrue(
            self.controls["all_source_semantic_samples_replay"]
        )

    def test_verifier_dlp_has_no_candidate_credit(self):
        self.assertFalse(
            self.controls["candidate_discrete_log_oracle_consumed"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control[
                    "diagnostic_build_consumes_verifier_scalar_labels"
                ]
            )
            self.assertFalse(
                control["candidate_operator_consumes_scalar_labels"]
            )
            self.assertFalse(
                control["finite_control_receives_asymptotic_credit"]
            )

    def test_conditional_exponent_envelope_is_exact(self):
        envelope = R144.conditional_exponent_envelope()
        self.assertEqual(
            envelope["conditional_precomputation_exponent_B"]["exact"],
            "9/4",
        )
        self.assertEqual(
            envelope["conditional_precomputation_exponent_N"]["exact"],
            "9/20",
        )
        self.assertEqual(
            envelope["conditional_shifted_descent_exponent_B"]["exact"],
            "3/4",
        )
        self.assertEqual(
            envelope["pollard_rho_exponent_N"]["exact"],
            "1/2",
        )
        self.assertEqual(envelope["credit"], "conditional_reduction_only")

    def test_offline_online_derivative_boundary_is_explicit(self):
        weighted = self.report["theorem"]["weighted_fiber_operator"]
        self.assertIn(
            "does not by itself preserve",
            weighted["offline_online_boundary"],
        )
        self.assertIn(
            "without replaying",
            self.report["next_action"],
        )

    def test_count_index_rank_and_generic_transfer_remain_open(self):
        cost = self.bundle["cost"]
        logs = self.bundle["logs"]
        self.assertFalse(cost["weighted_count_circuit_supplied"])
        self.assertFalse(
            cost["offline_online_transposed_derivative_index_supplied"]
        )
        self.assertFalse(
            cost["structured_rank_density_theorem_supplied"]
        )
        self.assertFalse(
            logs["candidate_factor_logs_computed_without_verifier_labels"]
        )
        self.assertFalse(logs["candidate_identical_target_descent_computed"])
        self.assertFalse(logs["generic_prime_family_transfer_supplied"])

    def test_gate_admits_only_the_conditional_reduction(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 19)
        self.assertEqual(admission["obligation_count"], 29)
        self.assertTrue(admission["aggregate_identity_reduction_admitted"])
        self.assertTrue(
            admission["source_free_log_descent_reduction_admitted"]
        )
        self.assertTrue(
            admission["conditional_exponent_envelope_admitted"]
        )
        self.assertFalse(admission["weighted_count_circuit_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
