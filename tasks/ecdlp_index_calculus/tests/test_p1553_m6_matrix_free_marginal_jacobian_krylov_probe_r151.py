import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / (
    "p1553_m6_matrix_free_marginal_jacobian_krylov_probe_r151.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r151_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R151 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R151)


class M6MatrixFreeMarginalJacobianKrylovTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R151.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R151.verify_source_bindings()), 17)

    def test_marginal_matrix_is_frozen_as_log_weight_jacobian(self):
        theorem = self.report["theorem"]
        self.assertIn(
            "M=D_log Z(1)",
            theorem["marginal_jacobian"],
        )
        self.assertIn("Mx", theorem["forward_action"])
        self.assertIn("M^T lambda", theorem["transpose_action"])

    def test_all_actual_jacobian_transpose_identities_are_exact(self):
        self.assertEqual(self.controls["actual_control_count"], 8)
        self.assertTrue(
            self.controls["all_jacobian_transpose_identities_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control["jacobian_transpose_identity_exact"]
            )
            self.assertEqual(
                control["bilinear_pairing_forward"],
                control["bilinear_pairing_reverse"],
            )

    def test_all_selected_linear_systems_replay(self):
        self.assertTrue(
            self.controls["all_selected_linear_system_replays_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control["selected_linear_system_replay_exact"]
            )
            self.assertTrue(control["r144_factor_logs_recovered"])
            self.assertTrue(
                control["r144_factor_logs_and_shifted_descent_replay"]
            )
            self.assertEqual(
                control["matrix_free_recovered_rhs_sha256"],
                control["selected_rhs_sha256"],
            )

    def test_meaningful_dimensions_are_frozen(self):
        self.assertEqual(
            self.controls["meaningful_dimensions"],
            [4, 4, 7, 7, 8, 8, 10, 10],
        )

    def test_baur_strassen_scope_is_exact(self):
        theorem = self.report["theorem"]
        self.assertIn(
            "constant-factor nonscalar overhead",
            theorem["baur_strassen_use"],
        )
        self.assertIn(
            "does not compile a reusable",
            theorem["offline_online_boundary"],
        )
        self.assertIn(
            "10.1016/0304-3975(83)90110-X",
            theorem["primary_references"]["baur_strassen"],
        )

    def test_wiedemann_operator_scope_is_exact(self):
        theorem = self.report["theorem"]
        self.assertIn(
            "O(n) matrix-vector applications",
            theorem["wiedemann_use"],
        )
        self.assertIn(
            "exact residual check",
            theorem["wiedemann_use"],
        )
        self.assertIn(
            "10.1109/TIT.1986.1057137",
            theorem["primary_references"]["wiedemann"],
        )

    def test_conditional_matrix_free_cost_is_b2(self):
        cost = self.bundle["cost"]
        self.assertEqual(
            cost["scalar_marker_batch_exponent_B"]["exact"],
            "5/4",
        )
        self.assertEqual(
            cost["conditional_wiedemann_iteration_exponent_B"]["exact"],
            "3/4",
        )
        self.assertEqual(
            cost["conditional_matrix_free_solve_exponent_B"]["exact"],
            "2",
        )
        self.assertTrue(
            cost["conditional_matrix_free_solve_inside_setup_cap"]
        )
        self.assertTrue(
            cost["conditional_matrix_free_solve_inside_pollard_rho"]
        )

    def test_missing_operator_and_derivative_state_remain_explicit(self):
        cost = self.bundle["cost"]
        self.assertFalse(
            cost["bidirectional_marker_batch_operator_supplied"]
        )
        self.assertFalse(
            cost["reusable_weight_parametric_derivative_state_supplied"]
        )
        self.assertFalse(
            cost["generic_prime_rank_and_density_supplied"]
        )
        self.assertFalse(cost["unconditional_total_attack_cost_supplied"])

    def test_finite_controls_receive_no_candidate_credit(self):
        self.assertFalse(
            self.controls["candidate_discrete_log_oracle_consumed"]
        )
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_credit"]
        )
        for control in self.controls["actual_controls"]:
            self.assertFalse(
                control["candidate_discrete_log_oracle_consumed"]
            )
            self.assertFalse(
                control["verifier_matrix_receives_candidate_credit"]
            )
            self.assertFalse(
                control["finite_control_receives_asymptotic_credit"]
            )

    def test_gate_admits_only_conditional_interface_reduction(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 16)
        self.assertEqual(admission["obligation_count"], 25)
        self.assertTrue(
            admission["matrix_free_jacobian_interface_admitted"]
        )
        self.assertTrue(
            admission["conditional_B2_linear_algebra_envelope_admitted"]
        )
        self.assertFalse(
            admission[
                "bidirectional_weight_parametric_count_circuit_admitted"
            ]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
