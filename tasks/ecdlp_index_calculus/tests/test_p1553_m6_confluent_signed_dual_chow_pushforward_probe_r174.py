import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_r174.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R174 = load_module("p1553_r174_test", PRODUCER)


class ConfluentSignedDualChowPushforwardProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R174.build_bundle()
        cls.report = cls.bundle["report"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 12)
        self.assertEqual(len(R174.verify_source_bindings()), 12)

    def test_signed_line_biconditional_is_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_signed_line_zero_biconditionals_exact"])
        self.assertEqual(controls["line_factor_identity_count"], 68326)
        self.assertEqual(controls["line_zero_biconditional_count"], 68326)
        self.assertIn("P+Q=T", self.report["theorem"]["signed_collinearity"])

    def test_diagonal_uses_curve_tangent_not_interpolant_derivative(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["tangent_factor_count"], 1486)
        self.assertEqual(controls["tangent_zero_count"], 5)
        self.assertEqual(controls["secant_zero_count"], 236)
        theorem = self.report["theorem"]["confluent_deflation"]
        self.assertIn("interpolant derivative V'(X) is not the curve tangent", theorem)
        self.assertIn("(X-u)(3X^2+a)-2V(X)(V(X)+v)", theorem)

    def test_deflated_norm_and_selected_chow_derivative_are_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_deflated_line_norms_exact"])
        self.assertTrue(controls["all_selected_chow_confluent_derivatives_exact"])
        self.assertEqual(controls["deflated_norm_identity_count"], 1486)
        self.assertEqual(controls["selected_chow_confluent_identity_count"], 1486)
        self.assertIn(
            "partial_gamma(C_S)",
            self.report["theorem"]["selected_dual_chow_derivative"],
        )

    def test_target_dual_chow_pushforward_is_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_target_dual_chow_evaluations_exact"])
        self.assertTrue(controls["all_target_first_and_target_last_aggregates_equal"])
        self.assertEqual(controls["target_chow_identity_count"], 8922)
        theorem = self.report["theorem"]["target_dual_chow"]
        self.assertIn("C_T(-Delta_V,-1,X*Delta_V-V(X))", theorem)
        self.assertIn("two-chart", theorem)

    def test_signed_roots_replay_r166_without_opposite_branch(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_candidate_roots_match_r166_verified_roots"])
        self.assertEqual(controls["candidate_root_count"], 140)
        for row in controls["controls"]:
            self.assertEqual(row["candidate_roots"], row["r166_verified_roots"])
        self.assertIn(
            "without the opposite-sign Kummer branch",
            self.report["theorem"]["candidate_semantics"],
        )

    def test_target_chow_represented_body_is_at_rho(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["target_chow_coefficient_slot_count"], 204)
        self.assertEqual(controls["target_chow_nonzero_coefficient_count"], 204)
        for row in controls["controls"]:
            target_count = row["retained_target_count"]
            self.assertEqual(
                row["target_dual_chow"]["coefficient_slot_count"],
                (target_count + 2) * (target_count + 1) // 2,
            )
        cost = self.report["cost"]
        self.assertEqual(
            cost["represented_target_dual_chow_body_exponent_B"]["exact"],
            "5/2",
        )
        self.assertFalse(cost["represented_target_dual_chow_strictly_inside_rho"])

    def test_selected_chow_represented_body_is_above_rho(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["selected_chow_coefficient_slot_count"], 4770)
        self.assertEqual(controls["selected_chow_nonzero_coefficient_count"], 4770)
        cost = self.report["cost"]
        self.assertEqual(
            cost["represented_selected_dual_chow_body_exponent_B"]["exact"],
            "9/2",
        )
        self.assertFalse(cost["represented_selected_dual_chow_inside_rho"])

    def test_query_grids_and_factored_route_are_fully_charged(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["selected_target_evaluation_grid_exponent_B"]["exact"], "7/2"
        )
        self.assertEqual(
            cost["target_first_selected_pair_grid_exponent_B"]["exact"], "9/2"
        )
        self.assertEqual(
            cost["factored_target_chow_on_selected_pair_grid_exponent_B"]["exact"],
            "23/4",
        )
        self.assertFalse(cost["factored_confluent_outer_norm_supplied"])

    def test_multipoint_contract_does_not_hide_input_body(self) -> None:
        fit = self.report["literature"]["fast_multivariate_multipoint_evaluation"][
            "fit"
        ]
        self.assertIn("Theta(N^2)", fit)
        self.assertIn("does not accept the factored linear forms", fit)

    def test_no_oracle_or_asymptotic_credit_is_consumed(self) -> None:
        controls = self.report["controls"]
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(controls["candidate_oracle_consumed"])
        self.assertFalse(controls["finite_chow_density_receives_asymptotic_credit"])
        self.assertFalse(controls["finite_candidate_count_receives_attack_credit"])

    def test_no_breakthrough_flags(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(admission["confluent_signed_dual_chow_identity_admitted"])
        self.assertFalse(admission["factored_confluent_outer_norm_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_next_action_preserves_only_fused_factored_outer_norm(self) -> None:
        action = self.report["next_action"]
        self.assertIn("fused factored dual-Chow outer norm", action)
        self.assertIn("Delta_V", action)
        self.assertIn("tangent diagonal", action)
        self.assertIn("below B^(5/2)", action)
        self.assertIn("N^2 or n^2", action)

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R174.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "chow"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
