import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_principal_target_pontryagin_resultant_probe_r176.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R176 = load_module("p1553_r176_test", PRODUCER)


class PrincipalTargetPontryaginResultantProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R176.build_bundle()
        cls.report = cls.bundle["report"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 24)
        self.assertEqual(len(R176.verify_source_bindings()), 24)

    def test_principal_target_function_is_signed_and_pole_free(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["selected_pair_evaluation_count"], 8922)
        self.assertEqual(controls["selected_pair_denominator_unit_count"], 8922)
        self.assertEqual(controls["principal_pair_incidence_count"], 241)
        theorem = self.report["theorem"]["principal_target_signed_incidence"]
        self.assertIn("h(P+Q)=0 if and only if P+Q", theorem)
        self.assertIn("sign-sensitive", theorem)

    def test_diagonal_uses_group_law_without_interpolant_tangent(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["diagonal_pair_incidence_count"], 5)
        self.assertEqual(controls["off_diagonal_pair_incidence_count"], 236)
        theorem = self.report["theorem"]["principal_target_signed_incidence"]
        self.assertIn("P=Q through the elliptic group law", theorem)
        self.assertIn("rather than an interpolant derivative", theorem)

    def test_principal_roots_replay_r175_and_r167(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_principal_leaf_roots_match_r175_and_r167"])
        self.assertEqual(controls["candidate_root_count"], 140)
        for row in controls["controls"]:
            self.assertEqual(row["principal_leaf_roots"], row["r175_candidate_roots"])
            self.assertEqual(row["principal_leaf_roots"], row["r167_candidate_roots"])

    def test_principal_tree_replays_all_r175_queries(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_principal_tree_zero_patterns_match_r175"])
        self.assertEqual(controls["principal_tree_query_count"], 358)
        for row in controls["controls"]:
            self.assertEqual(
                row["principal_tree_zero_projection_sha256"],
                row["r175_tree_zero_projection_sha256"],
            )

    def test_pontryagin_cycle_degree_and_group_sum_are_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_pair_cycle_group_sum_identities_exact"])
        self.assertEqual(controls["queried_pair_cycle_degree_sum"], 56468)
        theorem = self.report["theorem"]["subset_pontryagin_norm"]
        self.assertIn("degree is m*n", theorem)
        completion = self.report["theorem"]["principal_completion"]
        self.assertIn("n*sum(A)+m*sum(D)", completion)

    def test_principal_completions_and_auxiliary_poles_are_units(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_completion_values_units"])
        self.assertTrue(controls["all_miller_pole_products_units"])
        self.assertLessEqual(controls["maximum_miller_shuffle_attempt"], 1)

    def test_all_completed_weil_reciprocity_identities_are_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(
            controls["all_principal_pontryagin_reciprocity_identities_exact"]
        )
        self.assertEqual(controls["reciprocity_identity_count"], 358)
        self.assertEqual(controls["literal_disjoint_support_identity_count"], 42)
        self.assertEqual(
            controls["candidate_specialized_zero_identity_count"], 316
        )
        theorem = self.report["theorem"]["weil_reciprocity_scalar"]
        self.assertIn("h(div f)=f(div h)", theorem)
        self.assertIn("must not be inverted", theorem)

    def test_standard_miller_route_retains_pair_cycle_size(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["miller_merge_count"], 56468)
        self.assertEqual(controls["charged_miller_probe_evaluation_count"], 868648)
        cost = self.report["cost"]
        self.assertEqual(
            cost["represented_principal_completion_miller_state_exponent_B"][
                "exact"
            ],
            "9/2",
        )
        self.assertFalse(cost["standard_reciprocity_route_inside_rho"])

    def test_root_pair_cycle_and_balanced_volume_are_above_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["root_pontryagin_cycle_degree_exponent_B"]["exact"], "9/2"
        )
        self.assertEqual(
            cost["balanced_queried_pair_cycle_volume_exponent_B"]["exact"],
            "9/2",
        )
        self.assertEqual(cost["global_pollard_rho_exponent_B"]["exact"], "5/2")
        self.assertFalse(cost["standard_represented_pair_cycle_route_inside_rho"])

    def test_direct_and_fast_represented_routes_are_charged(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["direct_target_slp_on_pair_cycle_exponent_B"]["exact"],
            "23/4",
        )
        self.assertEqual(
            cost[
                "represented_fast_evaluation_after_pair_enumeration_exponent_B"
            ]["exact"],
            "9/2",
        )

    def test_conditional_factored_resultant_remains_unsupplied(self) -> None:
        cost = self.report["cost"]
        admission = self.report["admission"]
        self.assertEqual(
            cost["conditional_factored_trilinear_resultant_total_exponent_B"][
                "exact"
            ],
            "9/4",
        )
        self.assertTrue(
            cost["conditional_factored_trilinear_resultant_strictly_inside_rho"]
        )
        self.assertFalse(cost["factored_trilinear_elliptic_resultant_supplied"])
        self.assertFalse(
            admission["factored_trilinear_elliptic_resultant_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])

    def test_represented_degree_is_not_promoted_to_a_lower_bound(self) -> None:
        cost = self.report["cost"]
        self.assertFalse(
            cost["represented_degree_observation_claimed_as_circuit_lower_bound"]
        )
        self.assertIn("does not prove a lower bound", self.report["theorem"]["scope"])

    def test_no_oracle_or_finite_attack_credit_is_consumed(self) -> None:
        controls = self.report["controls"]
        self.assertFalse(
            controls[
                "finite_pair_and_miller_enumeration_receives_asymptotic_credit"
            ]
        )
        self.assertFalse(controls["candidate_oracle_consumed"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])

    def test_no_breakthrough_flags(self) -> None:
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("FACTORED_TRILINEAR_RESULTANT_OPEN", self.report["classification"])

    def test_next_action_is_the_factored_pontryagin_resultant(self) -> None:
        action = self.report["next_action"]
        self.assertIn("factored trilinear elliptic resultant", action)
        self.assertIn("U_A,V_A", action)
        self.assertIn("O(m+N)", action)
        self.assertIn("Reject explicit mn pair sums", action)

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R176.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "resultant"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
