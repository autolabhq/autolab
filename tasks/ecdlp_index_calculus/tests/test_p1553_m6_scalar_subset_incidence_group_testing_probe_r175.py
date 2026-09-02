import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_scalar_subset_incidence_group_testing_probe_r175.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R175 = load_module("p1553_r175_test", PRODUCER)


class ScalarSubsetIncidenceGroupTestingProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R175.build_bundle()
        cls.report = cls.bundle["report"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 15)
        self.assertEqual(len(R175.verify_source_bindings()), 15)

    def test_scalar_subset_zero_biconditional_is_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_query_zero_biconditionals_exact"])
        self.assertEqual(controls["query_count"], 358)
        self.assertEqual(controls["zero_query_count"], 316)
        self.assertEqual(controls["nonzero_query_count"], 42)
        theorem = self.report["theorem"]["scalar_subset_incidence"]
        self.assertIn("integral domain", theorem)
        self.assertIn("if and only if", theorem)

    def test_all_r174_roots_are_recovered(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_scalar_leaf_roots_match_r174"])
        self.assertTrue(controls["all_balanced_trees_recover_r174_roots"])
        self.assertEqual(controls["candidate_root_count"], 140)
        self.assertEqual(controls["leaf_query_count"], 169)
        for row in controls["controls"]:
            self.assertEqual(
                row["tree"]["recovered_roots"], row["r174_candidate_roots"]
            )

    def test_query_count_bound_holds(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_query_count_bounds_hold"])
        self.assertEqual(controls["query_count_bound"], 1622)
        self.assertLessEqual(controls["query_count"], controls["query_count_bound"])
        theorem = self.report["theorem"]["balanced_zero_product_recovery"]
        self.assertIn("1+2K*ceil(log2(n))", theorem)

    def test_amortized_subset_volume_bound_holds(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_subset_volume_bounds_hold"])
        self.assertEqual(controls["queried_subset_size_sum"], 1238)
        self.assertEqual(controls["queried_subset_size_bound"], 1374)
        self.assertEqual(controls["queried_descriptor_slot_count"], 2834)
        theorem = self.report["theorem"]["amortized_subset_volume"]
        self.assertIn("n*(1+ceil(log2(n)))", theorem)
        self.assertIn("O(n+KN)", theorem)

    def test_subset_descriptors_replay_uv_remainders(self) -> None:
        for control in self.report["controls"]["controls"]:
            root = control["tree"]["queries"][0]
            self.assertEqual(root["path"], "r")
            self.assertEqual(root["descriptor"]["u_degree"], root["subset_size"])
            self.assertEqual(
                root["descriptor"]["u_sha256"], control["root_u_sha256"]
            )
            self.assertEqual(
                root["descriptor"]["v_sha256"], control["root_v_sha256"]
            )

    def test_conditional_cost_envelope_is_below_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["conditional_scalar_oracle_total_exponent_B"]["exact"], "9/4"
        )
        self.assertEqual(cost["global_pollard_rho_exponent_B"]["exact"], "5/2")
        self.assertTrue(
            cost["conditional_scalar_oracle_total_strictly_inside_rho"]
        )
        self.assertEqual(
            cost["conditional_tree_target_overhead_KN_exponent_B"]["exact"],
            "2",
        )

    def test_r163_output_and_postprocessing_costs_are_charged(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["r163_charged_candidate_output_exponent_B"]["exact"], "3/4"
        )
        self.assertEqual(
            cost[
                "r163_target_label_and_backpointer_postprocessing_exponent_B"
            ]["exact"],
            "2",
        )
        self.assertTrue(
            cost["candidate_output_exponent_is_r163_charged_contract_not_finite_fit"]
        )

    def test_standard_routes_are_fully_charged(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["direct_expanded_tree_factor_work_exponent_B"]["exact"],
            "23/4",
        )
        self.assertEqual(
            cost["represented_target_dual_chow_body_exponent_B"]["exact"],
            "5/2",
        )
        self.assertEqual(
            cost["represented_selected_pair_query_exponent_B"]["exact"],
            "9/2",
        )
        self.assertFalse(cost["standard_direct_route_inside_rho"])

    def test_missing_oracle_is_not_admitted(self) -> None:
        cost = self.report["cost"]
        admission = self.report["admission"]
        self.assertFalse(cost["scalar_subset_oracle_supplied"])
        self.assertFalse(cost["unconditional_total_attack_cost_supplied"])
        self.assertTrue(admission["scalar_subset_reduction_admitted"])
        self.assertTrue(admission["conditional_below_rho_envelope_admitted"])
        self.assertFalse(admission["reusable_scalar_subset_oracle_admitted"])
        self.assertFalse(admission["lane_admitted"])

    def test_no_oracle_or_finite_asymptotic_credit_is_consumed(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(
            controls["full_leaf_enumeration_performed_for_finite_controls"]
        )
        self.assertFalse(controls["finite_tree_counts_receive_asymptotic_credit"])
        self.assertFalse(controls["scalar_subset_oracle_supplied"])
        self.assertFalse(controls["candidate_oracle_consumed"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])

    def test_generic_boundary_is_explicit(self) -> None:
        theorem = self.report["theorem"]["generic_boundary"]
        self.assertIn("R160 and Shoup", theorem)
        self.assertIn("coordinate representation", theorem)
        self.assertIn("not supplied", theorem)

    def test_no_breakthrough_flags(self) -> None:
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("ORACLE_UNSUPPLIED", self.report["classification"])

    def test_next_action_preserves_the_reusable_oracle_contract(self) -> None:
        action = self.report["next_action"]
        self.assertIn("reusable scalar subset-incidence oracle", action)
        self.assertIn("U_S,V mod U_S", action)
        self.assertIn("O(|S|+N)", action)
        self.assertIn("tangent-aware", action)
        self.assertIn("Reject leaf enumeration", action)

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R175.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "tree"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
