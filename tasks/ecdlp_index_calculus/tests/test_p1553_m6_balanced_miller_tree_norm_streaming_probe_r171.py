import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PRODUCER_PATH = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_r171.py"
REPORT_PATH = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_report_r171.json"


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r171_test", PRODUCER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load R171 producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R171 = load_module()


class BalancedMillerTreeNormStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT_PATH.read_text())
        cls.bundle = R171.build_bundle()

    def test_source_bindings_are_exact(self) -> None:
        actual = R171.verify_source_bindings()
        self.assertEqual(len(actual), 14)
        for name, _, expected in R171.SOURCE_BINDINGS:
            self.assertEqual(actual[name], expected)

    def test_schema_and_control_count(self) -> None:
        self.assertEqual(self.report["schema"], R171.SCHEMA)
        self.assertEqual(self.report["controls"]["control_count"], 6)
        self.assertEqual(self.report["controls"]["family_count"], 3)

    def test_balanced_trees_have_exact_size_and_depth(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_balanced_trees_close_at_infinity"])
        self.assertEqual(controls["total_line_merge_count"], 80)
        self.assertEqual(controls["maximum_tree_depth"], 4)
        for row in controls["controls"]:
            size = row["zero_list_size"]
            self.assertEqual(row["numerator_tree"]["merge_count"], size - 1)
            self.assertEqual(row["denominator_tree"]["merge_count"], size - 1)
            self.assertIsNone(row["numerator_tree"]["sum_point"])
            self.assertIsNone(row["denominator_tree"]["sum_point"])

    def test_trees_equal_dense_witnesses_up_to_scalar(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_dense_tree_ratios_constant"])
        self.assertEqual(controls["dense_tree_comparison_count"], 192)
        for row in controls["controls"]:
            self.assertTrue(row["all_dense_tree_ratios_constant"])
            self.assertNotEqual(row["numerator_dense_tree_scalar_ratio"], 0)
            self.assertNotEqual(row["denominator_dense_tree_scalar_ratio"], 0)

    def test_tree_quotients_replay_r167_candidate_roots(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_tree_corrected_norm_replays_exact"])
        self.assertTrue(controls["all_candidate_roots_match_r167"])
        self.assertEqual(controls["candidate_root_count"], 140)
        for row in controls["controls"]:
            self.assertEqual(row["candidate_roots"], row["r167_candidate_roots"])

    def test_line_reciprocity_and_specialization_boundary(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_admissible_line_reciprocity_rows_exact"])
        self.assertEqual(controls["line_reciprocity_admissible_row_count"], 1222)
        self.assertEqual(
            controls["line_reciprocity_skipped_pole_or_nonunit_row_count"], 58
        )
        self.assertTrue(
            controls["all_selected_endpoint_linewise_origin_factors_zero"]
        )
        self.assertIn("cancelled symbolically", self.report["theorem"]["specialization_boundary"])

    def test_signed_telescoping_leaves_exactly_targets(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_residual_leaf_factors_equal_targets"])
        self.assertEqual(controls["residual_leaf_factor_count"], 40)
        for row in controls["controls"]:
            self.assertEqual(
                row["residual_leaf_factor_count"], row["retained_target_count"]
            )
            self.assertFalse(row["balanced_depth_reduces_total_leaf_factor_count"])
            self.assertTrue(row["residual_leaf_factors_equal_targets"])

    def test_cost_boundary(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["node_local_line_norm_work_exponent_B"]["exact"], "7/2")
        self.assertEqual(cost["telescoped_leaf_translate_work_exponent_B"]["exact"], "7/2")
        self.assertEqual(cost["raw_signed_grid_tree_expansion_exponent_B"]["exact"], "23/4")
        self.assertEqual(cost["live_endpoint_value_vector_exponent_B"]["exact"], "9/4")
        self.assertFalse(cost["node_local_line_norm_work_inside_rho"])
        self.assertFalse(cost["nonlocal_batched_leaf_translate_operator_supplied"])

    def test_literature_input_contracts_are_not_promoted(self) -> None:
        literature = self.report["literature"]
        self.assertIn("represented bivariate", literature["moroz_schost_truncated_resultant"]["fit"])
        self.assertIn("coefficient vector", literature["bhargava_et_al_multipoint"]["fit"])
        self.assertIn("does not", literature["bhargava_et_al_multipoint"]["fit"])

    def test_scoped_negative_is_not_a_general_lower_bound(self) -> None:
        cost = self.report["cost"]
        self.assertFalse(cost["finite_tree_cancellation_receives_lower_bound_credit"])
        self.assertFalse(cost["arithmetic_circuit_lower_bound_claimed"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])

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
        admission = self.report["admission"]
        self.assertTrue(admission["balanced_generalized_miller_tree_admitted"])
        self.assertTrue(admission["node_local_norm_homomorphism_admitted"])
        self.assertFalse(admission["node_local_slp_streaming_below_rho_admitted"])
        self.assertFalse(admission["nonlocal_batched_leaf_translate_operator_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_next_action_requires_nonlocal_fusion(self) -> None:
        action = self.report["next_action"]
        self.assertIn("nonlocal batched", action)
        self.assertIn("n-by-N pair grid", action)
        self.assertIn("uncharged norm/resultant/multipoint oracle", action)

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R171.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle), {"report", "frozen", "cost", "replay", "controls", "slp"}
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
