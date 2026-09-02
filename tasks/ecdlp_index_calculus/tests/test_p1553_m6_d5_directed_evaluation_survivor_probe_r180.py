from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_d5_directed_evaluation_survivor_probe_r180.py"


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r180_tested", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R180 = load_module()


class D5DirectedEvaluationSurvivorProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R180.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["controls"]
        cls.cost = cls.report["cost"]

    def test_source_bindings_are_complete(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 18)
        self.assertEqual(
            set(self.report["source_bindings"]),
            {name for name, _, _ in R180.SOURCE_BINDINGS},
        )

    def test_six_r174_r178_candidate_controls_replay(self) -> None:
        self.assertEqual(self.controls["control_count"], 6)
        self.assertTrue(self.controls["all_candidate_factors_equal_r174_and_r178"])
        self.assertEqual(self.controls["selected_divisor_degree_sum"], 202)
        self.assertEqual(self.controls["target_factor_count_sum"], 40)
        self.assertEqual(self.controls["candidate_degree_sum"], 140)

    def test_target_factor_zero_unions_are_exact(self) -> None:
        self.assertTrue(self.controls["all_target_factor_zero_unions_exact"])
        self.assertEqual(self.controls["target_factor_zero_incidence_count"], 241)
        for row in self.controls["controls"]:
            self.assertEqual(
                row["candidate_roots"],
                sorted(
                    set().union(
                        *(set(factor["zero_roots"]) for factor in row["target_factors"])
                    )
                ),
            )

    def test_optimal_early_split_order_is_exact(self) -> None:
        self.assertTrue(self.controls["all_optimal_orders_exact"])
        self.assertEqual(self.controls["natural_component_factor_visit_count"], 869)
        self.assertEqual(self.controls["optimal_component_factor_visit_count"], 762)
        for row in self.controls["controls"]:
            self.assertLessEqual(
                row["optimal_order"]["component_factor_visit_count"],
                row["natural_order"]["component_factor_visit_count"],
            )

    def test_noncandidate_components_survive_every_factor(self) -> None:
        self.assertTrue(self.controls["all_survivor_lower_bounds_exact"])
        self.assertEqual(self.controls["noncandidate_survivor_degree_sum"], 62)
        self.assertEqual(self.controls["component_factor_visit_lower_bound"], 466)
        for row in self.controls["controls"]:
            self.assertEqual(
                row["component_factor_visit_lower_bound"],
                row["noncandidate_survivor_degree"] * row["target_factor_count"],
            )
            self.assertGreaterEqual(
                row["optimal_order"]["component_factor_visit_count"],
                row["component_factor_visit_lower_bound"],
            )

    def test_d5_splits_only_at_actual_nonunits(self) -> None:
        self.assertTrue(self.controls["all_splits_are_actual_nonunits"])
        self.assertTrue(self.controls["all_final_survivor_factors_are_units"])
        for row in self.controls["controls"]:
            split_steps = [
                step
                for step in row["optimal_order"]["steps"]
                if step["new_candidate_degree"] > 0
            ]
            self.assertEqual(
                len(split_steps), row["optimal_order"]["actual_nonunit_split_count"]
            )
            self.assertTrue(
                all(step["split_is_actual_nonunit"] for step in split_steps)
            )

    def test_finite_materialization_is_fully_charged(self) -> None:
        self.assertEqual(
            self.controls["materialized_target_factor_residue_slot_count"], 1486
        )
        self.assertEqual(self.controls["finite_pair_evaluation_count"], 68326)
        self.assertFalse(
            self.controls["finite_materialization_receives_asymptotic_credit"]
        )

    def test_literal_d5_and_directed_routes_charge_n_times_N(self) -> None:
        self.assertEqual(
            self.cost["literal_factor_stream_lower_bound_exponent_B"]["exact"],
            "7/2",
        )
        self.assertEqual(
            self.cost["successive_d5_zero_test_exponent_B"]["exact"], "7/2"
        )
        self.assertEqual(
            self.cost["directed_evaluation_same_tree_exponent_B"]["exact"],
            "7/2",
        )
        self.assertFalse(self.cost["literal_factor_stream_strictly_below_rho"])
        self.assertFalse(self.cost["directed_evaluation_same_tree_strictly_below_rho"])

    def test_standard_d5_half_gcd_charges_n_squared(self) -> None:
        self.assertEqual(self.cost["standard_d5_half_gcd_exponent_B"]["exact"], "9/2")
        self.assertFalse(self.cost["standard_d5_half_gcd_strictly_below_rho"])

    def test_generic_algebraic_modcomp_does_not_cross_rho(self) -> None:
        self.assertEqual(self.cost["best_cited_generic_algebraic_modcomp_exponent_n"], 1.343)
        self.assertEqual(self.cost["best_cited_generic_algebraic_modcomp_exponent_B"], 3.02175)
        self.assertFalse(self.cost["cited_generic_algebraic_modcomp_strictly_below_rho"])

    def test_one_shot_finite_field_fold_remains_open(self) -> None:
        self.assertFalse(self.cost["finite_field_near_linear_modcomp_fold_applicability_proved"])
        self.assertFalse(self.cost["one_shot_monogenic_compiler_supplied"])
        self.assertIn("one-shot monogenic compiler", self.report["next_action"])

    def test_admission_is_scoped(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 18)
        self.assertEqual(admission["obligation_count"], 28)
        self.assertTrue(admission["d5_directed_interface_scope_admitted"])
        self.assertTrue(admission["literal_d5_factor_stream_route_closed"])
        self.assertTrue(admission["standard_d5_half_gcd_route_closed"])
        self.assertFalse(admission["one_shot_monogenic_compiler_admitted"])
        self.assertFalse(admission["lane_admitted"])

    def test_no_lower_bound_or_attack_credit(self) -> None:
        self.assertIn("not a lower bound", self.report["theorem"]["scope"])
        self.assertFalse(self.cost["standard_route_negative_claimed_as_circuit_lower_bound"])
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])

    def test_no_breakthrough_flags(self) -> None:
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("ONE_SHOT_MONOGENIC_FINITE_FIELD_FOLD_OPEN", self.report["classification"])

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R180.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "applicability"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
