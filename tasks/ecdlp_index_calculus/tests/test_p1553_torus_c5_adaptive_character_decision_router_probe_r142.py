import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT
    / "p1553_torus_c5_adaptive_character_decision_router_probe_r142.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r142_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R142 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R142)


class TorusC5AdaptiveCharacterDecisionRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R142.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R142.verify_source_bindings()), 10)

    def test_all_actual_supports_are_injective_and_inverse_empty(self):
        self.assertEqual(self.controls["actual_control_count"], 8)
        self.assertTrue(
            self.controls["all_actual_c5_supports_injective"]
        )
        self.assertTrue(
            self.controls["all_actual_positive_inverses_empty"]
        )

    def test_exhaustive_router_counts_are_stable(self):
        self.assertEqual(self.controls["exact_router_control_count"], 3)
        self.assertEqual(
            self.controls["impossible_router_control_count"],
            5,
        )
        leaf_counts = sorted(
            count
            for count in self.controls["minimum_leaf_counts"].values()
            if count is not None
        )
        self.assertEqual(leaf_counts, [33, 360, 645])

    def test_optimal_depths_are_stable(self):
        depths = sorted(
            depth
            for depth in self.controls[
                "minimum_maximum_depths"
            ].values()
            if depth is not None
        )
        self.assertEqual(depths, [3, 5, 5])

    def test_impossible_controls_have_full_signature_obstructions(self):
        counts = self.controls["full_signature_obstruction_counts"]
        self.assertEqual(sum(count > 0 for count in counts.values()), 5)

    def test_every_exact_tree_replays_c2_c3_and_inverse_rejection(self):
        self.assertTrue(
            self.controls["all_exact_router_replays_valid"]
        )
        replay_controls = self.bundle["replay"]["controls"]
        for control in replay_controls:
            replay = control["replay"]
            if replay is None:
                continue
            self.assertTrue(replay["all_rows_valid"])
            for row in replay["rows"]:
                if row["target_kind"] == "positive":
                    self.assertEqual(row["leaf_kind"], "accept")
                    self.assertTrue(row["c2_c3_product_exact"])
                    self.assertTrue(row["reverse_five_source_exact"])
                else:
                    self.assertEqual(row["leaf_kind"], "reject")
                    self.assertIsNone(row["c2_pointer"])

    def test_every_control_fails_the_finite_leaf_cap_comparator(self):
        self.assertTrue(
            self.controls[
                "all_controls_fail_finite_B9_over_4_leaf_cap"
            ]
        )

    def test_dynamic_program_scope_is_explicit(self):
        theorem = self.report["theorem"]
        self.assertIn("every deterministic tree", theorem[
            "finite_grammar_completeness"
        ])
        self.assertIn(
            "not a lower bound",
            theorem["scope"],
        )
        self.assertIn("arbitrary parameters", theorem["scope"])

    def test_random_comparator_has_no_candidate_credit(self):
        comparator = self.controls["random_source_comparator"]
        self.assertEqual(
            comparator["predicted_leaf_exponent_B"],
            "5",
        )
        self.assertFalse(comparator["candidate_credit"])

    def test_no_discrete_log_is_consumed(self):
        self.assertFalse(
            self.controls["candidate_discrete_log_oracle_consumed"]
        )
        self.assertFalse(
            self.bundle["cost"]["candidate_field_dlp_used"]
        )

    def test_gate_does_not_promote_router_or_algorithm(self):
        admission = self.report["admission"]
        self.assertTrue(
            admission["frozen_grammar_negative_admitted"]
        )
        self.assertFalse(admission["nonlinear_source_router_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
