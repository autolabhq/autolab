from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_piecewise_selector_decision_dag_probe_r129.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r129", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R129 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R129)


class TorusC5PiecewiseSelectorDecisionDagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R129.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R129.verify_source_bindings()), 12)

    def test_balanced_four_part_turan_formula(self) -> None:
        for n in range(0, 16):
            parts = R129.balanced_four_parts(n)
            self.assertEqual(sum(map(len, parts)), n)
            self.assertLessEqual(
                max(map(len, parts), default=0)
                - min(map(len, parts), default=0),
                1,
            )
            self.assertEqual(
                R129.minimum_cross_branch_count(n),
                sum(len(part) * (len(part) - 1) // 2 for part in parts),
            )

    def test_finite_turan_minima_match_exhaustion(self) -> None:
        expected = {3: 0, 5: 1, 6: 2, 7: 3}
        for n, count in expected.items():
            self.assertEqual(
                R129.exhaustive_minimum_cross_branch_count(n),
                count,
            )
            self.assertEqual(R129.minimum_cross_branch_count(n), count)

    def test_all_eight_support_controls_are_injective(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertTrue(self.controls["all_supports_injective"])
        self.assertFalse(self.controls["candidate_discrete_logs_consumed"])

    def test_actual_optimal_branch_counts(self) -> None:
        expected = {3: 3, 5: 6, 6: 8, 7: 10}
        self.assertTrue(
            self.controls["all_selected_branch_sets_turan_optimal"]
        )
        self.assertTrue(self.controls["all_source_monomial_covers_exact"])
        for row in self.controls["controls"]:
            self.assertEqual(
                row["selected_branch_count"],
                expected[row["deck_size"]],
            )

    def test_positive_targets_return_exact_sources(self) -> None:
        self.assertTrue(self.controls["all_positive_targets_located"])
        self.assertTrue(self.controls["all_returned_sources_replay"])
        for row in self.controls["controls"]:
            self.assertTrue(row["all_positive_targets_located"])
            self.assertTrue(row["all_returned_c2_c3_sources_replay"])

    def test_empty_targets_force_full_sequential_scan(self) -> None:
        self.assertTrue(
            self.controls["all_empty_targets_rejected_after_full_scan"]
        )
        for row in self.controls["controls"]:
            self.assertTrue(row["empty_target_rejected"])
            self.assertEqual(
                row["empty_target_branch_probes"],
                row["selected_branch_count"],
            )

    def test_sequential_scan_and_explicit_router_miss_caps(self) -> None:
        scan = self.routes[
            "optimal_piecewise_constant_branches_sequential_scan"
        ]
        self.assertEqual(scan["branch_count_exponent_B"]["exact"], "3/2")
        self.assertEqual(scan["query_exponent_B"]["exact"], "3/2")
        self.assertTrue(scan["inside_setup_cap"])
        self.assertFalse(scan["inside_polylog_query_cap"])
        table = self.routes["explicit_target_to_branch_router"]
        self.assertEqual(table["state_exponent_B"]["exact"], "15/4")
        self.assertFalse(table["inside_setup_cap"])

    def test_low_slp_and_shared_predicate_dag_remain_open(self) -> None:
        dag = self.routes["balanced_shared_predicate_decision_dag"]
        slp = self.routes["high_degree_low_slp_rational_selector"]
        self.assertFalse(dag["compact_predicates_constructed"])
        self.assertEqual(dag["status"], "open")
        self.assertEqual(slp["multiplication_lower_bound"], "Omega(log B)")
        self.assertFalse(slp["polylog_query_excluded"])
        self.assertEqual(slp["status"], "open")
        self.assertIn("shared-predicate", self.frozen["preserved_interface"])

    def test_scoped_turan_bound_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 12)
        self.assertEqual(admission["obligation_count"], 20)
        self.assertTrue(admission["piecewise_selector_semantics_admitted"])
        self.assertTrue(
            admission["scoped_turan_branch_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("shared-predicate", self.report["next_action"])


if __name__ == "__main__":
    unittest.main()
