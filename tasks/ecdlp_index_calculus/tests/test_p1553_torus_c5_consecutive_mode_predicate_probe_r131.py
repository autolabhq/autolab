from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_torus_c5_consecutive_mode_predicate_probe_r131.py"
SPEC = importlib.util.spec_from_file_location("p1553_r131", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R131 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R131)


class TorusC5ConsecutiveModePredicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R131.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R131.verify_source_bindings()), 10)

    def test_color_acceptance_count_formula(self) -> None:
        for deck_size in range(1, 8):
            sources = tuple(
                __import__("itertools").combinations_with_replacement(
                    range(deck_size),
                    5,
                )
            )
            for color, part in enumerate(
                R131.R129.balanced_four_parts(deck_size)
            ):
                observed = sum(
                    R131.source_color_multiplicity(source, color) >= 2
                    for source in sources
                )
                self.assertEqual(
                    observed,
                    R131.color_acceptance_count(deck_size, len(part)),
                )

    def test_all_actual_controls_are_order_two_and_injective(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertTrue(
            self.controls["all_fields_have_order_two_characteristic"]
        )
        self.assertTrue(self.controls["all_supports_injective"])

    def test_actual_vandermonde_determinants_are_nonzero(self) -> None:
        self.assertGreater(
            self.controls["active_color_control_count"],
            0,
        )
        self.assertTrue(
            self.controls["all_vandermonde_determinants_nonzero"]
        )
        self.assertTrue(self.controls["all_color_targets_distinct"])
        for control in self.controls["controls"]:
            for color in control["color_controls"]:
                self.assertEqual(
                    color["vandermonde_size"],
                    color["accepted_target_count"],
                )
                self.assertTrue(
                    color["vandermonde_determinant_nonzero"]
                )

    def test_dense_annihilators_have_exact_color_zero_sets(self) -> None:
        self.assertTrue(
            self.controls["all_dense_annihilator_zero_sets_exact"]
        )
        self.assertTrue(self.controls["all_zero_targets_rejected"])
        for control in self.controls["controls"]:
            for color in control["color_controls"]:
                self.assertEqual(
                    color[
                        "dense_annihilator_serialized_coefficient_count"
                    ],
                    color["accepted_target_count"] + 1,
                )
                self.assertEqual(
                    color["annihilator_degree"],
                    color["accepted_target_count"],
                )
                self.assertTrue(color["annihilator_zero_set_exact"])

    def test_pigeonhole_c2_c3_sources_replay(self) -> None:
        self.assertTrue(
            self.controls["all_selected_pairs_are_optimal_branches"]
        )
        self.assertTrue(
            self.controls["all_c2_c3_source_products_replay"]
        )

    def test_dense_consecutive_route_is_over_both_caps(self) -> None:
        route = self.routes[
            "dense_consecutive_mode_color_zero_predicate"
        ]
        self.assertEqual(
            route["minimum_serialized_mode_exponent_B"]["exact"],
            "15/4",
        )
        self.assertEqual(
            route["sequential_evaluation_exponent_B"]["exact"],
            "15/4",
        )
        self.assertFalse(route["inside_setup_cap"])
        self.assertFalse(route["inside_polylog_query_cap"])

    def test_root_product_tree_is_fully_charged(self) -> None:
        route = self.routes["explicit_color_root_product_tree"]
        self.assertEqual(route["stored_root_exponent_B"]["exact"], "15/4")
        self.assertEqual(
            route["single_target_leaf_evaluation_exponent_B"]["exact"],
            "15/4",
        )
        self.assertEqual(route["status"], "rejected")

    def test_lacunary_and_nonfourier_routes_remain_open(self) -> None:
        lacunary = self.routes[
            "lacunary_order_two_finite_field_predicate"
        ]
        nonfourier = self.routes[
            "nonfourier_shared_predicate_decision_dag"
        ]
        self.assertFalse(
            lacunary["field_specific_sparse_zero_theorem_supplied"]
        )
        self.assertEqual(lacunary["status"], "open")
        self.assertFalse(nonfourier["exact_dag_constructed"])
        self.assertEqual(nonfourier["status"], "open")
        self.assertIn(
            "lacunary",
            self.frozen["preserved_interface"],
        )

    def test_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 12)
        self.assertEqual(admission["obligation_count"], 20)
        self.assertTrue(
            admission["actual_field_consecutive_mode_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("lacunary", self.report["next_action"])


if __name__ == "__main__":
    unittest.main()
