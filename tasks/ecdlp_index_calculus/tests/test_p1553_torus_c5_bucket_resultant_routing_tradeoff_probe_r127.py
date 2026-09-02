from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_bucket_resultant_routing_tradeoff_probe_r127.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r127", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R127 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R127)


class TorusC5BucketResultantRoutingTradeoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R127.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R127.verify_source_bindings()), 13)

    def test_all_twenty_four_actual_controls_complete(self) -> None:
        self.assertEqual(self.controls["control_count"], 24)
        self.assertFalse(self.controls["candidate_discrete_logs_consumed"])
        for row in self.controls["controls"]:
            self.assertEqual(row["empty_query_count"], 1)
            self.assertGreater(row["positive_query_count"], 0)

    def test_all_pair_bucket_resultants_are_exact(self) -> None:
        self.assertTrue(
            self.controls["all_pair_resultant_membership_exact"]
        )
        for row in self.controls["controls"]:
            self.assertTrue(row["all_pair_resultant_membership_exact"])
            self.assertGreater(row["all_pair_optimistic_degree_work"], 0)

    def test_target_bucket_routing_is_exact(self) -> None:
        self.assertTrue(
            self.controls["all_target_bucket_routed_membership_exact"]
        )
        for row in self.controls["controls"]:
            self.assertTrue(
                row["target_bucket_routed_resultant_membership_exact"]
            )
            self.assertGreater(
                row["target_bucket_routing_tensor_edge_count"],
                0,
            )

    def test_routed_positive_sources_replay(self) -> None:
        self.assertTrue(
            self.controls["all_positive_routed_sources_replay"]
        )
        for row in self.controls["controls"]:
            self.assertTrue(row["all_positive_routed_sources_replay"])

    def test_symbolic_bucket_outputs_recover_full_split_degree(self) -> None:
        self.assertTrue(
            self.controls["all_symbolic_degrees_equal_c2_times_c3"]
        )
        for row in self.controls["controls"]:
            self.assertTrue(
                row["symbolic_output_degree_equals_c2_times_c3"]
            )
            self.assertGreater(
                row["symbolic_bucket_resultant_total_output_degree"],
                0,
            )

    def test_bucket_and_routing_tradeoff_formulas(self) -> None:
        tradeoff = self.report["tradeoff"]
        self.assertEqual(
            tradeoff["all_H2_pairs_query_work"],
            "B^(9/4+gamma)",
        )
        self.assertEqual(
            tradeoff["R_pair_query_work"],
            "B^(rho+9/4-gamma)",
        )
        self.assertEqual(
            tradeoff["polylog_query_necessary_inequality"],
            "rho<=gamma-9/4",
        )
        self.assertEqual(
            tradeoff["quotient_or_latin_slice_R_equals_H"][
                "query_work"
            ],
            "B^(9/4)",
        )

    def test_standard_bucket_resultant_routes_miss_caps(self) -> None:
        all_pairs = self.routes["all_bucket_pair_resultants"]
        quotient = self.routes[
            "quotient_style_H_routed_pair_resultants"
        ]
        symbolic = self.routes["represented_symbolic_bucket_resultants"]
        self.assertFalse(all_pairs["inside_polylog_query_cap"])
        self.assertFalse(quotient["inside_polylog_query_cap"])
        self.assertEqual(
            symbolic["setup_output_exponent_B"]["exact"],
            "15/4",
        )
        self.assertFalse(symbolic["inside_setup_cap"])

    def test_extreme_singleton_c3_router_remains_open(self) -> None:
        route = self.routes["extreme_implicit_singleton_C3_router"]
        self.assertEqual(route["bucket_exponent_gamma"]["exact"], "9/4")
        self.assertEqual(
            route["routed_pair_exponent_rho"]["exact"],
            "0",
        )
        self.assertTrue(
            route["inside_setup_and_query_caps_if_constructed"]
        )
        self.assertFalse(route["exact_router_constructed"])
        self.assertEqual(route["status"], "open")
        self.assertIn(
            "C3 singleton buckets",
            self.frozen["preserved_interface"],
        )

    def test_scoped_tradeoff_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 12)
        self.assertEqual(admission["obligation_count"], 20)
        self.assertTrue(
            admission["bucket_resultant_routing_semantics_admitted"]
        )
        self.assertTrue(
            admission["scoped_resultant_tradeoff_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "C3 singleton-bucket index",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
