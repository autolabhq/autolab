from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_m6_nonlinear_value_sensitive_c6_source_locator_probe_r118.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r118", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R118 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R118)


class M6NonlinearValueSensitiveC6SourceLocatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R118.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.replay = cls.bundle["replay"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R118.verify_source_bindings()), 12)

    def test_one_c_plus_c5_branch_identity_is_exact(self) -> None:
        self.assertTrue(self.controls["all_one_c_plus_c5_counts_exact"])
        self.assertTrue(self.controls["all_binary_split_counts_exact"])
        for control in self.controls["controls"]:
            for target in control["targets"]:
                self.assertEqual(
                    target["one_c_plus_c5"]["branch_count"],
                    target["direct_c6_count"],
                )
                self.assertTrue(
                    all(
                        row["matches_direct"]
                        for row in target["all_binary_split_counts"]
                    )
                )

    def test_positive_repeated_empty_identity_and_sources_replay(
        self,
    ) -> None:
        self.assertTrue(
            self.controls[
                "positive_repeated_empty_and_identity_controls_present"
            ]
        )
        self.assertTrue(self.controls["all_sources_replay"])
        for control in self.controls["controls"]:
            rows = {row["kind"]: row for row in control["targets"]}
            self.assertTrue(rows["repeated_atom_positive"]["positive"])
            self.assertFalse(rows["empty"]["positive"])
            self.assertTrue(
                rows["repeated_atom_positive"][
                    "one_c_plus_c5"
                ]["source_replay_exact"]
            )

    def test_occurrence_split_table_cap_theorem(self) -> None:
        theorem = self.cost["explicit_occurrence_split_table_theorem"]
        self.assertFalse(theorem["any_row_meets_both_caps"])
        self.assertEqual(theorem["best_setup_compatible_stored_arity"], 3)
        self.assertEqual(
            theorem["best_setup_compatible_batch_exponent_B"]["exact"],
            "11/4",
        )
        self.assertEqual(theorem["minimum_stored_arity_for_batch_cap"], 5)
        self.assertEqual(
            theorem["minimum_online_compatible_setup_exponent_B"][
                "exact"
            ],
            "15/4",
        )
        self.assertFalse(
            theorem["arbitrary_endpoint_compressed_index_lower_bound_claimed"]
        )

    def test_one_atom_branch_consumes_all_online_slack(self) -> None:
        reduction = self.cost["one_atom_branch_reduction"]
        self.assertTrue(reduction["outer_query_count_equals_batch_cap"])
        self.assertEqual(
            reduction["outer_query_count_exponent_B"]["exact"],
            "5/4",
        )
        self.assertEqual(
            reduction["required_c5_query_exponent_B"]["exact"],
            "0",
        )
        self.assertTrue(
            reduction["negative_queries_require_exact_rejection"]
        )

    def test_current_k6_index_and_c2_scan_miss_setup_or_batch(
        self,
    ) -> None:
        theorem = self.cost["dinur_golovnev_k6"]
        self.assertEqual(
            theorem["zero_online_slack_forces_delta"]["exact"],
            "0",
        )
        self.assertEqual(theorem["state_exponent_B"]["exact"], "33/8")
        routes = {row["route_id"]: row for row in self.cost["routes"]}
        k6 = routes["one_c_branch_plus_dinur_golovnev_k6_index"]
        scan = routes["one_c_branch_plus_c2_scan_against_c3_hash"]
        self.assertFalse(k6["inside_setup_cap"])
        self.assertTrue(k6["inside_batch_cap"])
        self.assertTrue(scan["inside_setup_cap"])
        self.assertFalse(scan["inside_batch_cap"])
        self.assertEqual(
            scan["outer_query_batch_exponent_B"]["exact"],
            "11/4",
        )

    def test_standard_ffe_and_resultant_scopes_are_charged(self) -> None:
        routes = {row["route_id"]: row for row in self.cost["routes"]}
        quotient = routes["five_variable_deck_quotient_or_grid"]
        multipoint = routes["all_grid_multivariate_multipoint"]
        resultant = routes[
            "constant_order_truncated_c2_c3_resultant_per_c5_query"
        ]
        self.assertEqual(
            quotient["represented_dimension_exponent_B"]["exact"],
            "15/4",
        )
        self.assertEqual(quotient["degree_per_source_variable"], 16)
        self.assertFalse(
            quotient["x_only_s6_fixed_sign_source_biconditional"]
        )
        self.assertTrue(
            quotient["fixed_sign_point_source_verification_required"]
        )
        self.assertEqual(
            multipoint["represented_coefficient_or_output_exponent_B"][
                "exact"
            ],
            "15/4",
        )
        self.assertEqual(
            resultant["explicit_degree_body_exponent_B"]["exact"],
            "9/4",
        )
        self.assertFalse(
            self.cost[
                "general_arithmetic_circuit_or_data_structure_lower_bound_claimed"
            ]
        )

    def test_exact_reduction_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 10)
        self.assertEqual(admission["obligation_count"], 17)
        self.assertTrue(admission["exact_reduction_admitted"])
        self.assertTrue(admission["scoped_negative_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "output-sensitive nonlinear C5 membership",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
