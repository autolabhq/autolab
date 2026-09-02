import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "p1553_m6_weighted_c3_mobius_gcd_trace_probe_r145.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r145_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R145 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R145)


class M6WeightedC3MobiusGcdTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R145.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R145.verify_source_bindings()), 14)

    def test_all_actual_controls_are_present(self):
        self.assertEqual(self.controls["actual_control_count"], 8)
        degrees = sorted(
            row["c3_support_degree"]
            for row in self.controls["actual_controls"]
        )
        self.assertEqual(degrees, [10, 10, 35, 35, 56, 56, 84, 84])

    def test_support_and_weight_polynomials_are_exact(self):
        self.assertTrue(
            self.controls["all_c3_supports_in_cayley_chart"]
        )
        self.assertTrue(
            self.controls[
                "all_support_polynomials_and_weight_interpolants_exact"
            ]
        )

    def test_positive_gcd_degrees_and_weighted_traces_are_exact(self):
        self.assertTrue(
            self.controls[
                "all_gcd_intersections_and_quotient_traces_exact"
            ]
        )
        self.assertTrue(
            self.controls[
                "all_positive_queries_match_direct_c6_counts"
            ]
        )
        for control in self.controls["actual_controls"]:
            self.assertEqual(len(control["positive_queries"]), 12)
            self.assertTrue(
                control["all_positive_queries_match_direct_c6_count"]
            )
            for query in control["positive_queries"]:
                self.assertTrue(
                    query["gcd_degree_matches_direct_intersection"]
                )
                self.assertTrue(query["trace_count_exact_integer"])
                self.assertTrue(query["trace_matches_direct_c6_counter"])

    def test_empty_gcds_and_weighted_traces_are_zero(self):
        self.assertTrue(self.controls["all_empty_queries_rejected"])
        for control in self.controls["actual_controls"]:
            self.assertEqual(len(control["empty_queries"]), 6)
            self.assertTrue(control["all_empty_queries_rejected"])
            for query in control["empty_queries"]:
                self.assertEqual(query["gcd_degree"], 0)
                self.assertEqual(query["direct_ordered_pair_count"], 0)
                self.assertEqual(
                    query["quotient_trace_count_mod_field_prime"],
                    0,
                )

    def test_candidate_uses_neither_roots_nor_discrete_logs(self):
        self.assertFalse(self.controls["candidate_root_oracle_consumed"])
        self.assertFalse(
            self.controls["candidate_discrete_log_oracle_consumed"]
        )
        for control in self.controls["actual_controls"]:
            self.assertFalse(control["candidate_root_oracle_consumed"])
            self.assertFalse(
                control["candidate_discrete_log_oracle_consumed"]
            )
            self.assertFalse(
                control["finite_control_receives_asymptotic_credit"]
            )

    def test_standard_explicit_route_cost_is_fully_charged(self):
        cost = R145.standard_route_cost()
        self.assertEqual(
            cost["explicit_support_polynomial_state_exponent_B"]["exact"],
            "9/4",
        )
        self.assertEqual(
            cost[
                "one_mobius_pullback_coefficient_output_exponent_B"
            ]["exact"],
            "9/4",
        )
        self.assertEqual(
            cost["one_full_six_factor_count_exponent_B"]["exact"],
            "11/4",
        )
        self.assertEqual(
            cost["one_full_six_factor_count_exponent_N"]["exact"],
            "11/20",
        )
        self.assertFalse(
            cost["standard_one_c6_query_inside_fresh_cap"]
        )
        self.assertFalse(cost["standard_a6_batch_inside_rho"])

    def test_negative_scope_is_not_an_implicit_lower_bound(self):
        scope = self.report["theorem"]["scope"]
        self.assertIn("not a lower bound", scope)
        self.assertIn("implicit modular resultants", scope)
        self.assertIn("target-batched remainders", scope)

    def test_count_index_and_transposed_marginals_remain_open(self):
        cost = self.bundle["cost"]
        self.assertFalse(cost["inside_cap_weighted_count_index_supplied"])
        self.assertFalse(
            cost["offline_online_transposed_derivative_index_supplied"]
        )
        self.assertFalse(
            cost["structured_rank_density_theorem_supplied"]
        )
        self.assertFalse(cost["unconditional_total_attack_cost_supplied"])

    def test_logs_descent_and_generic_transfer_remain_open(self):
        logs = self.bundle["logs"]
        self.assertFalse(logs["candidate_factor_logs_computed"])
        self.assertFalse(
            logs["candidate_identical_target_descent_computed"]
        )
        self.assertFalse(logs["generic_prime_family_transfer_supplied"])

    def test_gate_admits_only_identity_and_scoped_negative(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 17)
        self.assertEqual(admission["obligation_count"], 28)
        self.assertTrue(admission["weighted_gcd_trace_identity_admitted"])
        self.assertTrue(
            admission["standard_explicit_route_negative_admitted"]
        )
        self.assertFalse(
            admission["implicit_batched_count_operator_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
