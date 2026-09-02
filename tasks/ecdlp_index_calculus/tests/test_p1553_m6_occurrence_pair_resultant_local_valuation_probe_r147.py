import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT
    / "p1553_m6_occurrence_pair_resultant_local_valuation_probe_r147.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r147_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R147 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R147)


class M6OccurrencePairResultantLocalValuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R147.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R147.verify_source_bindings()), 11)

    def test_all_actual_controls_and_degrees_are_frozen(self):
        actual = self.controls["actual_controls"]
        self.assertEqual(len(actual), 8)
        self.assertEqual(
            sorted(row["occurrence_divisor_degree"] for row in actual),
            [27, 27, 125, 125, 216, 216, 343, 343],
        )
        self.assertEqual(
            sorted(row["squarefree_c3_support_degree"] for row in actual),
            [10, 10, 35, 35, 56, 56, 84, 84],
        )
        self.assertEqual(
            sorted(row["implicit_pair_resultant_degree"] for row in actual),
            [
                729,
                729,
                15625,
                15625,
                46656,
                46656,
                117649,
                117649,
            ],
        )

    def test_occurrence_root_multiplicities_are_exact(self):
        self.assertTrue(
            self.controls["all_occurrence_root_multiplicities_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertEqual(
                control["occurrence_polynomial_degree"],
                control["ordered_c3_occurrence_count"],
            )
            self.assertTrue(
                control["all_occurrence_root_multiplicities_exact"]
            )

    def test_positive_local_valuations_are_exact(self):
        self.assertTrue(
            self.controls["all_positive_pair_resultant_valuations_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertEqual(len(control["positive_queries"]), 12)
            self.assertTrue(control["all_positive_valuations_exact"])
            for query in control["positive_queries"]:
                self.assertTrue(query["valuation_equals_ordered_c6_count"])
                self.assertEqual(
                    query["pair_resultant_local_valuation"],
                    query["direct_ordered_c6_count"],
                )
                self.assertEqual(
                    query[
                        "required_truncation_order_for_first_nonzero_term"
                    ],
                    query["direct_ordered_c6_count"] + 1,
                )

    def test_empty_local_valuations_are_zero(self):
        self.assertTrue(
            self.controls["all_empty_pair_resultant_valuations_zero"]
        )
        for control in self.controls["actual_controls"]:
            self.assertEqual(len(control["empty_queries"]), 6)
            self.assertTrue(control["all_empty_valuations_zero"])
            for query in control["empty_queries"]:
                self.assertEqual(query["pair_resultant_local_valuation"], 0)
                self.assertTrue(query["valuation_equals_ordered_c6_count"])

    def test_finite_controls_use_no_candidate_oracles_or_credit(self):
        self.assertFalse(
            self.controls["candidate_discrete_log_oracle_consumed"]
        )
        self.assertFalse(self.controls["candidate_root_oracle_consumed"])
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_credit"]
        )
        for control in self.controls["actual_controls"]:
            self.assertFalse(
                control["candidate_discrete_log_oracle_consumed"]
            )
            self.assertFalse(control["candidate_root_oracle_consumed"])
            self.assertFalse(
                control["finite_control_receives_asymptotic_credit"]
            )

    def test_full_pair_resultant_is_not_materialized(self):
        pair_resultant = self.report["theorem"]["pair_resultant"]
        self.assertEqual(pair_resultant["degree"], "|C|^6=B^(9/2+o(1))")
        self.assertFalse(pair_resultant["root_extraction_required"])
        self.assertFalse(
            self.bundle["cost"]["full_pair_resultant_materialized"]
        )

    def test_standard_truncation_costs_are_frozen(self):
        cost = self.bundle["cost"]["standard_truncated_resultant_cost"]
        self.assertEqual(
            cost["occurrence_divisor_degree_exponent_B"]["exact"],
            "9/4",
        )
        self.assertEqual(
            cost["one_local_valuation_exponent_B"]["exact"],
            "9/4",
        )
        self.assertEqual(
            cost["one_known_target_a6_batch_exponent_B"]["exact"],
            "11/4",
        )
        self.assertEqual(
            cost["componentwise_full_relation_batch_exponent_B"]["exact"],
            "7/2",
        )
        self.assertEqual(
            cost["componentwise_full_relation_batch_exponent_N"]["exact"],
            "7/10",
        )
        self.assertFalse(cost["standard_one_known_target_batch_inside_rho"])
        self.assertFalse(cost["standard_full_relation_batch_inside_rho"])

    def test_cost_boundary_is_not_a_computational_lower_bound(self):
        cost = self.bundle["cost"]
        scope = cost["standard_truncated_resultant_cost"]["scope"]
        self.assertIn("componentwise or direct-product", scope)
        self.assertIn("not a lower bound", scope)
        self.assertFalse(
            cost["unconditional_computational_lower_bound_claimed"]
        )
        self.assertFalse(cost["unconditional_total_attack_cost_supplied"])

    def test_shared_index_marginals_logs_and_descent_remain_open(self):
        cost = self.bundle["cost"]
        logs = self.bundle["logs"]
        self.assertFalse(
            cost["inside_cap_multi_target_valuation_index_supplied"]
        )
        self.assertFalse(
            cost["offline_online_transposed_marginal_index_supplied"]
        )
        self.assertFalse(logs["candidate_factor_logs_computed"])
        self.assertFalse(
            logs["candidate_identical_target_descent_computed"]
        )
        self.assertFalse(logs["generic_prime_family_transfer_supplied"])

    def test_gate_admits_only_identity_and_standard_route_negative(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 25)
        self.assertTrue(admission["local_valuation_count_identity_admitted"])
        self.assertTrue(
            admission[
                "standard_componentwise_truncation_negative_admitted"
            ]
        )
        self.assertFalse(admission["shared_multi_target_operator_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
