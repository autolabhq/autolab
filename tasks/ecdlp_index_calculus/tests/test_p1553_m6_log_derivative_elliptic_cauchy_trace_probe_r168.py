from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_log_derivative_elliptic_cauchy_trace_probe_r168.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r168", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R168 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R168)


class M6LogDerivativeEllipticCauchyTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R168.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R168.verify_source_bindings()
        self.assertEqual(len(actual), 15)
        self.assertEqual(
            actual["r167_parent"],
            "0952a627b3bd5acd95e622533627f54b339605f92e9187be68d72431e7931db7",
        )
        self.assertEqual(
            actual["eagen_2022_596"],
            "5310b35d288a9462ff704eb77e7651d18f681a5b560cfb2919d1cfd0e01ae09e",
        )

    def test_neighboring_lanes_and_primary_source_are_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("denominator-aware additive trace", dedup["r167"])
        self.assertIn("attributes no ECDLP", dedup["eagen_2022_596"])
        self.assertIn("local valuations", dedup["r147"])
        self.assertIn("true-or-opposite Kummer", dedup["r166"])

    def test_polynomial_derivative_uses_ascending_coefficients(self) -> None:
        self.assertEqual(R168.polynomial_derivative([3, 4, 5], 101), [4, 10])
        self.assertEqual(R168.polynomial_derivative([7], 101), [0])

    def test_preregistered_control_grid_is_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)

    def test_every_direct_and_swapped_zero_is_simple_per_occurrence(self) -> None:
        self.assertTrue(
            self.controls["all_direct_and_swapped_zero_derivatives_nonzero"]
        )
        for row in self.rows:
            self.assertTrue(row["all_direct_zero_derivatives_nonzero"])
            self.assertTrue(row["all_swapped_zero_derivatives_nonzero"])

    def test_direct_and_swapped_multiplicities_are_identical(self) -> None:
        self.assertTrue(
            self.controls["all_direct_and_swapped_multiplicities_equal"]
        )
        self.assertEqual(self.controls["candidate_zero_occurrence_count"], 241)
        self.assertEqual(self.controls["true_orientation_occurrence_count"], 241)
        self.assertEqual(self.controls["opposite_orientation_occurrence_count"], 0)
        for row in self.rows:
            self.assertTrue(row["all_direct_and_swapped_multiplicities_equal"])

    def test_candidate_poles_match_r167_candidate_roots(self) -> None:
        self.assertTrue(self.controls["all_candidate_poles_match_r167_roots"])
        self.assertEqual(self.controls["candidate_pole_count"], 140)
        for row in self.rows:
            self.assertTrue(row["candidate_poles_match_r167_roots"])
            self.assertEqual(row["candidate_roots"], row["r167_candidate_roots"])

    def test_all_candidate_residues_survive_characteristic(self) -> None:
        self.assertTrue(
            self.controls["all_candidate_multiplicity_residues_nonzero"]
        )
        self.assertEqual(self.controls["maximum_candidate_multiplicity"], 4)
        for row in self.rows:
            self.assertTrue(row["all_candidate_multiplicity_residues_nonzero"])
            self.assertLess(row["maximum_candidate_multiplicity"], row["field_prime"])

    def test_public_target_equality_poles_are_regularized(self) -> None:
        self.assertTrue(
            self.controls["all_public_equality_poles_detected_and_regularized"]
        )
        self.assertEqual(self.controls["public_target_equality_correction_count"], 6)
        for row in self.rows:
            self.assertEqual(row["public_target_equality_correction_count"], 1)
            correction = row["public_target_equality_corrections"][0]
            self.assertEqual(correction["target_role"], "denominator_exception")
            self.assertTrue(correction["selected_sign_matches"])
            self.assertEqual(
                correction["rational_factor_pole_order"],
                2 * row["c3_divisor_degree"],
            )
            self.assertTrue(correction["rational_logarithmic_residue_nonzero"])
            self.assertEqual(correction["semantic_regularized_factor"], 1)
            self.assertEqual(
                correction["semantic_regularized_logarithmic_derivative"], 0
            )

    def test_logarithmic_trace_interface_is_denominator_aware(self) -> None:
        theorem = self.report["theorem"]
        self.assertIn("simple pole with residue m", theorem["invariant_logarithmic_derivative"])
        self.assertIn("exactly the true-or-opposite-sign", theorem["candidate_pole_biconditional"])
        self.assertIn("additive trace", theorem["logarithmic_weil_swap"])
        self.assertIn("value-only trace", theorem["linearized_interface"])
        self.assertIn("Fitting/subresultant", theorem["linearized_interface"])

    def test_compact_state_and_public_prefilter_are_inside_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["compact_h_and_dlog_h_state_exponent_B"]["exact"], "5/4")
        self.assertEqual(
            cost["public_target_equality_prefilter_exponent_B"]["exact"], "9/4"
        )
        self.assertTrue(cost["compact_dlog_h_state_inside_rho"])
        self.assertTrue(cost["public_target_equality_prefilter_inside_rho"])

    def test_explicit_trace_and_tensor_routes_are_above_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["direct_log_trace_table_exponent_B"]["exact"], "7/2")
        self.assertEqual(
            cost["raw_swapped_log_trace_table_exponent_B"]["exact"], "9/2"
        )
        self.assertEqual(
            cost["standard_tensor_quotient_state_exponent_B"]["exact"], "9/2"
        )
        self.assertFalse(cost["direct_log_trace_table_inside_rho"])
        self.assertFalse(cost["raw_swapped_log_trace_table_inside_rho"])
        self.assertFalse(cost["standard_tensor_quotient_inside_rho"])

    def test_value_only_trace_receives_no_locator_credit(self) -> None:
        cost = self.report["cost"]
        admission = self.report["admission"]
        self.assertFalse(cost["value_only_trace_receives_candidate_locator_credit"])
        self.assertFalse(
            cost["denominator_aware_elliptic_cauchy_trace_mod_u_supplied"]
        )
        self.assertTrue(admission["log_derivative_candidate_pole_interface_admitted"])
        self.assertTrue(admission["additive_elliptic_trace_linearization_admitted"])
        self.assertFalse(admission["denominator_aware_trace_constructor_admitted"])
        self.assertFalse(admission["lane_admitted"])

    def test_successor_forbids_inverting_away_candidates(self) -> None:
        interface = self.bundle["frozen"]["successor_interface"]
        forbidden = interface["forbidden_credit"]
        self.assertIn("nN or n^2", forbidden)
        self.assertIn("tensor quotient", forbidden)
        self.assertIn("value-only inversion", forbidden)
        self.assertIn("denominator-aware", interface["open_primitive"])

    def test_candidate_oracles_and_finite_credit_are_absent(self) -> None:
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_attack_credit"]
        )
        for row in self.rows:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )
            self.assertFalse(
                row["finite_pair_enumeration_receives_asymptotic_attack_credit"]
            )

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        obligations = self.report["admission"]["obligations"]
        self.assertFalse(
            obligations["denominator_aware_elliptic_cauchy_trace_mod_u_complete"]
        )
        self.assertFalse(obligations["candidate_safe_zero_divisor_handling_complete"])
        self.assertFalse(obligations["unconditional_total_attack_cost_complete"])
        self.assertFalse(obligations["generic_prime_coordinate_family_algorithm"])
        self.assertFalse(obligations["pollard_rho_improvement_complete"])
        self.assertFalse(obligations["shoup_improvement_complete"])
        self.assertFalse(obligations["breakthrough_complete"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
