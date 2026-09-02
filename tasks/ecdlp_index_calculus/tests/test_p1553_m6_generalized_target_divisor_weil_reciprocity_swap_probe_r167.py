from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_r167.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r167", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R167 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R167)


class M6GeneralizedTargetDivisorWeilReciprocitySwapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R167.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R167.verify_source_bindings()
        self.assertEqual(len(actual), 14)
        self.assertEqual(
            actual["eagen_2022_596"],
            "5310b35d288a9462ff704eb77e7651d18f681a5b560cfb2919d1cfd0e01ae09e",
        )
        self.assertEqual(
            actual["miller_1986"],
            "39c76c7643278b87b3d8c24b9a07d0b4cbfb561cd13735548990848e0f0bd166",
        )

    def test_neighboring_lanes_and_literature_are_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("fast resultant modulo U", dedup["r166"])
        self.assertIn("refined", dedup["r165"])
        self.assertIn("does not attribute an ECDLP speedup", dedup["eagen_2022_596"])
        self.assertIn("line-function divisor arithmetic", dedup["miller_1986"])
        self.assertIn("arbitrary target divisor", dedup["r113"])

    def test_preregistered_control_grid_and_pole_filter_are_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)
        self.assertEqual(
            self.controls["dropped_selected_endpoint_target_count"], 6
        )
        for row in self.rows:
            self.assertEqual(row["dropped_selected_endpoint_target_count"], 1)
            self.assertEqual(row["dropped_target_roles"], ["denominator_exception"])

    def test_riemann_roch_basis_has_exact_dimension_and_weights(self) -> None:
        for pole_order in range(2, 17):
            basis = R167.riemann_roch_basis(pole_order)
            self.assertEqual(len(basis), pole_order)
            self.assertEqual([entry["weight"] for entry in basis], sorted(
                entry["weight"] for entry in basis
            ))
            self.assertEqual(basis[-1]["weight"], pole_order)

    def test_both_completed_zero_lists_sum_to_infinity(self) -> None:
        self.assertTrue(self.controls["all_zero_lists_sum_to_infinity"])
        for row in self.rows:
            self.assertTrue(row["numerator_point_sum_is_infinity"])
            self.assertTrue(row["denominator_point_sum_is_infinity"])
            self.assertTrue(row["numerator_points_distinct"])
            self.assertTrue(row["denominator_points_distinct"])

    def test_witness_nullspaces_and_full_pole_orders_are_exact(self) -> None:
        self.assertTrue(self.controls["all_riemann_roch_witnesses_exact"])
        for row in self.rows:
            expected = row["riemann_roch_ambient_pole_order"]
            for witness_name in ("numerator_witness", "denominator_witness"):
                witness = row[witness_name]
                self.assertEqual(witness["nullity"], 1)
                self.assertEqual(witness["matrix_rank"], expected - 1)
                self.assertEqual(witness["pole_order"], expected)
                self.assertTrue(witness["all_prescribed_values_zero"])

    def test_common_zero_and_infinity_poles_cancel(self) -> None:
        self.assertTrue(self.controls["all_common_zero_cancellations_recorded"])
        for row in self.rows:
            self.assertTrue(row["common_zero_cancels_from_quotient_divisor"])
            self.assertEqual(
                row["quotient_zero_divisor_degree"],
                row["retained_target_count"],
            )
            self.assertEqual(
                row["quotient_pole_divisor_degree"],
                row["retained_target_count"],
            )
            self.assertNotEqual(row["h_at_infinity"], 0)

    def test_auxiliary_corrections_are_units(self) -> None:
        self.assertTrue(self.controls["all_auxiliary_corrections_are_units"])
        self.assertEqual([row["auxiliary_offset"] for row in self.rows], [2, 1, 1, 1, 1, 1])
        self.assertTrue(
            all(
                row["all_correction_factors_units_on_selected_divisor"]
                for row in self.rows
            )
        )

    def test_direct_and_reciprocity_transcripts_are_identical(self) -> None:
        self.assertTrue(self.controls["all_reciprocity_identities_exact"])
        self.assertEqual(self.controls["evaluated_selected_endpoint_count"], 202)
        self.assertEqual(self.controls["candidate_zero_count"], 140)
        self.assertEqual(self.controls["disjoint_support_row_count"], 62)
        self.assertEqual(
            self.controls["specialized_common_zero_row_count"], 140
        )
        for row in self.rows:
            self.assertTrue(row["all_reciprocity_identities_exact"])
            self.assertEqual(
                row["direct_product_transcript_sha256"],
                row["reciprocity_transcript_sha256"],
            )

    def test_candidate_zero_rows_are_scoped_as_specializations(self) -> None:
        theorem = self.report["theorem"]["weil_reciprocity_swap"]
        self.assertIn("extends by specialization", theorem)
        self.assertIn("not claimed as literal disjoint-support", theorem)
        self.assertGreater(self.controls["specialized_common_zero_row_count"], 0)
        self.assertGreater(self.controls["disjoint_support_row_count"], 0)

    def test_raw_swap_is_larger_than_direct_table(self) -> None:
        self.assertEqual(self.controls["direct_target_evaluation_count"], 1486)
        self.assertEqual(self.controls["raw_swapped_h_evaluation_count"], 17844)
        self.assertGreater(
            self.controls["raw_swapped_h_evaluation_count"],
            self.controls["direct_target_evaluation_count"],
        )
        cost = self.report["cost"]
        self.assertEqual(cost["direct_target_table_exponent_B"]["exact"], "7/2")
        self.assertEqual(cost["raw_weil_swapped_table_exponent_B"]["exact"], "9/2")
        self.assertFalse(cost["direct_target_table_inside_rho"])
        self.assertFalse(cost["raw_weil_swap_inside_rho"])

    def test_standard_resultant_remains_above_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["standard_elliptic_resultant_representation_exponent_B"]["exact"],
            "7/2",
        )
        self.assertFalse(cost["standard_represented_elliptic_resultant_inside_rho"])
        self.assertFalse(cost["output_sensitive_elliptic_resultant_mod_u_supplied"])

    def test_compact_target_state_is_not_promoted_to_fast_restriction(self) -> None:
        cost = self.report["cost"]
        admission = self.report["admission"]
        self.assertEqual(
            cost["compact_target_divisor_slp_state_exponent_B"]["exact"], "5/4"
        )
        self.assertTrue(cost["compact_target_divisor_slp_inside_rho"])
        self.assertTrue(admission["compact_target_divisor_witness_admitted"])
        self.assertFalse(admission["output_sensitive_resultant_constructor_admitted"])
        self.assertFalse(admission["lane_admitted"])

    def test_successor_forbids_hidden_large_intermediates(self) -> None:
        interface = self.bundle["frozen"]["successor_interface"]
        forbidden = interface["forbidden_credit"]
        self.assertIn("n-by-N", forbidden)
        self.assertIn("2n-by-n", forbidden)
        self.assertIn("degree-Theta(nN)", forbidden)
        self.assertIn("resultant", interface["open_primitive"])

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
                row["dense_finite_nullspace_receives_asymptotic_attack_credit"]
            )

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        obligations = self.report["admission"]["obligations"]
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
