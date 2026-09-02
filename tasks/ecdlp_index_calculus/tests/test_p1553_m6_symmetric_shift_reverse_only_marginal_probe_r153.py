import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / (
    "p1553_m6_symmetric_shift_reverse_only_marginal_probe_r153.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r153_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R153 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R153)


class M6SymmetricShiftReverseOnlyMarginalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R153.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R153.verify_source_bindings()), 13)

    def test_known_a_and_symmetric_c_constructions_are_exact(self):
        self.assertEqual(self.controls["actual_control_count"], 8)
        for control in self.controls["actual_controls"]:
            a_scalars = control["known_symmetric_a_scalars"]
            modulus = control["subgroup_order"]
            self.assertEqual(
                {(-value) % modulus for value in a_scalars},
                set(a_scalars),
            )
            self.assertFalse(
                control["known_a_scalar_generation_requires_dlp"]
            )
            self.assertTrue(control["c_deck_inversion_symmetric"])
            self.assertTrue(
                control[
                    "symmetric_c_closure_has_constant_factor_width"
                ]
            )

    def test_c5_and_a6_evenness_are_exact(self):
        for control in self.controls["actual_controls"]:
            self.assertTrue(control["c5_kernel_even"])
            self.assertTrue(control["a6_shift_multiplicity_even"])

    def test_opposite_shift_blocks_are_transposes(self):
        self.assertTrue(
            self.controls["all_symmetric_block_identities_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control[
                    "all_opposite_shift_block_transposes_exact"
                ]
            )
            self.assertTrue(
                control[
                    "all_forward_actions_from_opposite_reverse_exact"
                ]
            )
            for block in control["blocks"]:
                self.assertTrue(
                    block["opposite_block_transpose_exact"]
                )
                self.assertTrue(
                    block[
                        "forward_action_from_opposite_reverse_exact"
                    ]
                )

    def test_counts_are_exact_row_sums(self):
        self.assertTrue(
            self.controls["all_counts_recovered_from_row_sums"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control["all_row_sums_recover_six_times_count"]
            )
            for block in control["blocks"]:
                self.assertTrue(
                    block["row_sum_recovers_six_times_count"]
                )

    def test_known_shift_relation_identities_are_exact(self):
        self.assertTrue(
            self.controls["all_known_shift_relation_identities_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control["all_known_shift_relation_identities_exact"]
            )
            for block in control["blocks"]:
                self.assertTrue(
                    block["known_shift_relation_identity_exact"]
                )

    def test_finite_rank_deficit_is_frozen(self):
        ranks = [
            control["stacked_relation_rank"]
            for control in self.controls["actual_controls"]
        ]
        self.assertEqual(ranks, [4, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(
            self.controls["finite_stacked_rank_range"],
            [0, 4],
        )
        self.assertEqual(
            self.controls["finite_full_rank_control_count"],
            0,
        )
        self.assertFalse(
            self.controls[
                "all_stacked_relation_matrices_full_c_log_rank"
            ]
        )
        self.assertFalse(
            self.controls["all_verifier_only_c_logs_recovered"]
        )

    def test_reverse_only_operator_consequence_is_explicit(self):
        theorem = self.report["theorem"]
        self.assertIn(
            "D_s^T=D_(-s)",
            theorem["transpose_pairing"],
        )
        self.assertIn(
            "A separate forward tangent",
            theorem["reverse_only_consequence"],
        )
        self.assertFalse(
            self.bundle["cost"]["separate_forward_tangent_operator_required"]
        )

    def test_density_is_model_bound_and_rank_transfer_is_absent(self):
        theorem = self.report["theorem"]
        cost = self.bundle["cost"]
        self.assertIn("model-bound", theorem["density_model"])
        self.assertIn(
            "ranks from zero to four",
            theorem["scope"],
        )
        self.assertEqual(
            cost[
                "uniform_model_expected_relation_count_exponent_B"
            ]["exact"],
            "3/4",
        )
        self.assertFalse(
            cost["structured_generic_prime_rank_and_density_supplied"]
        )

    def test_cost_preserves_exponents_but_supplies_no_operator(self):
        cost = self.bundle["cost"]
        self.assertEqual(
            cost["known_a_deck_exponent_B"]["exact"],
            "1/12",
        )
        self.assertEqual(
            cost["symmetric_c_deck_exponent_B"]["exact"],
            "3/4",
        )
        self.assertEqual(
            cost["structured_relation_row_count_exponent_B"]["exact"],
            "5/4",
        )
        self.assertTrue(cost["symmetric_closure_changes_no_exponent"])
        self.assertFalse(cost["reverse_only_marker_operator_supplied"])
        self.assertFalse(cost["signed_weight_separable_ffe_dag_supplied"])
        self.assertFalse(cost["unconditional_total_attack_cost_supplied"])

    def test_finite_labels_receive_no_candidate_credit(self):
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_credit"]
        )
        for control in self.controls["actual_controls"]:
            self.assertFalse(
                control["candidate_discrete_log_oracle_consumed"]
            )
            self.assertFalse(
                control["verifier_bsgs_labels_receive_candidate_credit"]
            )
            self.assertFalse(
                control[
                    "finite_rank_and_log_control_receives_asymptotic_credit"
                ]
            )

    def test_gate_admits_symmetry_and_rank_deficit_only(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 16)
        self.assertEqual(admission["obligation_count"], 27)
        self.assertTrue(
            admission["reverse_only_operator_reduction_admitted"]
        )
        self.assertTrue(admission["finite_rank_deficit_admitted"])
        self.assertFalse(
            admission[
                "finite_full_rank_controls_admitted_without_asymptotic_credit"
            ]
        )
        self.assertFalse(
            admission["reverse_only_signed_marker_operator_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
