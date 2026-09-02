import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "p1553_m6_actual_c6_shift_krylov_rank_probe_r149.py"
SPEC = importlib.util.spec_from_file_location("p1553_r149_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R149 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R149)


class M6ActualC6ShiftKrylovRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R149.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R149.verify_source_bindings()), 13)

    def test_all_actual_prime_order_decks_are_present(self):
        actual = self.controls["actual_controls"]
        self.assertEqual(len(actual), 8)
        self.assertTrue(
            self.controls[
                "all_prime_order_proper_binary_deck_conditions_hold"
            ]
        )
        for control in actual:
            self.assertTrue(control["subgroup_order_is_prime"])
            self.assertTrue(control["proper_nonempty_binary_deck"])
            self.assertTrue(control["labels_are_distinct"])

    def test_torus_to_scalar_c6_replays_are_exact(self):
        self.assertTrue(
            self.controls["all_torus_and_scalar_c6_replays_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control["torus_target_to_scalar_label_is_consistent"]
            )
            self.assertTrue(
                control["torus_and_scalar_c6_multiplicities_match"]
            )
            self.assertEqual(
                control["ordered_c6_total_multiplicity"],
                control["expected_ordered_c6_total_multiplicity"],
            )

    def test_actual_support_sizes_and_totals_are_frozen(self):
        actual = self.controls["actual_controls"]
        self.assertEqual(
            sorted(control["positive_c6_support_size"] for control in actual),
            [28, 28, 210, 210, 462, 462, 924, 924],
        )
        self.assertEqual(
            sorted(
                control["ordered_c6_total_multiplicity"]
                for control in actual
            ),
            [729, 729, 15625, 15625, 46656, 46656, 117649, 117649],
        )

    def test_count_fourier_nonvanishing_theorem_is_explicit(self):
        theorem = self.report["theorem"]
        self.assertIn("Phi_q", theorem["cyclotomic_nonvanishing"])
        self.assertIn(
            "DFT(u)_j^6",
            theorem["ordered_c6_fourier_transform"],
        )
        self.assertEqual(
            theorem["ordered_c6_shift_krylov_dimension"],
            "q",
        )
        self.assertEqual(
            theorem["ordered_c6_minimum_cyclic_linear_recurrence_order"],
            "q",
        )

    def test_atom_marginal_identity_and_rank_are_explicit(self):
        theorem = self.report["theorem"]
        self.assertEqual(
            theorem["atom_marginal_identity"],
            "For atom a, d_a(y)=6*u^(*5)(y-a).",
        )
        self.assertIn(
            "DFT(u)_j^5",
            theorem["atom_marginal_fourier_transform"],
        )
        self.assertEqual(
            theorem["atom_marginal_shift_krylov_dimension"],
            "q",
        )

    def test_count_and_marginal_theorems_apply_to_all_controls(self):
        self.assertTrue(
            self.controls[
                "all_count_and_marginal_full_support_theorems_apply"
            ]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control[
                    "characteristic_zero_count_fourier_support_theorem_applies"
                ]
            )
            self.assertTrue(
                control[
                    "characteristic_zero_c_atom_marginal_fourier_support_theorem_applies"
                ]
            )

    def test_full_shift_krylov_state_is_over_cap(self):
        cost = self.bundle["cost"]
        self.assertEqual(cost["group_order_exponent_B"]["exact"], "5")
        self.assertEqual(
            cost["count_linear_recurrence_state_exponent_B"]["exact"],
            "5",
        )
        self.assertEqual(
            cost[
                "shared_atom_marginal_linear_recurrence_state_exponent_B"
            ]["exact"],
            "5",
        )
        self.assertFalse(cost["count_recurrence_state_inside_setup_cap"])
        self.assertFalse(cost["marginal_recurrence_state_inside_setup_cap"])
        self.assertFalse(cost["full_target_generation_inside_rho"])

    def test_finite_labels_receive_no_candidate_credit(self):
        self.assertFalse(
            self.controls["candidate_discrete_log_oracle_consumed"]
        )
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
                control["finite_control_receives_asymptotic_credit"]
            )

    def test_scope_preserves_nonlinear_and_ffe_routes(self):
        scope = self.report["theorem"]["scope"]
        cost = self.bundle["cost"]
        self.assertIn("not a lower bound", scope)
        self.assertIn("nonlinear preprocessing", scope)
        self.assertIn("summation-polynomial/FFE", scope)
        self.assertFalse(
            cost["unconditional_computational_lower_bound_claimed"]
        )
        self.assertFalse(
            cost["nonlinear_compact_divisor_count_circuit_supplied"]
        )

    def test_gate_admits_only_full_linear_rank_boundary(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 25)
        self.assertTrue(
            admission[
                "actual_count_and_marginal_full_krylov_rank_admitted"
            ]
        )
        self.assertTrue(
            admission["linear_shift_recurrence_negative_admitted"]
        )
        self.assertFalse(
            admission["nonlinear_compact_divisor_circuit_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
