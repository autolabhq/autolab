import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / (
    "p1553_m6_rational_convolution_subalgebra_rigidity_probe_r150.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r150_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R150 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R150)


class M6RationalConvolutionSubalgebraRigidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R150.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R150.verify_source_bindings()), 10)

    def test_prime_cyclic_rational_crt_rigidity_is_explicit(self):
        theorem = self.report["theorem"]
        self.assertIn(
            "Q direct-product Q(zeta_q)",
            theorem["ambient_crt_decomposition"],
        )
        self.assertIn(
            "1, q-1, and q",
            theorem["ambient_quotient_rigidity"],
        )
        self.assertIn(
            "augmentation",
            theorem["ambient_quotient_rigidity"],
        )

    def test_deck_galois_stabilizer_degree_is_explicit(self):
        theorem = self.report["theorem"]
        self.assertEqual(
            theorem["multiplier_stabilizer"],
            "H_C={k in F_q^*: kC=C}.",
        )
        self.assertEqual(
            theorem["cyclotomic_degree"],
            "[Q(U(zeta_q)):Q]=(q-1)/|H_C|.",
        )
        self.assertIn(
            "(q-1)/|C|",
            theorem["stabilizer_size_bound"],
        )
        self.assertIn(
            "C={0}",
            theorem["stabilizer_size_bound"],
        )

    def test_all_actual_multiplier_orbits_are_exact(self):
        self.assertEqual(self.controls["actual_control_count"], 8)
        self.assertTrue(self.controls["all_actual_decks_valid"])
        self.assertTrue(
            self.controls["all_multiplier_orbit_checks_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(control["labels_are_distinct"])
            self.assertTrue(control["proper_nonempty_binary_deck"])
            self.assertTrue(
                control["stabilizer_order_divides_q_minus_one"]
            )
            self.assertTrue(
                control["stabilizer_orbits_cover_nonzero_deck"]
            )
            self.assertTrue(
                control["orbit_degree_meets_deck_size_bound"]
            )
            self.assertEqual(
                control["cyclotomic_conjugate_orbit_degree"],
                (control["subgroup_order"] - 1)
                // control["multiplier_stabilizer_order"],
            )

    def test_actual_control_stabilizers_are_frozen(self):
        actual = self.controls["actual_controls"]
        self.assertEqual(
            [row["multiplier_stabilizer_order"] for row in actual],
            [1] * 8,
        )
        self.assertEqual(
            [row["deck_size"] for row in actual],
            [3, 3, 5, 5, 6, 6, 7, 7],
        )
        self.assertEqual(
            [row["cyclotomic_conjugate_orbit_degree"] for row in actual],
            [
                16426,
                16426,
                524682,
                524682,
                1600606,
                1600606,
                16780522,
                16780522,
            ],
        )

    def test_augmentation_only_collision_witnesses_are_exact(self):
        self.assertTrue(
            self.controls[
                "all_augmentation_collision_witnesses_exact"
            ]
        )
        for control in self.controls["actual_controls"]:
            witness = control["augmentation_only_collision_witness"]
            self.assertTrue(witness["same_augmentation"])
            self.assertTrue(witness["different_selected_c6_count"])
            self.assertTrue(
                control[
                    "augmentation_only_cannot_recover_selected_c6_counts"
                ]
            )

    def test_generated_subalgebra_bound_is_over_both_caps(self):
        cost = self.bundle["cost"]
        self.assertEqual(cost["group_order_exponent_B"]["exact"], "5")
        self.assertEqual(cost["c_atom_deck_exponent_B"]["exact"], "3/4")
        self.assertEqual(
            cost[
                "universal_deck_generated_subalgebra_lower_bound_exponent_B"
            ]["exact"],
            "17/4",
        )
        self.assertFalse(
            cost["universal_subalgebra_bound_inside_setup_cap"]
        )
        self.assertFalse(
            cost["universal_subalgebra_bound_inside_pollard_rho"]
        )

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

    def test_scope_preserves_fixed_depth_nonhomomorphic_and_ffe_routes(self):
        theorem = self.report["theorem"]
        cost = self.bundle["cost"]
        self.assertIn(
            "reusable characteristic-zero",
            theorem["actual_subalgebra_scope"],
        )
        self.assertIn(
            "not a time lower bound",
            theorem["excluded_models"],
        )
        self.assertIn(
            "bounded-depth circuit",
            theorem["excluded_models"],
        )
        self.assertIn(
            "summation-polynomial/FFE",
            theorem["excluded_models"],
        )
        self.assertFalse(
            cost["unconditional_computational_lower_bound_claimed"]
        )
        self.assertFalse(cost["finite_depth_u6_marker_circuit_supplied"])

    def test_novelty_is_not_overclaimed(self):
        self.assertEqual(
            self.report["theorem"]["novelty_status"],
            "elementary_derivation_novelty_unverified",
        )

    def test_gate_admits_only_multiplication_closed_boundary(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 14)
        self.assertEqual(admission["obligation_count"], 24)
        self.assertTrue(
            admission[
                "rational_convolution_subalgebra_rigidity_admitted"
            ]
        )
        self.assertTrue(
            admission[
                "reusable_multiplication_closed_route_negative_admitted"
            ]
        )
        self.assertFalse(
            admission[
                "finite_depth_target_specialized_u6_circuit_admitted"
            ]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
