import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / (
    "p1553_m6_geometry_only_weight_interpolation_adjoint_probe_r152.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r152_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R152 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R152)


class M6GeometryOnlyWeightInterpolationAdjointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R152.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R152.verify_source_bindings()), 14)

    def test_all_actual_decks_have_distinct_public_x_coordinates(self):
        self.assertEqual(self.controls["actual_control_count"], 8)
        for control in self.controls["actual_controls"]:
            self.assertTrue(control["a_deck"]["x_coordinates_distinct"])
            self.assertTrue(control["c_deck"]["x_coordinates_distinct"])
            self.assertEqual(
                control["a_deck"]["product_polynomial_degree"],
                control["a_deck"]["atom_count"],
            )
            self.assertEqual(
                control["c_deck"]["product_polynomial_degree"],
                control["c_deck"]["atom_count"],
            )

    def test_weight_interpolation_roundtrips_are_exact(self):
        self.assertTrue(
            self.controls["all_weight_interpolation_roundtrips_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(control["a_deck"]["weight_roundtrip_exact"])
            self.assertTrue(control["c_deck"]["weight_roundtrip_exact"])

    def test_dual_tangent_linearity_is_exact(self):
        self.assertTrue(
            self.controls["all_tangent_linearity_checks_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(control["a_deck"]["tangent_linearity_exact"])
            self.assertTrue(control["c_deck"]["tangent_linearity_exact"])

    def test_transpose_interpolation_pairings_are_exact(self):
        self.assertTrue(
            self.controls["all_transpose_interpolation_identities_exact"]
        )
        for control in self.controls["actual_controls"]:
            for deck_id in ("a_deck", "c_deck"):
                deck = control[deck_id]
                self.assertTrue(
                    deck["transpose_interpolation_identity_exact"]
                )
                self.assertEqual(
                    deck["transpose_pairing_forward"],
                    deck["transpose_pairing_reverse"],
                )

    def test_all_divisions_are_public_geometry_constants(self):
        theorem = self.report["theorem"]
        self.assertIn("P'(x_i)^(-1)", theorem["division_safety"])
        for control in self.controls["actual_controls"]:
            for deck_id in ("a_deck", "c_deck"):
                deck = control[deck_id]
                self.assertTrue(
                    deck["all_barycentric_denominators_nonzero"]
                )
                self.assertTrue(
                    deck[
                        "divisions_depend_only_on_distinct_public_x_coordinates"
                    ]
                )
                self.assertTrue(
                    deck["geometry_setup_independent_of_weights"]
                )

    def test_signed_point_boundary_is_preserved(self):
        theorem = self.report["theorem"]
        self.assertIn(
            "y-coordinate side table",
            theorem["signed_point_boundary"],
        )
        self.assertIn(
            "sign branches",
            theorem["signed_point_boundary"],
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control["a_deck"][
                    "signed_point_branches_retained_as_side_table"
                ]
            )
            self.assertTrue(
                control["c_deck"][
                    "signed_point_branches_retained_as_side_table"
                ]
            )

    def test_leaf_state_and_apply_cost_are_inside_caps(self):
        cost = self.bundle["cost"]
        self.assertEqual(
            cost["geometry_subproduct_tree_state_exponent_B"]["exact"],
            "3/4",
        )
        self.assertEqual(
            cost["one_tangent_payload_compile_exponent_B"]["exact"],
            "3/4",
        )
        self.assertEqual(
            cost["one_adjoint_payload_apply_exponent_B"]["exact"],
            "3/4",
        )
        self.assertTrue(cost["leaf_geometry_state_inside_setup_cap"])
        self.assertTrue(
            cost["leaf_tangent_and_adjoint_inside_fresh_cap"]
        )
        self.assertTrue(
            cost["weight_independent_leaf_derivative_state_supplied"]
        )

    def test_internal_elimination_remains_open(self):
        cost = self.bundle["cost"]
        scope = self.report["theorem"]["scope"]
        self.assertIn("leaf tangent and adjoint state only", scope)
        self.assertFalse(
            cost["weight_separable_internal_elimination_dag_supplied"]
        )
        self.assertFalse(
            cost["bidirectional_marker_batch_operator_supplied"]
        )
        self.assertFalse(cost["generic_prime_rank_and_density_supplied"])
        self.assertFalse(cost["unconditional_total_attack_cost_supplied"])

    def test_controls_are_scalar_blind(self):
        self.assertTrue(self.controls["all_controls_scalar_blind"])
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_credit"]
        )
        for control in self.controls["actual_controls"]:
            self.assertFalse(control["candidate_scalar_labels_consumed"])
            self.assertFalse(
                control["candidate_discrete_log_oracle_consumed"]
            )

    def test_gate_admits_only_leaf_derivative_state(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 25)
        self.assertTrue(
            admission[
                "weight_independent_leaf_derivative_state_admitted"
            ]
        )
        self.assertFalse(
            admission["bidirectional_marker_batch_operator_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
