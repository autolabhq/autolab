from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_base_field_frobenius_predicate_dag_probe_r132.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r132", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R132 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R132)


class TorusC5BaseFieldFrobeniusPredicateDagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R132.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R132.verify_source_bindings()), 10)

    def test_frobenius_is_conjugation_and_inversion(self) -> None:
        self.assertTrue(
            self.controls["all_fields_have_characteristic_minus_one_mod_q"]
        )
        self.assertTrue(
            self.controls["all_frobenius_maps_equal_inversion"]
        )
        for control in self.controls["controls"]:
            self.assertTrue(
                control["all_target_frobenius_values_equal_inverses"]
            )

    def test_all_actual_positive_inverses_are_empty(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertTrue(
            self.controls["all_positive_supports_inversion_disjoint"]
        )
        self.assertTrue(self.controls["all_positive_inverses_are_empty"])
        for control in self.controls["controls"]:
            self.assertEqual(
                control["positive_inverse_empty_count"],
                control["c5_source_count"],
            )
            self.assertEqual(control["positive_inverse_in_c5_count"], 0)

    def test_base_polynomial_frobenius_identity_is_exact(self) -> None:
        self.assertTrue(
            self.controls[
                "all_sample_base_polynomial_frobenius_identities_exact"
            ]
        )
        self.assertTrue(
            self.controls[
                "all_sample_base_polynomial_zero_outcomes_invariant"
            ]
        )

    def test_exact_annihilators_require_extension_coefficients(self) -> None:
        self.assertTrue(
            self.controls[
                "all_support_annihilators_use_extension_coefficients"
            ]
        )
        self.assertTrue(
            self.controls[
                "all_color_annihilators_use_extension_coefficients"
            ]
        )
        for control in self.controls["controls"]:
            self.assertFalse(control["support_annihilator_in_base_field"])
            self.assertTrue(
                control["support_annihilator_uses_extension_coefficient"]
            )

    def test_all_color_inverses_are_empty(self) -> None:
        self.assertTrue(
            self.controls["all_color_supports_inversion_disjoint"]
        )
        self.assertTrue(self.controls["all_color_inverses_are_empty"])
        for control in self.controls["controls"]:
            for color in control["color_controls"]:
                self.assertEqual(
                    color["accepted_inverse_empty_count"],
                    color["accepted_target_count"],
                )

    def test_random_subset_comparator_is_model_bound(self) -> None:
        theorem = self.cost["theorem"]
        self.assertEqual(
            theorem["random_subset_overlap_exponent_B"]["exact"],
            "5/2",
        )
        self.assertTrue(
            theorem["random_subset_comparator_is_model_bound"]
        )
        self.assertFalse(
            theorem["random_subset_comparator_receives_candidate_credit"]
        )

    def test_base_field_zero_test_dags_are_rejected(self) -> None:
        polynomial = self.routes[
            "base_field_univariate_zero_predicate_dag"
        ]
        trace = self.routes[
            "base_field_trace_or_dickson_predicate_dag"
        ]
        self.assertTrue(polynomial["inversion_invariant"])
        self.assertFalse(
            polynomial["exact_on_actual_inversion_disjoint_supports"]
        )
        self.assertEqual(
            polynomial["status"],
            "rejected_on_actual_controls",
        )
        self.assertEqual(trace["status"], "rejected_on_actual_controls")

    def test_asymmetric_routes_remain_open(self) -> None:
        extension = self.routes["extension_field_lacunary_predicate"]
        coordinate = self.routes[
            "frobenius_aware_coordinate_predicate_dag"
        ]
        self.assertFalse(extension["inversion_invariant_forced"])
        self.assertEqual(extension["status"], "open")
        self.assertTrue(coordinate["may_distinguish_z_from_inverse"])
        self.assertEqual(coordinate["status"], "open")
        self.assertIn(
            "asymmetric extension-field",
            self.frozen["preserved_interface"],
        )

    def test_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 12)
        self.assertEqual(admission["obligation_count"], 20)
        self.assertTrue(
            admission["base_field_frobenius_dag_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("asymmetric", self.report["next_action"])


if __name__ == "__main__":
    unittest.main()
