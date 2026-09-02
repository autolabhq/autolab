from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_sparse_fourier_predicate_transfer_probe_r130.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r130", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R130 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R130)


class TorusC5SparseFourierPredicateTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R130.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.controls = cls.bundle["controls"]
        cls.counterexample = cls.controls["finite_field_counterexample"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R130.verify_source_bindings()), 12)

    def test_gf2_10_modulus_and_inverses_are_exact(self) -> None:
        self.assertTrue(R130.gf2_10_modulus_is_irreducible())
        for value in (1, 2, 7, 713, 1023):
            self.assertEqual(
                R130.gf2_10_mul(value, R130.gf2_10_inv(value)),
                1,
            )

    def test_counterexample_root_has_exact_order_eleven(self) -> None:
        omega = self.counterexample["primitive_root_encoding"]
        self.assertNotEqual(omega, 1)
        self.assertEqual(R130.gf2_10_pow(omega, 11), 1)
        self.assertTrue(
            self.counterexample["primitive_root_has_exact_order_11"]
        )

    def test_five_by_five_fourier_minor_is_singular(self) -> None:
        self.assertEqual(self.counterexample["minor_size"], 5)
        self.assertEqual(self.counterexample["minor_rank"], 4)
        self.assertTrue(self.counterexample["minor_singular"])

    def test_five_mode_polynomial_has_exact_five_zero_set(self) -> None:
        self.assertEqual(self.counterexample["nonzero_mode_count"], 5)
        self.assertEqual(self.counterexample["zero_count"], 5)
        self.assertEqual(
            self.counterexample["observed_zero_exponents"],
            self.counterexample["zero_exponents"],
        )
        self.assertTrue(self.counterexample["zero_set_exact"])
        self.assertTrue(
            self.counterexample[
                "complex_sharp_zero_bound_violated_after_reduction"
            ]
        )

    def test_all_actual_pairing_fields_have_order_two(self) -> None:
        self.assertEqual(self.controls["actual_pairing_field_count"], 4)
        self.assertTrue(
            self.controls[
                "all_actual_pairing_fields_have_order_two_characteristic"
            ]
        )
        self.assertTrue(
            self.controls[
                "all_actual_subgroups_embed_in_quadratic_extension"
            ]
        )
        for row in self.controls["actual_pairing_fields"]:
            self.assertTrue(row["characteristic_is_minus_one_mod_subgroup_order"])
            self.assertEqual(
                row["multiplicative_order_of_characteristic_mod_subgroup"],
                2,
            )

    def test_actual_fields_fail_primitive_chebotarev_condition(self) -> None:
        self.assertTrue(
            self.controls[
                "all_actual_pairing_fields_fail_primitive_order_condition"
            ]
        )
        for row in self.controls["actual_pairing_fields"]:
            self.assertFalse(
                row["primitive_chebotarev_order_condition_passes"]
            )
            self.assertGreater(
                row["primitive_chebotarev_order_required"],
                2,
            )

    def test_complex_fourier_costs_receive_no_finite_field_credit(
        self,
    ) -> None:
        zero_test = self.routes["complex_sparse_fourier_color_zero_test"]
        indicator = self.routes["complex_exact_color_indicator"]
        self.assertEqual(
            zero_test["minimum_mode_exponent_B"]["exact"],
            "15/4",
        )
        self.assertEqual(
            indicator["minimum_mode_exponent_B"]["exact"],
            "5",
        )
        self.assertFalse(zero_test["transfers_to_actual_finite_fields"])
        self.assertFalse(indicator["transfers_to_actual_finite_fields"])
        self.assertFalse(zero_test["candidate_work_credit"])
        self.assertFalse(indicator["candidate_work_credit"])

    def test_order_two_and_nonfourier_routes_remain_open(self) -> None:
        order_two = self.routes[
            "order_two_finite_field_fourier_minor_theorem"
        ]
        nonfourier = self.routes[
            "nonfourier_shared_predicate_or_low_slp_selector_dag"
        ]
        self.assertFalse(order_two["all_minors_theorem_supplied"])
        self.assertEqual(order_two["status"], "open")
        self.assertFalse(nonfourier["exact_circuit_constructed"])
        self.assertEqual(nonfourier["status"], "open")
        self.assertIn("order-two", self.frozen["preserved_interface"])

    def test_transfer_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 12)
        self.assertEqual(admission["obligation_count"], 20)
        self.assertTrue(
            admission["complex_fourier_predicate_negative_admitted"]
        )
        self.assertTrue(admission["finite_field_transfer_rejection_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "ord_q(characteristic)=2",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
