from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_torus_c5_linear_sketch_circulant_probe_r124.py"
SPEC = importlib.util.spec_from_file_location("p1553_r124", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R124 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R124)


class TorusC5LinearSketchCirculantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R124.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R124.verify_source_bindings()), 12)

    def test_cyclic_convolution_power_preserves_total_multiplicity(
        self,
    ) -> None:
        deck = (1, 1, 0, 0, 0)
        for exponent in range(1, 6):
            values = R124.convolution_power(deck, exponent)
            self.assertEqual(sum(values), 2**exponent)

    def test_translated_c3_c2_queries_equal_c5(self) -> None:
        self.assertEqual(self.controls["control_count"], 5)
        self.assertTrue(
            self.controls[
                "all_translated_inner_product_identities_exact"
            ]
        )
        for row in self.controls["controls"]:
            self.assertEqual(
                row["c5_total_multiplicity"],
                row["deck_size"] ** 5,
            )

    def test_all_control_circulants_have_full_rational_rank(self) -> None:
        self.assertTrue(
            self.controls["all_rational_circulant_ranks_full"]
        )
        for row in self.controls["controls"]:
            self.assertEqual(
                row["rational_circulant_rank"],
                row["group_order"],
            )

    def test_selected_finite_field_spectra_are_full(self) -> None:
        self.assertTrue(
            self.controls[
                "all_selected_finite_field_spectral_ranks_full"
            ]
        )
        for row in self.controls["controls"]:
            self.assertEqual(
                row["finite_field_nonzero_deck_fourier_mode_count"],
                row["group_order"],
            )
            self.assertEqual(
                row["finite_field_nonzero_c2_kernel_mode_count"],
                row["group_order"],
            )

    def test_row_space_theorem_forces_q_measurements(self) -> None:
        theorem = self.report["theorem"]
        self.assertEqual(theorem["minimum_linear_sketch_dimension"], "q")
        self.assertIn("rowspace(S)", theorem["row_space_argument"])
        self.assertIn("rank(C_w)", theorem["row_space_argument"])
        self.assertEqual(
            theorem["dimension_exponent_B"]["exact"],
            "5",
        )
        self.assertFalse(theorem["inside_setup_cap"])

    def test_prime_order_cyclotomic_scope_is_explicit(self) -> None:
        theorem = self.report["theorem"]
        self.assertIn(
            "proper nonempty binary deck",
            theorem["prime_order_binary_deck_criterion"],
        )
        self.assertIn(
            "Phi_q",
            theorem["prime_order_binary_deck_criterion"],
        )
        self.assertIn(
            "characteristic-zero rank argument",
            theorem["scope_limits"],
        )

    def test_represented_split_still_misses_query_cap(self) -> None:
        route = self.routes["represented_C3_hash_then_C2_scan"]
        self.assertEqual(route["setup_exponent_B"]["exact"], "9/4")
        self.assertEqual(route["query_exponent_B"]["exact"], "3/2")
        self.assertTrue(route["inside_setup_cap"])
        self.assertFalse(route["inside_polylog_query_cap"])

    def test_coupled_nonlinear_route_remains_open(self) -> None:
        route = self.routes["coupled_nonlinear_membership_data_structure"]
        self.assertFalse(route["scoped_lower_bound_proved"])
        self.assertFalse(route["exact_structure_constructed"])
        self.assertEqual(route["status"], "open")
        self.assertIn(
            "coupled deck powers",
            self.frozen["preserved_interface"],
        )
        self.assertFalse(
            self.frozen[
                "general_data_structure_or_arithmetic_circuit_"
                "lower_bound_claimed"
            ]
        )

    def test_scoped_negative_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 12)
        self.assertEqual(admission["obligation_count"], 20)
        self.assertTrue(
            admission["translated_inner_product_semantics_admitted"]
        )
        self.assertTrue(
            admission["scoped_universal_linear_sketch_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "target-specialized nonlinear zero-test",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
