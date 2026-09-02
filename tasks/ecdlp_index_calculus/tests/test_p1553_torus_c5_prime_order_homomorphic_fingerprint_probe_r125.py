from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_torus_c5_prime_order_homomorphic_fingerprint_probe_r125.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r125", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R125 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R125)


class TorusC5PrimeOrderHomomorphicFingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R125.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R125.verify_source_bindings()), 12)

    def test_prime_power_map_image_dichotomy(self) -> None:
        for order in (5, 7, 11, 13):
            image_sizes = {
                R125.power_map_image_size(order, exponent)
                for exponent in range(order)
            }
            self.assertEqual(image_sizes, {1, order})

    def test_prime_controls_have_positive_and_empty_targets(self) -> None:
        self.assertEqual(self.controls["prime_control_count"], 4)
        self.assertTrue(
            self.controls[
                "all_prime_controls_have_positive_and_empty_targets"
            ]
        )
        for row in self.controls["prime_controls"]:
            self.assertGreater(row["positive_target_count"], 0)
            self.assertGreater(row["empty_target_count"], 0)

    def test_nontrivial_prime_power_maps_are_injective_and_exact(
        self,
    ) -> None:
        self.assertTrue(
            self.controls[
                "all_prime_nontrivial_power_maps_injective_and_exact"
            ]
        )
        self.assertTrue(
            self.controls["all_prime_intermediate_image_counts_zero"]
        )

    def test_trivial_prime_power_maps_fail_empty_targets(self) -> None:
        self.assertTrue(
            self.controls[
                "all_prime_trivial_power_maps_fail_empty_targets"
            ]
        )
        for control in self.controls["prime_controls"]:
            trivial = [
                row for row in control["maps"] if row["is_trivial"]
            ]
            self.assertEqual(len(trivial), 1)
            self.assertGreater(trivial[0]["false_positive_count"], 0)
            self.assertEqual(trivial[0]["false_negative_count"], 0)

    def test_composite_controls_expose_proper_quotients(self) -> None:
        self.assertEqual(self.controls["composite_control_count"], 3)
        self.assertTrue(
            self.controls[
                "all_composite_controls_have_intermediate_images"
            ]
        )
        for row in self.controls["composite_controls"]:
            self.assertGreater(row["intermediate_image_size_count"], 0)

    def test_finite_tuple_of_maps_retains_dichotomy(self) -> None:
        self.assertTrue(self.controls["tuple_dichotomy_exact"])
        tuple_control = self.controls["tuple_control"]
        self.assertTrue(tuple_control["all_tuple_images_are_one_or_q"])
        self.assertTrue(
            tuple_control["all_nontrivial_tuples_injective_and_exact"]
        )
        self.assertTrue(tuple_control["all_trivial_tuples_inexact"])

    def test_exact_homomorphic_image_cost_is_B5(self) -> None:
        theorem = self.report["theorem"]
        self.assertEqual(
            theorem["minimum_exact_image_cardinality"],
            "q",
        )
        self.assertEqual(theorem["image_exponent_B"]["exact"], "5")
        self.assertFalse(
            theorem["inside_setup_cap_for_full_image_table"]
        )
        route = self.routes[
            "nontrivial_prime_order_homomorphic_fingerprint"
        ]
        self.assertTrue(route["injective"])
        self.assertFalse(route["inside_full_table_setup_cap"])

    def test_nonhomomorphic_adaptive_route_remains_open(self) -> None:
        route = self.routes["nonhomomorphic_or_adaptive_fingerprint"]
        self.assertFalse(route["scoped_lower_bound_proved"])
        self.assertFalse(route["exact_structure_constructed"])
        self.assertEqual(route["status"], "open")
        self.assertIn(
            "nonhomomorphic or adaptive target fingerprint",
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
            admission["prime_order_homomorphism_dichotomy_admitted"]
        )
        self.assertTrue(
            admission["scoped_homomorphic_fingerprint_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "nonhomomorphic or adaptive target fingerprint",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
