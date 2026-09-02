from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_explicit_hash_correction_support_probe_r126.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r126", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R126 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R126)


class TorusC5ExplicitHashCorrectionSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R126.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R126.verify_source_bindings()), 12)

    def test_coordinate_hash_is_deterministic(self) -> None:
        point = (17, 29)
        self.assertEqual(R126.coordinate_hash(point, 5, 3), 4)
        self.assertEqual(R126.coordinate_hash(point, 5, 3), 4)
        with self.assertRaises(ValueError):
            R126.coordinate_hash(point, 0, 3)

    def test_all_twenty_four_actual_controls_complete(self) -> None:
        self.assertEqual(self.controls["control_count"], 24)
        self.assertTrue(
            self.controls[
                "all_hashes_use_field_coordinates_without_dlp"
            ]
        )
        for row in self.controls["controls"]:
            self.assertIn(row["hash_modulus"], (2, 3, 5))
            self.assertFalse(row["candidate_discrete_logs_consumed"])

    def test_correction_union_equals_exact_c5_support(self) -> None:
        self.assertTrue(
            self.controls["all_correction_unions_equal_c5_support"]
        )
        for row in self.controls["controls"]:
            self.assertTrue(row["correction_union_equals_c5_support"])
            self.assertEqual(
                row["c5_source_count"],
                row["distinct_c5_product_count"],
            )

    def test_explicit_correction_entries_cover_at_least_c5(self) -> None:
        self.assertTrue(
            self.controls[
                "all_correction_lists_cover_at_least_c5_support"
            ]
        )
        self.assertGreaterEqual(
            self.controls["minimum_correction_duplication_ratio"],
            1.0,
        )
        for row in self.controls["controls"]:
            self.assertGreaterEqual(
                row["explicit_correction_entry_count"],
                row["distinct_c5_product_count"],
            )

    def test_every_correction_entry_has_replaying_sources(self) -> None:
        self.assertTrue(
            self.controls[
                "all_source_backpointers_complete_and_exact"
            ]
        )
        for row in self.controls["controls"]:
            self.assertTrue(
                row["all_correction_entries_have_source_backpointers"]
            )
            self.assertTrue(row["all_source_backpointers_replay"])

    def test_explicit_correction_support_cost_is_B15O4(self) -> None:
        theorem = self.report["theorem"]
        self.assertIn("|C5|", theorem["entry_lower_bound"])
        self.assertEqual(
            theorem[
                "minimum_explicit_correction_state_exponent_B"
            ]["exact"],
            "15/4",
        )
        self.assertFalse(theorem["inside_setup_cap"])
        for route_id in (
            "explicit_per_bucket_pair_product_corrections",
            "global_deduplicated_product_dictionary",
        ):
            route = self.routes[route_id]
            self.assertTrue(route["scoped_lower_bound_proved"])
            self.assertFalse(route["inside_setup_cap"])

    def test_hash_without_corrections_is_not_admitted(self) -> None:
        route = self.routes["coordinate_hash_without_corrections"]
        self.assertFalse(route["exact_product_composition_proved"])
        self.assertFalse(route["admitted"])

    def test_implicit_adaptive_correction_route_remains_open(self) -> None:
        route = self.routes["implicit_or_adaptive_correction_circuit"]
        self.assertFalse(route["scoped_lower_bound_proved"])
        self.assertFalse(route["exact_structure_constructed"])
        self.assertEqual(route["status"], "open")
        self.assertIn(
            "implicit correction circuit",
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
            admission["coordinate_hash_correction_semantics_admitted"]
        )
        self.assertTrue(
            admission[
                "scoped_explicit_correction_support_negative_admitted"
            ]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "implicit correction circuit",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
