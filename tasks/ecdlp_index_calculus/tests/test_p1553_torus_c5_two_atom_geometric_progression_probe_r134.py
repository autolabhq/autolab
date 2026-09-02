from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_two_atom_geometric_progression_probe_r134.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r134", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R134 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R134)


class TorusC5TwoAtomGeometricProgressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R134.build_bundle()
        cls.report = cls.bundle["report"]
        cls.theorem = cls.report["theorem"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.bundle["cost"]["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R134.verify_source_bindings()), 10)

    def test_progression_sources_cover_all_degree_five_mixtures(self) -> None:
        sources = R134.progression_sources(2, 6)
        self.assertEqual(len(sources), 6)
        self.assertEqual(sources[0], (2, 2, 2, 2, 2))
        self.assertEqual(sources[-1], (6, 6, 6, 6, 6))
        self.assertTrue(all(len(source) == 5 for source in sources))

    def test_theorem_excludes_through_six_modes(self) -> None:
        self.assertEqual(self.theorem["source_degree"], 5)
        self.assertEqual(self.theorem["progression_length"], 6)
        self.assertEqual(self.theorem["maximum_excluded_mode_count"], 6)
        self.assertEqual(self.theorem["first_mode_count_not_excluded"], 7)
        self.assertTrue(self.theorem["field_independent"])
        self.assertFalse(self.theorem["random_deck_model_required"])

    def test_actual_progression_witness_count(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertEqual(self.controls["active_color_control_count"], 30)
        self.assertEqual(
            self.controls["available_progression_witness_count"],
            12,
        )
        self.assertFalse(
            self.controls["all_active_finite_colors_have_witness"]
        )

    def test_actual_progressions_and_prime_order_ratios_are_exact(self) -> None:
        self.assertTrue(
            self.controls["all_available_ratios_have_exact_prime_order"]
        )
        self.assertTrue(self.controls["all_available_targets_distinct"])
        self.assertTrue(self.controls["all_available_progressions_exact"])

    def test_actual_sources_replay_and_are_color_accepted(self) -> None:
        self.assertTrue(self.controls["all_available_sources_replay"])
        self.assertTrue(self.controls["all_available_sources_accepted"])
        for control in self.controls["controls"]:
            for witness in control["color_witnesses"]:
                if not witness["witness_available"]:
                    continue
                self.assertEqual(witness["source_count"], 6)
                self.assertEqual(witness["target_count"], 6)
                self.assertTrue(
                    witness["all_sources_have_color_multiplicity_five"]
                )

    def test_sample_vandermonde_determinants_are_nonzero(self) -> None:
        self.assertTrue(
            self.controls[
                "all_available_sample_vandermonde_determinants_nonzero"
            ]
        )

    def test_single_zero_and_pole_routes_are_rejected(self) -> None:
        self.assertEqual(
            self.routes[
                "one_to_six_mode_single_color_zero_predicate"
            ]["status"],
            "rejected_deterministically_by_two_atom_progression",
        )
        self.assertEqual(
            self.routes[
                "one_to_six_mode_single_color_pole_predicate"
            ]["status"],
            "rejected_deterministically_by_two_atom_progression",
        )

    def test_surviving_selector_classes_remain_open(self) -> None:
        for route_id in (
            "seven_or_more_mode_extension_zero_predicate",
            "multiple_small_mode_predicate_dag",
            "low_slp_expanded_extension_predicate",
            "nonzero_value_frobenius_coordinate_dag",
        ):
            self.assertEqual(self.routes[route_id]["status"], "open")

    def test_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 13)
        self.assertEqual(admission["obligation_count"], 21)
        self.assertTrue(
            admission["deterministic_one_to_six_mode_negative_admitted"]
        )
        self.assertTrue(
            admission["structured_factor_base_progression_theorem_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("seven-or-more-mode", self.report["next_action"])


if __name__ == "__main__":
    unittest.main()
