from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_5a5c_fixed_marker_scalar_recurrence_probe_r89.py"
SPEC = importlib.util.spec_from_file_location("p1553_r89", MODULE_PATH)
assert SPEC and SPEC.loader
R89 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R89)


class FixedMarkerScalarRecurrenceTests(unittest.TestCase):
    def test_equal_size_nonzero_fixed_marker_collision(self) -> None:
        witness = R89.fixed_marker_collision_witness()
        self.assertEqual(witness["deck_sizes"], [2, 2, 2, 2, 2])
        self.assertTrue(witness["local_jets_equal"])
        self.assertTrue(witness["local_norm_nonzero"])
        self.assertTrue(witness["translated_jets_differ"])

    def test_collision_changes_multiplicity_branch(self) -> None:
        controls = R89.exceptional_controls()
        self.assertTrue(controls["nonzero_local_state_collision"])
        self.assertTrue(controls["one_translation_is_simple"])
        self.assertTrue(
            controls["other_translation_is_multiple_or_nonreduced"]
        )
        self.assertTrue(
            controls["local_state_cannot_predict_multiplicity_branch"]
        )

    def test_all_fixed_marker_translation_channels_have_full_rank(self) -> None:
        control = R89.translation_orbit_controls()
        self.assertEqual(control["norm_polynomial_degree"], 32)
        self.assertEqual(control["marker_polynomial_degrees"], [31] * 5)
        self.assertEqual(control["norm_translation_remainder_rank"], 16)
        self.assertEqual(
            control["marker_translation_remainder_ranks"],
            [16] * 5,
        )
        self.assertTrue(control["all_channels_full_quotient_rank"])

    def test_prefix_support_controls_are_exact(self) -> None:
        controls = [
            R89.prefix_support_control(deck_size)
            for deck_size in (2, 3, 4)
        ]
        self.assertTrue(controls[0]["collision_free_every_prefix"])
        self.assertTrue(controls[1]["collision_free_every_prefix"])
        self.assertFalse(controls[2]["collision_free_every_prefix"])
        self.assertGreater(
            controls[2]["minimum_prefix_occupancy_fraction"],
            0.99,
        )
        self.assertEqual(
            controls[-1]["prefix_distinct_support_sizes"],
            [4, 16, 64, 255, 1020],
        )

    def test_bundle_preserves_nonlocal_translation_sketch(self) -> None:
        bundle = R89.build_bundle()
        report = bundle["report"]
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertIn("nonlocal nonlinear", report["next_action"])
        self.assertFalse(report["factor_log_solve_complete"])
        self.assertFalse(report["fresh_target_descent_complete"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
