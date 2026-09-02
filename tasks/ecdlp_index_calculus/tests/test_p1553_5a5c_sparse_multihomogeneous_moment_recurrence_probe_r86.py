from __future__ import annotations

import collections
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "p1553_5a5c_sparse_multihomogeneous_moment_recurrence_probe_r86.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r86", MODULE_PATH)
assert SPEC and SPEC.loader
R86 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R86)


class SparseMultihomogeneousMomentRecurrenceTests(unittest.TestCase):
    def test_coloring_is_complete_and_rectangular(self) -> None:
        colors = R86.colored_factor_indices(28)
        self.assertEqual(sorted(index for color in colors for index in color), list(range(28)))
        rectangles = R86.rectangular_color_description(4, 7)
        self.assertTrue(all(row["rectangle_count"] <= 5 for row in rectangles))

    def test_synthetic_nonreduced_first_jet_rejects(self) -> None:
        control = R86.synthetic_nonreduced_control(101)
        self.assertTrue(control["norm_and_first_jet_vanish"])

    def test_absent_multiple_fiber_is_vacuous_not_failed(self) -> None:
        control = R86.multiple_norm_control(collections.Counter({1: 1, 2: 1}))
        self.assertFalse(control["multiple_fiber_present"])
        self.assertFalse(control["all_first_derivatives_zero"])
        self.assertTrue(control["branch_exact_or_vacuous"])

    def test_constructor_ledger_exposes_cubic_atom_side(self) -> None:
        ledger = R86.multihomogeneous_constructor_ledger()
        refinement = ledger["r82_addition_pushforward_refinement"]
        self.assertEqual(
            refinement["five_colored_a_choices_exponent_B"],
            2.0,
        )
        self.assertEqual(refinement["five_c_choices_exponent_B"], 3.0)
        self.assertFalse(refinement["larger_side_inside_setup_cap"])
        self.assertFalse(
            ledger["standard_sparse_multihomogeneous_constructor_inside_caps"]
        )

    def test_small_bundle_recovers_supplied_jet_but_withholds_credit(self) -> None:
        bundle = R86.build_bundle(
            families=R86.R82.FAMILIES[:1],
            offsets=(0,),
        )
        report = bundle["report"]
        self.assertTrue(
            report["aggregate"]["all_supplied_simple_jets_recover_source"]
        )
        self.assertTrue(report["aggregate"]["all_empty_fibers_detected"])
        self.assertFalse(
            bundle["replay"]["public_input_moment_constructor_inside_caps"]
        )
        self.assertIn("intertwiner", report["next_action"])
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["factor_log_solve_complete"])
        self.assertFalse(report["fresh_target_descent_complete"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
