from __future__ import annotations

import importlib.util
import pathlib
import unittest
from fractions import Fraction


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "p1553_5a5c_unequal_list_subfunction_inversion_probe_r91.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r91", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R91 probe")
R91 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R91)


class UnequalListSubfunctionInversionTests(unittest.TestCase):
    def test_intended_five_a_five_c_cap_interval_is_empty(self) -> None:
        control = R91.intended_five_a_five_c_application()
        self.assertFalse(control["cap_interval_nonempty"])
        self.assertEqual(
            control["delta_required_by_setup_cap"]["exact"],
            "15/8",
        )
        self.assertEqual(
            control["delta_allowed_by_online_cap"]["exact"],
            "5/8",
        )
        self.assertEqual(
            control["best_point_under_online_cap"][
                "space_exponent_B"
            ]["exact"],
            "19/4",
        )

    def test_every_ten_deck_partition_misses_setup_cap(self) -> None:
        control = R91.all_partition_cap_control()
        self.assertEqual(control["unique_nonempty_partition_count"], 11)
        self.assertFalse(control["any_partition_meets_both_caps"])
        self.assertFalse(
            control["any_partition_auxiliary_list_inside_setup_cap"]
        )
        self.assertEqual(
            control["minimum_space_exponent_B"]["exact"],
            "22/5",
        )

    def test_balanced_five_sum_index_also_misses_setup_cap(self) -> None:
        control = R91.balanced_factor_base_ksum_control()
        self.assertEqual(control["paper_k_parameter"], 6)
        self.assertEqual(control["space_exponent_B"]["exact"], "9/2")
        self.assertFalse(control["setup_cap_satisfied"])
        self.assertTrue(control["online_cap_satisfied"])

    def test_integer_subfunction_control_reports_exact_sources(self) -> None:
        replay = R91.integer_subfunction_source_replay()
        self.assertTrue(
            replay["all_present_targets_report_exact_source"]
        )
        self.assertTrue(
            replay["absent_target_rejected_after_exact_verification"]
        )

    def test_prime_group_control_has_no_proper_additive_filter(self) -> None:
        control = R91.prime_order_transfer_control()
        self.assertTrue(
            control["all_proper_homomorphic_filters_trivial"]
        )
        self.assertFalse(control["generic_prime_group_transfer_complete"])
        self.assertFalse(control["dlog_labels_available_to_candidate"])

    def test_fractional_theorem_formula_matches_direct_substitution(self) -> None:
        exponents = R91.unequal_list_exponents(
            Fraction(2),
            Fraction(3),
            Fraction(5, 8),
        )
        self.assertEqual(exponents["space"], Fraction(19, 4))
        self.assertEqual(exponents["query"], Fraction(5, 4))

    def test_bundle_preserves_compact_elliptic_subfunction_map(self) -> None:
        report = R91.build_bundle()["report"]
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertIn("compact elliptic", report["scope_boundary"])
        self.assertIn("public MAP1, MAP2, f_d, and TR", report["next_action"])


if __name__ == "__main__":
    unittest.main()
