from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "p1553_5a5c_actual_divisor_image_entropy_merge_probe_r101.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r101", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R101 probe")
R101 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R101)


class ActualDivisorImageEntropyMergeTests(unittest.TestCase):
    def test_all_actual_side_counts_and_sources_are_exact(self) -> None:
        controls = R101.actual_image_controls()
        self.assertEqual(controls["instance_count"], 8)
        self.assertEqual(controls["side_query_count"], 16)
        self.assertTrue(controls["all_attained_counts_exact"])
        self.assertTrue(controls["all_attained_sources_replay"])

    def test_blind_identity_and_repeated_atom_paths_are_exact(
        self,
    ) -> None:
        controls = R101.actual_image_controls()
        self.assertTrue(controls["all_blind_targets_return_bottom"])
        self.assertTrue(controls["all_identity_target_queries_exact"])
        self.assertTrue(controls["all_repeated_atom_targets_replay"])

    def test_finite_state_and_scan_counts_match_combinatorics(
        self,
    ) -> None:
        for instance in R101.actual_image_controls()["instances"]:
            u = instance["atom_a_size"]
            v = instance["atom_c_size"]
            left = instance["left"]
            right = instance["right"]
            self.assertEqual(
                left["stored_c_multiset_occurrences"],
                R101.math.comb(v + 2, 3),
            )
            self.assertEqual(
                left["scanned_a_multisets_per_query"],
                R101.math.comb(u + 1, 2),
            )
            self.assertEqual(
                right["stored_c_multiset_occurrences"],
                R101.math.comb(v + 1, 2),
            )
            self.assertEqual(
                right["scanned_a_multisets_per_query"],
                R101.math.comb(u + 2, 3),
            )

    def test_synthetic_collision_multiplicity_is_exact(self) -> None:
        control = R101.cyclic_multiset_control()
        self.assertTrue(control["duplicate_atom_values_present"])
        self.assertTrue(control["all_target_counts_exact"])
        self.assertTrue(control["all_positive_sources_replay"])
        self.assertTrue(control["multiplicity_above_one"])
        self.assertTrue(control["identity_target_positive"])

    def test_both_local_oracles_fit_direct_caps(self) -> None:
        costs = R101.asymptotic_cost_control()
        right = costs["right_3A_plus_2C"]
        left = costs["left_2A_plus_3C"]
        self.assertEqual(
            right["stored_state_exponent_B"]["exact"], "6/5"
        )
        self.assertEqual(
            right["fresh_work_exponent_B"]["exact"], "6/5"
        )
        self.assertEqual(
            left["stored_state_exponent_B"]["exact"], "9/5"
        )
        self.assertEqual(
            left["fresh_work_exponent_B"]["exact"], "4/5"
        )
        self.assertTrue(right["inside_setup_cap"])
        self.assertTrue(right["inside_online_cap"])
        self.assertTrue(left["inside_setup_cap"])
        self.assertTrue(left["inside_online_cap"])

    def test_full_two_sided_join_remains_unsupplied(self) -> None:
        join = R101.asymptotic_cost_control()["full_two_sided_join"]
        self.assertFalse(join["supplied"])
        self.assertFalse(join["inside_online_cap"])
        self.assertEqual(
            join["enumerating_left_endpoints_exponent_B"]["exact"],
            "13/5",
        )
        self.assertEqual(
            join["enumerating_right_endpoints_exponent_B"]["exact"],
            "12/5",
        )

    def test_bundle_credits_local_positive_only(self) -> None:
        report = R101.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"], 19
        )
        self.assertEqual(report["admission"]["obligation_count"], 32)
        self.assertTrue(report["local_side_oracles_admitted"])
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertIn("real local positive", report["scope_boundary"])
        self.assertIn("two-sided implicit intersection", report["next_action"])


if __name__ == "__main__":
    unittest.main()
