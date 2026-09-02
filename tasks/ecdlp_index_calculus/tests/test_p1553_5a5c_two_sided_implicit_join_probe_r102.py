from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "p1553_5a5c_two_sided_implicit_join_probe_r102.py"
SPEC = importlib.util.spec_from_file_location("p1553_r102_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load R102 probe")
R102 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R102)


class TwoSidedImplicitJoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R102.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["actual_controls"]
        cls.costs = cls.report["asymptotic_control"]

    def test_canonical_all_target_counts_and_sources_are_exact(self) -> None:
        self.assertTrue(self.controls["all_canonical_histograms_exact"])
        self.assertTrue(self.controls["all_canonical_source_keys_exact"])
        self.assertTrue(self.controls["all_first_sources_replay"])

    def test_fresh_query_samples_and_exceptional_targets_are_exact(
        self,
    ) -> None:
        self.assertTrue(self.controls["all_sampled_queries_exact"])
        self.assertTrue(self.controls["all_blind_queries_bottom"])
        self.assertTrue(self.controls["all_identity_queries_exact"])
        self.assertTrue(self.controls["all_repeated_targets_exact"])

    def test_canonical_split_is_best_direct_setup_eligible_split(
        self,
    ) -> None:
        split = self.costs["exhaustive_split_ledger"]
        self.assertEqual(split["eligible_split_count"], 15)
        self.assertEqual(split["best_state_exponent_B"]["exact"], "11/5")
        self.assertEqual(split["best_query_exponent_B"]["exact"], "14/5")
        self.assertFalse(split["best_query_inside_online_cap"])
        best_counts = {
            (row["stored_a_count"], row["stored_c_count"])
            for row in split["best_setup_eligible_splits"]
        }
        self.assertEqual(best_counts, {(1, 3), (4, 1)})

    def test_prefix_state_passes_but_suffix_query_misses_caps(self) -> None:
        join = self.costs["canonical_prefix_join"]
        self.assertTrue(join["inside_setup_cap"])
        self.assertFalse(join["inside_online_cap"])
        self.assertEqual(join["state_exponent_B"]["exact"], "11/5")
        self.assertEqual(join["query_exponent_B"]["exact"], "14/5")

    def test_density_one_partition_filter_cost_is_conserved(self) -> None:
        density = self.costs["representation_density"]
        filter_cost = self.costs["disjoint_filter_conservation"]
        self.assertEqual(
            density["ordered_factor_source_exponent_B"]["exact"],
            density["subgroup_order_exponent_B"]["exact"],
        )
        self.assertEqual(
            density["representation_surplus_exponent_B"]["exact"],
            "0",
        )
        self.assertEqual(
            filter_cost["required_ideal_pruning_exponent_B"]["exact"],
            "31/20",
        )
        self.assertEqual(
            filter_cost["constant_success_repetition_exponent_B"]["exact"],
            "31/20",
        )
        self.assertEqual(
            filter_cost["restored_total_work_exponent_B"]["exact"],
            "14/5",
        )
        self.assertTrue(
            self.controls["all_partition_incidence_sums_exact"]
        )
        self.assertTrue(self.controls["all_partition_unions_complete"])

    def test_synthetic_joint_collision_multiplicity_is_exact(self) -> None:
        control = self.controls["cyclic_collision_control"]
        self.assertTrue(control["duplicate_atom_values_present"])
        self.assertTrue(control["all_target_counts_exact"])
        self.assertTrue(control["all_positive_sources_replay"])
        self.assertTrue(control["identity_target_positive"])
        self.assertTrue(control["multiplicity_above_one"])

    def test_bundle_closes_only_direct_and_thinning_filters(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 24)
        self.assertEqual(admission["obligation_count"], 34)
        self.assertFalse(admission["lane_admitted"])
        self.assertTrue(
            self.report["exact_overcap_join_baseline_admitted"]
        )
        self.assertFalse(self.report["factor_log_solve_complete"])
        self.assertFalse(self.report["fresh_target_descent_complete"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "target-forced Semaev, FFE",
            self.report["scope_boundary"],
        )


if __name__ == "__main__":
    unittest.main()
