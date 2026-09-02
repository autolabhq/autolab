from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_s6_subset_incidence_mobius_probe_r76.py"
SPEC = importlib.util.spec_from_file_location("p1553_r76", MODULE_PATH)
assert SPEC and SPEC.loader
R76 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R76)


class S6SubsetIncidenceMobiusProbeTests(unittest.TestCase):
    def test_mobius_corrects_multiple_common_root_overcount(self) -> None:
        control = R76.multiple_common_root_control()
        self.assertEqual(control["direct_tuple_count"], 4)
        self.assertEqual(control["mobius_tuple_count"], 4)
        self.assertGreater(
            control["singleton_endpoint_incidence_count"],
            control["direct_tuple_count"],
        )

    def test_small_prefix_replays_blind_and_forced_sources(self) -> None:
        curve = dict(R76.R74.R72.CURVES[0])
        family = R76.probe_curve(curve, prefix_sizes=(4,))
        targets = family["prefixes"][0]["targets"]
        blind = next(
            row for row in targets if row["target_id"] == "blind_hash_target"
        )
        forced = next(
            row
            for row in targets
            if row["target_id"] == "forced_positive_target"
        )
        self.assertEqual(blind["mobius_tuple_count"], 0)
        self.assertTrue(blind["counts_match"])
        self.assertTrue(forced["counts_match"])
        self.assertTrue(forced["source_relation_verified"])

    def test_small_report_keeps_cost_and_nonclaim_boundary(self) -> None:
        report = R76.build_report(prefix_sizes=(4,))
        aggregate = report["aggregate"]
        self.assertTrue(
            aggregate["all_mobius_counts_match_direct_tuple_counts"]
        )
        self.assertTrue(aggregate["all_dyadic_child_counts_conserve"])
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertEqual(
            report["cost_ledger"][
                "prefix_occurrence_and_state_exponent_B"
            ],
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
