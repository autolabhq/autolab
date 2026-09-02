from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_5a5c_gauge_normalized_endpoint_query2p1_probe_r112.py"
SPEC = importlib.util.spec_from_file_location("p1553_r112", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R112 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R112)


class GaugeNormalizedEndpointQuery2P1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R112.build_bundle()
        cls.report = cls.bundle["report"]
        cls.replay = cls.bundle["source_replay"]
        cls.rows = [
            *cls.replay["actual"],
            *cls.replay["matched_random_decks"],
        ]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R112.verify_source_bindings()), 10)

    def test_typed_endpoint_queries_replay_r111(self) -> None:
        self.assertTrue(
            self.replay["summary"]["all_direct_queries_match_r111"]
        )
        self.assertTrue(
            self.replay["summary"]["all_pair_triple_queries_match_r111"]
        )
        self.assertTrue(
            all(
                row["projective_key_exact_on_all_endpoints"]
                for row in self.rows
            )
        )

    def test_canonical_pair_triple_join_is_one_to_one_with_c5(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["canonical_pair_triple_query_traffic"],
                row["c5_source_count"],
            )
            self.assertEqual(
                row["direct_query_c5_traffic"],
                row["c5_source_count"],
            )

    def test_thin_tables_fit_setup_but_query_misses_online_cap(self) -> None:
        ledger = self.report["cost_ledger"]
        self.assertTrue(
            ledger["target_independent_setup"]["inside_setup_cap"]
        )
        self.assertEqual(
            ledger["target_independent_setup"][
                "combined_state_exponent_B"
            ]["exact"],
            "2",
        )
        self.assertFalse(
            ledger["fresh_target_query"]["inside_online_cap"]
        )
        self.assertEqual(
            ledger["fresh_target_query"][
                "canonical_c2_c3_join_exponent_B"
            ]["exact"],
            "3",
        )

    def test_dyadic_recovery_and_no_relation_controls_are_exact(self) -> None:
        self.assertTrue(
            self.replay["summary"]["all_dyadic_recoveries_exact"]
        )
        self.assertTrue(
            self.replay["summary"]["all_no_relation_controls_exact"]
        )
        self.assertLessEqual(
            self.replay["summary"]["maximum_dyadic_inspection_ratio"],
            2.0,
        )
        self.assertEqual(
            self.replay["summary"]["maximum_no_relation_step_multiplier"],
            1,
        )

    def test_double_fibers_markers_and_weight_replay(self) -> None:
        doubles = [
            row
            for row in self.replay["actual"]
            if row["target_class"] == "r105_actual_double_fiber"
        ]
        self.assertEqual(len(doubles), 2)
        self.assertTrue(all(row["source_count"] == 2 for row in doubles))
        self.assertTrue(
            all(
                source["canonical_cycle_weight"]
                == R112.R108.FULL_CYCLE_SCALE
                and len(source["marker"]) == R112.R105.MARKER_DIMENSION
                for row in self.rows
                for source in row["sources"]
            )
        )

    def test_scoped_negative_preserves_nonlinear_orbit_exception(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(admission["typed_endpoint_normalization_admitted"])
        self.assertTrue(admission["scoped_negative_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "nonlinear elliptic orbit-product recurrence",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
