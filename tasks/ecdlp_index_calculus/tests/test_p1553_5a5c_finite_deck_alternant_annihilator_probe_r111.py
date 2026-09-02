from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_5a5c_finite_deck_alternant_annihilator_probe_r111.py"
SPEC = importlib.util.spec_from_file_location("p1553_r111", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R111 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R111)


class FiniteDeckAlternantAnnihilatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R111.build_bundle()
        cls.report = cls.bundle["report"]
        cls.replay = cls.bundle["source_replay"]
        cls.rows = [
            *cls.replay["actual"],
            *cls.replay["matched_random_decks"],
        ]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R111.verify_source_bindings()), 10)

    def test_every_canonical_source_is_enumerated(self) -> None:
        expected = 0
        for row in self.rows:
            count_a = math.comb(
                round(row["factor_base_size_B"] ** (2 / 5)) + 4,
                5,
            )
            count_c = math.comb(
                round(row["factor_base_size_B"] ** (3 / 5)) + 4,
                5,
            )
            self.assertEqual(row["a_source_count"], count_a)
            self.assertEqual(row["c_source_count"], count_c)
            expected += count_a * count_c
        self.assertEqual(expected, 135744)
        self.assertEqual(
            self.replay["summary"]["full_source_count"],
            expected,
        )

    def test_exterior_pairing_and_zero_biconditional_are_exact(self) -> None:
        self.assertTrue(
            self.replay["summary"]["all_exterior_laplace_checks_exact"]
        )
        self.assertTrue(
            self.replay["summary"][
                "all_full_body_zero_biconditionals_exact"
            ]
        )
        self.assertTrue(
            all(
                check["direct_determinant"] == check["exterior_pairing"]
                for row in self.rows
                for check in row["laplace_checks"]
            )
        )

    def test_raw_value_mask_degree_is_exact_and_dense(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["minimum_exact_raw_value_zero_mask_degree"],
                row["distinct_nonzero_determinant_value_count"],
            )
            self.assertGreater(
                row["distinct_nonzero_value_ratio"],
                0.98,
            )
        self.assertFalse(
            self.report["cost_ledger"]["raw_value_zero_mask"][
                "global_asymptotic_degree_claimed"
            ]
        )

    def test_low_rank_zero_mask_is_not_given_constructor_credit(self) -> None:
        self.assertEqual(
            self.replay["summary"]["maximum_zero_mask_rank"],
            2,
        )
        zero_incidence = self.report["cost_ledger"]["zero_incidence"]
        self.assertFalse(
            zero_incidence[
                "low_rank_mask_is_available_without_constructing_mask"
            ]
        )
        self.assertFalse(
            zero_incidence["subset_stable_existence_oracle_supplied"]
        )

    def test_double_fibers_markers_and_weight_replay(self) -> None:
        doubles = [
            row
            for row in self.replay["actual"]
            if row["target_class"] == "r105_actual_double_fiber"
        ]
        self.assertEqual(len(doubles), 2)
        self.assertTrue(
            all(row["zero_source_count"] == 2 for row in doubles)
        )
        self.assertTrue(
            all(
                source["canonical_cycle_weight"]
                == R111.R108.FULL_CYCLE_SCALE
                and len(source["marker"]) == R111.R105.MARKER_DIMENSION
                for row in self.rows
                for source in row["zero_sources"]
            )
        )

    def test_scoped_negative_does_not_admit_algorithm_lane(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(admission["scoped_negative_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "gauge-normalized endpoint Query2P1",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
