from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_5a5c_nonlinear_elliptic_orbit_product_probe_r113.py"
SPEC = importlib.util.spec_from_file_location("p1553_r113", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R113 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R113)


class NonlinearEllipticOrbitProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R113.build_bundle()
        cls.report = cls.bundle["report"]
        cls.replay = cls.bundle["source_replay"]
        cls.rows = [
            *cls.replay["actual"],
            *cls.replay["matched_random_decks"],
        ]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R113.verify_source_bindings()), 10)

    def test_product_roots_and_sources_replay_r112(self) -> None:
        self.assertTrue(
            self.replay["summary"]["all_product_roots_exact"]
        )
        self.assertTrue(
            self.replay["summary"]["all_sources_match_r112"]
        )
        self.assertEqual(
            self.replay["summary"]["source_count"],
            18,
        )

    def test_c3_sets_are_not_fixed_translation_orbits(self) -> None:
        self.assertTrue(
            self.replay["summary"][
                "all_c3_translation_stabilizers_trivial"
            ]
        )
        self.assertTrue(
            self.replay["summary"]["all_c3_orderings_nonorbit"]
        )
        for row in self.rows:
            self.assertTrue(
                row["c3_support_is_nonempty_proper_subset_of_prime_group"]
            )
            self.assertEqual(
                row["nonzero_translation_orbit_length"],
                row["subgroup_order"],
            )

    def test_bounded_prefix_state_does_not_reduce_leaf_work(self) -> None:
        ledger = self.report["cost_ledger"]["nonlinear_product_tree"]
        self.assertEqual(ledger["prefix_recurrence_order"], 1)
        self.assertEqual(ledger["prefix_state_dimension"], 1)
        self.assertEqual(
            ledger["leaf_generator_exponent_B"]["exact"],
            "3",
        )
        self.assertFalse(
            ledger["bounded_product_state_reduces_leaf_work"]
        )
        self.assertFalse(ledger["leaf_generation_inside_online_cap"])

    def test_target_independent_compilation_restores_source_body_state(
        self,
    ) -> None:
        compiled = self.report["cost_ledger"][
            "target_independent_compilation"
        ]
        self.assertEqual(
            compiled["endpoint_dictionary_state_exponent_B"]["exact"],
            "5",
        )
        self.assertGreater(
            compiled["minimum_observed_support_ratio"],
            0.99,
        )
        self.assertFalse(compiled["inside_setup_cap"])

    def test_markers_and_weight_replay(self) -> None:
        self.assertTrue(
            all(
                source["canonical_cycle_weight"]
                == R113.R108.FULL_CYCLE_SCALE
                and len(source["marker"]) == R113.R105.MARKER_DIMENSION
                for row in self.rows
                for source in row["sources"]
            )
        )

    def test_scoped_negative_preserves_transposed_leaf_generator(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(admission["scoped_negative_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "transposed nonuniform C5 leaf generator",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
