from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_m6_a6_batched_c3_pair_sum_source_locator_probe_r116.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r116", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R116 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R116)


class M6A6BatchedC3PairSumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R116.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.replay = cls.bundle["replay"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R116.verify_source_bindings()), 12)

    def test_exact_reduction_exponents(self) -> None:
        interface = self.frozen["exact_interface"]
        self.assertEqual(
            interface["preprocessed_occurrence_exponent_B"]["exact"],
            "9/4",
        )
        self.assertEqual(
            interface["fresh_target_batch_exponent_B"]["exact"],
            "1/2",
        )
        self.assertEqual(
            interface["average_query_allowance_exponent_B"]["exact"],
            "3/4",
        )
        self.assertEqual(
            interface["requested_source"],
            "one A6 and two C3 occurrence backpointers",
        )

    def test_finite_count_identity_and_empty_branch(self) -> None:
        self.assertTrue(self.replay["all_instances_exact"])
        self.assertTrue(self.replay["negative_control_present"])
        controls = [
            control
            for instance in self.replay["instances"]
            for control in instance["target_controls"]
        ]
        self.assertTrue(any(control["positive"] for control in controls))
        self.assertTrue(any(not control["positive"] for control in controls))
        for control in controls:
            self.assertTrue(control["all_three_counts_equal"])
            self.assertTrue(control["source_presence_exact"])

    def test_positive_sources_and_s3_chains_replay(self) -> None:
        positives = [
            control
            for instance in self.replay["instances"]
            for control in instance["target_controls"]
            if control["positive"]
        ]
        self.assertTrue(positives)
        for control in positives:
            self.assertTrue(control["direct_source_replay_exact"])
            self.assertTrue(control["batched_atom_source_replay_exact"])
            self.assertTrue(control["batched_factor_source_replay_exact"])
            self.assertTrue(
                control["self_convolution_source_replay_exact"]
            )
            self.assertEqual(
                len(control["batched_source"]["a6_indices"]),
                6,
            )
            self.assertEqual(
                len(control["batched_source"]["left_c3_indices"]),
                3,
            )
            self.assertEqual(
                len(control["batched_source"]["right_c3_indices"]),
                3,
            )
            self.assertTrue(
                control["factor_s3_chain"]["all_nonprojective_s3_zero"]
            )

    def test_dinur_golovnev_boundary_misses_setup(self) -> None:
        indexing = self.cost["indexing"]
        theorem = indexing["dinur_golovnev_v2"]
        self.assertEqual(
            theorem["batch_cap_forces_delta_at_most"]["exact"],
            "1/3",
        )
        self.assertEqual(
            theorem["setup_cap_would_require_delta_at_least"]["exact"],
            "3/2",
        )
        self.assertEqual(
            theorem["online_compatible_setup_exponent_B"]["exact"],
            "39/8",
        )
        self.assertFalse(
            indexing["any_bound_indexing_route_meets_both_caps"]
        )
        self.assertFalse(
            indexing["prime_order_transfer_boundary"][
                "public_ec_coordinate_encoding_is_addition_compatible"
            ]
        )

    def test_2026_preprocessed_universe_route_is_charged(self) -> None:
        routes = {
            row["route_id"]: row
            for row in self.cost["indexing"]["routes"]
        }
        best_space = routes[
            "preprocessed_unknown_universe_2026_best_space"
        ]
        best_query = routes[
            "preprocessed_unknown_universe_2026_best_query"
        ]
        self.assertEqual(
            best_space["setup_exponent_B"]["exact"],
            "15/4",
        )
        self.assertEqual(
            best_query["query_exponent_B"]["exact"],
            "27/8",
        )
        self.assertEqual(
            best_space["preprocessing_work_exponent_B"]["exact"],
            "9/2",
        )
        self.assertFalse(best_space["model_matches_point_challenge"])

    def test_standard_algebraic_route_degrees_are_charged(self) -> None:
        algebraic = self.cost["algebraic"]
        endpoints = algebraic["endpoint_bodies"]
        self.assertEqual(
            endpoints["c3_pair_occurrence_exponent_B"]["exact"],
            "9/2",
        )
        self.assertEqual(
            endpoints["full_source_occurrence_exponent_B"]["exact"],
            "5",
        )
        degrees = algebraic["summation_polynomial_degrees"]
        self.assertEqual(
            degrees["factor_level_s7"]["degree_per_variable"],
            32,
        )
        self.assertEqual(
            degrees["expanded_atom_level_s13"]["degree_per_variable"],
            2048,
        )
        self.assertFalse(algebraic["unconditional_lower_bound_claimed"])

    def test_exact_interface_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(admission["exact_interface_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(
            admission["obligations"][
                "implicit_target_batched_coefficient_functional_complete"
            ]
        )
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "target-batched elliptic coefficient functional",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
