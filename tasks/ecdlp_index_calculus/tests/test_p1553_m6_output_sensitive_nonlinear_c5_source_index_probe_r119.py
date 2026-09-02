from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_m6_output_sensitive_nonlinear_c5_source_index_probe_r119.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r119", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R119 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R119)


class M6OutputSensitiveNonlinearC5SourceIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R119.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.replay = cls.bundle["replay"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R119.verify_source_bindings()), 12)

    def test_iid_collision_theorem_has_exact_finite_control(self) -> None:
        control = self.controls["exact_iid_collision_control"]
        self.assertEqual(control["group_order"], 11)
        self.assertEqual(control["deck_assignments_enumerated"], 1331)
        self.assertEqual(control["canonical_source_count"], 21)
        self.assertEqual(control["distinct_source_pair_count"], 210)
        self.assertEqual(control["observed_total_collision_pairs"], 25410)
        self.assertEqual(control["expected_total_collision_pairs"], 25410)
        self.assertTrue(control["total_matches_pairwise_uniform_theorem"])
        self.assertTrue(
            control["support_deficiency_bounded_on_every_assignment"]
        )
        self.assertFalse(
            control["finite_enumeration_receives_asymptotic_credit"]
        )

    def test_eight_actual_hash_decks_are_injective_and_replay(self) -> None:
        self.assertEqual(self.controls["actual_control_count"], 8)
        self.assertTrue(
            self.controls["all_actual_c5_endpoint_maps_injective"]
        )
        self.assertTrue(self.controls["all_actual_sources_replay"])
        self.assertTrue(
            self.controls["all_actual_empty_targets_rejected"]
        )
        for row in self.controls["actual_r82_controls"]:
            self.assertEqual(
                row["canonical_c5_source_count"],
                row["expected_canonical_source_count"],
            )
            self.assertEqual(row["support_deficiency"], 0)
            self.assertEqual(row["endpoint_collision_pair_count"], 0)
            self.assertTrue(row["all_selected_sources_replay"])
            self.assertTrue(row["empty_target_rejected_exactly"])
            self.assertFalse(row["candidate_scalar_labels_consumed"])

    def test_random_deck_support_exponents_are_exact(self) -> None:
        theorem = self.cost["random_deck_support_theorem"]
        campaign = theorem["campaign_substitution"]
        self.assertEqual(campaign["M_exponent_B"]["exact"], "15/4")
        self.assertEqual(
            campaign["expected_collision_pair_exponent_B"]["exact"],
            "5/2",
        )
        self.assertEqual(campaign["M_over_q_exponent_B"]["exact"], "-5/4")
        self.assertEqual(
            theorem["pair_collision_probability"]["value"],
            "1/q",
        )
        self.assertIn(
            "(1-o(1))M",
            campaign["conclusion"],
        )

    def test_random_model_scope_does_not_transfer_to_filtered_deck(
        self,
    ) -> None:
        theorem = self.cost["random_deck_support_theorem"]
        self.assertFalse(
            theorem["transfer_to_filtered_r82_hash_deck_claimed"]
        )
        self.assertFalse(
            theorem["generic_deterministic_deck_theorem_claimed"]
        )
        self.assertTrue(self.frozen["random_model_only"])
        self.assertFalse(
            self.controls["finite_enumeration_receives_asymptotic_credit"]
        )

    def test_output_linear_routes_exceed_setup_cap(self) -> None:
        routes = {row["route_id"]: row for row in self.cost["routes"]}
        for route_id in (
            "explicit_endpoint_membership_hash_and_source",
            "radical_endpoint_polynomial_and_source_selector",
            "output_linear_radical_image_or_trace_compiler",
        ):
            self.assertFalse(routes[route_id]["inside_setup_cap"])
        self.assertFalse(
            self.cost["any_output_linear_route_meets_setup_cap"]
        )
        self.assertEqual(
            routes["universal_characteristic_zero_linear_shift_index"][
                "state_exponent_B"
            ]["exact"],
            "5",
        )
        self.assertEqual(
            routes["current_dinur_golovnev_k6_zero_query_index"][
                "state_exponent_B"
            ]["exact"],
            "33/8",
        )

    def test_known_lower_bound_boundary_preserves_implicit_index(
        self,
    ) -> None:
        boundary = self.cost["known_data_structure_lower_bound_boundary"]
        self.assertFalse(
            boundary["polylog_query_with_n3_space_unconditionally_excluded"]
        )
        self.assertFalse(boundary["cell_probe_or_ram_lower_bound_claimed"])
        self.assertFalse(
            self.cost["general_arithmetic_circuit_lower_bound_claimed"]
        )
        self.assertFalse(
            self.replay["inside_cap_implicit_membership_circuit_constructed"]
        )
        self.assertFalse(
            self.replay["inside_cap_implicit_source_recovery_constructed"]
        )
        self.assertIn(
            "sub-output implicit nonlinear C5",
            self.cost["preserved_interface"],
        )

    def test_scoped_theorem_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 10)
        self.assertEqual(admission["obligation_count"], 17)
        self.assertTrue(admission["random_model_theorem_admitted"])
        self.assertTrue(admission["scoped_negative_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "sub-output implicit nonlinear C5",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
