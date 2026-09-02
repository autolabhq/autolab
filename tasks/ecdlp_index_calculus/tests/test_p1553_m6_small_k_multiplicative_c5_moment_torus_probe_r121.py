from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_r121.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r121", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R121 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R121)


class M6SmallKMultiplicativeC5MomentTorusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R121.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.replay = cls.bundle["replay"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R121.verify_source_bindings()), 14)

    def test_all_pairing_images_have_complete_cayley_chart(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertTrue(
            self.controls["all_pairing_images_norm_one_and_charted"]
        )
        for row in self.controls["r82_torus_moment_controls"]:
            self.assertTrue(row["all_pairing_images_norm_one"])
            self.assertTrue(
                row["minus_one_absent_from_odd_q_subgroup_deck"]
            )
            self.assertTrue(row["all_cayley_chart_roundtrips_exact"])
            self.assertFalse(row["candidate_scalar_labels_consumed"])

    def test_binary_law_and_degree_five_form_are_exact(self) -> None:
        self.assertTrue(
            self.controls["all_torus_laws_and_c5_forms_exact"]
        )
        for row in self.controls["r82_torus_moment_controls"]:
            self.assertTrue(row["all_binary_torus_laws_exact"])
            self.assertTrue(row["all_c5_torus_forms_exact"])
            self.assertTrue(row["all_c5_even_denominators_nonzero"])
        form = self.report["algebraic_identities"]["five_product_form"]
        self.assertEqual(form["degree"], 5)
        self.assertFalse(form["requires_discrete_logarithms"])

    def test_complete_homogeneous_moments_match_direct_products(
        self,
    ) -> None:
        self.assertTrue(
            self.controls["all_compact_moment_identities_exact"]
        )
        for row in self.controls["r82_torus_moment_controls"]:
            self.assertEqual(
                row["direct_moment_sha256"],
                row["compact_h5_moment_sha256"],
            )
            self.assertEqual(
                row["moment_prefix_length"],
                2 * row["canonical_c5_source_count"],
            )

    def test_bm_order_and_annihilator_equal_full_support(self) -> None:
        self.assertTrue(
            self.controls["all_bm_orders_equal_distinct_c5_support"]
        )
        for row in self.controls["r82_torus_moment_controls"]:
            self.assertEqual(
                row["berlekamp_massey_order"],
                row["distinct_product_count"],
            )
            self.assertEqual(
                row["recurrence_coefficient_sha256"],
                row["annihilator_coefficient_sha256"],
            )
            self.assertTrue(
                row["recurrence_equals_full_product_annihilator"]
            )

    def test_full_annihilator_membership_controls_are_exact(self) -> None:
        self.assertTrue(
            self.controls["all_annihilator_membership_controls_exact"]
        )
        for row in self.controls["r82_torus_moment_controls"]:
            self.assertTrue(
                row["all_positive_annihilator_evaluations_zero"]
            )
            self.assertTrue(row["empty_annihilator_evaluation_nonzero"])

    def test_c2_c3_split_returns_sources_and_rejects_empty(self) -> None:
        self.assertTrue(
            self.controls["all_c2_c3_sources_and_empty_queries_exact"]
        )
        for row in self.controls["r82_torus_moment_controls"]:
            self.assertTrue(row["all_c2_c3_split_sources_exact"])
            self.assertTrue(
                row["all_c2_c3_projective_sources_replay"]
            )
            self.assertTrue(row["empty_c2_c3_split_query_rejected"])

    def test_moment_and_split_routes_miss_frozen_caps(self) -> None:
        moment = self.routes[
            "complete_homogeneous_full_moment_recurrence"
        ]
        split = self.routes["c2_table_c3_table_split_query"]
        self.assertEqual(
            moment["represented_state_or_output_exponent_B"]["exact"],
            "15/4",
        )
        self.assertFalse(moment["inside_setup_cap"])
        self.assertEqual(split["c3_setup_exponent_B"]["exact"], "9/4")
        self.assertTrue(split["inside_setup_cap"])
        self.assertEqual(split["c2_query_exponent_B"]["exact"], "3/2")
        self.assertFalse(split["inside_polylog_query_cap"])

    def test_dinur_golovnev_routes_are_scoped_not_general_bounds(
        self,
    ) -> None:
        direct = self.routes[
            "dinur_golovnev_balanced_k6_index_delta_zero"
        ]
        template = self.routes[
            "dinur_golovnev_theorem4p1_subfunction_template"
        ]
        self.assertEqual(direct["state_exponent_B"]["exact"], "33/8")
        self.assertFalse(direct["inside_setup_cap"])
        self.assertFalse(
            direct["integer_addition_or_xor_theorem_directly_transferred"]
        )
        self.assertFalse(
            template["ambient_universe_meets_paper_requirement"]
        )
        self.assertEqual(
            template["granted_state_exponent_B"]["exact"],
            "15/4",
        )
        self.assertFalse(template["inside_setup_cap"])
        self.assertFalse(
            template["general_data_structure_lower_bound_claimed"]
        )

    def test_semantic_dedup_preserves_nonlinear_torus_route(self) -> None:
        dedup = self.cost["semantic_dedup"]
        self.assertIn(
            "R90 additive exponential-moment Hankel translation",
            dedup["nearby_lanes"],
        )
        self.assertIn("norm-one torus", dedup["r121_distinct_scope"])
        self.assertFalse(dedup["new_idea_id_claimed"])
        self.assertIn(
            "target-specialized nonlinear norm-one-torus C5",
            self.frozen["preserved_interface"],
        )

    def test_scoped_controls_do_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 13)
        self.assertEqual(admission["obligation_count"], 20)
        self.assertTrue(admission["torus_and_moment_controls_admitted"])
        self.assertTrue(
            admission["scoped_representation_negatives_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "target-specialized nonlinear torus C5",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
