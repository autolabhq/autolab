from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_m6_suboutput_implicit_c5_character_pairing_probe_r120.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r120", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R120 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R120)


class M6SuboutputImplicitC5CharacterPairingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R120.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.replay = cls.bundle["replay"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R120.verify_source_bindings()), 13)

    def test_base_field_and_self_pairing_characters_are_closed(self) -> None:
        theorems = self.cost["character_theorems"]
        self.assertFalse(
            theorems["base_field_algebraic_group_character"][
                "nonconstant_base_field_algebraic_character_exists"
            ]
        )
        self.assertFalse(
            theorems["self_weil_pairing"][
                "nontrivial_character_from_two_inputs_in_G"
            ]
        )
        pairing = theorems["independent_torsion_pairing_character"]
        self.assertTrue(pairing["source_backpointer_preserved"])
        self.assertIn("iff", pairing["five_sum_equivalence"])

    def test_all_r82_pairing_characters_are_exact(self) -> None:
        self.assertEqual(self.controls["r82_control_count"], 8)
        self.assertTrue(
            self.controls[
                "all_r82_pairing_characters_nontrivial_q_roots"
            ]
        )
        for row in self.controls["r82_pairing_controls"]:
            self.assertTrue(row["pairing_character_nontrivial"])
            self.assertTrue(row["pairing_character_has_order_q"])
            self.assertTrue(row["distortion_point_on_curve"])
            self.assertTrue(row["distortion_point_has_q_torsion"])
            self.assertFalse(row["candidate_scalar_labels_consumed"])

    def test_pairing_preserves_c5_products_sources_and_empty_answers(
        self,
    ) -> None:
        self.assertTrue(
            self.controls[
                "all_r82_c5_endpoint_product_identities_exact"
            ]
        )
        self.assertTrue(self.controls["all_r82_product_maps_injective"])
        self.assertTrue(
            self.controls["all_r82_sources_and_empty_images_replay"]
        )
        for row in self.controls["r82_pairing_controls"]:
            self.assertEqual(
                row["distinct_product_image_count"],
                row["canonical_c5_source_count"],
            )
            self.assertTrue(
                row["all_c5_endpoint_images_equal_source_products"]
            )
            self.assertTrue(row["all_projective_sources_replay"])
            self.assertTrue(
                row[
                    "empty_target_image_absent_from_c5_product_support"
                ]
            )

    def test_r82_fixtures_are_embedding_degree_two_exceptions(
        self,
    ) -> None:
        self.assertTrue(self.controls["all_r82_embedding_degrees_two"])
        for row in self.controls["r82_pairing_controls"]:
            self.assertEqual(row["field_prime_mod_subgroup_order"], -1 % row["subgroup_order"])
            self.assertEqual(row["embedding_degree"], 2)
            self.assertTrue(row["supersingular_j_zero_fixture"])
            self.assertFalse(
                row["finite_pairing_control_receives_asymptotic_credit"]
            )

    def test_prime_order_curves_realize_maximal_embedding_degree(
        self,
    ) -> None:
        rows = self.controls[
            "maximal_embedding_degree_prime_order_curve_controls"
        ]
        self.assertEqual(len(rows), 4)
        self.assertTrue(
            self.controls["all_maximal_embedding_controls_exact"]
        )
        for row in rows:
            self.assertEqual(
                row["observed_curve_order"],
                row["subgroup_order"],
            )
            self.assertEqual(
                row["embedding_degree"],
                row["subgroup_order"] - 1,
            )
            self.assertFalse(
                row["finite_control_receives_asymptotic_credit"]
            )

    def test_embedding_degree_is_charged_per_target(self) -> None:
        charge = self.cost["pairing_cost_parameter"]
        self.assertEqual(charge["embedding_degree"], "k=ord_q(p)")
        self.assertEqual(
            charge["one_target_character_base_field_exponent_B"],
            "log_B(k)",
        )
        self.assertEqual(
            charge["r118_full_batch_base_field_exponent_B"],
            "5/4+log_B(k)",
        )
        self.assertEqual(charge["polylog_query_requires"], "k=B^(o(1))")
        self.assertFalse(
            charge["uniform_generic_prime_requirement_proved"]
        )

    def test_pairing_reencoding_does_not_close_product_index(self) -> None:
        routes = {row["route_id"]: row for row in self.cost["routes"]}
        explicit = routes[
            "small_embedding_degree_pairing_then_explicit_c5"
        ]
        current = routes[
            "small_embedding_degree_pairing_then_current_k6_index"
        ]
        residual = routes[
            "small_embedding_degree_suboutput_multiplicative_"
            "c5_membership_source_circuit"
        ]
        self.assertEqual(explicit["state_exponent_B"]["exact"], "15/4")
        self.assertFalse(explicit["inside_setup_cap"])
        self.assertEqual(current["state_exponent_B"]["exact"], "33/8")
        self.assertFalse(current["inside_setup_cap"])
        self.assertFalse(residual["exact_circuit_constructed"])
        self.assertFalse(residual["general_lower_bound_claimed"])
        self.assertEqual(residual["status"], "open")

    def test_pairing_lift_semantic_dedup_is_explicit(self) -> None:
        dedup = self.cost["semantic_dedup"]
        self.assertEqual(
            dedup["nearby_idea"],
            "ECDLP-IDEA-008 / P1542 pairing lift-return",
        )
        self.assertIn("forward pairing character", dedup["r120_distinct_scope"])
        self.assertFalse(dedup["new_idea_id_claimed"])
        self.assertFalse(self.cost["candidate_work_credit"])

    def test_scoped_pairing_control_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 12)
        self.assertEqual(admission["obligation_count"], 19)
        self.assertTrue(admission["pairing_character_control_admitted"])
        self.assertTrue(admission["scoped_negative_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "sub-output multiplicative five-product",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
