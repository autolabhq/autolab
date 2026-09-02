from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_m6_target_batched_c3_elliptic_transpose_probe_r117.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r117", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R117 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R117)


class M6TargetBatchedC3EllipticTransposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R117.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.replay = cls.bundle["replay"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R117.verify_source_bindings()), 12)

    def test_finite_dft_factorization_and_full_rank_are_exact(self) -> None:
        self.assertTrue(self.controls["all_dft_factorizations_exact"])
        self.assertTrue(self.controls["all_translation_orbits_full_rank"])
        self.assertTrue(
            self.controls["sparse_support_full_rank_control_present"]
        )
        self.assertFalse(
            self.controls[
                "finite_labels_and_enumeration_receive_asymptotic_credit"
            ]
        )
        for control in self.controls["controls"]:
            self.assertEqual(
                control["translation_orbit_rank_over_auxiliary_field"],
                control["curve"]["subgroup_order"],
            )
            self.assertEqual(
                control["dft_factorization_failure_count"],
                0,
            )
            self.assertEqual(
                control["target_count_zero_frequency_count"],
                0,
            )

    def test_sparse_canonical_source_control_has_many_zero_targets(
        self,
    ) -> None:
        sparse = self.controls["controls"][1]
        self.assertEqual(sparse["ordered_source_count"], 46656)
        self.assertEqual(sparse["canonical_source_count"], 196)
        self.assertEqual(sparse["target_support_size"], 196)
        self.assertEqual(sparse["zero_target_count"], 16231)
        self.assertTrue(sparse["ordered_and_canonical_support_equal"])
        self.assertTrue(sparse["full_translation_orbit_rank"])

    def test_prime_cyclotomic_theorem_scope_is_frozen(self) -> None:
        theorem = self.cost["prime_cyclotomic_translation_rank_theorem"]
        self.assertEqual(theorem["translation_orbit_rank"], "q")
        self.assertEqual(theorem["state_exponent_B"]["exact"], "5")
        self.assertIn("Phi_q", theorem["proof"])
        self.assertFalse(theorem["base_field_linear_rank_theorem_claimed"])
        self.assertFalse(
            self.cost["unconditional_data_structure_lower_bound_claimed"]
        )

    def test_canonical_support_and_regular_section_cost_are_charged(
        self,
    ) -> None:
        theorem = self.cost["canonical_source_support_theorem"]
        self.assertEqual(theorem["leading_denominator"], 518400)
        self.assertEqual(
            theorem["leading_ratio_to_group_order"]["exact"],
            "1/518400",
        )
        self.assertEqual(
            theorem["regular_section_pole_degree_exponent_B"]["exact"],
            "5",
        )
        self.assertFalse(
            theorem["straight_line_circuit_lower_bound_claimed"]
        )

    def test_original_c_deck_k7_routes_miss_setup(self) -> None:
        theorem = self.cost["dinur_golovnev_k7"]
        self.assertEqual(theorem["online_compatible_delta"]["exact"], "1")
        self.assertEqual(theorem["state_exponent_B"]["exact"], "33/8")
        routes = {row["route_id"]: row for row in self.cost["routes"]}
        bound = routes["dinur_golovnev_k7_index_on_original_c_deck"]
        trivial = routes["trivial_k7_store_five_c_sums"]
        self.assertEqual(bound["fresh_batch_exponent_B"]["exact"], "5/4")
        self.assertEqual(trivial["state_exponent_B"]["exact"], "15/4")
        self.assertTrue(bound["inside_fresh_batch_cap"])
        self.assertTrue(trivial["inside_fresh_batch_cap"])
        self.assertFalse(bound["inside_setup_cap"])
        self.assertFalse(trivial["inside_setup_cap"])
        self.assertFalse(self.cost["any_scoped_route_meets_both_caps"])

    def test_exact_source_is_inherited_but_no_adjoint_is_constructed(
        self,
    ) -> None:
        inherited = self.replay["inherited_r116_replay"]
        self.assertTrue(inherited["all_instances_exact"])
        self.assertTrue(inherited["negative_control_present"])
        self.assertFalse(
            self.replay["linear_translation_sketch_source_adjoint_constructed"]
        )
        self.assertFalse(
            self.replay[
                "nonlinear_value_sensitive_source_locator_constructed"
            ]
        )

    def test_scoped_negative_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 10)
        self.assertEqual(admission["obligation_count"], 16)
        self.assertTrue(admission["scoped_negative_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "nonlinear value-sensitive six-C source locator",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
