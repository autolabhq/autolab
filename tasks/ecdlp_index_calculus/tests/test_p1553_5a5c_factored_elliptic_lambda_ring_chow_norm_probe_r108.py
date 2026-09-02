from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "p1553_5a5c_factored_elliptic_lambda_ring_chow_"
    "norm_probe_r108.py"
)


def load_producer():
    spec = importlib.util.spec_from_file_location("p1553_r108_test", PRODUCER)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R108 producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R108 = load_producer()


class FactoredEllipticLambdaRingChowNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R108.build_bundle()

    def test_degree_five_cycle_index_coefficients(self) -> None:
        rows = R108.cycle_types()
        self.assertEqual(len(rows), 7)
        self.assertEqual(sum(row["class_size"] for row in rows), 120)
        self.assertEqual(
            {
                tuple(row["parts"]): row["class_size"]
                for row in rows
            },
            {
                (5,): 24,
                (4, 1): 30,
                (3, 2): 20,
                (3, 1, 1): 20,
                (2, 2, 1): 15,
                (2, 1, 1, 1): 10,
                (1, 1, 1, 1, 1): 1,
            },
        )

    def test_actual_and_matched_cycle_identities_are_exact(self) -> None:
        controls = R108.actual_and_matched_controls()
        self.assertEqual(controls["actual_count"], 8)
        self.assertEqual(controls["matched_random_deck_count"], 8)
        self.assertTrue(controls["all_deck_cycle_identities_exact"])
        self.assertTrue(controls["all_full_49_term_identities_exact"])
        self.assertTrue(controls["all_target_counts_divide_exactly"])
        self.assertTrue(controls["all_cycle_scales_invertible_in_field"])

    def test_cycle_expansion_gives_uniform_canonical_marker_weight(
        self,
    ) -> None:
        controls = R108.marker_weight_controls()
        self.assertEqual(controls["cycle_scale_per_deck"], 120)
        self.assertEqual(controls["cycle_scale_per_full_source"], 14_400)
        self.assertTrue(controls["all_side_sources_weight_120"])
        self.assertTrue(controls["all_full_sources_weight_14400"])
        self.assertTrue(
            controls["marker_vector_preserved_by_cycle_expansion"]
        )

    def test_identity_cycle_term_preserves_dominant_body(self) -> None:
        ledger = R108.termwise_cost_ledger()
        self.assertEqual(ledger["term_count"], 49)
        self.assertTrue(ledger["identity_cycle_term_coefficient_one"])
        self.assertTrue(ledger["identity_cycle_term_source_exponent_B5"])
        self.assertTrue(ledger["identity_cycle_term_root_interface_B13O5"])
        self.assertFalse(ledger["termwise_evaluation_inside_caps"])

    def test_theorem_of_cube_factors_bundle_class_not_section(self) -> None:
        control = R108.theorem_of_cube_control()
        self.assertTrue(
            control[
                "iterated_line_bundle_class_uses_one_body_and_pairwise_factors"
            ]
        )
        self.assertTrue(control["all_pair_tables_inside_online_cap"])
        self.assertFalse(
            control["line_bundle_isomorphism_is_section_factorization"]
        )
        self.assertFalse(
            control["target_section_pure_tensor_factorization_supplied"]
        )
        self.assertFalse(
            control["poincare_trivialization_and_scalar_evaluator_supplied"]
        )

    def test_cost_scope_preserves_poincare_theta_lane(self) -> None:
        scope = R108.cost_ledger()["scope_boundary"]
        self.assertTrue(scope["cycle_index_canonical_correction_admitted"])
        self.assertTrue(scope["termwise_49_norm_constructor_rejected"])
        self.assertTrue(
            scope["theorem_of_cube_line_bundle_factorization_admitted"]
        )
        self.assertTrue(scope["poincare_theta_section_factorization_open"])
        self.assertFalse(
            scope["general_section_tensor_rank_lower_bound_claimed"]
        )

    def test_bundle_preserves_nonclaim_and_routes_section_factorization(
        self,
    ) -> None:
        report = self.bundle["report"]
        admission = report["admission"]
        self.assertFalse(admission["lane_admitted"])
        self.assertEqual(admission["passed_obligation_count"], 17)
        self.assertEqual(admission["obligation_count"], 30)
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])
        self.assertIn("Poincare/theta", report["next_action"])
        self.assertIn(
            "The theorem of the cube does not factor the target section.",
            report["non_claims"],
        )


if __name__ == "__main__":
    unittest.main()
