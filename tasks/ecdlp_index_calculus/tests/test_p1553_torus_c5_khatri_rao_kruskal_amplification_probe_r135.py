from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_torus_c5_khatri_rao_kruskal_amplification_probe_r135.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r135", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R135 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R135)


class TorusC5KhatriRaoKruskalAmplificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R135.build_bundle()
        cls.report = cls.bundle["report"]
        cls.theorem = cls.report["theorem"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.bundle["cost"]["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R135.verify_source_bindings()), 11)
        self.assertEqual(
            R135.sha256_file(R135.BHASKARA_PDF),
            R135.BHASKARA_PDF_SHA256,
        )

    def test_fp2_rank_and_kruskal_rank_primitives(self) -> None:
        field = R135.Field(R135.R82.FAMILIES[0]["field_prime"])
        matrix = [
            [field.elt(1), field.elt(0), field.elt(1)],
            [field.elt(0), field.elt(1), field.elt(1)],
        ]
        self.assertEqual(R135.fp2_matrix_rank(matrix, field), 2)
        self.assertEqual(R135.kruskal_rank(matrix, field), 2)

    def test_finite_field_khatri_rao_proof_is_self_contained(self) -> None:
        self.assertIn(
            "dual functionals",
            self.theorem["finite_field_khatri_rao_lemma"],
        )
        self.assertIn(
            "5*(krank(A)-1)+1",
            self.theorem["fifth_power_amplification"],
        )
        self.assertTrue(
            self.theorem["field_independent_ordinary_rank_lemma"]
        )
        self.assertFalse(
            self.theorem["robust_real_rank_claim_imported"]
        )

    def test_c5_matrix_is_deduplicated_fifth_khatri_rao_power(self) -> None:
        self.assertIn(
            "degree-five source product",
            self.theorem["c5_matrix_identification"],
        )
        self.assertTrue(
            self.controls["all_available_khatri_rao_rows_replay"]
        )
        for control in self.controls["controls"]:
            for color in control["color_controls"]:
                if not color["witness_available"]:
                    continue
                self.assertEqual(
                    color["ordered_khatri_rao_row_count"],
                    32,
                )
                self.assertEqual(
                    color["ordered_khatri_rao_unique_row_count"],
                    6,
                )

    def test_exact_atom_and_product_kruskal_ranks(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertEqual(self.controls["active_color_control_count"], 30)
        self.assertEqual(
            self.controls["available_two_atom_control_count"],
            12,
        )
        self.assertTrue(
            self.controls["all_available_atom_kranks_equal_two"]
        )
        self.assertTrue(
            self.controls["all_available_product_kranks_equal_six"]
        )
        self.assertTrue(
            self.controls[
                "all_available_seven_column_matrices_dependent"
            ]
        )

    def test_actual_c5_sources_replay_and_are_color_accepted(self) -> None:
        self.assertTrue(
            self.controls[
                "all_available_sources_replay_and_are_accepted"
            ]
        )

    def test_random_deck_bound_is_model_bound(self) -> None:
        random_model = self.theorem["random_deck_model"]
        self.assertIn(
            "4*T*36^T*q^(3T)*rho_epsilon^m",
            random_model["union_bound"],
        )
        self.assertEqual(
            self.theorem["random_model_excluded_mode_asymptotic"],
            "(5-o(1))*log2(q)",
        )
        self.assertFalse(
            random_model["structured_factor_base_transfer_proved"]
        )
        self.assertFalse(random_model["receives_candidate_credit"])
        self.assertFalse(
            self.controls["all_actual_random_union_bounds_below_one"]
        )

    def test_deterministic_credit_stops_at_six_modes(self) -> None:
        self.assertEqual(
            self.theorem["maximum_deterministic_excluded_mode_count"],
            6,
        )
        self.assertFalse(
            self.report["admission"][
                "structured_factor_base_above_six_admitted"
            ]
        )

    def test_surviving_selector_classes_remain_open(self) -> None:
        for route_id in (
            "seven_plus_mode_structured_zero_or_pole",
            "multiple_small_mode_predicate_dag",
            "low_slp_expanded_extension_predicate",
            "nonzero_value_frobenius_coordinate_dag",
        ):
            self.assertEqual(self.routes[route_id]["status"], "open")

    def test_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 14)
        self.assertEqual(admission["obligation_count"], 22)
        self.assertTrue(
            admission["deterministic_one_to_six_mode_negative_admitted"]
        )
        self.assertTrue(
            admission["random_model_near_five_log_mode_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("multiple-small-predicate", self.report["next_action"])


if __name__ == "__main__":
    unittest.main()
