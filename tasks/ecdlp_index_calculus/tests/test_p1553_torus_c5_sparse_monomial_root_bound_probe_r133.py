from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_torus_c5_sparse_monomial_root_bound_probe_r133.py"
SPEC = importlib.util.spec_from_file_location("p1553_r133", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R133 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R133)


class TorusC5SparseMonomialRootBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R133.build_bundle()
        cls.report = cls.bundle["report"]
        cls.theorem = cls.report["theorem"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.bundle["cost"]["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R133.verify_source_bindings()), 11)

    def test_primary_sparse_root_source_is_pinned(self) -> None:
        source = self.theorem["primary_source"]
        self.assertEqual(source["arxiv"], "1602.00208")
        self.assertEqual(source["pinned_sha256"], R133.KELLEY_PDF_SHA256)

    def test_subgroup_adaptation_has_prime_coset_parameter(self) -> None:
        self.assertEqual(self.theorem["prime_order_coset_parameter"], 1)
        self.assertEqual(
            len(self.theorem["adaptation_proof_dependencies"]),
            5,
        )
        self.assertIn(
            "2 q^(1-1/(t-1))",
            self.theorem["subgroup_adaptation"],
        )

    def test_one_through_four_modes_are_excluded(self) -> None:
        rows = self.theorem["term_threshold_rows"]
        self.assertTrue(
            all(
                row["excluded_for_q_to_three_quarters_roots"]
                for row in rows[:4]
            )
        )
        self.assertEqual(rows[3]["root_exponent_q"]["exact"], "2/3")
        self.assertTrue(rows[4]["first_not_excluded_by_bound"])
        self.assertEqual(rows[4]["root_exponent_q"]["exact"], "3/4")

    def test_small_log_density_bound_is_one_eighth(self) -> None:
        for q in (2**20 + 7, 2**40 + 15, 2**80 + 13):
            threshold = R133.logarithmic_mode_threshold(q)
            self.assertLessEqual(threshold - 1, R133.math.log2(q) / 4)
            density = 2 * q ** (-1 / (threshold - 1))
            self.assertLessEqual(density, 1 / 8 + 1e-12)

    def test_random_deck_union_bound_is_model_only(self) -> None:
        model = self.theorem["random_deck_model"]
        self.assertTrue(model["model_bound"])
        self.assertFalse(model["transfers_to_structured_factor_base"])
        self.assertFalse(model["receives_candidate_credit"])
        self.assertIn("8^(-m)", model["union_bound"])

    def test_actual_pure_fifth_witnesses_replay(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertTrue(self.controls["all_fifth_power_maps_are_permutations"])
        self.assertTrue(self.controls["all_pure_fifth_targets_distinct"])
        self.assertTrue(self.controls["all_pure_fifth_sources_replay"])
        self.assertTrue(
            self.controls["all_pure_fifth_targets_in_color_acceptance"]
        )

    def test_actual_small_controls_receive_no_probability_credit(self) -> None:
        self.assertFalse(
            self.controls["all_actual_random_deck_union_bounds_below_one"]
        )
        self.assertFalse(
            self.controls["actual_random_deck_union_bounds_receive_credit"]
        )
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_credit"]
        )

    def test_surviving_selector_classes_remain_open(self) -> None:
        self.assertEqual(
            self.routes["five_mode_extension_zero_predicate"]["status"],
            "open",
        )
        self.assertEqual(
            self.routes[
                "larger_polylog_sparse_extension_predicate"
            ]["status"],
            "open",
        )
        self.assertEqual(
            self.routes["low_slp_expanded_extension_predicate"]["status"],
            "open",
        )
        self.assertEqual(
            self.routes[
                "multi_predicate_frobenius_coordinate_dag"
            ]["status"],
            "open",
        )

    def test_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 13)
        self.assertEqual(admission["obligation_count"], 21)
        self.assertTrue(
            admission["deterministic_one_to_four_mode_negative_admitted"]
        )
        self.assertTrue(
            admission["random_deck_small_log_negative_admitted_model_only"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("five-mode", self.report["next_action"])


if __name__ == "__main__":
    unittest.main()
