from __future__ import annotations

from fractions import Fraction
import importlib.util
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "p1553_m6_singleton_relation_hypergraph_rank_probe_r155.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r155_test", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R155 = load_module()


class M6SingletonRelationHypergraphRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R155.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_and_primary_papers_are_exact(self):
        self.assertEqual(len(R155.verify_source_bindings()), 12)
        self.assertEqual(len(self.report["literature"]), 2)

    def test_endpoint_multisets_enumerate_every_unordered_source(self):
        labels = (1, 2, 3, 4)
        endpoints = R155.endpoint_multisets(labels, 6, 101)
        self.assertEqual(
            sum(len(sources) for sources in endpoints.values()),
            math.comb(len(labels) + 5, 6),
        )

    def test_permutation_counts_preserve_ordered_multiplicity(self):
        self.assertEqual(R155.permutation_count((6, 0, 0)), 1)
        self.assertEqual(R155.permutation_count((3, 2, 1)), 60)
        self.assertEqual(R155.permutation_count((1, 1, 1, 1, 1, 1)), 720)

    def test_independent_coverage_comparator_is_exact(self):
        probability = R155.independent_uniform_coverage_probability(
            3, (1, 1, 1)
        )
        expected_uncovered = R155.independent_uniform_expected_uncovered(
            3, (1, 1, 1)
        )
        self.assertEqual(probability, Fraction(2, 9))
        self.assertEqual(expected_uncovered, Fraction(8, 9))

    def test_all_singleton_rows_normalize_from_aggregate_rows(self):
        self.assertEqual(self.controls["control_count"], 48)
        self.assertTrue(
            self.controls["all_singleton_normalizations_exact"]
        )
        for row in self.controls["controls"]:
            self.assertTrue(
                row["all_singleton_aggregate_rows_normalize_exactly"]
            )
            self.assertTrue(row["inherited_aggregate_rank_matches"])

    def test_singleton_support_bound_is_respected(self):
        self.assertEqual(
            self.controls["max_observed_singleton_support_size"],
            6,
        )
        self.assertLessEqual(
            self.controls["max_observed_singleton_support_size"],
            7,
        )

    def test_singleton_and_aggregate_full_rank_outcomes_agree(self):
        self.assertEqual(
            self.controls["aggregate_full_rank_control_count"],
            11,
        )
        self.assertEqual(
            self.controls["singleton_full_rank_control_count"],
            11,
        )
        self.assertTrue(
            self.controls[
                "aggregate_and_singleton_full_rank_outcomes_agree"
            ]
        )

    def test_coverage_is_necessary_but_not_sufficient(self):
        self.assertTrue(
            self.controls[
                "all_full_rank_singleton_controls_cover_every_column"
            ]
        )
        self.assertEqual(
            self.controls["covered_but_rank_deficient_control_count"],
            17,
        )

    def test_opposite_shift_rows_expose_exact_dependency(self):
        self.assertTrue(
            self.controls[
                "all_singleton_row_sets_closed_under_opposites"
            ]
        )
        for row in self.controls["controls"]:
            self.assertTrue(row["opposite_row_closure_exact"])
            self.assertLessEqual(
                row["projectively_distinct_singleton_row_count"],
                row["unique_singleton_row_count"],
            )

    def test_finite_full_rank_pattern_is_frozen(self):
        pattern = [
            row["singleton_full_rank_count"]
            for row in self.controls["grouped_summary"]
        ]
        self.assertEqual(
            pattern,
            [0, 0, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 2],
        )

    def test_literature_hypothesis_gap_is_explicit(self):
        for record in self.report["literature"].values():
            self.assertIn("applicable_model", record)
            self.assertIn("campaign_gap", record)
        for row in self.controls["controls"]:
            self.assertFalse(
                row["independent_uniform_support_model"][
                    "model_applies_to_actual_rows"
                ]
            )

    def test_logarithmic_oversampling_preserves_exponents_only(self):
        cost = self.bundle["cost"]
        self.assertEqual(
            cost["required_relation_count_polylog_factor"],
            "log(B)",
        )
        self.assertEqual(
            cost["reverse_batch_polylog_factor"],
            "log(B)",
        )
        self.assertTrue(
            cost["logarithmic_oversampling_changes_no_exponent"]
        )
        self.assertFalse(cost["singleton_hypergraph_contiguity_supplied"])

    def test_finite_controls_receive_no_attack_credit(self):
        self.assertFalse(
            self.report["finite_controls_receive_asymptotic_credit"]
        )
        self.assertFalse(
            self.report["admission"][
                "published_sparse_rank_theorem_transfer_admitted"
            ]
        )
        self.assertFalse(self.report["admission"]["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_open_outputs_remain_explicit(self):
        required = self.bundle["frozen"]["required_open_outputs"]
        self.assertEqual(
            required[
                "singleton_hypergraph_contiguity_or_dependency_theorem"
            ],
            "open",
        )
        self.assertEqual(
            required["reverse_only_signed_marker_operator"],
            "open",
        )
        self.assertEqual(
            required["factor_logs_without_verifier_labels"],
            "open",
        )
        self.assertEqual(required["shoup_bound_improvement"], "open")


if __name__ == "__main__":
    unittest.main()
