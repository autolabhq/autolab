from __future__ import annotations

import importlib.util
from fractions import Fraction
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "p1553_m6_short_relation_near_injectivity_supply_probe_r158.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r158_test", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R158 = load_module()


class M6ShortRelationNearInjectivitySupplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R158.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R158.verify_source_bindings()), 10)

    def test_preregistered_grid_is_frozen(self):
        design = self.bundle["frozen"]["design"]
        self.assertEqual(design["arity"], 6)
        self.assertEqual(design["a_pair_counts"], [2, 3])
        self.assertEqual(design["c_pair_counts"], [3, 4, 5])
        self.assertEqual(
            design["independent_a_batch_counts"], [1, 2, 4]
        )
        self.assertEqual(design["seeds"], [15801, 15802])
        self.assertEqual(self.controls["control_count"], 36)

    def test_signed_weight_counts_match_enumeration(self):
        for dimension in range(1, 6):
            for weight in range(1, 8):
                self.assertEqual(
                    len(R158.signed_weight_vectors(dimension, weight)),
                    R158.signed_weight_count(dimension, weight),
                )

    def test_relation_row_universe_is_exactly_l1_five_or_seven(self):
        for dimension in range(1, 6):
            rows = R158.singleton_relation_rows(dimension)
            self.assertEqual(
                len(rows), R158.relation_row_count(dimension)
            )
            self.assertTrue(
                all(
                    sum(abs(value) for value in row) in (5, 7)
                    for row in rows
                )
            )
        self.assertTrue(
            self.controls["all_relation_row_formulas_exact"]
        )

    def test_incident_row_count_is_coordinate_symmetric(self):
        for dimension in range(2, 6):
            rows = R158.singleton_relation_rows(dimension)
            counts = [
                sum(row[index] != 0 for row in rows)
                for index in range(dimension)
            ]
            self.assertEqual(len(set(counts)), 1)
            self.assertEqual(
                counts[0],
                R158.incident_relation_row_count(dimension),
            )

    def test_projective_a_vector_count_quotients_only_sign(self):
        for dimension in (2, 3, 4):
            all_vectors = R158.feasible_six_vectors(dimension)
            canonical = R158.canonical_nonzero_a_vectors(dimension)
            self.assertEqual(len(canonical), (len(all_vectors) - 1) // 2)

    def test_all_moduli_exceed_minor_bound(self):
        self.assertTrue(
            self.controls[
                "all_moduli_exceed_coefficient_minor_bound"
            ]
        )
        self.assertTrue(
            all(
                row[
                    "subgroup_order_exceeds_coefficient_minor_bound_84"
                ]
                for row in self.rows
            )
        )

    def test_collision_pruning_is_applied_before_relation_events(self):
        self.assertGreater(
            sum(
                row["unusable_relation_row_count_due_c6_collisions"]
                for row in self.rows
            ),
            0,
        )
        for row in self.rows:
            self.assertEqual(
                row["usable_singleton_relation_row_count"]
                + row[
                    "unusable_relation_row_count_due_c6_collisions"
                ],
                row[
                    "complete_relation_row_universe_count_l1_five_or_seven"
                ],
            )
            self.assertLessEqual(
                row["good_l1_six_c6_vector_count"],
                row["l1_six_c6_vector_count"],
            )

    def test_candidate_form_expectation_and_variance_are_exact(self):
        for row in self.rows:
            count = row["candidate_projective_form_count"]
            modulus = row["subgroup_order"]
            expected = Fraction(count, modulus)
            self.assertEqual(
                row["expected_relation_event_count"]["exact"],
                (
                    str(expected.numerator)
                    if expected.denominator == 1
                    else f"{expected.numerator}/{expected.denominator}"
                ),
            )
            self.assertLess(
                row["pairwise_independent_event_variance"]["decimal"],
                row["expected_relation_event_count"]["decimal"],
            )

    def test_event_supply_grows_with_independent_batches(self):
        summaries = self.controls[
            "event_counts_by_independent_a_batch_count"
        ]
        self.assertEqual(
            [
                row["total_observed_relation_event_count"]
                for row in summaries
            ],
            [185, 323, 692],
        )
        self.assertEqual(
            [
                row["total_distinct_relation_row_count"]
                for row in summaries
            ],
            [183, 320, 673],
        )

    def test_finite_rank_transition_is_frozen_without_credit(self):
        self.assertEqual(self.controls["full_rank_control_count"], 36)
        self.assertEqual(
            self.controls[
                "full_rank_counts_by_independent_a_batch_count"
            ],
            [12, 12, 12],
        )
        self.assertTrue(
            all(row["uncovered_column_count"] == 0 for row in self.rows)
        )
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_credit"]
        )

    def test_asymptotic_exponents_are_frozen(self):
        exponents = self.bundle["report"]["exponents"]
        self.assertEqual(
            exponents["feasible_c6_vector_count_exponent_B"]["exact"],
            "9/2",
        )
        self.assertEqual(
            exponents["c6_collision_pair_count_exponent_B"]["exact"],
            "4",
        )
        self.assertEqual(
            exponents[
                "expected_bad_singleton_fraction_exponent_B"
            ]["exact"],
            "-1/2",
        )
        self.assertEqual(
            exponents["relation_row_universe_exponent_B"]["exact"],
            "21/4",
        )
        self.assertEqual(
            exponents[
                "projective_candidate_form_count_exponent_B"
            ]["exact"],
            "23/4",
        )
        self.assertEqual(
            exponents["expected_relation_count_exponent_B"]["exact"],
            "3/4",
        )
        self.assertEqual(
            exponents["duplicate_row_event_exponent_B"]["exact"],
            "-15/4",
        )

    def test_pairwise_independence_boundary_is_explicit(self):
        exponents = self.bundle["report"]["exponents"]
        self.assertFalse(
            exponents["pairwise_independence_proves_full_coverage"]
        )
        self.assertFalse(
            exponents["pairwise_independence_proves_full_rank"]
        )
        self.assertFalse(
            self.report["admission"][
                "exact_conditioned_hash_to_curve_transfer_admitted"
            ]
        )
        self.assertFalse(
            self.report["admission"]["full_coverage_or_rank_admitted"]
        )

    def test_explicit_cost_remains_above_rho(self):
        cost = self.bundle["cost"]
        self.assertEqual(
            cost["explicit_c6_enumeration_exponent_B"]["exact"], "9/2"
        )
        self.assertTrue(cost["explicit_control_exceeds_pollard_rho"])
        self.assertFalse(
            cost["finite_explicit_controls_receive_attack_credit"]
        )

    def test_no_algorithm_or_breakthrough_is_promoted(self):
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        self.assertFalse(
            self.report["finite_controls_receive_asymptotic_credit"]
        )
        self.assertFalse(self.report["admission"]["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_open_outputs_remain_explicit(self):
        required = self.bundle["frozen"]["open_obligations"]
        self.assertEqual(
            required["exact_conditioned_hash_to_curve_transfer"], "open"
        )
        self.assertEqual(
            required["full_column_coverage_theorem"], "open"
        )
        self.assertEqual(
            required["full_projective_rank_theorem"], "open"
        )
        self.assertEqual(
            required["reverse_only_signed_marker_operator"], "open"
        )
        self.assertFalse(required["shoup_bound_improvement"])


if __name__ == "__main__":
    unittest.main()
