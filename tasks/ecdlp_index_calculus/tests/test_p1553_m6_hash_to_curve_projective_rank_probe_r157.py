from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "p1553_m6_hash_to_curve_projective_rank_probe_r157.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r157_test", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R157 = load_module()


class M6HashToCurveProjectiveRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R157.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R157.verify_source_bindings()), 10)

    def test_preregistered_public_group_grid_is_frozen(self):
        frozen = self.bundle["frozen"]
        self.assertEqual(frozen["arity"], 6)
        self.assertEqual(frozen["log_oversampling_factors"], [2, 4, 8])
        self.assertEqual(frozen["offsets"], [0, 1])
        self.assertEqual(self.controls["control_count"], 24)

    def test_coefficient_fiber_formula(self):
        self.assertEqual(
            R157.coefficient_fiber_size((6, 0, 0), 3), 1
        )
        self.assertEqual(
            R157.coefficient_fiber_size((4, 0, 0), 3), 3
        )
        self.assertEqual(
            R157.coefficient_fiber_size((2, 0, 0), 3), 6
        )
        self.assertEqual(
            R157.coefficient_fiber_size((0, 0, 0), 3), 10
        )

    def test_actual_c6_maps_are_injective_and_fibers_exact(self):
        self.assertTrue(
            self.controls["all_c6_signed_coefficient_maps_injective"]
        )
        self.assertTrue(
            self.controls["all_coefficient_fiber_formulas_exact"]
        )
        self.assertTrue(
            self.controls["all_singleton_iff_l1_six_criteria_exact"]
        )
        for row in self.rows:
            self.assertTrue(row["c6_signed_coefficient_map_injective"])
            self.assertTrue(row["c6_coefficient_fiber_formula_exact"])
            self.assertTrue(row["c6_singleton_iff_l1_six_exact"])

    def test_public_group_relations_and_opposites_are_exact(self):
        self.assertTrue(
            self.controls[
                "all_public_group_relations_and_opposites_exact"
            ]
        )
        for row in self.rows:
            self.assertTrue(
                row["all_public_group_relation_identities_exact"]
            )
            self.assertTrue(row["all_opposite_rows_exact"])
            self.assertTrue(row["projective_duplicate_rhs_consistent"])

    def test_no_candidate_or_verifier_label_oracle_is_consumed(self):
        self.assertTrue(self.controls["all_candidate_oracles_avoided"])
        for row in self.rows:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_count_marginal_rank_or_source_oracle_consumed"
                ]
            )
            self.assertFalse(row["verifier_bsgs_labels_consumed"])

    def test_rank_deficit_decomposition_is_exact(self):
        for row in self.rows:
            expected = max(
                0,
                min(
                    row["signed_log_dimension"],
                    row["projectively_distinct_row_count"],
                )
                - row["signed_rank"],
            )
            self.assertEqual(
                row["dependency_nullity_after_projective_dedup"],
                expected,
            )
            if row["signed_full_rank"]:
                self.assertEqual(row["uncovered_column_count"], 0)
                self.assertEqual(row["projective_count_deficit"], 0)
                self.assertEqual(expected, 0)

    def test_full_rank_and_factor_log_counts_are_frozen(self):
        self.assertEqual(self.controls["full_rank_control_count"], 23)
        self.assertEqual(self.controls["factor_log_control_count"], 23)
        self.assertEqual(
            self.controls["full_rank_counts_by_log_factor"],
            [7, 8, 8],
        )
        self.assertTrue(
            self.bundle["logs"][
                "all_full_rank_factor_log_controls_verified"
            ]
        )

    def test_factor_logs_are_publicly_verified_when_rank_is_full(self):
        for row in self.rows:
            self.assertEqual(
                row["factor_logs_computed_without_dlp_oracle"],
                row["signed_full_rank"],
            )
            self.assertEqual(
                row[
                    "factor_logs_verified_by_public_scalar_multiplication"
                ],
                row["signed_full_rank"],
            )
            if row["signed_full_rank"]:
                self.assertIsNotNone(
                    row["recovered_factor_logs_sha256"]
                )

    def test_only_failure_is_projective_row_count(self):
        failures = [row for row in self.rows if not row["signed_full_rank"]]
        self.assertEqual(len(failures), 1)
        failure = failures[0]
        self.assertEqual(failure["signed_log_dimension"], 3)
        self.assertEqual(failure["signed_rank"], 2)
        self.assertEqual(failure["projectively_distinct_row_count"], 2)
        self.assertEqual(failure["uncovered_column_count"], 0)
        self.assertEqual(failure["projective_count_deficit"], 1)
        self.assertEqual(
            failure["dependency_nullity_after_projective_dedup"], 0
        )

    def test_selected_a_pair_counts_are_preregistered_by_supports(self):
        for row in self.rows:
            selected, target, nominal = R157.choose_a_pair_count(
                row["c_pair_count"],
                row["subgroup_order"],
                row["log_oversampling_factor"],
            )
            self.assertEqual(row["selected_a_pair_count"], selected)
            self.assertEqual(row["target_occupancy_multiplier"], target)
            self.assertEqual(
                row["nominal_max_support_occupancy"]["exact"],
                str(nominal.numerator)
                if nominal.denominator == 1
                else f"{nominal.numerator}/{nominal.denominator}",
            )

    def test_explicit_control_cost_is_above_rho(self):
        cost = self.bundle["cost"]
        self.assertEqual(
            cost["explicit_c6_endpoint_enumeration_exponent_B"]["exact"],
            "9/2",
        )
        self.assertEqual(
            cost["explicit_control_total_exponent_B"]["exact"], "9/2"
        )
        self.assertTrue(cost["explicit_control_exceeds_pollard_rho"])
        self.assertFalse(
            cost["finite_explicit_enumeration_receives_attack_credit"]
        )

    def test_finite_transfer_receives_no_asymptotic_credit(self):
        self.assertFalse(
            self.report["finite_controls_receive_asymptotic_credit"]
        )
        self.assertFalse(
            self.report["admission"][
                "asymptotic_hash_to_curve_rank_admitted"
            ]
        )
        self.assertFalse(self.report["admission"]["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_open_outputs_remain_explicit(self):
        required = self.bundle["frozen"]["open_obligations"]
        self.assertEqual(
            required[
                "asymptotic_hash_to_curve_coefficient_map_injectivity"
            ],
            "open",
        )
        self.assertEqual(
            required["asymptotic_short_relation_rank_theorem"], "open"
        )
        self.assertEqual(
            required["reverse_only_signed_marker_operator"], "open"
        )
        self.assertFalse(required["shoup_bound_improvement"])


if __name__ == "__main__":
    unittest.main()
