from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "p1553_m6_a_diversity_projective_rank_probe_r156.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r156_test", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R156 = load_module()


class M6ADiversityProjectiveRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R156.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R156.verify_source_bindings()), 10)

    def test_preregistered_grid_is_frozen(self):
        design = self.bundle["frozen"]["design"]
        self.assertEqual(design["a_pair_counts"], [2, 3, 4])
        self.assertEqual(design["c_pair_counts"], [5, 6, 7, 8])
        self.assertEqual(
            design["log_oversampling_factors"],
            [2, 4, 8],
        )
        self.assertEqual(design["seeds"], [15601, 15602, 15603])
        self.assertEqual(design["arity"], 6)

    def test_logarithmic_occupancy_multipliers_are_exact(self):
        self.assertEqual(
            [
                R156.occupancy_multiplier(5, factor)
                for factor in R156.LOG_OVERSAMPLING_FACTORS
            ],
            [4, 7, 13],
        )
        self.assertEqual(
            [
                R156.occupancy_multiplier(8, factor)
                for factor in R156.LOG_OVERSAMPLING_FACTORS
            ],
            [5, 9, 17],
        )

    def test_projective_normalization_quotients_opposites(self):
        modulus = 101
        row = (2, 7, 0, 11)
        opposite = tuple((-value) % modulus for value in row)
        self.assertEqual(
            R156.projective_normalize(row, modulus),
            R156.projective_normalize(opposite, modulus),
        )

    def test_all_one_hundred_eight_controls_are_exact(self):
        self.assertEqual(self.controls["control_count"], 108)
        self.assertTrue(
            self.controls[
                "all_relation_and_opposite_row_identities_exact"
            ]
        )
        for row in self.controls["controls"]:
            self.assertTrue(row["all_relation_identities_exact"])
            self.assertTrue(row["all_opposite_rows_exact"])

    def test_moduli_are_fixed_from_preregistered_supports(self):
        for row in self.controls["controls"]:
            expected = R156.R154.next_prime(
                math.ceil(
                    row["max_a6_signed_coefficient_support"]
                    * row["max_c6_signed_coefficient_support"]
                    / row["occupancy_multiplier"]
                )
            )
            self.assertEqual(row["subgroup_order"], expected)

    def test_rank_deficit_decomposition_is_exact(self):
        for row in self.controls["controls"]:
            expected_dependency = max(
                0,
                min(
                    row["signed_log_dimension"],
                    row["projectively_distinct_row_count"],
                )
                - row["signed_rank"],
            )
            self.assertEqual(
                row["dependency_nullity_after_projective_dedup"],
                expected_dependency,
            )
            if row["signed_full_rank"]:
                self.assertEqual(row["uncovered_column_count"], 0)
                self.assertEqual(row["projective_count_deficit"], 0)
                self.assertEqual(expected_dependency, 0)

    def test_full_rank_counts_by_a_diversity_are_frozen(self):
        self.assertEqual(self.controls["full_rank_control_count"], 62)
        self.assertEqual(
            self.controls["full_rank_counts_by_a_pair_count"],
            [18, 22, 22],
        )
        self.assertTrue(
            self.controls[
                "full_rank_counts_monotone_in_a_diversity"
            ]
        )

    def test_logarithmic_oversampling_transition_is_frozen(self):
        self.assertEqual(
            self.controls[
                "full_rank_counts_by_log_oversampling_factor"
            ],
            [6, 21, 35],
        )
        self.assertTrue(
            self.controls[
                "full_rank_counts_monotone_in_log_oversampling"
            ]
        )
        self.assertEqual(
            [
                row["control_count"]
                for row in self.controls["log_factor_summary"]
            ],
            [36, 36, 36],
        )

    def test_high_log_factor_has_one_finite_failure(self):
        high = [
            row
            for row in self.controls["controls"]
            if row["log_oversampling_factor"] == 8
        ]
        failures = [row for row in high if not row["signed_full_rank"]]
        self.assertEqual(len(high), 36)
        self.assertEqual(len(failures), 1)

    def test_costs_preserve_only_the_selected_exponents(self):
        cost = self.bundle["cost"]
        self.assertEqual(cost["relation_count_polylog_factor"], "log(B)")
        self.assertEqual(
            cost["structured_row_batch_polylog_factor"],
            "log(B)",
        )
        self.assertTrue(
            cost[
                "opposite_row_projective_quotient_changes_no_exponent"
            ]
        )
        self.assertFalse(cost["convolution_tanner_contiguity_supplied"])

    def test_finite_controls_receive_no_attack_credit(self):
        self.assertFalse(
            self.report["finite_controls_receive_asymptotic_credit"]
        )
        self.assertFalse(
            self.report["admission"][
                "asymptotic_rank_or_hash_to_curve_transfer_admitted"
            ]
        )
        self.assertFalse(self.report["admission"]["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_open_outputs_remain_explicit(self):
        required = self.bundle["frozen"]["required_open_outputs"]
        self.assertEqual(
            required["asymptotic_a_diversity_rank_theorem"],
            "open",
        )
        self.assertEqual(
            required[
                "convolution_tanner_contiguity_or_direct_rank_theorem"
            ],
            "open",
        )
        self.assertEqual(
            required["reverse_only_signed_marker_operator"],
            "open",
        )
        self.assertEqual(required["shoup_bound_improvement"], "open")


if __name__ == "__main__":
    unittest.main()
