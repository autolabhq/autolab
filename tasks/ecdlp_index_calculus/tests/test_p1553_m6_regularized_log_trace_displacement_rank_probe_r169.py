from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_regularized_log_trace_displacement_rank_probe_r169.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r169", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R169 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R169)


class M6RegularizedLogTraceDisplacementRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R169.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]
        cls.sweeps = cls.controls["degree_sweep_controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R169.verify_source_bindings()
        self.assertEqual(len(actual), 16)
        self.assertEqual(
            actual["r168_parent"],
            "390b4ec16a24b79742901f4d8a9894a23bf387edc80998010a0fc0128dc4d68c",
        )
        self.assertEqual(
            actual["bostan_displacement_2017"],
            "e0bff4ecbd9309d4e9c1a050426546ca21ca8eba6d6097a5fca932f5578ecd05",
        )

    def test_neighboring_lanes_and_literature_are_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("scalar pencil", dedup["r168"])
        self.assertIn("short generator", dedup["bostan_displacement_2017"])
        self.assertIn("candidate-safe scalar resolvent", dedup["eagen_2022_596"])
        self.assertIn("different fixed rational convolution", dedup["r150"])
        self.assertIn("compact target witness", dedup["r167"])

    def test_preregistered_control_grids_are_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)
        self.assertEqual(self.controls["degree_sweep_control_count"], 14)

    def test_polynomial_pencil_helpers_are_exact(self) -> None:
        prime = 101
        poly = R169.lambda_characteristic_polynomial([0, 0, 3], prime)
        self.assertEqual(poly, [0, 0, 3, 1])
        self.assertEqual(R169.lambda_valuation(poly), 2)
        self.assertEqual(R169.polynomial_mul([1, 1], [2, 1], prime), [2, 3, 1])

    def test_lambda_pencils_are_monic_and_full_degree(self) -> None:
        self.assertTrue(self.controls["all_lambda_pencils_exact"])
        for row in self.rows:
            self.assertTrue(row["all_lambda_pencils_monic_full_degree"])
            self.assertEqual(
                row["lambda_characteristic_degree"],
                2 * row["c3_divisor_degree"],
            )
            self.assertEqual(
                row["generic_lambda_sample_count_for_interpolation"],
                row["lambda_characteristic_degree"] + 1,
            )

    def test_lambda_valuations_equal_r168_candidate_multiplicities(self) -> None:
        for row in self.rows:
            self.assertTrue(
                row["all_lambda_valuations_equal_candidate_multiplicity"]
            )
            self.assertTrue(row["pencil_candidates_match_r168"])
            self.assertEqual(
                row["pencil_candidate_roots"], row["r168_candidate_roots"]
            )

    def test_regularized_kernels_have_full_row_rank(self) -> None:
        self.assertTrue(self.controls["all_regularized_kernels_full_row_rank"])
        for row in self.rows:
            self.assertEqual(row["regularizing_scalar"], 1)
            self.assertEqual(row["kernel_column_count"], 2 * row["kernel_row_count"])
            self.assertEqual(row["kernel_rank"], row["kernel_row_count"])
            self.assertTrue(row["kernel_full_row_rank"])

    def test_all_six_natural_displacements_have_full_row_rank(self) -> None:
        self.assertTrue(
            self.controls["all_six_natural_displacements_full_row_rank"]
        )
        expected = {
            "x_sylvester_minus",
            "x_sylvester_plus",
            "y_sylvester_minus",
            "y_sylvester_plus",
            "x_stein",
            "y_stein",
        }
        for row in self.rows:
            self.assertEqual(set(row["displacement_ranks"]), expected)
            self.assertTrue(row["all_six_natural_displacements_full_row_rank"])
            self.assertTrue(
                all(
                    rank == row["kernel_row_count"]
                    for rank in row["displacement_ranks"].values()
                )
            )

    def test_first_four_x_displacement_powers_remain_full_rank(self) -> None:
        self.assertTrue(
            self.controls["all_four_x_sylvester_powers_full_row_rank"]
        )
        for row in self.rows:
            self.assertEqual(set(row["x_sylvester_power_ranks"]), {"1", "2", "3", "4"})
            self.assertTrue(row["all_four_x_sylvester_powers_full_row_rank"])

    def test_degree_two_through_eight_sweeps_remain_full_rank(self) -> None:
        self.assertTrue(
            self.controls[
                "all_degree_sweep_kernels_and_x_displacements_full_row_rank"
            ]
        )
        self.assertEqual(self.controls["minimum_tested_witness_degree"], 2)
        self.assertEqual(self.controls["maximum_tested_witness_degree"], 8)
        for seed in (16001, 16002):
            rows = [row for row in self.sweeps if row["seed"] == seed]
            self.assertEqual([row["witness_degree"] for row in rows], list(range(2, 9)))
            self.assertTrue(all(row["kernel_full_row_rank"] for row in rows))
            self.assertTrue(
                all(
                    row["x_sylvester_displacement_full_row_rank"] for row in rows
                )
            )

    def test_generic_pencil_and_full_generator_are_above_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["lambda_pencil_coefficient_or_sample_state_exponent_B"]["exact"],
            "9/2",
        )
        self.assertEqual(
            cost["full_rank_displacement_generator_state_exponent_B"]["exact"],
            "9/2",
        )
        self.assertFalse(cost["generic_lambda_interpolation_inside_rho"])
        self.assertFalse(cost["full_rank_generator_inside_rho"])

    def test_compact_witness_does_not_promote_displacement_compression(self) -> None:
        cost = self.report["cost"]
        admission = self.report["admission"]
        self.assertEqual(cost["compact_log_witness_state_exponent_B"]["exact"], "5/4")
        self.assertTrue(cost["compact_log_witness_inside_rho"])
        self.assertFalse(cost["ordinary_diagonal_displacement_compression_observed"])
        self.assertTrue(admission["ordinary_diagonal_displacement_route_closed_on_controls"])
        self.assertFalse(admission["fraction_free_fitting_constructor_admitted"])
        self.assertFalse(admission["lane_admitted"])

    def test_scope_does_not_refute_custom_elliptic_operators(self) -> None:
        theorem = self.report["theorem"]["scope"]
        cost = self.report["cost"]
        self.assertIn("not an arithmetic-circuit lower bound", theorem)
        self.assertIn("custom elliptic companion displacement", theorem)
        self.assertFalse(cost["custom_elliptic_companion_displacement_refuted"])
        self.assertFalse(cost["fraction_free_fitting_subresultant_supplied"])

    def test_successor_forbids_reusing_closed_materializations(self) -> None:
        interface = self.bundle["frozen"]["successor_interface"]
        forbidden = interface["forbidden_credit"]
        self.assertIn("2n+1 lambda samples", forbidden)
        self.assertIn("full-rank diagonal x/y", forbidden)
        self.assertIn("nN or n^2", forbidden)
        self.assertIn("fraction-free", interface["open_primitive"])

    def test_candidate_oracles_and_finite_lower_bound_credit_are_absent(self) -> None:
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        self.assertFalse(
            self.controls["finite_controls_receive_asymptotic_lower_bound_credit"]
        )
        for row in self.rows:
            self.assertFalse(row["finite_rank_receives_asymptotic_lower_bound_credit"])
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        obligations = self.report["admission"]["obligations"]
        self.assertFalse(obligations["custom_elliptic_companion_displacement_complete"])
        self.assertFalse(obligations["fraction_free_fitting_subresultant_mod_u_complete"])
        self.assertFalse(obligations["unconditional_total_attack_cost_complete"])
        self.assertFalse(obligations["generic_prime_coordinate_family_algorithm"])
        self.assertFalse(obligations["pollard_rho_improvement_complete"])
        self.assertFalse(obligations["shoup_improvement_complete"])
        self.assertFalse(obligations["breakthrough_complete"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
