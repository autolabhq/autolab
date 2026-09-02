from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_monogenic_kernel_bidegree_probe_r181.py"


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r181_tested", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R181 = load_module()


class MonogenicKernelBidegreeProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R181.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["controls"]
        cls.cost = cls.report["cost"]

    def test_source_bindings_are_complete(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 14)
        self.assertEqual(
            set(self.report["source_bindings"]),
            {name for name, _, _ in R181.SOURCE_BINDINGS},
        )

    def test_development_and_held_out_controls_replay(self) -> None:
        self.assertEqual(self.controls["control_count"], 9)
        self.assertEqual(self.controls["development_control_count"], 6)
        self.assertEqual(self.controls["held_out_control_count"], 3)
        self.assertEqual(self.controls["held_out_seeds"], [18104])

    def test_canonical_normal_forms_have_exact_size(self) -> None:
        self.assertTrue(self.controls["all_target_pole_orders_equal_3n"])
        self.assertTrue(self.controls["all_coefficient_bodies_fully_dense"])
        self.assertEqual(self.controls["coefficient_slot_count"], 40149)
        self.assertEqual(
            self.controls["coefficient_nonzero_count"],
            self.controls["coefficient_slot_count"],
        )
        for row in self.controls["controls"]:
            n = row["selected_divisor_degree"]
            self.assertEqual(row["coefficient_slot_count_per_source"], 3 * n)
            self.assertEqual(row["maximum_target_pole_order"], 3 * n)
            self.assertEqual(row["minimum_target_pole_order"], 3 * n)

    def test_exact_kernel_matrices_have_full_source_rank(self) -> None:
        self.assertTrue(
            self.controls["all_coefficient_matrices_full_source_rank"]
        )
        self.assertTrue(
            self.controls["all_evaluation_matrices_full_source_rank"]
        )
        for row in self.controls["controls"]:
            n = row["selected_divisor_degree"]
            self.assertEqual(row["coefficient_matrix_rank"], n)
            self.assertEqual(row["evaluation_matrix_rank"], n)
            self.assertEqual(row["exact_scalar_source_degree_lower_bound"], n - 1)

    def test_canonical_kernel_replays_r180_factor_values(self) -> None:
        self.assertTrue(
            self.controls["all_canonical_kernels_equal_r180_factors"]
        )
        self.assertEqual(self.controls["scan_pair_evaluation_count"], 658473)

    def test_all_desired_incidence_zeros_are_exact(self) -> None:
        self.assertTrue(self.controls["all_desired_incidence_zeros_exact"])
        self.assertEqual(self.controls["desired_incidence_zero_count"], 13383)
        for row in self.controls["controls"]:
            self.assertTrue(row["desired_target_sets_are_distinct"])
            self.assertTrue(row["nonincident_witness_per_source"])
            self.assertTrue(row["pole_order_zero_bound_respected"])

    def test_signed_candidate_factors_replay(self) -> None:
        self.assertTrue(
            self.controls["all_candidate_factors_equal_signed_replay"]
        )
        sources = [
            row["candidate_replay_source"] for row in self.controls["controls"]
        ]
        self.assertEqual(sources.count("r180_frozen_control"), 6)
        self.assertEqual(sources.count("fresh_r174_held_out_replay"), 3)

    def test_flattened_norm_and_full_resultant_are_charged(self) -> None:
        self.assertEqual(
            self.cost["flattened_tensor_algebra_dimension_exponent_B"]["exact"],
            "7/2",
        )
        self.assertEqual(
            self.cost["full_composed_resultant_output_degree_exponent_B"]["exact"],
            "7/2",
        )
        self.assertFalse(
            self.cost["flattened_near_linear_norm_strictly_below_rho"]
        )
        self.assertFalse(
            self.cost["full_composed_resultant_output_strictly_below_rho"]
        )

    def test_postcompiled_modcomp_does_not_receive_constructor_credit(self) -> None:
        self.assertTrue(
            self.cost["postcompiled_degree_n_modcomp_strictly_below_rho"]
        )
        self.assertFalse(self.cost["postcompiled_input_coefficients_supplied"])
        self.assertIn("a=X and H=C_h", self.report["theorem"]["monogenic_tautology"])

    def test_gcd_equivalent_resultant_remains_open(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 21)
        self.assertEqual(admission["obligation_count"], 33)
        self.assertTrue(admission["explicit_exact_bounded_bidegree_fold_closed"])
        self.assertTrue(admission["flattened_norm_route_closed"])
        self.assertFalse(
            admission["gcd_equivalent_output_sensitive_resultant_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])

    def test_no_asymptotic_lower_bound_or_attack_credit(self) -> None:
        self.assertFalse(
            self.cost["finite_rank_claimed_as_asymptotic_lower_bound"]
        )
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])

    def test_no_breakthrough_flags(self) -> None:
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "GCD_EQUIVALENT_OUTPUT_SENSITIVE_RESULTANT_OPEN",
            self.report["classification"],
        )

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R181.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "applicability"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
