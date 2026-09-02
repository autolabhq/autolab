from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_m6_signed_c3_divisor_translation_gcd_probe_r161.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r161", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {MODULE_PATH}")
R161 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R161)


class M6SignedC3DivisorTranslationGcdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R161.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.rows = cls.controls["controls"]

    def test_source_bindings_are_exact(self) -> None:
        actual = R161.verify_source_bindings()
        self.assertEqual(len(actual), 16)
        self.assertEqual(
            actual["semaev_summation_polynomials"],
            "991f85d58ab68551a229266d03c2f88a5fc42e81b2a5f8f4432937bcceff16df",
        )

    def test_prior_translated_divisor_cost_is_deduplicated(self) -> None:
        dedup = self.report["deduplication"]
        self.assertIn("Already charges", dedup["r117"])
        self.assertIn("exact signed U,V", dedup["r117"])
        self.assertIn("explicit coordinate operation", dedup["r160"])

    def test_preregistered_control_grid_is_complete(self) -> None:
        self.assertEqual(self.controls["family_count"], 3)
        self.assertEqual(self.controls["seeds"], [16001, 16002])
        self.assertEqual(self.controls["control_count"], 6)

    def test_signed_c3_divisor_degree_is_exact(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["c3_degree"],
                math.comb(row["factor_base_dimension"] + 2, 3),
            )
            self.assertTrue(row["c3_degree_formula_exact"])
            self.assertTrue(row["c3_x_coordinates_injective"])

    def test_signed_divisor_lies_on_curve_mod_u(self) -> None:
        self.assertEqual(self.controls["exact_signed_divisor_control_count"], 6)
        for row in self.rows:
            self.assertTrue(row["signed_divisor_curve_residual_zero"])

    def test_every_positive_source_is_recovered_exactly(self) -> None:
        self.assertEqual(self.controls["exact_positive_source_control_count"], 6)
        for row in self.rows:
            self.assertTrue(row["all_positive_sources_exact"])
            for target in row["positive_targets"]:
                self.assertTrue(target["source_presence_exact"])
                self.assertTrue(target["source_matches_expected_coefficients"])
                self.assertTrue(target["source_identity_exact"])

    def test_positive_targets_have_unique_c6_sources(self) -> None:
        for row in self.rows:
            self.assertTrue(row["all_positive_targets_unique_c6"])
            self.assertTrue(
                all(target["verifier_unique_c6_endpoint"] for target in row["positive_targets"])
            )

    def test_source_gcd_degree_bound_is_exactly_guarded(self) -> None:
        for row in self.rows:
            self.assertLessEqual(row["maximum_positive_source_gcd_degree"], 20)
            self.assertTrue(row["unique_source_gcd_degree_bound_20_holds"])

    def test_empty_targets_are_rejected_exactly(self) -> None:
        self.assertEqual(self.controls["exact_empty_target_control_count"], 6)
        for row in self.rows:
            target = row["empty_target"]
            self.assertTrue(row["empty_target_exact"])
            self.assertEqual(target["gcd_degree"], 0)
            self.assertIsNone(target["located_source"])

    def test_denominator_exception_branch_is_exercised(self) -> None:
        self.assertEqual(self.controls["exact_exceptional_branch_control_count"], 6)
        for row in self.rows:
            self.assertTrue(row["denominator_exception_branch_exercised"])
            self.assertTrue(row["denominator_exception_semantics_exact"])
            self.assertGreaterEqual(
                row["denominator_exception_target"]["exceptional_left_count"], 1
            )

    def test_candidate_oracles_are_not_consumed(self) -> None:
        self.assertFalse(self.controls["candidate_oracle_consumed"])
        for row in self.rows:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_independent_target_cost_is_fully_charged(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["c3_divisor_degree_exponent_B"]["exact"], "9/4")
        self.assertEqual(cost["r159_target_batch_exponent_B"]["exact"], "5/4")
        self.assertEqual(
            cost["independent_complete_batch_exponent_B"]["exact"], "7/2"
        )
        self.assertTrue(cost["independent_target_route_exceeds_pollard_rho"])

    def test_coordinate_operation_is_explicit_and_non_generic(self) -> None:
        cost = self.report["cost"]
        self.assertTrue(cost["coordinate_specific_operation_explicit"])
        self.assertFalse(cost["generic_encoding_invariant"])
        self.assertIn("coordinate-specific", self.report["theorem"]["scope"])

    def test_many_target_composition_and_gcd_remain_open(self) -> None:
        cost = self.report["cost"]
        admission = self.report["admission"]
        self.assertFalse(cost["target_batched_many_inner_modular_composition_supplied"])
        self.assertFalse(cost["target_batched_gcd_source_adjoint_supplied"])
        self.assertFalse(admission["target_batched_composition_admitted"])
        self.assertIn("many-inner", self.report["next_action"])

    def test_finite_controls_receive_no_attack_credit(self) -> None:
        self.assertFalse(self.controls["finite_controls_receive_attack_credit"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(
            self.report["cost"]["finite_polynomial_and_dictionary_work_receives_attack_credit"]
        )

    def test_no_algorithm_or_breakthrough_is_promoted(self) -> None:
        admission = self.report["admission"]
        obligations = admission["obligations"]
        self.assertTrue(admission["signed_divisor_translation_gcd_interface_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(obligations["unconditional_total_attack_cost_complete"])
        self.assertFalse(obligations["generic_prime_coordinate_family_algorithm"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
