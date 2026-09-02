from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_squarefree_truncated_resultant_applicability_probe_r179.py"


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r179_tested", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R179 = load_module()


class SquarefreeTruncatedResultantApplicabilityProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R179.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["controls"]
        cls.cost = cls.report["cost"]

    def test_source_bindings_are_complete(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 15)
        self.assertEqual(
            set(self.report["source_bindings"]),
            {name for name, _, _ in R179.SOURCE_BINDINGS},
        )

    def test_six_r178_controls_replay(self) -> None:
        self.assertEqual(self.controls["control_count"], 6)
        self.assertTrue(self.controls["all_aggregates_equal_r178"])
        self.assertEqual(self.controls["selected_divisor_degree_sum"], 202)
        self.assertEqual(self.controls["target_count_sum"], 40)

    def test_selected_moduli_are_squarefree(self) -> None:
        self.assertTrue(self.controls["all_squarefree_moduli_verified"])
        for row in self.controls["controls"]:
            self.assertEqual(row["squarefree_gcd_degree"], 0)

    def test_order_one_crt_reconstructs_every_aggregate(self) -> None:
        self.assertTrue(self.controls["all_crt_reconstructions_exact"])
        self.assertEqual(self.controls["crt_component_count"], 202)

    def test_local_power_alias_does_not_determine_mod_u(self) -> None:
        self.assertIn("0 and (X-a)^n", self.report["theorem"]["arbitrary_modulus_mismatch"])
        self.assertTrue(self.controls["all_local_power_alias_witnesses_exact"])
        for row in self.controls["controls"]:
            self.assertTrue(row["local_power_aliases_zero_mod_local_power"])
            self.assertTrue(row["local_power_does_not_alias_zero_mod_u"])

    def test_two_direct_truncation_routes_charge_n_squared(self) -> None:
        self.assertEqual(self.controls["pair_evaluation_count"], 8922)
        self.assertEqual(self.controls["global_degree_precision_product_sum"], 8922)
        self.assertEqual(self.controls["crt_degree_precision_product_sum"], 8922)
        self.assertEqual(self.cost["moroz_single_expansion_total_exponent_B"]["exact"], "9/2")
        self.assertEqual(self.cost["moroz_squarefree_crt_total_exponent_B"]["exact"], "9/2")

    def test_represented_target_chow_reaches_rho(self) -> None:
        self.assertEqual(self.controls["target_chow_coefficient_slot_sum"], 204)
        self.assertEqual(self.cost["represented_target_chow_exponent_B"]["exact"], "5/2")
        self.assertFalse(self.cost["represented_target_chow_strictly_below_rho"])

    def test_factored_dynamic_evaluation_remains_open(self) -> None:
        self.assertTrue(
            self.cost["conditional_factored_dynamic_evaluation_strictly_below_rho"]
        )
        self.assertFalse(self.cost["factored_dynamic_evaluation_constructor_supplied"])
        self.assertIn("factored arbitrary-squarefree dynamic-evaluation", self.report["next_action"])

    def test_admission_is_scoped(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 11)
        self.assertEqual(admission["obligation_count"], 20)
        self.assertTrue(admission["xadic_applicability_scope_admitted"])
        self.assertTrue(admission["direct_truncated_resultant_route_closed"])
        self.assertFalse(admission["factored_dynamic_evaluation_constructor_admitted"])
        self.assertFalse(admission["lane_admitted"])

    def test_no_lower_bound_or_attack_credit(self) -> None:
        self.assertFalse(
            self.cost["standard_route_negative_claimed_as_circuit_lower_bound"]
        )
        self.assertIn("not a lower bound", self.report["theorem"]["scope"])
        self.assertFalse(self.controls["finite_pair_scan_receives_asymptotic_credit"])
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])

    def test_no_breakthrough_flags(self) -> None:
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("DIRECT_TRUNCATED_RESULTANT_ROUTE_CLOSED", self.report["classification"])

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R179.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "applicability"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
