from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "p1553_5a5c_target_uniform_precoefficient_circuit_probe_r85.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r85", MODULE_PATH)
assert SPEC and SPEC.loader
R85 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R85)


class TargetUniformPreCoefficientCircuitTests(unittest.TestCase):
    def test_prime_order_theorem_is_not_homomorphism_limited(self) -> None:
        theorem = R85.target_equivariant_fiber_theorem()
        self.assertFalse(theorem["requires_pi_to_be_a_homomorphism"])
        self.assertIn("injective or constant", theorem["prime_order_consequence"])

    def test_composite_quotient_positive_control(self) -> None:
        control = R85.composite_order_positive_control()
        self.assertTrue(control["exact_target_action"])
        self.assertTrue(control["proper_compression"])
        self.assertEqual(control["proper_kernel_size"], 101)

    def test_best_binary_payload_misses_both_caps(self) -> None:
        ledger = R85.split_payload_ledger()
        self.assertAlmostEqual(
            ledger["minimum_larger_payload_exponent_B"],
            2.6,
        )
        self.assertAlmostEqual(
            ledger["corresponding_smaller_payload_exponent_B"],
            2.4,
        )
        self.assertFalse(ledger["smaller_payload_inside_setup_cap"])
        self.assertFalse(ledger["smaller_payload_inside_online_cap"])

    def test_small_bundle_routes_to_sparse_target_specialization(self) -> None:
        bundle = R85.build_bundle(
            families=R85.R82.FAMILIES[:1],
            offsets=(0,),
        )
        report = bundle["report"]
        self.assertTrue(
            report["aggregate"]["all_fixed_label_controls_match_theorem"]
        )
        self.assertFalse(
            report["aggregate"][
                "proper_fixed_compressing_target_action_found"
            ]
        )
        self.assertIn("sparse multihomogeneous", report["next_action"])
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["factor_log_solve_complete"])
        self.assertFalse(report["fresh_target_descent_complete"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
