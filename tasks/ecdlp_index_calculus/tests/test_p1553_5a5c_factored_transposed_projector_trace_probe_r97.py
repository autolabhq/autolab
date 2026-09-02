from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "p1553_5a5c_factored_transposed_projector_trace_probe_r97.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r97", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R97 probe")
R97 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R97)


class FactoredTransposedProjectorTraceTests(unittest.TestCase):
    def test_projector_and_derivative_identities(self) -> None:
        prime = 101
        for value in range(prime):
            expected = int(value == 0)
            self.assertEqual(R97.zero_projector(value, prime), expected)
            derivative = R97.zero_projector_derivative(value, prime)
            if value == 0:
                self.assertEqual(derivative, 0)
            else:
                self.assertNotEqual(derivative, 0)

    def test_blind_jacobian_and_dyadic_adjoints_have_full_rank(
        self,
    ) -> None:
        controls = R97.adjoint_rank_controls()
        self.assertTrue(controls["all_jacobians_full_rank"])
        self.assertTrue(
            controls["all_dyadic_adjoint_families_full_rank"]
        )
        for row in controls["blind_nonzero_sweep"]:
            self.assertEqual(row["jacobian_rank"], row["dimension"])
            self.assertEqual(
                row["dyadic_adjoint_rank"], row["dimension"]
            )
            self.assertEqual(
                row["dyadic_mask_count"],
                row["expected_full_binary_mask_count"],
            )

    def test_unique_zero_projector_and_dyadic_source_are_exact(
        self,
    ) -> None:
        unique = R97.source_and_gradient_controls()["unique_zero"]
        self.assertEqual(unique["integer_projector_count"], 1)
        self.assertEqual(unique["dyadic_source"]["source_index"], 2)
        self.assertTrue(
            unique["dyadic_source"]["returned_source_is_zero"]
        )
        self.assertFalse(
            unique["zero_coordinate_has_nonzero_projector_gradient"]
        )

    def test_unique_product_gradient_localizes_zero(self) -> None:
        unique = R97.source_and_gradient_controls()["unique_zero"]
        self.assertEqual(
            unique["product_gradient_nonzero_indices"], [2]
        )
        self.assertTrue(
            unique["product_gradient_localizes_unique_zero"]
        )

    def test_two_zero_product_gradient_collapses(self) -> None:
        duplicate = R97.source_and_gradient_controls()["two_zeros"]
        self.assertEqual(duplicate["integer_projector_count"], 2)
        self.assertTrue(duplicate["product_gradient_is_zero_vector"])
        self.assertFalse(any(duplicate["product_gradient"]))

    def test_blind_control_returns_bottom(self) -> None:
        blind = R97.source_and_gradient_controls()["blind_nonzero"]
        self.assertEqual(blind["integer_projector_count"], 0)
        self.assertTrue(blind["dyadic_source"]["returned_bottom"])
        self.assertIsNone(blind["dyadic_source"]["source_index"])

    def test_standard_transpose_misses_both_caps(self) -> None:
        costs = R97.asymptotic_cost_control()
        adjoint = costs["full_transposed_adjoint"]
        product = costs["balanced_product_tree"]
        self.assertEqual(
            adjoint["state_exponent_B"]["exact"], "12/5"
        )
        self.assertFalse(adjoint["inside_setup_cap"])
        self.assertFalse(product["inside_setup_cap"])
        self.assertFalse(product["inside_online_cap"])
        self.assertFalse(
            costs["compact_nonlinear_tensor_tower_trace_supplied"]
        )

    def test_bundle_preserves_nonlinear_tensor_tower_boundary(
        self,
    ) -> None:
        report = R97.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"], 13
        )
        self.assertEqual(report["admission"]["obligation_count"], 28)
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertIn("not an asymptotic lower bound", report["scope_boundary"])
        self.assertIn("nonlinear tensor-tower", report["next_action"])


if __name__ == "__main__":
    unittest.main()
