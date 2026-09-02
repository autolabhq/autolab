from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT / "p1553_torus_c5_all_nonzero_path_product_probe_r136.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r136", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R136 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R136)


class TorusC5AllNonzeroPathProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R136.build_bundle()
        cls.report = cls.bundle["report"]
        cls.theorem = cls.report["theorem"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.bundle["cost"]["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R136.verify_source_bindings()), 10)

    def test_polynomial_product_and_union_identity(self) -> None:
        field = R136.Field(R136.R82.FAMILIES[0]["field_prime"])
        roots = (field.elt(2), field.elt(3), field.elt(5))
        factors = [R136.linear_factor(root, field) for root in roots]
        product = R136.polynomial_product(factors, field)
        self.assertEqual(len(product) - 1, 3)
        for value in roots + (field.elt(7),):
            self.assertEqual(
                R136.polynomial_eval(product, value, field) == field.zero,
                any(
                    R136.polynomial_eval(factor, value, field)
                    == field.zero
                    for factor in factors
                ),
            )

    def test_all_nonzero_path_dichotomy_is_explicit(self) -> None:
        statement = self.theorem["all_nonzero_path_product"]
        self.assertIn("leaf rejects", statement)
        self.assertIn("leaf accepts", statement)
        self.assertIn("union of the root sets", statement)
        self.assertIn(
            "M>6",
            self.theorem["deterministic_structured_dichotomy"],
        )

    def test_random_model_consequence_preserves_low_slp_escape(self) -> None:
        self.assertEqual(
            self.theorem[
                "random_model_path_product_mode_count_excluded"
            ],
            "(5-o(1))*log2(q)",
        )
        self.assertIn(
            "Omega(log log q)",
            self.theorem["random_deck_amplification"],
        )
        self.assertFalse(self.theorem["high_expansion_low_slp_covered"])

    def test_actual_path_control_count(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertEqual(self.controls["active_color_control_count"], 30)
        self.assertEqual(
            self.controls["available_path_control_count"],
            12,
        )

    def test_five_factor_products_leave_one_positive(self) -> None:
        self.assertTrue(
            self.controls["all_available_product_mode_bounds_respected"]
        )
        self.assertTrue(
            self.controls["all_available_products_leave_one_positive"]
        )
        for control in self.controls["controls"]:
            for color in control["color_controls"]:
                if not color["witness_available"]:
                    continue
                self.assertEqual(color["factor_count"], 5)
                self.assertEqual(color["factor_mode_counts"], [2] * 5)
                self.assertEqual(color["product_degree"], 5)
                self.assertEqual(color["product_mode_count"], 6)
                self.assertEqual(
                    color["product_zero_mask_on_progression"],
                    [True, True, True, True, True, False],
                )

    def test_positive_and_inverse_empty_share_all_nonzero_path(self) -> None:
        self.assertTrue(
            self.controls[
                "all_available_inverse_empties_share_all_nonzero_path"
            ]
        )

    def test_product_union_identity_and_sources_replay(self) -> None:
        self.assertTrue(
            self.controls["all_available_union_identities_exact"]
        )
        self.assertTrue(
            self.controls[
                "all_available_sources_replay_and_are_accepted"
            ]
        )

    def test_surviving_selector_classes_remain_open(self) -> None:
        for route_id in (
            "growing_expanded_support_low_slp_dag",
            "nonzero_value_frobenius_coordinate_dag",
            "structured_seven_plus_mode_path_product",
        ):
            self.assertEqual(self.routes[route_id]["status"], "open")

    def test_gate_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 13)
        self.assertEqual(admission["obligation_count"], 21)
        self.assertTrue(
            admission[
                "structured_six_mode_path_product_negative_admitted"
            ]
        )
        self.assertFalse(
            admission["growing_support_or_nonzero_value_selector_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("growing-support", self.report["next_action"])


if __name__ == "__main__":
    unittest.main()
