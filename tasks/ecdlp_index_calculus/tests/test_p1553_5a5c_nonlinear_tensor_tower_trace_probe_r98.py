from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "p1553_5a5c_nonlinear_tensor_tower_trace_probe_r98.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r98", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R98 probe")
R98 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R98)


class NonlinearTensorTowerTraceTests(unittest.TestCase):
    def test_monic_quadratic_resultant_is_squared_difference(
        self,
    ) -> None:
        for prime in (3, 5, 7, 11, 13):
            for left in range(prime):
                for right in range(prime):
                    self.assertEqual(
                        R98.monic_quadratic_resultant(
                            left, right, prime
                        ),
                        (right - left) ** 2 % prime,
                    )

    def test_resultant_projector_is_equality_kernel(self) -> None:
        for prime in (3, 5, 7, 11, 13):
            labels = list(range(prime))
            self.assertEqual(
                R98.resultant_projector_kernel(labels, prime),
                R98.equality_kernel(labels, prime),
            )

    def test_full_field_one_bond_rank_is_prime(self) -> None:
        controls = R98.separation_rank_controls()
        self.assertTrue(controls["all_full_field_kernels_rank_p"])
        for row in controls["full_field_sweep"]:
            self.assertEqual(row["equality_rank"], row["prime"])
            self.assertEqual(
                row["resultant_projector_rank"], row["prime"]
            )
            self.assertTrue(row["one_hot_factorization_exact"])

    def test_restricted_distinct_message_rank_is_dimension(
        self,
    ) -> None:
        controls = R98.separation_rank_controls()
        self.assertTrue(controls["all_restricted_kernels_full_rank"])
        for row in controls["restricted_distinct_message_sweep"]:
            self.assertEqual(row["rank"], row["distinct_label_count"])

    def test_duplicate_occurrence_count_and_source_are_exact(
        self,
    ) -> None:
        controls = R98.occurrence_source_controls()
        self.assertEqual(controls["positive_integer_count"], 2)
        self.assertEqual(controls["duplicate_source_indices"], [0, 1])
        self.assertTrue(
            controls["positive_dyadic_source"][
                "returned_source_matches_target"
            ]
        )
        self.assertTrue(
            controls["occurrence_count_exceeds_value_collapsed_count"]
        )

    def test_blind_control_returns_bottom(self) -> None:
        controls = R98.occurrence_source_controls()
        self.assertEqual(controls["blind_integer_count"], 0)
        self.assertTrue(
            controls["blind_dyadic_source"]["returned_bottom"]
        )
        self.assertIsNone(
            controls["blind_dyadic_source"]["source_index"]
        )

    def test_one_bond_and_restricted_widths_miss_caps(self) -> None:
        costs = R98.asymptotic_cost_control()
        full = costs["full_field_one_bond_tensor"]
        restricted = costs["restricted_distinct_source_messages"]
        self.assertEqual(
            full["state_and_contraction_exponent_B"]["exact"], "5"
        )
        self.assertEqual(
            restricted["state_and_contraction_exponent_B"]["exact"],
            "12/5",
        )
        self.assertFalse(full["inside_setup_cap"])
        self.assertFalse(full["inside_online_cap"])
        self.assertFalse(restricted["inside_setup_cap"])
        self.assertFalse(restricted["inside_online_cap"])

    def test_bundle_preserves_multi_edge_digitized_boundary(
        self,
    ) -> None:
        report = R98.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"], 14
        )
        self.assertEqual(report["admission"]["obligation_count"], 30)
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertIn("not a lower bound", report["scope_boundary"])
        self.assertIn("multi-edge digitized", report["next_action"])


if __name__ == "__main__":
    unittest.main()
