from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "p1553_5a5c_aggregate_veronese_projector_recurrence_probe_r95.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r95", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R95 probe")
R95 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R95)


class AggregateVeroneseProjectorRecurrenceTests(unittest.TestCase):
    def test_resultant_has_plucker_discriminant_form(self) -> None:
        prime = 29
        for left in ((1, 2, 3), (4, 5, 6), (0, 1, 1)):
            for right in ((7, 8, 9), (1, 0, 2), (3, 4, 0)):
                a, b, c = left
                d, e, f = right
                plucker = (
                    (a * f - c * d) ** 2
                    - (a * e - b * d) * (b * f - c * e)
                ) % prime
                self.assertEqual(
                    R95.quadratic_resultant(left, right, prime),
                    plucker,
                )

    def test_coefficient_pairing_is_full_on_frozen_sweep(self) -> None:
        sweep = R95.coefficient_rank_sweep()
        self.assertEqual(
            [row["field_prime"] for row in sweep],
            [3, 5, 7, 11, 13, 17, 19, 29],
        )
        self.assertTrue(all(row["full_rank"] for row in sweep))
        self.assertEqual(
            [row["coefficient_pairing_rank"] for row in sweep],
            [15, 45, 91, 231, 325, 561, 703, 1653],
        )

    def test_veronese_quotient_dimension_formula_is_exact(self) -> None:
        for row in R95.coefficient_rank_sweep():
            prime = row["field_prime"]
            self.assertEqual(
                row["veronese_quotient_dimension"],
                prime * (2 * prime - 1),
            )
            self.assertEqual(
                row["left_monomial_count"],
                row["veronese_quotient_dimension"],
            )
            self.assertEqual(
                row["right_monomial_count"],
                row["veronese_quotient_dimension"],
            )

    def test_aggregate_count_and_controls_are_exact(self) -> None:
        replay = R95.toy_source_replay()
        self.assertTrue(
            replay["aggregate_count"][
                "aggregate_count_equals_direct_integer"
            ]
        )
        self.assertTrue(replay["blind_control"]["blind_zero_exact"])
        for quadratic in replay["blind_control"]["right_quadratics"]:
            a, b, c = quadratic
            self.assertEqual(a, 1)
            self.assertNotEqual((b * b - 4 * a * c) % 11, 0)
        self.assertTrue(
            replay["duplicate_occurrence_control"][
                "duplicate_occurrence_equality_exact"
            ]
        )

    def test_dyadic_aggregate_route_returns_a_zero_source(self) -> None:
        source = R95.toy_source_replay()["dyadic_source"]
        self.assertTrue(source["all_counts_exact"])
        self.assertTrue(source["returned_source_is_zero"])
        self.assertIsNotNone(source["source"])

    def test_canonical_moment_state_is_b10_and_over_caps(self) -> None:
        control = R95.asymptotic_cost_control()
        moments = control["veronese_quotient_moment_coordinates"]
        self.assertEqual(moments["state_exponent_B"]["exact"], "10")
        self.assertFalse(moments["inside_setup_cap"])
        self.assertFalse(moments["inside_online_cap"])
        self.assertFalse(control["nonlinear_nonmoment_recurrence_refuted"])

    def test_bundle_preserves_nonmoment_recurrence_boundary(self) -> None:
        report = R95.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"],
            11,
        )
        self.assertEqual(report["admission"]["obligation_count"], 25)
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertIn("not an asymptotic lower bound", report["scope_boundary"])
        self.assertIn("modular trace recurrence", report["next_action"])


if __name__ == "__main__":
    unittest.main()
