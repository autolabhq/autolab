from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "p1553_5a5c_modular_frobenius_trace_recurrence_probe_r96.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r96", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R96 probe")
R96 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R96)


class ModularFrobeniusTraceRecurrenceTests(unittest.TestCase):
    def test_root_polynomial_vanishes_with_occurrence_multiplicity(
        self,
    ) -> None:
        prime = 11
        roots = [1, 1, 2, 4, 7]
        polynomial = R96.polynomial_from_roots(roots, prime)
        self.assertEqual(len(polynomial) - 1, len(roots))
        for root in set(roots):
            self.assertEqual(
                R96.poly_mod(polynomial, [-root, 1], prime),
                [0],
            )

    def test_reduced_split_trace_is_exact_and_frobenius_identity(
        self,
    ) -> None:
        reduced = R96.quotient_trace_controls()[
            "reduced_split_quotient"
        ]
        self.assertTrue(reduced["trace_equals_direct_integer"])
        self.assertEqual(reduced["direct_integer_zero_count"], 2)
        self.assertTrue(reduced["frobenius_is_identity"])
        self.assertEqual(
            reduced["frobenius_rank"],
            reduced["quotient_dimension"],
        )

    def test_nonreduced_trace_counts_duplicate_but_frobenius_drops_rank(
        self,
    ) -> None:
        controls = R96.quotient_trace_controls()
        nonreduced = controls["nonreduced_occurrence_quotient"]
        self.assertTrue(nonreduced["trace_equals_direct_integer"])
        self.assertEqual(nonreduced["direct_integer_zero_count"], 3)
        self.assertTrue(
            controls["radicalization_loses_duplicate_zero_occurrence"]
        )
        self.assertTrue(
            controls["nonreduced_frobenius_loses_nilpotent_rank"]
        )
        self.assertEqual(nonreduced["frobenius_rank"], 4)
        self.assertEqual(nonreduced["quotient_dimension"], 5)

    def test_blind_projector_trace_is_exact_zero(self) -> None:
        blind = R96.quotient_trace_controls()["blind_split_quotient"]
        self.assertEqual(blind["direct_integer_zero_count"], 0)
        self.assertEqual(blind["projector_trace_mod_prime"], 0)
        self.assertTrue(blind["trace_equals_direct_integer"])

    def test_dyadic_trace_returns_an_exact_occurrence(self) -> None:
        source = R96.quotient_trace_controls()["dyadic_source"]
        self.assertTrue(source["all_trace_counts_exact"])
        self.assertTrue(source["returned_source_is_zero"])
        self.assertEqual(source["source_index"], 0)
        self.assertTrue(
            source["query_dimension_sum_below_three_root_bodies"]
        )

    def test_standard_quotient_and_matrix_miss_caps(self) -> None:
        costs = R96.asymptotic_cost_control()
        quotient = costs["source_complete_split_quotient"]
        matrix = costs["explicit_multiplication_or_frobenius_matrix"]
        self.assertEqual(
            quotient["basis_dimension_exponent_B"]["exact"],
            "12/5",
        )
        self.assertFalse(quotient["inside_setup_cap"])
        self.assertFalse(quotient["inside_online_cap"])
        self.assertEqual(matrix["state_exponent_B"]["exact"], "24/5")
        self.assertFalse(matrix["inside_setup_cap"])

    def test_bundle_preserves_factored_trace_boundary(self) -> None:
        report = R96.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"],
            13,
        )
        self.assertEqual(report["admission"]["obligation_count"], 27)
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertIn("not a lower bound", report["scope_boundary"])
        self.assertIn(
            "factored transposed projector-trace",
            report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
