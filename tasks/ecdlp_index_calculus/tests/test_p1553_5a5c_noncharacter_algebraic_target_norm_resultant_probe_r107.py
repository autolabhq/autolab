from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "p1553_5a5c_noncharacter_algebraic_target_norm_"
    "resultant_probe_r107.py"
)


def load_producer():
    spec = importlib.util.spec_from_file_location("p1553_r107_test", PRODUCER)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R107 producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R107 = load_producer()


class NoncharacterAlgebraicTargetNormResultantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R107.build_bundle()

    def test_all_root_partitions_have_b13o5_minimum(self) -> None:
        ledger = R107.root_partition_ledger()
        self.assertEqual(ledger["partition_count"], 34)
        self.assertEqual(
            ledger["minimum_interface_exponent_B"]["exact"],
            "13/5",
        )
        self.assertEqual(ledger["minimizer_count"], 4)
        self.assertTrue(ledger["balanced_mixed_split_present"])
        self.assertTrue(ledger["minimum_interface_above_setup_cap"])
        self.assertTrue(ledger["minimum_interface_above_online_cap"])

    def test_actual_and_matched_partition_weight_identities(self) -> None:
        controls = R107.actual_and_matched_controls()
        self.assertEqual(controls["actual_count"], 8)
        self.assertEqual(controls["matched_random_deck_count"], 8)
        self.assertEqual(controls["query_count"], 80)
        self.assertTrue(controls["all_partition_weight_identities_exact"])
        self.assertTrue(controls["all_markers_recover_sources"])
        self.assertTrue(controls["all_recovered_sources_replay"])

    def test_actual_double_fibers_refute_constant_normalization(self) -> None:
        controls = R107.actual_and_matched_controls()
        actual_double = [
            row
            for row in controls["double_fiber_queries"]
            if row["control_class"] == "actual"
        ]
        self.assertEqual(len(actual_double), 2)
        self.assertTrue(
            all(
                row["canonical_count"] == 2
                and len(set(row["partition_weights"])) == 2
                and row["balanced_resultant_vanishing_order"]
                == sum(row["partition_weights"])
                for row in actual_double
            )
        )
        self.assertTrue(controls["actual_nonuniform_double_fiber_present"])

    def test_cyclic_resultant_jet_replays_duplicate_roots(self) -> None:
        control = R107.cyclic_resultant_jet_control()
        self.assertTrue(control["duplicate_roots_present"])
        self.assertTrue(control["order_equals_convolution_count"])
        self.assertTrue(control["first_nonzero_coefficient_nonzero"])

    def test_collapsed_deck_blocks_fixed_generic_jet_order(self) -> None:
        control = R107.collapsed_deck_multiplicity_control()
        self.assertTrue(control["weighted_identity_exact"])
        self.assertTrue(control["fixed_order_two_jet_insufficient"])
        self.assertEqual(
            control["asymptotic_canonical_multiplicity_exponent_B"]["exact"],
            "5",
        )

    def test_standard_resultant_scope_is_closed_only(self) -> None:
        ledger = R107.algebraic_grammar_ledger()
        scope = ledger["scope_boundary"]
        self.assertTrue(scope["standard_explicit_resultant_grammar_closed"])
        self.assertFalse(
            scope["arbitrary_factored_algebraic_or_rational_circuit_closed"]
        )
        self.assertFalse(scope["general_arithmetic_circuit_lower_bound_claimed"])
        self.assertTrue(scope["elliptic_lambda_ring_or_chow_recurrence_open"])

    def test_bundle_preserves_nonclaim_and_routes_factored_recurrence(
        self,
    ) -> None:
        report = self.bundle["report"]
        admission = report["admission"]
        self.assertFalse(admission["lane_admitted"])
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 27)
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])
        self.assertIn("lambda-ring/Chow-form", report["next_action"])
        self.assertIn(
            "No general arithmetic-circuit lower bound is proved.",
            report["non_claims"],
        )


if __name__ == "__main__":
    unittest.main()
