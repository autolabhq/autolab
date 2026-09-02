from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "p1553_5a5c_multiedge_digitized_equality_projector_probe_r99.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r99", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R99 probe")
R99 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R99)


class MultiEdgeDigitizedEqualityProjectorTests(unittest.TestCase):
    def test_radix_digits_are_injective_and_equality_is_exact(
        self,
    ) -> None:
        controls = R99.digit_channel_controls()
        self.assertTrue(controls["all_digit_encodings_injective"])
        self.assertTrue(controls["all_digit_equalities_exact"])
        self.assertTrue(controls["all_alphabet_products_cover_field"])

    def test_fiber_indicator_interpolation_is_exact_full_degree(
        self,
    ) -> None:
        controls = R99.digit_channel_controls()
        self.assertTrue(controls["all_fiber_interpolations_exact"])
        self.assertTrue(
            controls["all_fiber_indicators_degree_p_minus_1"]
        )
        for sweep in controls["sweep"]:
            for row in sweep["fiber_rows"]:
                self.assertEqual(
                    row["interpolation_degree"], sweep["prime"] - 1
                )

    def test_fiber_root_lists_partition_each_digit_channel(
        self,
    ) -> None:
        for row in R99.digit_channel_controls()["sweep"]:
            self.assertEqual(
                row["all_fiber_root_words"],
                row["prime"] * row["bit_count"],
            )
            self.assertGreaterEqual(
                row["alphabet_product"], row["prime"]
            )

    def test_duplicate_occurrence_count_and_source_are_exact(
        self,
    ) -> None:
        controls = R99.occurrence_digit_controls()
        self.assertEqual(controls["positive_integer_count"], 2)
        self.assertEqual(controls["duplicate_source_indices"], [0, 1])
        self.assertTrue(
            controls["positive_dyadic_source"][
                "returned_source_matches_target"
            ]
        )

    def test_blind_control_returns_bottom(self) -> None:
        controls = R99.occurrence_digit_controls()
        self.assertEqual(controls["blind_integer_count"], 0)
        self.assertTrue(
            controls["blind_dyadic_source"]["returned_bottom"]
        )
        self.assertIsNone(
            controls["blind_dyadic_source"]["source_index"]
        )

    def test_standard_digit_constructors_miss_caps(self) -> None:
        costs = R99.asymptotic_cost_control()
        full = costs["full_field_materialized_constructors"]
        source = costs["sourcewise_digit_extraction"]
        self.assertEqual(full["exponent_B"]["exact"], "5")
        self.assertEqual(source["exponent_B"]["exact"], "12/5")
        self.assertFalse(full["inside_setup_cap"])
        self.assertFalse(source["inside_setup_cap"])
        self.assertFalse(source["inside_online_cap"])
        self.assertFalse(
            costs[
                "succinct_aggregate_digit_index_from_compact_divisors_supplied"
            ]
        )

    def test_bundle_credits_representation_but_not_constructor(
        self,
    ) -> None:
        report = R99.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"], 15
        )
        self.assertEqual(report["admission"]["obligation_count"], 31)
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertIn("positively verifies", report["scope_boundary"])
        self.assertIn("aggregate digit trie", report["next_action"])


if __name__ == "__main__":
    unittest.main()
