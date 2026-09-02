from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "p1553_5a5c_succinct_aggregate_digit_trie_probe_r100.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r100", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R100 probe")
R100 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R100)


class SuccinctAggregateDigitTrieTests(unittest.TestCase):
    def test_exact_subset_state_counts_and_word_bounds(self) -> None:
        controls = R100.arbitrary_set_information_controls()
        self.assertTrue(controls["all_finite_word_bounds_exact"])
        for row in controls["finite_sweep"]:
            self.assertEqual(
                row["possible_distinct_subsets"],
                R100.math.comb(row["prime"], row["subset_size"]),
            )
            words = row["minimum_field_words"]
            self.assertLess(
                row["prime"] ** max(words - 1, 0),
                row["possible_distinct_subsets"],
            )
            self.assertGreaterEqual(
                row["prime"] ** words,
                row["possible_distinct_subsets"],
            )

    def test_patricia_and_explicit_leaf_state_are_linear(self) -> None:
        for row in R100.arbitrary_set_information_controls()[
            "finite_sweep"
        ]:
            self.assertEqual(
                row["explicit_leaf_records"], row["subset_size"]
            )
            self.assertEqual(
                row["patricia_total_nodes"],
                2 * row["subset_size"] - 1,
            )
            self.assertGreaterEqual(
                row["binary_trie_nodes"], row["subset_size"] + 1
            )

    def test_campaign_entropy_substitution_is_exact(self) -> None:
        substitution = R100.arbitrary_set_information_controls()[
            "campaign_substitution"
        ]
        self.assertEqual(substitution["log_p_D"]["exact"], "12/25")
        self.assertEqual(
            substitution["constant_fraction_of_D"]["exact"], "13/25"
        )
        self.assertIn("B^(12/5)", substitution["minimum_field_words"])

    def test_structured_interval_is_constant_state_positive(self) -> None:
        controls = R100.structured_and_occurrence_controls()
        self.assertTrue(
            controls[
                "structured_interval_is_constant_state_counterexample"
            ]
        )
        self.assertEqual(
            controls["structured_interval_positive"]["summary_words"], 2
        )
        self.assertEqual(
            controls["structured_interval_positive"]["source_index"], 7
        )
        self.assertTrue(
            controls["structured_interval_blind"]["returned_bottom"]
        )

    def test_duplicate_occurrence_payload_is_charged(self) -> None:
        controls = R100.structured_and_occurrence_controls()
        self.assertEqual(controls["positive_integer_count"], 2)
        self.assertEqual(controls["positive_source_index"], 0)
        self.assertEqual(
            controls["duplicate_occurrence_payload_words"],
            len(controls["occurrence_list"]),
        )
        self.assertTrue(
            controls[
                "occurrence_payload_needed_for_complete_source_return"
            ]
        )

    def test_universal_index_misses_setup_cap(self) -> None:
        costs = R100.asymptotic_cost_control()
        universal = costs["arbitrary_D_subset_exact_index"]
        self.assertEqual(universal["exponent_B"]["exact"], "12/5")
        self.assertFalse(universal["inside_setup_cap"])
        self.assertTrue(
            costs["structured_interval_positive_control"][
                "inside_direct_caps"
            ]
        )
        self.assertFalse(
            costs["r84_smaller_side"][
                "actual_divisor_image_short_generator_proved"
            ]
        )

    def test_bundle_preserves_actual_image_boundary(self) -> None:
        report = R100.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"], 16
        )
        self.assertEqual(report["admission"]["obligation_count"], 31)
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertIn("interval positive control", report["scope_boundary"])
        self.assertIn("actual-image entropy", report["next_action"])


if __name__ == "__main__":
    unittest.main()
