from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_resultant_valuation_trace_grammar_r73.py"
SPEC = importlib.util.spec_from_file_location("p1553_r73", MODULE_PATH)
assert SPEC and SPEC.loader
R73 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R73)


class ResultantValuationTraceGrammarTests(unittest.TestCase):
    def test_root_valuation_preserves_duplicate_occurrence_count(self) -> None:
        first = R73.pair_occurrences([2, 2], [3])
        second = R73.pair_occurrences([5, 5], [7])
        resultant = R73.product_resultant_polynomial(first, second)
        target = 2 * 3 * 5 * 7 % R73.FIELD_PRIME
        query = R73.resultant_query(first, second, resultant, target)
        self.assertEqual(query["valuation_count"], 4)
        self.assertEqual(query["direct_count"], 4)
        self.assertIsNotNone(query["source"])

    def test_zero_strata_and_rank_two_replay_are_exact(self) -> None:
        payloads = R73.build_payloads()
        control = payloads[str(R73.RANK_TWO_PATH)]
        mutation = next(
            row
            for row in control["instances"]
            if row["instance_id"] == "zero_signature_mutation"
        )
        self.assertGreater(mutation["zero_stratum_count"], 0)
        self.assertTrue(mutation["count_matches"])
        self.assertTrue(control["all_exact_counts_match"])
        self.assertTrue(control["all_query_valuations_match"])

    def test_cost_gate_rejects_only_frozen_grammar(self) -> None:
        payloads = R73.build_payloads()
        report = payloads[str(R73.REPORT_PATH)]
        ledger = payloads[str(R73.COST_LEDGER_PATH)]
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(ledger["lane_inside_caps"])
        self.assertTrue(ledger["not_a_lower_bound"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
