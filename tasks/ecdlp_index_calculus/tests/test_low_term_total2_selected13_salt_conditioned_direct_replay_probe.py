from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import sys
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "low_term_total2_selected13_salt_conditioned_direct_replay_probe.py"
SPEC = importlib.util.spec_from_file_location("low_term_direct_replay", MODULE_PATH)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ProbeTweetIntakeTests(unittest.TestCase):
    def test_build_tweet_source_embeds_known_text(self) -> None:
        source = PROBE.build_tweet_source(
            "https://x.com/askalphaxiv/status/2076737985559822734?s=46"
        )["tweet_source"]

        self.assertTrue(source["tweet_text_included"])
        self.assertTrue(source["tweet_summary_is_verbatim"])
        self.assertEqual(source["tweet_intake_mode"], "public_snapshot_summary")
        self.assertEqual(source["tweet_text"], source["tweet_summary"])
        self.assertEqual(source["tweet_author"], "askalphaxiv")
        self.assertEqual(source["tweet_post_id"], "2076737985559822734")
        expected_text = PROBE.KNOWN_TWEET_SOURCE_TEXT["askalphaxiv:2076737985559822734"]
        self.assertEqual(source["tweet_text"], expected_text)
        self.assertEqual(
            source["tweet_text_sha256"],
            hashlib.sha256(expected_text.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            source["tweet_posted_at"],
            "2026-07-13T18:38:14+00:00",
        )
        self.assertEqual(
            source["tweet_source_title"],
            "Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in LLM Fine-tuning",
        )
        self.assertEqual(
            source["tweet_source_url"],
            "https://arxiv.org/abs/2607.08393",
        )
        self.assertEqual(
            source["tweet_media_urls"],
            [
                "https://video.twimg.com/amplify_video/2076736796508192768/vid/avc1/3448x1914/"
                "vhhIWh90nyKEtBhc.mp4"
            ],
        )
        self.assertEqual(source["tweet_media_types"], ["video"])
        self.assertEqual(source["tweet_media_count"], 1)
        self.assertTrue(source["tweet_has_media"])
        self.assertEqual(source["tweet_hashtags"], [])

    def test_parse_note_url_normalizes_known_variants(self) -> None:
        aliases = [
            "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
            "http://x.com/askalphaxiv/status/2076737985559822734?s=46",
            "x.com/askalphaxiv/status/2076737985559822734?s=46",
            "www.x.com/askalphaxiv/status/2076737985559822734",
            "https://x.com/status/2076737985559822734?s=46",
            "https://x.com/i/web/status/2076737985559822734?s=46",
            "http://x.com/i/status/2076737985559822734?s=46",
            "https://mobile.x.com/askalphaxiv/status/2076737985559822734?s=46",
            "https://twitter.com/askalphaxiv/status/2076737985559822734?s=46",
        ]
        for alias in aliases:
            parsed = PROBE.parse_note_url(alias)
            source = PROBE.build_tweet_source(alias)["tweet_source"]
            self.assertEqual(parsed["source_post_id"], "2076737985559822734")
            if "?" in alias:
                self.assertEqual(parsed["source_query"], "?s=46")
            else:
                self.assertEqual(parsed["source_query"], "")
            self.assertEqual(parsed["source_author"], "askalphaxiv")
            self.assertEqual(source["tweet_post_id"], "2076737985559822734")
            self.assertEqual(source["tweet_author"], "askalphaxiv")
            self.assertTrue(source["tweet_text_included"])

    def test_build_tweet_source_falls_back_for_unknown_post(self) -> None:
        source = PROBE.build_tweet_source(
            "https://x.com/altstatus/status/9999999999999999999?s=46"
        )["tweet_source"]

        self.assertFalse(source["tweet_text_included"])
        self.assertEqual(source["tweet_summary_is_verbatim"], False)
        self.assertEqual(source["tweet_intake_mode"], "fallback_summary_text")
        self.assertEqual(source["tweet_author"], "altstatus")
        self.assertEqual(source["tweet_post_id"], "9999999999999999999")
        self.assertEqual(source["tweet_query"], "?s=46")
        self.assertFalse(source["tweet_text"])
        self.assertNotEqual(source["tweet_summary"], "")
        self.assertEqual(source["tweet_posted_at"], "")
        self.assertEqual(source["tweet_source_title"], "")
        self.assertEqual(source["tweet_source_url"], "")
        self.assertEqual(source["tweet_media_urls"], [])
        self.assertEqual(source["tweet_media_types"], [])
        self.assertEqual(source["tweet_media_count"], 0)
        self.assertFalse(source["tweet_has_media"])
        self.assertEqual(source["tweet_hashtags"], [])

    def test_default_top_k_is_five(self) -> None:
        previous = sys.argv[:]
        try:
            sys.argv = ["script"]
            args = PROBE.parse_args()
        finally:
            sys.argv = previous
        self.assertEqual(args.top_k, 5)

    def test_default_selection_strategy_is_scan_until_relation(self) -> None:
        previous = sys.argv[:]
        try:
            sys.argv = ["script"]
            args = PROBE.parse_args()
        finally:
            sys.argv = previous
        self.assertEqual(args.selection_strategy, "scan_until_relation")

    def test_default_common_sweep_source(self) -> None:
        previous = sys.argv[:]
        try:
            sys.argv = ["script"]
            args = PROBE.parse_args()
        finally:
            sys.argv = previous
        self.assertEqual(args.common_sweep_source, PROBE.DEFAULT_COMMON_SWEEP_SOURCE)

    def test_default_term_shape_filters(self) -> None:
        previous = sys.argv[:]
        try:
            sys.argv = ["script"]
            args = PROBE.parse_args()
        finally:
            sys.argv = previous
        self.assertEqual(args.term_shape_filters, PROBE.DEFAULT_TERM_SHAPE_FILTERS)

    def test_parse_term_shape_filters(self) -> None:
        previous = sys.argv[:]
        try:
            sys.argv = ["script", "--term-shape-filters", "2+2,3+1, 4+2"]
            args = PROBE.parse_args()
        finally:
            sys.argv = previous
        self.assertEqual(args.term_shape_filters, ("2+2", "3+1", "4+2"))

    def test_parse_args_accepts_common_sweep_source(self) -> None:
        previous = sys.argv[:]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                common = Path(tmp) / "sweep.json"
                sys.argv = ["script", "--common-sweep-source", str(common)]
                args = PROBE.parse_args()
        finally:
            sys.argv = previous
        self.assertEqual(args.common_sweep_source, common)

    def test_parse_args_accepts_compare_to_path(self) -> None:
        previous = sys.argv[:]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                compare_path = Path(tmp) / "baseline.json"
                sys.argv = ["script", "--compare-to", str(compare_path)]
                args = PROBE.parse_args()
        finally:
            sys.argv = previous
        self.assertEqual(args.compare_to, compare_path)

    def test_replay_attempt_record_marks_accepted_relation(self) -> None:
        accepted = PROBE.replay_attempt_record(
            0,
            {"leaf_index": 7},
            {"below_rho": True, "public_key_verified": True, "derived": True},
            ["row_a", "row_b"],
        )
        self.assertEqual(accepted["leaf_index"], 7)
        self.assertTrue(accepted["accepted_relation_export"])

    def test_replay_attempt_record_rejects_non_accepted_relation(self) -> None:
        rejected = PROBE.replay_attempt_record(
            1,
            {"leaf_index": 11},
            {"below_rho": True, "rank": 1, "relation_count": 1},
            ["row_a", "row_b"],
        )
        self.assertFalse(rejected["accepted_relation_export"])

    def test_build_selection_strategy_comparison_not_requested(self) -> None:
        comparison = PROBE.build_selection_strategy_comparison(SimpleNamespace(compare_to=None), [])
        self.assertFalse(comparison["enabled"])
        self.assertEqual(comparison["comparison_status"], "not_requested")

    def test_build_selection_strategy_comparison_missing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            comparison = PROBE.build_selection_strategy_comparison(SimpleNamespace(compare_to=missing), [])
        self.assertTrue(comparison["enabled"])
        self.assertEqual(comparison["comparison_status"], "compare_to_missing")
        self.assertEqual(comparison["compare_to"], str(missing))

    def test_build_selection_strategy_comparison_detects_improvement_and_regression(self) -> None:
        baseline = {
            "parameters": {"selection_strategy": "parallel_top_k", "top_k": 2},
            "direct_replay_records": [
                {
                    "transfer_index": 1,
                    "accepted_relation_export": False,
                    "below_rho": True,
                    "status": "RANK1_RELATION_ONLY",
                    "selection_attempt_count": 1,
                    "selection_attempted_leaf_indices": [99],
                },
                {
                    "transfer_index": 2,
                    "accepted_relation_export": True,
                    "below_rho": True,
                    "status": "ACCEPTED_DERIVED_OVER_RHO",
                    "selection_attempt_count": 2,
                    "selection_attempted_leaf_indices": [88, 89],
                },
                {
                    "transfer_index": 3,
                    "accepted_relation_export": True,
                    "below_rho": True,
                    "status": "ACCEPTED_DERIVED_BELOW_RHO",
                    "selection_attempt_count": 3,
                    "selection_attempted_leaf_indices": [77],
                },
            ],
        }
        current = [
            {
                "transfer_index": 1,
                "accepted_relation_export": True,
                "below_rho": True,
                "status": "ACCEPTED_DERIVED_BELOW_RHO",
                "selection_attempt_count": 2,
                "selection_attempted_leaf_indices": [99, 100],
            },
            {
                "transfer_index": 2,
                "accepted_relation_export": False,
                "below_rho": False,
                "status": "RANK1_RELATION_ONLY",
                "selection_attempt_count": 1,
                "selection_attempted_leaf_indices": [11],
            },
            {
                "transfer_index": 4,
                "accepted_relation_export": True,
                "below_rho": True,
                "status": "ACCEPTED_DERIVED_BELOW_RHO",
                "selection_attempt_count": 2,
                "selection_attempted_leaf_indices": [66],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline))
            comparison = PROBE.build_selection_strategy_comparison(
                SimpleNamespace(compare_to=baseline_path, selection_strategy="scan_until_relation", top_k=5),
                current,
            )
        self.assertEqual(comparison["comparison_status"], "ok")
        self.assertEqual(comparison["improved_transfers"], [1])
        self.assertEqual(comparison["regressed_transfers"], [2])
        self.assertEqual(comparison["missing_in_baseline_transfers"], [4])
        self.assertEqual(comparison["before_summary"]["selection_strategy"], "parallel_top_k")
        self.assertEqual(comparison["after_summary"]["selection_strategy"], "scan_until_relation")
        self.assertEqual(comparison["outcome_counts"]["unchanged"], 0)

    def test_ranked_leaf_records_prefers_term_shape_matches(self) -> None:
        target = {"transfer_index": 7, "row_keys": ["r0", "r1"]}
        contexts = {
            "r0": {"components": {"leaves": [0, 1, 2]}},
            "r1": {"components": {"leaves": [0, 1, 2]}},
        }
        common_lookup = {7: {0: [], 1: ["2+2"], 2: ["1+1"]}}
        expected = [1, 0, 2]

        with mock.patch.object(
            PROBE,
            "association_leaf_record",
            side_effect=lambda target, leaf_index, contexts: {
                "leaf_index": leaf_index,
                "active_scout_count_sum": 0,
                "hit_root_count_sum": 0,
                "min_term_span": 0,
                "row_hit_total_sum": 0,
                "scout_hit_total_sum": 0,
            },
        ):
            with mock.patch.object(
                PROBE.assoc_predictor,
                "score_key",
                return_value=(0,),
            ):
                ranked = PROBE.ranked_leaf_records(
                    target,
                    contexts,
                    "active_scout_sum_desc",
                    None,
                    ("2+2",),
                    common_lookup,
                )

        self.assertEqual([record["leaf_index"] for record in ranked], expected)

    def test_ranked_leaf_records_without_filters_preserves_leaf_order(self) -> None:
        target = {"transfer_index": 7, "row_keys": ["r0", "r1"]}
        contexts = {
            "r0": {"components": {"leaves": [0, 1, 2]}},
            "r1": {"components": {"leaves": [0, 1, 2]}},
        }
        common_lookup = {7: {0: ["3+2"], 1: ["2+2"], 2: ["1+1"]}}

        with mock.patch.object(
            PROBE,
            "association_leaf_record",
            side_effect=lambda target, leaf_index, contexts: {
                "leaf_index": leaf_index,
                "active_scout_count_sum": 0,
                "hit_root_count_sum": 0,
                "min_term_span": 0,
                "row_hit_total_sum": 0,
                "scout_hit_total_sum": 0,
            },
        ):
            with mock.patch.object(
                PROBE.assoc_predictor,
                "score_key",
                return_value=(0,),
            ):
                ranked = PROBE.ranked_leaf_records(
                    target,
                    contexts,
                    "active_scout_sum_desc",
                    None,
                    (),
                    common_lookup,
                )

        self.assertEqual([record["leaf_index"] for record in ranked], [0, 1, 2])

    def test_summarize_reflects_term_shape_ranking_availability(self) -> None:
        records = [
            {
                "accepted_relation_export": False,
                "below_rho": False,
                "status": "NO_RELATION",
                "selection_attempt_count": 1,
                "selection_strategy": "scan_until_relation",
                "selection_term_shape_filters": ["2+2"],
                "transfer_index": 123,
                "known_positive_transfer": False,
            },
        ]

        summary = PROBE.summarize(
            records,
            failures=[],
            term_shape_filters=("2+2",),
            common_sweep_term_shape_available=False,
        )

        self.assertTrue(summary["direct_replay_without_common_sweep_labels"])
        self.assertFalse(summary["selection_term_shape_ranking_used"])

    def test_summarize_reflects_term_shape_ranking_used_when_available(self) -> None:
        records = [
            {
                "accepted_relation_export": True,
                "below_rho": True,
                "status": "ACCEPTED_DERIVED_BELOW_RHO",
                "selection_attempt_count": 1,
                "selection_strategy": "scan_until_relation",
                "selection_term_shape_filters": ["2+2"],
                "transfer_index": 123,
                "known_positive_transfer": False,
            },
        ]

        summary = PROBE.summarize(
            records,
            failures=[],
            term_shape_filters=("2+2",),
            common_sweep_term_shape_available=True,
        )

        self.assertFalse(summary["direct_replay_without_common_sweep_labels"])
        self.assertTrue(summary["selection_term_shape_ranking_used"])
