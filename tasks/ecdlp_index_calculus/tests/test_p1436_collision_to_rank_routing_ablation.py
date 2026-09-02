from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "p1436_collision_to_rank_routing_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("p1436_collision", MODULE_PATH)
assert SPEC and SPEC.loader
ABLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ABLATION)


def fixture_config(
    *,
    collisions: int = 8,
    cross_shift: int = 3,
    rows: int = 4,
    rank: int = 2,
    unknowns: int = 5,
    raw_records: list[dict] | None = None,
) -> dict:
    config = {
        "collision_edge_count": collisions,
        "cross_shift_collision_count": cross_shift,
        "within_shift_collision_count": collisions - cross_shift,
        "relation_row_count": rows,
        "duplicate_or_zero_row_count": collisions - rows,
        "relation_rank": rank,
        "augmented_rank": rank,
        "unknown_factor_count": unknowns,
    }
    if raw_records is not None:
        config["collision_records"] = raw_records
    return config


def valid_collision_records() -> list[dict]:
    relation_rows = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [2, 3, 0, 0, 0],
    ]
    records = []
    for index in range(8):
        admitted = index < len(relation_rows)
        records.append(
            {
                "edge_id": f"edge-{index}",
                "left_shift": index,
                "right_shift": index + 1 if index < 3 else index,
                "relation_coefficients": (
                    relation_rows[index] if admitted else [0, 0, 0, 0, 0]
                ),
                "relation_rhs": 0,
                "relation_admitted": admitted,
                "source_equation_exact": True,
                "residual_equality_exact": True,
            }
        )
    return records


def fixture_payload() -> dict:
    return {
        "schema": "ecdlp.p1436_large_prime_residual_collision_collector.v1",
        "curve_records": [
            {
                "split": "prospective",
                "bits": 24,
                "seed": 1432401,
                "order": 1009,
                "policies": {
                    "two_map_union": {
                        "full": {
                            "factor_base_size_B": 12,
                            "configurations": {
                                "random_hash_mask1": fixture_config(
                                    raw_records=valid_collision_records()
                                ),
                                "mixed_balanced_stride_mask1": fixture_config(),
                            },
                        }
                    }
                },
            }
        ],
    }


class AblationHarnessTests(unittest.TestCase):
    def test_artifacts_surface_missing_and_present_collision_sources(self) -> None:
        payload = fixture_payload()
        routing, matrices = ABLATION.build_artifacts(payload, ABLATION.harness.DEFAULT_NOTE_URL)

        routing_status = routing["execution_status"]
        self.assertEqual(routing_status["status"], "missing_exact_inputs")
        self.assertEqual(routing_status["records_with_missing_collision_sources"], 1)
        self.assertEqual(routing_status["records_missing_exact_inputs"], 1)
        self.assertEqual(routing_status["replay_ready_records"], 1)
        self.assertEqual(routing_status["routing_ablation_records"], 2)

        matrices_status = matrices["execution_status"]
        self.assertEqual(matrices_status["status"], "missing_exact_inputs")
        self.assertEqual(matrices_status["entries_missing_exact_inputs"], 3)
        self.assertEqual(matrices_status["matrix_entries"], 6)
        self.assertEqual(matrices_status["entries_replay_ready"], 3)

        first_record = routing["records"][0]
        self.assertEqual(
            first_record["matrix_variants"][0]["status"],
            "exact_replay_compiled",
        )
        self.assertEqual(first_record["matrix_variants"][0]["replay_mode"], "exact")
        self.assertEqual(first_record["matrix_variants"][0]["relation_row_count"], 4)
        self.assertEqual(first_record["matrix_variants"][0]["relation_rank_observed"], 2)
        self.assertEqual(first_record["matrix_variants"][0]["augmented_rank_observed"], 2)
        self.assertEqual(len(first_record["matrix_variants"][0]["augmented_rows"]), 4)
        second_record = routing["records"][1]
        self.assertEqual(
            second_record["matrix_variants"][0]["status"],
            "synthetic_replay_planned",
        )
        self.assertEqual(second_record["matrix_variants"][0]["replay_mode"], "synthetic")
        self.assertEqual(second_record["replay_mode"], "synthetic")

        first_matrix = matrices["matrices"][0]
        self.assertEqual(first_matrix["matrix"]["status"], "exact_replay_compiled")
        self.assertEqual(first_matrix["matrix"]["replay_mode"], "exact")
        self.assertEqual(first_matrix["matrix"]["raw_source"]["present"], True)
        self.assertTrue(first_matrix["matrix"]["compiled_from_exact_records"])
        self.assertEqual(first_matrix["matrix"]["relation_rank_observed"], 2)
        self.assertEqual(len(first_matrix["matrix"]["coefficient_rows"]), 4)
        self.assertIn("all_edge", first_matrix["matrix"]["matrix_id"])

        second_matrix = next(
            matrix
            for matrix in matrices["matrices"]
            if matrix["configuration"] == "mixed_balanced_stride_mask1"
            and matrix["matrix"]["matrix_id"].endswith("within_shift_only")
        )
        self.assertEqual(second_matrix["matrix"]["status"], "synthetic_replay_planned")
        self.assertEqual(second_matrix["matrix"]["replay_mode"], "synthetic")
        self.assertEqual(second_matrix["matrix"]["raw_source"]["present"], False)

    def test_known_tweet_url_is_normalized_in_script_metadata(self) -> None:
        aliases = [
            "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
            "https://x.com/i/web/status/2076737985559822734?s=46",
            "http://x.com/i/status/2076737985559822734?s=46",
            "x.com/askalphaxiv/status/2076737985559822734?s=46",
        ]
        for alias in aliases:
            with self.subTest(alias=alias):
                source_metadata = ABLATION.build_source_metadata(alias)

                self.assertEqual(
                    source_metadata["source_post_url"],
                    "https://x.com/askalphaxiv/status/2076737985559822734",
                )
                self.assertEqual(
                    source_metadata["source_post_url_with_query"],
                    "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
                )
                self.assertEqual(
                    source_metadata["source_query"],
                    "?s=46",
                )
                self.assertEqual(
                    source_metadata["tweet_posted_at"],
                    "2026-07-13T18:38:14.340000+00:00",
                )
                self.assertEqual(
                    source_metadata["tweet_source_title"],
                    "Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning",
                )
                self.assertEqual(
                    source_metadata["tweet_referenced_paper_title"],
                    "Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in LLM Finetuning",
                )
                self.assertEqual(
                    source_metadata["tweet_source_url"],
                    "https://arxiv.org/abs/2607.08393",
                )
                self.assertTrue(source_metadata["tweet_has_media"])
                self.assertEqual(source_metadata["tweet_media_count"], 1)
                self.assertIn(
                    "https://x.com/askalphaxiv/status/2076737985559822734/photo/1",
                    source_metadata["tweet_media_urls"],
                )
                self.assertIn("video", source_metadata["tweet_media_types"])
                self.assertEqual(
                    source_metadata["tweet_intake_status"],
                    "Exact tweet text snapshot captured for this known source.",
                )
                self.assertEqual(
                    source_metadata["tweet_text_source_note"],
                    "Exact tweet text is embedded for this known source as an auditable source snapshot.",
                )
                self.assertTrue(source_metadata["tweet_text_included"])
                self.assertIn(
                    "GPT-5.6 stayed more focused",
                    source_metadata["tweet_summary"],
                )

    def test_unknown_tweet_url_marks_text_as_not_included(self) -> None:
        source_metadata = ABLATION.build_source_metadata(
            "https://x.com/status/0000000000000000001?s=46"
        )
        self.assertFalse(source_metadata["tweet_text_included"])
        self.assertFalse(source_metadata["tweet_summary_is_verbatim"])
        self.assertEqual(
            source_metadata["tweet_summary"],
            (
                "The linked post summarized focused autoresearch behavior and "
                "critical-experiment selection; exact text was not embedded in "
                "this runtime context."
            ),
        )
        self.assertEqual(
            source_metadata["tweet_intake_status"],
            "Source URL supplied by user; exact tweet text is not embedded in this harness.",
        )
        self.assertEqual(
            source_metadata["tweet_text_source_note"],
            (
                "Exact tweet text was not retrievable from this harness context; "
                "guidance was embedded as a non-verbatim, source-linked summary."
            ),
        )
        self.assertEqual(source_metadata["tweet_text"], "")
        self.assertEqual(source_metadata["tweet_text_sha256"], "")
        self.assertFalse(source_metadata["tweet_has_media"])
        self.assertEqual(source_metadata["tweet_media_count"], 0)
        self.assertEqual(source_metadata["tweet_media_urls"], [])
        self.assertEqual(source_metadata["tweet_media_types"], [])

    def test_nonempty_but_invalid_collision_list_is_not_exact(self) -> None:
        config = fixture_config(raw_records=[{"a": 1}])
        validation = ABLATION.validate_collision_source(config, modulus=1009)

        self.assertTrue(validation["source_present"])
        self.assertFalse(validation["exact"])
        self.assertIn("record_0_edge_id_invalid", validation["errors"])
        self.assertEqual(
            ABLATION._determine_replay_mode(config, validation),
            ABLATION.REPLAY_MODE_INVALID,
        )
        variant = ABLATION.matrix_variant_payload(
            config,
            "all_edge",
            validation,
            modulus=1009,
        )
        self.assertEqual(variant["status"], "replay_source_inconsistent")
        self.assertTrue(variant["raw_collision_records_present"])
        self.assertFalse(variant["raw_collision_records_exact"])
        self.assertEqual(variant["coefficient_rows"], [])

    def test_summary_mismatch_rejects_otherwise_well_formed_records(self) -> None:
        config = fixture_config(
            collisions=9,
            raw_records=valid_collision_records(),
        )
        validation = ABLATION.validate_collision_source(config, modulus=1009)

        self.assertFalse(validation["exact"])
        self.assertIn("collision_edge_count_mismatch", validation["errors"])

    def test_relation_admission_must_match_nonzero_unknown_row(self) -> None:
        records = valid_collision_records()
        records[0]["relation_admitted"] = False
        config = fixture_config(raw_records=records)

        validation = ABLATION.validate_collision_source(config, modulus=1009)

        self.assertFalse(validation["exact"])
        self.assertIn("record_0_relation_admission_mismatch", validation["errors"])

    def test_script_writes_artifacts_via_cli(self) -> None:
        payload = fixture_payload()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "collector.json"
            source.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            routing_output = root / "routing_ablation.json"
            matrices_output = root / "relation_matrices.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    str(source),
                    "--note-url",
                    "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
                    "--routing-output",
                    str(routing_output),
                    "--matrices-output",
                    str(matrices_output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(routing_output.exists())
            self.assertTrue(matrices_output.exists())
            routing = json.loads(routing_output.read_text(encoding="utf-8"))
            matrices = json.loads(matrices_output.read_text(encoding="utf-8"))

        self.assertIn("routing_status=missing_exact_inputs", completed.stdout)
        self.assertEqual(routing["schema"], ABLATION.ROUTING_ABLATION_SCHEMA)
        self.assertEqual(matrices["schema"], ABLATION.RELATION_MATRICES_SCHEMA)
        self.assertEqual(routing["note_url"], "https://x.com/askalphaxiv/status/2076737985559822734?s=46")
        self.assertEqual(matrices["note_url"], "https://x.com/askalphaxiv/status/2076737985559822734?s=46")


if __name__ == "__main__":
    unittest.main()
