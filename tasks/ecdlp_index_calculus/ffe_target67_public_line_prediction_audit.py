#!/usr/bin/env python3
"""Audit whether public target-67 features predict preserving FFE lines.

The target-67 replay package has a useful line-present activation rule and a
rank-aware all-xmatch replay baseline.  This script checks the remaining gap:
can the preserving degree-1 line itself be predicted from public row/leaf
features before exact FFE factorization?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LINE_STAGE_AUDIT = DEFAULT_STATE_DIR / "ffe_target67_line_stage_audit_328_511.json"
DEFAULT_FEATURE_PROBE = DEFAULT_STATE_DIR / "ffe_target67_orientation_feature_probe_328_511.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_public_line_prediction_audit_328_511.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_leaf_signature(signature: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(signature).split(",") if part != "")


def feature_probe_by_replay_source(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    probe = load_json(path)
    return {
        str(record.get("replay_source")): dict(record)
        for record in probe.get("records") or []
        if record.get("replay_source")
    }


def line_records(line_stage_path: Path, feature_probe_path: Path) -> list[dict[str, Any]]:
    line_stage = load_json(line_stage_path)
    feature_by_replay = feature_probe_by_replay_source(feature_probe_path)
    records = []
    for record in line_stage.get("records") or []:
        if not record.get("has_preserving_line") or not record.get("replay_source"):
            continue
        replay_source = str(record["replay_source"])
        features = feature_by_replay.get(replay_source, {})
        line = (record.get("preserving_line") or {}).get("line")
        leaf_signature = str(record.get("leaf_signature") or "")
        records.append(
            {
                "record_id": (
                    f"transfer{record.get('transfer_index')}_top{record.get('top_k')}_"
                    f"{leaf_signature}"
                ),
                "bucket": record.get("bucket"),
                "label_replay_success": record.get("bucket") == "line_present_replay_success",
                "transfer_index": int(record.get("transfer_index") or 0),
                "top_k": int(record.get("top_k") or 0),
                "salt": int(record.get("salt") or 0),
                "leaf_signature": leaf_signature,
                "leaf_tuple": parse_leaf_signature(leaf_signature),
                "leaf_count": len(parse_leaf_signature(leaf_signature)),
                "leaf_selector": str(record.get("leaf_selector") or ""),
                "preserving_line": str(line),
                "preserving_factor_index": record.get("preserving_factor_index"),
                "line_root_scan_ops_over_rho": record.get("line_root_scan_ops_over_rho"),
                "replay_source": replay_source,
                "xmatch_count": int(features.get("xmatch_count") or 0),
                "scheduled1_count": int(features.get("scheduled1_count") or 0),
                "candidate_eq_scheduled_xmatch_count": int(
                    features.get("candidate_eq_scheduled_xmatch_count") or 0
                ),
                "valid_xmatch_count": int(features.get("valid_xmatch_count") or 0),
            }
        )
    return sorted(records, key=lambda row: (row["transfer_index"], row["leaf_signature"]))


def ge_atoms(prefix: str, value: int, thresholds: tuple[int, ...]) -> list[str]:
    return [f"{prefix}>={threshold}" for threshold in thresholds if value >= threshold]


def feature_atoms(record: dict[str, Any]) -> set[str]:
    leaves = tuple(record["leaf_tuple"])
    atoms = {
        f"top_k={record['top_k']}",
        f"leaf_signature={record['leaf_signature']}",
        f"top_k={record['top_k']}&leaf_signature={record['leaf_signature']}",
        f"leaf_count={record['leaf_count']}",
        f"top_k={record['top_k']}&leaf_count={record['leaf_count']}",
        f"leaf_selector={record['leaf_selector']}",
        f"top_k={record['top_k']}&leaf_selector={record['leaf_selector']}",
        f"xmatch_count={record['xmatch_count']}",
        f"top_k={record['top_k']}&xmatch_count={record['xmatch_count']}",
        f"scheduled1_count={record['scheduled1_count']}",
        f"top_k={record['top_k']}&scheduled1_count={record['scheduled1_count']}",
        f"candidate_eq_scheduled_xmatch_count={record['candidate_eq_scheduled_xmatch_count']}",
        (
            f"top_k={record['top_k']}&candidate_eq_scheduled_xmatch_count="
            f"{record['candidate_eq_scheduled_xmatch_count']}"
        ),
    }
    for atom in ge_atoms("xmatch_count", record["xmatch_count"], (3, 4, 6, 8, 9)):
        atoms.add(atom)
        atoms.add(f"top_k={record['top_k']}&{atom}")
    for atom in ge_atoms("scheduled1_count", record["scheduled1_count"], (3, 4)):
        atoms.add(atom)
        atoms.add(f"top_k={record['top_k']}&{atom}")
    for size in (2, 3):
        for subset in combinations(leaves, size):
            joined = ",".join(str(part) for part in subset)
            atoms.add(f"leaf_subset{size}={joined}")
            atoms.add(f"top_k={record['top_k']}&leaf_subset{size}={joined}")
    return atoms


def build_index(train_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        for atom in feature_atoms(row):
            grouped[atom].append(row)
    index = {}
    for atom, rows in grouped.items():
        line_counts = Counter(row["preserving_line"] for row in rows)
        index[atom] = {
            "atom": atom,
            "train_support_count": len(rows),
            "line_counts": dict(sorted(line_counts.items())),
            "line_count": len(line_counts),
            "success_count": sum(1 for row in rows if row["label_replay_success"]),
            "failure_count": sum(1 for row in rows if not row["label_replay_success"]),
            "examples": [
                {
                    "record_id": row["record_id"],
                    "transfer_index": row["transfer_index"],
                    "label_replay_success": row["label_replay_success"],
                    "preserving_line": row["preserving_line"],
                }
                for row in rows
            ],
        }
    return index


def case_prediction(
    record: dict[str, Any],
    index: dict[str, dict[str, Any]],
    min_train_support: int,
    max_predictions: int,
) -> dict[str, Any]:
    unique_predictions = []
    ambiguous_predictions = []
    true_line = record["preserving_line"]
    for atom in sorted(feature_atoms(record)):
        entry = index.get(atom)
        if not entry or int(entry["train_support_count"]) < min_train_support:
            continue
        line_counts = entry["line_counts"]
        prediction = {
            "atom": atom,
            "train_support_count": entry["train_support_count"],
            "line_counts": line_counts,
            "contains_true_line": true_line in line_counts,
        }
        if len(line_counts) == 1:
            predicted_line = next(iter(line_counts))
            unique_predictions.append(
                {
                    **prediction,
                    "predicted_line": predicted_line,
                    "line_match": predicted_line == true_line,
                }
            )
        else:
            ambiguous_predictions.append(prediction)
    unique_predictions.sort(
        key=lambda item: (
            not item["line_match"],
            -int(item["train_support_count"]),
            len(item["atom"]),
            item["atom"],
        )
    )
    ambiguous_predictions.sort(
        key=lambda item: (
            not item["contains_true_line"],
            len(item["line_counts"]),
            -int(item["train_support_count"]),
            len(item["atom"]),
            item["atom"],
        )
    )
    return {
        "record_id": record["record_id"],
        "transfer_index": record["transfer_index"],
        "top_k": record["top_k"],
        "leaf_signature": record["leaf_signature"],
        "label_replay_success": record["label_replay_success"],
        "true_line": true_line,
        "unique_prediction_count": len(unique_predictions),
        "unique_line_match": any(item["line_match"] for item in unique_predictions),
        "ambiguous_prediction_count": len(ambiguous_predictions),
        "ambiguous_contains_true_line": any(
            item["contains_true_line"] for item in ambiguous_predictions
        ),
        "best_unique_predictions": unique_predictions[:max_predictions],
        "best_ambiguous_predictions": ambiguous_predictions[:max_predictions],
    }


def evaluate_cutoff(
    records: list[dict[str, Any]],
    cutoff: int,
    min_train_support_values: list[int],
    max_predictions: int,
) -> dict[str, Any]:
    train_rows = [row for row in records if row["transfer_index"] <= cutoff]
    validation_rows = [row for row in records if row["transfer_index"] > cutoff]
    index = build_index(train_rows)
    support_results = []
    for min_support in min_train_support_values:
        predictions = [
            case_prediction(row, index, min_support, max_predictions)
            for row in validation_rows
        ]
        unique_attempts = [row for row in predictions if row["unique_prediction_count"]]
        unique_hits = [row for row in predictions if row["unique_line_match"]]
        success_rows = [row for row in predictions if row["label_replay_success"]]
        support_results.append(
            {
                "min_train_support": min_support,
                "validation_case_count": len(predictions),
                "validation_success_count": len(success_rows),
                "unique_attempt_case_count": len(unique_attempts),
                "unique_line_match_case_count": len(unique_hits),
                "unique_success_line_match_case_count": sum(
                    1 for row in unique_hits if row["label_replay_success"]
                ),
                "unique_success_attempt_case_count": sum(
                    1 for row in unique_attempts if row["label_replay_success"]
                ),
                "ambiguous_contains_true_case_count": sum(
                    1 for row in predictions if row["ambiguous_contains_true_line"]
                ),
                "ambiguous_success_contains_true_case_count": sum(
                    1
                    for row in predictions
                    if row["label_replay_success"] and row["ambiguous_contains_true_line"]
                ),
                "case_predictions": predictions,
            }
        )
    repeated_atoms = [
        entry
        for entry in index.values()
        if int(entry["train_support_count"]) >= 2
    ]
    repeated_atoms.sort(
        key=lambda entry: (
            len(entry["line_counts"]),
            -int(entry["train_support_count"]),
            -int(entry["success_count"]),
            entry["atom"],
        )
    )
    return {
        "train_cutoff": cutoff,
        "train_case_count": len(train_rows),
        "validation_case_count": len(validation_rows),
        "train_success_count": sum(1 for row in train_rows if row["label_replay_success"]),
        "validation_success_count": sum(
            1 for row in validation_rows if row["label_replay_success"]
        ),
        "feature_atom_count": len(index),
        "repeated_feature_atom_count": len(repeated_atoms),
        "line_count": len({row["preserving_line"] for row in train_rows}),
        "support_results": support_results,
        "top_repeated_feature_atoms": repeated_atoms[:25],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-stage-audit", type=Path, default=DEFAULT_LINE_STAGE_AUDIT)
    parser.add_argument("--feature-probe", type=Path, default=DEFAULT_FEATURE_PROBE)
    parser.add_argument("--train-cutoff", type=int, action="append", default=[])
    parser.add_argument("--min-train-support", type=int, action="append", default=[])
    parser.add_argument("--max-predictions", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records = line_records(args.line_stage_audit, args.feature_probe)
    cutoffs = args.train_cutoff or [431, 463, 495]
    min_support_values = args.min_train_support or [1, 2]
    evaluations = [
        evaluate_cutoff(records, cutoff, min_support_values, max(1, args.max_predictions))
        for cutoff in cutoffs
    ]
    first_support = [
        result
        for evaluation in evaluations
        for result in evaluation["support_results"]
        if result["min_train_support"] == min(min_support_values)
    ]
    stable_support = [
        result
        for evaluation in evaluations
        for result in evaluation["support_results"]
        if result["min_train_support"] == max(min_support_values)
    ]
    output = {
        "schema": "ecdlp_target67_public_line_prediction_audit_v1",
        "method": "train_public_feature_to_preserving_line_dictionary_then_score_later_transfers",
        "parameters": {
            "line_stage_audit": str(args.line_stage_audit),
            "feature_probe": str(args.feature_probe),
            "train_cutoffs": cutoffs,
            "min_train_support_values": min_support_values,
            "max_predictions": int(args.max_predictions),
        },
        "summary": {
            "line_present_case_count": len(records),
            "replay_success_count": sum(1 for row in records if row["label_replay_success"]),
            "distinct_preserving_line_count": len({row["preserving_line"] for row in records}),
            "unique_line_match_case_count_min_support_1": sum(
                result["unique_line_match_case_count"] for result in first_support
            ),
            "unique_success_line_match_case_count_min_support_1": sum(
                result["unique_success_line_match_case_count"] for result in first_support
            ),
            "unique_line_match_case_count_stable_support": sum(
                result["unique_line_match_case_count"] for result in stable_support
            ),
            "unique_success_line_match_case_count_stable_support": sum(
                result["unique_success_line_match_case_count"] for result in stable_support
            ),
            "interpretation": (
                "Simple public feature dictionaries do not yet predict fresh preserving "
                "target-67 lines. Repeated subset features are useful diagnostics, but "
                "they are ambiguous or replay-failure-only in these windows."
            ),
        },
        "records": [
            {
                key: value
                for key, value in record.items()
                if key != "leaf_tuple"
            }
            for record in records
        ],
        "evaluations": evaluations,
        "non_claims": [
            "This audit does not run exact factorization on new rows.",
            "A set-valued feature containing the true line is not a public line predictor unless it chooses one line before exact factorization.",
            "Replay-failure rows are mechanism evidence only and are not counted as ECDLP recoveries.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
