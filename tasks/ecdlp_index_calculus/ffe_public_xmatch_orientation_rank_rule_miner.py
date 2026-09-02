#!/usr/bin/env python3
"""Rank public x-match orientation rules by key verification, not relation hits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LINE_STAGE_AUDIT = DEFAULT_STATE_DIR / "ffe_target67_line_stage_audit_328_511.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_xmatch_orientation_rank_rule_miner_target67_328_511.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKTREE_ROOT / path


def case_key(case: dict[str, Any]) -> str:
    return str(case.get("case_key") or "")


def charged_ops_over_rho(result: dict[str, Any]) -> float | None:
    charged = result.get("charged_models") or {}
    value = charged.get("measured_oriented_ops_over_rho")
    return float(value) if value is not None else None


def selected_valid_count(result: dict[str, Any]) -> int:
    return sum(1 for xmatch in result.get("selected_xmatches") or [] if xmatch.get("valid_relation"))


def label_cases(line_stage_path: Path) -> dict[str, dict[str, Any]]:
    line_stage = load_json(line_stage_path)
    labels: dict[str, dict[str, Any]] = {}
    for record in line_stage.get("records") or []:
        if not record.get("has_preserving_line") or not record.get("replay_source"):
            continue
        replay_path = resolve_path(str(record["replay_source"]))
        replay = load_json(replay_path)
        cases = replay.get("cases") or []
        if len(cases) != 1:
            raise ValueError(f"expected one replay case in {replay_path}, found {len(cases)}")
        key = case_key(cases[0])
        labels[key] = {
            "bucket": record.get("bucket"),
            "label_success": record.get("bucket") == "line_present_replay_success",
            "replay_source": str(replay_path.relative_to(WORKTREE_ROOT)),
            "transfer_index": record.get("transfer_index"),
            "top_k": record.get("top_k"),
            "leaf_signature": record.get("leaf_signature"),
            "preserving_line": (record.get("preserving_line") or {}).get("line"),
        }
    return labels


def default_artifacts(labels: dict[str, dict[str, Any]]) -> list[Path]:
    replay_paths = [resolve_path(str(row["replay_source"])) for row in labels.values()]
    where_paths = sorted(
        DEFAULT_STATE_DIR.glob(
            "ffe_public_linear_factor_xmatch_orientation_target67_*where_candidate_eq_scheduled*.json"
        )
    )
    return sorted({*replay_paths, *where_paths})


def load_rule_results(paths: list[Path], labels: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {
        key: {**label, "rules": {}, "artifacts": []}
        for key, label in labels.items()
    }
    for path in paths:
        artifact = load_json(path)
        for case in artifact.get("cases") or []:
            key = case_key(case)
            if key not in cases:
                continue
            cases[key]["artifacts"].append(str(path.relative_to(WORKTREE_ROOT)))
            for result in case.get("rule_results") or []:
                rule = str(result.get("rule") or "")
                if not rule:
                    continue
                cases[key]["rules"][rule] = {
                    "public_key_verified": bool(result.get("public_key_verified")),
                    "rank": int(result.get("rank") or 0),
                    "relation_count": int(result.get("relation_count") or 0),
                    "selected_xmatch_count": int(result.get("selected_xmatch_count") or 0),
                    "selected_valid_relation_count": selected_valid_count(result),
                    "ops_over_rho": charged_ops_over_rho(result),
                    "artifact": str(path.relative_to(WORKTREE_ROOT)),
                }
    return cases


def evaluate_rule(rule: str, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for key, case in sorted(cases.items()):
        result = case["rules"].get(rule)
        verified = bool(result and result.get("public_key_verified"))
        label_success = bool(case["label_success"])
        rows.append(
            {
                "case_key": key,
                "label_success": label_success,
                "bucket": case["bucket"],
                "transfer_index": case["transfer_index"],
                "top_k": case["top_k"],
                "leaf_signature": case["leaf_signature"],
                "preserving_line": case["preserving_line"],
                "rule_present": result is not None,
                "public_key_verified": verified,
                "rank": result.get("rank") if result else 0,
                "relation_count": result.get("relation_count") if result else 0,
                "selected_xmatch_count": result.get("selected_xmatch_count") if result else 0,
                "selected_valid_relation_count": (
                    result.get("selected_valid_relation_count") if result else 0
                ),
                "ops_over_rho": result.get("ops_over_rho") if result else None,
                "artifact": result.get("artifact") if result else None,
            }
        )
    tp = sum(1 for row in rows if row["label_success"] and row["public_key_verified"])
    fp = sum(1 for row in rows if not row["label_success"] and row["public_key_verified"])
    fn = sum(1 for row in rows if row["label_success"] and not row["public_key_verified"])
    tn = sum(1 for row in rows if not row["label_success"] and not row["public_key_verified"])
    verified_ops = [
        float(row["ops_over_rho"])
        for row in rows
        if row["public_key_verified"] and row["ops_over_rho"] is not None
    ]
    verified_selected = [
        int(row["selected_xmatch_count"])
        for row in rows
        if row["public_key_verified"]
    ]
    return {
        "rule": rule,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "true_negative_count": tn,
        "selected_case_count": tp + fp,
        "verified_ops_over_rho_max": max(verified_ops) if verified_ops else None,
        "verified_ops_over_rho_values": sorted(verified_ops),
        "verified_selected_xmatch_count_max": max(verified_selected) if verified_selected else None,
        "case_results": rows,
    }


def score(row: dict[str, Any]) -> tuple[Any, ...]:
    max_ops = row["verified_ops_over_rho_max"]
    max_selected = row["verified_selected_xmatch_count_max"]
    return (
        int(row["false_positive_count"]),
        int(row["false_negative_count"]),
        -int(row["true_positive_count"]),
        float(max_ops) if max_ops is not None else float("inf"),
        int(max_selected) if max_selected is not None else 10**9,
        len(str(row["rule"])),
        str(row["rule"]),
    )


def rule_names(cases: dict[str, dict[str, Any]]) -> list[str]:
    names = set()
    for case in cases.values():
        names.update(case["rules"].keys())
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-stage-audit", type=Path, default=DEFAULT_LINE_STAGE_AUDIT)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--top-rules", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    labels = label_cases(args.line_stage_audit)
    artifacts = args.artifact or default_artifacts(labels)
    cases = load_rule_results(artifacts, labels)
    evaluations = [evaluate_rule(rule, cases) for rule in rule_names(cases)]
    evaluations.sort(key=score)
    best = evaluations[0] if evaluations else None
    by_rule_family = defaultdict(int)
    for evaluation in evaluations:
        by_rule_family[str(evaluation["rule"]).split(":", 1)[0]] += 1
    output = {
        "schema": "ecdlp_public_xmatch_orientation_rank_rule_miner_v1",
        "method": "rank_public_orientation_rules_by_public_key_verification_across_line_present_cases",
        "parameters": {
            "line_stage_audit": str(args.line_stage_audit),
            "artifacts": [str(path.relative_to(WORKTREE_ROOT)) for path in artifacts],
            "label_use": (
                "Replay success labels are used to rank already-replayed public rules. "
                "The selected rule still needs fresh holdout validation."
            ),
        },
        "summary": {
            "case_count": len(cases),
            "success_case_count": sum(1 for case in cases.values() if case["label_success"]),
            "failure_case_count": sum(1 for case in cases.values() if not case["label_success"]),
            "candidate_rule_count": len(evaluations),
            "rule_family_counts": dict(sorted(by_rule_family.items())),
            "best_rule": best["rule"] if best else None,
            "best_rule_true_positive_count": best["true_positive_count"] if best else 0,
            "best_rule_false_positive_count": best["false_positive_count"] if best else 0,
            "best_rule_false_negative_count": best["false_negative_count"] if best else 0,
            "interpretation": (
                "This miner penalizes rank-1 public slices that contain a valid relation "
                "but do not derive the key. It is retrospective over the supplied replay artifacts."
            ),
        },
        "best_rule_evaluation": best,
        "top_rules": evaluations[: max(1, int(args.top_rules))],
        "non_claims": [
            "This is not a fresh holdout validation.",
            "A no-op all-xmatches rule still depends on public line prediction or amortized line confirmation.",
            "Rules absent from a case's replay artifact are treated as non-verifying for that case.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
