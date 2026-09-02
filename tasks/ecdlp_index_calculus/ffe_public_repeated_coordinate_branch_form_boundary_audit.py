#!/usr/bin/env python3
"""Audit branch-aware candidate-position form boundaries.

The frozen branch-aware two-row replay uses a strict same-position guard.  The
680 backtest exposed verified relation systems with duplicate positions that
are not same-position aligned.  This diagnostic keeps that boundary explicit
without promoting a relaxed guard.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_branch_form_boundary_audit.json"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text())


def parse_named_path(raw: str) -> tuple[str, Path]:
    parts = raw.split("|", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("value must be name|path")
    return parts[0], Path(parts[1])


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def duplicate_count(values: list[int]) -> int:
    counts = Counter(values)
    return max(counts.values()) if counts else 0


def form_class(features: dict[str, Any]) -> str:
    values = [int(value) for value in features.get("candidate_pos_values") or []]
    span = int(features.get("candidate_pos_span") or 0)
    count = int(features.get("candidate_pos_count") or 0)
    unique_count = int(features.get("candidate_pos_unique_count") or 0)
    if span == 0 and count >= 2 and unique_count == 1:
        return "strict_same_position_duplicate"
    if unique_count == 2 and count >= 3 and duplicate_count(values) >= 2:
        return "partial_duplicate_two_position"
    if count >= 2:
        return "multi_event_misaligned"
    if count == 1:
        return "singleton"
    return "no_candidate_position"


def compact_case(case: dict[str, Any], source: str) -> dict[str, Any]:
    replay = case.get("guarded_replay") or {}
    features = case.get("form_guard_features") or {}
    relation_count = int(replay.get("relation_count") or 0)
    ops_over_rho = round_or_none(replay.get("ops_over_rho"))
    return {
        "source": source,
        "case_key": case.get("case_key"),
        "window": case.get("window"),
        "target": case.get("target"),
        "transfer_index": case.get("transfer_index"),
        "coordinate": case.get("coordinate"),
        "row_keys": replay.get("row_keys") or [],
        "rank": int(replay.get("rank") or 0),
        "relation_count": relation_count,
        "public_key_verified": bool(replay.get("public_key_verified")),
        "below_rho": bool(replay.get("below_rho")),
        "guard_passed": bool(replay.get("guard_passed") or case.get("form_guard_passed")),
        "ops_over_rho": ops_over_rho,
        "candidate_pos_signature": features.get("candidate_pos_signature") or "",
        "candidate_pos_span": features.get("candidate_pos_span"),
        "candidate_pos_count": features.get("candidate_pos_count"),
        "candidate_pos_unique_count": features.get("candidate_pos_unique_count"),
        "candidate_pos_values": features.get("candidate_pos_values") or [],
        "form_class": form_class(features),
    }


def load_relation_cases(named_paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    relation_cases: list[dict[str, Any]] = []
    for name, path in named_paths:
        data = load_json(path)
        for case in data.get("cases") or []:
            replay = case.get("guarded_replay") or {}
            if int(replay.get("relation_count") or 0) >= 2:
                relation_cases.append(compact_case(case, name))
    return relation_cases


def summarize_group(cases: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [case for case in cases if case["public_key_verified"]]
    below = [case for case in verified if case["below_rho"]]
    guarded = [case for case in cases if case["guard_passed"]]
    ops = [float(case["ops_over_rho"]) for case in cases if case["ops_over_rho"] is not None]
    verified_ops = [
        float(case["ops_over_rho"])
        for case in verified
        if case["ops_over_rho"] is not None
    ]
    return {
        "relation_case_count": len(cases),
        "guard_passed_case_count": len(guarded),
        "verified_case_count": len(verified),
        "verified_below_rho_count": len(below),
        "min_ops_over_rho": round(min(ops), 8) if ops else None,
        "mean_ops_over_rho": mean_or_none(ops),
        "min_verified_ops_over_rho": round(min(verified_ops), 8) if verified_ops else None,
        "mean_verified_ops_over_rho": mean_or_none(verified_ops),
    }


def group_summary(cases: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault(str(case.get(key) or ""), []).append(case)
    return {name: summarize_group(group) for name, group in sorted(groups.items())}


def representative_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        cases,
        key=lambda case: (
            not case["public_key_verified"],
            not case["below_rho"],
            float(case["ops_over_rho"]) if case["ops_over_rho"] is not None else 999.0,
            str(case.get("source")),
            str(case.get("case_key")),
        ),
    )
    return ordered[:limit]


def run(args: argparse.Namespace) -> dict[str, Any]:
    named_paths = [parse_named_path(raw) for raw in args.branch_replay]
    cases = load_relation_cases(named_paths)
    signature_summary = group_summary(cases, "candidate_pos_signature")
    class_summary = group_summary(cases, "form_class")
    partial = [case for case in cases if case["form_class"] == "partial_duplicate_two_position"]
    strict = [case for case in cases if case["form_class"] == "strict_same_position_duplicate"]
    return {
        "schema": "ecdlp_public_repeated_coordinate_branch_form_boundary_audit_v1",
        "method": "branch_aware_relation_case_candidate_position_boundary_audit",
        "parameters": {
            "branch_replays": [f"{name}|{path}" for name, path in named_paths],
            "representative_case_limit": int(args.case_limit),
        },
        "summary": {
            **summarize_group(cases),
            "source_summary": group_summary(cases, "source"),
            "candidate_pos_signature_summary": signature_summary,
            "form_class_summary": class_summary,
            "strict_same_position_verified_below_rho_count": summarize_group(strict)[
                "verified_below_rho_count"
            ],
            "partial_duplicate_verified_below_rho_count": summarize_group(partial)[
                "verified_below_rho_count"
            ],
        },
        "representative_cases": representative_cases(cases, int(args.case_limit)),
        "non_claims": [
            "This is a replay-form audit, not a public pre-event selector.",
            "Partial-duplicate cases are diagnostic until a frozen public guard rejects known decoys.",
            "Relaxing the strict same-position guard would need a new validation run.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-replay", action="append", required=True, help="name|path")
    parser.add_argument("--case-limit", type=int, default=24)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    out_path = resolve_path(Path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
