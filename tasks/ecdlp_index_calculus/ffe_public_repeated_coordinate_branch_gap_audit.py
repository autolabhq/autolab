#!/usr/bin/env python3
"""Audit gaps between repeated-coordinate pipeline stages.

Fresh windows can contain full-coordinate relation material while the frozen
strict pair rule selects a different public slice.  This diagnostic keeps those
branches separate: coordinate-gate verification, row/form relation reach, and
strict pair-rule reach.
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
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_branch_gap_audit.json"


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


def coordinate_key(source: str, transfer_index: Any, coordinate: Any) -> str:
    if not isinstance(coordinate, dict):
        return f"{source}:{transfer_index}:None"
    return f"{source}:{transfer_index}:{coordinate.get('b')},{coordinate.get('c')}"


def case_stub(case: dict[str, Any], source: str, replay_key: str) -> dict[str, Any]:
    replay = case.get(replay_key) or {}
    coordinate = case.get("coordinate")
    return {
        "key": coordinate_key(source, case.get("transfer_index"), coordinate),
        "source": source,
        "transfer_index": case.get("transfer_index"),
        "top_k": case.get("top_k"),
        "policy": case.get("policy"),
        "leaf_selector": case.get("leaf_selector"),
        "coordinate": coordinate,
        "ops_over_rho": round_or_none(replay.get("ops_over_rho")),
        "rank": replay.get("rank"),
        "relation_count": replay.get("relation_count"),
        "public_key_verified": bool(replay.get("public_key_verified")),
        "below_rho": bool(replay.get("below_rho")),
    }


def load_coordinate_cases(named_paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in named_paths:
        data = load_json(path)
        for case in data.get("candidates") or []:
            item = case_stub(case, source, "exact_coordinate_replay")
            item["salts"] = case.get("salts")
            item["leaf_indices"] = case.get("leaf_indices")
            rows.append(item)
    return rows


def load_guard_cases(named_paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in named_paths:
        data = load_json(path)
        for case in data.get("cases") or []:
            item = case_stub(case, source, "guarded_replay")
            item["form_guard_passed"] = bool(case.get("form_guard_passed"))
            item["form_guard_features"] = case.get("form_guard_features") or {}
            rows.append(item)
    return rows


def load_pair_cases(named_paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in named_paths:
        data = load_json(path)
        for case in data.get("cases") or []:
            item = case_stub(case, source, "guarded_replay")
            item["form_guard_passed"] = bool(case.get("form_guard_passed"))
            item["pair_features"] = case.get("pair_features") or {}
            rows.append(item)
    return rows


def summarize_replay_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [row for row in rows if row["public_key_verified"]]
    below = [row for row in verified if row["below_rho"]]
    relation = [row for row in rows if int(row.get("relation_count") or 0) >= 2]
    ops = [row["ops_over_rho"] for row in rows if row.get("ops_over_rho") is not None]
    return {
        "case_count": len(rows),
        "unique_coordinate_transfer_count": len({row["key"] for row in rows}),
        "relation_case_count": len(relation),
        "verified_case_count": len(verified),
        "verified_below_rho_count": len(below),
        "min_ops_over_rho": round(min(ops), 8) if ops else None,
        "mean_ops_over_rho": mean_or_none([float(value) for value in ops]),
    }


def key_gap_summary(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_keys = {row["key"] for row in left}
    right_keys = {row["key"] for row in right}
    missing_keys = sorted(left_keys - right_keys)
    examples = []
    for key in missing_keys[:16]:
        first = next(row for row in left if row["key"] == key)
        examples.append(first)
    return {
        f"{left_label}_unique_key_count": len(left_keys),
        f"{right_label}_unique_key_count": len(right_keys),
        f"{left_label}_not_in_{right_label}_count": len(missing_keys),
        "examples": examples,
    }


def form_signature_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        features = row.get("form_guard_features") or {}
        signature = str(features.get("candidate_pos_signature"))
        counts[signature] += 1
    return dict(sorted(counts.items()))


def run(args: argparse.Namespace) -> dict[str, Any]:
    coordinate_cases = load_coordinate_cases(
        [parse_named_path(raw) for raw in args.coordinate_gate]
    )
    guard_cases = load_guard_cases([parse_named_path(raw) for raw in args.guard_replay])
    pair_cases = load_pair_cases([parse_named_path(raw) for raw in args.pair_replay])

    coordinate_verified = [row for row in coordinate_cases if row["public_key_verified"]]
    guard_relation = [
        row for row in guard_cases if int(row.get("relation_count") or 0) >= 2
    ]
    pair_selected = pair_cases
    pair_relation = [
        row for row in pair_cases if int(row.get("relation_count") or 0) >= 2
    ]

    return {
        "schema": "ecdlp_public_repeated_coordinate_branch_gap_audit_v1",
        "method": "coordinate_gate_vs_row_guard_vs_strict_pair_gap_audit",
        "parameters": {
            "coordinate_gate": args.coordinate_gate,
            "guard_replay": args.guard_replay,
            "pair_replay": args.pair_replay,
        },
        "summary": {
            "coordinate_gate": summarize_replay_rows(coordinate_cases),
            "coordinate_gate_verified": summarize_replay_rows(coordinate_verified),
            "row_guard": summarize_replay_rows(guard_cases),
            "row_guard_relation": summarize_replay_rows(guard_relation),
            "strict_pair": summarize_replay_rows(pair_selected),
            "strict_pair_relation": summarize_replay_rows(pair_relation),
            "row_guard_relation_form_signatures": form_signature_counts(guard_relation),
            "coordinate_verified_missing_from_strict_pair": key_gap_summary(
                coordinate_verified,
                pair_selected,
                "coordinate_verified",
                "strict_pair",
            ),
            "row_guard_relation_missing_from_strict_pair": key_gap_summary(
                guard_relation,
                pair_selected,
                "row_guard_relation",
                "strict_pair",
            ),
        },
        "non_claims": [
            "This is a diagnostic branch-gap audit, not a selector and not a speedup claim.",
            "Exact coordinates are used only to identify missed branches across existing artifacts.",
            "Coordinate-gate verification over rho does not promote the strict pair rule.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate-gate", action="append", required=True, help="name|path")
    parser.add_argument("--guard-replay", action="append", required=True, help="name|path")
    parser.add_argument("--pair-replay", action="append", required=True, help="name|path")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    out_path = resolve_path(Path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
