#!/usr/bin/env python3
"""Audit relation-case reach for repeated-coordinate pair replays.

Recent fresh windows can select public pair/subset cases without producing any
relation systems.  This probe keeps that bottleneck separate from downstream
candidate-position alignment and public-key verification.
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
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_relation_reach_audit.json"
)

EXCLUDED_FEATURE_KEYS = {
    "coordinate_key",
    "full_ops_millirhos",
    "full_rank",
    "full_relation_count",
    "pair_salt_signature",
    "row_keys",
    "target",
    "transfer_index",
    "window",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


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


def compact_case(case: dict[str, Any], source: str) -> dict[str, Any]:
    replay = case.get("guarded_replay") or {}
    features = case.get("pair_features") or {}
    relation_count = int(replay.get("relation_count") or 0)
    public_key_verified = bool(replay.get("public_key_verified"))
    below_rho = bool(replay.get("below_rho"))
    return {
        "case_key": case.get("case_key"),
        "source": source,
        "window": case.get("window"),
        "target": case.get("target"),
        "transfer_index": case.get("transfer_index"),
        "top_k": case.get("top_k"),
        "policy": case.get("policy"),
        "leaf_selector": case.get("leaf_selector"),
        "coordinate": case.get("coordinate"),
        "pair_features": features,
        "relation_reached": relation_count >= 2,
        "guard_passed": bool(case.get("form_guard_passed")),
        "public_key_verified": public_key_verified,
        "below_rho": below_rho,
        "ops_over_rho": round_or_none(replay.get("ops_over_rho")),
        "rank": int(replay.get("rank") or 0),
        "relation_count": relation_count,
        "selected_row_count": int(replay.get("selected_row_count") or 0),
        "selected_leaf_count": int(replay.get("selected_leaf_count") or 0),
    }


def load_cases(named_paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for name, path in named_paths:
        data = load_json(resolve_path(path))
        for case in data.get("cases") or []:
            cases.append(compact_case(case, name))
    return cases


def feature_atoms(features: dict[str, Any]) -> set[str]:
    atoms: set[str] = set()
    for key, value in sorted(features.items()):
        if key in EXCLUDED_FEATURE_KEYS or value is None or value == "":
            continue
        if isinstance(value, list):
            continue
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float, str)):
            atoms.add(f"{key}={value}")
    return atoms


def atom_separators(cases: list[dict[str, Any]]) -> dict[str, Any]:
    reached = [case for case in cases if case["relation_reached"]]
    missed = [case for case in cases if not case["relation_reached"]]
    counts = {"reached": Counter(), "missed": Counter()}
    for case in reached:
        counts["reached"].update(feature_atoms(case.get("pair_features") or {}))
    for case in missed:
        counts["missed"].update(feature_atoms(case.get("pair_features") or {}))
    reached_only = [
        {"atom": atom, "reached_count": count, "missed_count": counts["missed"].get(atom, 0)}
        for atom, count in counts["reached"].items()
        if count and not counts["missed"].get(atom)
    ]
    missed_only = [
        {"atom": atom, "missed_count": count, "reached_count": counts["reached"].get(atom, 0)}
        for atom, count in counts["missed"].items()
        if count and not counts["reached"].get(atom)
    ]
    reached_only.sort(key=lambda item: (-int(item["reached_count"]), item["atom"]))
    missed_only.sort(key=lambda item: (-int(item["missed_count"]), item["atom"]))
    return {
        "reached_only_atoms": reached_only[:32],
        "missed_only_atoms": missed_only[:32],
    }


def summarize_group(cases: list[dict[str, Any]]) -> dict[str, Any]:
    reached = [case for case in cases if case["relation_reached"]]
    guard_passed = [case for case in cases if case["guard_passed"]]
    verified = [case for case in cases if case["public_key_verified"]]
    below = [case for case in cases if case["public_key_verified"] and case["below_rho"]]
    ops = [case["ops_over_rho"] for case in cases if case.get("ops_over_rho") is not None]
    return {
        "selected_pair_case_count": len(cases),
        "relation_pair_case_count": len(reached),
        "guard_passed_case_count": len(guard_passed),
        "verified_case_count": len(verified),
        "verified_below_rho_count": len(below),
        "min_ops_over_rho": round(min(ops), 8) if ops else None,
        "mean_ops_over_rho": mean_or_none([float(value) for value in ops]),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_window: dict[str, list[dict[str, Any]]] = {}
    by_transfer: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_window.setdefault(str(case.get("window")), []).append(case)
        by_transfer.setdefault(f"{case.get('window')}:{case.get('transfer_index')}", []).append(case)
    window_summary = {
        window: summarize_group(group) for window, group in sorted(by_window.items())
    }
    transfer_summary = {
        transfer: summarize_group(group) for transfer, group in sorted(by_transfer.items())
    }
    dead_windows = [
        window
        for window, item in window_summary.items()
        if item["selected_pair_case_count"] and not item["relation_pair_case_count"]
    ]
    relation_windows = [
        window for window, item in window_summary.items() if item["relation_pair_case_count"]
    ]
    return {
        **summarize_group(cases),
        "window_summary": window_summary,
        "transfer_summary": transfer_summary,
        "dead_selection_windows": dead_windows,
        "relation_reach_windows": relation_windows,
        "separator_summary": atom_separators(cases),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    named_paths = [parse_named_path(raw) for raw in args.pair_replay]
    cases = load_cases(named_paths)
    output = {
        "schema": "ecdlp_public_repeated_coordinate_relation_reach_audit_v1",
        "method": "selected_pair_case_relation_reach_audit",
        "parameters": {
            "pair_replays": [f"{name}|{path}" for name, path in named_paths],
        },
        "summary": summarize(cases),
        "non_claims": [
            "Relation reach is not public-key verification and is not an ECDLP speedup claim.",
            "Feature separators are diagnostic until frozen and replayed on future windows.",
            "This audit separates dead public-pair selection from candidate-position alignment.",
        ],
    }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
