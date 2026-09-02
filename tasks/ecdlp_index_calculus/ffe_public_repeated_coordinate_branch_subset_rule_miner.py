#!/usr/bin/env python3
"""Mine public subset rules inside branch-aware repeated-coordinate replays.

The branch-aware replay first filters coordinate-gate candidates by a frozen
public branch selector, then tries public row subsets.  This miner treats
verified-below-rho guarded subset recoveries as labels and searches for small
pair-feature clauses that could replace the blunt "try all subsets" stage.
Exact coordinates, exact salts, row keys, and full-coordinate replay labels are
excluded from candidate atoms.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_branch_subset_rule_miner.json"

EXCLUDED_FEATURE_KEYS = {
    "coordinate_key",
    "full_ops_millirhos",
    "full_rank",
    "full_relation_count",
    "pair_salt_signature",
    "row_keys",
    "source_ops_millirhos",
    "target",
    "window",
}

THRESHOLD_KEYS = {
    "pair_salt_span",
    "pair_salt_index_span",
    "salt_span",
    "subset_size",
    "top_k",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text())


def parse_named_path(raw: str) -> tuple[str, Path]:
    parts = raw.split("|", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("value must be name|path")
    return parts[0], Path(parts[1])


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def label_case(case: dict[str, Any]) -> str:
    replay = case.get("guarded_replay") or {}
    if replay.get("guarded_public_key_verified") and replay.get("guarded_below_rho"):
        return "verified_below_rho"
    if replay.get("guarded_public_key_verified"):
        return "verified_over_rho"
    if int(replay.get("relation_count") or 0) >= 2 and case.get("form_guard_passed"):
        return "unverified_relation"
    if int(replay.get("relation_count") or 0) >= 2:
        return "guard_rejected_relation"
    return "no_relation"


def compact_case(case: dict[str, Any], source: str) -> dict[str, Any]:
    replay = case.get("guarded_replay") or {}
    pair = case.get("pair_features") or {}
    return {
        "source": source,
        "window": case.get("window"),
        "target": case.get("target"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "coordinate": case.get("coordinate"),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "leaf_selector": case.get("leaf_selector"),
        "candidate_clauses": case.get("candidate_clauses") or [],
        "pair_features": pair,
        "label": label_case(case),
        "public_key_verified": bool(replay.get("guarded_public_key_verified")),
        "below_rho": bool(replay.get("guarded_below_rho")),
        "ops_over_rho": replay.get("ops_over_rho"),
        "rank": replay.get("rank"),
        "relation_count": int(replay.get("relation_count") or 0),
        "row_keys": replay.get("row_keys") or [],
    }


def load_cases(named_paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for source, path in named_paths:
        data = load_json(path)
        for case in data.get("cases") or []:
            if isinstance(case, dict):
                cases.append(compact_case(case, source))
    return cases


def atom_matches(atom: str, features: dict[str, Any]) -> bool:
    if ">=" in atom:
        key, raw_value = atom.split(">=", 1)
        return int(features[key]) >= int(raw_value)
    if "<=" in atom:
        key, raw_value = atom.split("<=", 1)
        return int(features[key]) <= int(raw_value)
    key, raw_value = atom.split("=", 1)
    value = features.get(key)
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        return value == int(raw_value)
    return str(value) == raw_value


def clause_matches(clause: tuple[str, ...], case: dict[str, Any]) -> bool:
    features = case.get("pair_features") or {}
    return all(atom_matches(atom, features) for atom in clause)


def selected_cases(clause: tuple[str, ...], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [case for case in cases if clause_matches(clause, case)]


def atoms_for_case(case: dict[str, Any]) -> set[str]:
    atoms: set[str] = set()
    for key, value in sorted((case.get("pair_features") or {}).items()):
        if key in EXCLUDED_FEATURE_KEYS or value is None or value == "":
            continue
        if isinstance(value, list):
            continue
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int):
            atoms.add(f"{key}={value}")
            if key in THRESHOLD_KEYS:
                atoms.add(f"{key}>={value}")
                atoms.add(f"{key}<={value}")
        elif isinstance(value, (float, str)):
            atoms.add(f"{key}={value}")
    return atoms


def candidate_atoms(cases: list[dict[str, Any]], max_atoms: int) -> list[str]:
    counts = {"pos": Counter(), "neg": Counter()}
    for case in cases:
        bucket = "pos" if case["label"] == "verified_below_rho" else "neg"
        counts[bucket].update(atoms_for_case(case))
    ranked = []
    for atom, pos_count in counts["pos"].items():
        neg_count = counts["neg"].get(atom, 0)
        precision = pos_count / (pos_count + neg_count)
        ranked.append((precision, pos_count, -neg_count, atom))
    ranked.sort(reverse=True)
    return [atom for _precision, _pos_count, _neg_count, atom in ranked[:max_atoms]]


def compact_example(case: dict[str, Any]) -> dict[str, Any]:
    pair = case.get("pair_features") or {}
    return {
        "window": case.get("window"),
        "transfer_index": case.get("transfer_index"),
        "coordinate": case.get("coordinate"),
        "label": case.get("label"),
        "ops_over_rho": case.get("ops_over_rho"),
        "row_keys": case.get("row_keys"),
        "pair_summary": {
            key: pair.get(key)
            for key in (
                "subset_size",
                "policy_family",
                "leaf_selector",
                "leaf_signature",
                "pair_leaf_signature",
                "pair_salt_index_signature",
                "pair_salt_delta_from_min_signature",
                "pair_salt_span",
                "pair_salt_sum_mod8",
                "transfer_mod3",
            )
        },
    }


def score_clause(clause: tuple[str, ...], cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected = selected_cases(clause, cases)
    positives = [case for case in cases if case["label"] == "verified_below_rho"]
    true_pos = [case for case in selected if case["label"] == "verified_below_rho"]
    false_pos = [case for case in selected if case["label"] != "verified_below_rho"]
    ops = [
        float(case["ops_over_rho"])
        for case in true_pos
        if case.get("ops_over_rho") is not None
    ]
    return {
        "clause": "&".join(clause),
        "selected_count": len(selected),
        "true_positive_count": len(true_pos),
        "false_positive_count": len(false_pos),
        "false_negative_count": len(positives) - len(true_pos),
        "positive_count": len(positives),
        "precision": round(len(true_pos) / len(selected), 8) if selected else None,
        "recall": round(len(true_pos) / len(positives), 8) if positives else None,
        "positive_windows": sorted({str(case.get("window")) for case in true_pos}),
        "min_true_positive_ops_over_rho": round(min(ops), 8) if ops else None,
        "mean_true_positive_ops_over_rho": mean_or_none(ops),
        "selected_label_counts": dict(sorted(Counter(case["label"] for case in selected).items())),
        "selected_examples": [compact_example(case) for case in selected[:12]],
    }


def rank_score(score: dict[str, Any]) -> tuple[Any, ...]:
    return (
        score["false_positive_count"],
        -score["true_positive_count"],
        score["false_negative_count"],
        len(score["clause"].split("&")),
        score["clause"],
    )


def mine(cases: list[dict[str, Any]], max_clause_size: int, max_atoms: int, top_rules: int) -> list[dict[str, Any]]:
    atoms = candidate_atoms(cases, max_atoms)
    scores: list[dict[str, Any]] = []
    for size in range(1, max_clause_size + 1):
        for clause in itertools.combinations(atoms, size):
            score = score_clause(clause, cases)
            if score["true_positive_count"] == 0:
                continue
            scores.append(score)
    scores.sort(key=rank_score)
    return scores[:top_rules]


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(case["label"] for case in cases)
    by_window: dict[str, Counter[str]] = {}
    for case in cases:
        by_window.setdefault(str(case.get("window")), Counter()).update([case["label"]])
    return {
        "case_count": len(cases),
        "label_counts": dict(sorted(label_counts.items())),
        "window_label_counts": {
            window: dict(sorted(counts.items())) for window, counts in sorted(by_window.items())
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    named_paths = [parse_named_path(raw) for raw in args.replay]
    cases = load_cases(named_paths)
    return {
        "schema": "ecdlp_public_repeated_coordinate_branch_subset_rule_miner_v1",
        "method": "public_pair_feature_rule_search_inside_frozen_branch_replay",
        "parameters": {
            "replays": [f"{name}|{path}" for name, path in named_paths],
            "max_clause_size": int(args.max_clause_size),
            "max_atoms": int(args.max_atoms),
            "top_rules": int(args.top_rules),
        },
        "summary": summarize(cases),
        "top_rules": mine(cases, int(args.max_clause_size), int(args.max_atoms), int(args.top_rules)),
        "non_claims": [
            "Rules are mined from replay labels and must be frozen before future-window validation.",
            "Candidate atoms exclude exact coordinates, exact salts, row keys, and full-coordinate replay labels.",
            "A subset rule is not a standalone speedup until branch selection, subset selection, and replay costs are all charged.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="append", required=True, help="name|path")
    parser.add_argument("--max-clause-size", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=256)
    parser.add_argument("--top-rules", type=int, default=16)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": output["summary"], "top_rules": output["top_rules"][:8]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
