#!/usr/bin/env python3
"""Audit a frozen proxy reject guard on repeated-coordinate relation cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_AUDIT = (
    DEFAULT_STATE_DIR
    / "ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_815.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_815.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


def parse_clause(raw: str) -> tuple[str, ...]:
    atoms = tuple(atom.strip() for atom in raw.split("&") if atom.strip())
    if not atoms:
        raise argparse.ArgumentTypeError("reject clause must contain at least one atom")
    return atoms


def feature_atoms(features: dict[str, Any]) -> set[str]:
    atoms: set[str] = set()
    for key, value in features.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float, str)):
            atoms.add(f"{key}={value}")
    return atoms


def relation_cases(audit: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in audit.get("cases") or []:
        replay = case.get("pair_case_replay") or {}
        if int(replay.get("relation_count") or 0) < 2:
            continue
        out.append(case)
    return out


def case_label(case: dict[str, Any]) -> str:
    replay = case.get("pair_case_replay") or {}
    if replay.get("public_key_verified") and replay.get("below_rho"):
        return "verified_below_rho"
    if replay.get("public_key_verified"):
        return "verified_over_rho"
    return "unverified"


def matching_clauses(case_atoms: set[str], clauses: list[tuple[str, ...]]) -> list[str]:
    matched: list[str] = []
    for clause in clauses:
        if all(atom in case_atoms for atom in clause):
            matched.append("&".join(clause))
    return matched


def compact_case(case: dict[str, Any], matched: list[str]) -> dict[str, Any]:
    replay = case.get("pair_case_replay") or {}
    features = case.get("system_features") or {}
    coordinate = case.get("coordinate") or {}
    return {
        "window": case.get("window"),
        "transfer_index": case.get("transfer_index"),
        "coordinate": coordinate,
        "label": case_label(case),
        "matched_reject_clauses": matched,
        "public_key_verified": bool(replay.get("public_key_verified")),
        "below_rho": bool(replay.get("below_rho")),
        "ops_over_rho": replay.get("ops_over_rho"),
        "rank": replay.get("rank"),
        "relation_count": replay.get("relation_count"),
        "candidate_pos_signature": features.get("candidate_pos_signature"),
        "candidate_pos_span": features.get("candidate_pos_span"),
        "candidate_pos_aligned": features.get("candidate_pos_aligned"),
    }


def summarize(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(item["label"]) for item in evaluated)
    rejected = [item for item in evaluated if item["matched_reject_clauses"]]
    kept = [item for item in evaluated if not item["matched_reject_clauses"]]
    by_window: dict[str, Counter[str]] = {}
    by_clause: dict[str, Counter[str]] = {}
    for item in evaluated:
        state = "rejected" if item["matched_reject_clauses"] else "kept"
        by_window.setdefault(str(item["window"]), Counter()).update([f"{state}_{item['label']}"])
        for clause in item["matched_reject_clauses"]:
            by_clause.setdefault(clause, Counter()).update([str(item["label"])])
    return {
        "input_relation_case_count": len(evaluated),
        "label_counts": dict(sorted(label_counts.items())),
        "kept_case_count": len(kept),
        "rejected_case_count": len(rejected),
        "kept_verified_below_rho_count": sum(
            1 for item in kept if item["label"] == "verified_below_rho"
        ),
        "rejected_verified_count": sum(1 for item in rejected if item["label"].startswith("verified")),
        "rejected_unverified_count": sum(1 for item in rejected if item["label"] == "unverified"),
        "kept_unverified_count": sum(1 for item in kept if item["label"] == "unverified"),
        "by_window": {key: dict(sorted(value.items())) for key, value in sorted(by_window.items())},
        "by_clause": {key: dict(sorted(value.items())) for key, value in sorted(by_clause.items())},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_clauses = args.reject_clause or ["transfer_index_mod3=1"]
    clauses = [parse_clause(raw) for raw in raw_clauses]
    audit_path = resolve_path(Path(args.audit))
    cases = relation_cases(load_json(audit_path))
    evaluated: list[dict[str, Any]] = []
    for case in cases:
        atoms = feature_atoms(case.get("system_features") or {})
        matched = matching_clauses(atoms, clauses)
        evaluated.append(compact_case(case, matched))
    output = {
        "schema": "ecdlp_public_repeated_coordinate_proxy_guard_filter_audit_v1",
        "method": "frozen_proxy_reject_guard_filter_audit",
        "parameters": {
            "audit": str(args.audit),
            "reject_clauses": ["&".join(clause) for clause in clauses],
        },
        "summary": summarize(evaluated),
        "rejected_cases": [item for item in evaluated if item["matched_reject_clauses"]],
        "kept_unverified_cases": [
            item for item in evaluated if item["label"] == "unverified" and not item["matched_reject_clauses"]
        ],
        "non_claims": [
            "This filter audit uses existing replay labels; it is not a fresh-window validation.",
            "A proxy guard must be frozen before future replay to support any selection claim.",
            "Rejecting decoys is useful only if verified below-rho cases are not false-rejected.",
        ],
    }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument(
        "--reject-clause",
        action="append",
        default=None,
        help="Conjunctive atom clause; repeat for OR rejection.",
    )
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
