#!/usr/bin/env python3
"""Audit candidate-position guard extensions for partial duplicates.

The strict branch-aware guard accepts same-position duplicate relation events.
The 680 backtest found verified below-rho systems with signature ``1,3,3``.
This probe evaluates whether small verifier-free guard variants accept that
partial-duplicate class while rejecting known misaligned decoys.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "ffe_public_repeated_coordinate_partial_duplicate_guard_audit_target67.json"
)
DEFAULT_BRANCH_REPLAYS = [
    (
        "branch_680_687",
        DEFAULT_STATE_DIR
        / "ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_680_687.json",
    ),
    (
        "branch_704_711",
        DEFAULT_STATE_DIR
        / "ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_704_711.json",
    ),
    (
        "branch_728_735",
        DEFAULT_STATE_DIR
        / "ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_728_735.json",
    ),
    (
        "branch_736_743",
        DEFAULT_STATE_DIR
        / "ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_736_743.json",
    ),
    (
        "branch_752_847",
        DEFAULT_STATE_DIR
        / "ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_752_847.json",
    ),
]
DEFAULT_ALIGNMENT_AUDIT = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_847.json"
)
DEFAULT_BRANCH_GAP_AUDIT = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_branch_gap_audit_target67_832_839.json"
)
DEFAULT_ACTIVATION_SCAN = (
    DEFAULT_STATE_DIR
    / "ffe_public_repeated_coordinate_branch_activation_scan_target67_bmod4_saltmod2_subset2_848_903.json"
)


Case = dict[str, Any]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    return json.loads(resolved.read_text()) if resolved.exists() else {}


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


def parse_signature(signature: str) -> list[int]:
    if not signature:
        return []
    out: list[int] = []
    for part in signature.split(","):
        try:
            out.append(int(part))
        except ValueError:
            return []
    return out


def max_duplicate(values: list[int]) -> int:
    counts = Counter(values)
    return max(counts.values()) if counts else 0


def label(public_key_verified: bool, below_rho: bool) -> str:
    if public_key_verified and below_rho:
        return "verified_below_rho"
    if public_key_verified:
        return "verified_over_rho"
    return "unverified"


def normalize_case(
    *,
    source_family: str,
    source: str,
    window: Any,
    transfer_index: Any,
    coordinate: Any,
    signature: str,
    candidate_pos_count: Any,
    candidate_pos_unique_count: Any,
    candidate_pos_span: Any,
    candidate_pos_values: list[int] | None,
    public_key_verified: bool,
    below_rho: bool,
    ops_over_rho: Any,
    rank: Any,
    relation_count: Any,
    weight: int = 1,
    provenance: str = "",
) -> Case:
    values = candidate_pos_values or parse_signature(signature)
    count = int(candidate_pos_count if candidate_pos_count is not None else len(values))
    unique_count = int(
        candidate_pos_unique_count
        if candidate_pos_unique_count is not None
        else len(set(values))
    )
    span = int(
        candidate_pos_span
        if candidate_pos_span is not None
        else ((max(values) - min(values)) if values else -1)
    )
    return {
        "source_family": source_family,
        "source": source,
        "window": window,
        "transfer_index": transfer_index,
        "coordinate": coordinate,
        "candidate_pos_signature": signature,
        "candidate_pos_values": values,
        "candidate_pos_count": count,
        "candidate_pos_unique_count": unique_count,
        "candidate_pos_span": span,
        "candidate_pos_max_duplicate": max_duplicate(values),
        "public_key_verified": public_key_verified,
        "below_rho": below_rho,
        "label": label(public_key_verified, below_rho),
        "ops_over_rho": round_or_none(ops_over_rho),
        "rank": int(rank or 0),
        "relation_count": int(relation_count or 0),
        "weight": int(weight),
        "provenance": provenance,
    }


def relation_cases_from_branch_replay(name: str, path: Path) -> list[Case]:
    data = load_json(path)
    out: list[Case] = []
    for case in data.get("cases") or []:
        replay = case.get("guarded_replay") or {}
        relation_count = int(replay.get("relation_count") or 0)
        if relation_count < 2:
            continue
        features = case.get("form_guard_features") or {}
        out.append(
            normalize_case(
                source_family="branch_aware_subset_replay",
                source=name,
                window=case.get("window"),
                transfer_index=case.get("transfer_index"),
                coordinate=case.get("coordinate"),
                signature=str(features.get("candidate_pos_signature") or ""),
                candidate_pos_count=features.get("candidate_pos_count"),
                candidate_pos_unique_count=features.get("candidate_pos_unique_count"),
                candidate_pos_span=features.get("candidate_pos_span"),
                candidate_pos_values=[
                    int(value) for value in features.get("candidate_pos_values") or []
                ],
                public_key_verified=bool(replay.get("public_key_verified")),
                below_rho=bool(replay.get("below_rho")),
                ops_over_rho=replay.get("ops_over_rho"),
                rank=replay.get("rank"),
                relation_count=relation_count,
                provenance=str(path),
            )
        )
    return out


def relation_cases_from_alignment_audit(path: Path) -> list[Case]:
    data = load_json(path)
    out: list[Case] = []
    for case in data.get("cases") or []:
        replay = case.get("pair_case_replay") or {}
        relation_count = int(replay.get("relation_count") or 0)
        if relation_count < 2:
            continue
        features = case.get("system_features") or {}
        out.append(
            normalize_case(
                source_family="strict_pair_alignment_audit",
                source="alignment_752_847",
                window=case.get("window"),
                transfer_index=case.get("transfer_index"),
                coordinate=case.get("coordinate"),
                signature=str(features.get("candidate_pos_signature") or ""),
                candidate_pos_count=features.get("candidate_pos_count"),
                candidate_pos_unique_count=features.get("candidate_pos_unique_count"),
                candidate_pos_span=features.get("candidate_pos_span"),
                candidate_pos_values=None,
                public_key_verified=bool(replay.get("public_key_verified")),
                below_rho=bool(replay.get("below_rho")),
                ops_over_rho=replay.get("ops_over_rho"),
                rank=replay.get("rank"),
                relation_count=relation_count,
                provenance=str(path),
            )
        )
    return out


def row_guard_decoys_from_branch_gap(path: Path) -> list[Case]:
    data = load_json(path)
    summary = data.get("summary") or {}
    relation = summary.get("row_guard_relation") or {}
    missing = summary.get("row_guard_relation_missing_from_strict_pair") or {}
    examples = missing.get("examples") or []
    if not examples:
        return []
    total_weight = int(relation.get("case_count") or len(examples))
    per_example_weight = max(1, total_weight // len(examples))
    out: list[Case] = []
    for example in examples:
        features = example.get("form_guard_features") or {}
        out.append(
            normalize_case(
                source_family="row_form_branch_gap_aggregate",
                source="branch_gap_832_839_row_guard_relation",
                window=example.get("source"),
                transfer_index=example.get("transfer_index"),
                coordinate=example.get("coordinate"),
                signature=str(features.get("candidate_pos_signature") or ""),
                candidate_pos_count=features.get("candidate_pos_count"),
                candidate_pos_unique_count=features.get("candidate_pos_unique_count"),
                candidate_pos_span=features.get("candidate_pos_span"),
                candidate_pos_values=[
                    int(value) for value in features.get("candidate_pos_values") or []
                ],
                public_key_verified=bool(example.get("public_key_verified")),
                below_rho=bool(example.get("below_rho")),
                ops_over_rho=example.get("ops_over_rho"),
                rank=example.get("rank"),
                relation_count=example.get("relation_count"),
                weight=per_example_weight,
                provenance=str(path),
            )
        )
    return out


def dry_window_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    summary = data.get("summary") or {}
    windows = [
        window
        for window in data.get("windows") or []
        if isinstance(window, dict)
        and not window.get("selected_public_case_count")
        and not window.get("coordinate_candidate_count")
        and not window.get("branch_clause_match_count")
    ]
    return {
        "path": str(path),
        "dry_window_count": len(windows),
        "dry_windows": [window.get("window") for window in windows],
        "scan_summary": summary,
    }


def strict_same_position(case: Case) -> bool:
    return (
        int(case["candidate_pos_span"]) == 0
        and int(case["candidate_pos_count"]) >= 2
        and int(case["candidate_pos_unique_count"]) == 1
    )


def partial_signature_133(case: Case) -> bool:
    return str(case["candidate_pos_signature"]) == "1,3,3"


def partial_count3_unique2(case: Case) -> bool:
    return (
        int(case["candidate_pos_count"]) == 3
        and int(case["candidate_pos_unique_count"]) == 2
        and int(case["candidate_pos_max_duplicate"]) >= 2
    )


def broad_partial_duplicate(case: Case) -> bool:
    return (
        int(case["candidate_pos_count"]) >= 3
        and int(case["candidate_pos_unique_count"]) == 2
        and int(case["candidate_pos_max_duplicate"]) >= 2
    )


def any_misaligned_relation(case: Case) -> bool:
    return int(case["candidate_pos_count"]) >= 2 and int(case["candidate_pos_span"]) > 0


GUARDS: dict[str, Callable[[Case], bool]] = {
    "strict_same_position": strict_same_position,
    "strict_or_partial_signature_1_3_3": lambda case: strict_same_position(case)
    or partial_signature_133(case),
    "strict_or_partial_count3_unique2": lambda case: strict_same_position(case)
    or partial_count3_unique2(case),
    "strict_or_broad_partial_duplicate": lambda case: strict_same_position(case)
    or broad_partial_duplicate(case),
    "strict_or_any_misaligned_relation": lambda case: strict_same_position(case)
    or any_misaligned_relation(case),
}


def weighted_count(cases: list[Case]) -> int:
    return sum(int(case.get("weight") or 1) for case in cases)


def summarize_cases(cases: list[Case]) -> dict[str, Any]:
    by_label: Counter[str] = Counter()
    by_signature: Counter[str] = Counter()
    by_source_family: Counter[str] = Counter()
    ops = [float(case["ops_over_rho"]) for case in cases if case["ops_over_rho"] is not None]
    for case in cases:
        weight = int(case.get("weight") or 1)
        by_label.update({str(case["label"]): weight})
        by_signature.update({str(case["candidate_pos_signature"]): weight})
        by_source_family.update({str(case["source_family"]): weight})
    return {
        "case_count": len(cases),
        "weighted_case_count": weighted_count(cases),
        "label_counts": dict(sorted(by_label.items())),
        "signature_counts": dict(sorted(by_signature.items())),
        "source_family_counts": dict(sorted(by_source_family.items())),
        "min_ops_over_rho": round(min(ops), 8) if ops else None,
        "mean_ops_over_rho": mean_or_none(ops),
    }


def compact_case(case: Case) -> dict[str, Any]:
    keys = [
        "source_family",
        "source",
        "window",
        "transfer_index",
        "coordinate",
        "candidate_pos_signature",
        "candidate_pos_values",
        "label",
        "ops_over_rho",
        "rank",
        "relation_count",
        "weight",
    ]
    return {key: case.get(key) for key in keys}


def evaluate_guard(name: str, predicate: Callable[[Case], bool], cases: list[Case]) -> dict[str, Any]:
    accepted = [case for case in cases if predicate(case)]
    rejected = [case for case in cases if not predicate(case)]
    accepted_labels = Counter()
    rejected_labels = Counter()
    for case in accepted:
        accepted_labels.update({str(case["label"]): int(case.get("weight") or 1)})
    for case in rejected:
        rejected_labels.update({str(case["label"]): int(case.get("weight") or 1)})
    accepted_ops = [
        float(case["ops_over_rho"]) for case in accepted if case["ops_over_rho"] is not None
    ]
    return {
        "guard": name,
        "accepted_case_count": len(accepted),
        "accepted_weighted_case_count": weighted_count(accepted),
        "rejected_weighted_case_count": weighted_count(rejected),
        "accepted_label_counts": dict(sorted(accepted_labels.items())),
        "rejected_label_counts": dict(sorted(rejected_labels.items())),
        "accepted_unverified_count": accepted_labels.get("unverified", 0),
        "accepted_verified_over_rho_count": accepted_labels.get("verified_over_rho", 0),
        "accepted_verified_below_rho_count": accepted_labels.get("verified_below_rho", 0),
        "rejected_verified_below_rho_count": rejected_labels.get("verified_below_rho", 0),
        "accepted_signature_counts": dict(
            sorted(
                Counter(
                    {
                        signature: weighted_count(
                            [
                                case
                                for case in accepted
                                if case["candidate_pos_signature"] == signature
                            ]
                        )
                        for signature in {case["candidate_pos_signature"] for case in accepted}
                    }
                ).items()
            )
        ),
        "accepted_min_ops_over_rho": round(min(accepted_ops), 8) if accepted_ops else None,
        "accepted_mean_ops_over_rho": mean_or_none(accepted_ops),
        "accepted_unverified_examples": [
            compact_case(case) for case in accepted if case["label"] == "unverified"
        ][:16],
        "accepted_verified_over_rho_examples": [
            compact_case(case) for case in accepted if case["label"] == "verified_over_rho"
        ][:16],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    branch_replays = (
        [parse_named_path(raw) for raw in args.branch_replay]
        if args.branch_replay
        else DEFAULT_BRANCH_REPLAYS
    )
    cases: list[Case] = []
    for name, path in branch_replays:
        cases.extend(relation_cases_from_branch_replay(name, path))
    cases.extend(relation_cases_from_alignment_audit(Path(args.alignment_audit)))
    cases.extend(row_guard_decoys_from_branch_gap(Path(args.branch_gap_audit)))
    dry = dry_window_summary(Path(args.activation_scan))
    evaluations = {name: evaluate_guard(name, predicate, cases) for name, predicate in GUARDS.items()}
    output = {
        "schema": "ecdlp_public_repeated_coordinate_partial_duplicate_guard_audit_v1",
        "method": "retrospective_candidate_position_guard_extension_audit",
        "parameters": {
            "branch_replays": [f"{name}|{path}" for name, path in branch_replays],
            "alignment_audit": str(args.alignment_audit),
            "branch_gap_audit": str(args.branch_gap_audit),
            "activation_scan": str(args.activation_scan),
        },
        "summary": {
            **summarize_cases(cases),
            "dry_window_summary": dry,
            "recommended_candidate_guard": "strict_or_partial_count3_unique2",
            "recommendation_boundary": (
                "Current corpus accepts the 680 1,3,3 below-rho class and rejects known "
                "1,2, 1,3, and 1,1,3,3 decoys/controls, but support is one older window."
            ),
        },
        "guard_evaluations": evaluations,
        "representative_cases": [compact_case(case) for case in cases[:64]],
        "non_claims": [
            "This audit is retrospective and does not prove a new ECDLP speedup.",
            "Candidate-position signatures are relation-stage form features, not pre-activation selectors.",
            "A guard extension must be frozen before a future nonempty replay window to become a claim.",
        ],
    }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-replay", action="append", default=None, help="name|path")
    parser.add_argument("--alignment-audit", default=str(DEFAULT_ALIGNMENT_AUDIT))
    parser.add_argument("--branch-gap-audit", default=str(DEFAULT_BRANCH_GAP_AUDIT))
    parser.add_argument("--activation-scan", default=str(DEFAULT_ACTIVATION_SCAN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    out_path = resolve_path(Path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(json.dumps(output["guard_evaluations"], indent=2, sort_keys=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
