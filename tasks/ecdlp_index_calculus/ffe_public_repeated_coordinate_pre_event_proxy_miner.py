#!/usr/bin/env python3
"""Mine candidate-position proxy guards from repeated-coordinate audits.

The alignment audit showed that verified pair-rule systems have duplicate
candidate-position events while decoys have split positions.  This probe asks a
more conservative question: can public pair metadata or form residual metadata,
excluding candidate-position labels, reject the known split-position decoys
without rejecting verified systems?

The output is diagnostic.  Any mined guard must be frozen before use on a future
window.
"""

from __future__ import annotations

import argparse
import itertools
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
    / "ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_815.json"
)


PUBLIC_PAIR_KEYS = {
    "b_minus_c_mod16",
    "b_mod5",
    "b_mod8",
    "b_mod16",
    "c_mod5",
    "c_mod8",
    "c_mod16",
    "leaf_selector",
    "leaf_signature",
    "leaf_total",
    "materialized_row_count",
    "pair_leaf_signature",
    "pair_salt_delta_from_min_signature",
    "pair_salt_delta_to_max_signature",
    "pair_salt_index_signature",
    "pair_salt_index_span",
    "pair_salt_max_mod5",
    "pair_salt_min_mod5",
    "pair_salt_span",
    "pair_salt_sum_mod5",
    "pair_salt_sum_mod8",
    "policy_family",
    "row_count",
    "salt_count",
    "salt_span",
    "selected_leaf_count",
    "selected_row_count",
    "source_ops_millirhos",
    "subset_size",
    "top_k",
    "transfer_index_mod2",
    "transfer_index_mod3",
    "transfer_index_mod4",
    "transfer_index_mod5",
    "transfer_index_mod8",
    "transfer_index_mod16",
}

FORM_RESIDUAL_EXTRA_KEYS = {
    "coeff_mod2_signature",
    "coeff_mod4_signature",
    "coeff_support_signature",
    "event_leaf_signature",
    "rhs_mod16_signature",
    "row_event_count_signature",
    "scheduled_trial_signature",
    "term_shape_signature",
}

EXACT_OR_POSITION_PREFIXES = (
    "candidate_",
)

EXACT_OR_LABEL_KEYS = {
    "b",
    "c",
    "duplicate_form_count",
    "event_count",
    "event_salt_signature",
    "event_salt_span",
    "full_ops_millirhos",
    "leaf_min",
    "max_events_per_row",
    "ops_over_rho_bucket",
    "original_trial_signature",
    "pair_rule_name",
    "pair_salt_signature",
    "rank",
    "relation_count",
    "target",
    "transfer_index",
    "unique_form_count",
    "window",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


def case_label(case: dict[str, Any]) -> str:
    replay = case.get("pair_case_replay") or {}
    if replay.get("public_key_verified") and replay.get("below_rho"):
        return "verified_below_rho"
    if replay.get("public_key_verified"):
        return "verified_over_rho"
    return "unverified"


def relation_cases(audit: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in audit.get("cases") or []:
        replay = case.get("pair_case_replay") or {}
        if int(replay.get("relation_count") or 0) < 2:
            continue
        enriched = dict(case)
        enriched["label"] = case_label(case)
        out.append(enriched)
    return out


def atoms_for(case: dict[str, Any], keys: set[str]) -> set[str]:
    features = case.get("system_features") or {}
    atoms: set[str] = set()
    for key in sorted(keys):
        if key in EXACT_OR_LABEL_KEYS or key.startswith(EXACT_OR_POSITION_PREFIXES):
            continue
        value = features.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float, str)):
            atoms.add(f"{key}={value}")
    return atoms


def clause_text(clause: tuple[str, ...]) -> str:
    return "&".join(clause)


def matches_clause(case_atoms: set[str], clause: tuple[str, ...]) -> bool:
    return all(atom in case_atoms for atom in clause)


def clause_stats(
    clause: tuple[str, ...],
    cases: list[dict[str, Any]],
    atom_cache: dict[str, set[str]],
) -> dict[str, Any]:
    matched = [case for case in cases if matches_clause(atom_cache[case["case_key"]], clause)]
    verified = [case for case in matched if case.get("label") != "unverified"]
    unverified = [case for case in matched if case.get("label") == "unverified"]
    windows = sorted({str(case.get("window")) for case in matched})
    transfer_indices = sorted({int(case.get("transfer_index") or 0) for case in matched})
    coordinates = sorted(
        {
            f"{(case.get('coordinate') or {}).get('b')},{(case.get('coordinate') or {}).get('c')}"
            for case in matched
        }
    )
    return {
        "clause": clause_text(clause),
        "size": len(clause),
        "matched_case_count": len(matched),
        "verified_count": len(verified),
        "unverified_count": len(unverified),
        "windows": windows,
        "transfer_indices": transfer_indices,
        "coordinates": coordinates,
    }


def mine_clauses(
    cases: list[dict[str, Any]],
    atom_cache: dict[str, set[str]],
    *,
    max_size: int,
    max_atoms: int,
) -> dict[str, Any]:
    positive_cases = [case for case in cases if case.get("label") != "unverified"]
    negative_cases = [case for case in cases if case.get("label") == "unverified"]
    atom_counts: Counter[str] = Counter()
    for case in cases:
        atom_counts.update(atom_cache[case["case_key"]])
    ranked_atoms = [
        atom
        for atom, _ in sorted(
            atom_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ][:max_atoms]

    negative_guard_candidates: list[dict[str, Any]] = []
    positive_selector_candidates: list[dict[str, Any]] = []
    for size in range(1, max_size + 1):
        for clause in itertools.combinations(ranked_atoms, size):
            stats = clause_stats(clause, cases, atom_cache)
            if stats["unverified_count"] and stats["verified_count"] == 0:
                negative_guard_candidates.append(stats)
            if stats["verified_count"] and stats["unverified_count"] == 0:
                positive_selector_candidates.append(stats)

    negative_guard_candidates.sort(
        key=lambda item: (-int(item["unverified_count"]), int(item["size"]), item["clause"])
    )
    positive_selector_candidates.sort(
        key=lambda item: (-int(item["verified_count"]), int(item["size"]), item["clause"])
    )
    return {
        "atom_count": len(atom_counts),
        "ranked_atoms_considered": ranked_atoms,
        "negative_guard_candidates": negative_guard_candidates[:32],
        "positive_selector_candidates": positive_selector_candidates[:32],
    }


def fold_stats(
    clause: tuple[str, ...],
    cases: list[dict[str, Any]],
    atom_cache: dict[str, set[str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for window in sorted({str(case.get("window")) for case in cases}):
        train = [case for case in cases if str(case.get("window")) != window]
        test = [case for case in cases if str(case.get("window")) == window]
        train_stats = clause_stats(clause, train, atom_cache)
        test_stats = clause_stats(clause, test, atom_cache)
        out.append(
            {
                "heldout_window": window,
                "train_verified_count": train_stats["verified_count"],
                "train_unverified_count": train_stats["unverified_count"],
                "test_verified_count": test_stats["verified_count"],
                "test_unverified_count": test_stats["unverified_count"],
            }
        )
    return out


def stable_negative_candidates(
    cases: list[dict[str, Any]],
    atom_cache: dict[str, set[str]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in candidates:
        clause = tuple(item["clause"].split("&"))
        folds = fold_stats(clause, cases, atom_cache)
        false_rejects = sum(int(fold["test_verified_count"]) for fold in folds)
        rejected_negatives = sum(int(fold["test_unverified_count"]) for fold in folds)
        out.append(
            {
                **item,
                "leave_one_window_false_rejects": false_rejects,
                "leave_one_window_rejected_unverified": rejected_negatives,
                "leave_one_window": folds,
            }
        )
    out.sort(
        key=lambda item: (
            int(item["leave_one_window_false_rejects"]),
            -int(item["leave_one_window_rejected_unverified"]),
            int(item["size"]),
            item["clause"],
        )
    )
    return out[:16]


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(case.get("label")) for case in cases)
    by_window: dict[str, Counter[str]] = {}
    for case in cases:
        by_window.setdefault(str(case.get("window")), Counter()).update([str(case.get("label"))])
    return {
        "relation_case_count": len(cases),
        "label_counts": dict(sorted(labels.items())),
        "window_label_counts": {
            window: dict(sorted(counts.items())) for window, counts in sorted(by_window.items())
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = resolve_path(Path(args.audit))
    audit = load_json(audit_path)
    cases = relation_cases(audit)
    surfaces = {
        "public_pair_no_position_no_exact": PUBLIC_PAIR_KEYS,
        "form_residual_no_position_no_exact": PUBLIC_PAIR_KEYS | FORM_RESIDUAL_EXTRA_KEYS,
    }
    surface_results: dict[str, Any] = {}
    for name, keys in surfaces.items():
        atom_cache = {case["case_key"]: atoms_for(case, keys) for case in cases}
        mined = mine_clauses(
            cases,
            atom_cache,
            max_size=int(args.max_clause_size),
            max_atoms=int(args.max_atoms),
        )
        mined["stable_negative_guard_candidates"] = stable_negative_candidates(
            cases,
            atom_cache,
            mined["negative_guard_candidates"],
        )
        surface_results[name] = mined

    output = {
        "schema": "ecdlp_public_repeated_coordinate_pre_event_proxy_miner_v1",
        "method": "candidate_position_proxy_mining_without_position_labels",
        "parameters": {
            "audit": str(args.audit),
            "max_clause_size": int(args.max_clause_size),
            "max_atoms": int(args.max_atoms),
        },
        "summary": summarize_cases(cases),
        "surfaces": surface_results,
        "non_claims": [
            "Mined proxy guards are diagnostic and are not a completed ECDLP speedup.",
            "Candidate-position labels and exact coordinates/salts are excluded from the mined surfaces.",
            "Any proxy must be frozen before being replayed on a future transfer window.",
        ],
    }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--max-clause-size", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=128)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    out_path = resolve_path(Path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    public_surface = output["surfaces"]["public_pair_no_position_no_exact"]
    print(
        json.dumps(
            {
                "summary": output["summary"],
                "best_public_negative_guards": public_surface[
                    "stable_negative_guard_candidates"
                ][:8],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
