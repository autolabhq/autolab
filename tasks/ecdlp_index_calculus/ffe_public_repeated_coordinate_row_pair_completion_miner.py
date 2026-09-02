#!/usr/bin/env python3
"""Mine public row rules for repeated-coordinate rank completion.

The root-positive row miner found public rules that can identify individual
rows with selected roots, but held-out rows often carried only one relation.
This miner scores the same public row predicates at the coordinate-case level:
a rule is useful only when the selected rows in a case carry at least two
relation events.  Public-key verification is deliberately left to the replay
probe; this script only proposes row rules that may complete a rank/relation
system without selecting one-row decoys.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_activation_rule_miner as activation_miner
import ffe_public_repeated_coordinate_root_positive_row_miner as row_miner


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_row_pair_completion_miner.json"


def parse_window(raw: str) -> tuple[str, Path]:
    return row_miner.parse_window(raw)


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def case_key(record: dict[str, Any]) -> tuple[Any, ...]:
    key = record.get("case_key")
    if key:
        return (record.get("window"), key)
    return (
        record.get("window"),
        record.get("target"),
        record.get("transfer_index"),
        record.get("top_k"),
        record.get("policy"),
        record.get("leaf_selector"),
        record.get("coordinate_key"),
    )


def group_cases(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(case_key(record), []).append(record)
    return grouped


def relation_events(records: list[dict[str, Any]]) -> int:
    return sum(int(record.get("relation_event_count") or 0) for record in records)


def row_ops_sum(records: list[dict[str, Any]]) -> float:
    return sum(
        float(record.get("row_ops_over_rho") or 0.0)
        for record in records
        if record.get("row_ops_over_rho") is not None
    )


def selected_records(rule: tuple[tuple[str, ...], ...], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if row_miner.dnf_matches(rule, record)]


def full_case_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = group_cases(records)
    relation_complete = [
        rows for rows in grouped.values() if relation_events(rows) >= 2
    ]
    one_relation = [
        rows for rows in grouped.values() if relation_events(rows) == 1
    ]
    return {
        "row_count": len(records),
        "case_count": len(grouped),
        "relation_complete_case_count": len(relation_complete),
        "one_relation_case_count": len(one_relation),
        "root_positive_row_count": sum(1 for record in records if record.get("label_root_positive")),
        "relation_positive_row_count": sum(1 for record in records if record.get("label_relation_positive")),
    }


def score_rule_split(
    rule: tuple[tuple[str, ...], ...],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped = group_cases(records)
    selected_case_count = 0
    selected_complete_count = 0
    selected_incomplete_count = 0
    selected_one_relation_count = 0
    true_complete_count = 0
    false_negative_count = 0
    selected_row_count = 0
    selected_ops: list[float] = []
    selected_complete_ops: list[float] = []
    examples: list[dict[str, Any]] = []
    incomplete_examples: list[dict[str, Any]] = []

    for rows in grouped.values():
        selected = selected_records(rule, rows)
        available_events = relation_events(rows)
        available_complete = available_events >= 2
        selected_events = relation_events(selected)
        selected_complete = selected_events >= 2
        if selected:
            selected_case_count += 1
            selected_row_count += len(selected)
            ops_sum = row_ops_sum(selected)
            selected_ops.append(ops_sum)
            if selected_complete:
                selected_complete_count += 1
                selected_complete_ops.append(ops_sum)
            else:
                selected_incomplete_count += 1
                if selected_events == 1:
                    selected_one_relation_count += 1
            example = compact_case(rows, selected)
            if selected_complete and len(examples) < 16:
                examples.append(example)
            if selected and not selected_complete and len(incomplete_examples) < 16:
                incomplete_examples.append(example)
        if available_complete and selected_complete:
            true_complete_count += 1
        if available_complete and not selected_complete:
            false_negative_count += 1

    available_complete_count = sum(1 for rows in grouped.values() if relation_events(rows) >= 2)
    return {
        "case_count": len(grouped),
        "available_complete_case_count": available_complete_count,
        "selected_case_count": selected_case_count,
        "selected_row_count": selected_row_count,
        "selected_complete_count": selected_complete_count,
        "selected_incomplete_count": selected_incomplete_count,
        "selected_one_relation_count": selected_one_relation_count,
        "true_complete_count": true_complete_count,
        "false_negative_count": false_negative_count,
        "completion_precision": (
            round(true_complete_count / selected_complete_count, 8)
            if selected_complete_count
            else None
        ),
        "completion_recall": (
            round(true_complete_count / available_complete_count, 8)
            if available_complete_count
            else None
        ),
        "selected_completion_yield": (
            round(selected_complete_count / selected_case_count, 8)
            if selected_case_count
            else None
        ),
        "mean_selected_ops_over_rho": mean_or_none(selected_ops),
        "mean_complete_ops_over_rho": mean_or_none(selected_complete_ops),
        "complete_examples": examples,
        "incomplete_examples": incomplete_examples,
    }


def compact_case(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    return {
        "window": first.get("window"),
        "target": first.get("target"),
        "transfer_index": first.get("transfer_index"),
        "top_k": first.get("top_k"),
        "policy_family": first.get("policy_family"),
        "leaf_selector": first.get("leaf_selector"),
        "coordinate": {"b": first.get("b"), "c": first.get("c")},
        "available_relation_events": relation_events(rows),
        "selected_relation_events": relation_events(selected),
        "selected_row_count": len(selected),
        "selected_ops_over_rho_sum": round_or_none(row_ops_sum(selected)),
        "selected_rows": [
            {
                "salt": record.get("salt"),
                "leaf_index": record.get("leaf_index"),
                "relation_event_count": record.get("relation_event_count"),
                "selected_hit_roots": record.get("selected_hit_roots"),
                "row_ops_over_rho": record.get("row_ops_over_rho"),
            }
            for record in selected[:8]
        ],
    }


def score_rule(
    rule: tuple[tuple[str, ...], ...],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "train": score_rule_split(rule, train_records),
        "validation": score_rule_split(rule, validation_records),
    }


def clause_text(clause: tuple[str, ...]) -> str:
    return "&".join(clause)


def rule_text(rule: tuple[tuple[str, ...], ...]) -> str:
    return row_miner.rule_text(rule)


def rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
    train = item["score"]["train"]
    validation = item["score"]["validation"]
    val_yield = validation["selected_completion_yield"]
    train_yield = train["selected_completion_yield"]
    return (
        validation["selected_incomplete_count"],
        train["selected_incomplete_count"],
        -validation["true_complete_count"],
        -train["true_complete_count"],
        -(val_yield if val_yield is not None else -1.0),
        -(train_yield if train_yield is not None else -1.0),
        train["false_negative_count"],
        item["clause_count"],
        item["atom_count"],
        item["row_rule"],
    )


def enumerate_clauses(
    atoms: list[str],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    max_clause_size: int,
) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for size in range(1, max_clause_size + 1):
        for clause in itertools.combinations(atoms, size):
            train_selected = [
                record for record in train_records if row_miner.clause_matches(clause, record)
            ]
            if not train_selected:
                continue
            rule = (clause,)
            score = score_rule(rule, train_records, validation_records)
            if score["train"]["true_complete_count"] == 0 and score["validation"]["true_complete_count"] == 0:
                continue
            clauses.append(
                {
                    "rule": rule,
                    "row_rule": rule_text(rule),
                    "score": score,
                    "clause_count": 1,
                    "atom_count": len(clause),
                }
            )
    return sorted(clauses, key=rank_key)


def combine_clauses(
    clauses: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    max_clauses: int,
    beam_size: int,
) -> list[dict[str, Any]]:
    beam = clauses[:beam_size]
    all_rules = list(beam)
    for _size in range(2, max_clauses + 1):
        candidates: dict[str, dict[str, Any]] = {}
        for base in beam:
            base_rule = base["rule"]
            base_texts = {clause_text(clause) for clause in base_rule}
            for add in clauses[:beam_size]:
                add_clause = add["rule"][0]
                if clause_text(add_clause) in base_texts:
                    continue
                ordered = tuple(sorted((*base_rule, add_clause), key=clause_text))
                text = rule_text(ordered)
                if text in candidates:
                    continue
                score = score_rule(ordered, train_records, validation_records)
                if score["train"]["true_complete_count"] == 0 and score["validation"]["true_complete_count"] == 0:
                    continue
                candidates[text] = {
                    "rule": ordered,
                    "row_rule": text,
                    "score": score,
                    "clause_count": len(ordered),
                    "atom_count": sum(len(clause) for clause in ordered),
                }
        beam = sorted(candidates.values(), key=rank_key)[:beam_size]
        all_rules.extend(beam)
        if not beam:
            break
    unique: dict[str, dict[str, Any]] = {}
    for item in all_rules:
        unique[item["row_rule"]] = item
    return sorted(unique.values(), key=rank_key)


def window_split(records: list[dict[str, Any]], validation_names: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_names:
        return (
            [record for record in records if str(record.get("window")) not in validation_names],
            [record for record in records if str(record.get("window")) in validation_names],
        )
    return records, []


def strip_examples(item: dict[str, Any], sample_limit: int) -> dict[str, Any]:
    def shrink(split: dict[str, Any]) -> dict[str, Any]:
        out = dict(split)
        out["complete_examples"] = out.get("complete_examples", [])[:sample_limit]
        out["incomplete_examples"] = out.get("incomplete_examples", [])[:sample_limit]
        return out

    return {
        "row_rule": item["row_rule"],
        "clause_count": item["clause_count"],
        "atom_count": item["atom_count"],
        "score": {
            "train": shrink(item["score"]["train"]),
            "validation": shrink(item["score"]["validation"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=parse_window, action="append", required=True)
    parser.add_argument("--validation-name", action="append", default=[])
    parser.add_argument("--activation-rule", type=activation_miner.parse_activation_rule)
    parser.add_argument("--activated-only", action="store_true")
    parser.add_argument("--allow-exact-coordinate", action="store_true")
    parser.add_argument("--max-candidates-per-window", type=int, default=0)
    parser.add_argument("--max-clause-size", type=int, default=2)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=96)
    parser.add_argument("--top-rules", type=int, default=12)
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--event-summary-limit", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records, diagnostics = row_miner.load_row_records(
        args.window,
        args.activation_rule,
        bool(args.activated_only),
        int(args.max_candidates_per_window),
        int(args.event_summary_limit),
    )
    train_records, validation_records = window_split(records, set(args.validation_name or []))
    atoms = row_miner.atoms_for_records(records, bool(args.allow_exact_coordinate))
    clauses = enumerate_clauses(
        atoms,
        train_records,
        validation_records,
        int(args.max_clause_size),
    )
    rules = combine_clauses(
        clauses,
        train_records,
        validation_records,
        int(args.max_clauses),
        int(args.beam_size),
    )
    top_rules = [strip_examples(rule, int(args.sample_limit)) for rule in rules[: int(args.top_rules)]]
    best = top_rules[0] if top_rules else None
    output = {
        "schema": "ecdlp_public_repeated_coordinate_row_pair_completion_miner_v1",
        "method": "public_feature_rule_mining_for_relation_complete_coordinate_cases",
        "parameters": {
            "windows": [{"name": name, "artifact": str(path)} for name, path in args.window],
            "validation_names": list(args.validation_name or []),
            "activation_rule": (
                activation_miner.rule_text(args.activation_rule) if args.activation_rule else None
            ),
            "activated_only": bool(args.activated_only),
            "allow_exact_coordinate": bool(args.allow_exact_coordinate),
            "max_candidates_per_window": int(args.max_candidates_per_window),
            "max_clause_size": int(args.max_clause_size),
            "max_clauses": int(args.max_clauses),
            "beam_size": int(args.beam_size),
            "event_summary_limit": int(args.event_summary_limit),
        },
        "diagnostics": diagnostics,
        "record_summary": full_case_summary(records),
        "train_summary": full_case_summary(train_records),
        "validation_summary": full_case_summary(validation_records),
        "candidate_atom_count": len(atoms),
        "candidate_clause_count": len(clauses),
        "frozen_row_rule": (best or {}).get("row_rule"),
        "best_rule": best,
        "top_rules": top_rules,
        "non_claims": [
            "Relation-event completion is still a mining label, not a public-key proof.",
            "Candidate row rules must be replayed with ffe_public_repeated_coordinate_row_rule_replay_probe.py before any speedup claim.",
            "A rule with no held-out relation-complete cases is only an in-sample selector.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "frozen_row_rule": output["frozen_row_rule"],
                "record_summary": output["record_summary"],
                "train_summary": output["train_summary"],
                "validation_summary": output["validation_summary"],
                "best_rule_score": (best or {}).get("score"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
