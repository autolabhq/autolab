#!/usr/bin/env python3
"""Mine public activation rules for repeated-coordinate gates.

The repeated-coordinate miner turns public selected leaves into monic ``(b,c)``
gate candidates.  This script searches small public predicates over those
candidates and ranks rules by replay labels.  Labels are used only for scoring;
the predicates themselves are restricted to public metadata available before
exact Sage factorization and before verifier-derived labels are inspected.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_activation_rule_miner.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_window(raw: str) -> tuple[str, Path]:
    parts = raw.split("|", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("window must be name|artifact_path")
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


def selector_total(leaf_selector: str) -> int | None:
    if "total3" in leaf_selector:
        return 3
    if "total4" in leaf_selector:
        return 4
    return None


def policy_family(policy: str) -> str:
    if "target_cap3" in policy:
        return "target_cap3"
    if "global_cap3" in policy:
        return "global_cap3"
    if "target_cap1" in policy:
        return "target_cap1"
    return policy or "unknown"


def replay_label(record: dict[str, Any]) -> dict[str, Any]:
    replay = record.get("exact_coordinate_replay") or {}
    axis_b = record.get("b_axis_replay") or {}
    axis_c = record.get("c_axis_replay") or {}
    verified = bool(replay.get("public_key_verified"))
    return {
        "label_verified": verified,
        "label_below_rho": verified and bool(replay.get("below_rho")),
        "label_ops_over_rho": replay.get("ops_over_rho"),
        "label_rank": replay.get("rank"),
        "label_relation_count": replay.get("relation_count"),
        "label_derived_secret": replay.get("derived_secret"),
        "label_b_axis_verified": bool(axis_b.get("public_key_verified")),
        "label_c_axis_verified": bool(axis_c.get("public_key_verified")),
    }


def feature_record(record: dict[str, Any], window_name: str, source: str) -> dict[str, Any]:
    b = int(record.get("b") if record.get("b") is not None else (record.get("coordinate") or {}).get("b") or 0)
    c = int(record.get("c") if record.get("c") is not None else (record.get("coordinate") or {}).get("c") or 0)
    p = int(record.get("p") or 9803)
    salts = sorted(int(salt) for salt in record.get("salts") or [])
    leaf_indices = sorted(int(leaf) for leaf in record.get("leaf_indices") or [])
    transfer_index = int(record.get("transfer_index") or 0)
    top_k = int(record.get("top_k") or 0)
    source_ops = round_or_none(record.get("source_ops_over_rho"))
    source_ops_millirhos = int(round(float(source_ops) * 1000)) if source_ops is not None else -1
    salt_min = min(salts) if salts else -1
    salt_max = max(salts) if salts else -1
    salt_span = salt_max - salt_min if salts else -1
    leaf_min = min(leaf_indices) if leaf_indices else -1
    leaf_max = max(leaf_indices) if leaf_indices else -1
    policy = str(record.get("policy") or "")
    leaf_selector = str(record.get("leaf_selector") or "")
    features = {
        "window": window_name,
        "source": source,
        "case_key": record.get("case_key"),
        "target": str(record.get("target") or ""),
        "transfer_index": transfer_index,
        "top_k": top_k,
        "policy": policy,
        "policy_family": policy_family(policy),
        "leaf_selector": leaf_selector,
        "leaf_total": selector_total(leaf_selector) or -1,
        "leaf_selector_costed": int(leaf_selector.startswith("mode_cost_")),
        "p": p,
        "b": b,
        "c": c,
        "b_mod2": b % 2,
        "b_mod3": b % 3,
        "b_mod4": b % 4,
        "b_mod5": b % 5,
        "b_mod8": b % 8,
        "b_mod16": b % 16,
        "c_mod2": c % 2,
        "c_mod3": c % 3,
        "c_mod4": c % 4,
        "c_mod5": c % 5,
        "c_mod8": c % 8,
        "c_mod16": c % 16,
        "b_plus_c_mod8": (b + c) % 8,
        "b_plus_c_mod16": (b + c) % 16,
        "b_minus_c_mod8": (b - c) % 8,
        "b_minus_c_mod16": (b - c) % 16,
        "coordinate_key": f"{b},{c}",
        "row_count": int(record.get("row_count") or 0),
        "salt_count": int(record.get("salt_count") or 0),
        "profile_count": int(record.get("profile_count") or 0),
        "leaf_indices": leaf_indices,
        "leaf_signature": ",".join(str(leaf) for leaf in leaf_indices),
        "leaf_min": leaf_min,
        "leaf_max": leaf_max,
        "salts": salts,
        "salt_signature": ",".join(str(salt) for salt in salts),
        "salt_min": salt_min,
        "salt_max": salt_max,
        "salt_span": salt_span,
        "salt_sum_mod4": sum(salts) % 4 if salts else -1,
        "salt_sum_mod8": sum(salts) % 8 if salts else -1,
        "salt_mod2_pattern": ",".join(str(salt % 2) for salt in salts),
        "salt_mod3_pattern": ",".join(str(salt % 3) for salt in salts),
        "salt_mod4_pattern": ",".join(str(salt % 4) for salt in salts),
        "source_ops_millirhos": source_ops_millirhos,
        "source_below_rho": int(bool(record.get("source_below_rho"))),
    }
    for modulus in (2, 3, 4, 5, 8, 16):
        features[f"transfer_mod{modulus}"] = transfer_index % modulus
        features[f"salt_min_mod{modulus}"] = salt_min % modulus if salt_min >= 0 else -1
        features[f"salt_max_mod{modulus}"] = salt_max % modulus if salt_max >= 0 else -1
        features[f"leaf_min_mod{modulus}"] = leaf_min % modulus if leaf_min >= 0 else -1
    for modulus in (2, 3, 4):
        for residue in range(modulus):
            features[f"salt_mod{modulus}_residue{residue}_count"] = sum(
                1 for salt in salts if salt % modulus == residue
            )
    features.update(replay_label(record))
    return features


def load_records(windows: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, path in windows:
        artifact = load_json(path)
        for record in artifact.get("candidates") or []:
            if not isinstance(record, dict):
                continue
            records.append(feature_record(record, name, str(path)))
    return records


NUMERIC_KEYS = {
    "transfer_index",
    "top_k",
    "leaf_total",
    "leaf_selector_costed",
    "p",
    "b",
    "c",
    "b_mod2",
    "b_mod3",
    "b_mod4",
    "b_mod5",
    "b_mod8",
    "b_mod16",
    "c_mod2",
    "c_mod3",
    "c_mod4",
    "c_mod5",
    "c_mod8",
    "c_mod16",
    "b_plus_c_mod8",
    "b_plus_c_mod16",
    "b_minus_c_mod8",
    "b_minus_c_mod16",
    "row_count",
    "salt_count",
    "profile_count",
    "leaf_min",
    "leaf_max",
    "salt_min",
    "salt_max",
    "salt_span",
    "salt_sum_mod4",
    "salt_sum_mod8",
    "source_ops_millirhos",
    "source_below_rho",
}
for _modulus in (2, 3, 4, 5, 8, 16):
    NUMERIC_KEYS.update(
        {
            f"transfer_mod{_modulus}",
            f"salt_min_mod{_modulus}",
            f"salt_max_mod{_modulus}",
            f"leaf_min_mod{_modulus}",
        }
    )
for _modulus in (2, 3, 4):
    for _residue in range(_modulus):
        NUMERIC_KEYS.add(f"salt_mod{_modulus}_residue{_residue}_count")

STRING_KEYS = {
    "target",
    "policy",
    "policy_family",
    "leaf_selector",
    "coordinate_key",
    "leaf_signature",
    "salt_signature",
    "salt_mod2_pattern",
    "salt_mod3_pattern",
    "salt_mod4_pattern",
}


def atoms_for_records(records: list[dict[str, Any]], allow_exact_coordinate: bool) -> list[str]:
    atoms: set[str] = set()
    for record in records:
        for key in sorted(STRING_KEYS):
            if key == "coordinate_key" and not allow_exact_coordinate:
                continue
            value = record.get(key)
            if value not in {None, ""}:
                atoms.add(f"{key}={value}")
        for key in sorted(NUMERIC_KEYS):
            value = int(record[key])
            atoms.add(f"{key}={value}")
            if key in {
                "source_ops_millirhos",
                "salt_min",
                "salt_max",
                "salt_span",
                "leaf_min",
                "leaf_max",
                "top_k",
                "row_count",
                "salt_count",
                "profile_count",
            }:
                atoms.add(f"{key}>={value}")
                atoms.add(f"{key}<={value}")
    return sorted(atoms)


def atom_matches(atom: str, record: dict[str, Any]) -> bool:
    if ">=" in atom:
        key, value = atom.split(">=", 1)
        return int(record[key]) >= int(value)
    if "<=" in atom:
        key, value = atom.split("<=", 1)
        return int(record[key]) <= int(value)
    key, value = atom.split("=", 1)
    if key in NUMERIC_KEYS:
        return int(record[key]) == int(value)
    return str(record.get(key)) == value


def clause_matches(clause: tuple[str, ...], record: dict[str, Any]) -> bool:
    return all(atom_matches(atom, record) for atom in clause)


def dnf_matches(rule: tuple[tuple[str, ...], ...], record: dict[str, Any]) -> bool:
    return any(clause_matches(clause, record) for clause in rule)


def selected_records(rule: tuple[tuple[str, ...], ...], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if dnf_matches(rule, record)]


def score_selection(selected: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [record for record in records if record["label_verified"]]
    true_pos = [record for record in selected if record["label_verified"]]
    false_pos = [record for record in selected if not record["label_verified"]]
    false_neg = len(positives) - len(true_pos)
    verified_ops = [
        float(record["label_ops_over_rho"])
        for record in true_pos
        if record.get("label_ops_over_rho") is not None
    ]
    return {
        "selected_count": len(selected),
        "true_positive_count": len(true_pos),
        "false_positive_count": len(false_pos),
        "false_negative_count": false_neg,
        "positive_count": len(positives),
        "precision": round(len(true_pos) / len(selected), 8) if selected else None,
        "recall": round(len(true_pos) / len(positives), 8) if positives else None,
        "min_verified_ops_over_rho": round(min(verified_ops), 8) if verified_ops else None,
        "mean_verified_ops_over_rho": mean_or_none(verified_ops),
    }


def score_rule(
    rule: tuple[tuple[str, ...], ...],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_selected = selected_records(rule, train_records)
    validation_selected = selected_records(rule, validation_records)
    train_score = score_selection(train_selected, train_records)
    validation_score = score_selection(validation_selected, validation_records)
    return {
        "train": train_score,
        "validation": validation_score,
    }


def clause_text(clause: tuple[str, ...]) -> str:
    return "&".join(clause)


def rule_text(rule: tuple[tuple[str, ...], ...]) -> str:
    return "activate:" + "|".join(clause_text(clause) for clause in rule)


def parse_activation_rule(raw: str) -> tuple[tuple[str, ...], ...]:
    if not raw.startswith("activate:"):
        raise argparse.ArgumentTypeError("activation rule must start with activate:")
    expression = raw.split(":", 1)[1].strip()
    if not expression:
        raise argparse.ArgumentTypeError("activation rule has no clauses")
    clauses = []
    for clause_text_value in expression.split("|"):
        atoms = tuple(atom.strip() for atom in clause_text_value.split("&") if atom.strip())
        if not atoms:
            raise argparse.ArgumentTypeError("activation rule contains an empty clause")
        clauses.append(atoms)
    return tuple(clauses)


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "window": record.get("window"),
        "target": record.get("target"),
        "transfer_index": record.get("transfer_index"),
        "top_k": record.get("top_k"),
        "policy": record.get("policy"),
        "leaf_selector": record.get("leaf_selector"),
        "coordinate": {"b": record.get("b"), "c": record.get("c")},
        "leaf_signature": record.get("leaf_signature"),
        "salt_signature": record.get("salt_signature"),
        "source_ops_millirhos": record.get("source_ops_millirhos"),
        "label_verified": record.get("label_verified"),
        "label_ops_over_rho": record.get("label_ops_over_rho"),
        "label_derived_secret": record.get("label_derived_secret"),
    }


def rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
    train = item["score"]["train"]
    validation = item["score"]["validation"]
    return (
        validation["false_positive_count"],
        train["false_positive_count"],
        -validation["true_positive_count"],
        -train["true_positive_count"],
        validation["false_negative_count"],
        train["false_negative_count"],
        item["clause_count"],
        item["atom_count"],
        item["activation_rule"],
    )


def enumerate_clauses(
    atoms: list[str],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    max_clause_size: int,
) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for size in range(1, int(max_clause_size) + 1):
        for clause in itertools.combinations(atoms, size):
            selected = [record for record in train_records if clause_matches(clause, record)]
            if not selected:
                continue
            rule = (clause,)
            score = score_rule(rule, train_records, validation_records)
            if score["train"]["true_positive_count"] == 0 and score["validation"]["true_positive_count"] == 0:
                continue
            clauses.append(
                {
                    "rule": rule,
                    "score": score,
                    "activation_rule": rule_text(rule),
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
    for _size in range(2, int(max_clauses) + 1):
        candidates: dict[str, dict[str, Any]] = {}
        for base in beam:
            base_rule = base["rule"]
            base_texts = {clause_text(clause) for clause in base_rule}
            for add in clauses[:beam_size]:
                add_clause = add["rule"][0]
                if clause_text(add_clause) in base_texts:
                    continue
                ordered = tuple(
                    sorted(
                        (*base_rule, add_clause),
                        key=clause_text,
                    )
                )
                activation_rule = rule_text(ordered)
                if activation_rule in candidates:
                    continue
                score = score_rule(ordered, train_records, validation_records)
                if score["train"]["true_positive_count"] == 0 and score["validation"]["true_positive_count"] == 0:
                    continue
                candidates[activation_rule] = {
                    "rule": ordered,
                    "score": score,
                    "activation_rule": activation_rule,
                    "clause_count": len(ordered),
                    "atom_count": sum(len(clause) for clause in ordered),
                }
        next_rules = sorted(candidates.values(), key=rank_key)[:beam_size]
        all_rules.extend(next_rules)
        beam = next_rules
        if not beam:
            break
    unique: dict[str, dict[str, Any]] = {}
    for item in all_rules:
        unique[item["activation_rule"]] = item
    return sorted(unique.values(), key=rank_key)


def enrich_rule(
    item: dict[str, Any],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    sample_limit: int,
) -> dict[str, Any]:
    rule = item["rule"]
    train_selected = selected_records(rule, train_records)
    validation_selected = selected_records(rule, validation_records)
    return {
        "activation_rule": item["activation_rule"],
        "clause_count": item["clause_count"],
        "atom_count": item["atom_count"],
        "score": item["score"],
        "train_selected_cases": [compact_record(record) for record in train_selected[:sample_limit]],
        "validation_selected_cases": [
            compact_record(record) for record in validation_selected[:sample_limit]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-window", type=parse_window, action="append", required=True)
    parser.add_argument("--validation-window", type=parse_window, action="append", default=[])
    parser.add_argument("--max-clause-size", type=int, default=2)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=128)
    parser.add_argument("--top-rules", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--allow-exact-coordinate", action="store_true")
    parser.add_argument("--activation-rule", type=parse_activation_rule)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    train_records = load_records(args.training_window)
    validation_records = load_records(args.validation_window)
    atoms = atoms_for_records(train_records + validation_records, bool(args.allow_exact_coordinate))
    if args.activation_rule:
        fixed_rule = args.activation_rule
        fixed_score = score_rule(fixed_rule, train_records, validation_records)
        clauses = []
        rules = [
            {
                "rule": fixed_rule,
                "score": fixed_score,
                "activation_rule": rule_text(fixed_rule),
                "clause_count": len(fixed_rule),
                "atom_count": sum(len(clause) for clause in fixed_rule),
            }
        ]
    else:
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
    enriched = [
        enrich_rule(rule, train_records, validation_records, int(args.sample_limit))
        for rule in rules[: int(args.top_rules)]
    ]
    best = enriched[0] if enriched else None
    output = {
        "schema": "ecdlp_public_repeated_coordinate_activation_rule_miner_v1",
        "method": "label_ranked_public_coordinate_gate_activation_search",
        "parameters": {
            "training_windows": [
                {"name": name, "artifact": str(path)} for name, path in args.training_window
            ],
            "validation_windows": [
                {"name": name, "artifact": str(path)} for name, path in args.validation_window
            ],
            "max_clause_size": int(args.max_clause_size),
            "max_clauses": int(args.max_clauses),
            "beam_size": int(args.beam_size),
            "allow_exact_coordinate": bool(args.allow_exact_coordinate),
            "fixed_activation_rule": rule_text(args.activation_rule) if args.activation_rule else None,
        },
        "training_case_count": len(train_records),
        "training_positive_count": sum(1 for record in train_records if record["label_verified"]),
        "validation_case_count": len(validation_records),
        "validation_positive_count": sum(1 for record in validation_records if record["label_verified"]),
        "candidate_atom_count": len(atoms),
        "candidate_clause_count": len(clauses),
        "frozen_activation_rule": (best or {}).get("activation_rule"),
        "best_rule": best,
        "top_rules": enriched,
        "non_claims": [
            "Rules are ranked with replay labels but only use public repeated-coordinate metadata.",
            "A rule that verifies over rho is still a mechanism selector, not an end-to-end ECDLP speedup.",
            "Exact coordinate atoms are disabled unless --allow-exact-coordinate is set.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "frozen_activation_rule": output["frozen_activation_rule"],
                "best_rule_score": (best or {}).get("score"),
                "candidate_clause_count": len(clauses),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
