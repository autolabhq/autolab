#!/usr/bin/env python3
"""Mine public rules for repeated-coordinate relation-row subsets.

Cost decompositions show that several target-67 repeated-coordinate recoveries
are over rho only because they carry one or more dead rows.  This probe treats
the verifier-informed best below-rho subset as a label, then searches small
public predicates over row-pair metadata.  A mined rule is only a candidate:
it must be frozen and replayed on a later window before it can support a
speedup claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_root_positive_row_miner as row_miner


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_subset_pair_rule_miner.json"


STRING_KEYS = {
    "target",
    "policy_family",
    "leaf_selector",
    "leaf_signature",
    "pair_leaf_signature",
    "pair_salt_index_signature",
    "pair_salt_delta_from_min_signature",
    "pair_salt_delta_to_max_signature",
}

OPTIONAL_EXACT_KEYS = {
    "coordinate_key",
    "pair_salt_signature",
}

NUMERIC_KEYS = {
    "subset_size",
    "transfer_index",
    "top_k",
    "leaf_total",
    "b_mod5",
    "b_mod16",
    "c_mod5",
    "c_mod16",
    "b_minus_c_mod16",
    "row_count",
    "salt_count",
    "salt_span",
    "pair_salt_span",
    "pair_salt_min_mod5",
    "pair_salt_max_mod5",
    "pair_salt_sum_mod5",
    "pair_salt_sum_mod8",
    "pair_salt_index_span",
    "leaf_min",
    "leaf_min_mod8",
    "source_ops_millirhos",
    "full_ops_millirhos",
    "full_relation_count",
    "full_rank",
}

for _modulus in (2, 3, 4, 5, 8, 16):
    NUMERIC_KEYS.add(f"transfer_mod{_modulus}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def int_or_default(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clause_text(clause: tuple[str, ...]) -> str:
    return "&".join(clause)


def rule_text(rule: tuple[tuple[str, ...], ...]) -> str:
    return "pair_activate:" + "|".join(clause_text(clause) for clause in rule)


def parse_pair_rule(raw: str) -> tuple[tuple[str, ...], ...]:
    if not raw.startswith("pair_activate:"):
        raise argparse.ArgumentTypeError("pair rule must start with pair_activate:")
    expression = raw.split(":", 1)[1].strip()
    if not expression:
        raise argparse.ArgumentTypeError("pair rule has no clauses")
    clauses = []
    for clause_value in expression.split("|"):
        atoms = tuple(atom.strip() for atom in clause_value.split("&") if atom.strip())
        if not atoms:
            raise argparse.ArgumentTypeError("pair rule contains an empty clause")
        clauses.append(atoms)
    return tuple(clauses)


def add_mod_features(record: dict[str, Any]) -> None:
    transfer_index = int(record["transfer_index"])
    for modulus in (2, 3, 4, 5, 8, 16):
        record[f"transfer_mod{modulus}"] = transfer_index % modulus


def source_ops_millirhos(value: Any) -> int:
    rounded = round_or_none(value)
    return int(round(float(rounded) * 1000)) if rounded is not None else -1


def leaf_values(profile: dict[str, Any]) -> list[int]:
    raw = profile.get("leaf_indices") or []
    return sorted(int(value) for value in raw)


def row_record_from_profile(
    recovery: dict[str, Any],
    profile: dict[str, Any],
    window_name: str,
) -> dict[str, Any]:
    coordinate = recovery.get("coordinate") or {}
    b_value = int_or_default(coordinate.get("b"), 0)
    c_value = int_or_default(coordinate.get("c"), 0)
    kept_profiles = recovery.get("kept_profiles") or []
    salts = sorted(int_or_default(item.get("salt")) for item in kept_profiles)
    leaves = leaf_values(profile)
    all_leaves = sorted({leaf for item in kept_profiles for leaf in leaf_values(item)})
    salt = int_or_default(profile.get("salt"))
    salt_index = salts.index(salt) if salt in salts else -1
    salt_min = min(salts) if salts else -1
    salt_max = max(salts) if salts else -1
    full_replay = recovery.get("full_replay") or {}
    record = {
        "window": window_name,
        "target": str(recovery.get("target") or ""),
        "transfer_index": int_or_default(recovery.get("transfer_index"), 0),
        "top_k": int_or_default(recovery.get("top_k"), 0),
        "policy": str(recovery.get("policy") or ""),
        "policy_family": row_miner.policy_family(str(recovery.get("policy") or "")),
        "leaf_selector": str(recovery.get("leaf_selector") or ""),
        "leaf_total": row_miner.selector_total(str(recovery.get("leaf_selector") or "")) or -1,
        "b": b_value,
        "c": c_value,
        "b_mod5": b_value % 5,
        "b_mod16": b_value % 16,
        "c_mod5": c_value % 5,
        "c_mod16": c_value % 16,
        "b_minus_c_mod16": (b_value - c_value) % 16,
        "coordinate_key": f"{b_value},{c_value}",
        "row_count": len({str(item.get("row_key") or "") for item in kept_profiles}),
        "salt_count": len(set(salts)),
        "salt": salt,
        "salt_index": salt_index,
        "salt_min": salt_min,
        "salt_max": salt_max,
        "salt_span": salt_max - salt_min if salts else -1,
        "salt_delta_from_min": salt - salt_min if salt >= 0 and salt_min >= 0 else -1,
        "salt_delta_to_max": salt_max - salt if salt >= 0 and salt_max >= 0 else -1,
        "leaf_min": min(all_leaves) if all_leaves else -1,
        "leaf_signature": ",".join(str(value) for value in all_leaves),
        "row_leaf_signature": ",".join(str(value) for value in leaves),
        "row_key": str(profile.get("row_key") or ""),
        "source_ops_millirhos": source_ops_millirhos(recovery.get("source_ops_over_rho")),
        "full_ops_millirhos": source_ops_millirhos(full_replay.get("ops_over_rho")),
        "full_relation_count": int_or_default(full_replay.get("relation_count"), 0),
        "full_rank": int_or_default(full_replay.get("rank"), 0),
    }
    record["leaf_min_mod8"] = int(record["leaf_min"]) % 8 if int(record["leaf_min"]) >= 0 else -1
    add_mod_features(record)
    return record


def pair_record(
    window_name: str,
    decomposition_name: str,
    recovery: dict[str, Any],
    rows: tuple[dict[str, Any], ...],
    positive_row_keys: set[str],
    allow_exact_coordinate: bool,
    allow_exact_salt: bool,
) -> dict[str, Any]:
    salts = sorted(int(row["salt"]) for row in rows)
    salt_indices = sorted(int(row["salt_index"]) for row in rows)
    salt_deltas_from_min = sorted(int(row["salt_delta_from_min"]) for row in rows)
    salt_deltas_to_max = sorted(int(row["salt_delta_to_max"]) for row in rows)
    row_keys = sorted(str(row["row_key"]) for row in rows)
    first = rows[0]
    record = {
        "window": window_name,
        "decomposition": decomposition_name,
        "target": first["target"],
        "transfer_index": first["transfer_index"],
        "top_k": first["top_k"],
        "policy_family": first["policy_family"],
        "leaf_selector": first["leaf_selector"],
        "leaf_total": first["leaf_total"],
        "b_mod5": first["b_mod5"],
        "b_mod16": first["b_mod16"],
        "c_mod5": first["c_mod5"],
        "c_mod16": first["c_mod16"],
        "b_minus_c_mod16": first["b_minus_c_mod16"],
        "row_count": first["row_count"],
        "salt_count": first["salt_count"],
        "salt_span": first["salt_span"],
        "leaf_min": first["leaf_min"],
        "leaf_min_mod8": first["leaf_min_mod8"],
        "leaf_signature": first["leaf_signature"],
        "source_ops_millirhos": first["source_ops_millirhos"],
        "full_ops_millirhos": first["full_ops_millirhos"],
        "full_relation_count": first["full_relation_count"],
        "full_rank": first["full_rank"],
        "subset_size": len(rows),
        "pair_salt_span": max(salts) - min(salts) if salts else -1,
        "pair_salt_min_mod5": min(salts) % 5 if salts else -1,
        "pair_salt_max_mod5": max(salts) % 5 if salts else -1,
        "pair_salt_sum_mod5": sum(salts) % 5 if salts else -1,
        "pair_salt_sum_mod8": sum(salts) % 8 if salts else -1,
        "pair_salt_index_span": max(salt_indices) - min(salt_indices) if salt_indices else -1,
        "pair_salt_index_signature": ",".join(str(value) for value in salt_indices),
        "pair_salt_delta_from_min_signature": ",".join(str(value) for value in salt_deltas_from_min),
        "pair_salt_delta_to_max_signature": ",".join(str(value) for value in salt_deltas_to_max),
        "pair_leaf_signature": "|".join(str(row["row_leaf_signature"]) for row in rows),
        "row_keys": row_keys,
        "label_subset_positive": set(row_keys) == set(positive_row_keys),
        "label_row_overlap_count": len(set(row_keys) & positive_row_keys),
    }
    if allow_exact_coordinate:
        record["coordinate_key"] = first["coordinate_key"]
    if allow_exact_salt:
        record["pair_salt_signature"] = ",".join(str(value) for value in salts)
    add_mod_features(record)
    return record


def best_positive_subset(recovery: dict[str, Any]) -> dict[str, Any] | None:
    subset = recovery.get("best_below_rho_verified_subset")
    if subset and subset.get("public_key_verified") and subset.get("below_rho"):
        return subset
    subset = recovery.get("best_verified_subset")
    if subset and subset.get("public_key_verified") and subset.get("below_rho"):
        return subset
    return None


def records_from_decomposition(
    name: str,
    artifact: dict[str, Any],
    allow_exact_coordinate: bool,
    allow_exact_salt: bool,
    max_subset_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for recovery in artifact.get("recoveries") or []:
        if not isinstance(recovery, dict):
            continue
        positive_subset = best_positive_subset(recovery)
        if not positive_subset:
            continue
        positive_row_keys = set(str(row_key) for row_key in positive_subset.get("row_keys") or [])
        if not positive_row_keys:
            continue
        row_records = [
            row_record_from_profile(recovery, profile, str(recovery.get("source_name") or name))
            for profile in recovery.get("kept_profiles") or []
            if profile.get("row_key")
        ]
        by_key = {row["row_key"]: row for row in row_records}
        row_records = [by_key[key] for key in sorted(by_key)]
        for size in range(1, min(max_subset_size, len(row_records)) + 1):
            for rows in itertools.combinations(row_records, size):
                records.append(
                    pair_record(
                        str(recovery.get("source_name") or name),
                        name,
                        recovery,
                        rows,
                        positive_row_keys,
                        allow_exact_coordinate,
                        allow_exact_salt,
                    )
                )
    return records


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


def rule_matches(rule: tuple[tuple[str, ...], ...], record: dict[str, Any]) -> bool:
    return any(clause_matches(clause, record) for clause in rule)


def selected_records(rule: tuple[tuple[str, ...], ...], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if rule_matches(rule, record)]


def score_selection(selected: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [record for record in records if record.get("label_subset_positive")]
    true_pos = [record for record in selected if record.get("label_subset_positive")]
    false_pos = [record for record in selected if not record.get("label_subset_positive")]
    false_neg = [record for record in positives if record not in true_pos]
    selected_ops = [float(record["source_ops_millirhos"]) / 1000.0 for record in true_pos]
    return {
        "row_count": len(records),
        "positive_count": len(positives),
        "selected_count": len(selected),
        "true_positive_count": len(true_pos),
        "false_positive_count": len(false_pos),
        "false_negative_count": len(false_neg),
        "precision": round(len(true_pos) / len(selected), 8) if selected else None,
        "recall": round(len(true_pos) / len(positives), 8) if positives else None,
        "mean_true_positive_source_ops_over_rho": mean_or_none(selected_ops),
    }


def score_rule(
    rule: tuple[tuple[str, ...], ...],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "train": score_selection(selected_records(rule, train_records), train_records),
        "validation": score_selection(selected_records(rule, validation_records), validation_records),
    }


def atoms_for_records(records: list[dict[str, Any]]) -> list[str]:
    atoms: set[str] = set()
    for record in records:
        for key in sorted(STRING_KEYS | OPTIONAL_EXACT_KEYS):
            if key not in record:
                continue
            value = record.get(key)
            if value not in {None, ""}:
                atoms.add(f"{key}={value}")
        for key in sorted(NUMERIC_KEYS):
            if key not in record:
                continue
            value = int(record[key])
            atoms.add(f"{key}={value}")
            if key in {
                "top_k",
                "source_ops_millirhos",
                "full_ops_millirhos",
                "pair_salt_span",
                "pair_salt_index_span",
            }:
                atoms.add(f"{key}>={value}")
                atoms.add(f"{key}<={value}")
    return sorted(atoms)


def rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
    train = item["score"]["train"]
    validation = item["score"]["validation"]
    validation_precision = validation["precision"] if validation["precision"] is not None else -1.0
    train_precision = train["precision"] if train["precision"] is not None else -1.0
    validation_recall = validation["recall"] if validation["recall"] is not None else 0.0
    train_recall = train["recall"] if train["recall"] is not None else 0.0
    return (
        validation["false_positive_count"],
        train["false_positive_count"],
        -validation_precision,
        -train_precision,
        -validation_recall,
        -train_recall,
        item["atom_count"],
        item["clause_count"],
        item["pair_rule"],
    )


def enumerate_clauses(
    atoms: list[str],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    max_clause_size: int,
) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for size in range(1, max_clause_size + 1):
        for combo in itertools.combinations(atoms, size):
            rule = (tuple(combo),)
            score = score_rule(rule, train_records, validation_records)
            if score["train"]["selected_count"] == 0 and score["validation"]["selected_count"] == 0:
                continue
            clauses.append(
                {
                    "rule": rule,
                    "pair_rule": rule_text(rule),
                    "score": score,
                    "clause_count": 1,
                    "atom_count": len(combo),
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
    best: dict[str, dict[str, Any]] = {item["pair_rule"]: item for item in beam}
    for _depth in range(2, max_clauses + 1):
        next_items: list[dict[str, Any]] = []
        for item in beam:
            existing = set(item["rule"])
            for clause_item in clauses[: beam_size * 2]:
                clause = clause_item["rule"][0]
                if clause in existing:
                    continue
                rule = tuple(sorted((*item["rule"], clause)))
                text = rule_text(rule)
                if text in best:
                    continue
                score = score_rule(rule, train_records, validation_records)
                candidate = {
                    "rule": rule,
                    "pair_rule": text,
                    "score": score,
                    "clause_count": len(rule),
                    "atom_count": sum(len(clause_value) for clause_value in rule),
                }
                best[text] = candidate
                next_items.append(candidate)
        beam = sorted(next_items, key=rank_key)[:beam_size]
        if not beam:
            break
    return sorted(best.values(), key=rank_key)


def window_split(records: list[dict[str, Any]], validation_names: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_names:
        return (
            [record for record in records if str(record.get("window")) not in validation_names],
            [record for record in records if str(record.get("window")) in validation_names],
        )
    return records, []


def summary_for_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [record for record in records if record.get("label_subset_positive")]
    return {
        "pair_record_count": len(records),
        "positive_pair_count": len(positives),
        "positive_rate": round(len(positives) / len(records), 8) if records else None,
        "windows": sorted({str(record.get("window") or "") for record in records}),
    }


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "window",
        "target",
        "transfer_index",
        "top_k",
        "policy_family",
        "leaf_selector",
        "subset_size",
        "pair_salt_signature",
        "pair_salt_index_signature",
        "pair_salt_delta_from_min_signature",
        "pair_salt_delta_to_max_signature",
        "source_ops_millirhos",
        "full_ops_millirhos",
        "full_relation_count",
        "full_rank",
        "label_subset_positive",
        "row_keys",
    ]
    return {key: record.get(key) for key in keys if key in record}


def enrich_rule(
    rule: dict[str, Any],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    sample_limit: int,
) -> dict[str, Any]:
    enriched = dict(rule)
    enriched.pop("rule", None)
    enriched["train_selected_examples"] = [
        compact_record(record) for record in selected_records(rule["rule"], train_records)[:sample_limit]
    ]
    enriched["validation_selected_examples"] = [
        compact_record(record) for record in selected_records(rule["rule"], validation_records)[:sample_limit]
    ]
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=parse_named_path, action="append", required=True)
    parser.add_argument("--validation-name", action="append", default=[])
    parser.add_argument("--allow-exact-coordinate", action="store_true")
    parser.add_argument("--allow-exact-salt", action="store_true")
    parser.add_argument("--pair-rule", type=parse_pair_rule)
    parser.add_argument("--max-subset-size", type=int, default=2)
    parser.add_argument("--max-clause-size", type=int, default=2)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=128)
    parser.add_argument("--top-rules", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    loaded_decompositions = []
    for name, path in args.decomposition:
        artifact = load_json(path)
        loaded_decompositions.append({"name": name, "artifact": str(path)})
        records.extend(
            records_from_decomposition(
                name,
                artifact,
                bool(args.allow_exact_coordinate),
                bool(args.allow_exact_salt),
                int(args.max_subset_size),
            )
        )

    train_records, validation_records = window_split(records, set(args.validation_name or []))
    if args.pair_rule:
        fixed_rule = args.pair_rule
        rules = [
            {
                "rule": fixed_rule,
                "score": score_rule(fixed_rule, train_records, validation_records),
                "pair_rule": rule_text(fixed_rule),
                "clause_count": len(fixed_rule),
                "atom_count": sum(len(clause) for clause in fixed_rule),
            }
        ]
        clauses: list[dict[str, Any]] = []
    else:
        atoms = atoms_for_records(records)
        clauses = enumerate_clauses(atoms, train_records, validation_records, int(args.max_clause_size))
        rules = combine_clauses(clauses, train_records, validation_records, int(args.max_clauses), int(args.beam_size))

    enriched = [enrich_rule(rule, train_records, validation_records, int(args.sample_limit)) for rule in rules[: int(args.top_rules)]]
    best = enriched[0] if enriched else None
    output = {
        "schema": "ecdlp_public_repeated_coordinate_subset_pair_rule_miner_v1",
        "method": "public_feature_rule_mining_for_verifier_labeled_relation_row_subsets",
        "parameters": {
            "decompositions": loaded_decompositions,
            "validation_names": list(args.validation_name or []),
            "allow_exact_coordinate": bool(args.allow_exact_coordinate),
            "allow_exact_salt": bool(args.allow_exact_salt),
            "fixed_pair_rule": rule_text(args.pair_rule) if args.pair_rule else None,
            "max_subset_size": int(args.max_subset_size),
            "max_clause_size": int(args.max_clause_size),
            "max_clauses": int(args.max_clauses),
            "beam_size": int(args.beam_size),
        },
        "record_summary": summary_for_records(records),
        "train_summary": summary_for_records(train_records),
        "validation_summary": summary_for_records(validation_records),
        "candidate_clause_count": len(clauses),
        "frozen_pair_rule": (best or {}).get("pair_rule"),
        "best_rule": best,
        "top_rules": enriched,
        "pair_records_sample": [compact_record(record) for record in records[: int(args.sample_limit)]],
        "non_claims": [
            "Positive labels come from verifier-informed cost decompositions.",
            "Rules are candidate subset predictors only until frozen and replayed on a later window.",
            "Exact coordinate and exact salt features are disabled unless explicitly requested.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "frozen_pair_rule": output["frozen_pair_rule"],
                "best_rule_score": (best or {}).get("score"),
                "record_summary": output["record_summary"],
                "candidate_clause_count": len(clauses),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
