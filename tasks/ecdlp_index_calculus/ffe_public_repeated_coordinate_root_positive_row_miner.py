#!/usr/bin/env python3
"""Mine public row-pruning rules for repeated-coordinate gates.

Coordinate-gate decompositions show that verifier-informed root-positive row
subsets can replay below rho.  This miner reruns repeated-coordinate candidates
row-by-row, labels each row by ``selected_hit_roots > 0``, and searches small
public predicates that predict those root-positive rows before relation events
or public-key verification are inspected.

The label is still produced by the current preassociation scan and is not free.
Rules found here are row-pruning candidates, not completed speedups, until the
same selection can be charged honestly in a staged replay.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_activation_rule_miner as activation_miner
import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_root_positive_row_miner.json"


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


def replay_args(params: dict[str, Any], event_summary_limit: int) -> argparse.Namespace:
    return argparse.Namespace(
        radius=int(params.get("radius") or 4),
        event_summary_limit=int(event_summary_limit),
        row_pool=512,
        row_count=128,
        scout_limit=192,
        scout_mode="s3_coeff_spread",
        scout_order="eval_cover_hits_high",
        selected_limit=64,
        factor_base_size=16,
        max_relations=96,
        min_distinct_indices=4,
        min_unsigned_distinct_indices=2,
        require_unit_coefficients=True,
        row_factor=512,
        product_factor=4096,
        seed=str(params.get("seed") or "ecdlp-frontier-signed-dual-sieve-v1"),
    )


def candidate_activation_match(
    candidate: dict[str, Any],
    window_name: str,
    source_path: str,
    activation_rule: tuple[tuple[str, ...], ...] | None,
) -> tuple[bool, list[str]]:
    if activation_rule is None:
        return True, ["all"]
    feature = activation_miner.feature_record(candidate, window_name, source_path)
    matches = [
        "&".join(clause)
        for clause in activation_rule
        if activation_miner.clause_matches(clause, feature)
    ]
    return bool(matches), matches


def context_sources(artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    params = artifact.get("parameters") or {}
    bank_source = load_json(Path(params["bank_source"]))
    config_source = load_json(Path(params["config_source"]))
    direct_source = load_json(Path(params["direct_source"]))
    transfer_source = load_json(Path(params["transfer_source"]))
    transfer_params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(params.get("radius") or (transfer_params or {}).get("radius") or 4)
    return bank_source, config_source, direct_source, radius


def row_public_features(
    candidate: dict[str, Any],
    profile: dict[str, Any],
    window_name: str,
    source_path: str,
    activation_selected: bool,
    activation_clauses: list[str],
) -> dict[str, Any]:
    b_value = int(candidate.get("b") if candidate.get("b") is not None else (candidate.get("coordinate") or {}).get("b") or 0)
    c_value = int(candidate.get("c") if candidate.get("c") is not None else (candidate.get("coordinate") or {}).get("c") or 0)
    p = int(candidate.get("p") or profile.get("p") or 9803)
    salts = sorted(int(salt) for salt in candidate.get("salts") or [])
    leaf_indices = sorted(int(leaf) for leaf in candidate.get("leaf_indices") or [])
    salt = int(profile.get("salt") if profile.get("salt") is not None else -1)
    leaf_index = int(profile.get("leaf_index") or 0)
    transfer_index = int(candidate.get("transfer_index") or 0)
    top_k = int(candidate.get("top_k") or 0)
    leaf_selector = str(candidate.get("leaf_selector") or "")
    policy = str(candidate.get("policy") or "")
    source_ops = round_or_none(candidate.get("source_ops_over_rho"))
    source_ops_millirhos = int(round(float(source_ops) * 1000)) if source_ops is not None else -1
    salt_index = salts.index(salt) if salt in salts else -1
    salt_min = min(salts) if salts else -1
    salt_max = max(salts) if salts else -1
    return {
        "window": window_name,
        "source": source_path,
        "case_key": candidate.get("case_key"),
        "target": str(candidate.get("target") or ""),
        "transfer_index": transfer_index,
        "top_k": top_k,
        "policy": policy,
        "policy_family": policy_family(policy),
        "leaf_selector": leaf_selector,
        "leaf_total": selector_total(leaf_selector) or -1,
        "leaf_selector_costed": int(leaf_selector.startswith("mode_cost_")),
        "candidate_activation_selected": int(activation_selected),
        "candidate_activation_signature": "|".join(sorted(activation_clauses)),
        "p": p,
        "b": b_value,
        "c": c_value,
        "b_mod2": b_value % 2,
        "b_mod3": b_value % 3,
        "b_mod4": b_value % 4,
        "b_mod5": b_value % 5,
        "b_mod8": b_value % 8,
        "b_mod16": b_value % 16,
        "c_mod2": c_value % 2,
        "c_mod3": c_value % 3,
        "c_mod4": c_value % 4,
        "c_mod5": c_value % 5,
        "c_mod8": c_value % 8,
        "c_mod16": c_value % 16,
        "b_plus_c_mod8": (b_value + c_value) % 8,
        "b_plus_c_mod16": (b_value + c_value) % 16,
        "b_minus_c_mod8": (b_value - c_value) % 8,
        "b_minus_c_mod16": (b_value - c_value) % 16,
        "coordinate_key": f"{b_value},{c_value}",
        "row_count": int(candidate.get("row_count") or 0),
        "salt_count": int(candidate.get("salt_count") or 0),
        "profile_count": int(candidate.get("profile_count") or 0),
        "salt": salt,
        "salt_index": salt_index,
        "salt_min": salt_min,
        "salt_max": salt_max,
        "salt_span": salt_max - salt_min if salts else -1,
        "salt_delta_from_min": salt - salt_min if salts and salt >= 0 else -1,
        "salt_delta_to_max": salt_max - salt if salts and salt >= 0 else -1,
        "salt_signature": ",".join(str(item) for item in salts),
        "leaf_index": leaf_index,
        "leaf_min": min(leaf_indices) if leaf_indices else -1,
        "leaf_max": max(leaf_indices) if leaf_indices else -1,
        "leaf_signature": ",".join(str(item) for item in leaf_indices),
        "source_ops_millirhos": source_ops_millirhos,
        "source_below_rho": int(bool(candidate.get("source_below_rho"))),
        "source_public_key_verified": int(bool(candidate.get("source_public_key_verified"))),
        "row_key": str(profile.get("row_key") or ""),
        "surface_id": str(profile.get("surface_id") or ""),
    }


for _modulus in (2, 3, 4, 5, 8, 16):
    # Populated dynamically in row_public_features.
    pass


NUMERIC_KEYS = {
    "transfer_index",
    "top_k",
    "leaf_total",
    "leaf_selector_costed",
    "candidate_activation_selected",
    "p",
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
    "salt",
    "salt_index",
    "salt_min",
    "salt_max",
    "salt_span",
    "salt_delta_from_min",
    "salt_delta_to_max",
    "leaf_index",
    "leaf_min",
    "leaf_max",
    "source_ops_millirhos",
    "source_below_rho",
}
for _modulus in (2, 3, 4, 5, 8, 16):
    NUMERIC_KEYS.update(
        {
            f"transfer_mod{_modulus}",
            f"salt_mod{_modulus}",
            f"salt_min_mod{_modulus}",
            f"salt_max_mod{_modulus}",
            f"leaf_mod{_modulus}",
            f"leaf_min_mod{_modulus}",
        }
    )

STRING_KEYS = {
    "target",
    "policy",
    "policy_family",
    "leaf_selector",
    "coordinate_key",
    "salt_signature",
    "leaf_signature",
}


def add_mod_features(record: dict[str, Any]) -> None:
    transfer_index = int(record["transfer_index"])
    salt = int(record["salt"])
    salt_min = int(record["salt_min"])
    salt_max = int(record["salt_max"])
    leaf_index = int(record["leaf_index"])
    leaf_min = int(record["leaf_min"])
    for modulus in (2, 3, 4, 5, 8, 16):
        record[f"transfer_mod{modulus}"] = transfer_index % modulus
        record[f"salt_mod{modulus}"] = salt % modulus if salt >= 0 else -1
        record[f"salt_min_mod{modulus}"] = salt_min % modulus if salt_min >= 0 else -1
        record[f"salt_max_mod{modulus}"] = salt_max % modulus if salt_max >= 0 else -1
        record[f"leaf_mod{modulus}"] = leaf_index % modulus
        record[f"leaf_min_mod{modulus}"] = leaf_min % modulus if leaf_min >= 0 else -1


def scan_row_label(
    verifier: Any,
    records: list[dict[str, Any]],
    config_source: dict[str, Any],
    specs_by_target: dict[str, dict[str, dict[str, Any]]],
    candidate: dict[str, Any],
    profile: dict[str, Any],
    args: argparse.Namespace,
    replay_local_args: argparse.Namespace,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]],
) -> dict[str, Any]:
    row_key = str(profile.get("row_key") or "")
    leaf_index = int(profile.get("leaf_index") or 0)
    case = {
        "target": candidate.get("target"),
        "transfer_index": int(candidate.get("transfer_index") or 0),
        "top_k": int(candidate.get("top_k") or 0),
    }
    contexts, errors = replay_probe.materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        case,
        [row_key],
        replay_local_args,
        context_cache,
    )
    if errors or row_key not in contexts:
        return {
            "context_error": True,
            "context_errors": errors,
            "selected_hit_roots": 0,
            "selected_hit_events": 0,
            "candidate_verifications": 0,
            "relation_event_count": 0,
            "row_ops_over_rho": None,
            "row_ops": None,
            "generic_rho_steps": None,
        }
    replay, _events = replay_probe.replay_selection(
        verifier,
        {row_key: {leaf_index}},
        contexts,
        scan_cache,
        int(args.event_summary_limit),
    )
    row = (replay.get("rows") or [{}])[0]
    scan = row.get("scan") or {}
    return {
        "context_error": False,
        "context_errors": [],
        "selected_hit_roots": int(scan.get("selected_hit_roots") or 0),
        "selected_hit_events": int(scan.get("selected_hit_events") or 0),
        "candidate_verifications": int(scan.get("candidate_verifications") or 0),
        "relation_event_count": int(scan.get("event_summary_count") or 0),
        "row_rank": int(scan.get("rank") or 0),
        "row_public_key_verified": bool(scan.get("row_public_key_verified")),
        "row_derived_secret": scan.get("derived_secret"),
        "row_ops": int(scan.get("preassociation_filter_ops") or 0),
        "row_ops_over_rho": scan.get("preassociation_filter_ops_over_rho"),
        "generic_rho_steps": int(row.get("generic_rho_steps") or 0),
    }


def load_row_records(
    windows: list[tuple[str, Path]],
    activation_rule: tuple[tuple[str, ...], ...] | None,
    activated_only: bool,
    max_candidates_per_window: int,
    event_summary_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_records: list[dict[str, Any]] = []
    context_errors: list[dict[str, Any]] = []
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    verifier = replay_probe.relation_probe.load_verifier_module()
    verifier_records = verifier.load_records()
    candidate_count = 0
    scanned_candidate_count = 0
    for window_name, path in windows:
        artifact = load_json(path)
        bank_source, config_source, direct_source, radius = context_sources(artifact)
        specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
        params = artifact.get("parameters") or {}
        local_args = replay_args(params, event_summary_limit)
        scanned_for_window = 0
        for candidate in artifact.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            candidate_count += 1
            if max_candidates_per_window > 0 and scanned_for_window >= max_candidates_per_window:
                continue
            activation_selected, activation_clauses = candidate_activation_match(
                candidate,
                window_name,
                str(path),
                activation_rule,
            )
            if activated_only and not activation_selected:
                continue
            scanned_for_window += 1
            scanned_candidate_count += 1
            for profile in candidate.get("profiles") or []:
                if not isinstance(profile, dict):
                    continue
                public = row_public_features(
                    candidate,
                    profile,
                    window_name,
                    str(path),
                    activation_selected,
                    activation_clauses,
                )
                add_mod_features(public)
                label = scan_row_label(
                    verifier,
                    verifier_records,
                    config_source,
                    specs_by_target,
                    candidate,
                    profile,
                    argparse.Namespace(event_summary_limit=event_summary_limit),
                    local_args,
                    context_cache,
                    scan_cache,
                )
                context_errors.extend(label.get("context_errors") or [])
                public.update(label)
                public["label_root_positive"] = bool(int(label.get("selected_hit_roots") or 0) > 0)
                public["label_relation_positive"] = bool(int(label.get("relation_event_count") or 0) > 0)
                row_records.append(public)
    diagnostics = {
        "input_candidate_count": candidate_count,
        "scanned_candidate_count": scanned_candidate_count,
        "row_record_count": len(row_records),
        "context_error_count": len(context_errors),
        "context_errors": context_errors[:32],
        "context_errors_truncated": len(context_errors) > 32,
    }
    return row_records, diagnostics


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
                "salt",
                "salt_index",
                "salt_delta_from_min",
                "salt_delta_to_max",
                "leaf_index",
                "top_k",
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
    positives = [record for record in records if record["label_root_positive"]]
    true_pos = [record for record in selected if record["label_root_positive"]]
    false_pos = [record for record in selected if not record["label_root_positive"]]
    false_neg = len(positives) - len(true_pos)
    selected_ops = [
        float(record["row_ops_over_rho"])
        for record in selected
        if record.get("row_ops_over_rho") is not None
    ]
    true_ops = [
        float(record["row_ops_over_rho"])
        for record in true_pos
        if record.get("row_ops_over_rho") is not None
    ]
    return {
        "row_count": len(records),
        "positive_count": len(positives),
        "selected_count": len(selected),
        "true_positive_count": len(true_pos),
        "false_positive_count": len(false_pos),
        "false_negative_count": false_neg,
        "precision": round(len(true_pos) / len(selected), 8) if selected else None,
        "recall": round(len(true_pos) / len(positives), 8) if positives else None,
        "selected_ops_over_rho_sum": round(sum(selected_ops), 8) if selected_ops else 0.0,
        "true_positive_ops_over_rho_sum": round(sum(true_ops), 8) if true_ops else 0.0,
        "mean_selected_ops_over_rho": mean_or_none(selected_ops),
    }


def score_rule(
    rule: tuple[tuple[str, ...], ...],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_selected = selected_records(rule, train_records)
    validation_selected = selected_records(rule, validation_records)
    return {
        "train": score_selection(train_selected, train_records),
        "validation": score_selection(validation_selected, validation_records),
    }


def clause_text(clause: tuple[str, ...]) -> str:
    return "&".join(clause)


def rule_text(rule: tuple[tuple[str, ...], ...]) -> str:
    return "row_activate:" + "|".join(clause_text(clause) for clause in rule)


def parse_row_rule(raw: str) -> tuple[tuple[str, ...], ...]:
    if not raw.startswith("row_activate:"):
        raise argparse.ArgumentTypeError("row rule must start with row_activate:")
    expression = raw.split(":", 1)[1].strip()
    if not expression:
        raise argparse.ArgumentTypeError("row rule has no clauses")
    clauses = []
    for clause_value in expression.split("|"):
        atoms = tuple(atom.strip() for atom in clause_value.split("&") if atom.strip())
        if not atoms:
            raise argparse.ArgumentTypeError("row rule contains an empty clause")
        clauses.append(atoms)
    return tuple(clauses)


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
        -validation["true_positive_count"],
        -train["true_positive_count"],
        -validation_recall,
        -train_recall,
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
                    "row_rule": rule_text(rule),
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
                ordered = tuple(sorted((*base_rule, add_clause), key=clause_text))
                text = rule_text(ordered)
                if text in candidates:
                    continue
                score = score_rule(ordered, train_records, validation_records)
                if score["train"]["true_positive_count"] == 0 and score["validation"]["true_positive_count"] == 0:
                    continue
                candidates[text] = {
                    "rule": ordered,
                    "score": score,
                    "row_rule": text,
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
        unique[item["row_rule"]] = item
    return sorted(unique.values(), key=rank_key)


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "window": record.get("window"),
        "target": record.get("target"),
        "transfer_index": record.get("transfer_index"),
        "top_k": record.get("top_k"),
        "policy": record.get("policy"),
        "leaf_selector": record.get("leaf_selector"),
        "coordinate": {"b": record.get("b"), "c": record.get("c")},
        "salt": record.get("salt"),
        "leaf_index": record.get("leaf_index"),
        "candidate_activation_selected": record.get("candidate_activation_selected"),
        "selected_hit_roots": record.get("selected_hit_roots"),
        "selected_hit_events": record.get("selected_hit_events"),
        "relation_event_count": record.get("relation_event_count"),
        "row_ops_over_rho": record.get("row_ops_over_rho"),
        "label_root_positive": record.get("label_root_positive"),
    }


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
        "row_rule": item["row_rule"],
        "clause_count": item["clause_count"],
        "atom_count": item["atom_count"],
        "score": item["score"],
        "train_selected_rows": [compact_record(record) for record in train_selected[:sample_limit]],
        "validation_selected_rows": [
            compact_record(record) for record in validation_selected[:sample_limit]
        ],
    }


def window_split(records: list[dict[str, Any]], validation_names: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_names:
        train = [record for record in records if str(record.get("window")) not in validation_names]
        validation = [record for record in records if str(record.get("window")) in validation_names]
        return train, validation
    return records, []


def summary_for_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [record for record in records if record.get("label_root_positive")]
    relation_positive = [record for record in records if record.get("label_relation_positive")]
    return {
        "row_count": len(records),
        "root_positive_count": len(positives),
        "relation_positive_count": len(relation_positive),
        "context_error_count": sum(1 for record in records if record.get("context_error")),
        "root_positive_rate": round(len(positives) / len(records), 8) if records else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=parse_window, action="append", required=True)
    parser.add_argument("--validation-name", action="append", default=[])
    parser.add_argument("--activation-rule", type=activation_miner.parse_activation_rule)
    parser.add_argument("--activated-only", action="store_true")
    parser.add_argument("--allow-exact-coordinate", action="store_true")
    parser.add_argument("--row-rule", type=parse_row_rule)
    parser.add_argument("--max-candidates-per-window", type=int, default=0)
    parser.add_argument("--max-clause-size", type=int, default=2)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=128)
    parser.add_argument("--top-rules", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--event-summary-limit", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records, diagnostics = load_row_records(
        args.window,
        args.activation_rule,
        bool(args.activated_only),
        int(args.max_candidates_per_window),
        int(args.event_summary_limit),
    )
    train_records, validation_records = window_split(records, set(args.validation_name or []))
    atoms = atoms_for_records(records, bool(args.allow_exact_coordinate))
    if args.row_rule:
        fixed_rule = args.row_rule
        rules = [
            {
                "rule": fixed_rule,
                "score": score_rule(fixed_rule, train_records, validation_records),
                "row_rule": rule_text(fixed_rule),
                "clause_count": len(fixed_rule),
                "atom_count": sum(len(clause) for clause in fixed_rule),
            }
        ]
        clauses: list[dict[str, Any]] = []
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
        "schema": "ecdlp_public_repeated_coordinate_root_positive_row_miner_v1",
        "method": "public_feature_rule_mining_for_selected_hit_roots_positive_rows",
        "parameters": {
            "windows": [{"name": name, "artifact": str(path)} for name, path in args.window],
            "validation_names": list(args.validation_name or []),
            "activation_rule": (
                activation_miner.rule_text(args.activation_rule) if args.activation_rule else None
            ),
            "activated_only": bool(args.activated_only),
            "allow_exact_coordinate": bool(args.allow_exact_coordinate),
            "fixed_row_rule": rule_text(args.row_rule) if args.row_rule else None,
            "max_candidates_per_window": int(args.max_candidates_per_window),
            "max_clause_size": int(args.max_clause_size),
            "max_clauses": int(args.max_clauses),
            "beam_size": int(args.beam_size),
            "event_summary_limit": int(args.event_summary_limit),
        },
        "diagnostics": diagnostics,
        "record_summary": summary_for_records(records),
        "train_summary": summary_for_records(train_records),
        "validation_summary": summary_for_records(validation_records),
        "candidate_atom_count": len(atoms),
        "candidate_clause_count": len(clauses),
        "frozen_row_rule": (best or {}).get("row_rule"),
        "best_rule": best,
        "top_rules": enriched,
        "row_records_sample": [compact_record(record) for record in records[: int(args.sample_limit)]],
        "non_claims": [
            "selected_hit_roots>0 is used as a training label and is not treated as a free selector.",
            "Rules use public row and coordinate metadata before relation events or public-key verification.",
            "A root-positive row rule must still be charged in staged replay before it can support a speedup claim.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "frozen_row_rule": output["frozen_row_rule"],
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
