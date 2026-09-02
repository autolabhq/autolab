#!/usr/bin/env python3
"""Mine public branch selectors for repeated-coordinate branch gaps.

This miner aggregates coordinate-gate candidates, row/form replay cases, and
strict pair-rule replay cases by transfer plus repeated coordinate.  It then
scores small public predicates that explain branch material missed by the
strict pair rule.  Replay/verifier labels are used only for scoring; candidate
atoms exclude exact coordinates, exact salts, verifier labels, and
candidate-position labels.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_branch_selector_miner.json"

LABELS = (
    "coordinate_verified",
    "coordinate_verified_missing_strict",
    "row_guard_relation",
    "row_guard_relation_missing_strict",
    "branch_gap",
    "strict_pair_dead",
)

EXACT_OR_LABEL_PREFIXES = (
    "coordinate_key",
    "salt_signature",
    "pair_salt_signature",
    "candidate_pos",
    "label_",
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text())


def parse_named_path(raw: str) -> tuple[str, Path]:
    parts = raw.split("|", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("value must be name|path")
    return parts[0], Path(parts[1])


def parse_clause(raw: str) -> tuple[str, ...]:
    atoms = tuple(atom.strip() for atom in raw.split("&") if atom.strip())
    if not atoms:
        raise argparse.ArgumentTypeError("clause must contain at least one atom")
    for atom in atoms:
        if "=" not in atom:
            raise argparse.ArgumentTypeError(f"atom must be key=value: {atom}")
    return atoms


def clause_text(clause: tuple[str, ...]) -> str:
    return "&".join(clause)


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def policy_family(policy: str) -> str:
    if "target_cap3" in policy:
        return "target_cap3"
    if "global_cap3" in policy:
        return "global_cap3"
    if "target_cap1" in policy:
        return "target_cap1"
    return policy or "unknown"


def leaf_total(leaf_selector: str) -> int:
    if "total3" in leaf_selector:
        return 3
    if "total4" in leaf_selector:
        return 4
    return -1


def coordinate_from_case(case: dict[str, Any]) -> tuple[int, int] | None:
    coordinate = case.get("coordinate") or {}
    if not isinstance(coordinate, dict):
        return None
    if coordinate.get("b") is None or coordinate.get("c") is None:
        return None
    return int(coordinate["b"]), int(coordinate["c"])


def aggregate_key(source: str, case: dict[str, Any]) -> str | None:
    coordinate = coordinate_from_case(case)
    if coordinate is None:
        return None
    transfer_index = case.get("transfer_index")
    if transfer_index is None:
        return None
    b, c = coordinate
    window = str(case.get("window") or source)
    return f"{window}:{int(transfer_index)}:{b},{c}"


def selected_profiles(case: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = case.get("profiles") or case.get("selected_profiles") or []
    return [profile for profile in profiles if isinstance(profile, dict)]


def salts_from_case(case: dict[str, Any]) -> list[int]:
    salts = case.get("salts")
    if salts is not None:
        return sorted({int(salt) for salt in salts})
    found: set[int] = set()
    for profile in selected_profiles(case):
        if profile.get("salt") is not None:
            found.add(int(profile["salt"]))
            continue
        row_key = str(profile.get("row_key") or "")
        if "salt" in row_key:
            try:
                found.add(int(row_key.rsplit("salt", 1)[1].split(":", 1)[0]))
            except ValueError:
                pass
    pair_features = case.get("pair_features") or {}
    if pair_features.get("pair_salt_signature"):
        for raw in str(pair_features["pair_salt_signature"]).split(","):
            if raw.strip():
                found.add(int(raw))
    return sorted(found)


def leaves_from_case(case: dict[str, Any]) -> list[int]:
    leaves = case.get("leaf_indices")
    if leaves is not None:
        return sorted({int(leaf) for leaf in leaves})
    found: set[int] = set()
    for profile in selected_profiles(case):
        for leaf in profile.get("leaf_indices") or []:
            found.add(int(leaf))
    pair_features = case.get("pair_features") or {}
    if pair_features.get("leaf_signature"):
        for raw in str(pair_features["leaf_signature"]).split(","):
            if raw.strip():
                found.add(int(raw))
    return sorted(found)


def int_or_default(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def replay_for_stage(case: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage in {"coordinate", "coordinate_gate"}:
        return case.get("exact_coordinate_replay") or {}
    return case.get("guarded_replay") or {}


def source_ops_millirhos(case: dict[str, Any], replay: dict[str, Any]) -> int:
    source_ops = round_or_none(case.get("source_ops_over_rho"))
    if source_ops is None:
        source_ops = round_or_none(replay.get("ops_over_rho"))
    return int(round(float(source_ops) * 1000)) if source_ops is not None else -1


def public_feature_values(case: dict[str, Any], stage: str) -> dict[str, Any]:
    coordinate = coordinate_from_case(case)
    if coordinate is None:
        return {}
    b, c = coordinate
    transfer_index = int_or_default(case.get("transfer_index"), 0)
    top_k = int_or_default(case.get("top_k"), 0)
    policy = str(case.get("policy") or "")
    leaf_selector = str(case.get("leaf_selector") or "")
    replay = replay_for_stage(case, stage)
    salts = salts_from_case(case)
    leaves = leaves_from_case(case)
    salt_min = min(salts) if salts else -1
    salt_max = max(salts) if salts else -1
    salt_span = salt_max - salt_min if salts else -1
    leaf_min = min(leaves) if leaves else -1
    leaf_max = max(leaves) if leaves else -1
    values: dict[str, Any] = {
        "target": str(case.get("target") or ""),
        "transfer_index": transfer_index,
        "top_k": top_k,
        "policy_family": policy_family(policy),
        "leaf_selector": leaf_selector,
        "leaf_total": leaf_total(leaf_selector),
        "leaf_selector_costed": int(leaf_selector.startswith("mode_cost_")),
        "row_count": int_or_default(case.get("row_count"), len(selected_profiles(case))),
        "salt_count": int_or_default(case.get("salt_count"), len(salts)),
        "profile_count": int_or_default(case.get("profile_count"), len(selected_profiles(case))),
        "leaf_signature": ",".join(str(leaf) for leaf in leaves),
        "leaf_min": leaf_min,
        "leaf_max": leaf_max,
        "salt_span": salt_span,
        "salt_sum_mod4": sum(salts) % 4 if salts else -1,
        "salt_sum_mod8": sum(salts) % 8 if salts else -1,
        "salt_mod2_pattern": ",".join(str(salt % 2) for salt in salts),
        "salt_mod3_pattern": ",".join(str(salt % 3) for salt in salts),
        "salt_mod4_pattern": ",".join(str(salt % 4) for salt in salts),
        "source_ops_millirhos": source_ops_millirhos(case, replay),
    }
    for modulus in (2, 3, 4, 5, 8, 16):
        values[f"transfer_mod{modulus}"] = transfer_index % modulus
        values[f"b_mod{modulus}"] = b % modulus
        values[f"c_mod{modulus}"] = c % modulus
        values[f"b_plus_c_mod{modulus}"] = (b + c) % modulus
        values[f"b_minus_c_mod{modulus}"] = (b - c) % modulus
        values[f"leaf_min_mod{modulus}"] = leaf_min % modulus if leaf_min >= 0 else -1
        values[f"salt_min_mod{modulus}"] = salt_min % modulus if salt_min >= 0 else -1
        values[f"salt_max_mod{modulus}"] = salt_max % modulus if salt_max >= 0 else -1
    for modulus in (2, 3, 4):
        for residue in range(modulus):
            values[f"salt_mod{modulus}_residue{residue}_count"] = sum(
                1 for salt in salts if salt % modulus == residue
            )
    pair_features = case.get("pair_features") or {}
    for key, value in pair_features.items():
        if key.startswith(EXACT_OR_LABEL_PREFIXES):
            continue
        if value is None or isinstance(value, list):
            continue
        values[f"pair_{key}"] = value
    return values


def empty_record(source: str, key: str, case: dict[str, Any]) -> dict[str, Any]:
    b, c = coordinate_from_case(case) or (-1, -1)
    window = str(case.get("window") or source)
    return {
        "key": key,
        "source": source,
        "window": window,
        "transfer_index": int_or_default(case.get("transfer_index"), 0),
        "coordinate": {"b": b, "c": c},
        "features": defaultdict(set),
        "counts": Counter(),
        "min_ops_over_rho": None,
        "form_signatures": Counter(),
        "samples": [],
    }


def add_feature_values(record: dict[str, Any], values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float, str)):
            record["features"][key].add(str(value))


def add_replay(record: dict[str, Any], case: dict[str, Any], stage: str) -> None:
    replay = replay_for_stage(case, stage)
    record["counts"][f"{stage}_case"] += 1
    if int_or_default(replay.get("relation_count"), 0) >= 2:
        record["counts"][f"{stage}_relation"] += 1
    if replay.get("public_key_verified"):
        record["counts"][f"{stage}_verified"] += 1
    if replay.get("public_key_verified") and replay.get("below_rho"):
        record["counts"][f"{stage}_verified_below_rho"] += 1
    ops = round_or_none(replay.get("ops_over_rho"))
    if ops is not None:
        current = record.get("min_ops_over_rho")
        record["min_ops_over_rho"] = ops if current is None else min(float(current), ops)
    form_features = case.get("form_guard_features") or {}
    if form_features.get("candidate_pos_signature"):
        record["form_signatures"][str(form_features["candidate_pos_signature"])] += 1


def add_case(
    records: dict[str, dict[str, Any]],
    source: str,
    stage: str,
    case: dict[str, Any],
) -> None:
    key = aggregate_key(source, case)
    if key is None:
        return
    record = records.setdefault(key, empty_record(source, key, case))
    add_feature_values(record, public_feature_values(case, stage))
    add_replay(record, case, stage)
    if len(record["samples"]) < 4:
        record["samples"].append(
            {
                "stage": stage,
                "top_k": case.get("top_k"),
                "policy": case.get("policy"),
                "leaf_selector": case.get("leaf_selector"),
            }
        )


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for source, path in [parse_named_path(raw) for raw in args.coordinate_gate]:
        data = load_json(path)
        for case in data.get("candidates") or []:
            add_case(records, source, "coordinate", case)
    for source, path in [parse_named_path(raw) for raw in args.guard_replay]:
        data = load_json(path)
        for case in data.get("cases") or []:
            add_case(records, source, "row_guard", case)
    for source, path in [parse_named_path(raw) for raw in args.strict_pair]:
        data = load_json(path)
        for case in data.get("cases") or []:
            add_case(records, source, "strict_pair", case)
    output: list[dict[str, Any]] = []
    for record in records.values():
        counts = record["counts"]
        labels = {
            "coordinate_verified": counts["coordinate_verified"] > 0,
            "coordinate_verified_missing_strict": counts["coordinate_verified"] > 0
            and counts["strict_pair_case"] == 0,
            "row_guard_relation": counts["row_guard_relation"] > 0,
            "row_guard_relation_missing_strict": counts["row_guard_relation"] > 0
            and counts["strict_pair_case"] == 0,
            "strict_pair_dead": counts["strict_pair_case"] > 0
            and counts["strict_pair_relation"] == 0,
        }
        labels["branch_gap"] = labels["coordinate_verified_missing_strict"] or labels[
            "row_guard_relation_missing_strict"
        ]
        record["labels"] = labels
        record["features"] = {key: sorted(values) for key, values in record["features"].items()}
        record["counts"] = dict(counts)
        record["form_signatures"] = dict(record["form_signatures"])
        output.append(record)
    return output


def atom_candidates(records: list[dict[str, Any]], label: str, max_atoms: int) -> list[str]:
    positive = [record for record in records if record["labels"][label]]
    counts: dict[str, Counter[str]] = {"pos": Counter(), "neg": Counter()}
    for record in records:
        target = "pos" if record["labels"][label] else "neg"
        for key, values in record["features"].items():
            if key.startswith(EXACT_OR_LABEL_PREFIXES):
                continue
            for value in values:
                counts[target][f"{key}={value}"] += 1
    atoms = []
    for atom, pos_count in counts["pos"].items():
        if pos_count <= 0:
            continue
        neg_count = counts["neg"].get(atom, 0)
        precision = pos_count / (pos_count + neg_count)
        atoms.append((precision, pos_count, -neg_count, atom))
    atoms.sort(reverse=True)
    return [atom for _precision, _pos_count, _neg_count, atom in atoms[:max_atoms]]


def atom_matches(atom: str, record: dict[str, Any]) -> bool:
    key, value = atom.split("=", 1)
    return value in record["features"].get(key, [])


def clause_matches(clause: tuple[str, ...], record: dict[str, Any]) -> bool:
    return all(atom_matches(atom, record) for atom in clause)


def selected_records(clause: tuple[str, ...], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if clause_matches(clause, record)]


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": record["key"],
        "window": record["window"],
        "transfer_index": record["transfer_index"],
        "coordinate": record["coordinate"],
        "counts": record["counts"],
        "labels": record["labels"],
        "min_ops_over_rho": record["min_ops_over_rho"],
        "form_signatures": record["form_signatures"],
        "sample_features": {
            key: values
            for key, values in record["features"].items()
            if key
            in {
                "policy_family",
                "leaf_signature",
                "leaf_min",
                "b_mod16",
                "c_mod16",
                "b_minus_c_mod16",
                "transfer_mod3",
                "salt_span",
                "salt_mod2_pattern",
                "salt_mod3_pattern",
            }
        },
    }


def score_clause(clause: tuple[str, ...], records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = selected_records(clause, records)
    positives = [record for record in records if record["labels"][label]]
    true_pos = [record for record in selected if record["labels"][label]]
    false_pos = [record for record in selected if not record["labels"][label]]
    windows = sorted({record["window"] for record in true_pos})
    ops = [
        float(record["min_ops_over_rho"])
        for record in true_pos
        if record.get("min_ops_over_rho") is not None
    ]
    return {
        "label": label,
        "clause": clause_text(clause),
        "selected_count": len(selected),
        "true_positive_count": len(true_pos),
        "false_positive_count": len(false_pos),
        "false_negative_count": len(positives) - len(true_pos),
        "positive_count": len(positives),
        "precision": round(len(true_pos) / len(selected), 8) if selected else None,
        "recall": round(len(true_pos) / len(positives), 8) if positives else None,
        "positive_windows": windows,
        "min_positive_ops_over_rho": round(min(ops), 8) if ops else None,
        "mean_positive_ops_over_rho": mean_or_none(ops),
        "selected_examples": [compact_record(record) for record in selected[:8]],
    }


def rank_score(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["false_positive_count"],
        -item["true_positive_count"],
        item["false_negative_count"],
        len(item["clause"].split("&")),
        item["clause"],
    )


def mine_label(
    records: list[dict[str, Any]],
    label: str,
    max_clause_size: int,
    max_atoms: int,
    top_rules: int,
) -> dict[str, Any]:
    atoms = atom_candidates(records, label, max_atoms)
    scored: list[dict[str, Any]] = []
    for size in range(1, max_clause_size + 1):
        for clause in itertools.combinations(atoms, size):
            score = score_clause(clause, records, label)
            if score["true_positive_count"] == 0:
                continue
            scored.append(score)
    scored.sort(key=rank_score)
    return {
        "label": label,
        "positive_count": sum(1 for record in records if record["labels"][label]),
        "candidate_atom_count": len(atoms),
        "top_rules": scored[:top_rules],
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "window_count": len({record["window"] for record in records}),
        "label_counts": {
            label: sum(1 for record in records if record["labels"][label])
            for label in LABELS
        },
        "by_window": {
            window: {
                label: sum(
                    1
                    for record in records
                    if record["window"] == window and record["labels"][label]
                )
                for label in LABELS
            }
            for window in sorted({record["window"] for record in records})
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(args)
    labels = args.label or list(LABELS)
    fixed_clause_scores = {
        clause_text(clause): {
            label: score_clause(clause, records, label)
            for label in labels
        }
        for clause in (args.fixed_clause or [])
    }
    mined = {
        label: mine_label(
            records,
            label,
            int(args.max_clause_size),
            int(args.max_atoms),
            int(args.top_rules),
        )
        for label in labels
    }
    return {
        "schema": "ecdlp_public_repeated_coordinate_branch_selector_miner_v1",
        "method": "public_no_exact_coordinate_no_exact_salt_branch_selector_search",
        "parameters": {
            "coordinate_gate": args.coordinate_gate,
            "guard_replay": args.guard_replay,
            "strict_pair": args.strict_pair,
            "max_clause_size": int(args.max_clause_size),
            "max_atoms": int(args.max_atoms),
            "top_rules": int(args.top_rules),
            "labels": labels,
            "fixed_clauses": [clause_text(clause) for clause in (args.fixed_clause or [])],
        },
        "summary": summarize_records(records),
        "fixed_clause_scores": fixed_clause_scores,
        "mined_rules": mined,
        "non_claims": [
            "Rules are diagnostic branch selectors, not ECDLP speedup claims.",
            "Replay and verifier labels are used only for scoring, not as candidate atoms.",
            "Candidate atoms exclude exact coordinate keys, exact salt signatures, and candidate-position labels.",
            "Rules must be frozen before being tested on a future fresh window.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate-gate", action="append", required=True, help="name|path")
    parser.add_argument("--guard-replay", action="append", required=True, help="name|path")
    parser.add_argument("--strict-pair", action="append", required=True, help="name|path")
    parser.add_argument("--label", action="append", choices=LABELS)
    parser.add_argument("--max-clause-size", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=256)
    parser.add_argument("--top-rules", type=int, default=16)
    parser.add_argument("--fixed-clause", action="append", type=parse_clause)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
