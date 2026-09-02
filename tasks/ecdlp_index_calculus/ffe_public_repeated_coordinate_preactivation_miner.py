#!/usr/bin/env python3
"""Mine public pre-activation cues for the target-67 branch.

The branch-aware replay can only help on windows where the frozen public
stress pipeline emits target-67 repeated-coordinate candidates.  This miner
labels activation from the frozen target-67 activation scan, then searches
stress-artifact public features for small clauses that distinguish activation
windows from dry windows before relation replay.
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
DEFAULT_ACTIVATION_SCAN = (
    DEFAULT_STATE_DIR
    / "ffe_public_repeated_coordinate_branch_activation_scan_target67_bmod4_saltmod2_subset2_752_903.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_preactivation_miner.json"
TARGET = "67.a1@9803"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text())


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def safe_key(value: Any) -> str:
    text = str(value)
    for old, new in ((".", "_"), ("@", "_"), (":", "_"), (",", "_"), ("|", "_"), ("-", "_")):
        text = text.replace(old, new)
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")


def row_salt(row_key: str) -> int | None:
    try:
        return int(str(row_key).rsplit("salt", 1)[1])
    except (IndexError, ValueError):
        return None


def policy_family(policy: str) -> str:
    if "target_cap1" in policy:
        return "target_cap1"
    if "target_cap3" in policy:
        return "target_cap3"
    if "global_cap3" in policy:
        return "global_cap3"
    return safe_key(policy) or "unknown"


def leaf_total(selector: str) -> int:
    if "total4" in selector:
        return 4
    if "total3" in selector:
        return 3
    if "total2" in selector:
        return 2
    return 0


def add_int(features: dict[str, Any], key: str, value: int) -> None:
    features[key] = int(value)
    features[f"{key}_nonzero"] = int(value > 0)


def add_min_millirhos(features: dict[str, Any], key: str, values: list[float]) -> None:
    if not values:
        features[f"{key}_present"] = 0
        features[f"{key}_millirhos"] = -1
        features[f"{key}_lt1000"] = 0
        features[f"{key}_le1100"] = 0
        features[f"{key}_le1250"] = 0
        return
    best = min(values)
    millirhos = int(round(best * 1000))
    features[f"{key}_present"] = 1
    features[f"{key}_millirhos"] = millirhos
    features[f"{key}_lt1000"] = int(best < 1.0)
    features[f"{key}_le1100"] = int(best <= 1.1)
    features[f"{key}_le1250"] = int(best <= 1.25)


def leaf_signature(row_leaf_keys: list[dict[str, Any]]) -> str:
    leaves: list[int] = []
    for item in row_leaf_keys:
        if isinstance(item, dict):
            leaves.extend(int(leaf) for leaf in item.get("leaf_indices") or [])
    return ",".join(str(leaf) for leaf in sorted(leaves))


def salt_pattern(row_keys: list[str], modulus: int) -> str:
    salts = sorted(salt for key in row_keys if (salt := row_salt(key)) is not None)
    return ",".join(str(salt % modulus) for salt in salts)


def collect_policy_results(stress: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    results = []
    for policy, payload in sorted((stress.get("policies") or {}).items()):
        if not isinstance(payload, dict):
            continue
        for case in payload.get("stress_leaf_results") or []:
            if isinstance(case, dict):
                results.append((str(policy), payload, case))
    return results


def stress_features(stress: dict[str, Any], target: str) -> dict[str, Any]:
    features: dict[str, Any] = {}
    policy_payloads = stress.get("policies") or {}
    add_int(features, "policy_count", len(policy_payloads))

    row_case_total = 0
    leaf_case_total = 0
    for policy, payload in sorted(policy_payloads.items()):
        if not isinstance(payload, dict):
            continue
        family = policy_family(str(policy))
        row_summary = payload.get("stress_row_summary") or {}
        leaf_summary = payload.get("stress_leaf_summary") or {}
        row_count = int(row_summary.get("case_count") or 0)
        leaf_count = int(leaf_summary.get("case_count") or 0)
        row_case_total += row_count
        leaf_case_total += leaf_count
        add_int(features, f"policy_{family}_row_case_count", row_count)
        add_int(features, f"policy_{family}_leaf_case_count", leaf_count)

    add_int(features, "stress_row_case_count", row_case_total)
    add_int(features, "stress_leaf_case_count", leaf_case_total)

    all_results = collect_policy_results(stress)
    target_results = [(policy, payload, case) for policy, payload, case in all_results if case.get("target") == target]
    add_int(features, "stress_leaf_result_count", len(all_results))
    add_int(features, "target67_leaf_result_count", len(target_results))
    add_int(features, "target67_transfer_count", len({int(case.get("transfer_index") or 0) for _, _, case in target_results}))
    add_int(features, "target67_topk_count", len({int(case.get("top_k") or 0) for _, _, case in target_results}))
    add_int(features, "target67_policy_family_count", len({policy_family(policy) for policy, _, _ in target_results}))

    target_ops = [float(case["ops_over_rho"]) for _, _, case in target_results if case.get("ops_over_rho") is not None]
    add_min_millirhos(features, "target67_best_public_cost", target_ops)

    target_row_keys = []
    target_leaf_signatures = Counter()
    target_salt_mod2_patterns = Counter()
    target_salt_mod3_patterns = Counter()
    target_salt_spans: list[int] = []
    for policy, _, case in target_results:
        family = policy_family(policy)
        selector = str(case.get("selector") or "")
        top_k = int(case.get("top_k") or 0)
        row_keys = [str(key) for key in case.get("row_keys") or []]
        salts = sorted(salt for key in row_keys if (salt := row_salt(key)) is not None)
        target_row_keys.extend(row_keys)
        features[f"has_target67_policy_{family}"] = 1
        features[f"has_target67_selector_{safe_key(selector)}"] = 1
        features[f"has_target67_leaf_total{leaf_total(selector)}"] = 1
        features[f"has_target67_topk_{top_k}"] = 1
        signature = leaf_signature(case.get("row_leaf_keys") or [])
        if signature:
            target_leaf_signatures[signature] += 1
        pattern2 = salt_pattern(row_keys, 2)
        pattern3 = salt_pattern(row_keys, 3)
        if pattern2:
            target_salt_mod2_patterns[pattern2] += 1
        if pattern3:
            target_salt_mod3_patterns[pattern3] += 1
        if salts:
            target_salt_spans.append(max(salts) - min(salts))

    for key in ("target_cap1", "target_cap3", "global_cap3"):
        features.setdefault(f"has_target67_policy_{key}", 0)
    for total in (3, 4):
        features.setdefault(f"has_target67_leaf_total{total}", 0)
    for top_k in (4, 7, 12, 16):
        features.setdefault(f"has_target67_topk_{top_k}", 0)

    add_int(features, "target67_row_key_count", len(target_row_keys))
    add_int(features, "target67_unique_row_key_count", len(set(target_row_keys)))
    add_int(features, "target67_leaf_signature_count", len(target_leaf_signatures))
    add_int(features, "target67_salt_mod2_pattern_count", len(target_salt_mod2_patterns))
    add_int(features, "target67_salt_mod3_pattern_count", len(target_salt_mod3_patterns))
    add_min_millirhos(features, "target67_min_salt_span", [float(span) for span in target_salt_spans])

    for signature, count in target_leaf_signatures.most_common(12):
        features[f"target67_leaf_signature_{safe_key(signature)}_count"] = int(count)
    for pattern, count in target_salt_mod2_patterns.most_common(12):
        features[f"target67_salt_mod2_pattern_{safe_key(pattern)}_count"] = int(count)
    for pattern, count in target_salt_mod3_patterns.most_common(12):
        features[f"target67_salt_mod3_pattern_{safe_key(pattern)}_count"] = int(count)

    return features


def atom_matches(atom: str, features: dict[str, Any]) -> bool:
    if ">=" in atom:
        key, raw_value = atom.split(">=", 1)
        return int(features.get(key, 0)) >= int(raw_value)
    if "<=" in atom:
        key, raw_value = atom.split("<=", 1)
        return int(features.get(key, 0)) <= int(raw_value)
    key, raw_value = atom.split("=", 1)
    return str(features.get(key, 0)) == raw_value


def clause_matches(clause: tuple[str, ...], features: dict[str, Any]) -> bool:
    return all(atom_matches(atom, features) for atom in clause)


def atoms_for_features(features: dict[str, Any]) -> set[str]:
    atoms: set[str] = set()
    for key, value in sorted(features.items()):
        if value is None:
            continue
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int):
            continue
        atoms.add(f"{key}={value}")
        if value > 0:
            atoms.add(f"{key}>=1")
        if value >= 2:
            atoms.add(f"{key}>=2")
        if value >= 4:
            atoms.add(f"{key}>=4")
        if value >= 8:
            atoms.add(f"{key}>=8")
        if value >= 16:
            atoms.add(f"{key}>=16")
        if value >= 32:
            atoms.add(f"{key}>=32")
        if value == 0:
            atoms.add(f"{key}<=0")
    return atoms


def load_records(activation_scan: Path, target: str) -> list[dict[str, Any]]:
    scan = load_json(activation_scan)
    records = []
    for window in scan.get("windows") or []:
        stress_source = window.get("stress_source")
        stress = load_json(Path(stress_source)) if stress_source else {}
        features = stress_features(stress, target)
        activated = int(window.get("coordinate_candidate_count") or 0) > 0
        branch_clause = int(window.get("branch_clause_match_count") or 0) > 0
        records.append(
            {
                "window": window.get("window"),
                "stress_source": stress_source,
                "activated": bool(activated),
                "branch_clause_matched": bool(branch_clause),
                "selected_public_case_count": int(window.get("selected_public_case_count") or 0),
                "coordinate_candidate_count": int(window.get("coordinate_candidate_count") or 0),
                "branch_clause_match_count": int(window.get("branch_clause_match_count") or 0),
                "features": features,
            }
        )
    return records


def score_clause(clause: tuple[str, ...], records: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    selected = [record for record in records if clause_matches(clause, record["features"])]
    positives = [record for record in records if record[label_key]]
    true_pos = [record for record in selected if record[label_key]]
    false_pos = [record for record in selected if not record[label_key]]
    false_neg = [record for record in positives if not clause_matches(clause, record["features"])]
    precision = len(true_pos) / len(selected) if selected else 0.0
    recall = len(true_pos) / len(positives) if positives else 0.0
    return {
        "clause": "&".join(clause),
        "selected_count": len(selected),
        "true_positive_count": len(true_pos),
        "false_positive_count": len(false_pos),
        "false_negative_count": len(false_neg),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "selected_windows": [record["window"] for record in selected],
        "false_positive_windows": [record["window"] for record in false_pos],
        "false_negative_windows": [record["window"] for record in false_neg],
    }


def mine_clauses(records: list[dict[str, Any]], label_key: str, max_clause_size: int, max_atoms: int) -> list[dict[str, Any]]:
    positives = [record for record in records if record[label_key]]
    atom_counts = Counter()
    for record in positives:
        atom_counts.update(atoms_for_features(record["features"]))
    atoms = [atom for atom, _ in atom_counts.most_common(max_atoms)]
    scored = []
    for size in range(1, max_clause_size + 1):
        for clause in itertools.combinations(atoms, size):
            scored.append(score_clause(clause, records, label_key))
    scored.sort(
        key=lambda item: (
            int(item["false_positive_count"]),
            -float(item["recall"]),
            -int(item["true_positive_count"]),
            int(item["selected_count"]),
            item["clause"],
        )
    )
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-scan", type=Path, default=DEFAULT_ACTIVATION_SCAN)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--label-key", choices=["activated", "branch_clause_matched"], default="activated")
    parser.add_argument("--max-clause-size", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=256)
    parser.add_argument("--top-rules", type=int, default=24)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records = load_records(args.activation_scan, args.target)
    rules = mine_clauses(records, args.label_key, int(args.max_clause_size), int(args.max_atoms))
    labels = Counter("positive" if record[args.label_key] else "negative" for record in records)
    output = {
        "schema": "ecdlp_public_repeated_coordinate_preactivation_miner_v1",
        "parameters": {
            "activation_scan": str(args.activation_scan),
            "target": args.target,
            "label_key": args.label_key,
            "max_clause_size": int(args.max_clause_size),
            "max_atoms": int(args.max_atoms),
        },
        "summary": {
            "record_count": len(records),
            "label_counts": dict(sorted(labels.items())),
            "positive_windows": [record["window"] for record in records if record[args.label_key]],
            "negative_windows": [record["window"] for record in records if not record[args.label_key]],
            "top_rules": rules[: int(args.top_rules)],
        },
        "records": records,
    }
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
