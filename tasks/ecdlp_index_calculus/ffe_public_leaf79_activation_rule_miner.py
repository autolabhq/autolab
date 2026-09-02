#!/usr/bin/env python3
"""Mine public activation rules for the intermittent leaf-79 gate.

The orientation rule ``where:support=0+5&scheduled_trial=1`` can produce
measured below-rho recoveries, but only in some windows.  This script ranks
case-level activation filters over public metadata from prior audit artifacts.
Labels are used only to rank rules on training windows; the emitted
``activation_rule`` must be frozen before replaying a later holdout.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def row_salt(row_key: str) -> int | None:
    if ":salt" not in row_key:
        return None
    try:
        return int(row_key.rsplit(":salt", 1)[1])
    except ValueError:
        return None


def support_key(match: dict[str, Any]) -> str:
    support = sorted({int(index) for index in match.get("unsigned_indices") or []})
    return "+".join(str(index) for index in support)


def feature_record(case: dict[str, Any], source: str) -> dict[str, Any]:
    rows = case.get("rows") or []
    xmatches = [match for row in rows for match in (row.get("xmatches") or [])]
    support_counts: dict[str, int] = {}
    scheduled_counts: dict[int, int] = {}
    salt_mods: dict[int, list[int]] = {2: [], 3: [], 4: []}
    row_xmatch_counts: list[int] = []
    for row in rows:
        row_xmatch_counts.append(len(row.get("xmatches") or []))
        salt = row_salt(str(row.get("row_key") or ""))
        if salt is not None:
            for modulus in salt_mods:
                salt_mods[modulus].append(salt % modulus)
        for match in row.get("xmatches") or []:
            support_counts[support_key(match)] = support_counts.get(support_key(match), 0) + 1
            scheduled = int(match.get("scheduled_trial") or 0)
            scheduled_counts[scheduled] = scheduled_counts.get(scheduled, 0) + 1
    rule_result = (case.get("rule_results") or [{}])[0]
    charged = rule_result.get("charged_models") or {}
    verified = bool(rule_result.get("public_key_verified"))
    measured = charged.get("measured_oriented_ops_over_rho")
    return {
        "source": source,
        "case_key": case.get("case_key"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "leaf_selector": case.get("leaf_selector"),
        "row_count": len(rows),
        "factor_zero_profile_count": len(case.get("factor_zero_profiles") or []),
        "xmatch_count": len(xmatches),
        "support05_count": int(support_counts.get("0+5") or 0),
        "support15_count": int(support_counts.get("1+5") or 0),
        "scheduled1_count": int(scheduled_counts.get(1) or 0),
        "row_xmatch_counts": sorted(row_xmatch_counts),
        "salt_mod2_pattern": sorted(salt_mods[2]),
        "salt_mod3_pattern": sorted(salt_mods[3]),
        "salt_mod4_pattern": sorted(salt_mods[4]),
        "salt_mod2_residue_counts": {str(residue): salt_mods[2].count(residue) for residue in range(2)},
        "salt_mod3_residue_counts": {str(residue): salt_mods[3].count(residue) for residue in range(3)},
        "salt_mod4_residue_counts": {str(residue): salt_mods[4].count(residue) for residue in range(4)},
        "label_verified": verified,
        "label_measured_below_rho": verified and measured is not None and float(measured) < 1.0,
        "label_measured_ops_over_rho": measured,
        "label_rank": rule_result.get("rank"),
        "label_relation_count": rule_result.get("relation_count"),
        "label_derived_secret": rule_result.get("derived_secret"),
    }


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        artifact = load_json(path)
        for case in artifact.get("cases") or []:
            record = feature_record(case, str(path))
            if record["factor_zero_profile_count"] or record["xmatch_count"]:
                records.append(record)
    return records


def atoms_for_records(records: list[dict[str, Any]]) -> list[str]:
    atoms: set[str] = set()
    for record in records:
        for key in ("policy", "leaf_selector"):
            atoms.add(f"{key}={record[key]}")
        for key in (
            "top_k",
            "row_count",
            "factor_zero_profile_count",
            "xmatch_count",
            "support05_count",
            "support15_count",
            "scheduled1_count",
        ):
            value = int(record[key])
            atoms.add(f"{key}={value}")
            atoms.add(f"{key}>={value}")
            atoms.add(f"{key}<={value}")
        for modulus in (2, 3, 4, 5, 8):
            atoms.add(f"transfer_mod{modulus}={record['transfer_index'] % modulus}")
        for key in ("row_xmatch_counts", "salt_mod2_pattern", "salt_mod3_pattern", "salt_mod4_pattern"):
            atoms.add(f"{key}={','.join(str(part) for part in record[key])}")
        for modulus in (2, 3, 4):
            counts = record[f"salt_mod{modulus}_residue_counts"]
            for residue_text, count in counts.items():
                key = f"salt_mod{modulus}_residue{residue_text}_count"
                atoms.add(f"{key}={count}")
                atoms.add(f"{key}>={count}")
                atoms.add(f"{key}<={count}")
    return sorted(atom for atom in atoms if not atom.endswith("=None"))


def numeric_feature(record: dict[str, Any], key: str) -> int:
    if key.startswith("salt_mod") and "_residue" in key and key.endswith("_count"):
        prefix, residue_text = key.removesuffix("_count").split("_residue", 1)
        return int(record[f"{prefix}_residue_counts"].get(residue_text, 0))
    return int(record[key])


def atom_matches(atom: str, record: dict[str, Any]) -> bool:
    if ">=" in atom:
        key, value = atom.split(">=", 1)
        return numeric_feature(record, key) >= int(value)
    if "<=" in atom:
        key, value = atom.split("<=", 1)
        return numeric_feature(record, key) <= int(value)
    key, value = atom.split("=", 1)
    if key.startswith("transfer_mod"):
        return int(record["transfer_index"]) % int(key.removeprefix("transfer_mod")) == int(value)
    if key.startswith("salt_mod") and "_residue" in key and key.endswith("_count"):
        return numeric_feature(record, key) == int(value)
    if key in {"row_xmatch_counts", "salt_mod2_pattern", "salt_mod3_pattern", "salt_mod4_pattern"}:
        return ",".join(str(part) for part in record[key]) == value
    if key in {
        "top_k",
        "row_count",
        "factor_zero_profile_count",
        "xmatch_count",
        "support05_count",
        "support15_count",
        "scheduled1_count",
    }:
        return int(record[key]) == int(value)
    return str(record.get(key)) == value


def rule_matches(rule: tuple[str, ...], record: dict[str, Any]) -> bool:
    return all(atom_matches(atom, record) for atom in rule)


def score_rule(rule: tuple[str, ...], records: list[dict[str, Any]]) -> tuple[Any, ...]:
    selected = [record for record in records if rule_matches(rule, record)]
    true_pos = sum(1 for record in selected if record["label_measured_below_rho"])
    false_pos = sum(1 for record in selected if not record["label_measured_below_rho"])
    positives = sum(1 for record in records if record["label_measured_below_rho"])
    false_neg = positives - true_pos
    return (
        false_pos,
        false_neg,
        -true_pos,
        len(selected),
        len(rule),
        rule,
    )


def rule_text(rule: tuple[str, ...]) -> str:
    return "activate:" + "&".join(rule)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-artifact", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-clause-size", type=int, default=3)
    parser.add_argument("--top-rules", type=int, default=16)
    args = parser.parse_args()

    records = load_records(args.training_artifact)
    atoms = atoms_for_records(records)
    scored = []
    for size in range(1, int(args.max_clause_size) + 1):
        for rule in itertools.combinations(atoms, size):
            score = score_rule(rule, records)
            selected = [record for record in records if rule_matches(rule, record)]
            if not selected:
                continue
            scored.append(
                {
                    "activation_rule": rule_text(rule),
                    "score": score[:-1],
                    "selected_case_count": len(selected),
                    "selected_positive_count": sum(1 for record in selected if record["label_measured_below_rho"]),
                    "selected_negative_count": sum(1 for record in selected if not record["label_measured_below_rho"]),
                    "selected_cases": [
                        {
                            key: record[key]
                            for key in (
                                "case_key",
                                "transfer_index",
                                "policy",
                                "leaf_selector",
                                "support05_count",
                                "support15_count",
                                "xmatch_count",
                                "row_xmatch_counts",
                                "salt_mod4_pattern",
                                "salt_mod4_residue_counts",
                                "label_measured_below_rho",
                                "label_measured_ops_over_rho",
                                "label_derived_secret",
                            )
                        }
                        for record in selected[:24]
                    ],
                }
            )
    scored.sort(key=lambda item: tuple(item["score"]))
    output = {
        "schema": "ecdlp_leaf79_public_activation_rule_miner_v1",
        "method": "label_ranked_case_level_public_activation_search",
        "training_artifacts": [str(path) for path in args.training_artifact],
        "training_case_count": len(records),
        "positive_training_case_count": sum(1 for record in records if record["label_measured_below_rho"]),
        "candidate_atom_count": len(atoms),
        "candidate_atoms": atoms,
        "frozen_activation_rule": scored[0]["activation_rule"] if scored else None,
        "top_rules": scored[: int(args.top_rules)],
        "interpretation": (
            "Activation rules are ranked with training verifier labels but are "
            "restricted to public case/x-match metadata.  The frozen rule must "
            "be chosen before inspecting the next holdout's verifier outcomes."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"frozen_activation_rule": output["frozen_activation_rule"], "top_rules": output["top_rules"][:3]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
