#!/usr/bin/env python3
"""Audit stage-split costs for hand-scored full-remainder DNF families.

This is deliberately a charge-model audit, not a speedup proof.  The evaluated
families may use Sage factorization metadata such as preserving candidate count,
so this script reports that timing boundary explicitly and only uses existing
exact-profile artifacts to estimate how much full-remainder follow-up would be
avoided after the factor stage.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import ffe_full_remainder_public_sparsity_miner as miner


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_forbidden(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def load_records_from_evaluation(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = data.get("metadata_inputs") or {}
    artifact_inputs = [
        (str(item["label"]), Path(str(item["path"])))
        for item in data.get("artifact_inputs") or []
    ]
    bank_rows = miner.load_bank_rows(Path(metadata["bank_source"])) if metadata.get("bank_source") else {}
    schedule_rows = (
        miner.load_schedule_rows(Path(metadata["schedule_source"]))
        if metadata.get("schedule_source")
        else {}
    )
    return miner.load_records(artifact_inputs, bank_rows, schedule_rows), metadata


def family_from_metadata(
    data: dict[str, Any],
    requested_name: str | None,
) -> dict[str, Any]:
    families = (data.get("metadata_inputs") or {}).get("evaluated_rule_families") or []
    if requested_name:
        matches = [family for family in families if family.get("family_name") == requested_name]
    else:
        matches = families[:1]
    if not matches:
        raise SystemExit(f"family not found: {requested_name or '<first>'}")
    return matches[0]


def parsed_clauses(family: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    clauses = []
    for clause in family.get("clauses") or []:
        clauses.append(
            (
                str(clause["mode"]),
                miner.parse_rule_text(str(clause["activation_rule"])),
            )
        )
    if not clauses:
        raise SystemExit("family has no clauses")
    return clauses


def matching_clause_indexes(
    record: dict[str, Any],
    clauses: list[tuple[str, tuple[str, ...]]],
    reference_transfers: list[int],
    reference_salts: list[int],
    forbidden_atom_patterns: list[re.Pattern[str]],
) -> list[int]:
    atom_sets_by_mode: dict[str, set[str]] = {}
    matches: list[int] = []
    for index, (mode, rule) in enumerate(clauses):
        if mode not in atom_sets_by_mode:
            atom_sets_by_mode[mode] = miner.mined_atoms_for_record(
                record,
                mode,
                reference_transfers,
                reference_salts,
                forbidden_atom_patterns,
            )
        if all(atom in atom_sets_by_mode[mode] for atom in rule):
            matches.append(index)
    return matches


def sum_field(records: list[dict[str, Any]], field: str) -> float:
    values = [miner.as_float(record.get(field)) for record in records]
    return round(sum(value for value in values if value is not None), 8)


def avg_field(records: list[dict[str, Any]], field: str) -> float | None:
    values = [miner.as_float(record.get(field)) for record in records]
    parsed = [value for value in values if value is not None]
    if not parsed:
        return None
    return round(sum(parsed) / len(parsed), 8)


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 8)


def compact_group(records: list[dict[str, Any]], label_mode: str) -> dict[str, Any]:
    positives = [record for record in records if miner.label_for_record(record, label_mode)]
    return {
        "record_count": len(records),
        "positive_count": len(positives),
        "negative_count": len(records) - len(positives),
        "candidate_remainder_ops_over_rho_sum": sum_field(records, "remainder_ffe_ops_over_rho"),
        "candidate_remainder_ops_over_rho_avg": avg_field(records, "remainder_ffe_ops_over_rho"),
        "factor_root_scan_ops_over_rho_sum": sum_field(records, "factor_root_scan_ops_over_rho"),
        "factor_root_scan_ops_over_rho_avg": avg_field(records, "factor_root_scan_ops_over_rho"),
        "surface_full_remainder_ops_over_rho_sum": sum_field(
            records,
            "surface_level_full_remainder_ffe_ops_over_rho",
        ),
        "surface_full_remainder_ops_over_rho_avg": avg_field(
            records,
            "surface_level_full_remainder_ffe_ops_over_rho",
        ),
    }


def clause_summary(
    records: list[dict[str, Any]],
    labels: list[bool],
    clause_matches: list[list[int]],
    clauses: list[tuple[str, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, (mode, rule) in enumerate(clauses):
        selected = [
            record for record, matches in zip(records, clause_matches) if index in matches
        ]
        selected_labels = [
            label for label, matches in zip(labels, clause_matches) if index in matches
        ]
        output.append(
            {
                "clause_index": index,
                "mode": mode,
                "activation_rule": miner.rule_text(rule),
                "selected_record_count": len(selected),
                "selected_positive_count": sum(1 for label in selected_labels if label),
                "selected_negative_count": sum(1 for label in selected_labels if not label),
                "candidate_remainder_ops_over_rho_sum": sum_field(
                    selected,
                    "remainder_ffe_ops_over_rho",
                ),
                "factor_root_scan_ops_over_rho_sum": sum_field(
                    selected,
                    "factor_root_scan_ops_over_rho",
                ),
            }
        )
    return output


def group_by_source(
    records: list[dict[str, Any]],
    labels: list[bool],
    selected_flags: list[bool],
) -> list[dict[str, Any]]:
    by_source: dict[str, list[tuple[dict[str, Any], bool, bool]]] = defaultdict(list)
    for record, label, selected in zip(records, labels, selected_flags):
        by_source[str(record.get("source_label"))].append((record, label, selected))
    output = []
    for source, rows in sorted(by_source.items()):
        selected_rows = [record for record, _label, selected in rows if selected]
        selected_labels = [label for _record, label, selected in rows if selected]
        labels_for_source = [label for _record, label, _selected in rows]
        if not any(labels_for_source) and not selected_rows:
            continue
        output.append(
            {
                "source_label": source,
                "record_count": len(rows),
                "positive_count": sum(1 for label in labels_for_source if label),
                "selected_record_count": len(selected_rows),
                "selected_positive_count": sum(1 for label in selected_labels if label),
                "selected_negative_count": sum(1 for label in selected_labels if not label),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-source", type=Path, required=True)
    parser.add_argument("--family-name")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = load_json(args.evaluation_source)
    records, metadata = load_records_from_evaluation(data)
    family = family_from_metadata(data, args.family_name)
    clauses = parsed_clauses(family)
    reference_transfers = [int(value) for value in metadata.get("reference_transfers") or []]
    reference_salts = [int(value) for value in metadata.get("reference_salts") or []]
    forbidden_atom_patterns = parse_forbidden(metadata.get("forbidden_atom_regexes") or [])
    label_mode = str(metadata.get("label_mode") or "full_remainder_below_rho")

    clause_matches = [
        matching_clause_indexes(
            record,
            clauses,
            reference_transfers,
            reference_salts,
            forbidden_atom_patterns,
        )
        for record in records
    ]
    selected_flags = [bool(matches) for matches in clause_matches]
    labels = [miner.label_for_record(record, label_mode) for record in records]
    selected = [record for record, flag in zip(records, selected_flags) if flag]
    rejected = [record for record, flag in zip(records, selected_flags) if not flag]
    selected_labels = [label for label, flag in zip(labels, selected_flags) if flag]
    rejected_labels = [label for label, flag in zip(labels, selected_flags) if not flag]

    all_remainder_sum = sum_field(records, "remainder_ffe_ops_over_rho")
    selected_remainder_sum = sum_field(selected, "remainder_ffe_ops_over_rho")
    rejected_remainder_sum = sum_field(rejected, "remainder_ffe_ops_over_rho")
    all_factor_scan_sum = sum_field(records, "factor_root_scan_ops_over_rho")
    factor_scan_plus_selected_remainder = round(
        all_factor_scan_sum + selected_remainder_sum,
        8,
    )
    all_surface_full_sum = sum_field(records, "surface_level_full_remainder_ffe_ops_over_rho")
    selected_surface_full_sum = sum_field(
        selected,
        "surface_level_full_remainder_ffe_ops_over_rho",
    )
    factor_scan_plus_selected_surface_full = round(
        all_factor_scan_sum + selected_surface_full_sum,
        8,
    )

    output = {
        "schema": "ecdlp_full_remainder_stage_split_charge_audit_v1",
        "method": "replay_evaluated_rule_family_and_proxy_charge_full_remainder_followup",
        "evaluation_source": str(args.evaluation_source),
        "family_name": family.get("family_name"),
        "activation_family": " || ".join(
            f"{mode}:{miner.rule_text(rule)}" for mode, rule in clauses
        ),
        "label_mode": label_mode,
        "stage_boundary": {
            "selector_uses_pre_factor_stage": True,
            "selector_requires_exact_surface_materialization": True,
            "selector_requires_sage_factorization": True,
            "selector_uses_preserving_candidate_count": any(
                any("preserving_candidate_count=" in atom for atom in rule)
                for _mode, rule in clauses
            ),
            "selector_uses_factor_root_scan_cost": any(
                any("factor_root_scan_ops_over_rho" in atom for atom in rule)
                for _mode, rule in clauses
            ),
            "sage_factorization_cost_is_charged": False,
            "materialization_cost_is_charged": False,
            "interpretation": (
                "Proxy charge only: the family can reject full-remainder follow-up "
                "after exact materialization and Sage factorization, but this artifact "
                "does not prove a wall-clock or asymptotic speedup."
            ),
        },
        "summary": {
            "record_count": len(records),
            "positive_count": sum(1 for label in labels if label),
            "selected_record_count": len(selected),
            "selected_positive_count": sum(1 for label in selected_labels if label),
            "selected_negative_count": sum(1 for label in selected_labels if not label),
            "rejected_record_count": len(rejected),
            "rejected_positive_count": sum(1 for label in rejected_labels if label),
            "rejected_negative_count": sum(1 for label in rejected_labels if not label),
            "precision": ratio(sum(1 for label in selected_labels if label), len(selected)),
            "recall": ratio(sum(1 for label in selected_labels if label), sum(1 for label in labels if label)),
        },
        "groups": {
            "all_records": compact_group(records, label_mode),
            "selected_records": compact_group(selected, label_mode),
            "rejected_records": compact_group(rejected, label_mode),
        },
        "proxy_charge": {
            "candidate_remainder_all_records_sum": all_remainder_sum,
            "candidate_remainder_selected_only_sum": selected_remainder_sum,
            "candidate_remainder_rejected_sum": rejected_remainder_sum,
            "selected_candidate_remainder_fraction_of_all": ratio(
                selected_remainder_sum,
                all_remainder_sum,
            ),
            "factor_root_scan_all_records_sum": all_factor_scan_sum,
            "factor_scan_plus_selected_candidate_remainder_sum": factor_scan_plus_selected_remainder,
            "factor_scan_plus_selected_candidate_remainder_vs_all_candidate_remainder": ratio(
                factor_scan_plus_selected_remainder,
                all_remainder_sum,
            ),
            "surface_full_remainder_all_records_sum": all_surface_full_sum,
            "surface_full_remainder_selected_only_sum": selected_surface_full_sum,
            "selected_surface_full_remainder_fraction_of_all": ratio(
                selected_surface_full_sum,
                all_surface_full_sum,
            ),
            "factor_scan_plus_selected_surface_full_remainder_sum": (
                factor_scan_plus_selected_surface_full
            ),
            "factor_scan_plus_selected_surface_full_remainder_vs_all_surface_full_remainder": ratio(
                factor_scan_plus_selected_surface_full,
                all_surface_full_sum,
            ),
        },
        "clause_summaries": clause_summary(records, labels, clause_matches, clauses),
        "source_summaries": group_by_source(records, labels, selected_flags),
        "selected_false_positive_records": [
            miner.compact_record(record)
            for record, label, selected_flag in zip(records, labels, selected_flags)
            if selected_flag and not label
        ],
        "missed_positive_records": [
            miner.compact_record(record)
            for record, label, selected_flag in zip(records, labels, selected_flags)
            if label and not selected_flag
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(
        "records={records} selected={selected} positives={positives} selected_pos={selected_pos} selected_neg={selected_neg}".format(
            records=output["summary"]["record_count"],
            selected=output["summary"]["selected_record_count"],
            positives=output["summary"]["positive_count"],
            selected_pos=output["summary"]["selected_positive_count"],
            selected_neg=output["summary"]["selected_negative_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
