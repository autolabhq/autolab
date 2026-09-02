#!/usr/bin/env python3
"""Mine public sparsity rules for Sage full-remainder FFE surfaces.

The Sage factor probes expose ``surfaces[]`` records, not the older
``cases[]``/xmatch format.  This miner keeps the rule stages separate so a
frozen public selector is not accidentally justified with post-materialization
signals such as full-remainder monomial count.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MODULI = (2, 3, 4, 5, 6, 8, 12, 16, 24, 32)
SMALL_REMAINDER_CUTS = (79, 92, 106, 121, 137, 154, 172, 191, 232, 254)
SOURCE_OPS_CUTS = (0.5, 0.75, 1.0)
BANK_OPS_CUTS = (0.4, 0.5, 0.75, 1.0)
REFERENCE_DISTANCE_CUTS = (0, 1, 2, 4, 8, 16, 32)
LEAF_SHAPE_CUTS = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80)
FACTOR_ROOT_SCAN_OPS_OVER_RHO_CUTS = (0.46, 0.5, 0.55, 0.75, 1.0)
LABEL_DEFINITIONS = {
    "full_remainder_below_rho": "best_preserving_candidate.remainder_ffe_beats_rho == true",
    "factor_root_scan_below_rho": "best_preserving_candidate.factor_root_scan_ops_over_rho < 1",
    "surface_ffe_below_rho": "best_preserving_candidate.surface_ffe_ops_over_rho < 1",
}
RULE_MODE_ALIASES = {
    "pre": "pre_materialization",
    "pre_materialization": "pre_materialization",
    "factor": "factor_stage",
    "factor_stage": "factor_stage",
    "pre_factor": "pre_factor_stage",
    "pre_factor_stage": "pre_factor_stage",
    "diagnostic": "diagnostic_post_materialization",
    "diagnostic_post_materialization": "diagnostic_post_materialization",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_artifact(raw: str) -> tuple[str, Path]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("--artifact must be label:path")
    label, path_text = raw.split(":", 1)
    if not label:
        raise argparse.ArgumentTypeError("artifact label cannot be empty")
    return label, Path(path_text)


def parse_regex(raw: str) -> re.Pattern[str]:
    try:
        return re.compile(raw)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"invalid regex {raw!r}: {exc}") from exc


def parse_row_salt(row_key: str) -> int | None:
    match = re.search(r":salt(\d+)$", row_key)
    return int(match.group(1)) if match else None


def parse_transfer_index(surface_id: str) -> int | None:
    match = re.search(r"shared-transfer:(\d+):", surface_id)
    return int(match.group(1)) if match else None


def parse_candidate_index(candidate_name: str | None) -> int | None:
    if not candidate_name:
        return None
    match = re.search(r"sage_resultant_factor_(\d+)$", candidate_name)
    return int(match.group(1)) if match else None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def below_rho(value: float | None) -> bool:
    return value is not None and value < 1.0


def load_bank_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = load_json(path)
    return {
        str(row.get("row_key")): row
        for row in data.get("bank_rows") or []
        if isinstance(row, dict) and row.get("row_key") is not None
    }


def parse_row_components(row_key: str) -> dict[str, Any]:
    match = re.match(r"^(?P<target>.+):(?P<mode>[^:]+):(?P<count>\d+):salt(?P<salt>\d+)$", row_key)
    if not match:
        return {}
    return {
        "row_schedule_mode": match.group("mode"),
        "row_schedule_count": int(match.group("count")),
        "row_schedule_salt": int(match.group("salt")),
    }


def load_schedule_rows(path: Path | None) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    if path is None:
        return {}
    data = load_json(path)
    rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in data.get("results") or []:
        if not isinstance(row, dict):
            continue
        target = row.get("target")
        mode = row.get("row_schedule_mode")
        count = as_int(row.get("row_schedule_count"))
        salt = as_int(row.get("row_schedule_salt"))
        if target is None or mode is None or count is None or salt is None:
            continue
        rows[(str(target), str(mode), count, salt)] = row
    return rows


def public_mod_atoms(prefix: str, value: int | None) -> set[str]:
    if value is None:
        return set()
    return {f"{prefix}_mod{modulus}={value % modulus}" for modulus in MODULI}


def public_cut_atoms(prefix: str, value: int | None, cuts: tuple[int, ...]) -> set[str]:
    if value is None:
        return set()
    return {
        atom
        for cut in cuts
        for atom in (
            f"{prefix}<={cut}:{value <= cut}",
            f"{prefix}>={cut}:{value >= cut}",
        )
    }


def leaf_signature(leaves: list[Any]) -> str:
    return ",".join(str(int(leaf)) for leaf in sorted({int(leaf) for leaf in leaves}))


def int_values(values: list[Any]) -> list[int]:
    output: list[int] = []
    for value in values:
        parsed = as_int(value)
        if parsed is not None:
            output.append(parsed)
    return output


def leaf_shape(leaves: list[Any]) -> dict[str, Any]:
    values = sorted(set(int_values(leaves)))
    gaps = [right - left for left, right in zip(values, values[1:])]
    return {
        "selected_leaf_min": min(values) if values else None,
        "selected_leaf_max": max(values) if values else None,
        "selected_leaf_span": (max(values) - min(values)) if values else None,
        "selected_leaf_sum": sum(values) if values else None,
        "selected_leaf_gap_signature": ",".join(str(gap) for gap in gaps),
        "selected_leaf_gap_min": min(gaps) if gaps else None,
        "selected_leaf_gap_max": max(gaps) if gaps else None,
        "selected_leaf_parity_signature": ",".join(str(leaf % 2) for leaf in values),
    }


def label_kind(label: str) -> str:
    return label.split("_", 1)[0] if label else ""


def artifact_stem(path_text: str) -> str:
    if not path_text:
        return ""
    return Path(path_text).stem


def selector_details(selector: str) -> dict[str, Any]:
    seed_match = re.search(r"validation-(\d+)", selector)
    family_match = re.search(r"(hash\d+:\d+:sketch_hit_row_count>=\d+)", selector)
    hash_match = re.search(r"hash(\d+):(\d+):sketch_hit_row_count>=(\d+)", selector)
    seed_hits_match = re.search(r"schedule_scoped_seed_hits>=(\d+)", selector)
    return {
        "family": family_match.group(1) if family_match else None,
        "hash_bits": int(hash_match.group(1)) if hash_match else None,
        "hash_width": int(hash_match.group(2)) if hash_match else None,
        "hit_threshold": int(hash_match.group(3)) if hash_match else None,
        "seed_hit_threshold": int(seed_hits_match.group(1)) if seed_hits_match else None,
        "validation_seed_id": int(seed_match.group(1)) if seed_match else None,
    }


def feature_record(
    surface: dict[str, Any],
    label: str,
    path: Path,
    bank_rows: dict[str, dict[str, Any]],
    schedule_rows: dict[tuple[str, str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    candidate = surface.get("best_preserving_candidate") or {}
    exact_profile = surface.get("exact_profile") or {}
    source_cases = surface.get("source_cases") or []
    first_source_case = source_cases[0] if source_cases and isinstance(source_cases[0], dict) else {}
    row_key = str(surface.get("row_key") or "")
    target = str(surface.get("target") or "")
    surface_id = str(surface.get("surface_id") or "")
    transfer_index = parse_transfer_index(surface_id)
    row_salt = parse_row_salt(row_key)
    row_components = parse_row_components(row_key)
    bank_row = bank_rows.get(row_key) or {}
    schedule_row = schedule_rows.get(
        (
            target,
            str(row_components.get("row_schedule_mode")),
            int(row_components.get("row_schedule_count") or -1),
            int(row_components.get("row_schedule_salt") or -1),
        )
    ) or {}
    bank_source_window_labels = sorted(
        {
            str(window.get("window_label"))
            for window in bank_row.get("source_windows") or []
            if isinstance(window, dict) and window.get("window_label") is not None
        }
    )
    bank_source_selectors = sorted(
        {
            str(window.get("source_selector"))
            for window in bank_row.get("source_windows") or []
            if isinstance(window, dict) and window.get("source_selector") is not None
        }
    )
    bank_window_starts = sorted(
        {
            int(start)
            for window in bank_row.get("source_windows") or []
            if isinstance(window, dict)
            for start in [as_int(window.get("salt_start"))]
            if start is not None
        }
    )
    bank_window_offsets = sorted(
        {
            row_salt - start
            for start in bank_window_starts
            if row_salt is not None
        }
    )
    bank_source_window_label_kinds = sorted({label_kind(label) for label in bank_source_window_labels})
    bank_source_window_guards = sorted(
        {
            str(window.get("guard"))
            for window in bank_row.get("source_windows") or []
            if isinstance(window, dict) and window.get("guard") is not None
        }
    )
    bank_source_window_artifact_stems = sorted(
        {
            artifact_stem(str(window.get("source_artifact") or ""))
            for window in bank_row.get("source_windows") or []
            if isinstance(window, dict) and window.get("source_artifact")
        }
    )
    bank_source_positive_guarded_selector_count = sum(
        1
        for window in bank_row.get("source_windows") or []
        if isinstance(window, dict) and bool(window.get("positive_guarded_selector"))
    )
    source_selector_details = [
        selector_details(selector) for selector in bank_row.get("source_selectors") or []
    ]
    bank_source_selector_families = sorted(
        {str(detail["family"]) for detail in source_selector_details if detail["family"]}
    )
    bank_source_hash_bits = sorted(
        {int(detail["hash_bits"]) for detail in source_selector_details if detail["hash_bits"] is not None}
    )
    bank_source_hash_widths = sorted(
        {int(detail["hash_width"]) for detail in source_selector_details if detail["hash_width"] is not None}
    )
    bank_source_hit_thresholds = sorted(
        {
            int(detail["hit_threshold"])
            for detail in source_selector_details
            if detail["hit_threshold"] is not None
        }
    )
    bank_source_seed_hit_thresholds = sorted(
        {
            int(detail["seed_hit_threshold"])
            for detail in source_selector_details
            if detail["seed_hit_threshold"] is not None
        }
    )
    bank_source_validation_seed_ids = sorted(
        {
            int(detail["validation_seed_id"])
            for detail in source_selector_details
            if detail["validation_seed_id"] is not None
        }
    )
    candidate_name = candidate.get("candidate_name")
    selected_leaf_indices = surface.get("selected_leaf_indices") or []
    leaf_shape_fields = leaf_shape(selected_leaf_indices)
    remainder_ops_over_rho = as_float(candidate.get("remainder_ffe_ops_over_rho"))
    if remainder_ops_over_rho is None:
        remainder_ops_over_rho = as_float(candidate.get("full_remainder_ffe_ops_over_rho"))
    factor_root_scan_ops_over_rho = as_float(candidate.get("factor_root_scan_ops_over_rho"))
    surface_ffe_ops_over_rho = as_float(candidate.get("surface_ffe_ops_over_rho"))
    return {
        "source_label": label,
        "source_path": str(path),
        "surface_id": surface_id,
        "target": target,
        "p": as_int(surface.get("p")),
        "row_key": row_key,
        "row_salt": row_salt,
        "row_schedule_mode": row_components.get("row_schedule_mode"),
        "row_schedule_count": row_components.get("row_schedule_count"),
        "row_schedule_salt": row_components.get("row_schedule_salt"),
        "transfer_index": transfer_index,
        "selected_leaf_indices": selected_leaf_indices,
        "selected_leaf_signature": leaf_signature(selected_leaf_indices),
        "selected_leaf_index_count": len(selected_leaf_indices),
        **leaf_shape_fields,
        "profile_policy": exact_profile.get("policy"),
        "profile_leaf_selector": exact_profile.get("leaf_selector"),
        "profile_top_k": as_int(exact_profile.get("top_k")),
        "source_policy": first_source_case.get("policy"),
        "source_row_selector": first_source_case.get("row_selector"),
        "source_leaf_selector": first_source_case.get("leaf_selector"),
        "source_top_k": as_int(first_source_case.get("top_k")),
        "source_ops_over_rho": as_float(first_source_case.get("source_ops_over_rho")),
        "bank_row_present": bool(bank_row),
        "bank_best_filter_mode": bank_row.get("best_filter_mode"),
        "bank_best_filter_top_k": as_int(bank_row.get("best_filter_top_k")),
        "bank_best_filter_ops_over_rho": as_float(bank_row.get("best_filter_ops_over_rho")),
        "bank_selected_relation_count": as_int(bank_row.get("selected_relation_count")),
        "bank_surface_hit_row_count": as_int(bank_row.get("surface_hit_row_count")),
        "bank_source_window_labels": bank_source_window_labels,
        "bank_source_window_label_kinds": bank_source_window_label_kinds,
        "bank_source_window_starts": bank_window_starts,
        "bank_source_window_row_salt_offsets": bank_window_offsets,
        "bank_source_window_contains_row_salt": any(
            row_salt is not None and start <= row_salt < start + int(window.get("salt_count") or 0)
            for window in bank_row.get("source_windows") or []
            if isinstance(window, dict)
            for start in [as_int(window.get("salt_start"))]
            if start is not None
        ),
        "bank_source_window_guards": bank_source_window_guards,
        "bank_source_window_artifact_stems": bank_source_window_artifact_stems,
        "bank_source_positive_guarded_selector_count": bank_source_positive_guarded_selector_count,
        "bank_source_selectors": bank_source_selectors,
        "bank_source_selector_families": bank_source_selector_families,
        "bank_source_hash_bits": bank_source_hash_bits,
        "bank_source_hash_widths": bank_source_hash_widths,
        "bank_source_hit_thresholds": bank_source_hit_thresholds,
        "bank_source_seed_hit_thresholds": bank_source_seed_hit_thresholds,
        "bank_source_validation_seed_ids": bank_source_validation_seed_ids,
        "schedule_row_present": bool(schedule_row),
        "schedule_best_filter_mode": schedule_row.get("best_filter_mode"),
        "schedule_best_filter_top_k": as_int(schedule_row.get("best_filter_top_k")),
        "schedule_best_filter_ops_over_rho": as_float(schedule_row.get("best_filter_ops_over_rho")),
        "original_selected_root_pair_count": as_int(surface.get("original_selected_root_pair_count")),
        "source_case_count": as_int(surface.get("source_case_count")),
        "candidate_count": as_int(surface.get("candidate_count")),
        "preserving_candidate_count": as_int(surface.get("preserving_candidate_count")),
        "is_preserving_candidate": bool(candidate),
        "candidate_name": candidate_name,
        "candidate_index": parse_candidate_index(candidate_name),
        "selected_leaf_count": as_int(candidate.get("selected_leaf_count")),
        "known_hit_root_count": as_int(candidate.get("known_hit_root_count")),
        "factor_root_scan_ops": as_int(candidate.get("factor_root_scan_ops")),
        "surface_monomials": as_int(candidate.get("surface_monomials")),
        "factor_total_degree": as_int(candidate.get("factor_total_degree")),
        "factor_monomials": as_int(candidate.get("factor_monomials")),
        "factor_root_scan_ops_over_rho": factor_root_scan_ops_over_rho,
        "surface_ffe_ops_over_rho": surface_ffe_ops_over_rho,
        "full_remainder_monomials": as_int(candidate.get("full_remainder_monomials")),
        "full_resultant_monomials": as_int(candidate.get("full_resultant_monomials")),
        "remainder_ffe_ops_over_rho": remainder_ops_over_rho,
        "remainder_ffe_beats_rho": bool(candidate.get("remainder_ffe_beats_rho")),
        "factor_root_scan_beats_rho": below_rho(factor_root_scan_ops_over_rho),
        "surface_ffe_beats_rho": below_rho(surface_ffe_ops_over_rho),
        "surface_level_full_remainder_ffe_ops_over_rho": as_float(surface.get("full_remainder_ffe_ops_over_rho")),
        "is_79_profile": as_int(candidate.get("full_remainder_monomials")) == 79,
    }


def label_for_record(record: dict[str, Any], label_mode: str) -> bool:
    if label_mode == "full_remainder_below_rho":
        return bool(record["remainder_ffe_beats_rho"])
    if label_mode == "factor_root_scan_below_rho":
        return below_rho(record["factor_root_scan_ops_over_rho"])
    if label_mode == "surface_ffe_below_rho":
        return below_rho(record["surface_ffe_ops_over_rho"])
    raise ValueError(f"unknown label mode: {label_mode}")


def load_records(
    artifacts: list[tuple[str, Path]],
    bank_rows: dict[str, dict[str, Any]],
    schedule_rows: dict[tuple[str, str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, path in artifacts:
        artifact = load_json(path)
        for surface in artifact.get("surfaces") or []:
            records.append(feature_record(surface, label, path, bank_rows, schedule_rows))
    return records


def reference_distance_atoms(prefix: str, value: int | None, references: list[int]) -> set[str]:
    if value is None:
        return set()
    atoms: set[str] = set()
    for reference in references:
        delta = value - reference
        abs_delta = abs(delta)
        atoms.add(f"{prefix}_delta_from_{reference}={delta}")
        for cut in REFERENCE_DISTANCE_CUTS:
            atoms.add(f"{prefix}_abs_delta_from_{reference}<={cut}:{abs_delta <= cut}")
        if delta < 0:
            atoms.add(f"{prefix}_side_of_{reference}=below")
        elif delta > 0:
            atoms.add(f"{prefix}_side_of_{reference}=above")
        else:
            atoms.add(f"{prefix}_side_of_{reference}=equal")
    return atoms


def atoms_for_record(
    record: dict[str, Any],
    mode: str,
    reference_transfers: list[int] | None = None,
    reference_salts: list[int] | None = None,
) -> set[str]:
    atoms: set[str] = set()
    reference_transfers = reference_transfers or []
    reference_salts = reference_salts or []
    target = record["target"]
    row_salt = record["row_salt"]
    transfer = record["transfer_index"]

    if mode == "pre_materialization":
        atoms.add(f"target={target}")
        if record["p"] is not None:
            atoms.add(f"p={record['p']}")
        if row_salt is not None:
            atoms.add(f"row_salt={row_salt}")
            atoms.add(f"target_row_salt={target}|{row_salt}")
            atoms.update(public_mod_atoms("row_salt", row_salt))
            atoms.update(reference_distance_atoms("row_salt", row_salt, reference_salts))
        for key in ("row_schedule_mode", "row_schedule_count", "row_schedule_salt"):
            if record.get(key) is not None:
                atoms.add(f"{key}={record[key]}")
        if transfer is not None:
            atoms.update(public_mod_atoms("transfer", transfer))
            atoms.update(reference_distance_atoms("transfer", transfer, reference_transfers))
        if row_salt is not None and transfer is not None:
            for modulus in MODULI:
                atoms.add(f"row_salt_transfer_mod{modulus}={row_salt}|{transfer % modulus}")
                atoms.add(f"target_transfer_mod{modulus}={target}|{transfer % modulus}")
                atoms.add(f"target_row_salt_transfer_mod{modulus}={target}|{row_salt}|{transfer % modulus}")
        for key in ("selected_leaf_index_count", "original_selected_root_pair_count", "source_case_count"):
            if record[key] is not None:
                atoms.add(f"{key}={record[key]}")
        for key in (
            "selected_leaf_min",
            "selected_leaf_max",
            "selected_leaf_span",
            "selected_leaf_sum",
            "selected_leaf_gap_min",
            "selected_leaf_gap_max",
        ):
            if record[key] is not None:
                atoms.add(f"{key}={record[key]}")
                atoms.update(public_cut_atoms(key, record[key], LEAF_SHAPE_CUTS))
        if record["selected_leaf_gap_signature"]:
            atoms.add(f"selected_leaf_gap_signature={record['selected_leaf_gap_signature']}")
        if record["selected_leaf_parity_signature"]:
            atoms.add(f"selected_leaf_parity_signature={record['selected_leaf_parity_signature']}")
        if record["selected_leaf_sum"] is not None:
            atoms.update(public_mod_atoms("selected_leaf_sum", record["selected_leaf_sum"]))
        if record["selected_leaf_signature"]:
            atoms.add(f"selected_leaf_signature={record['selected_leaf_signature']}")
        for leaf in record["selected_leaf_indices"]:
            atoms.add(f"selected_leaf_has={int(leaf)}")
        for key in (
            "profile_policy",
            "profile_leaf_selector",
            "source_policy",
            "source_row_selector",
            "source_leaf_selector",
        ):
            if record.get(key):
                atoms.add(f"{key}={record[key]}")
        for key in ("profile_top_k", "source_top_k"):
            if record[key] is not None:
                atoms.add(f"{key}={record[key]}")
        source_ops = record["source_ops_over_rho"]
        if source_ops is not None:
            for cut in SOURCE_OPS_CUTS:
                atoms.add(f"source_ops_over_rho<={cut}:{source_ops <= cut}")
        atoms.add(f"bank_row_present={record['bank_row_present']}")
        if record["bank_row_present"]:
            for key in ("bank_best_filter_mode",):
                if record.get(key):
                    atoms.add(f"{key}={record[key]}")
            for key in (
                "bank_best_filter_top_k",
                "bank_selected_relation_count",
                "bank_surface_hit_row_count",
            ):
                if record[key] is not None:
                    atoms.add(f"{key}={record[key]}")
            bank_ops = record["bank_best_filter_ops_over_rho"]
            if bank_ops is not None:
                for cut in BANK_OPS_CUTS:
                    atoms.add(f"bank_best_filter_ops_over_rho<={cut}:{bank_ops <= cut}")
            for label in record["bank_source_window_labels"]:
                atoms.add(f"bank_source_window_label={label}")
            for label_kind_value in record["bank_source_window_label_kinds"]:
                atoms.add(f"bank_source_window_label_kind={label_kind_value}")
            for start in record["bank_source_window_starts"]:
                atoms.add(f"bank_source_window_start={int(start)}")
                atoms.update(public_mod_atoms("bank_source_window_start", int(start)))
            for offset in record["bank_source_window_row_salt_offsets"]:
                atoms.add(f"bank_source_window_row_salt_offset={int(offset)}")
            atoms.add(
                f"bank_source_window_contains_row_salt={record['bank_source_window_contains_row_salt']}"
            )
            for guard in record["bank_source_window_guards"]:
                atoms.add(f"bank_source_window_guard={guard}")
            for stem in record["bank_source_window_artifact_stems"]:
                atoms.add(f"bank_source_window_artifact_stem={stem}")
            atoms.add(
                "bank_source_positive_guarded_selector_count="
                f"{record['bank_source_positive_guarded_selector_count']}"
            )
            for selector in record["bank_source_selectors"]:
                atoms.add(f"bank_source_selector={selector}")
            for family in record["bank_source_selector_families"]:
                atoms.add(f"bank_source_selector_family={family}")
            for key in (
                "bank_source_hash_bits",
                "bank_source_hash_widths",
                "bank_source_hit_thresholds",
                "bank_source_seed_hit_thresholds",
            ):
                for value in record[key]:
                    atoms.add(f"{key}={int(value)}")
            for seed_id in record["bank_source_validation_seed_ids"]:
                atoms.add(f"bank_source_validation_seed_id={int(seed_id)}")
                atoms.update(public_mod_atoms("bank_source_validation_seed_id", int(seed_id)))
        atoms.add(f"schedule_row_present={record['schedule_row_present']}")
        if record["schedule_row_present"]:
            if record["schedule_best_filter_mode"]:
                atoms.add(f"schedule_best_filter_mode={record['schedule_best_filter_mode']}")
            if record["schedule_best_filter_top_k"] is not None:
                atoms.add(f"schedule_best_filter_top_k={record['schedule_best_filter_top_k']}")
            schedule_ops = record["schedule_best_filter_ops_over_rho"]
            if schedule_ops is not None:
                for cut in BANK_OPS_CUTS:
                    atoms.add(f"schedule_best_filter_ops_over_rho<={cut}:{schedule_ops <= cut}")
    elif mode == "factor_stage":
        for key in (
            "candidate_count",
            "preserving_candidate_count",
            "candidate_index",
            "selected_leaf_count",
            "factor_root_scan_ops",
            "surface_monomials",
            "factor_total_degree",
            "factor_monomials",
        ):
            if record[key] is not None:
                atoms.add(f"{key}={record[key]}")
        factor_root_scan_ops_over_rho = record["factor_root_scan_ops_over_rho"]
        if factor_root_scan_ops_over_rho is not None:
            for cut in FACTOR_ROOT_SCAN_OPS_OVER_RHO_CUTS:
                atoms.add(
                    "factor_root_scan_ops_over_rho<="
                    f"{cut}:{factor_root_scan_ops_over_rho <= cut}"
                )
        if record["is_preserving_candidate"]:
            atoms.add("has_preserving_candidate=true")
        else:
            atoms.add("has_preserving_candidate=false")
    elif mode == "pre_factor_stage":
        atoms.update(
            atoms_for_record(
                record,
                "pre_materialization",
                reference_transfers,
                reference_salts,
            )
        )
        atoms.update(atoms_for_record(record, "factor_stage", reference_transfers, reference_salts))
    elif mode == "diagnostic_post_materialization":
        for key in ("known_hit_root_count", "full_remainder_monomials", "full_resultant_monomials"):
            if record[key] is not None:
                atoms.add(f"{key}={record[key]}")
        for cut in SMALL_REMAINDER_CUTS:
            monomials = record["full_remainder_monomials"]
            if monomials is not None:
                atoms.add(f"full_remainder_monomials<={cut}:{monomials <= cut}")
        for key in (
            "factor_root_scan_ops_over_rho",
            "surface_ffe_ops_over_rho",
            "remainder_ffe_ops_over_rho",
        ):
            value = record[key]
            if value is not None:
                atoms.add(f"{key}<1:{value < 1.0}")
        atoms.add(f"is_79_profile={record['is_79_profile']}")
    else:
        raise ValueError(f"unknown atom mode: {mode}")
    return atoms


def filter_atoms(
    atoms: set[str],
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
) -> set[str]:
    if not forbidden_atom_patterns:
        return atoms
    return {
        atom
        for atom in atoms
        if not any(pattern.search(atom) for pattern in forbidden_atom_patterns)
    }


def mined_atoms_for_record(
    record: dict[str, Any],
    mode: str,
    reference_transfers: list[int],
    reference_salts: list[int],
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
) -> set[str]:
    return filter_atoms(
        atoms_for_record(record, mode, reference_transfers, reference_salts),
        forbidden_atom_patterns,
    )


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source_label",
        "target",
        "row_key",
        "row_salt",
        "row_schedule_mode",
        "row_schedule_count",
        "row_schedule_salt",
        "transfer_index",
        "selected_leaf_indices",
        "selected_leaf_signature",
        "selected_leaf_index_count",
        "selected_leaf_min",
        "selected_leaf_max",
        "selected_leaf_span",
        "selected_leaf_sum",
        "selected_leaf_gap_signature",
        "selected_leaf_gap_min",
        "selected_leaf_gap_max",
        "selected_leaf_parity_signature",
        "profile_policy",
        "profile_leaf_selector",
        "profile_top_k",
        "source_row_selector",
        "source_ops_over_rho",
        "bank_row_present",
        "bank_best_filter_mode",
        "bank_best_filter_top_k",
        "bank_best_filter_ops_over_rho",
        "bank_selected_relation_count",
        "bank_surface_hit_row_count",
        "bank_source_window_labels",
        "bank_source_window_label_kinds",
        "bank_source_window_starts",
        "bank_source_window_row_salt_offsets",
        "bank_source_window_contains_row_salt",
        "bank_source_window_guards",
        "bank_source_window_artifact_stems",
        "bank_source_positive_guarded_selector_count",
        "bank_source_selectors",
        "bank_source_selector_families",
        "bank_source_hash_bits",
        "bank_source_hash_widths",
        "bank_source_hit_thresholds",
        "bank_source_seed_hit_thresholds",
        "bank_source_validation_seed_ids",
        "schedule_row_present",
        "schedule_best_filter_mode",
        "schedule_best_filter_top_k",
        "schedule_best_filter_ops_over_rho",
        "original_selected_root_pair_count",
        "source_case_count",
        "is_preserving_candidate",
        "candidate_name",
        "candidate_index",
        "selected_leaf_count",
        "known_hit_root_count",
        "factor_root_scan_ops",
        "factor_root_scan_ops_over_rho",
        "factor_root_scan_beats_rho",
        "surface_ffe_ops_over_rho",
        "surface_ffe_beats_rho",
        "full_remainder_monomials",
        "remainder_ffe_ops_over_rho",
        "remainder_ffe_beats_rho",
        "is_79_profile",
    )
    return {key: record[key] for key in keys}


def rule_text(rule: tuple[str, ...]) -> str:
    return "activate:" + "&".join(rule)


def score_rule(
    rule: tuple[str, ...],
    records: list[dict[str, Any]],
    atom_sets: list[set[str]],
    positive_count: int,
    label_mode: str,
) -> dict[str, Any] | None:
    selected_indexes = [
        index for index, atoms in enumerate(atom_sets) if all(atom in atoms for atom in rule)
    ]
    if not selected_indexes:
        return None
    selected = [records[index] for index in selected_indexes]
    selected_positive_count = sum(1 for record in selected if label_for_record(record, label_mode))
    if selected_positive_count == 0:
        return None
    selected_negative_count = len(selected) - selected_positive_count
    selected_sources = sorted({record["source_label"] for record in selected})
    positive_sources = sorted(
        {record["source_label"] for record in selected if label_for_record(record, label_mode)}
    )
    return {
        "activation_rule": rule_text(rule),
        "clause_size": len(rule),
        "selected_record_count": len(selected),
        "selected_positive_count": selected_positive_count,
        "selected_negative_count": selected_negative_count,
        "missed_positive_count": positive_count - selected_positive_count,
        "precision": round(selected_positive_count / len(selected), 6),
        "recall": round(selected_positive_count / positive_count, 6) if positive_count else 0.0,
        "selected_sources": selected_sources,
        "positive_sources": positive_sources,
        "selected_records": [compact_record(record) for record in selected[:32]],
    }


def evaluate_rule(
    rule: tuple[str, ...],
    records: list[dict[str, Any]],
    mode: str,
    reference_transfers: list[int],
    reference_salts: list[int],
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
    label_mode: str = "full_remainder_below_rho",
) -> dict[str, Any]:
    atom_sets = [
        mined_atoms_for_record(
            record,
            mode,
            reference_transfers,
            reference_salts,
            forbidden_atom_patterns,
        )
        for record in records
    ]
    available_atoms = set().union(*atom_sets) if atom_sets else set()
    selected = [
        record for record, atoms in zip(records, atom_sets) if all(atom in atoms for atom in rule)
    ]
    selected_positive_count = sum(1 for record in selected if label_for_record(record, label_mode))
    positive_count = sum(1 for record in records if label_for_record(record, label_mode))
    selected_negative_count = len(selected) - selected_positive_count
    return {
        "activation_rule": rule_text(rule),
        "record_count": len(records),
        "positive_count": positive_count,
        "selected_record_count": len(selected),
        "selected_positive_count": selected_positive_count,
        "selected_negative_count": selected_negative_count,
        "missed_positive_count": positive_count - selected_positive_count,
        "precision": round(selected_positive_count / len(selected), 6) if selected else 0.0,
        "recall": round(selected_positive_count / positive_count, 6) if positive_count else 0.0,
        "unavailable_atoms": sorted(atom for atom in rule if atom not in available_atoms),
        "selected_sources": sorted({record["source_label"] for record in selected}),
        "positive_sources": sorted(
            {record["source_label"] for record in selected if label_for_record(record, label_mode)}
        ),
        "selected_records": [compact_record(record) for record in selected[:32]],
    }


def strip_embedded_record_samples(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_embedded_record_samples(item) for item in value]
    if isinstance(value, dict):
        return {
            key: strip_embedded_record_samples(item)
            for key, item in value.items()
            if key != "selected_records"
        }
    return value


def sort_key(rule: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rule["selected_negative_count"],
        rule["missed_positive_count"],
        -rule["selected_positive_count"],
        rule["selected_record_count"],
        rule["clause_size"],
        rule["activation_rule"],
    )


def mine_rules(
    records: list[dict[str, Any]],
    mode: str,
    max_clause_size: int,
    top_rules: int,
    min_positive_atom_support: int = 1,
    reference_transfers: list[int] | None = None,
    reference_salts: list[int] | None = None,
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
    label_mode: str = "full_remainder_below_rho",
) -> list[dict[str, Any]]:
    reference_transfers = reference_transfers or []
    reference_salts = reference_salts or []
    atom_sets = [
        mined_atoms_for_record(
            record,
            mode,
            reference_transfers,
            reference_salts,
            forbidden_atom_patterns,
        )
        for record in records
    ]
    positive_count = sum(1 for record in records if label_for_record(record, label_mode))
    positive_atom_counts = Counter(
        atom
        for atoms, record in zip(atom_sets, records)
        if label_for_record(record, label_mode)
        for atom in atoms
    )
    candidate_rules: set[tuple[str, ...]] = set()
    for atoms, record in zip(atom_sets, records):
        if not label_for_record(record, label_mode):
            continue
        sorted_atoms = sorted(
            atom for atom in atoms if positive_atom_counts[atom] >= min_positive_atom_support
        )
        for size in range(1, max_clause_size + 1):
            for rule in itertools.combinations(sorted_atoms, size):
                candidate_rules.add(rule)
    scored = [
        score
        for rule in candidate_rules
        if (score := score_rule(rule, records, atom_sets, positive_count, label_mode)) is not None
    ]
    scored.sort(key=sort_key)
    return scored[:top_rules]


def parse_rule_text(raw: str) -> tuple[str, ...]:
    if raw.startswith("activate:"):
        raw = raw[len("activate:") :]
    if not raw:
        raise argparse.ArgumentTypeError("rule cannot be empty")
    return tuple(part for part in raw.split("&") if part)


def parse_evaluate_rule(raw: str) -> tuple[str, tuple[str, ...]]:
    for alias, mode in sorted(RULE_MODE_ALIASES.items(), key=lambda item: -len(item[0])):
        prefix = f"{alias}:"
        if raw.startswith(prefix):
            return mode, parse_rule_text(raw[len(prefix) :])
    return "pre_materialization", parse_rule_text(raw)


def parse_evaluate_rule_family(raw: str) -> tuple[str, list[tuple[str, tuple[str, ...]]]]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "rule family must be NAME=CLAUSE||CLAUSE, with optional clause stage prefixes"
        )
    name, body = raw.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("rule family name cannot be empty")
    clauses = [parse_evaluate_rule(part) for part in body.split("||") if part]
    if not clauses:
        raise argparse.ArgumentTypeError("rule family must contain at least one clause")
    return name, clauses


def group_key(record: dict[str, Any], key: str) -> str:
    if key == "transfer_index":
        return str(record.get("transfer_index"))
    if key == "source_label":
        return str(record.get("source_label"))
    if key == "row_salt":
        return str(record.get("row_salt"))
    if key == "source_label_transfer":
        return f"{record.get('source_label')}|{record.get('transfer_index')}"
    raise ValueError(f"unsupported holdout key: {key}")


def holdout_evaluations(
    records: list[dict[str, Any]],
    mode: str,
    holdout_keys: list[str],
    max_clause_size: int,
    top_rules: int,
    min_positive_atom_support: int,
    reference_transfers: list[int],
    reference_salts: list[int],
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
    label_mode: str = "full_remainder_below_rho",
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for key in holdout_keys:
        groups = sorted({group_key(record, key) for record in records})
        entries: list[dict[str, Any]] = []
        for value in groups:
            train_records = [record for record in records if group_key(record, key) != value]
            test_records = [record for record in records if group_key(record, key) == value]
            train_positive_count = sum(1 for record in train_records if label_for_record(record, label_mode))
            test_positive_count = sum(1 for record in test_records if label_for_record(record, label_mode))
            if not train_positive_count or not test_records:
                continue
            train_rules = mine_rules(
                train_records,
                mode,
                max_clause_size,
                top_rules,
                min_positive_atom_support,
                reference_transfers,
                reference_salts,
                forbidden_atom_patterns,
                label_mode,
            )
            evaluated = [
                {
                    "train": rule,
                    "test": evaluate_rule(
                        parse_rule_text(rule["activation_rule"]),
                        test_records,
                        mode,
                        reference_transfers,
                        reference_salts,
                        forbidden_atom_patterns,
                        label_mode,
                    ),
                }
                for rule in train_rules[: min(top_rules, 5)]
            ]
            entries.append(
                {
                    "holdout_key": key,
                    "holdout_value": value,
                    "train_record_count": len(train_records),
                    "train_positive_count": train_positive_count,
                    "test_record_count": len(test_records),
                    "test_positive_count": test_positive_count,
                    "top_train_rules_on_test": evaluated,
                }
            )
        output[key] = entries
    return output


def manual_rule_evaluations(
    records: list[dict[str, Any]],
    rules: list[tuple[str, tuple[str, ...]]],
    reference_transfers: list[int],
    reference_salts: list[int],
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
    label_mode: str = "full_remainder_below_rho",
) -> list[dict[str, Any]]:
    return [
        {
            "mode": mode,
            **evaluate_rule(
                rule,
                records,
                mode,
                reference_transfers,
                reference_salts,
                forbidden_atom_patterns,
                label_mode,
            ),
        }
        for mode, rule in rules
    ]


def evaluate_rule_family(
    name: str,
    clauses: list[tuple[str, tuple[str, ...]]],
    records: list[dict[str, Any]],
    reference_transfers: list[int],
    reference_salts: list[int],
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
    label_mode: str = "full_remainder_below_rho",
) -> dict[str, Any]:
    atom_sets_by_mode = {
        mode: [
            mined_atoms_for_record(
                record,
                mode,
                reference_transfers,
                reference_salts,
                forbidden_atom_patterns,
            )
            for record in records
        ]
        for mode in sorted({mode for mode, _rule in clauses})
    }
    available_atoms_by_mode = {
        mode: set().union(*atom_sets) if atom_sets else set()
        for mode, atom_sets in atom_sets_by_mode.items()
    }
    selected_indexes: list[int] = []
    for index in range(len(records)):
        if any(
            all(atom in atom_sets_by_mode[mode][index] for atom in rule)
            for mode, rule in clauses
        ):
            selected_indexes.append(index)
    selected = [records[index] for index in selected_indexes]
    selected_positive_count = sum(1 for record in selected if label_for_record(record, label_mode))
    positive_count = sum(1 for record in records if label_for_record(record, label_mode))
    selected_negative_count = len(selected) - selected_positive_count
    return {
        "family_name": name,
        "activation_family": " || ".join(
            f"{mode}:{rule_text(rule)}" for mode, rule in clauses
        ),
        "clauses": [
            {
                "mode": mode,
                "activation_rule": rule_text(rule),
                "unavailable_atoms": sorted(
                    atom for atom in rule if atom not in available_atoms_by_mode[mode]
                ),
            }
            for mode, rule in clauses
        ],
        "record_count": len(records),
        "positive_count": positive_count,
        "selected_record_count": len(selected),
        "selected_positive_count": selected_positive_count,
        "selected_negative_count": selected_negative_count,
        "missed_positive_count": positive_count - selected_positive_count,
        "precision": round(selected_positive_count / len(selected), 6) if selected else 0.0,
        "recall": round(selected_positive_count / positive_count, 6) if positive_count else 0.0,
        "selected_sources": sorted({record["source_label"] for record in selected}),
        "positive_sources": sorted(
            {record["source_label"] for record in selected if label_for_record(record, label_mode)}
        ),
        "selected_records": [compact_record(record) for record in selected[:32]],
    }


def manual_rule_family_evaluations(
    records: list[dict[str, Any]],
    families: list[tuple[str, list[tuple[str, tuple[str, ...]]]]],
    reference_transfers: list[int],
    reference_salts: list[int],
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
    label_mode: str = "full_remainder_below_rho",
) -> list[dict[str, Any]]:
    return [
        evaluate_rule_family(
            name,
            clauses,
            records,
            reference_transfers,
            reference_salts,
            forbidden_atom_patterns,
            label_mode,
        )
        for name, clauses in families
    ]


def manual_rule_group_evaluations(
    records: list[dict[str, Any]],
    rules: list[tuple[str, tuple[str, ...]]],
    group_keys: list[str],
    reference_transfers: list[int],
    reference_salts: list[int],
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
    label_mode: str = "full_remainder_below_rho",
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for key in group_keys:
        entries: list[dict[str, Any]] = []
        for rule_index, (mode, rule) in enumerate(rules):
            for value in sorted({group_key(record, key) for record in records}):
                group_records = [
                    record for record in records if group_key(record, key) == value
                ]
                score = evaluate_rule(
                    rule,
                    group_records,
                    mode,
                    reference_transfers,
                    reference_salts,
                    forbidden_atom_patterns,
                    label_mode,
                )
                if score["selected_record_count"] == 0 and score["positive_count"] == 0:
                    continue
                entries.append(
                    {
                        "rule_index": rule_index,
                        "mode": mode,
                        "holdout_key": key,
                        "holdout_value": value,
                        **score,
                    }
                )
        output[key] = entries
    return output


def manual_rule_family_group_evaluations(
    records: list[dict[str, Any]],
    families: list[tuple[str, list[tuple[str, tuple[str, ...]]]]],
    group_keys: list[str],
    reference_transfers: list[int],
    reference_salts: list[int],
    forbidden_atom_patterns: list[re.Pattern[str]] | None = None,
    label_mode: str = "full_remainder_below_rho",
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for key in group_keys:
        entries: list[dict[str, Any]] = []
        for family_index, (name, clauses) in enumerate(families):
            for value in sorted({group_key(record, key) for record in records}):
                group_records = [
                    record for record in records if group_key(record, key) == value
                ]
                score = evaluate_rule_family(
                    name,
                    clauses,
                    group_records,
                    reference_transfers,
                    reference_salts,
                    forbidden_atom_patterns,
                    label_mode,
                )
                if score["selected_record_count"] == 0 and score["positive_count"] == 0:
                    continue
                entries.append(
                    {
                        "family_index": family_index,
                        "holdout_key": key,
                        "holdout_value": value,
                        **score,
                    }
                )
        output[key] = entries
    return output


def summarize(records: list[dict[str, Any]], label_mode: str) -> dict[str, Any]:
    positive_records = [record for record in records if label_for_record(record, label_mode)]
    full_remainder_positive_records = [
        record for record in records if record["remainder_ffe_beats_rho"]
    ]
    root_scan_positive_records = [
        record for record in records if label_for_record(record, "factor_root_scan_below_rho")
    ]
    surface_positive_records = [
        record for record in records if label_for_record(record, "surface_ffe_below_rho")
    ]
    preserving_records = [record for record in records if record["is_preserving_candidate"]]
    negative_preserving = [
        record
        for record in preserving_records
        if not label_for_record(record, label_mode) and record["remainder_ffe_ops_over_rho"] is not None
    ]
    min_negative = min(
        negative_preserving,
        key=lambda record: float(record["remainder_ffe_ops_over_rho"]),
        default=None,
    )
    by_source = Counter(record["source_label"] for record in records)
    positives_by_source = Counter(record["source_label"] for record in positive_records)
    return {
        "record_count": len(records),
        "label_mode": label_mode,
        "label_positive_count": len(positive_records),
        "source_counts": dict(sorted(by_source.items())),
        "preserving_candidate_count": len(preserving_records),
        "nonpreserving_surface_count": len(records) - len(preserving_records),
        "positive_below_rho_count": len(full_remainder_positive_records),
        "factor_root_scan_below_rho_count": len(root_scan_positive_records),
        "surface_ffe_below_rho_count": len(surface_positive_records),
        "positive_79_profile_count": sum(1 for record in positive_records if record["is_79_profile"]),
        "positive_sources": dict(sorted(positives_by_source.items())),
        "positive_records": [compact_record(record) for record in positive_records],
        "minimum_negative_preserving_record": compact_record(min_negative) if min_negative else None,
        "row_salt_counts": dict(sorted(Counter(str(record["row_salt"]) for record in records).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=parse_artifact, action="append", required=True)
    parser.add_argument("--bank-source", type=Path)
    parser.add_argument("--schedule-source", type=Path)
    parser.add_argument("--reference-transfer", type=int, action="append", default=[])
    parser.add_argument("--reference-salt", type=int, action="append", default=[])
    parser.add_argument(
        "--label-mode",
        choices=tuple(LABEL_DEFINITIONS),
        default="full_remainder_below_rho",
        help="Which below-rho predicate to mine as the positive label.",
    )
    parser.add_argument(
        "--forbid-atom-regex",
        type=parse_regex,
        action="append",
        default=[],
        help="Drop mined atoms matching this regex; repeat to audit leakage-prone atom families.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-clause-size", type=int, default=2)
    parser.add_argument("--top-rules", type=int, default=20)
    parser.add_argument(
        "--min-positive-atom-support",
        type=int,
        default=1,
        help="Prune atoms that appear in fewer than this many positive records before making clauses.",
    )
    parser.add_argument(
        "--holdout-key",
        action="append",
        choices=("transfer_index", "source_label", "row_salt", "source_label_transfer"),
        default=[],
        help="Mine pre-materialization rules with one group held out, then score them on that group.",
    )
    parser.add_argument(
        "--evaluate-rule",
        type=parse_evaluate_rule,
        action="append",
        default=[],
        help=(
            "Evaluate a hand-written conjunction. Defaults to pre-materialization atoms; "
            "prefix with pre:, pre_factor:, factor:, or diagnostic: to choose another stage."
        ),
    )
    parser.add_argument(
        "--evaluate-rule-group-key",
        action="append",
        choices=("transfer_index", "source_label", "row_salt", "source_label_transfer"),
        default=[],
        help="For each --evaluate-rule and --evaluate-rule-family, also score per-group slices for this key.",
    )
    parser.add_argument(
        "--evaluate-rule-family",
        type=parse_evaluate_rule_family,
        action="append",
        default=[],
        help=(
            "Evaluate a named OR-family of hand-written conjunctions. Use "
            "NAME=CLAUSE||CLAUSE; each clause accepts the same optional stage prefixes as "
            "--evaluate-rule."
        ),
    )
    parser.add_argument(
        "--skip-mining",
        action="store_true",
        help="Only summarize records and evaluate --evaluate-rule clauses; do not enumerate mined rules.",
    )
    parser.add_argument(
        "--omit-rule-records",
        action="store_true",
        help="Omit embedded selected_records samples from mined rule and holdout blocks.",
    )
    args = parser.parse_args()

    bank_rows = load_bank_rows(args.bank_source)
    schedule_rows = load_schedule_rows(args.schedule_source)
    records = load_records(args.artifact, bank_rows, schedule_rows)
    if args.skip_mining:
        rules = {
            "pre_materialization": [],
            "factor_stage": [],
            "diagnostic_post_materialization": [],
        }
        holdouts: dict[str, list[dict[str, Any]]] = {}
    else:
        rules = {
            mode: mine_rules(
                records,
                mode,
                args.max_clause_size,
                args.top_rules,
                args.min_positive_atom_support,
                args.reference_transfer,
                args.reference_salt,
                args.forbid_atom_regex,
                args.label_mode,
            )
            for mode in (
                "pre_materialization",
                "factor_stage",
                "diagnostic_post_materialization",
            )
        }
        holdouts = holdout_evaluations(
            records,
            "pre_materialization",
            args.holdout_key,
            args.max_clause_size,
            args.top_rules,
            args.min_positive_atom_support,
            args.reference_transfer,
            args.reference_salt,
            args.forbid_atom_regex,
            args.label_mode,
        )
    candidate_rule_evaluations = manual_rule_evaluations(
        records,
        args.evaluate_rule,
        args.reference_transfer,
        args.reference_salt,
        args.forbid_atom_regex,
        args.label_mode,
    )
    candidate_rule_group_evaluations = manual_rule_group_evaluations(
        records,
        args.evaluate_rule,
        args.evaluate_rule_group_key,
        args.reference_transfer,
        args.reference_salt,
        args.forbid_atom_regex,
        args.label_mode,
    )
    candidate_rule_family_evaluations = manual_rule_family_evaluations(
        records,
        args.evaluate_rule_family,
        args.reference_transfer,
        args.reference_salt,
        args.forbid_atom_regex,
        args.label_mode,
    )
    candidate_rule_family_group_evaluations = manual_rule_family_group_evaluations(
        records,
        args.evaluate_rule_family,
        args.evaluate_rule_group_key,
        args.reference_transfer,
        args.reference_salt,
        args.forbid_atom_regex,
        args.label_mode,
    )
    output_rules = strip_embedded_record_samples(rules) if args.omit_rule_records else rules
    output_holdouts = (
        strip_embedded_record_samples(holdouts) if args.omit_rule_records else holdouts
    )
    output_candidate_rules = (
        strip_embedded_record_samples(candidate_rule_evaluations)
        if args.omit_rule_records
        else candidate_rule_evaluations
    )
    output_candidate_rule_groups = (
        strip_embedded_record_samples(candidate_rule_group_evaluations)
        if args.omit_rule_records
        else candidate_rule_group_evaluations
    )
    output_candidate_rule_families = (
        strip_embedded_record_samples(candidate_rule_family_evaluations)
        if args.omit_rule_records
        else candidate_rule_family_evaluations
    )
    output_candidate_rule_family_groups = (
        strip_embedded_record_samples(candidate_rule_family_group_evaluations)
        if args.omit_rule_records
        else candidate_rule_family_group_evaluations
    )
    output = {
        "schema": "ecdlp_full_remainder_public_sparsity_miner_v1",
        "method": "stage_separated_rule_mining_over_sage_surface_artifacts",
        "artifact_inputs": [
            {"label": label, "path": str(path)} for label, path in args.artifact
        ],
        "metadata_inputs": {
            "bank_source": str(args.bank_source) if args.bank_source else None,
            "bank_row_count": len(bank_rows),
            "schedule_source": str(args.schedule_source) if args.schedule_source else None,
            "schedule_row_count": len(schedule_rows),
            "reference_transfers": sorted(set(args.reference_transfer)),
            "reference_salts": sorted(set(args.reference_salt)),
            "forbidden_atom_regexes": [pattern.pattern for pattern in args.forbid_atom_regex],
            "label_mode": args.label_mode,
            "skip_mining": args.skip_mining,
            "evaluated_rules": [
                {"mode": mode, "activation_rule": rule_text(rule)}
                for mode, rule in args.evaluate_rule
            ],
            "evaluated_rule_families": [
                {
                    "family_name": name,
                    "clauses": [
                        {"mode": mode, "activation_rule": rule_text(rule)}
                        for mode, rule in clauses
                    ],
                }
                for name, clauses in args.evaluate_rule_family
            ],
            "evaluated_rule_group_keys": args.evaluate_rule_group_key,
            "omit_rule_records": args.omit_rule_records,
        },
        "label_definition": LABEL_DEFINITIONS[args.label_mode],
        "stage_notes": {
            "pre_materialization": "Uses public target, row salt, transfer residues, selected leaf envelope metadata, optional bank/schedule joins, and optional configured reference-distance atoms after applying any forbidden atom regex filters.",
            "factor_stage": "Uses Sage factorization/best preserving candidate metadata; useful for triage, not a pre-factor selector.",
            "pre_factor_stage": "Unions pre-materialization atoms with factor-stage atoms so two-stage gates can be audited before full-remainder scoring; this is not a pure public preselector.",
            "diagnostic_post_materialization": "Uses full-remainder and root-count metrics; explanatory only, not a frozen public selector.",
        },
        "summary": summarize(records, args.label_mode),
        "best_pre_materialization_rules": output_rules["pre_materialization"],
        "best_factor_stage_rules": output_rules["factor_stage"],
        "best_diagnostic_post_materialization_rules": output_rules[
            "diagnostic_post_materialization"
        ],
        "manual_rule_evaluations": output_candidate_rules,
        "manual_rule_group_evaluations": output_candidate_rule_groups,
        "manual_rule_family_evaluations": output_candidate_rule_families,
        "manual_rule_family_group_evaluations": output_candidate_rule_family_groups,
        "holdout_evaluations": output_holdouts,
        "records": [compact_record(record) for record in records],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(
        "records={records} positives={positives} best_pre={best}".format(
            records=output["summary"]["record_count"],
            positives=output["summary"]["label_positive_count"],
            best=(
                output["best_pre_materialization_rules"][0]["activation_rule"]
                if output["best_pre_materialization_rules"]
                else "none"
            ),
        )
    )
    if output["manual_rule_evaluations"]:
        print(f"manual_rules={len(output['manual_rule_evaluations'])}")
    if output["manual_rule_family_evaluations"]:
        print(f"manual_rule_families={len(output['manual_rule_family_evaluations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
