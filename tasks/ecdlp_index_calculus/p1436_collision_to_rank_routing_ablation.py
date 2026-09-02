#!/usr/bin/env python3
"""Create deterministic inputs for the next focus artifact set.

This utility emits both:
  - routing_ablation.json
  - relation_matrices.json

Those files are required by the first focus candidate in
`p1436_autoresearch_focus_harness.py` (`collision_to_rank_routing_ablation`).
The current collector payload does not yet include edge-level replay streams, so
this script records whether exact matrix replay is currently possible and stores a
safe, deterministic execution plan when missing sources are detected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import p1436_autoresearch_focus_harness as harness


ROUTING_ABLATION_SCHEMA = "ecdlp.p1436_collision_to_rank_routing_ablation.v3"
RELATION_MATRICES_SCHEMA = "ecdlp.p1436_relation_matrices.v3"
COLLISION_RECORD_SCHEMA = "ecdlp.p1436_collision_record.v2"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = WORKTREE_ROOT / "ecdlp_index_calculus_state" / (
    "p1436_large_prime_residual_collision_collector_after_p1435_probe_for_harness_v4_smoke.json"
)
DEFAULT_ROUTING_ABLATION_OUTPUT = (
    WORKTREE_ROOT / "ecdlp_index_calculus_state" / "routing_ablation.json"
)
DEFAULT_RELATION_MATRICES_OUTPUT = (
    WORKTREE_ROOT / "ecdlp_index_calculus_state" / "relation_matrices.json"
)

REPLAY_MODE_EXACT = "exact"
REPLAY_MODE_SYNTHETIC = "synthetic"
REPLAY_MODE_INVALID = "invalid"
REPLAY_MODE_MISSING = "missing"
STATUS_EXACT_REPLAY_COMPILED = "exact_replay_compiled"
STATUS_SYNTHETIC_REPLAY_PLANNED = "synthetic_replay_planned"
STATUS_REPLAY_SOURCE_INCONSISTENT = "replay_source_inconsistent"
STATUS_MISSING_COLLISION_SOURCES = "blocked_missing_collision_sources"

MATRIX_VARIANTS = ("all_edge", "cross_shift_only", "within_shift_only")


def sha256_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_source_metadata(note_url: str) -> dict[str, Any]:
    methodology = harness.build_methodology(note_url)
    tweet = harness.build_tweet_source_payload(methodology)
    return {
        "source_post_url": tweet["tweet_url"],
        "source_post_url_with_query": tweet["tweet_url_with_query"],
        "source_post_id": tweet["tweet_post_id"],
        "source_author": tweet["tweet_author"],
        "source_query": tweet["tweet_query"],
        "tweet_summary": tweet["tweet_summary"],
        "tweet_summary_is_verbatim": tweet["tweet_summary_is_verbatim"],
        "tweet_text_included": tweet["tweet_text_included"],
        "tweet_text": tweet["tweet_text"],
        "tweet_text_sha256": tweet["tweet_text_sha256"],
        "tweet_posted_at": tweet["tweet_posted_at"],
        "tweet_source_title": tweet["tweet_source_title"],
        "tweet_referenced_paper_title": tweet["tweet_referenced_paper_title"],
        "tweet_source_url": tweet["tweet_source_url"],
        "tweet_media_urls": tweet["tweet_media_urls"],
        "tweet_media_types": tweet["tweet_media_types"],
        "tweet_hashtags": tweet["tweet_hashtags"],
        "tweet_has_media": tweet["tweet_has_media"],
        "tweet_media_count": tweet["tweet_media_count"],
        "tweet_intake_status": tweet["tweet_intake_status"],
        "tweet_text_source_note": tweet["tweet_text_source_note"],
    }


def _collect_config_raw_collisions(
    config: dict[str, Any],
) -> tuple[str, list[Any] | None]:
    for key in (
        "collision_records",
        "collision_rows",
        "raw_collisions",
        "raw_rows",
        "residual_events",
    ):
        value = config.get(key)
        if isinstance(value, list):
            return key, value
    return "", None


def modular_rank(rows: list[list[int]], modulus: int) -> int:
    if not rows:
        return 0
    matrix = [[int(value) % modulus for value in row] for row in rows]
    rank = 0
    column_count = len(matrix[0])
    for column in range(column_count):
        pivot = next(
            (
                index
                for index in range(rank, len(matrix))
                if matrix[index][column] % modulus
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inverse = pow(matrix[rank][column], -1, modulus)
        matrix[rank] = [(value * inverse) % modulus for value in matrix[rank]]
        for index in range(len(matrix)):
            if index == rank:
                continue
            scale = matrix[index][column] % modulus
            if scale:
                matrix[index] = [
                    (value - scale * pivot_value) % modulus
                    for value, pivot_value in zip(matrix[index], matrix[rank])
                ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _normalize_collision_record(
    raw: Any,
    *,
    index: int,
    unknowns: int,
    modulus: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return None, [f"record_{index}_not_object"]

    edge_id = raw.get("edge_id")
    left_shift = raw.get("left_shift")
    right_shift = raw.get("right_shift")
    coefficients = raw.get("relation_coefficients")
    rhs = raw.get("relation_rhs")
    relation_admitted = raw.get("relation_admitted")
    if not isinstance(edge_id, (str, int)) or isinstance(edge_id, bool) or str(edge_id) == "":
        errors.append(f"record_{index}_edge_id_invalid")
    if not isinstance(left_shift, int) or isinstance(left_shift, bool):
        errors.append(f"record_{index}_left_shift_invalid")
    if not isinstance(right_shift, int) or isinstance(right_shift, bool):
        errors.append(f"record_{index}_right_shift_invalid")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != unknowns
        or any(not isinstance(value, int) or isinstance(value, bool) for value in coefficients)
    ):
        errors.append(f"record_{index}_relation_coefficients_invalid")
    if not isinstance(rhs, int) or isinstance(rhs, bool):
        errors.append(f"record_{index}_relation_rhs_invalid")
    if not isinstance(relation_admitted, bool):
        errors.append(f"record_{index}_relation_admitted_invalid")
    if raw.get("source_equation_exact") is not True:
        errors.append(f"record_{index}_source_equation_not_exact")
    if raw.get("residual_equality_exact") is not True:
        errors.append(f"record_{index}_residual_equality_not_exact")
    if errors:
        return None, errors

    coefficients_mod = [value % modulus for value in coefficients]
    if relation_admitted != any(coefficients_mod):
        errors.append(f"record_{index}_relation_admission_mismatch")
        return None, errors

    return {
        "schema": COLLISION_RECORD_SCHEMA,
        "edge_id": str(edge_id),
        "left_shift": left_shift,
        "right_shift": right_shift,
        "cross_shift": left_shift != right_shift,
        "relation_coefficients": coefficients_mod,
        "relation_rhs": rhs % modulus,
        "relation_admitted": relation_admitted,
        "source_equation_exact": True,
        "residual_equality_exact": True,
    }, []


def _admitted_relation_rows(records: list[dict[str, Any]]) -> list[list[int]]:
    return [
        [*record["relation_coefficients"], int(record["relation_rhs"])]
        for record in records
        if record["relation_admitted"]
    ]


def validate_collision_source(
    config: dict[str, Any],
    *,
    modulus: int,
) -> dict[str, Any]:
    source_field, raw_records = _collect_config_raw_collisions(config)
    expected_edges = int(config.get("collision_edge_count") or 0)
    unknowns = int(config.get("unknown_factor_count") or 0)
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []

    if not source_field:
        return {
            "source_field": "",
            "source_present": False,
            "source_sha256": "",
            "raw_record_count": 0,
            "records": [],
            "errors": ["collision_records_missing"],
            "exact": False,
        }
    if modulus <= 1:
        errors.append("curve_order_invalid")
    if unknowns <= 0:
        errors.append("unknown_factor_count_invalid")
    if raw_records is None:
        errors.append("collision_records_not_list")
        raw_records = []

    if not errors:
        for index, raw in enumerate(raw_records):
            record, record_errors = _normalize_collision_record(
                raw,
                index=index,
                unknowns=unknowns,
                modulus=modulus,
            )
            errors.extend(record_errors)
            if record is not None:
                normalized.append(record)

    edge_ids = [record["edge_id"] for record in normalized]
    if len(set(edge_ids)) != len(edge_ids):
        errors.append("duplicate_edge_id")
    if len(raw_records) != expected_edges:
        errors.append("collision_edge_count_mismatch")

    cross_shift_count = sum(record["cross_shift"] for record in normalized)
    within_shift_count = len(normalized) - cross_shift_count
    if cross_shift_count != int(config.get("cross_shift_collision_count") or 0):
        errors.append("cross_shift_collision_count_mismatch")
    if within_shift_count != int(config.get("within_shift_collision_count") or 0):
        errors.append("within_shift_collision_count_mismatch")

    rows = _admitted_relation_rows(normalized)
    coefficient_rows = [row[:-1] for row in rows]
    relation_rank = modular_rank(coefficient_rows, modulus) if modulus > 1 else 0
    augmented_rank = modular_rank(rows, modulus) if modulus > 1 else 0
    if len(rows) != int(config.get("relation_row_count") or 0):
        errors.append("relation_row_count_mismatch")
    rejected_rows = len(normalized) - len(rows)
    if rejected_rows != int(config.get("duplicate_or_zero_row_count") or 0):
        errors.append("duplicate_or_zero_row_count_mismatch")
    if relation_rank != int(config.get("relation_rank") or 0):
        errors.append("relation_rank_mismatch")
    if augmented_rank != int(config.get("augmented_rank") or 0):
        errors.append("augmented_rank_mismatch")

    serialized = json.dumps(raw_records, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "source_field": source_field,
        "source_present": True,
        "source_sha256": hashlib.sha256(serialized).hexdigest(),
        "raw_record_count": len(raw_records),
        "records": normalized,
        "errors": sorted(set(errors)),
        "exact": not errors,
        "compiled_all_edge_summary": {
            "collision_edge_count": len(normalized),
            "cross_shift_collision_count": cross_shift_count,
            "within_shift_collision_count": within_shift_count,
            "relation_row_count": len(rows),
            "duplicate_or_zero_row_count": rejected_rows,
            "relation_rank": relation_rank,
            "augmented_rank": augmented_rank,
        },
    }


def _determine_replay_mode(
    config: dict[str, Any],
    source_validation: dict[str, Any],
) -> str:
    if source_validation["exact"]:
        return REPLAY_MODE_EXACT
    if source_validation["source_present"]:
        return REPLAY_MODE_INVALID
    has_summary = any(
        key in config
        for key in (
            "collision_edge_count",
            "cross_shift_collision_count",
            "within_shift_collision_count",
            "relation_row_count",
        )
    )
    if has_summary:
        return REPLAY_MODE_SYNTHETIC
    return REPLAY_MODE_MISSING


def matrix_variant_payload(
    config: dict[str, Any],
    variant: str,
    source_validation: dict[str, Any],
    *,
    modulus: int,
) -> dict[str, Any]:
    unknowns = int(config.get("unknown_factor_count") or 0)
    replay_mode = _determine_replay_mode(config, source_validation)
    has_raw = replay_mode == REPLAY_MODE_EXACT
    if has_raw:
        records = source_validation["records"]
        if variant == "cross_shift_only":
            records = [record for record in records if record["cross_shift"]]
        elif variant == "within_shift_only":
            records = [record for record in records if not record["cross_shift"]]
        elif variant != "all_edge":
            records = []
        augmented_rows = _admitted_relation_rows(records)
        coefficient_rows = [row[:-1] for row in augmented_rows]
        relation_rank = modular_rank(coefficient_rows, modulus)
        augmented_rank = modular_rank(augmented_rows, modulus)
        status = STATUS_EXACT_REPLAY_COMPILED
    else:
        records = []
        augmented_rows = []
        coefficient_rows = []
        relation_rank = 0
        augmented_rank = 0
        if variant == "all_edge":
            observed_edges = int(config.get("collision_edge_count") or 0)
        elif variant == "cross_shift_only":
            observed_edges = int(config.get("cross_shift_collision_count") or 0)
        elif variant == "within_shift_only":
            observed_edges = int(config.get("within_shift_collision_count") or 0)
        else:
            observed_edges = 0
        if replay_mode == REPLAY_MODE_INVALID:
            status = STATUS_REPLAY_SOURCE_INCONSISTENT
        elif replay_mode == REPLAY_MODE_SYNTHETIC:
            status = (
                STATUS_REPLAY_SOURCE_INCONSISTENT
                if not observed_edges
                else STATUS_SYNTHETIC_REPLAY_PLANNED
            )
        else:
            status = STATUS_MISSING_COLLISION_SOURCES
    if has_raw:
        observed_edges = len(records)
        relation_rows = len(augmented_rows)
    else:
        relation_rows = int(config.get("relation_row_count") or 0)

    if replay_mode == REPLAY_MODE_INVALID:
        replay_hint = "Reject the supplied collision stream and repair every ABI or summary mismatch."
    elif replay_mode == REPLAY_MODE_SYNTHETIC:
        replay_hint = "Replay if ABI-valid raw collision records become available."
    elif replay_mode == REPLAY_MODE_EXACT:
        replay_hint = "Exact matrix compiled from hash-bound ABI-valid collision records."
    else:
        replay_hint = "Replay is blocked until collision streams are available."

    return {
        "variant": variant,
        "replay_mode": replay_mode,
        "status": status,
        "observed_collision_edges": observed_edges,
        "raw_collision_records_present": source_validation["source_present"],
        "raw_collision_records_exact": has_raw,
        "raw_collision_record_count": source_validation["raw_record_count"],
        "raw_collision_source_sha256": source_validation["source_sha256"],
        "source_validation_errors": source_validation["errors"],
        "relation_row_count": relation_rows,
        "relation_rank_observed": relation_rank if has_raw else int(config.get("relation_rank") or 0),
        "augmented_rank_observed": augmented_rank if has_raw else int(config.get("augmented_rank") or 0),
        "unknown_factor_count": unknowns,
        "max_rank_estimate": min(relation_rows, unknowns) if relation_rows and unknowns else 0,
        "coefficient_rows": coefficient_rows,
        "augmented_rows": augmented_rows,
        "replay_hint": replay_hint,
    }


def iter_full_cells(payload: dict[str, Any]):
    for curve in payload.get("curve_records") or []:
        for policy, prefixes in (curve.get("policies") or {}).items():
            for prefix, cell in prefixes.items():
                if prefix != "full":
                    continue
                yield curve, policy, prefix, cell


def build_routing_ablation(payload: dict[str, Any], note_url: str) -> dict[str, Any]:
    source_metadata = build_source_metadata(note_url)

    payload_records: list[dict[str, Any]] = []
    missing = 0
    invalid = 0
    total = 0

    for curve, policy, prefix, cell in iter_full_cells(payload):
        configs = cell.get("configurations") or {}
        factor_base_size = int(cell.get("factor_base_size_B") or 0)
        modulus = int(curve.get("order") or 0)
        for config_name, config in configs.items():
            total += 1
            source_validation = validate_collision_source(config, modulus=modulus)
            replay_mode = _determine_replay_mode(config, source_validation)
            if replay_mode != REPLAY_MODE_EXACT:
                missing += 1
            if replay_mode == REPLAY_MODE_INVALID:
                invalid += 1
            matrix_variants = [
                matrix_variant_payload(
                    config,
                    variant,
                    source_validation,
                    modulus=modulus,
                )
                for variant in MATRIX_VARIANTS
            ]

            record = {
                "curve": {
                    "split": curve.get("split"),
                    "bits": curve.get("bits"),
                    "seed": curve.get("seed"),
                },
                "policy": policy,
                "prefix": prefix,
                "factor_base_size_B": factor_base_size,
                "configuration": config_name,
                "collision_count": int(config.get("collision_edge_count") or 0),
                "cross_shift_collision_count": int(config.get("cross_shift_collision_count") or 0),
                "within_shift_collision_count": int(config.get("within_shift_collision_count") or 0),
                "relation_row_count": int(config.get("relation_row_count") or 0),
                "relation_rank": int(config.get("relation_rank") or 0),
                "augmented_rank": int(config.get("augmented_rank") or 0),
                "unknown_factor_count": int(config.get("unknown_factor_count") or 0),
                "raw_collision_records_present": source_validation["source_present"],
                "raw_collision_records_exact": replay_mode == REPLAY_MODE_EXACT,
                "raw_collision_source": {
                    "schema": COLLISION_RECORD_SCHEMA,
                    "field": source_validation["source_field"],
                    "present": source_validation["source_present"],
                    "count": source_validation["raw_record_count"],
                    "sha256": source_validation["source_sha256"],
                    "exact": source_validation["exact"],
                    "validation_errors": source_validation["errors"],
                },
                "replay_mode": replay_mode,
                "matrix_variants": matrix_variants,
            }
            payload_records.append(record)

    return {
        "schema": ROUTING_ABLATION_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "note_url": source_metadata["source_post_url_with_query"],
        "source": source_metadata,
        "required_variants": list(MATRIX_VARIANTS),
        "records": payload_records,
        "execution_status": {
            "routing_ablation_records": len(payload_records),
            "records_with_missing_collision_sources": missing,
            "records_missing_exact_inputs": missing,
            "records_with_invalid_collision_sources": invalid,
            "replay_ready_records": max(0, total - missing),
            "status": (
                "ready"
                if missing == 0
                else ("invalid_exact_inputs" if invalid else "missing_exact_inputs")
            ),
            "missing_inputs": (
                [f"collision_records conforming to {COLLISION_RECORD_SCHEMA}"]
                if missing
                else []
            ),
            "next_step": (
                (
                    "Audit the exact all-edge/cross-shift/within-shift matrices against "
                    "charged collection cost and the rho baseline."
                )
                if missing == 0
                else (
                    "Collect collision-level raw records in the original collection path "
                    "and rerun this script to compile exact matrix artifacts. Each record "
                    "needs edge_id, left/right shifts, a full relation coefficient vector "
                    "and RHS, an exact relation-admitted flag, plus exact source-equation "
                    "and residual-equality flags. Synthetic summaries are not durable "
                    "replay evidence."
                )
            ),
        },
    }


def build_relation_matrices(payload: dict[str, Any], note_url: str) -> dict[str, Any]:
    source_metadata = build_source_metadata(note_url)
    matrices = []
    missing_exact = 0
    invalid_exact = 0
    total = 0

    for curve, policy, prefix, cell in iter_full_cells(payload):
        configs = cell.get("configurations") or {}
        modulus = int(curve.get("order") or 0)
        for config_name, config in configs.items():
            total += 1
            source_validation = validate_collision_source(config, modulus=modulus)

            for variant in MATRIX_VARIANTS:
                variant_payload = matrix_variant_payload(
                    config,
                    variant,
                    source_validation,
                    modulus=modulus,
                )
                if variant_payload["replay_mode"] != REPLAY_MODE_EXACT:
                    missing_exact += 1
                if variant_payload["replay_mode"] == REPLAY_MODE_INVALID:
                    invalid_exact += 1
                has_raw = variant_payload["replay_mode"] == REPLAY_MODE_EXACT
                matrices.append(
                    {
                        "curve": {
                            "split": curve.get("split"),
                            "bits": curve.get("bits"),
                            "seed": curve.get("seed"),
                            "order": curve.get("order"),
                        },
                        "policy": policy,
                        "prefix": prefix,
                        "configuration": config_name,
                        "matrix": {
                            "matrix_id": f"{curve.get('seed')}|{policy}|{prefix}|{config_name}|{variant}",
                            "raw_source": {
                                "name": source_validation["source_field"] or "collision_records",
                                "schema": COLLISION_RECORD_SCHEMA,
                                "present": source_validation["source_present"],
                                "exact": has_raw,
                                "count": source_validation["raw_record_count"],
                                "sha256": source_validation["source_sha256"],
                                "validation_errors": source_validation["errors"],
                            },
                            "replay_mode": variant_payload["replay_mode"],
                            "status": variant_payload["status"],
                            "replay_hint": variant_payload["replay_hint"],
                            "edges_observed": variant_payload["observed_collision_edges"],
                            "relation_rows_observed": variant_payload["relation_row_count"],
                            "relation_rank_observed": variant_payload["relation_rank_observed"],
                            "augmented_rank_observed": variant_payload["augmented_rank_observed"],
                            "coefficient_rows": variant_payload["coefficient_rows"],
                            "augmented_rows": variant_payload["augmented_rows"],
                            "modulus": modulus,
                            "compiled_from_exact_records": has_raw,
                        },
                    }
                )

    return {
        "schema": RELATION_MATRICES_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "note_url": source_metadata["source_post_url_with_query"],
        "source": source_metadata,
        "variants": list(MATRIX_VARIANTS),
        "matrices": matrices,
        "execution_status": {
            "matrix_entries": len(matrices),
            "entries_missing_exact_inputs": missing_exact,
            "entries_with_invalid_exact_inputs": invalid_exact,
            "entries_replay_ready": len(matrices) - missing_exact,
            "status": (
                "ready"
                if missing_exact == 0
                else ("invalid_exact_inputs" if invalid_exact else "missing_exact_inputs")
            ),
            "missing_inputs": (
                [f"collision_records conforming to {COLLISION_RECORD_SCHEMA}"]
                if missing_exact
                else []
            ),
            "next_step": (
                (
                    "Audit exact variant ranks and charged collection costs before any "
                    "algorithmic promotion."
                )
                if missing_exact == 0
                else (
                    "Collect config-level collision streams under each policy/prefix and "
                    "rerun this script for exact compiled matrices; synthetic summaries "
                    "and ABI-invalid rows do not support durable rank claims."
                )
            ),
        },
    }


def build_artifacts(payload: dict[str, Any], note_url: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return build_routing_ablation(payload, note_url), build_relation_matrices(payload, note_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--note-url", type=str, default=harness.DEFAULT_NOTE_URL)
    parser.add_argument("--routing-output", type=Path, default=DEFAULT_ROUTING_ABLATION_OUTPUT)
    parser.add_argument("--matrices-output", type=Path, default=DEFAULT_RELATION_MATRICES_OUTPUT)
    parser.add_argument("--source-sha", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = read_json(args.input)

    routing_ablation, relation_matrices = build_artifacts(payload, args.note_url)
    write_json(args.routing_output, routing_ablation)
    write_json(args.matrices_output, relation_matrices)

    source_sha = args.source_sha or sha256_file(args.input)
    print(
        f"routing_ablation={args.routing_output} matrices={args.matrices_output} "
        f"routing_status={routing_ablation['execution_status']['status']} "
        f"relation_status={relation_matrices['execution_status']['status']} "
        f"source_sha={source_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
