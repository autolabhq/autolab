#!/usr/bin/env python3
"""Scout transfer-10376 residual terms 8/12 without confusing leaves for terms.

The selected13 priority-0 gap scanner leaves residual terms 8 and 12 open for
same-target evidence.  Several mounted public-factor/root artifacts contain
same-target `leaf_signature == [8]` rows, but those are leaf IDs, not residual
term IDs.  This scout records those rows as route-generation seeds only and
keeps the explicit residual-term closure count at zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_term812_route_scout.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
MOUNTED_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
DEFAULT_GAP_SCANNER = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_gap_closure_scanner_10376_probe.json"
DEFAULT_SIGNATURE = (
    MOUNTED_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_signature_probe.json"
)
DEFAULT_SIGNATURE_EXTENDED = (
    MOUNTED_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_signature_extended_72_79_probe.json"
)
DEFAULT_ROOT_MAP_RANK = (
    MOUNTED_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_root_map_rank_probe.json"
)
DEFAULT_LOW_RANK_REMAINDER = (
    MOUNTED_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_low_rank_remainder_probe.json"
)
DEFAULT_SURFACE_GATE_DELTA = (
    MOUNTED_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_forward_safe_multiblock_surface_gate_delta_200_231_232_375_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_term812_route_scout_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_term812_route_scout_10376_probe.h"

EXPECTED_TRANSFER = 10376
EXPECTED_OPEN_TERM_COUNT = 2
EXPECTED_EXPLICIT_SAME_TARGET_TERM_CLOSURE_COUNT = 0
EXPECTED_CROSS_TARGET_ANALOGUE_TERM_COUNT = 2
EXPECTED_SIGNATURE_LEAF8_SEED_COUNT = 1
EXPECTED_ROOT_MAP_LEAF8_SEED_COUNT = 1
EXPECTED_LOW_RANK_LEAF8_SEED_COUNT = 1
EXPECTED_SURFACE_GATE_UNIQUE_LEAF8_HIT_COUNT = 4
EXPECTED_SURFACE_GATE_BELOW_RHO_ROUTE_COUNT = 1
EXPECTED_LEAF12_SEED_COUNT = 0
EXPECTED_ACCEPTED_EXPORT_COUNT = 0


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_hash_u64(material: Any) -> int:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def leaf_values(row: dict[str, Any]) -> set[int]:
    values: set[int] = set()
    for key in ("leaf_indices", "unique_leaf_indices", "leaf_signature"):
        raw = row.get(key)
        if isinstance(raw, list):
            values.update(as_int(value, -1) for value in raw)
    for item in row.get("row_leaf_keys") or []:
        if isinstance(item, dict):
            values.update(as_int(value, -1) for value in item.get("leaf_indices") or [])
    return {value for value in values if value >= 0}


def signature_leaf_seed_rows(data: dict[str, Any], source_label: str, source_path: Path, target: str, leaf: int) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[Any, ...]] = set()
    for case in data.get("positive_cases") or []:
        if case.get("target") != target or leaf not in leaf_values(case):
            continue
        key = (
            source_label,
            as_int(case.get("transfer_index"), -1),
            tuple(as_int(value, -1) for value in case.get("row_salts") or []),
            tuple(sorted(leaf_values(case))),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "evidence_kind": "same_target_leaf_signature_seed",
                "leaf_id": leaf,
                "leaf_is_residual_term": False,
                "ops_over_rho": as_float(case.get("ops_over_rho")),
                "rank": as_int(case.get("rank")),
                "relation_count": as_int(case.get("relation_count")),
                "row_leaf_keys": case.get("row_leaf_keys") or [],
                "row_salts": case.get("row_salts") or [],
                "source_label": source_label,
                "source_path": str(source_path),
                "target": target,
                "transfer_index": as_int(case.get("transfer_index"), -1),
                "unique_leaf_indices": sorted(leaf_values(case)),
            }
        )
    return sorted(rows, key=lambda row: (row["transfer_index"], row["row_salts"], row["source_label"]))


def rank_case_leaf_seed_rows(
    data: dict[str, Any], source_label: str, source_path: Path, target: str, leaf: int
) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[Any, ...]] = set()
    for case in data.get("cases") or []:
        if case.get("target") != target or leaf not in leaf_values(case):
            continue
        row_keys = [item.get("row_key") for item in case.get("row_leaf_keys") or [] if isinstance(item, dict)]
        key = (source_label, as_int(case.get("transfer_index"), -1), tuple(row_keys), tuple(sorted(leaf_values(case))))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "evidence_kind": "same_target_leaf_rank_seed",
                "leaf_id": leaf,
                "leaf_is_residual_term": False,
                "public_key_verified": bool(
                    case.get("root_map_union_public_key_verified") or case.get("low_rank_union_public_key_verified")
                ),
                "row_keys": row_keys,
                "row_leaf_keys": case.get("row_leaf_keys") or [],
                "source_label": source_label,
                "source_ops_over_rho": as_float(case.get("source_ops_over_rho")),
                "source_path": str(source_path),
                "target": target,
                "transfer_index": as_int(case.get("transfer_index"), -1),
                "unique_leaf_indices": sorted(leaf_values(case)),
            }
        )
    return sorted(rows, key=lambda row: (row["transfer_index"], row["row_keys"], row["source_label"]))


def surface_gate_leaf_hits(data: dict[str, Any], source_path: Path, target: str, leaf: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hits: dict[tuple[Any, ...], dict[str, Any]] = {}
    routes: dict[tuple[Any, ...], dict[str, Any]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("target") == target and isinstance(value.get("retained_strict_hits"), list):
                route_key = (
                    value.get("block"),
                    value.get("schedule"),
                    as_int(value.get("prefix_size"), -1),
                    as_float(value.get("ops_over_rho")),
                )
                route_hits = []
                for hit in value.get("retained_strict_hits") or []:
                    key = hit.get("key") or []
                    if len(key) != 3 or key[0] != target or leaf not in leaf_values(hit):
                        continue
                    hit_key = (key[0], as_int(key[1], -1), key[2])
                    hits[hit_key] = {
                        "evidence_kind": "same_target_leaf8_public_factor_hit",
                        "evaluated_factor_count": as_int(hit.get("evaluated_factor_count")),
                        "factor_quadratic_root_ops": as_int(hit.get("factor_quadratic_root_ops")),
                        "leaf_id": leaf,
                        "leaf_is_residual_term": False,
                        "leaf_signature": hit.get("leaf_signature") or [],
                        "public_factor_quadratic_root_ops": as_int(hit.get("public_factor_quadratic_root_ops")),
                        "quadratic_root_work": as_int(hit.get("quadratic_root_work")),
                        "row_key": key[2],
                        "selector_eval_ops": as_int(hit.get("selector_eval_ops")),
                        "source_path": str(source_path),
                        "target": key[0],
                        "transfer_index": as_int(key[1], -1),
                        "unit_cost": as_int(hit.get("unit_cost")),
                    }
                    route_hits.append(hit_key)
                if route_hits and value.get("no_extra_root_work_below_rho") is True:
                    routes[route_key] = {
                        "block": value.get("block"),
                        "evidence_kind": "same_target_leaf8_below_rho_route_seed",
                        "leaf_id": leaf,
                        "leaf_is_residual_term": False,
                        "no_extra_root_work_below_rho": True,
                        "ops_over_rho": as_float(value.get("ops_over_rho")),
                        "prefix_size": as_int(value.get("prefix_size")),
                        "retained_hit_keys": [list(item) for item in sorted(set(route_hits))],
                        "rho": as_int(value.get("rho")),
                        "schedule": value.get("schedule"),
                        "target": target,
                        "total_ops": as_int(value.get("total_ops")),
                    }
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    return (
        sorted(hits.values(), key=lambda row: (row["transfer_index"], row["row_key"])),
        sorted(routes.values(), key=lambda row: (row["ops_over_rho"], row["prefix_size"], row["schedule"])),
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    gap_path = Path(args.gap_scanner)
    signature_path = Path(args.signature)
    signature_extended_path = Path(args.signature_extended)
    root_map_path = Path(args.root_map_rank)
    low_rank_path = Path(args.low_rank_remainder)
    surface_gate_path = Path(args.surface_gate_delta)
    gap = load_json(gap_path)
    target = str((gap.get("target_slot") or {}).get("target") or "")
    transfer = as_int((gap.get("target_slot") or {}).get("transfer_index"), -1)
    remaining_terms = [as_int(term, -1) for term in (gap.get("summary") or {}).get("remaining_same_target_terms") or []]
    cross_terms = [as_int(term, -1) for term in (gap.get("summary") or {}).get("cross_target_analogue_terms") or []]
    signature_rows = signature_leaf_seed_rows(load_json(signature_path), "signature", signature_path, target, 8)
    signature_extended_rows = signature_leaf_seed_rows(
        load_json(signature_extended_path), "signature_extended", signature_extended_path, target, 8
    )
    root_map_rows = rank_case_leaf_seed_rows(load_json(root_map_path), "root_map_rank", root_map_path, target, 8)
    low_rank_rows = rank_case_leaf_seed_rows(load_json(low_rank_path), "low_rank_remainder", low_rank_path, target, 8)
    surface_hits, surface_routes = surface_gate_leaf_hits(load_json(surface_gate_path), surface_gate_path, target, 8)
    leaf12_signature_rows = signature_leaf_seed_rows(load_json(signature_path), "signature", signature_path, target, 12)
    failures = []
    if gap.get("claim_status") != "SELECTED13_PRIORITY0_GAP_CLOSURE_SCANNER_READY":
        failures.append({"code": "gap_scanner_not_ready", "claim_status": gap.get("claim_status")})
    if transfer != EXPECTED_TRANSFER:
        failures.append({"code": "transfer_unexpected", "observed": transfer})
    if sorted(remaining_terms) != [8, 12]:
        failures.append({"code": "remaining_terms_unexpected", "observed": remaining_terms})
    if len(remaining_terms) != EXPECTED_OPEN_TERM_COUNT:
        failures.append({"code": "open_term_count_unexpected", "observed": len(remaining_terms)})
    explicit_same_target_term_closures = []
    if len(explicit_same_target_term_closures) != EXPECTED_EXPLICIT_SAME_TARGET_TERM_CLOSURE_COUNT:
        failures.append({"code": "explicit_same_target_term_closure_count_unexpected"})
    if len(cross_terms) != EXPECTED_CROSS_TARGET_ANALOGUE_TERM_COUNT:
        failures.append({"code": "cross_target_analogue_term_count_unexpected", "observed": len(cross_terms)})
    if len(signature_rows) != EXPECTED_SIGNATURE_LEAF8_SEED_COUNT:
        failures.append({"code": "signature_leaf8_seed_count_unexpected", "observed": len(signature_rows)})
    if len(root_map_rows) != EXPECTED_ROOT_MAP_LEAF8_SEED_COUNT:
        failures.append({"code": "root_map_leaf8_seed_count_unexpected", "observed": len(root_map_rows)})
    if len(low_rank_rows) != EXPECTED_LOW_RANK_LEAF8_SEED_COUNT:
        failures.append({"code": "low_rank_leaf8_seed_count_unexpected", "observed": len(low_rank_rows)})
    if len(surface_hits) != EXPECTED_SURFACE_GATE_UNIQUE_LEAF8_HIT_COUNT:
        failures.append({"code": "surface_gate_unique_leaf8_hit_count_unexpected", "observed": len(surface_hits)})
    if len(surface_routes) != EXPECTED_SURFACE_GATE_BELOW_RHO_ROUTE_COUNT:
        failures.append({"code": "surface_gate_below_rho_route_count_unexpected", "observed": len(surface_routes)})
    if len(leaf12_signature_rows) != EXPECTED_LEAF12_SEED_COUNT:
        failures.append({"code": "leaf12_seed_count_unexpected", "observed": len(leaf12_signature_rows)})
    summary = {
        "accepted_relation_export_count": 0,
        "cross_target_analogue_term_count": len(cross_terms),
        "cross_target_analogue_terms": sorted(cross_terms),
        "explicit_same_target_term_closure_count": len(explicit_same_target_term_closures),
        "failure_count": len(failures),
        "leaf12_seed_count": len(leaf12_signature_rows),
        "leaf8_low_rank_seed_count": len(low_rank_rows),
        "leaf8_root_map_seed_count": len(root_map_rows),
        "leaf8_signature_extended_seed_count": len(signature_extended_rows),
        "leaf8_signature_seed_count": len(signature_rows),
        "leaf8_surface_gate_below_rho_route_count": len(surface_routes),
        "leaf8_surface_gate_unique_hit_count": len(surface_hits),
        "native_preflight_verified": False,
        "open_term_count": len(remaining_terms),
        "open_terms": sorted(remaining_terms),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "verified": not failures,
        "worker_interpretation": (
            "Residual terms 8 and 12 still have no same-target residual-term closure. "
            "Same-target leaf-8 public-factor/root routes are useful generation seeds, "
            "but leaf 8 must not be counted as residual term 8."
        ),
    }
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_TERM812_ROUTE_SCOUT_READY"
            if not failures
            else "SELECTED13_PRIORITY0_TERM812_ROUTE_SCOUT_FAILED"
        ),
        "parameters": {
            "gap_scanner": str(gap_path),
            "low_rank_remainder": str(low_rank_path),
            "root_map_rank": str(root_map_path),
            "signature": str(signature_path),
            "signature_extended": str(signature_extended_path),
            "surface_gate_delta": str(surface_gate_path),
        },
        "packet_hash_u64": stable_hash_u64(
            {
                "leaf8_surface_hits": surface_hits,
                "open_terms": remaining_terms,
                "signature_rows": signature_rows,
                "surface_routes": surface_routes,
                "target": target,
                "transfer": transfer,
            }
        ),
        "target_slot": gap.get("target_slot") or {},
        "residual_term_status": [
            {
                "term": 8,
                "same_target_residual_term_closed": False,
                "cross_target_analogue_available": 8 in cross_terms,
                "route_seed_available": bool(signature_rows or root_map_rows or low_rank_rows or surface_hits),
                "next_action": "Use leaf-8 same-target public-factor routes as hit-stream seeds, then require explicit residual-term-8 dependency evidence.",
            },
            {
                "term": 12,
                "same_target_residual_term_closed": False,
                "cross_target_analogue_available": 12 in cross_terms,
                "route_seed_available": False,
                "next_action": "Search for a same-target 22050 residual-term-12 dependency or public-factor route; cross-target 67 evidence is a pattern hint only.",
            },
        ],
        "same_target_leaf8_route_seeds": {
            "signature_rows": signature_rows,
            "signature_extended_rows": signature_extended_rows,
            "root_map_rank_rows": root_map_rows,
            "low_rank_remainder_rows": low_rank_rows,
            "surface_gate_hits": surface_hits,
            "surface_gate_below_rho_routes": surface_routes,
        },
        "same_target_leaf12_route_seeds": leaf12_signature_rows,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "external_kernel_result_emitted": False,
            "general_ecdlp_algorithm_claimed": False,
            "leaf_id_equals_residual_term_id": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "residual_terms_8_12_closed": False,
            "validator_acceptance_ready": False,
        },
        "failures": failures,
        "summary": summary,
    }


def render_c_header(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    target = payload.get("target_slot") or {}
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_TERM812_ROUTE_SCOUT_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_TERM812_ROUTE_SCOUT_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_TERM812_SCOUT_TRANSFER {as_int(target.get('transfer_index'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_OPEN_TERM_COUNT {as_int(summary.get('open_term_count'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_EXPLICIT_SAME_TARGET_TERM_CLOSURES {as_int(summary.get('explicit_same_target_term_closure_count'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_CROSS_TARGET_ANALOGUE_TERMS {as_int(summary.get('cross_target_analogue_term_count'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_SIGNATURE_LEAF8_SEEDS {as_int(summary.get('leaf8_signature_seed_count'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_ROOT_MAP_LEAF8_SEEDS {as_int(summary.get('leaf8_root_map_seed_count'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_LOW_RANK_LEAF8_SEEDS {as_int(summary.get('leaf8_low_rank_seed_count'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_SURFACE_GATE_LEAF8_HITS {as_int(summary.get('leaf8_surface_gate_unique_hit_count'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_SURFACE_GATE_BELOW_RHO_ROUTES {as_int(summary.get('leaf8_surface_gate_below_rho_route_count'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_LEAF12_SEEDS {as_int(summary.get('leaf12_seed_count'))}ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_ACCEPTED_EXPORTS 0ULL
#define SELECTED13_PRIORITY0_TERM812_SCOUT_RELATION_DERIVED_ECDLP 0ULL

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  uint64_t failure_count = 0;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_OPEN_TERM_COUNT != {EXPECTED_OPEN_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_EXPLICIT_SAME_TARGET_TERM_CLOSURES != {EXPECTED_EXPLICIT_SAME_TARGET_TERM_CLOSURE_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_CROSS_TARGET_ANALOGUE_TERMS != {EXPECTED_CROSS_TARGET_ANALOGUE_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_SIGNATURE_LEAF8_SEEDS != {EXPECTED_SIGNATURE_LEAF8_SEED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_ROOT_MAP_LEAF8_SEEDS != {EXPECTED_ROOT_MAP_LEAF8_SEED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_LOW_RANK_LEAF8_SEEDS != {EXPECTED_LOW_RANK_LEAF8_SEED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_SURFACE_GATE_LEAF8_HITS != {EXPECTED_SURFACE_GATE_UNIQUE_LEAF8_HIT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_SURFACE_GATE_BELOW_RHO_ROUTES != {EXPECTED_SURFACE_GATE_BELOW_RHO_ROUTE_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_LEAF12_SEEDS != {EXPECTED_LEAF12_SEED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_ACCEPTED_EXPORTS != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TERM812_SCOUT_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;

  printf("selected13_priority0_term812_route_scout_preflight transfer=%llu open_terms=%llu explicit_closures=%llu leaf8_signature=%llu leaf8_root_map=%llu leaf8_low_rank=%llu leaf8_surface_hits=%llu leaf8_below_routes=%llu leaf12_seeds=%llu accepted=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_TRANSFER,
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_OPEN_TERM_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_EXPLICIT_SAME_TARGET_TERM_CLOSURES,
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_SIGNATURE_LEAF8_SEEDS,
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_ROOT_MAP_LEAF8_SEEDS,
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_LOW_RANK_LEAF8_SEEDS,
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_SURFACE_GATE_LEAF8_HITS,
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_SURFACE_GATE_BELOW_RHO_ROUTES,
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_LEAF12_SEEDS,
         (unsigned long long)SELECTED13_PRIORITY0_TERM812_SCOUT_ACCEPTED_EXPORTS,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path, compiler: str) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root)
    with tempfile.TemporaryDirectory(prefix="selected13_term812_scout_", dir=str(temp_root)) as tmp:
        tmp_path = Path(tmp)
        c_path = tmp_path / "preflight.c"
        exe_path = tmp_path / "preflight"
        c_path.write_text(source)
        command = [
            compiler,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-O2",
            "-I",
            str(header_path.parent),
            str(c_path),
            "-o",
            str(exe_path),
        ]
        compile_run = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        if compile_run.returncode != 0:
            return {
                "c_source_sha256": source_hash,
                "compile_command": command,
                "compile_returncode": compile_run.returncode,
                "compile_stderr": compile_run.stderr,
                "compile_stdout": compile_run.stdout,
                "compiled": False,
                "executed": False,
                "verified": False,
            }
        native_run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False, env=env)
        return {
            "c_source_sha256": source_hash,
            "compile_command": command,
            "compile_returncode": compile_run.returncode,
            "compile_stderr": compile_run.stderr,
            "compile_stdout": compile_run.stdout,
            "compiled": True,
            "executed": True,
            "preflight_returncode": native_run.returncode,
            "preflight_stderr": native_run.stderr.strip(),
            "preflight_stdout": native_run.stdout.strip(),
            "verified": native_run.returncode == 0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-scanner", default=str(DEFAULT_GAP_SCANNER))
    parser.add_argument("--signature", default=str(DEFAULT_SIGNATURE))
    parser.add_argument("--signature-extended", default=str(DEFAULT_SIGNATURE_EXTENDED))
    parser.add_argument("--root-map-rank", default=str(DEFAULT_ROOT_MAP_RANK))
    parser.add_argument("--low-rank-remainder", default=str(DEFAULT_LOW_RANK_REMAINDER))
    parser.add_argument("--surface-gate-delta", default=str(DEFAULT_SURFACE_GATE_DELTA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_TERM812_ROUTE_SCOUT_FAILED"
    payload["summary"]["native_preflight_verified"] = bool(native_preflight.get("verified"))
    payload["summary"]["failure_count"] = len(payload["failures"])
    payload["summary"]["verified"] = not payload["failures"]
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
