#!/usr/bin/env python3
"""Replay the 9842 -> 9943 selected13 salt-neighbor carryover path.

The source-gap preflight identifies transfer 9842 as the one-salt-neighbor
exact-support witness for the 9943 backfill target: source salts 167,176 map to
backfill salts 167,175.  This probe replays the 9842 source row/leaf systems,
maps their leaves by that salt mutation, and replays the mapped systems against
the 9943 verifier context.

This is a carryover test only.  A mapped rank-1 relation is not an accepted
export unless it public-key-verifies and derives the target relation system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import ffe_single_hit_root_relation_replay_probe as replay_probe


SCHEMA = "ecdlp.low_term_total2_selected13_9943_salt_neighbor_carryover_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SOURCE_GAP = DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_source_gap_preflight_selected13_9696_9999_probe.json"
DEFAULT_SIDECAR_COEFF = DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_sidecar_coeff_preflight_selected13_9696_9999_probe.json"
DEFAULT_SOURCE_POLICY = Path(
    "/Volumes/Volume/autolab/ecdlp_index_calculus_state/"
    "frontier_public_leaf_policy_p231_frozen_prefix_fixed_row_9840_9847_col15_selector_expanded_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_9943_salt_neighbor_carryover_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_9943_salt_neighbor_carryover_probe.h"

TARGET = "22050.cf1@11731"
SOURCE_TRANSFER = 9842
BACKFILL_TRANSFER = 9943
SOURCE_ROW_KEYS = [
    "22050.cf1@11731:uniform:256:salt167",
    "22050.cf1@11731:uniform:256:salt176",
]
BACKFILL_ROW_KEYS = [
    "22050.cf1@11731:uniform:256:salt167",
    "22050.cf1@11731:uniform:256:salt175",
]
SALT_MAP = {167: 167, 176: 175}

STATUS_CODES = {
    "SOURCE_VERIFIED_MUTATION_EXPORT": 1,
    "SOURCE_VERIFIED_MUTATION_NO_RELATION": 2,
    "SOURCE_VERIFIED_MUTATION_UNVERIFIED": 3,
    "MUTATION_RANK1_ONLY": 4,
    "MUTATION_NO_RELATION": 5,
    "CONTEXT_ERROR": 6,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
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


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def round_or_none(value: Any, digits: int = 8) -> float | None:
    numeric = as_float(value)
    return None if numeric is None else round(numeric, digits)


def digest_u64(raw: Any) -> int:
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def all_objects(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(value, dict):
        out.append(value)
        for child in value.values():
            out.extend(all_objects(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(all_objects(child))
    return out


def salt_from_row_key(row_key: str) -> int | None:
    match = re.search(r":salt(\d+)$", row_key)
    return int(match.group(1)) if match else None


def map_row_key(row_key: str) -> str | None:
    salt = salt_from_row_key(row_key)
    if salt not in SALT_MAP:
        return None
    return re.sub(r":salt\d+$", f":salt{SALT_MAP[salt]}", row_key)


def row_leaf_map(items: list[dict[str, Any]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        if not row_key:
            continue
        leaves = {as_int(leaf) for leaf in item.get("leaf_indices") or []}
        if leaves:
            out.setdefault(row_key, set()).update(leaves)
    return out


def compact_leaf_map(raw: dict[str, set[int]], row_order: list[str]) -> list[dict[str, Any]]:
    return [
        {"leaf_indices": sorted(raw[row_key]), "row_key": row_key}
        for row_key in row_order
        if raw.get(row_key)
    ]


def mapped_leaf_map(source_leaves: dict[str, set[int]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for row_key, leaves in source_leaves.items():
        mapped = map_row_key(row_key)
        if mapped is not None:
            out.setdefault(mapped, set()).update(leaves)
    return out


def source_rows(source_policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        item
        for item in all_objects(source_policy)
        if str(item.get("target") or "") == TARGET
        and as_int(item.get("transfer_index"), -1) == SOURCE_TRANSFER
        and item.get("row_leaf_keys")
    ]
    rows.sort(
        key=lambda row: (
            -int(bool(row.get("public_key_verified"))),
            -as_int(row.get("rank")),
            -as_int(row.get("relation_count")),
            as_float(row.get("ops_over_rho")) if as_float(row.get("ops_over_rho")) is not None else 999.0,
            str(row.get("selector") or row.get("row_selector") or ""),
        )
    )
    return rows


def source_gap_witnesses(source_gap: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in source_gap.get("source_gap_items") or []:
        if as_int(item.get("transfer_index"), -1) != BACKFILL_TRANSFER:
            continue
        for mask in item.get("mask_witnesses") or []:
            for witness in mask.get("witnesses") or []:
                if as_int(witness.get("transfer_index"), -1) == SOURCE_TRANSFER:
                    out.append(
                        {
                            "family_mask": mask.get("family_mask"),
                            "form_masks": witness.get("form_masks") or [],
                            "row_id": witness.get("row_id"),
                            "row_keys": witness.get("row_keys") or [],
                            "salt_delta": witness.get("salt_delta") or {},
                            "source_tier": mask.get("source_tier"),
                        }
                    )
    return out


def sidecar_forms(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    forms = []
    for form in sidecar.get("coefficient_forms") or []:
        if as_int(form.get("backfill_transfer_index"), -1) != BACKFILL_TRANSFER:
            continue
        if as_int(form.get("source_transfer_index"), -1) != SOURCE_TRANSFER:
            continue
        forms.append(
            {
                "form_hash_u64": as_int(form.get("form_hash_u64")),
                "form_index": as_int(form.get("form_index")),
                "source_form_support_mask": as_int(form.get("source_form_support_mask")),
                "source_public_key_verified": bool(form.get("source_public_key_verified")),
                "source_tier": form.get("source_tier"),
                "terms": form.get("terms") or [],
            }
        )
    return forms


def compact_rows(result: dict[str, Any], event_limit: int) -> list[dict[str, Any]]:
    rows = []
    for row in result.get("rows") or []:
        scan = row.get("scan") or {}
        rows.append(
            {
                "event_summaries": (scan.get("event_summaries") or [])[:event_limit],
                "event_summary_count": len(scan.get("event_summaries") or []),
                "hit_event_count": as_int(scan.get("selected_hit_events")),
                "rank": as_int(scan.get("rank")),
                "relation_count": as_int(scan.get("relation_count")),
                "row_key": row.get("row_key"),
                "selected_hit_roots": as_int(scan.get("selected_hit_roots")),
                "selected_leaf_indices": scan.get("selected_leaf_indices") or [],
            }
        )
    return rows


def replay_summary(result: dict[str, Any], event_limit: int) -> dict[str, Any]:
    return {
        "below_rho": bool(result.get("below_rho")),
        "derived": bool(result.get("derived")),
        "derived_secret": result.get("derived_secret"),
        "ops_over_rho": round_or_none(result.get("ops_over_rho")),
        "public_key_verified": bool(result.get("public_key_verified")),
        "rank": as_int(result.get("rank")),
        "relation_count": as_int(result.get("relation_count")),
        "row_event_count": sum(len((row.get("scan") or {}).get("event_summaries") or []) for row in result.get("rows") or []),
        "rows": compact_rows(result, event_limit),
        "selected_leaf_count": as_int(result.get("selected_leaf_count")),
        "selected_row_count": as_int(result.get("selected_row_count")),
    }


def status_for(source: dict[str, Any], mutated: dict[str, Any]) -> str:
    source_verified = bool(source.get("public_key_verified"))
    mutated_verified = bool(mutated.get("public_key_verified"))
    mutated_rank = as_int(mutated.get("rank"))
    mutated_relations = as_int(mutated.get("relation_count"))
    if source_verified and mutated_verified:
        return "SOURCE_VERIFIED_MUTATION_EXPORT"
    if source_verified and mutated_rank == 0 and mutated_relations == 0:
        return "SOURCE_VERIFIED_MUTATION_NO_RELATION"
    if source_verified:
        return "SOURCE_VERIFIED_MUTATION_UNVERIFIED"
    if mutated_rank > 0 or mutated_relations > 0:
        return "MUTATION_RANK1_ONLY"
    return "MUTATION_NO_RELATION"


def run_probe(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    bank_source = replay_probe.load_json(Path(args.bank_source))
    config_source = replay_probe.load_json(Path(args.config_source))
    direct_source = replay_probe.load_json(Path(args.direct_source))
    transfer_source = replay_probe.load_json(Path(args.transfer_source))
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    if not isinstance(params, dict):
        params = {}
    radius = as_int(args.radius if args.radius is not None else params.get("radius"), 4)
    specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
    verifier = replay_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()
    replay_args = argparse.Namespace(
        row_pool=args.row_pool,
        row_count=args.row_count,
        scout_limit=args.scout_limit,
        scout_mode=args.scout_mode,
        scout_order=args.scout_order,
        selected_limit=args.selected_limit,
        factor_base_size=args.factor_base_size,
        max_relations=args.max_relations,
        min_distinct_indices=args.min_distinct_indices,
        min_unsigned_distinct_indices=args.min_unsigned_distinct_indices,
        require_unit_coefficients=args.require_unit_coefficients,
        row_factor=args.row_factor,
        product_factor=args.product_factor,
        seed=args.seed,
        event_summary_limit=args.event_summary_limit,
    )
    source_contexts, source_errors = replay_probe.materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        {"target": TARGET, "transfer_index": SOURCE_TRANSFER, "top_k": args.top_k},
        SOURCE_ROW_KEYS,
        replay_args,
        {},
    )
    backfill_contexts, backfill_errors = replay_probe.materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        {"target": TARGET, "transfer_index": BACKFILL_TRANSFER, "top_k": args.top_k},
        BACKFILL_ROW_KEYS,
        replay_args,
        {},
    )
    context_errors = source_errors + backfill_errors
    if context_errors:
        return [], radius, context_errors
    source_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    backfill_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    out = []
    for index, row in enumerate(rows):
        source_leaves = row_leaf_map(row.get("row_leaf_keys") or [])
        backfill_leaves = mapped_leaf_map(source_leaves)
        source_replay, _source_events = replay_probe.replay_selection(
            verifier,
            source_leaves,
            source_contexts,
            source_cache,
            args.event_summary_limit,
        )
        mutated_replay, _mutated_events = replay_probe.replay_selection(
            verifier,
            backfill_leaves,
            backfill_contexts,
            backfill_cache,
            args.event_summary_limit,
        )
        source_summary = replay_summary(source_replay, args.event_summary_limit)
        mutated_summary = replay_summary(mutated_replay, args.event_summary_limit)
        status = status_for(source_summary, mutated_summary)
        out.append(
            {
                "backfill_leaf_map": compact_leaf_map(backfill_leaves, BACKFILL_ROW_KEYS),
                "candidate_index": index,
                "candidate_id": f"salt_neighbor_{digest_u64([row.get('selector'), compact_leaf_map(backfill_leaves, BACKFILL_ROW_KEYS)]):016x}",
                "candidate_id_u64": digest_u64([row.get("selector"), compact_leaf_map(backfill_leaves, BACKFILL_ROW_KEYS)]),
                "selector": row.get("selector") or row.get("row_selector"),
                "source_label": {
                    "below_rho": bool(row.get("below_rho")),
                    "ops_over_rho": round_or_none(row.get("ops_over_rho")),
                    "public_key_verified": bool(row.get("public_key_verified")),
                    "rank": as_int(row.get("rank")),
                    "relation_count": as_int(row.get("relation_count")),
                    "top_k": as_int(row.get("top_k")),
                },
                "source_leaf_map": compact_leaf_map(source_leaves, SOURCE_ROW_KEYS),
                "source_replay": source_summary,
                "status": status,
                "status_code": STATUS_CODES.get(status, 0),
                "target_replay": mutated_summary,
            }
        )
    return out, radius, []


def claim_status(failures: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> str:
    if failures:
        return "SELECTED13_9943_SALT_NEIGHBOR_CARRYOVER_FAILED"
    if any((item.get("target_replay") or {}).get("public_key_verified") for item in candidates):
        return "SELECTED13_9943_SALT_NEIGHBOR_CARRYOVER_FOUND_EXPORTABLE_RELATION"
    return "SELECTED13_9943_SALT_NEIGHBOR_CARRYOVER_NO_EXPORT"


def render_c_header(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for item in candidates:
        source = item.get("source_replay") or {}
        target = item.get("target_replay") or {}
        rows.append(
            "  {"
            f"{as_int(item.get('candidate_index'))}ULL, "
            f"{as_int(item.get('candidate_id_u64'))}ULL, "
            f"{1 if source.get('public_key_verified') else 0}ULL, "
            f"{as_int(source.get('rank'))}ULL, "
            f"{as_int(source.get('relation_count'))}ULL, "
            f"{1 if target.get('public_key_verified') else 0}ULL, "
            f"{as_int(target.get('rank'))}ULL, "
            f"{as_int(target.get('relation_count'))}ULL, "
            f"{1 if target.get('derived') else 0}ULL, "
            f"{as_int(item.get('status_code'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_9943_SALT_NEIGHBOR_CARRYOVER_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_9943_SALT_NEIGHBOR_CARRYOVER_PROBE_H

#include <stdint.h>

#define SELECTED13_9943_SALT_NEIGHBOR_CANDIDATE_COUNT {len(candidates)}
#define SELECTED13_9943_SALT_NEIGHBOR_SOURCE_VERIFIED_COUNT {sum(1 for item in candidates if (item.get('source_replay') or {}).get('public_key_verified'))}
#define SELECTED13_9943_SALT_NEIGHBOR_TARGET_VERIFIED_COUNT {sum(1 for item in candidates if (item.get('target_replay') or {}).get('public_key_verified'))}
#define SELECTED13_9943_SALT_NEIGHBOR_TARGET_DERIVED_COUNT {sum(1 for item in candidates if (item.get('target_replay') or {}).get('derived'))}

#define SELECTED13_9943_SALT_NEIGHBOR_STATUS_SOURCE_VERIFIED_EXPORT 1ULL
#define SELECTED13_9943_SALT_NEIGHBOR_STATUS_SOURCE_VERIFIED_NO_RELATION 2ULL
#define SELECTED13_9943_SALT_NEIGHBOR_STATUS_SOURCE_VERIFIED_UNVERIFIED 3ULL
#define SELECTED13_9943_SALT_NEIGHBOR_STATUS_MUTATION_RANK1 4ULL
#define SELECTED13_9943_SALT_NEIGHBOR_STATUS_MUTATION_NO_RELATION 5ULL
#define SELECTED13_9943_SALT_NEIGHBOR_STATUS_CONTEXT_ERROR 6ULL

typedef struct {{
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t source_public_key_verified;
  uint64_t source_rank;
  uint64_t source_relation_count;
  uint64_t target_public_key_verified;
  uint64_t target_rank;
  uint64_t target_relation_count;
  uint64_t target_relation_derived_ecdlp;
  uint64_t status_code;
}} selected13_9943_salt_neighbor_candidate_t;

static const selected13_9943_salt_neighbor_candidate_t SELECTED13_9943_SALT_NEIGHBOR_CANDIDATES[] = {{
{chr(10).join(rows)}
}};

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  uint64_t failure_count = 0;
  uint64_t candidate_count =
      sizeof(SELECTED13_9943_SALT_NEIGHBOR_CANDIDATES) / sizeof(SELECTED13_9943_SALT_NEIGHBOR_CANDIDATES[0]);
  uint64_t source_verified_count = 0;
  uint64_t target_verified_count = 0;
  uint64_t target_derived_count = 0;
  uint64_t target_rank1_count = 0;

  if (candidate_count != SELECTED13_9943_SALT_NEIGHBOR_CANDIDATE_COUNT) failure_count++;
  if (candidate_count == 0ULL) failure_count++;

  for (size_t i = 0; i < candidate_count; i++) {{
    const selected13_9943_salt_neighbor_candidate_t *candidate = &SELECTED13_9943_SALT_NEIGHBOR_CANDIDATES[i];
    source_verified_count += candidate->source_public_key_verified;
    target_verified_count += candidate->target_public_key_verified;
    target_derived_count += candidate->target_relation_derived_ecdlp;
    if (candidate->target_rank == 1ULL) target_rank1_count++;
    if (candidate->candidate_id_u64 == 0ULL) failure_count++;
    if (candidate->status_code == 0ULL) failure_count++;
    if (candidate->target_relation_derived_ecdlp && !candidate->target_public_key_verified) failure_count++;
  }}

  if (source_verified_count != SELECTED13_9943_SALT_NEIGHBOR_SOURCE_VERIFIED_COUNT) failure_count++;
  if (target_verified_count != SELECTED13_9943_SALT_NEIGHBOR_TARGET_VERIFIED_COUNT) failure_count++;
  if (target_derived_count != SELECTED13_9943_SALT_NEIGHBOR_TARGET_DERIVED_COUNT) failure_count++;

  printf("selected13_9943_salt_neighbor_carryover_preflight candidates=%llu source_verified=%llu target_verified=%llu target_derived=%llu target_rank1=%llu failures=%llu\\n",
         (unsigned long long)candidate_count,
         (unsigned long long)source_verified_count,
         (unsigned long long)target_verified_count,
         (unsigned long long)target_derived_count,
         (unsigned long long)target_rank1_count,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_9943_salt_neighbor_preflight_") as tmp:
        tmp_path = Path(tmp)
        c_path = tmp_path / "preflight.c"
        exe_path = tmp_path / "preflight"
        local_header = tmp_path / header_path.name
        c_path.write_text(source)
        local_header.write_text(header_path.read_text())
        compile_cmd = ["cc", "-std=c99", "-Wall", "-Wextra", "-O2", str(c_path), "-o", str(exe_path)]
        compile_run = subprocess.run(compile_cmd, text=True, capture_output=True, check=False)
        if compile_run.returncode != 0:
            return {
                "compile_command": compile_cmd,
                "compile_returncode": compile_run.returncode,
                "compile_stderr": compile_run.stderr,
                "verified": False,
            }
        preflight_run = subprocess.run([str(exe_path)], text=True, capture_output=True, check=False)
        return {
            "compile_command": compile_cmd,
            "compile_returncode": compile_run.returncode,
            "preflight_returncode": preflight_run.returncode,
            "preflight_stdout": preflight_run.stdout.strip(),
            "preflight_stderr": preflight_run.stderr.strip(),
            "verified": preflight_run.returncode == 0,
        }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_gap = load_json(Path(args.source_gap))
    sidecar = load_json(Path(args.sidecar_coeff))
    source_policy = load_json(Path(args.source_policy))
    failures: list[dict[str, Any]] = []
    if source_gap.get("claim_status") != "FFE_SHARP_LANE_SOURCE_GAP_PREFLIGHT_READY":
        failures.append({"code": "source_gap_status_unexpected", "claim_status": source_gap.get("claim_status")})
    if sidecar.get("claim_status") != "FFE_SHARP_LANE_SIDECAR_COEFF_PREFLIGHT_READY":
        failures.append({"code": "sidecar_coeff_status_unexpected", "claim_status": sidecar.get("claim_status")})
    rows = source_rows(source_policy)
    candidates, radius, context_errors = run_probe(args, rows)
    for error in context_errors:
        failures.append({"code": "context_materialization_error", "error": error})
    source_verified = [item for item in candidates if (item.get("source_replay") or {}).get("public_key_verified")]
    target_verified = [item for item in candidates if (item.get("target_replay") or {}).get("public_key_verified")]
    target_rank1 = [item for item in candidates if as_int((item.get("target_replay") or {}).get("rank")) == 1]
    status_counts = Counter(str(item.get("status")) for item in candidates)
    summary = {
        "accepted_relation_export_count": len(target_verified),
        "candidate_count": len(candidates),
        "context_error_count": len(context_errors),
        "failure_count": len(failures),
        "mapped_target_rank1_count": len(target_rank1),
        "max_mapped_target_rank": max((as_int((item.get("target_replay") or {}).get("rank")) for item in candidates), default=0),
        "pollard_rho_speedup_claimed": False,
        "source_gap_witness_count": len(source_gap_witnesses(source_gap)),
        "source_public_key_verified_count": len(source_verified),
        "source_sidecar_form_count": len(sidecar_forms(sidecar)),
        "source_transfer_index": SOURCE_TRANSFER,
        "status_counts": dict(sorted(status_counts.items())),
        "target_public_key_verified_count": len(target_verified),
        "target_relation_derived_ecdlp": any((item.get("target_replay") or {}).get("derived") for item in candidates),
        "transfer_index": BACKFILL_TRANSFER,
        "verified": not failures,
        "worker_interpretation": (
            "The verified 9842 salt-neighbor source row replays on its native "
            "transfer but maps to zero 9943 relations. The only mapped rank-1 "
            "9943 rows come from source-unverified low-leaf selectors, so this "
            "carryover path does not produce a direct/rank export."
        ),
    }
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, candidates),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "backfill_row_keys": BACKFILL_ROW_KEYS,
            "backfill_transfer_index": BACKFILL_TRANSFER,
            "config_source": str(Path(args.config_source)),
            "direct_source": str(Path(args.direct_source)),
            "radius": radius,
            "salt_map": SALT_MAP,
            "sidecar_coeff": str(Path(args.sidecar_coeff)),
            "source_gap": str(Path(args.source_gap)),
            "source_policy": str(Path(args.source_policy)),
            "source_row_keys": SOURCE_ROW_KEYS,
            "source_transfer_index": SOURCE_TRANSFER,
            "target": TARGET,
            "top_k": args.top_k,
            "transfer_source": str(Path(args.transfer_source)),
        },
        "summary": summary,
        "source_gap_witnesses": source_gap_witnesses(source_gap),
        "source_sidecar_forms": sidecar_forms(sidecar),
        "carryover_candidates": candidates,
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": len(target_verified),
            "mapped_rank1_is_not_export": True,
            "pollard_rho_speedup_claimed": False,
            "target_relation_derived_ecdlp": summary["target_relation_derived_ecdlp"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gap", type=Path, default=DEFAULT_SOURCE_GAP)
    parser.add_argument("--sidecar-coeff", type=Path, default=DEFAULT_SIDECAR_COEFF)
    parser.add_argument("--source-policy", type=Path, default=DEFAULT_SOURCE_POLICY)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--row-pool", type=int, default=512)
    parser.add_argument("--row-count", type=int, default=128)
    parser.add_argument("--scout-limit", type=int, default=192)
    parser.add_argument("--scout-mode", default="s3_coeff_spread")
    parser.add_argument("--scout-order", default="eval_cover_hits_high")
    parser.add_argument("--selected-limit", type=int, default=64)
    parser.add_argument("--factor-base-size", type=int, default=16)
    parser.add_argument("--max-relations", type=int, default=96)
    parser.add_argument("--min-distinct-indices", type=int, default=4)
    parser.add_argument("--min-unsigned-distinct-indices", type=int, default=2)
    parser.add_argument(
        "--allow-combined-coefficients",
        dest="require_unit_coefficients",
        action="store_false",
    )
    parser.set_defaults(require_unit_coefficients=True)
    parser.add_argument("--row-factor", type=int, default=512)
    parser.add_argument("--product-factor", type=int, default=4096)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--event-summary-limit", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["carryover_candidates"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = claim_status(payload["failures"], payload["carryover_candidates"])
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
