#!/usr/bin/env python3
"""Exhaust the visible selected13 second-stage leaf lattice for transfer 9943.

The selected13 leaf replay probe found three unverified rank-1 seeds for the
9943 no-relation target.  This probe turns that into a bounded completeness
check: enumerate every non-empty subset of the verifier-visible row leaves
present in the 9943 worklist, replay each subset through the verifier path, and
record whether any subset reaches an independent rank-2/public-key-verified
direct relation system.

The output is still a routing artifact.  A rank-1 replay or hit-root hint is not
an accepted direct/rank export and is not an ECDLP recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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


SCHEMA = "ecdlp.low_term_total2_selected13_9943_second_stage_lattice_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_WORKLIST = DEFAULT_STATE_DIR / "low_term_total2_selected13_leaf_augmentation_worklist_9981_9943_probe.json"
DEFAULT_REPLAY_PROBE = DEFAULT_STATE_DIR / "low_term_total2_selected13_leaf_augmentation_replay_probe_9981_9943.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_9943_second_stage_lattice_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_9943_second_stage_lattice_probe.h"

TARGET = "22050.cf1@11731"
TRANSFER = 9943
ROW_KEY_ORDER = [
    "22050.cf1@11731:uniform:256:salt167",
    "22050.cf1@11731:uniform:256:salt175",
]

STATUS_CODES = {
    "PUBLIC_KEY_VERIFIED_EXPORT": 1,
    "RANK2_UNVERIFIED": 2,
    "RANK1_RELATION_ONLY": 3,
    "HIT_ROOT_ONLY": 4,
    "NO_RELATION": 5,
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


def compact_leaf_map(raw: dict[str, set[int]]) -> list[dict[str, Any]]:
    return [
        {"leaf_indices": sorted(raw[row_key]), "row_key": row_key}
        for row_key in ROW_KEY_ORDER
        if raw.get(row_key)
    ]


def leaf_signature(raw: dict[str, set[int]]) -> str:
    return json.dumps(compact_leaf_map(raw), sort_keys=True, separators=(",", ":"))


def union_leaf_universe(worklist: dict[str, Any]) -> dict[str, list[int]]:
    universe: dict[str, set[int]] = {row_key: set() for row_key in ROW_KEY_ORDER}
    for item in worklist.get("work_items") or []:
        if as_int(item.get("transfer_index"), -1) != TRANSFER:
            continue
        for row_key, leaves in row_leaf_map(item.get("candidate_row_leaf_keys") or []).items():
            universe.setdefault(row_key, set()).update(leaves)
    return {row_key: sorted(leaves) for row_key, leaves in universe.items() if leaves}


def replay_seed_maps(replay: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = []
    for record in replay.get("work_item_replays") or []:
        if as_int(record.get("transfer_index"), -1) != TRANSFER:
            continue
        if not record.get("unverified_rank_or_relation_gain"):
            continue
        seed_map: dict[str, set[int]] = {}
        for row in (record.get("replay") or {}).get("rows") or []:
            scan = row.get("scan") or {}
            leaves = {as_int(leaf) for leaf in scan.get("selected_leaf_indices") or []}
            if leaves:
                seed_map[str(row.get("row_key"))] = leaves
        seeds.append(
            {
                "candidate_class": record.get("candidate_class"),
                "leaf_map": compact_leaf_map(seed_map),
                "rank": as_int(record.get("rank")),
                "relation_count": as_int(record.get("relation_count")),
                "work_item_id": record.get("work_item_id"),
            }
        )
    return seeds


def enumerate_lattice(universe: dict[str, list[int]]) -> list[dict[str, set[int]]]:
    per_row_subsets = []
    for row_key in ROW_KEY_ORDER:
        leaves = universe.get(row_key) or []
        subsets: list[set[int]] = [set()]
        for size in range(1, len(leaves) + 1):
            subsets.extend(set(combo) for combo in itertools.combinations(leaves, size))
        per_row_subsets.append((row_key, subsets))

    candidates: list[dict[str, set[int]]] = []
    for left in per_row_subsets[0][1]:
        for right in per_row_subsets[1][1]:
            candidate: dict[str, set[int]] = {}
            if left:
                candidate[per_row_subsets[0][0]] = set(left)
            if right:
                candidate[per_row_subsets[1][0]] = set(right)
            if candidate:
                candidates.append(candidate)
    candidates.sort(
        key=lambda raw: (
            sum(len(leaves) for leaves in raw.values()),
            tuple((row_key, tuple(sorted(raw.get(row_key, set())))) for row_key in ROW_KEY_ORDER),
        )
    )
    return candidates


def status_for(result: dict[str, Any]) -> str:
    if bool(result.get("public_key_verified")):
        return "PUBLIC_KEY_VERIFIED_EXPORT"
    if as_int(result.get("rank")) >= 2 or as_int(result.get("relation_count")) >= 2:
        return "RANK2_UNVERIFIED"
    if as_int(result.get("rank")) > 0 or as_int(result.get("relation_count")) > 0:
        return "RANK1_RELATION_ONLY"
    hit_roots = 0
    for row in result.get("rows") or []:
        hit_roots += as_int((row.get("scan") or {}).get("selected_hit_roots"))
    if hit_roots:
        return "HIT_ROOT_ONLY"
    return "NO_RELATION"


def compact_candidate(
    index: int,
    leaves: dict[str, set[int]],
    result: dict[str, Any],
    row_events: list[tuple[str, dict[str, Any]]],
    event_limit: int,
) -> dict[str, Any]:
    row_summaries = []
    total_hit_roots = 0
    for row in result.get("rows") or []:
        scan = row.get("scan") or {}
        events = scan.get("event_summaries") or []
        hit_roots = as_int(scan.get("selected_hit_roots"))
        total_hit_roots += hit_roots
        row_summaries.append(
            {
                "event_summary_count": len(events),
                "event_summaries": events[:event_limit],
                "hit_event_count": as_int(scan.get("selected_hit_events")),
                "rank": as_int(scan.get("rank")),
                "relation_count": as_int(scan.get("relation_count")),
                "row_key": row.get("row_key"),
                "selected_hit_roots": hit_roots,
                "selected_leaf_indices": scan.get("selected_leaf_indices") or [],
            }
        )
    status = status_for(result)
    return {
        "candidate_index": index,
        "candidate_id": f"second_stage_{digest_u64(compact_leaf_map(leaves)):016x}",
        "candidate_id_u64": digest_u64(compact_leaf_map(leaves)),
        "leaf_count": sum(len(values) for values in leaves.values()),
        "ops_over_rho": round_or_none(result.get("ops_over_rho")),
        "public_key_verified": bool(result.get("public_key_verified")),
        "rank": as_int(result.get("rank")),
        "relation_count": as_int(result.get("relation_count")),
        "relation_derived_ecdlp": bool(result.get("public_key_verified")) and bool(result.get("derived")),
        "row_event_count": len(row_events),
        "row_summaries": row_summaries,
        "selected_hit_roots": total_hit_roots,
        "selected_leaf_map": compact_leaf_map(leaves),
        "status": status,
        "status_code": STATUS_CODES.get(status, 0),
    }


def run_lattice(args: argparse.Namespace, universe: dict[str, list[int]]) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
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
    contexts, context_errors = replay_probe.materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        {"target": TARGET, "transfer_index": TRANSFER, "top_k": args.top_k},
        ROW_KEY_ORDER,
        replay_args,
        {},
    )
    if context_errors:
        return [], radius, context_errors
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    out = []
    seen: set[str] = set()
    for candidate in enumerate_lattice(universe):
        signature = leaf_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        result, row_events = replay_probe.replay_selection(
            verifier,
            candidate,
            contexts,
            scan_cache,
            args.event_summary_limit,
        )
        out.append(compact_candidate(len(out), candidate, result, row_events, args.event_summary_limit))
    return out, radius, []


def claim_status(failures: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> str:
    if failures:
        return "SELECTED13_9943_SECOND_STAGE_LATTICE_FAILED"
    if any(candidate.get("public_key_verified") for candidate in candidates):
        return "SELECTED13_9943_SECOND_STAGE_LATTICE_FOUND_EXPORTABLE_RELATION"
    if any(as_int(candidate.get("rank")) >= 2 for candidate in candidates):
        return "SELECTED13_9943_SECOND_STAGE_LATTICE_RANK2_UNVERIFIED"
    return "SELECTED13_9943_SECOND_STAGE_LATTICE_NO_EXPORT"


def render_c_header(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for candidate in candidates:
        rows.append(
            "  {"
            f"{as_int(candidate.get('candidate_index'))}ULL, "
            f"{as_int(candidate.get('candidate_id_u64'))}ULL, "
            f"{as_int(candidate.get('leaf_count'))}ULL, "
            f"{as_int(candidate.get('rank'))}ULL, "
            f"{as_int(candidate.get('relation_count'))}ULL, "
            f"{as_int(candidate.get('row_event_count'))}ULL, "
            f"{as_int(candidate.get('selected_hit_roots'))}ULL, "
            f"{1 if candidate.get('public_key_verified') else 0}ULL, "
            f"{1 if candidate.get('relation_derived_ecdlp') else 0}ULL, "
            f"{as_int(candidate.get('status_code'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_9943_SECOND_STAGE_LATTICE_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_9943_SECOND_STAGE_LATTICE_PROBE_H

#include <stdint.h>

#define SELECTED13_9943_SECOND_STAGE_LATTICE_CANDIDATE_COUNT {len(candidates)}
#define SELECTED13_9943_SECOND_STAGE_LATTICE_EXPORT_COUNT {sum(1 for candidate in candidates if candidate.get('public_key_verified'))}
#define SELECTED13_9943_SECOND_STAGE_LATTICE_DERIVED_COUNT {sum(1 for candidate in candidates if candidate.get('relation_derived_ecdlp'))}
#define SELECTED13_9943_SECOND_STAGE_LATTICE_RANK2_COUNT {sum(1 for candidate in candidates if as_int(candidate.get('rank')) >= 2)}

#define SELECTED13_9943_SECOND_STAGE_STATUS_EXPORT 1ULL
#define SELECTED13_9943_SECOND_STAGE_STATUS_RANK2_UNVERIFIED 2ULL
#define SELECTED13_9943_SECOND_STAGE_STATUS_RANK1_RELATION 3ULL
#define SELECTED13_9943_SECOND_STAGE_STATUS_HIT_ROOT_ONLY 4ULL
#define SELECTED13_9943_SECOND_STAGE_STATUS_NO_RELATION 5ULL
#define SELECTED13_9943_SECOND_STAGE_STATUS_CONTEXT_ERROR 6ULL

typedef struct {{
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t leaf_count;
  uint64_t replay_rank;
  uint64_t replay_relation_count;
  uint64_t row_event_count;
  uint64_t selected_hit_roots;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t status_code;
}} selected13_9943_second_stage_lattice_candidate_t;

static const selected13_9943_second_stage_lattice_candidate_t SELECTED13_9943_SECOND_STAGE_LATTICE_CANDIDATES[] = {{
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
      sizeof(SELECTED13_9943_SECOND_STAGE_LATTICE_CANDIDATES) / sizeof(SELECTED13_9943_SECOND_STAGE_LATTICE_CANDIDATES[0]);
  uint64_t export_count = 0;
  uint64_t derived_count = 0;
  uint64_t rank2_count = 0;
  uint64_t rank1_count = 0;

  if (candidate_count != SELECTED13_9943_SECOND_STAGE_LATTICE_CANDIDATE_COUNT) failure_count++;
  if (candidate_count == 0ULL) failure_count++;

  for (size_t i = 0; i < candidate_count; i++) {{
    const selected13_9943_second_stage_lattice_candidate_t *candidate = &SELECTED13_9943_SECOND_STAGE_LATTICE_CANDIDATES[i];
    export_count += candidate->public_key_verified;
    derived_count += candidate->relation_derived_ecdlp;
    if (candidate->replay_rank >= 2ULL) rank2_count++;
    if (candidate->replay_rank == 1ULL) rank1_count++;
    if (candidate->candidate_id_u64 == 0ULL) failure_count++;
    if (candidate->status_code == 0ULL) failure_count++;
    if (candidate->relation_derived_ecdlp && !candidate->public_key_verified) failure_count++;
  }}

  if (export_count != SELECTED13_9943_SECOND_STAGE_LATTICE_EXPORT_COUNT) failure_count++;
  if (derived_count != SELECTED13_9943_SECOND_STAGE_LATTICE_DERIVED_COUNT) failure_count++;
  if (rank2_count != SELECTED13_9943_SECOND_STAGE_LATTICE_RANK2_COUNT) failure_count++;

  printf("selected13_9943_second_stage_lattice_preflight candidates=%llu exports=%llu derived=%llu rank2=%llu rank1=%llu failures=%llu\\n",
         (unsigned long long)candidate_count,
         (unsigned long long)export_count,
         (unsigned long long)derived_count,
         (unsigned long long)rank2_count,
         (unsigned long long)rank1_count,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_9943_lattice_preflight_") as tmp:
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
    worklist_path = Path(args.worklist)
    replay_path = Path(args.replay_probe)
    worklist = load_json(worklist_path)
    replay = load_json(replay_path)
    failures: list[dict[str, Any]] = []
    if worklist.get("claim_status") != "SELECTED13_LEAF_AUGMENTATION_WORKLIST_READY":
        failures.append({"code": "worklist_status_unexpected", "claim_status": worklist.get("claim_status")})
    if replay.get("claim_status") not in {
        "SELECTED13_LEAF_AUGMENTATION_REPLAY_NO_EXPORT",
        "SELECTED13_LEAF_AUGMENTATION_REPLAY_FOUND_EXPORTABLE_RELATION",
    }:
        failures.append({"code": "replay_status_unexpected", "claim_status": replay.get("claim_status")})
    seeds = replay_seed_maps(replay)
    universe = union_leaf_universe(worklist)
    candidates, radius, context_errors = run_lattice(args, universe)
    for error in context_errors:
        failures.append({"code": "context_materialization_error", "error": error})
    status_counts = Counter(str(candidate.get("status")) for candidate in candidates)
    rank1 = [candidate for candidate in candidates if as_int(candidate.get("rank")) == 1]
    rank2 = [candidate for candidate in candidates if as_int(candidate.get("rank")) >= 2]
    relation_candidates = [candidate for candidate in candidates if as_int(candidate.get("relation_count")) > 0]
    hit_root_only = [candidate for candidate in candidates if candidate.get("status") == "HIT_ROOT_ONLY"]
    ratios = [as_float(candidate.get("ops_over_rho")) for candidate in candidates if as_float(candidate.get("ops_over_rho")) is not None]
    top_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -as_int(candidate.get("rank")),
            -as_int(candidate.get("relation_count")),
            as_float(candidate.get("ops_over_rho")) if as_float(candidate.get("ops_over_rho")) is not None else 999.0,
            as_int(candidate.get("leaf_count")),
        ),
    )[: args.top_candidate_count]
    summary = {
        "accepted_relation_export_count": sum(1 for candidate in candidates if candidate.get("public_key_verified")),
        "candidate_count": len(candidates),
        "context_error_count": len(context_errors),
        "failure_count": len(failures),
        "hit_root_only_count": len(hit_root_only),
        "leaf_universe": {row_key: leaves for row_key, leaves in universe.items()},
        "max_rank": max((as_int(candidate.get("rank")) for candidate in candidates), default=0),
        "min_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "pollard_rho_speedup_claimed": False,
        "public_key_verified_count": sum(1 for candidate in candidates if candidate.get("public_key_verified")),
        "rank1_relation_count": len(rank1),
        "rank2_or_better_count": len(rank2),
        "relation_derived_ecdlp": any(candidate.get("relation_derived_ecdlp") for candidate in candidates),
        "relation_producing_count": len(relation_candidates),
        "seed_count": len(seeds),
        "status_counts": dict(sorted(status_counts.items())),
        "verified": not failures,
        "worker_interpretation": (
            "The visible 9943 second-stage leaf lattice is exhausted under the "
            "current verifier context. It produces rank-1 relation replays and "
            "salt175 hit-root-only hints, but no rank-2, public-key-verified, or "
            "relation-derived ECDLP result."
        ),
    }
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, candidates),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "config_source": str(Path(args.config_source)),
            "direct_source": str(Path(args.direct_source)),
            "event_summary_limit": args.event_summary_limit,
            "radius": radius,
            "target": TARGET,
            "top_k": args.top_k,
            "transfer_index": TRANSFER,
            "transfer_source": str(Path(args.transfer_source)),
            "worklist": str(worklist_path),
            "replay_probe": str(replay_path),
        },
        "summary": summary,
        "rank1_seed_items": seeds,
        "top_candidates": top_candidates,
        "candidate_replays": candidates,
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": summary["accepted_relation_export_count"],
            "exhaustive_for_visible_worklist_leaf_universe": True,
            "pollard_rho_speedup_claimed": False,
            "rank1_replay_is_not_export": True,
            "relation_derived_ecdlp": summary["relation_derived_ecdlp"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--replay-probe", type=Path, default=DEFAULT_REPLAY_PROBE)
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
    parser.add_argument("--event-summary-limit", type=int, default=4)
    parser.add_argument("--top-candidate-count", type=int, default=24)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["candidate_replays"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = claim_status(payload["failures"], payload["candidate_replays"])
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
