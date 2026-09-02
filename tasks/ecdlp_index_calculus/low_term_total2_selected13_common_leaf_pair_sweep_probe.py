#!/usr/bin/env python3
"""Sweep common single-leaf event cores across selected13 backfill targets.

The 9943 and 9981 event-core probes both reduced to one common leaf index on
both public salts. This probe tests that rule family across every selected13
direct/rank backfill transfer in the 9999 kernel contract:

* materialize each full-family backfill row pair,
* replay `{leaf i on row A, leaf i on row B}` for every shared leaf index, and
* record relation-derived ECDLP recoveries against Pollard-rho at candidate
  level.

The result is intentionally framed as a rule-mining sweep. Candidate-level
below-rho rows are useful evidence for a public/FFE selector family, but the
full sweep cost is not promoted as a generalized ECDLP speedup.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import ffe_single_hit_root_relation_replay_probe as replay_probe
import low_term_total2_selected13_public_prefix_min_transfer_probe as min_transfer


SCHEMA = "ecdlp.low_term_total2_selected13_common_leaf_pair_sweep_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_CONTRACT = DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9999_probe.json"
DEFAULT_9981_GAP_CLOSURE = DEFAULT_STATE_DIR / "low_term_total2_selected13_9981_event_core_gap_closure_probe.json"
DEFAULT_MIN_TRANSFER = DEFAULT_STATE_DIR / "low_term_total2_selected13_public_prefix_min_transfer_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_common_leaf_pair_sweep_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_common_leaf_pair_sweep_probe.h"

TARGET = min_transfer.TARGET
KNOWN_POSITIVE_TRANSFERS = (9943, 9981)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: Any, default: int = 0) -> int:
    return min_transfer.as_int(value, default)


def as_float(value: Any) -> float | None:
    return min_transfer.as_float(value)


def round_or_none(value: Any, digits: int = 8) -> float | None:
    return min_transfer.round_or_none(value, digits)


def contract_backfill_targets(contract: dict[str, Any]) -> list[dict[str, Any]]:
    targets = []
    for item in (contract.get("contract") or {}).get("direct_rank_backfill_manifest") or []:
        public_first_pass = item.get("public_first_pass") or {}
        row_keys = [str(row_key) for row_key in public_first_pass.get("row_keys") or []]
        if len(row_keys) != 2:
            continue
        full_rows = [row for row in item.get("row_checks") or [] if row.get("is_full_family")]
        targets.append(
            {
                "first_pass_id": item.get("first_pass_id"),
                "full_family_row_id": (item.get("full_family_row_ids") or [None])[0],
                "group_id": item.get("group_id"),
                "known_positive": as_int(item.get("transfer_index")) in KNOWN_POSITIVE_TRANSFERS,
                "min_direct_ops_over_rho": round_or_none(item.get("min_direct_ops_over_rho")),
                "row_check_hash": full_rows[0].get("row_check_hash") if full_rows else None,
                "row_keys": row_keys,
                "salts": public_first_pass.get("salts") or [],
                "target": str(public_first_pass.get("target") or TARGET),
                "transfer_index": as_int(item.get("transfer_index")),
            }
        )
    return sorted(targets, key=lambda item: as_int(item.get("transfer_index")))


def materialize_contexts_for_target(
    verifier: Any,
    records: list[dict[str, Any]],
    config_source: dict[str, Any],
    specs_by_target: dict[str, dict[str, dict[str, Any]]],
    target: dict[str, Any],
    args: argparse.Namespace,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    case = {
        "target": target.get("target") or TARGET,
        "transfer_index": as_int(target.get("transfer_index")),
        "top_k": args.context_top_k,
    }
    return replay_probe.materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        case,
        [str(row_key) for row_key in target.get("row_keys") or []],
        args,
        context_cache,
    )


def candidate_input(target: dict[str, Any], leaf_index: int) -> dict[str, Any]:
    row_keys = [str(row_key) for row_key in target.get("row_keys") or []]
    selected_leaf_map = [
        {"leaf_indices": [leaf_index], "row_key": row_key}
        for row_key in row_keys
    ]
    return {
        "candidate_class": "common_single_leaf_pair",
        "candidate_class_code": 1,
        "known_positive_transfer": bool(target.get("known_positive")),
        "leaf_index": leaf_index,
        "selected_leaf_count": len(row_keys),
        "selected_leaf_map": selected_leaf_map,
        "transfer_index": as_int(target.get("transfer_index")),
    }


def compact_candidate(
    index: int,
    target: dict[str, Any],
    candidate: dict[str, Any],
    result: dict[str, Any],
    row_events: list[tuple[str, dict[str, Any]]],
    event_limit: int,
) -> dict[str, Any]:
    record = min_transfer.compact_candidate(index, candidate, result, row_events, event_limit)
    record["candidate_id"] = f"common_leaf_{min_transfer.digest_u64([candidate, index]):016x}"
    record["candidate_id_u64"] = min_transfer.digest_u64([candidate, index])
    record["full_family_row_id"] = target.get("full_family_row_id")
    record["row_check_hash"] = target.get("row_check_hash")
    record["salts"] = target.get("salts") or []
    return record


def replay_all(args: argparse.Namespace, targets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
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
        context_top_k=args.context_top_k,
    )

    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    records_out = []
    candidate_index = 0
    for target in targets:
        contexts, errors = materialize_contexts_for_target(
            verifier,
            records,
            config_source,
            specs_by_target,
            target,
            replay_args,
            context_cache,
        )
        for error in errors:
            failures.append(
                {
                    "code": "context_materialization_error",
                    "error": error,
                    "transfer_index": target.get("transfer_index"),
                }
            )
        if errors or len(contexts) != 2:
            continue
        leaf_count = min(len(context["components"]["leaves"]) for context in contexts.values())
        if args.leaf_limit is not None:
            leaf_count = min(leaf_count, args.leaf_limit)
        row_keys = [str(row_key) for row_key in target.get("row_keys") or []]
        for leaf_index in range(leaf_count):
            candidate = candidate_input(target, leaf_index)
            row_leaves = {row_key: {leaf_index} for row_key in row_keys}
            result, row_events = replay_probe.replay_selection(
                verifier,
                row_leaves,
                contexts,
                scan_cache,
                args.event_summary_limit,
            )
            records_out.append(compact_candidate(candidate_index, target, candidate, result, row_events, args.event_summary_limit))
            candidate_index += 1
    return records_out, radius, failures


def best_record(records: list[dict[str, Any]], *, heldout_only: bool = False, require_below: bool = False) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if record.get("accepted_relation_export")
        and (not heldout_only or not record.get("known_positive_transfer"))
        and (not require_below or record.get("below_rho"))
    ]
    candidates.sort(
        key=lambda item: (
            as_float(item.get("ops_over_rho")) if as_float(item.get("ops_over_rho")) is not None else 999.0,
            as_int(item.get("selected_leaf_count")),
            as_int(item.get("transfer_index")),
            as_int(item.get("leaf_index")),
        )
    )
    return candidates[0] if candidates else {}


def summarize_transfer(records: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any]:
    transfer_records = [row for row in records if as_int(row.get("transfer_index")) == as_int(target.get("transfer_index"))]
    accepted = [row for row in transfer_records if row.get("accepted_relation_export")]
    below = [row for row in accepted if row.get("below_rho")]
    rankish = [row for row in transfer_records if as_int(row.get("rank")) >= 2 or as_int(row.get("relation_count")) >= 2]
    best = best_record(transfer_records, require_below=True) or best_record(transfer_records)
    total_ops = sum(as_int(row.get("ops")) for row in transfer_records)
    generic_rho_steps = max((as_int(row.get("generic_rho_steps")) for row in transfer_records), default=0)
    statuses = Counter(str(row.get("status")) for row in transfer_records)
    return {
        "accepted_relation_export_count": len(accepted),
        "below_rho_accepted_relation_export_count": len(below),
        "best_candidate_id": best.get("candidate_id"),
        "best_derived_secret": best.get("derived_secret"),
        "best_leaf_index": best.get("leaf_index"),
        "best_ops_over_rho": best.get("ops_over_rho"),
        "best_rank": best.get("rank"),
        "best_relation_count": best.get("relation_count"),
        "candidate_count": len(transfer_records),
        "generic_rho_steps": generic_rho_steps,
        "known_positive": bool(target.get("known_positive")),
        "rank2_or_better_candidate_count": len(rankish),
        "row_keys": target.get("row_keys") or [],
        "salts": target.get("salts") or [],
        "status_counts": dict(sorted(statuses.items())),
        "sweep_ops": total_ops,
        "sweep_ops_over_rho": round_or_none(total_ops / generic_rho_steps if generic_rho_steps else None),
        "transfer_index": as_int(target.get("transfer_index")),
    }


def summarize(records: list[dict[str, Any]], targets: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    transfer_summaries = [summarize_transfer(records, target) for target in targets]
    transfer_count = len(transfer_summaries)
    heldout = [row for row in transfer_summaries if not row.get("known_positive")]
    below_transfers = [row for row in transfer_summaries if row.get("below_rho_accepted_relation_export_count")]
    heldout_below = [row for row in heldout if row.get("below_rho_accepted_relation_export_count")]
    no_below = [row.get("transfer_index") for row in transfer_summaries if not row.get("below_rho_accepted_relation_export_count")]
    accepted = [row for row in records if row.get("accepted_relation_export")]
    below = [row for row in accepted if row.get("below_rho")]
    best = best_record(records, require_below=True)
    best_heldout = best_record(records, heldout_only=True, require_below=True)
    return {
        "accepted_candidate_count": len(accepted),
        "below_rho_accepted_candidate_count": len(below),
        "best_candidate_id": best.get("candidate_id"),
        "best_derived_secret": best.get("derived_secret"),
        "best_heldout_candidate_id": best_heldout.get("candidate_id"),
        "best_heldout_derived_secret": best_heldout.get("derived_secret"),
        "best_heldout_leaf_index": best_heldout.get("leaf_index"),
        "best_heldout_ops_over_rho": best_heldout.get("ops_over_rho"),
        "best_heldout_transfer_index": best_heldout.get("transfer_index"),
        "best_leaf_index": best.get("leaf_index"),
        "best_ops_over_rho": best.get("ops_over_rho"),
        "best_transfer_index": best.get("transfer_index"),
        "candidate_count": len(records),
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "heldout_below_rho_transfer_count": len(heldout_below),
        "heldout_transfer_count": len(heldout),
        "known_positive_transfers": list(KNOWN_POSITIVE_TRANSFERS),
        "same_leaf_pair_sweep_promoted_as_end_to_end_speedup": False,
        "transfer_count": transfer_count,
        "transfer_summaries": transfer_summaries,
        "transfers_without_below_rho_export": no_below,
        "transfers_with_below_rho_export_count": len(below_transfers),
        "verified": not failures,
        "worker_interpretation": (
            "The common single-leaf pair sweep finds many candidate-level below-rho "
            "relation-derived event cores across selected13 backfill transfers. "
            "Because it scans leaf choices, it is a selector-mining artifact rather "
            "than an end-to-end generalized ECDLP speedup."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_COMMON_LEAF_PAIR_SWEEP_FAILED"
    if summary.get("heldout_below_rho_transfer_count", 0) >= 1:
        return "SELECTED13_COMMON_LEAF_PAIR_SWEEP_HELDOUT_BELOW_RHO_CANDIDATES"
    if summary.get("below_rho_accepted_candidate_count", 0) >= 1:
        return "SELECTED13_COMMON_LEAF_PAIR_SWEEP_KNOWN_ONLY"
    return "SELECTED13_COMMON_LEAF_PAIR_SWEEP_NO_EXPORT"


def render_c_header(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        ops_scaled = 0
        ops = as_float(record.get("ops_over_rho"))
        if ops is not None:
            ops_scaled = int(round(ops * 1_000_000))
        rows.append(
            "  {"
            f"{as_int(record.get('candidate_index'))}ULL, "
            f"{as_int(record.get('candidate_id_u64'))}ULL, "
            f"{as_int(record.get('transfer_index'))}ULL, "
            f"{as_int(record.get('leaf_index'))}ULL, "
            f"{1 if record.get('known_positive_transfer') else 0}ULL, "
            f"{1 if record.get('below_rho') else 0}ULL, "
            f"{1 if record.get('public_key_verified') else 0}ULL, "
            f"{1 if record.get('relation_derived_ecdlp') else 0}ULL, "
            f"{as_int(record.get('rank'))}ULL, "
            f"{as_int(record.get('relation_count'))}ULL, "
            f"{as_int(record.get('derived_secret'))}ULL, "
            f"{ops_scaled}ULL, "
            f"{as_int(record.get('status_code'))}ULL"
            "},"
        )
    accepted_count = sum(1 for row in records if row.get("relation_derived_ecdlp"))
    below_count = sum(1 for row in records if row.get("relation_derived_ecdlp") and row.get("below_rho"))
    heldout_below_count = sum(
        1
        for row in records
        if row.get("relation_derived_ecdlp") and row.get("below_rho") and not row.get("known_positive_transfer")
    )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_COMMON_LEAF_PAIR_SWEEP_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_COMMON_LEAF_PAIR_SWEEP_PROBE_H

#include <stdint.h>

#define SELECTED13_COMMON_LEAF_PAIR_SWEEP_CANDIDATE_COUNT {len(records)}
#define SELECTED13_COMMON_LEAF_PAIR_SWEEP_ACCEPTED_COUNT {accepted_count}
#define SELECTED13_COMMON_LEAF_PAIR_SWEEP_BELOW_RHO_ACCEPTED_COUNT {below_count}
#define SELECTED13_COMMON_LEAF_PAIR_SWEEP_HELDOUT_BELOW_RHO_ACCEPTED_COUNT {heldout_below_count}

typedef struct {{
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t transfer_index;
  uint64_t leaf_index;
  uint64_t known_positive_transfer;
  uint64_t below_rho;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t rank;
  uint64_t relation_count;
  uint64_t derived_secret;
  uint64_t ops_over_rho_scaled_1e6;
  uint64_t status_code;
}} selected13_common_leaf_pair_sweep_candidate_t;

static const selected13_common_leaf_pair_sweep_candidate_t SELECTED13_COMMON_LEAF_PAIR_SWEEP_CANDIDATES[] = {{
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
      sizeof(SELECTED13_COMMON_LEAF_PAIR_SWEEP_CANDIDATES) / sizeof(SELECTED13_COMMON_LEAF_PAIR_SWEEP_CANDIDATES[0]);
  uint64_t accepted = 0;
  uint64_t accepted_below = 0;
  uint64_t heldout_accepted_below = 0;
  uint64_t transfer_seen_below[10000] = {{0}};
  uint64_t heldout_transfer_seen_below[10000] = {{0}};
  uint64_t transfer_below_count = 0;
  uint64_t heldout_transfer_below_count = 0;

  if (candidate_count != SELECTED13_COMMON_LEAF_PAIR_SWEEP_CANDIDATE_COUNT) failure_count++;
  if (candidate_count == 0ULL) failure_count++;

  for (size_t i = 0; i < candidate_count; i++) {{
    const selected13_common_leaf_pair_sweep_candidate_t *candidate = &SELECTED13_COMMON_LEAF_PAIR_SWEEP_CANDIDATES[i];
    if (candidate->candidate_id_u64 == 0ULL) failure_count++;
    if (candidate->status_code == 0ULL) failure_count++;
    if (candidate->transfer_index >= 10000ULL) failure_count++;
    if (candidate->relation_derived_ecdlp && !candidate->public_key_verified) failure_count++;
    if (candidate->relation_derived_ecdlp && candidate->derived_secret == 0ULL) failure_count++;
    if (candidate->relation_derived_ecdlp) {{
      accepted++;
      if (candidate->below_rho) {{
        accepted_below++;
        if (!candidate->known_positive_transfer) heldout_accepted_below++;
        if (candidate->transfer_index < 10000ULL && !transfer_seen_below[candidate->transfer_index]) {{
          transfer_seen_below[candidate->transfer_index] = 1ULL;
          transfer_below_count++;
        }}
        if (candidate->transfer_index < 10000ULL && !candidate->known_positive_transfer && !heldout_transfer_seen_below[candidate->transfer_index]) {{
          heldout_transfer_seen_below[candidate->transfer_index] = 1ULL;
          heldout_transfer_below_count++;
        }}
      }}
    }}
  }}

  if (accepted != SELECTED13_COMMON_LEAF_PAIR_SWEEP_ACCEPTED_COUNT) failure_count++;
  if (accepted_below != SELECTED13_COMMON_LEAF_PAIR_SWEEP_BELOW_RHO_ACCEPTED_COUNT) failure_count++;
  if (heldout_accepted_below != SELECTED13_COMMON_LEAF_PAIR_SWEEP_HELDOUT_BELOW_RHO_ACCEPTED_COUNT) failure_count++;
  if (transfer_below_count < 12ULL) failure_count++;
  if (heldout_transfer_below_count < 10ULL) failure_count++;

  printf("selected13_common_leaf_pair_sweep_preflight candidates=%llu accepted=%llu accepted_below=%llu heldout_accepted_below=%llu transfer_below=%llu heldout_transfer_below=%llu failures=%llu\\n",
         (unsigned long long)candidate_count,
         (unsigned long long)accepted,
         (unsigned long long)accepted_below,
         (unsigned long long)heldout_accepted_below,
         (unsigned long long)transfer_below_count,
         (unsigned long long)heldout_transfer_below_count,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_common_leaf_pair_sweep_preflight_") as tmp:
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
    contract = load_json(Path(args.contract))
    min_transfer_payload = load_json(Path(args.min_transfer))
    gap_closure = load_json(Path(args.gap_closure))
    failures: list[dict[str, Any]] = []
    if contract.get("claim_status") != "FFE_SHARP_LANE_KERNEL_CONTRACT_READY":
        failures.append({"code": "contract_status_unexpected", "claim_status": contract.get("claim_status")})
    if min_transfer_payload.get("claim_status") != "SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_HELDOUT_OVER_RHO":
        failures.append({"code": "min_transfer_status_unexpected", "claim_status": min_transfer_payload.get("claim_status")})
    if gap_closure.get("claim_status") != "SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_BELOW_RHO_EXPORT":
        failures.append({"code": "gap_closure_status_unexpected", "claim_status": gap_closure.get("claim_status")})
    targets = contract_backfill_targets(contract)
    if not targets:
        failures.append({"code": "no_backfill_targets"})
    records, radius, replay_failures = replay_all(args, targets)
    failures.extend(replay_failures)
    summary = summarize(records, targets, failures)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, summary),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "config_source": str(Path(args.config_source)),
            "context_top_k": args.context_top_k,
            "contract": str(Path(args.contract)),
            "direct_source": str(Path(args.direct_source)),
            "event_summary_limit": args.event_summary_limit,
            "gap_closure": str(Path(args.gap_closure)),
            "known_positive_transfers": list(KNOWN_POSITIVE_TRANSFERS),
            "leaf_limit": args.leaf_limit,
            "min_transfer": str(Path(args.min_transfer)),
            "radius": radius,
            "target": TARGET,
            "transfer_source": str(Path(args.transfer_source)),
        },
        "target_specs": targets,
        "summary": summary,
        "common_leaf_pair_candidates": records,
        "failures": failures,
        "honesty_boundary": {
            "candidate_level_below_rho": summary["below_rho_accepted_candidate_count"] > 0,
            "general_ecdlp_algorithm_claimed": False,
            "same_leaf_pair_sweep_promoted_as_end_to_end_speedup": False,
            "selection_cost_note": "The probe sweeps leaf choices; individual candidate ops/rho values do not include the full cost of discovering the leaf without a predictive selector.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--min-transfer", type=Path, default=DEFAULT_MIN_TRANSFER)
    parser.add_argument("--gap-closure", type=Path, default=DEFAULT_9981_GAP_CLOSURE)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--leaf-limit", type=int)
    parser.add_argument("--context-top-k", type=int, default=16)
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["common_leaf_pair_candidates"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = claim_status(payload["failures"], payload["summary"])
    write_json(Path(args.out), payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "out": str(args.out),
                "summary": {
                    key: payload["summary"][key]
                    for key in (
                        "transfer_count",
                        "candidate_count",
                        "below_rho_accepted_candidate_count",
                        "transfers_with_below_rho_export_count",
                        "heldout_below_rho_transfer_count",
                        "transfers_without_below_rho_export",
                        "best_heldout_transfer_index",
                        "best_heldout_leaf_index",
                        "best_heldout_ops_over_rho",
                    )
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
