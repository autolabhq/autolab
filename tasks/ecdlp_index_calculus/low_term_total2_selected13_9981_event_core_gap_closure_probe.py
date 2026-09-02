#!/usr/bin/env python3
"""Close the selected13 9981 public-prefix gap by minimizing event cores.

The min/transfer probe showed that the public-prefix selector family transfers
to held-out 9981, but its full low-term-span top-k rule was just over rho. This
follow-up takes only the event-bearing leaves from accepted 9981 transfer
candidates, enumerates compact per-salt subsets, and replays them through the
same verifier path.

This is still a bounded selected13 verifier artifact. A below-rho held-out
event core is useful portability evidence, not a proof of a generalized ECDLP
index-calculus algorithm.
"""

from __future__ import annotations

import argparse
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
import low_term_total2_selected13_public_prefix_min_transfer_probe as min_transfer


SCHEMA = "ecdlp.low_term_total2_selected13_9981_event_core_gap_closure_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_MIN_TRANSFER = DEFAULT_STATE_DIR / "low_term_total2_selected13_public_prefix_min_transfer_probe.json"
DEFAULT_DIRECT_AUDIT = DEFAULT_STATE_DIR / "low_term_total2_selected13_direct_verification_audit_9981_9943_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_9981_event_core_gap_closure_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_9981_event_core_gap_closure_probe.h"

TARGET = min_transfer.TARGET
TRANSFER_INDEX = 9981
ROW_ORDER = min_transfer.TRANSFER_ROWS[TRANSFER_INDEX]

CLASS_CODES = {
    "accepted_candidate_event_core_subset": 1,
    "accepted_event_union_subset": 2,
}


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


def nonempty_subsets(values: set[int]) -> list[set[int]]:
    ordered = sorted(values)
    out = []
    for mask in range(1, 1 << len(ordered)):
        out.append({ordered[index] for index in range(len(ordered)) if mask & (1 << index)})
    out.sort(key=lambda item: (len(item), sorted(item)))
    return out


def event_leaf_map_from_candidate(candidate: dict[str, Any]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {row_key: set() for row_key in ROW_ORDER}
    for row in candidate.get("row_summaries") or []:
        row_key = str(row.get("row_key") or "")
        if row_key not in out:
            continue
        for event in row.get("event_summaries") or []:
            if event.get("leaf_index") is not None:
                out[row_key].add(as_int(event.get("leaf_index")))
    return {row_key: leaves for row_key, leaves in out.items() if leaves}


def compact_leaf_map(raw: dict[str, set[int]]) -> list[dict[str, Any]]:
    return min_transfer.compact_leaf_map(raw, ROW_ORDER)


def leaf_signature(raw: dict[str, set[int]]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return min_transfer.leaf_signature(raw, ROW_ORDER)


def accepted_9981_candidates(min_transfer_payload: dict[str, Any]) -> list[dict[str, Any]]:
    accepted = [
        candidate
        for candidate in min_transfer_payload.get("min_transfer_candidates") or []
        if as_int(candidate.get("transfer_index"), -1) == TRANSFER_INDEX
        and candidate.get("accepted_relation_export")
    ]
    accepted.sort(
        key=lambda item: (
            as_float(item.get("ops_over_rho")) if as_float(item.get("ops_over_rho")) is not None else 999.0,
            as_int(item.get("selected_leaf_count")),
            str(item.get("candidate_id")),
        )
    )
    return accepted


def build_seed_cores(min_transfer_payload: dict[str, Any]) -> list[dict[str, Any]]:
    accepted = accepted_9981_candidates(min_transfer_payload)
    seeds = []
    union: dict[str, set[int]] = {row_key: set() for row_key in ROW_ORDER}
    seen_seed_maps: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
    for candidate in accepted:
        leaf_map = event_leaf_map_from_candidate(candidate)
        if len(leaf_map) != len(ROW_ORDER):
            continue
        for row_key in ROW_ORDER:
            union[row_key].update(leaf_map.get(row_key, set()))
        signature = leaf_signature(leaf_map)
        if signature in seen_seed_maps:
            continue
        seen_seed_maps.add(signature)
        seeds.append(
            {
                "candidate_class": "accepted_candidate_event_core_subset",
                "candidate_class_code": CLASS_CODES["accepted_candidate_event_core_subset"],
                "parent_candidate_id": candidate.get("candidate_id"),
                "parent_mode": candidate.get("mode"),
                "parent_ops_over_rho": round_or_none(candidate.get("ops_over_rho")),
                "parent_top_k": candidate.get("top_k"),
                "seed_label": f"{candidate.get('mode')}_k{candidate.get('top_k')}_core",
                "seed_leaf_map": compact_leaf_map(leaf_map),
            }
        )
    if all(union.get(row_key) for row_key in ROW_ORDER):
        seeds.append(
            {
                "candidate_class": "accepted_event_union_subset",
                "candidate_class_code": CLASS_CODES["accepted_event_union_subset"],
                "parent_candidate_id": None,
                "parent_mode": "accepted_event_union",
                "parent_ops_over_rho": None,
                "parent_top_k": None,
                "seed_label": "accepted_event_union",
                "seed_leaf_map": compact_leaf_map(union),
            }
        )
    return seeds


def build_candidate_inputs(seed_cores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
    for seed in seed_cores:
        seed_map = min_transfer.row_leaf_map(seed.get("seed_leaf_map") or [])
        if len(seed_map) != len(ROW_ORDER):
            continue
        per_row_subsets = [nonempty_subsets(seed_map[row_key]) for row_key in ROW_ORDER]
        for left in per_row_subsets[0]:
            for right in per_row_subsets[1]:
                leaf_map = {ROW_ORDER[0]: set(left), ROW_ORDER[1]: set(right)}
                signature = leaf_signature(leaf_map)
                if signature in seen:
                    continue
                seen.add(signature)
                out.append(
                    {
                        "candidate_class": seed["candidate_class"],
                        "candidate_class_code": seed["candidate_class_code"],
                        "parent_candidate_id": seed.get("parent_candidate_id"),
                        "parent_mode": seed.get("parent_mode"),
                        "parent_top_k": seed.get("parent_top_k"),
                        "seed_label": seed.get("seed_label"),
                        "selected_leaf_count": sum(len(items) for items in leaf_map.values()),
                        "selected_leaf_map": compact_leaf_map(leaf_map),
                        "transfer_index": TRANSFER_INDEX,
                    }
                )
    out.sort(
        key=lambda item: (
            as_int(item.get("selected_leaf_count")),
            json.dumps(item.get("selected_leaf_map"), sort_keys=True),
            str(item.get("seed_label")),
        )
    )
    return out


def materialize_9981_contexts(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
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
    contexts, errors = min_transfer.materialize_contexts_for(
        verifier,
        records,
        config_source,
        specs_by_target,
        TRANSFER_INDEX,
        replay_args,
        {},
    )
    return contexts, errors, radius


def compact_candidate(index: int, candidate: dict[str, Any], result: dict[str, Any], row_events: list[tuple[str, dict[str, Any]]], event_limit: int) -> dict[str, Any]:
    record = min_transfer.compact_candidate(index, candidate, result, row_events, event_limit)
    record["candidate_id"] = f"event_core_{min_transfer.digest_u64([candidate, index]):016x}"
    record["candidate_id_u64"] = min_transfer.digest_u64([candidate, index])
    return record


def replay_candidates(args: argparse.Namespace, candidates: list[dict[str, Any]], contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    verifier = replay_probe.relation_probe.load_verifier_module()
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    records = []
    for index, candidate in enumerate(candidates):
        leaf_map = min_transfer.row_leaf_map(candidate.get("selected_leaf_map") or [])
        result, row_events = replay_probe.replay_selection(verifier, leaf_map, contexts, scan_cache, args.event_summary_limit)
        records.append(compact_candidate(index, candidate, result, row_events, args.event_summary_limit))
    return records


def best_record(records: list[dict[str, Any]], require_below: bool = False) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if record.get("accepted_relation_export") and (not require_below or record.get("below_rho"))
    ]
    candidates.sort(
        key=lambda item: (
            as_float(item.get("ops_over_rho")) if as_float(item.get("ops_over_rho")) is not None else 999.0,
            as_int(item.get("selected_leaf_count")),
            -as_int(item.get("rank")),
            str(item.get("candidate_id")),
        )
    )
    return candidates[0] if candidates else {}


def summarize(records: list[dict[str, Any]], failures: list[dict[str, Any]], source_payload: dict[str, Any]) -> dict[str, Any]:
    accepted = [row for row in records if row.get("accepted_relation_export")]
    below = [row for row in accepted if row.get("below_rho")]
    best = best_record(records)
    best_below = best_record(records, require_below=True)
    statuses = Counter(str(row.get("status")) for row in records)
    source_summary = source_payload.get("summary") or {}
    return {
        "accepted_relation_export_count": len(accepted),
        "below_rho_accepted_relation_export_count": len(below),
        "best_below_rho_candidate_id": best_below.get("candidate_id"),
        "best_below_rho_derived_secret": best_below.get("derived_secret"),
        "best_below_rho_ops_over_rho": best_below.get("ops_over_rho"),
        "best_below_rho_rank": best_below.get("rank"),
        "best_below_rho_relation_count": best_below.get("relation_count"),
        "best_below_rho_selected_leaf_count": best_below.get("selected_leaf_count"),
        "best_below_rho_selected_leaf_map": best_below.get("selected_leaf_map"),
        "best_candidate_id": best.get("candidate_id"),
        "best_derived_secret": best.get("derived_secret"),
        "best_ops_over_rho": best.get("ops_over_rho"),
        "best_selected_leaf_count": best.get("selected_leaf_count"),
        "candidate_count": len(records),
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "heldout_9981_below_rho_relation_derived": bool(best_below),
        "heldout_9981_relation_derived": bool(best),
        "source_best_over_rho_candidate_id": source_summary.get("transfer_9981_best_candidate_id"),
        "source_best_over_rho_ops_over_rho": source_summary.get("transfer_9981_best_ops_over_rho"),
        "source_claim_status": source_payload.get("claim_status"),
        "status_counts": dict(sorted(statuses.items())),
        "target_level_pollard_rho_speedup_claimed": bool(best_below),
        "verified": not failures,
        "worker_interpretation": (
            "The accepted 9981 public-prefix transfer closes below rho after event-core minimization. "
            "This is held-out target-level evidence, not a generalized ECDLP algorithm proof."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_FAILED"
    if summary.get("heldout_9981_below_rho_relation_derived"):
        return "SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_BELOW_RHO_EXPORT"
    if summary.get("heldout_9981_relation_derived"):
        return "SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_OVER_RHO_ONLY"
    return "SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_NO_EXPORT"


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
            f"{as_int(record.get('selected_leaf_count'))}ULL, "
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
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_PROBE_H

#include <stdint.h>

#define SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_CANDIDATE_COUNT {len(records)}
#define SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_ACCEPTED_COUNT {accepted_count}
#define SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_BELOW_RHO_ACCEPTED_COUNT {below_count}

typedef struct {{
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t selected_leaf_count;
  uint64_t below_rho;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t rank;
  uint64_t relation_count;
  uint64_t derived_secret;
  uint64_t ops_over_rho_scaled_1e6;
  uint64_t status_code;
}} selected13_9981_event_core_gap_closure_candidate_t;

static const selected13_9981_event_core_gap_closure_candidate_t SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_CANDIDATES[] = {{
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
      sizeof(SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_CANDIDATES) / sizeof(SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_CANDIDATES[0]);
  uint64_t accepted = 0;
  uint64_t accepted_below = 0;
  uint64_t best_below_ops = 0ULL;

  if (candidate_count != SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_CANDIDATE_COUNT) failure_count++;
  if (candidate_count == 0ULL) failure_count++;

  for (size_t i = 0; i < candidate_count; i++) {{
    const selected13_9981_event_core_gap_closure_candidate_t *candidate = &SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_CANDIDATES[i];
    if (candidate->candidate_id_u64 == 0ULL) failure_count++;
    if (candidate->selected_leaf_count == 0ULL) failure_count++;
    if (candidate->status_code == 0ULL) failure_count++;
    if (candidate->relation_derived_ecdlp && !candidate->public_key_verified) failure_count++;
    if (candidate->relation_derived_ecdlp && candidate->derived_secret == 0ULL) failure_count++;
    if (candidate->relation_derived_ecdlp) {{
      accepted++;
      if (candidate->below_rho) {{
        accepted_below++;
        if (best_below_ops == 0ULL || candidate->ops_over_rho_scaled_1e6 < best_below_ops) {{
          best_below_ops = candidate->ops_over_rho_scaled_1e6;
        }}
      }}
    }}
  }}

  if (accepted != SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_ACCEPTED_COUNT) failure_count++;
  if (accepted_below != SELECTED13_9981_EVENT_CORE_GAP_CLOSURE_BELOW_RHO_ACCEPTED_COUNT) failure_count++;
  if (accepted_below == 0ULL) failure_count++;
  if (best_below_ops == 0ULL || best_below_ops >= 1000000ULL) failure_count++;

  printf("selected13_9981_event_core_gap_closure_preflight candidates=%llu accepted=%llu accepted_below=%llu best_below_ops_scaled_1e6=%llu failures=%llu\\n",
         (unsigned long long)candidate_count,
         (unsigned long long)accepted,
         (unsigned long long)accepted_below,
         (unsigned long long)best_below_ops,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_9981_event_core_gap_closure_preflight_") as tmp:
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
    min_transfer_payload = load_json(Path(args.min_transfer))
    direct_audit = load_json(Path(args.direct_audit))
    failures: list[dict[str, Any]] = []
    if min_transfer_payload.get("claim_status") != "SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_HELDOUT_OVER_RHO":
        failures.append({"code": "min_transfer_status_unexpected", "claim_status": min_transfer_payload.get("claim_status")})
    if direct_audit.get("claim_status") != "SELECTED13_DIRECT_VERIFICATION_REQUIRES_FRESH_RELATIONS":
        failures.append({"code": "direct_audit_status_unexpected", "claim_status": direct_audit.get("claim_status")})

    seed_cores = build_seed_cores(min_transfer_payload)
    if not seed_cores:
        failures.append({"code": "no_accepted_9981_event_cores"})
    candidate_inputs = build_candidate_inputs(seed_cores)
    contexts, context_errors, radius = materialize_9981_contexts(args)
    for error in context_errors:
        failures.append({"code": "context_materialization_error", "error": error})
    records = [] if context_errors else replay_candidates(args, candidate_inputs, contexts)
    summary = summarize(records, failures, min_transfer_payload)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, summary),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "config_source": str(Path(args.config_source)),
            "context_top_k": args.context_top_k,
            "direct_audit": str(Path(args.direct_audit)),
            "direct_source": str(Path(args.direct_source)),
            "event_summary_limit": args.event_summary_limit,
            "min_transfer": str(Path(args.min_transfer)),
            "radius": radius,
            "target": TARGET,
            "transfer_index": TRANSFER_INDEX,
            "transfer_rows": ROW_ORDER,
            "transfer_source": str(Path(args.transfer_source)),
        },
        "seed_event_cores": seed_cores,
        "summary": summary,
        "event_core_candidates": records,
        "failures": failures,
        "honesty_boundary": {
            "general_ecdlp_algorithm_claimed": False,
            "heldout_9981_below_rho_relation_derived": summary["heldout_9981_below_rho_relation_derived"],
            "scope": "bounded selected13 9981 event-core replay under the existing preassociation cost model",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-transfer", type=Path, default=DEFAULT_MIN_TRANSFER)
    parser.add_argument("--direct-audit", type=Path, default=DEFAULT_DIRECT_AUDIT)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
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
    parser.add_argument("--event-summary-limit", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["event_core_candidates"]))
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
                "summary": payload["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
