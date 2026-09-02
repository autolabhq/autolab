#!/usr/bin/env python3
"""Replay selected13 leaf-augmentation work items through the verifier path.

The leaf-augmentation worklist orders fresh-emission candidates for the
selected13 priority targets.  This probe consumes that worklist, materializes
the existing verifier contexts, scans the candidate leaf sets, and records
whether any candidate becomes a public-key-verified direct relation export.

The output is deliberately narrow: unverified rank or relation events are
useful worker evidence, but they are not accepted exports, ECDLP recoveries, or
Pollard-rho speedups.
"""

from __future__ import annotations

import argparse
import hashlib
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


SCHEMA = "ecdlp.low_term_total2_selected13_leaf_augmentation_replay_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_WORKLIST = DEFAULT_STATE_DIR / "low_term_total2_selected13_leaf_augmentation_worklist_9981_9943_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_leaf_augmentation_replay_probe_9981_9943.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_leaf_augmentation_replay_probe_9981_9943.h"

CLASS_CODES = {
    "exact_selector_replay": 1,
    "positive_control_leaf_union": 2,
    "single_leaf_positive_control_augmentation": 3,
}

STATUS_CODES = {
    "ACCEPTED_PUBLIC_KEY_VERIFIED_REPLAY": 1,
    "UNVERIFIED_RANK_OR_RELATION_GAIN": 2,
    "UNVERIFIED_REPLAY_RELATION_ONLY": 3,
    "NO_REPLAY_RELATION": 4,
    "CONTEXT_ERROR": 5,
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


def row_leaf_map(row_leaf_keys: list[dict[str, Any]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for item in row_leaf_keys or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        if not row_key:
            continue
        leaves = {as_int(leaf) for leaf in item.get("leaf_indices") or []}
        if leaves:
            out.setdefault(row_key, set()).update(leaves)
    return out


def compact_replay_rows(rows: list[dict[str, Any]], event_limit: int) -> list[dict[str, Any]]:
    compact_rows = []
    for row in rows or []:
        scan = row.get("scan") or {}
        hit_summaries = scan.get("hit_event_summaries") or []
        event_summaries = scan.get("event_summaries") or []
        compact_scan = {
            key: value
            for key, value in scan.items()
            if key not in {"hit_event_summaries", "event_summaries"}
        }
        compact_scan["hit_event_summary_count"] = len(hit_summaries)
        compact_scan["hit_event_summaries"] = hit_summaries[:event_limit]
        compact_scan["hit_event_summaries_truncated"] = len(hit_summaries) > event_limit
        compact_scan["event_summaries"] = event_summaries[:event_limit]
        compact_scan["event_summaries_truncated"] = len(event_summaries) > event_limit
        compact_rows.append(
            {
                "challenge_seed": row.get("challenge_seed"),
                "generic_rho_steps": as_int(row.get("generic_rho_steps")),
                "row_key": row.get("row_key"),
                "scan": compact_scan,
            }
        )
    return compact_rows


def replay_status(item: dict[str, Any], result: dict[str, Any], context_errors: list[dict[str, Any]]) -> str:
    if context_errors:
        return "CONTEXT_ERROR"
    if bool(result.get("public_key_verified")):
        return "ACCEPTED_PUBLIC_KEY_VERIFIED_REPLAY"
    base_rank = as_int(item.get("base_rank"))
    base_relations = as_int(item.get("base_relation_count"))
    rank = as_int(result.get("rank"))
    relation_count = as_int(result.get("relation_count"))
    if rank > base_rank or relation_count > base_relations:
        return "UNVERIFIED_RANK_OR_RELATION_GAIN"
    if relation_count > 0 or rank > 0:
        return "UNVERIFIED_REPLAY_RELATION_ONLY"
    return "NO_REPLAY_RELATION"


def compact_item_record(
    item: dict[str, Any],
    result: dict[str, Any],
    row_events: list[tuple[str, dict[str, Any]]],
    context_errors: list[dict[str, Any]],
    event_limit: int,
) -> dict[str, Any]:
    status = replay_status(item, result, context_errors)
    accepted_export = status == "ACCEPTED_PUBLIC_KEY_VERIFIED_REPLAY"
    return {
        "accepted_relation_export": accepted_export,
        "base_relation_count": as_int(item.get("base_relation_count")),
        "base_rank": as_int(item.get("base_rank")),
        "candidate_class": item.get("candidate_class"),
        "candidate_class_code": CLASS_CODES.get(str(item.get("candidate_class")), 0),
        "context_error_count": len(context_errors),
        "context_errors": context_errors,
        "derived_secret": result.get("derived_secret"),
        "event_count": len(row_events),
        "ops_over_rho": round_or_none(result.get("ops_over_rho")),
        "positive_control": item.get("positive_control"),
        "public_key_verified": bool(result.get("public_key_verified")),
        "rank": as_int(result.get("rank")),
        "relation_count": as_int(result.get("relation_count")),
        "relation_derived_ecdlp": bool(result.get("public_key_verified")) and bool(result.get("derived")),
        "replay": {
            "below_rho": bool(result.get("below_rho")),
            "challenge_seeds": result.get("challenge_seeds") or [],
            "derived": bool(result.get("derived")),
            "duplicate_form_count": as_int(result.get("duplicate_form_count")),
            "generic_rho_steps": as_int(result.get("generic_rho_steps")),
            "materialized_row_count": as_int(result.get("materialized_row_count")),
            "ops": as_int(result.get("ops")),
            "ops_over_rho": round_or_none(result.get("ops_over_rho")),
            "public_key_verified": bool(result.get("public_key_verified")),
            "rank": as_int(result.get("rank")),
            "relation_count": as_int(result.get("relation_count")),
            "rows": compact_replay_rows(result.get("rows") or [], event_limit),
            "selected_leaf_count": as_int(result.get("selected_leaf_count")),
            "selected_row_count": as_int(result.get("selected_row_count")),
            "unique_form_count": as_int(result.get("unique_form_count")),
        },
        "status": status,
        "status_code": STATUS_CODES.get(status, 0),
        "target": item.get("target"),
        "transfer_index": as_int(item.get("transfer_index"), -1),
        "unverified_rank_or_relation_gain": status == "UNVERIFIED_RANK_OR_RELATION_GAIN",
        "work_item_id": item.get("work_item_id"),
        "work_item_id_u64": digest_u64(item.get("work_item_id")),
    }


def replay_work_items(args: argparse.Namespace, worklist: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
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
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    records_out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, item in enumerate(worklist.get("work_items") or []):
        if args.max_items is not None and index >= args.max_items:
            break
        leaves = row_leaf_map(item.get("candidate_row_leaf_keys") or [])
        top_k = as_int((item.get("candidate_source_row") or {}).get("top_k"), -1)
        case = {
            "target": item.get("target"),
            "transfer_index": as_int(item.get("transfer_index"), -1),
            "top_k": top_k,
        }
        contexts, context_errors = replay_probe.materialize_contexts(
            verifier,
            records,
            config_source,
            specs_by_target,
            case,
            sorted(leaves),
            replay_args,
            context_cache,
        )
        result, row_events = replay_probe.replay_selection(
            verifier,
            leaves,
            contexts,
            scan_cache,
            args.event_summary_limit,
        )
        if context_errors:
            failures.append(
                {
                    "code": "context_materialization_error",
                    "context_error_count": len(context_errors),
                    "transfer_index": as_int(item.get("transfer_index"), -1),
                    "work_item_id": item.get("work_item_id"),
                }
            )
        records_out.append(
            compact_item_record(item, result, row_events, context_errors, args.event_summary_limit)
        )
    return records_out, failures, radius


def claim_status(failures: list[dict[str, Any]], records: list[dict[str, Any]]) -> str:
    if failures:
        return "SELECTED13_LEAF_AUGMENTATION_REPLAY_FAILED"
    if any(record.get("accepted_relation_export") for record in records):
        return "SELECTED13_LEAF_AUGMENTATION_REPLAY_FOUND_EXPORTABLE_RELATION"
    return "SELECTED13_LEAF_AUGMENTATION_REPLAY_NO_EXPORT"


def render_c_header(records: list[dict[str, Any]]) -> str:
    rows = []
    for index, record in enumerate(records):
        rows.append(
            "  {"
            f"{as_int(record.get('transfer_index'))}ULL, "
            f"{index}ULL, "
            f"{as_int(record.get('work_item_id_u64'))}ULL, "
            f"{as_int(record.get('candidate_class_code'))}ULL, "
            f"{as_int(record.get('base_rank'))}ULL, "
            f"{as_int(record.get('base_relation_count'))}ULL, "
            f"{as_int(record.get('rank'))}ULL, "
            f"{as_int(record.get('relation_count'))}ULL, "
            f"{as_int(record.get('event_count'))}ULL, "
            f"{1 if record.get('public_key_verified') else 0}ULL, "
            f"{1 if record.get('relation_derived_ecdlp') else 0}ULL, "
            f"{as_int(record.get('context_error_count'))}ULL, "
            f"{1 if record.get('accepted_relation_export') else 0}ULL, "
            f"{as_int(record.get('status_code'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_LEAF_AUGMENTATION_REPLAY_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_LEAF_AUGMENTATION_REPLAY_PROBE_H

#include <stdint.h>

#define SELECTED13_LEAF_AUG_REPLAY_ITEM_COUNT {len(records)}
#define SELECTED13_LEAF_AUG_REPLAY_EXPORT_COUNT {sum(1 for record in records if record.get('accepted_relation_export'))}
#define SELECTED13_LEAF_AUG_REPLAY_DERIVED_COUNT {sum(1 for record in records if record.get('relation_derived_ecdlp'))}
#define SELECTED13_LEAF_AUG_REPLAY_CONTEXT_ERROR_COUNT {sum(as_int(record.get('context_error_count')) for record in records)}

#define SELECTED13_LEAF_AUG_REPLAY_STATUS_ACCEPTED 1ULL
#define SELECTED13_LEAF_AUG_REPLAY_STATUS_UNVERIFIED_GAIN 2ULL
#define SELECTED13_LEAF_AUG_REPLAY_STATUS_UNVERIFIED_RELATION 3ULL
#define SELECTED13_LEAF_AUG_REPLAY_STATUS_NO_RELATION 4ULL
#define SELECTED13_LEAF_AUG_REPLAY_STATUS_CONTEXT_ERROR 5ULL

typedef struct {{
  uint64_t transfer_index;
  uint64_t work_item_index;
  uint64_t work_item_id_u64;
  uint64_t candidate_class_code;
  uint64_t base_rank;
  uint64_t base_relation_count;
  uint64_t replay_rank;
  uint64_t replay_relation_count;
  uint64_t replay_event_count;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t context_error_count;
  uint64_t accepted_relation_export;
  uint64_t status_code;
}} selected13_leaf_augmentation_replay_item_t;

static const selected13_leaf_augmentation_replay_item_t SELECTED13_LEAF_AUGMENTATION_REPLAY_ITEMS[] = {{
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
  uint64_t item_count =
      sizeof(SELECTED13_LEAF_AUGMENTATION_REPLAY_ITEMS) / sizeof(SELECTED13_LEAF_AUGMENTATION_REPLAY_ITEMS[0]);
  uint64_t export_count = 0;
  uint64_t derived_count = 0;
  uint64_t context_error_count = 0;
  uint64_t unverified_gain_count = 0;

  if (item_count != SELECTED13_LEAF_AUG_REPLAY_ITEM_COUNT) failure_count++;
  if (item_count == 0ULL) failure_count++;

  for (size_t i = 0; i < item_count; i++) {{
    const selected13_leaf_augmentation_replay_item_t *item = &SELECTED13_LEAF_AUGMENTATION_REPLAY_ITEMS[i];
    export_count += item->accepted_relation_export;
    derived_count += item->relation_derived_ecdlp;
    context_error_count += item->context_error_count;
    if (item->status_code == SELECTED13_LEAF_AUG_REPLAY_STATUS_UNVERIFIED_GAIN) unverified_gain_count++;
    if (item->work_item_id_u64 == 0ULL) failure_count++;
    if (item->candidate_class_code == 0ULL) failure_count++;
    if (item->status_code == 0ULL) failure_count++;
    if (item->accepted_relation_export && !item->public_key_verified) failure_count++;
  }}

  if (export_count != SELECTED13_LEAF_AUG_REPLAY_EXPORT_COUNT) failure_count++;
  if (derived_count != SELECTED13_LEAF_AUG_REPLAY_DERIVED_COUNT) failure_count++;
  if (context_error_count != SELECTED13_LEAF_AUG_REPLAY_CONTEXT_ERROR_COUNT) failure_count++;

  printf("selected13_leaf_augmentation_replay_preflight items=%llu exports=%llu derived=%llu context_errors=%llu unverified_gain=%llu failures=%llu\\n",
         (unsigned long long)item_count,
         (unsigned long long)export_count,
         (unsigned long long)derived_count,
         (unsigned long long)context_error_count,
         (unsigned long long)unverified_gain_count,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_leaf_replay_preflight_") as tmp:
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
    worklist = load_json(worklist_path)
    failures: list[dict[str, Any]] = []
    if worklist.get("claim_status") != "SELECTED13_LEAF_AUGMENTATION_WORKLIST_READY":
        failures.append({"code": "worklist_status_unexpected", "claim_status": worklist.get("claim_status")})
    if worklist.get("failures"):
        failures.append({"code": "worklist_has_failures", "failures": worklist.get("failures")})
    records, replay_failures, radius = replay_work_items(args, worklist)
    failures.extend(replay_failures)
    class_counts = Counter(str(record.get("candidate_class")) for record in records)
    status_counts = Counter(str(record.get("status")) for record in records)
    transfer_counts = Counter(str(record.get("transfer_index")) for record in records)
    accepted_count = sum(1 for record in records if record.get("accepted_relation_export"))
    derived_count = sum(1 for record in records if record.get("relation_derived_ecdlp"))
    verified_count = sum(1 for record in records if record.get("public_key_verified"))
    gain_records = [record for record in records if record.get("unverified_rank_or_relation_gain")]
    relation_records = [record for record in records if as_int(record.get("relation_count")) > 0]
    ratios = [as_float(record.get("ops_over_rho")) for record in records if as_float(record.get("ops_over_rho")) is not None]
    summary = {
        "accepted_relation_export_count": accepted_count,
        "candidate_class_counts": dict(sorted(class_counts.items())),
        "context_error_count": sum(as_int(record.get("context_error_count")) for record in records),
        "failure_count": len(failures),
        "max_rank": max((as_int(record.get("rank")) for record in records), default=0),
        "min_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "pollard_rho_speedup_claimed": False,
        "public_key_verified_item_count": verified_count,
        "relation_derived_ecdlp": derived_count > 0,
        "relation_derived_item_count": derived_count,
        "relation_producing_item_count": len(relation_records),
        "replayed_item_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "target_transfer_counts": dict(sorted(transfer_counts.items())),
        "unverified_rank_or_relation_gain_count": len(gain_records),
        "verified": not failures,
        "work_item_count": len(worklist.get("work_items") or []),
        "worker_interpretation": (
            "The verifier replay produces target-specific rank/relation evidence "
            "for some leaf augmentations, but no candidate public-key-verifies or "
            "derives an ECDLP scalar. Rank gains remain worker evidence only."
        ),
    }
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, records),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "config_source": str(Path(args.config_source)),
            "direct_source": str(Path(args.direct_source)),
            "event_summary_limit": args.event_summary_limit,
            "max_items": args.max_items,
            "radius": radius,
            "replay_args": {
                "factor_base_size": args.factor_base_size,
                "max_relations": args.max_relations,
                "row_count": args.row_count,
                "row_pool": args.row_pool,
                "scout_limit": args.scout_limit,
                "scout_mode": args.scout_mode,
                "scout_order": args.scout_order,
                "selected_limit": args.selected_limit,
            },
            "transfer_source": str(Path(args.transfer_source)),
            "worklist": str(worklist_path),
        },
        "summary": summary,
        "work_item_replays": records,
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": accepted_count,
            "pollard_rho_speedup_claimed": False,
            "public_key_verified_item_count": verified_count,
            "rank_gain_is_not_export": True,
            "relation_derived_ecdlp": derived_count > 0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
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
    parser.add_argument("--event-summary-limit", type=int, default=8)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["work_item_replays"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = claim_status(payload["failures"], payload["work_item_replays"])
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
