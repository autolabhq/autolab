#!/usr/bin/env python3
"""Diagnose misses from the all-disjoint selected13 second-stage validation.

The frozen no-leaf rule recovers many disjoint selected13-like transfers with a
single replayed leaf. This probe replays every active-scout top-5 leaf for the
misses from that validation artifact, separating rule-selection misses from
shortlist/materialization misses.
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
import low_term_total2_selected13_common_leaf_pair_sweep_probe as common_sweep
import low_term_total2_selected13_public_prefix_min_transfer_probe as min_transfer
import low_term_total2_selected13_salt_conditioned_direct_replay_probe as direct_replay
import low_term_total2_selected13_second_stage_disjoint_validation_probe as disjoint_validation


SCHEMA = "ecdlp.low_term_total2_selected13_second_stage_disjoint_miss_diagnostic_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SOURCE_VALIDATION = DEFAULT_STATE_DIR / "low_term_total2_selected13_second_stage_disjoint_validation_all_probe.json"
DEFAULT_TRAINING_CONTRACT = disjoint_validation.DEFAULT_TRAINING_CONTRACT
DEFAULT_VALIDATION_MANIFEST = disjoint_validation.DEFAULT_VALIDATION_MANIFEST
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_second_stage_disjoint_miss_diagnostic_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_second_stage_disjoint_miss_diagnostic_probe.h"

TARGET = min_transfer.TARGET
DEFAULT_SHORTLIST_MODE = disjoint_validation.DEFAULT_SHORTLIST_MODE
DEFAULT_SHORTLIST_K = disjoint_validation.DEFAULT_SHORTLIST_K


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


def source_miss_transfers(source_payload: dict[str, Any]) -> list[int]:
    summary = source_payload.get("summary") if isinstance(source_payload, dict) else {}
    if not isinstance(summary, dict):
        return []
    return [as_int(transfer) for transfer in summary.get("missing_below_rho_transfers") or []]


def source_candidate_transfers(source_payload: dict[str, Any], *, diagnose_all: bool) -> list[int]:
    summary = source_payload.get("summary") if isinstance(source_payload, dict) else {}
    if not isinstance(summary, dict):
        return []
    if diagnose_all:
        transfers = summary.get("selected_validation_transfers") or []
        if not transfers:
            transfers = sorted(
                {
                    as_int(record.get("transfer_index"))
                    for record in source_payload.get("direct_replay_records") or []
                }
            )
        return [as_int(transfer) for transfer in transfers]
    return source_miss_transfers(source_payload)


def source_records_by_transfer(source_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = source_payload.get("direct_replay_records") if isinstance(source_payload, dict) else []
    out: dict[int, dict[str, Any]] = {}
    for record in records or []:
        out[as_int(record.get("transfer_index"))] = record
    return out


def replay_miss_shortlists(
    args: argparse.Namespace,
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
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
    verifier_records = verifier.load_records()
    replay_args = disjoint_validation.build_replay_args(args)
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    records_out: list[dict[str, Any]] = []
    replay_index = 0
    for target in targets:
        contexts, errors = common_sweep.materialize_contexts_for_target(
            verifier,
            verifier_records,
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
        ranked = direct_replay.ranked_leaf_records(target, contexts, args.shortlist_mode, args.leaf_limit)
        shortlist = ranked[: max(1, args.shortlist_k)]
        for position, association_record in enumerate(shortlist):
            leaf = as_int(association_record.get("leaf_index"))
            row_leaves = {str(row_key): {leaf} for row_key in target.get("row_keys") or []}
            result, row_events = replay_probe.replay_selection(
                verifier,
                row_leaves,
                contexts,
                scan_cache,
                args.event_summary_limit,
            )
            replay_record = direct_replay.direct_replay_record(
                replay_index,
                target,
                [association_record],
                result,
                row_events,
                args.event_summary_limit,
                args.shortlist_mode,
                1,
            )
            replay_record["manifest_queue_position"] = as_int(target.get("manifest_queue_position"))
            replay_record["manifest_range"] = target.get("manifest_range")
            replay_record["manifest_selector"] = target.get("manifest_selector")
            replay_record["shortlist_position"] = position
            replay_record["shortlist_mode"] = args.shortlist_mode
            records_out.append(replay_record)
            replay_index += 1
    return records_out, radius, failures


def accepted_below(record: dict[str, Any]) -> bool:
    return bool(record.get("accepted_relation_export")) and bool(record.get("below_rho"))


def best_below(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [record for record in records if accepted_below(record)]
    candidates.sort(
        key=lambda record: (
            as_float(record.get("ops_over_rho")) if as_float(record.get("ops_over_rho")) is not None else 999.0,
            as_int(record.get("shortlist_position")),
            as_int((record.get("selected_leaf_indices") or [0])[0]),
        )
    )
    return candidates[0] if candidates else {}


def per_transfer_diagnostics(
    replay_records: list[dict[str, Any]],
    source_by_transfer: dict[int, dict[str, Any]],
    transfers: list[int],
) -> list[dict[str, Any]]:
    by_transfer: dict[int, list[dict[str, Any]]] = {}
    for record in replay_records:
        by_transfer.setdefault(as_int(record.get("transfer_index")), []).append(record)
    out = []
    for transfer in sorted(transfers):
        records = sorted(by_transfer.get(transfer, []), key=lambda record: as_int(record.get("shortlist_position")))
        source_record = source_by_transfer.get(transfer, {})
        frozen_leaf = as_int((source_record.get("selected_leaf_indices") or [0])[0])
        best = best_below(records)
        accepted = [record for record in records if accepted_below(record)]
        out.append(
            {
                "accepted_below_rho_leaf_count": len(accepted),
                "accepted_below_rho_leaves": [
                    as_int((record.get("selected_leaf_indices") or [0])[0])
                    for record in accepted
                ],
                "best_alt_derived_secret": best.get("derived_secret"),
                "best_alt_leaf": as_int((best.get("selected_leaf_indices") or [0])[0]) if best else None,
                "best_alt_ops_over_rho": best.get("ops_over_rho"),
                "best_alt_shortlist_position": best.get("shortlist_position"),
                "frozen_leaf": frozen_leaf,
                "frozen_predicate": source_record.get("second_stage_predicate"),
                "frozen_shortlist_position": source_record.get("shortlist_position"),
                "frozen_status": source_record.get("status"),
                "shortlist_has_below_rho": bool(accepted),
                "shortlist_replay_count": len(records),
                "status_counts": dict(sorted(Counter(str(record.get("status")) for record in records).items())),
                "transfer_index": transfer,
            }
        )
    return out


def summarize(
    transfers: list[int],
    source_misses: list[int],
    replay_records: list[dict[str, Any]],
    per_transfer: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    training_transfers: set[int],
    source_mode: str,
) -> dict[str, Any]:
    accepted_records = [record for record in replay_records if accepted_below(record)]
    rescue_transfers = [
        as_int(row.get("transfer_index"))
        for row in per_transfer
        if row.get("shortlist_has_below_rho")
    ]
    no_shortlist = [
        as_int(row.get("transfer_index"))
        for row in per_transfer
        if not row.get("shortlist_has_below_rho")
    ]
    overlap = sorted(set(transfers) & training_transfers)
    status_counts = Counter(str(record.get("status")) for record in replay_records)
    return {
        "all_source_misses_diagnosed": len(per_transfer) == len(transfers),
        "diagnostic_source_mode": source_mode,
        "diagnostic_transfer_count": len(transfers),
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "missing_top5_below_rho_transfers": sorted(no_shortlist),
        "shortlist_below_rho_replay_count": len(accepted_records),
        "shortlist_below_rho_transfer_count": len(rescue_transfers),
        "shortlist_below_rho_transfers": sorted(rescue_transfers),
        "shortlist_replay_count": len(replay_records),
        "source_diagnostic_transfers": sorted(transfers),
        "source_missing_transfers": sorted(source_misses),
        "status_counts": dict(sorted(status_counts.items())),
        "training_overlap_count": len(overlap),
        "training_overlap_transfers": overlap,
        "verified": not failures,
        "worker_interpretation": (
            "This diagnostic replays every top-5 leaf for the all-disjoint misses. "
            "Transfers in shortlist_below_rho_transfers are rule-selection misses; "
            "the remaining misses need features beyond the current top-5 shortlist or "
            "different materialization."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_SECOND_STAGE_DISJOINT_MISS_DIAGNOSTIC_FAILED"
    if as_int(summary.get("training_overlap_count")) > 0:
        return "SELECTED13_SECOND_STAGE_DISJOINT_MISS_DIAGNOSTIC_HAS_OVERLAP"
    if summary.get("diagnostic_source_mode") == "all_source_transfers":
        if as_int(summary.get("shortlist_below_rho_transfer_count")) > 0:
            return "SELECTED13_SECOND_STAGE_DISJOINT_ALL_TOP5_DIAGNOSTIC_BELOW_RHO_COVERAGE"
        return "SELECTED13_SECOND_STAGE_DISJOINT_ALL_TOP5_DIAGNOSTIC_NO_COVERAGE"
    if as_int(summary.get("shortlist_below_rho_transfer_count")) > 0:
        return "SELECTED13_SECOND_STAGE_DISJOINT_MISS_DIAGNOSTIC_FOUND_RULE_SELECTION_MISSES"
    return "SELECTED13_SECOND_STAGE_DISJOINT_MISS_DIAGNOSTIC_NO_TOP5_RESCUES"


def render_c_header(per_transfer: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = []
    for row in per_transfer:
        best_ops_scaled = 0
        best_ops = as_float(row.get("best_alt_ops_over_rho"))
        if best_ops is not None:
            best_ops_scaled = int(round(best_ops * 1_000_000))
        rows.append(
            "  {"
            f"{as_int(row.get('transfer_index'))}ULL, "
            f"{as_int(row.get('frozen_leaf'))}ULL, "
            f"{as_int(row.get('frozen_shortlist_position'))}ULL, "
            f"{1 if row.get('shortlist_has_below_rho') else 0}ULL, "
            f"{as_int(row.get('accepted_below_rho_leaf_count'))}ULL, "
            f"{as_int(row.get('best_alt_leaf'))}ULL, "
            f"{as_int(row.get('best_alt_shortlist_position'))}ULL, "
            f"{as_int(row.get('best_alt_derived_secret'))}ULL, "
            f"{best_ops_scaled}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_SECOND_STAGE_DISJOINT_MISS_DIAGNOSTIC_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_SECOND_STAGE_DISJOINT_MISS_DIAGNOSTIC_PROBE_H

#include <stdint.h>

#define SELECTED13_DISJOINT_MISS_DIAGNOSTIC_TRANSFER_COUNT {len(per_transfer)}
#define SELECTED13_DISJOINT_MISS_DIAGNOSTIC_REPLAY_COUNT {as_int(summary.get("shortlist_replay_count"))}
#define SELECTED13_DISJOINT_MISS_DIAGNOSTIC_TOP5_RESCUE_COUNT {as_int(summary.get("shortlist_below_rho_transfer_count"))}
#define SELECTED13_DISJOINT_MISS_DIAGNOSTIC_TRAINING_OVERLAP_COUNT {as_int(summary.get("training_overlap_count"))}

typedef struct {{
  uint64_t transfer_index;
  uint64_t frozen_leaf;
  uint64_t frozen_shortlist_position;
  uint64_t shortlist_has_below_rho;
  uint64_t accepted_below_rho_leaf_count;
  uint64_t best_alt_leaf;
  uint64_t best_alt_shortlist_position;
  uint64_t best_alt_derived_secret;
  uint64_t best_alt_ops_over_rho_scaled_1e6;
}} selected13_disjoint_miss_diagnostic_transfer_t;

static const selected13_disjoint_miss_diagnostic_transfer_t SELECTED13_DISJOINT_MISS_DIAGNOSTICS[] = {{
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
  uint64_t transfer_count =
      sizeof(SELECTED13_DISJOINT_MISS_DIAGNOSTICS) / sizeof(SELECTED13_DISJOINT_MISS_DIAGNOSTICS[0]);
  uint64_t top5_rescues = 0;

  if (transfer_count != SELECTED13_DISJOINT_MISS_DIAGNOSTIC_TRANSFER_COUNT) failure_count++;
  if (transfer_count == 0ULL) failure_count++;
  if (SELECTED13_DISJOINT_MISS_DIAGNOSTIC_REPLAY_COUNT < transfer_count) failure_count++;
  if (SELECTED13_DISJOINT_MISS_DIAGNOSTIC_TRAINING_OVERLAP_COUNT != 0ULL) failure_count++;

  for (size_t i = 0; i < transfer_count; i++) {{
    const selected13_disjoint_miss_diagnostic_transfer_t *row = &SELECTED13_DISJOINT_MISS_DIAGNOSTICS[i];
    if (row->transfer_index == 0ULL) failure_count++;
    if (row->shortlist_has_below_rho) {{
      top5_rescues++;
      if (row->best_alt_leaf == 0ULL) failure_count++;
      if (row->best_alt_derived_secret == 0ULL) failure_count++;
    }}
  }}

  if (top5_rescues != SELECTED13_DISJOINT_MISS_DIAGNOSTIC_TOP5_RESCUE_COUNT) failure_count++;

  printf("selected13_disjoint_miss_diagnostic_preflight transfers=%llu replays=%llu top5_rescues=%llu overlap=%llu failures=%llu\\n",
         (unsigned long long)transfer_count,
         (unsigned long long)SELECTED13_DISJOINT_MISS_DIAGNOSTIC_REPLAY_COUNT,
         (unsigned long long)top5_rescues,
         (unsigned long long)SELECTED13_DISJOINT_MISS_DIAGNOSTIC_TRAINING_OVERLAP_COUNT,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_disjoint_miss_diagnostic_preflight_") as tmp:
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
    source_payload = load_json(Path(args.source_validation))
    training_contract = load_json(Path(args.training_contract))
    validation_manifest = load_json(Path(args.validation_manifest))
    failures: list[dict[str, Any]] = []
    source_misses = source_miss_transfers(source_payload)
    source_mode = "all_source_transfers" if args.diagnose_all_source_transfers else "source_misses"
    transfers = source_candidate_transfers(source_payload, diagnose_all=args.diagnose_all_source_transfers)
    if not transfers:
        failures.append({"code": "source_validation_has_no_diagnostic_transfers", "source_mode": source_mode})
    train_transfers = set(disjoint_validation.training_transfers(training_contract))
    targets, target_failures = disjoint_validation.select_validation_targets(
        validation_manifest,
        train_transfers,
        max_targets=None,
        requested_transfers=transfers,
    )
    failures.extend(target_failures)
    if len(targets) != len(transfers):
        failures.append({"code": "diagnostic_target_count_mismatch", "expected": len(transfers), "observed": len(targets)})
    replay_records, radius, replay_failures = replay_miss_shortlists(args, targets)
    failures.extend(replay_failures)
    per_transfer = per_transfer_diagnostics(replay_records, source_records_by_transfer(source_payload), transfers)
    summary = summarize(transfers, source_misses, replay_records, per_transfer, failures, train_transfers, source_mode)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, summary),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "config_source": str(Path(args.config_source)),
            "context_top_k": args.context_top_k,
            "diagnose_all_source_transfers": args.diagnose_all_source_transfers,
            "leaf_limit": args.leaf_limit,
            "radius": radius,
            "shortlist_k": args.shortlist_k,
            "shortlist_mode": args.shortlist_mode,
            "source_validation": str(Path(args.source_validation)),
            "target": TARGET,
            "training_contract": str(Path(args.training_contract)),
            "transfer_source": str(Path(args.transfer_source)),
            "validation_manifest": str(Path(args.validation_manifest)),
        },
        "summary": summary,
        "per_transfer_diagnostics": per_transfer,
        "shortlist_leaf_replays": replay_records,
        "failures": failures,
        "honesty_boundary": {
            "diagnostic_only": True,
            "general_ecdlp_algorithm_claimed": False,
            "replays_all_top5_for_misses": True,
            "selection_cost_note": "This probe is over the one-leaf policy cost and is used only to diagnose missed transfers.",
            "training_transfer_overlap_count": summary.get("training_overlap_count"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-validation", type=Path, default=DEFAULT_SOURCE_VALIDATION)
    parser.add_argument("--training-contract", type=Path, default=DEFAULT_TRAINING_CONTRACT)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--shortlist-mode", default=DEFAULT_SHORTLIST_MODE)
    parser.add_argument("--shortlist-k", type=int, default=DEFAULT_SHORTLIST_K)
    parser.add_argument("--diagnose-all-source-transfers", action="store_true")
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
    header_path.write_text(render_c_header(payload["per_transfer_diagnostics"], payload["summary"]))
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
