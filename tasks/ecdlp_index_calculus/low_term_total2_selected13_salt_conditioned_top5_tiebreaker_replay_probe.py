#!/usr/bin/env python3
"""Replay and mine public tie-breakers for the active-scout top-5 shortlist.

The direct top-1 salt-conditioned association policy gives five held-out
below-rho exports. The active-scout top-5 shortlist covers more winners in the
label audit, but scanning all five leaves is over rho. This probe direct-replays
each top-5 shortlisted leaf, then tests public tie-breakers that choose one leaf
from that shortlist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import ffe_single_hit_root_relation_replay_probe as replay_probe
import low_term_total2_selected13_common_leaf_pair_sweep_probe as common_sweep
import low_term_total2_selected13_public_prefix_min_transfer_probe as min_transfer
import low_term_total2_selected13_salt_conditioned_association_predictor_probe as assoc_predictor
import low_term_total2_selected13_salt_conditioned_direct_replay_probe as direct_replay


SCHEMA = "ecdlp.low_term_total2_selected13_salt_conditioned_top5_tiebreaker_replay_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_CONTRACT = DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9999_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_salt_conditioned_top5_tiebreaker_replay_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_salt_conditioned_top5_tiebreaker_replay_probe.h"

TARGET = min_transfer.TARGET
DEFAULT_SHORTLIST_MODE = "active_scout_sum_desc"
DEFAULT_SHORTLIST_K = 5

DEFAULT_TIEBREAKER_MODES = (
    "shortlist_pos0",
    "shortlist_pos1",
    "shortlist_pos2",
    "shortlist_pos3",
    "shortlist_pos4",
    "low_leaf_index",
    "high_leaf_index",
    "low_term_span",
    "low_support",
    "double_pair_first",
    "hit_root_sum_desc",
    "hit_root_min_desc",
    "hit_root_product_desc",
    "row_hit_sum_desc",
    "row_hit_min_desc",
    "scout_hit_sum_desc",
    "active_scout_sum_desc",
    "nonzero_row_count_desc",
    "asym_hit_root_desc",
    "asym_row_hit_desc",
    "hit_root_sum_low_span",
    "hit_root_min_low_span",
    "low_span_hit_root_sum",
    "double_pair_hit_root_sum",
    "double_pair_low_span_hit_root",
    "support_compact_hit_root_sum",
    "repeated_terms_hit_root_sum",
)
MODE_CODES = {mode: index + 1 for index, mode in enumerate(DEFAULT_TIEBREAKER_MODES)}


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


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def tie_key(association_record: dict[str, Any], mode: str, position: int) -> tuple[Any, ...]:
    leaf = as_int(association_record.get("leaf_index"))
    term_span = as_int(association_record.get("min_term_span"), 10**9)
    support = as_int(association_record.get("min_term_support_size"), 10**9)
    double_pair = 1 if association_record.get("has_double_pair") else 0
    if mode.startswith("shortlist_pos"):
        wanted = as_int(mode.removeprefix("shortlist_pos"), -1)
        return (0 if position == wanted else 1, position, leaf)
    if mode == "low_leaf_index":
        return (leaf, position)
    if mode == "high_leaf_index":
        return (-leaf, position)
    if mode == "low_term_span":
        return (term_span, support, leaf)
    if mode == "low_support":
        return (support, term_span, leaf)
    if mode == "double_pair_first":
        return (-double_pair, term_span, leaf)
    return assoc_predictor.score_key(association_record, mode)


def build_replay_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
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


def materialize_shortlist_replays(
    args: argparse.Namespace,
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]]]:
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
    replay_args = build_replay_args(args)
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    replay_records: list[dict[str, Any]] = []
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
            replay_record["shortlist_position"] = position
            replay_record["shortlist_mode"] = args.shortlist_mode
            entries.append(
                {
                    "association_record": association_record,
                    "position": position,
                    "replay_record": replay_record,
                    "transfer_index": as_int(target.get("transfer_index")),
                }
            )
            replay_records.append(replay_record)
            replay_index += 1
    return entries, replay_records, radius, failures


def accepted_below(record: dict[str, Any]) -> bool:
    return bool(record.get("accepted_relation_export")) and bool(record.get("below_rho"))


def evaluate_tiebreaker(selector_index: int, mode: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_transfer: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        by_transfer.setdefault(as_int(entry.get("transfer_index")), []).append(entry)
    per_transfer_records = []
    accepted = []
    heldout_accepted = []
    known_accepted = []
    for transfer_index in sorted(by_transfer):
        chosen = sorted(
            by_transfer[transfer_index],
            key=lambda entry: tie_key(entry["association_record"], mode, as_int(entry.get("position"))),
        )[0]
        replay_record = chosen["replay_record"]
        if accepted_below(replay_record):
            accepted.append(transfer_index)
            if replay_record.get("known_positive_transfer"):
                known_accepted.append(transfer_index)
            else:
                heldout_accepted.append(transfer_index)
        per_transfer_records.append(
            {
                "accepted_below_rho": accepted_below(replay_record),
                "derived_secret": replay_record.get("derived_secret"),
                "known_positive": bool(replay_record.get("known_positive_transfer")),
                "ops_over_rho": replay_record.get("ops_over_rho"),
                "selected_leaf_indices": replay_record.get("selected_leaf_indices") or [],
                "shortlist_position": as_int(chosen.get("position")),
                "status": replay_record.get("status"),
                "transfer_index": transfer_index,
            }
        )
    return {
        "selector_id": f"salt_assoc_top5_tiebreaker_{selector_index:03d}_{mode}",
        "selector_id_u64": min_transfer.digest_u64({"selector_index": selector_index, "mode": mode}),
        "selector_index": selector_index,
        "mode": mode,
        "mode_code": MODE_CODES.get(mode, 0),
        "accepted_below_rho_transfer_count": len(accepted),
        "accepted_below_rho_transfers": sorted(accepted),
        "heldout_accepted_below_rho_transfer_count": len(heldout_accepted),
        "heldout_accepted_below_rho_transfers": sorted(heldout_accepted),
        "known_positive_accepted_below_rho_transfers": sorted(known_accepted),
        "per_transfer_records": per_transfer_records,
    }


def choose_best(records: list[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    records.sort(
        key=lambda record: (
            -as_int(record.get("heldout_accepted_below_rho_transfer_count")),
            -as_int(record.get("accepted_below_rho_transfer_count")),
            as_int(record.get("selector_index")),
        )
    )
    return records[0] if records else {}


def summarize(
    selector_records: list[dict[str, Any]],
    replay_records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    best = choose_best(selector_records)
    accepted_replays = [record for record in replay_records if accepted_below(record)]
    heldout_replays = [record for record in accepted_replays if not record.get("known_positive_transfer")]
    improved = [
        record
        for record in selector_records
        if as_int(record.get("heldout_accepted_below_rho_transfer_count")) > 5
    ]
    return {
        "best_selector_id": best.get("selector_id"),
        "best_selector_mode": best.get("mode"),
        "best_accepted_below_rho_transfer_count": best.get("accepted_below_rho_transfer_count"),
        "best_accepted_below_rho_transfers": best.get("accepted_below_rho_transfers"),
        "best_heldout_accepted_below_rho_transfer_count": best.get("heldout_accepted_below_rho_transfer_count"),
        "best_heldout_accepted_below_rho_transfers": best.get("heldout_accepted_below_rho_transfers"),
        "direct_shortlist_replay_count": len(replay_records),
        "direct_shortlist_replay_accepted_below_rho_count": len(accepted_replays),
        "direct_shortlist_replay_heldout_accepted_below_rho_count": len(heldout_replays),
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "improving_selector_count": len(improved),
        "selector_count": len(selector_records),
        "verified": not failures,
        "worker_interpretation": (
            "Every active-scout top-5 leaf is direct-replayed, then public tie-breakers "
            "choose one leaf per transfer. A selector improves the top-1 frontier only "
            "if its held-out below-rho direct exports exceed five."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_SALT_CONDITIONED_TOP5_TIEBREAKER_REPLAY_FAILED"
    if as_int(summary.get("improving_selector_count")) > 0:
        return "SELECTED13_SALT_CONDITIONED_TOP5_TIEBREAKER_REPLAY_IMPROVES_HELDOUT"
    if as_int(summary.get("best_heldout_accepted_below_rho_transfer_count")) > 0:
        return "SELECTED13_SALT_CONDITIONED_TOP5_TIEBREAKER_REPLAY_NO_IMPROVEMENT"
    return "SELECTED13_SALT_CONDITIONED_TOP5_TIEBREAKER_REPLAY_NO_EXPORT"


def render_c_header(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = []
    for record in records:
        rows.append(
            "  {"
            f"{as_int(record.get('selector_index'))}ULL, "
            f"{as_int(record.get('selector_id_u64'))}ULL, "
            f"{as_int(record.get('mode_code'))}ULL, "
            f"{as_int(record.get('accepted_below_rho_transfer_count'))}ULL, "
            f"{as_int(record.get('heldout_accepted_below_rho_transfer_count'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_TOP5_TIEBREAKER_REPLAY_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_TOP5_TIEBREAKER_REPLAY_PROBE_H

#include <stdint.h>

#define SELECTED13_SALT_TOP5_TIEBREAKER_SELECTOR_COUNT {len(records)}
#define SELECTED13_SALT_TOP5_TIEBREAKER_DIRECT_REPLAY_COUNT {as_int(summary.get("direct_shortlist_replay_count"))}
#define SELECTED13_SALT_TOP5_TIEBREAKER_DIRECT_ACCEPTED_BELOW_COUNT {as_int(summary.get("direct_shortlist_replay_accepted_below_rho_count"))}
#define SELECTED13_SALT_TOP5_TIEBREAKER_BEST_HELDOUT_ACCEPTED_BELOW_COUNT {as_int(summary.get("best_heldout_accepted_below_rho_transfer_count"))}
#define SELECTED13_SALT_TOP5_TIEBREAKER_IMPROVING_SELECTOR_COUNT {as_int(summary.get("improving_selector_count"))}

typedef struct {{
  uint64_t selector_index;
  uint64_t selector_id_u64;
  uint64_t mode_code;
  uint64_t accepted_below_rho_transfer_count;
  uint64_t heldout_accepted_below_rho_transfer_count;
}} selected13_salt_top5_tiebreaker_selector_t;

static const selected13_salt_top5_tiebreaker_selector_t SELECTED13_SALT_TOP5_TIEBREAKER_SELECTORS[] = {{
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
  uint64_t selector_count =
      sizeof(SELECTED13_SALT_TOP5_TIEBREAKER_SELECTORS) / sizeof(SELECTED13_SALT_TOP5_TIEBREAKER_SELECTORS[0]);
  uint64_t best_heldout = 0;
  uint64_t improving = 0;

  if (selector_count != SELECTED13_SALT_TOP5_TIEBREAKER_SELECTOR_COUNT) failure_count++;
  if (selector_count == 0ULL) failure_count++;
  if (SELECTED13_SALT_TOP5_TIEBREAKER_DIRECT_REPLAY_COUNT == 0ULL) failure_count++;
  if (SELECTED13_SALT_TOP5_TIEBREAKER_DIRECT_ACCEPTED_BELOW_COUNT == 0ULL) failure_count++;

  for (size_t i = 0; i < selector_count; i++) {{
    const selected13_salt_top5_tiebreaker_selector_t *selector = &SELECTED13_SALT_TOP5_TIEBREAKER_SELECTORS[i];
    if (selector->selector_id_u64 == 0ULL) failure_count++;
    if (selector->mode_code == 0ULL) failure_count++;
    if (selector->heldout_accepted_below_rho_transfer_count > best_heldout) {{
      best_heldout = selector->heldout_accepted_below_rho_transfer_count;
    }}
    if (selector->heldout_accepted_below_rho_transfer_count > 5ULL) improving++;
  }}

  if (best_heldout != SELECTED13_SALT_TOP5_TIEBREAKER_BEST_HELDOUT_ACCEPTED_BELOW_COUNT) failure_count++;
  if (improving != SELECTED13_SALT_TOP5_TIEBREAKER_IMPROVING_SELECTOR_COUNT) failure_count++;
  if (best_heldout < 5ULL) failure_count++;

  printf("selected13_salt_top5_tiebreaker_preflight selectors=%llu shortlist_replays=%llu accepted_below=%llu best_heldout=%llu improving=%llu failures=%llu\\n",
         (unsigned long long)selector_count,
         (unsigned long long)SELECTED13_SALT_TOP5_TIEBREAKER_DIRECT_REPLAY_COUNT,
         (unsigned long long)SELECTED13_SALT_TOP5_TIEBREAKER_DIRECT_ACCEPTED_BELOW_COUNT,
         (unsigned long long)best_heldout,
         (unsigned long long)improving,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_salt_top5_tiebreaker_preflight_") as tmp:
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
    failures: list[dict[str, Any]] = []
    if contract.get("claim_status") != "FFE_SHARP_LANE_KERNEL_CONTRACT_READY":
        failures.append({"code": "contract_status_unexpected", "claim_status": contract.get("claim_status")})
    targets = common_sweep.contract_backfill_targets(contract)
    if not targets:
        failures.append({"code": "no_backfill_targets"})
    entries, replay_records, radius, replay_failures = materialize_shortlist_replays(args, targets)
    failures.extend(replay_failures)
    selector_records = [
        evaluate_tiebreaker(index, mode, entries)
        for index, mode in enumerate(args.tiebreaker_modes)
    ]
    summary = summarize(selector_records, replay_records, failures)
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
            "leaf_limit": args.leaf_limit,
            "radius": radius,
            "shortlist_k": args.shortlist_k,
            "shortlist_mode": args.shortlist_mode,
            "target": TARGET,
            "tiebreaker_modes": args.tiebreaker_modes,
            "transfer_source": str(Path(args.transfer_source)),
        },
        "summary": summary,
        "selector_records": selector_records,
        "shortlist_leaf_replays": replay_records,
        "failures": failures,
        "honesty_boundary": {
            "common_sweep_labels_used": False,
            "general_ecdlp_algorithm_claimed": False,
            "shortlist_leaf_replays_direct": True,
            "selection_cost_note": "Tie-breakers choose one leaf from a direct-replayed top-5 shortlist; a rule improves only if that one-leaf choice beats the top-1 held-out count.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--shortlist-mode", default=DEFAULT_SHORTLIST_MODE)
    parser.add_argument("--shortlist-k", type=int, default=DEFAULT_SHORTLIST_K)
    parser.add_argument("--tiebreaker-modes", type=parse_csv, default=list(DEFAULT_TIEBREAKER_MODES))
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
    header_path.write_text(render_c_header(payload["selector_records"], payload["summary"]))
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
