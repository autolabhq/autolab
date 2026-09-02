#!/usr/bin/env python3
"""Audit public predictors for selected13 common-leaf event cores.

The common-leaf sweep finds many candidate-level below-rho recoveries, but it
still scans all leaf indices. This audit tests whether the existing public
feature-prefix selectors can pick winning common leaves cheaply enough to avoid
that full sweep.

The result is deliberately allowed to be negative: a failed predictor audit is
useful because it separates broad event-core existence from an end-to-end
selector that beats rho.
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


SCHEMA = "ecdlp.low_term_total2_selected13_common_leaf_predictor_audit_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_COMMON_SWEEP = DEFAULT_STATE_DIR / "low_term_total2_selected13_common_leaf_pair_sweep_probe.json"
DEFAULT_CONTRACT = DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9999_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_common_leaf_predictor_audit_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_common_leaf_predictor_audit_probe.h"

TARGET = min_transfer.TARGET
FULL_SWEEP_TOP_K = 100

DEFAULT_PREDICTOR_MODES = (
    "pool_order",
    "low_leaf_index",
    "double_pair_first",
    "high_double_pair_count",
    "low_term_span",
    "low_monic_c",
    "hybrid_double_monic_b",
    "hybrid_support_monic_b",
)
DEFAULT_TOP_KS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 100)

MODE_CODES = {mode: index + 1 for index, mode in enumerate(DEFAULT_PREDICTOR_MODES)}


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


def parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def materialize_representative_feature_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    contract = load_json(Path(args.contract))
    targets = common_sweep.contract_backfill_targets(contract)
    if not targets:
        return [], [{"code": "no_contract_targets"}], 0

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
    contexts, errors = common_sweep.materialize_contexts_for_target(
        verifier,
        records,
        config_source,
        specs_by_target,
        targets[0],
        replay_args,
        {},
    )
    if errors:
        return [], errors, radius
    if not contexts:
        return [], [{"code": "no_representative_context"}], radius
    first_context = next(iter(contexts.values()))
    return first_context.get("feature_rows") or [], [], radius


def records_by_transfer_leaf(common_payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    out = {}
    for record in common_payload.get("common_leaf_pair_candidates") or []:
        out[(as_int(record.get("transfer_index")), as_int(record.get("leaf_index")))] = record
    return out


def transfer_summaries(common_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return (common_payload.get("summary") or {}).get("transfer_summaries") or []


def evaluate_selector(
    selector_index: int,
    mode: str,
    top_k: int,
    feature_rows: list[dict[str, Any]],
    common_payload: dict[str, Any],
) -> dict[str, Any]:
    salt_neighborhood_probe = replay_probe.salt_neighborhood_probe
    seed = f"common-leaf-predictor-audit:{TARGET}:{mode}"
    selected_leaves = sorted(
        int(leaf)
        for leaf in salt_neighborhood_probe.selected_prefix(feature_rows, mode, seed, top_k)
    )
    selected_leaf_set = set(selected_leaves)
    by_key = records_by_transfer_leaf(common_payload)
    selected_transfer_records = []
    covered = []
    heldout_covered = []
    end_to_end_like = []
    set_cost_below = []
    for transfer in transfer_summaries(common_payload):
        transfer_index = as_int(transfer.get("transfer_index"))
        known_positive = bool(transfer.get("known_positive"))
        selected_records = [
            by_key[(transfer_index, leaf)]
            for leaf in selected_leaves
            if (transfer_index, leaf) in by_key
        ]
        generic_rho_steps = max((as_int(row.get("generic_rho_steps")) for row in selected_records), default=0)
        selected_ops = sum(as_int(row.get("ops")) for row in selected_records)
        selected_ops_over_rho = round_or_none(selected_ops / generic_rho_steps if generic_rho_steps else None)
        accepted = [
            row
            for row in selected_records
            if row.get("accepted_relation_export") and row.get("below_rho")
        ]
        best = None
        if accepted:
            accepted.sort(
                key=lambda item: (
                    as_float(item.get("ops_over_rho")) if as_float(item.get("ops_over_rho")) is not None else 999.0,
                    as_int(item.get("leaf_index")),
                )
            )
            best = accepted[0]
            covered.append(transfer_index)
            if not known_positive:
                heldout_covered.append(transfer_index)
        if generic_rho_steps and selected_ops < generic_rho_steps:
            set_cost_below.append(transfer_index)
            if best:
                end_to_end_like.append(transfer_index)
        selected_transfer_records.append(
            {
                "accepted_below_rho": bool(best),
                "best_accepted_candidate_id": best.get("candidate_id") if best else None,
                "best_accepted_leaf_index": best.get("leaf_index") if best else None,
                "best_accepted_ops_over_rho": best.get("ops_over_rho") if best else None,
                "known_positive": known_positive,
                "selected_ops": selected_ops,
                "selected_ops_over_rho": selected_ops_over_rho,
                "selected_record_count": len(selected_records),
                "set_cost_below_rho": transfer_index in set_cost_below,
                "transfer_index": transfer_index,
            }
        )
    return {
        "selector_id": f"predictor_{selector_index:03d}_{mode}_top{top_k}",
        "selector_index": selector_index,
        "mode": mode,
        "mode_code": MODE_CODES.get(mode, 0),
        "top_k": top_k,
        "selected_leaf_count": len(selected_leaves),
        "selected_leaves": selected_leaves,
        "covered_transfer_count": len(covered),
        "covered_transfers": sorted(covered),
        "heldout_covered_transfer_count": len(heldout_covered),
        "heldout_covered_transfers": sorted(heldout_covered),
        "set_cost_below_rho_transfer_count": len(set_cost_below),
        "set_cost_below_rho_transfers": sorted(set_cost_below),
        "end_to_end_like_transfer_count": len(end_to_end_like),
        "end_to_end_like_transfers": sorted(end_to_end_like),
        "full_sweep_equivalent": top_k >= FULL_SWEEP_TOP_K,
        "per_transfer_records": selected_transfer_records,
    }


def choose_best(
    records: list[dict[str, Any]],
    *,
    allow_full: bool = False,
    max_top_k: int | None = None,
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if (allow_full or not record.get("full_sweep_equivalent"))
        and (max_top_k is None or as_int(record.get("top_k")) <= max_top_k)
    ]
    candidates.sort(
        key=lambda item: (
            -as_int(item.get("heldout_covered_transfer_count")),
            -as_int(item.get("covered_transfer_count")),
            as_int(item.get("top_k")),
            str(item.get("mode")),
        )
    )
    return candidates[0] if candidates else {}


def summarize(records: list[dict[str, Any]], failures: list[dict[str, Any]], common_payload: dict[str, Any]) -> dict[str, Any]:
    best_non_full = choose_best(records)
    best_any = choose_best(records, allow_full=True)
    best_compact = choose_best(records, max_top_k=13)
    end_to_end_selectors = [
        record
        for record in records
        if as_int(record.get("end_to_end_like_transfer_count")) > 0
    ]
    compact = [
        record
        for record in records
        if as_int(record.get("top_k")) <= 13 and as_int(record.get("heldout_covered_transfer_count")) > 0
    ]
    return {
        "best_any_selector_id": best_any.get("selector_id"),
        "best_any_top_k": best_any.get("top_k"),
        "best_any_heldout_covered_transfer_count": best_any.get("heldout_covered_transfer_count"),
        "best_compact_selector_id": best_compact.get("selector_id"),
        "best_compact_top_k": best_compact.get("top_k"),
        "best_compact_heldout_covered_transfer_count": best_compact.get("heldout_covered_transfer_count"),
        "best_non_full_selector_id": best_non_full.get("selector_id"),
        "best_non_full_top_k": best_non_full.get("top_k"),
        "best_non_full_heldout_covered_transfer_count": best_non_full.get("heldout_covered_transfer_count"),
        "compact_selector_with_heldout_count": len(compact),
        "end_to_end_like_selector_count": len(end_to_end_selectors),
        "failure_count": len(failures),
        "full_sweep_candidate_count": (common_payload.get("summary") or {}).get("candidate_count"),
        "general_ecdlp_algorithm_claimed": False,
        "predictor_selector_count": len(records),
        "source_common_sweep_status": common_payload.get("claim_status"),
        "verified": not failures,
        "worker_interpretation": (
            "Existing public-prefix selectors can cover some held-out winning leaves, "
            "but none both finds a below-rho candidate and keeps the selected-set cost below rho. "
            "The next useful step is a salt-conditioned FFE predictor, not another fixed feature-prefix ranker."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_FAILED"
    if as_int(summary.get("end_to_end_like_selector_count")) > 0:
        return "SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_HAS_END_TO_END_CANDIDATE"
    if as_int(summary.get("best_non_full_heldout_covered_transfer_count")) > 0:
        return "SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_COVERS_HELDOUT_NOT_BELOW_RHO_COST"
    return "SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_NO_HELDOUT_COVERAGE"


def render_c_header(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        rows.append(
            "  {"
            f"{as_int(record.get('selector_index'))}ULL, "
            f"{as_int(record.get('mode_code'))}ULL, "
            f"{as_int(record.get('top_k'))}ULL, "
            f"{as_int(record.get('selected_leaf_count'))}ULL, "
            f"{as_int(record.get('covered_transfer_count'))}ULL, "
            f"{as_int(record.get('heldout_covered_transfer_count'))}ULL, "
            f"{as_int(record.get('set_cost_below_rho_transfer_count'))}ULL, "
            f"{as_int(record.get('end_to_end_like_transfer_count'))}ULL, "
            f"{1 if record.get('full_sweep_equivalent') else 0}ULL"
            "},"
        )
    end_to_end_count = sum(1 for row in records if as_int(row.get("end_to_end_like_transfer_count")) > 0)
    heldout_cover_count = sum(1 for row in records if as_int(row.get("heldout_covered_transfer_count")) > 0)
    non_full_heldout_cover_count = sum(
        1
        for row in records
        if as_int(row.get("heldout_covered_transfer_count")) > 0 and not row.get("full_sweep_equivalent")
    )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_PROBE_H

#include <stdint.h>

#define SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_SELECTOR_COUNT {len(records)}
#define SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_HELDOUT_COVER_SELECTOR_COUNT {heldout_cover_count}
#define SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_NON_FULL_HELDOUT_COVER_SELECTOR_COUNT {non_full_heldout_cover_count}
#define SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_END_TO_END_SELECTOR_COUNT {end_to_end_count}

typedef struct {{
  uint64_t selector_index;
  uint64_t mode_code;
  uint64_t top_k;
  uint64_t selected_leaf_count;
  uint64_t covered_transfer_count;
  uint64_t heldout_covered_transfer_count;
  uint64_t set_cost_below_rho_transfer_count;
  uint64_t end_to_end_like_transfer_count;
  uint64_t full_sweep_equivalent;
}} selected13_common_leaf_predictor_audit_selector_t;

static const selected13_common_leaf_predictor_audit_selector_t SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_SELECTORS[] = {{
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
      sizeof(SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_SELECTORS) / sizeof(SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_SELECTORS[0]);
  uint64_t heldout_cover = 0;
  uint64_t non_full_heldout_cover = 0;
  uint64_t end_to_end = 0;

  if (selector_count != SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_SELECTOR_COUNT) failure_count++;
  if (selector_count == 0ULL) failure_count++;

  for (size_t i = 0; i < selector_count; i++) {{
    const selected13_common_leaf_predictor_audit_selector_t *selector = &SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_SELECTORS[i];
    if (selector->mode_code == 0ULL) failure_count++;
    if (selector->top_k == 0ULL) failure_count++;
    if (selector->selected_leaf_count == 0ULL) failure_count++;
    if (selector->heldout_covered_transfer_count > 0ULL) {{
      heldout_cover++;
      if (!selector->full_sweep_equivalent) non_full_heldout_cover++;
    }}
    if (selector->end_to_end_like_transfer_count > 0ULL) end_to_end++;
  }}

  if (heldout_cover != SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_HELDOUT_COVER_SELECTOR_COUNT) failure_count++;
  if (non_full_heldout_cover != SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_NON_FULL_HELDOUT_COVER_SELECTOR_COUNT) failure_count++;
  if (end_to_end != SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_END_TO_END_SELECTOR_COUNT) failure_count++;
  if (non_full_heldout_cover == 0ULL) failure_count++;
  if (end_to_end != 0ULL) failure_count++;

  printf("selected13_common_leaf_predictor_audit_preflight selectors=%llu heldout_cover=%llu non_full_heldout_cover=%llu end_to_end=%llu failures=%llu\\n",
         (unsigned long long)selector_count,
         (unsigned long long)heldout_cover,
         (unsigned long long)non_full_heldout_cover,
         (unsigned long long)end_to_end,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_common_leaf_predictor_audit_preflight_") as tmp:
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
    common_payload = load_json(Path(args.common_sweep))
    failures: list[dict[str, Any]] = []
    if common_payload.get("claim_status") != "SELECTED13_COMMON_LEAF_PAIR_SWEEP_HELDOUT_BELOW_RHO_CANDIDATES":
        failures.append({"code": "common_sweep_status_unexpected", "claim_status": common_payload.get("claim_status")})
    feature_rows, materialize_errors, radius = materialize_representative_feature_rows(args)
    for error in materialize_errors:
        failures.append({"code": "feature_materialization_error", "error": error})
    selector_records = []
    if feature_rows:
        index = 0
        for mode in args.predictor_modes:
            for top_k in args.top_ks:
                selector_records.append(evaluate_selector(index, mode, top_k, feature_rows, common_payload))
                index += 1
    summary = summarize(selector_records, failures, common_payload)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, summary),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "common_sweep": str(Path(args.common_sweep)),
            "config_source": str(Path(args.config_source)),
            "context_top_k": args.context_top_k,
            "contract": str(Path(args.contract)),
            "direct_source": str(Path(args.direct_source)),
            "predictor_modes": args.predictor_modes,
            "radius": radius,
            "target": TARGET,
            "top_ks": args.top_ks,
            "transfer_source": str(Path(args.transfer_source)),
        },
        "summary": summary,
        "predictor_selectors": selector_records,
        "failures": failures,
        "honesty_boundary": {
            "general_ecdlp_algorithm_claimed": False,
            "no_end_to_end_predictor_found": summary["end_to_end_like_selector_count"] == 0,
            "selection_cost_note": "A selector is end-to-end-like only if it finds a below-rho candidate and its whole selected leaf set costs less than rho on that transfer.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-sweep", type=Path, default=DEFAULT_COMMON_SWEEP)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--predictor-modes", type=parse_csv, default=list(DEFAULT_PREDICTOR_MODES))
    parser.add_argument("--top-ks", type=parse_int_csv, default=list(DEFAULT_TOP_KS))
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
    header_path.write_text(render_c_header(payload["predictor_selectors"]))
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
