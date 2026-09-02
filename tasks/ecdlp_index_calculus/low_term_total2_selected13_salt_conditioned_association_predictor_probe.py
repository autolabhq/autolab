#!/usr/bin/env python3
"""Audit salt-conditioned association predictors for selected13 common leaves.

The common-leaf pair sweep proved that many selected13 transfers have a
relation-derived below-rho event core, but only after scanning every common leaf
index. The fixed public-prefix audit then showed that global leaf rankers do
not cheaply choose those winners.

This probe moves one step closer to an end-to-end selector: for each transfer
row pair, it ranks leaves using only pre-relation FFE association signals from
the two public salts, then charges the whole selected leaf set against rho using
the already verified common-leaf sweep records.
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
import low_term_total2_selected13_common_leaf_predictor_audit_probe as predictor_audit
import low_term_total2_selected13_public_prefix_min_transfer_probe as min_transfer


SCHEMA = "ecdlp.low_term_total2_selected13_salt_conditioned_association_predictor_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_COMMON_SWEEP = DEFAULT_STATE_DIR / "low_term_total2_selected13_common_leaf_pair_sweep_probe.json"
DEFAULT_CONTRACT = DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9999_probe.json"
DEFAULT_FIXED_PREDICTOR_AUDIT = DEFAULT_STATE_DIR / "low_term_total2_selected13_common_leaf_predictor_audit_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_salt_conditioned_association_predictor_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_salt_conditioned_association_predictor_probe.h"

TARGET = min_transfer.TARGET
FULL_SWEEP_TOP_K = 100

DEFAULT_SCORER_MODES = (
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
DEFAULT_TOP_KS = (1, 2, 3, 5, 8, 13, 21, 34, 55)

MODE_CODES = {mode: index + 1 for index, mode in enumerate(DEFAULT_SCORER_MODES)}


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
    return min_transfer.parse_csv(raw)


def parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def truthy_int(value: Any) -> int:
    return 1 if bool(value) else 0


def feature_by_leaf(feature_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {as_int(row.get("leaf_index")): row for row in feature_rows or []}


def compact_feature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "double_pair_scout_count": as_int(row.get("double_pair_scout_count")),
        "has_double_pair": bool(row.get("has_double_pair")),
        "has_repeated_terms": bool(row.get("has_repeated_terms")),
        "leaf_index": as_int(row.get("leaf_index")),
        "max_shape_count": as_int(row.get("max_shape_count")),
        "min_scout_pos": as_int(row.get("min_scout_pos")),
        "min_term_span": as_int(row.get("min_term_span"), 10**9),
        "min_term_support_size": as_int(row.get("min_term_support_size"), 10**9),
        "monic_b": as_int(row.get("monic_b")),
        "monic_c": as_int(row.get("monic_c")),
        "repeated_scout_count": as_int(row.get("repeated_scout_count")),
        "scout_position_count": as_int(row.get("scout_position_count")),
        "sum_repeat_excess": as_int(row.get("sum_repeat_excess")),
        "term_support_size": as_int(row.get("term_support_size"), 10**9),
    }


def leaf_association_summary(context: dict[str, Any], leaf_index: int) -> dict[str, Any]:
    association_probe = replay_probe.direct_witness_probe.association_probe
    built = context["built"]
    components = context["components"]
    p = int(built["p"])
    leaves = components["leaves"]
    hits_by_scout = association_probe.empty_hits(built["scouts"])
    selected_hit_roots: set[int] = set()
    if 0 <= leaf_index < len(leaves):
        leaf = leaves[leaf_index]
        gcd_poly = association_probe.polyops.poly_gcd(components["hit_poly"], leaf["poly"], p)
        if max(0, len(gcd_poly) - 1) > 0:
            roots = association_probe.roots_in_hit_set(gcd_poly, components["hit_roots"], p)
            selected_hit_roots.update(int(root) for root in roots)
            association_probe.add_leaf_hits(hits_by_scout, leaf, roots, components["rows_by_x"])
    finalized_hits = association_probe.finalize_hits(hits_by_scout, components["row_order"])
    row_hit_count = association_probe.row_hit_counts(finalized_hits)
    scout_hit_lengths = [len(values) for values in finalized_hits.values()]
    return {
        "active_scout_count": sum(1 for value in scout_hit_lengths if value),
        "max_scout_hit_count": max(scout_hit_lengths) if scout_hit_lengths else 0,
        "row_hit_total": sum(int(value) for value in row_hit_count.values()),
        "row_hit_trial_count": len(row_hit_count),
        "scout_hit_total": sum(scout_hit_lengths),
        "selected_hit_root_count": len(selected_hit_roots),
        "selected_hit_root_values": sorted(selected_hit_roots),
    }


def combine_leaf_record(
    target: dict[str, Any],
    leaf_index: int,
    contexts: dict[str, dict[str, Any]],
    common_record: dict[str, Any] | None,
) -> dict[str, Any]:
    row_records = []
    for row_key in [str(row_key) for row_key in target.get("row_keys") or []]:
        context = contexts[row_key]
        feature = compact_feature(feature_by_leaf(context.get("feature_rows") or []).get(leaf_index, {}))
        association = leaf_association_summary(context, leaf_index)
        row_records.append(
            {
                "association": association,
                "feature": feature,
                "generic_rho_steps": as_int((context.get("built") or {}).get("generic_rho_steps")),
                "row_key": row_key,
            }
        )
    hit_root_counts = [as_int((row.get("association") or {}).get("selected_hit_root_count")) for row in row_records]
    row_hit_totals = [as_int((row.get("association") or {}).get("row_hit_total")) for row in row_records]
    scout_hit_totals = [as_int((row.get("association") or {}).get("scout_hit_total")) for row in row_records]
    active_scout_counts = [as_int((row.get("association") or {}).get("active_scout_count")) for row in row_records]
    first_feature = (row_records[0].get("feature") if row_records else {}) or {}
    common_record = common_record or {}
    return {
        "accepted_relation_export": bool(common_record.get("accepted_relation_export")),
        "active_scout_count_max": max(active_scout_counts) if active_scout_counts else 0,
        "active_scout_count_min": min(active_scout_counts) if active_scout_counts else 0,
        "active_scout_count_sum": sum(active_scout_counts),
        "below_rho": bool(common_record.get("below_rho")),
        "candidate_id": common_record.get("candidate_id"),
        "common_leaf_ops": as_int(common_record.get("ops")),
        "common_leaf_ops_over_rho": common_record.get("ops_over_rho"),
        "double_pair_scout_count": as_int(first_feature.get("double_pair_scout_count")),
        "feature": first_feature,
        "generic_rho_steps": as_int(common_record.get("generic_rho_steps")),
        "has_double_pair": bool(first_feature.get("has_double_pair")),
        "has_repeated_terms": bool(first_feature.get("has_repeated_terms")),
        "hit_root_count_max": max(hit_root_counts) if hit_root_counts else 0,
        "hit_root_count_min": min(hit_root_counts) if hit_root_counts else 0,
        "hit_root_count_product_plus_one": (
            (hit_root_counts[0] + 1) * (hit_root_counts[1] + 1) if len(hit_root_counts) >= 2 else 0
        ),
        "hit_root_count_sum": sum(hit_root_counts),
        "hit_root_nonzero_row_count": sum(1 for value in hit_root_counts if value),
        "known_positive_transfer": bool(target.get("known_positive")),
        "leaf_index": leaf_index,
        "min_term_span": as_int(first_feature.get("min_term_span"), 10**9),
        "min_term_support_size": as_int(first_feature.get("min_term_support_size"), 10**9),
        "monic_b": as_int(first_feature.get("monic_b")),
        "monic_c": as_int(first_feature.get("monic_c")),
        "relation_count": as_int(common_record.get("relation_count")),
        "relation_derived_ecdlp": bool(common_record.get("relation_derived_ecdlp")),
        "row_count": len(row_records),
        "row_hit_total_max": max(row_hit_totals) if row_hit_totals else 0,
        "row_hit_total_min": min(row_hit_totals) if row_hit_totals else 0,
        "row_hit_total_sum": sum(row_hit_totals),
        "row_records": row_records,
        "scout_hit_total_max": max(scout_hit_totals) if scout_hit_totals else 0,
        "scout_hit_total_min": min(scout_hit_totals) if scout_hit_totals else 0,
        "scout_hit_total_sum": sum(scout_hit_totals),
        "term_support_size": as_int(first_feature.get("term_support_size"), 10**9),
        "transfer_index": as_int(target.get("transfer_index")),
    }


def materialize_association_records(
    args: argparse.Namespace,
    targets: list[dict[str, Any]],
    common_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], list[dict[str, Any]]]:
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
    by_transfer_leaf = predictor_audit.records_by_transfer_leaf(common_payload)
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    materialized_targets = []
    records_out = []
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
        leaf_count = min(len(context["components"]["leaves"]) for context in contexts.values())
        if args.leaf_limit is not None:
            leaf_count = min(leaf_count, args.leaf_limit)
        materialized_targets.append(
            {
                "known_positive": bool(target.get("known_positive")),
                "leaf_count": leaf_count,
                "row_keys": target.get("row_keys") or [],
                "salts": target.get("salts") or [],
                "transfer_index": as_int(target.get("transfer_index")),
            }
        )
        transfer_index = as_int(target.get("transfer_index"))
        for leaf_index in range(leaf_count):
            records_out.append(
                combine_leaf_record(
                    target,
                    leaf_index,
                    contexts,
                    by_transfer_leaf.get((transfer_index, leaf_index)),
                )
            )
    return records_out, radius, failures, materialized_targets


def records_by_transfer_leaf(records: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (as_int(record.get("transfer_index")), as_int(record.get("leaf_index"))): record
        for record in records
    }


def transfer_indices(records: list[dict[str, Any]]) -> list[int]:
    return sorted({as_int(record.get("transfer_index")) for record in records})


def score_key(record: dict[str, Any], mode: str) -> tuple[Any, ...]:
    leaf = as_int(record.get("leaf_index"))
    root_sum = as_int(record.get("hit_root_count_sum"))
    root_min = as_int(record.get("hit_root_count_min"))
    root_max = as_int(record.get("hit_root_count_max"))
    root_product = as_int(record.get("hit_root_count_product_plus_one"))
    row_hit_sum = as_int(record.get("row_hit_total_sum"))
    row_hit_min = as_int(record.get("row_hit_total_min"))
    row_hit_max = as_int(record.get("row_hit_total_max"))
    scout_hit_sum = as_int(record.get("scout_hit_total_sum"))
    active_sum = as_int(record.get("active_scout_count_sum"))
    nonzero_rows = as_int(record.get("hit_root_nonzero_row_count"))
    term_span = as_int(record.get("min_term_span"), 10**9)
    support = as_int(record.get("min_term_support_size"), 10**9)
    double_pair = truthy_int(record.get("has_double_pair"))
    double_pair_count = as_int(record.get("double_pair_scout_count"))
    repeated_terms = truthy_int(record.get("has_repeated_terms"))
    monic_b = as_int(record.get("monic_b"))
    monic_c = as_int(record.get("monic_c"))
    root_asym = abs(root_max - root_min)
    row_asym = abs(row_hit_max - row_hit_min)
    base_tail = (term_span, support, monic_b, monic_c, leaf)
    if mode == "hit_root_sum_desc":
        return (-root_sum, -row_hit_sum, *base_tail)
    if mode == "hit_root_min_desc":
        return (-root_min, -root_sum, -row_hit_sum, *base_tail)
    if mode == "hit_root_product_desc":
        return (-root_product, -root_sum, -row_hit_sum, *base_tail)
    if mode == "row_hit_sum_desc":
        return (-row_hit_sum, -root_sum, *base_tail)
    if mode == "row_hit_min_desc":
        return (-row_hit_min, -row_hit_sum, -root_sum, *base_tail)
    if mode == "scout_hit_sum_desc":
        return (-scout_hit_sum, -row_hit_sum, -root_sum, *base_tail)
    if mode == "active_scout_sum_desc":
        return (-active_sum, -scout_hit_sum, -root_sum, *base_tail)
    if mode == "nonzero_row_count_desc":
        return (-nonzero_rows, -root_sum, -row_hit_sum, *base_tail)
    if mode == "asym_hit_root_desc":
        return (-root_max, root_min, -root_asym, -row_hit_sum, *base_tail)
    if mode == "asym_row_hit_desc":
        return (-row_hit_max, row_hit_min, -row_asym, -root_sum, *base_tail)
    if mode == "hit_root_sum_low_span":
        return (-root_sum, term_span, -row_hit_sum, support, leaf)
    if mode == "hit_root_min_low_span":
        return (-root_min, -root_sum, term_span, support, leaf)
    if mode == "low_span_hit_root_sum":
        return (term_span, -root_sum, -row_hit_sum, support, leaf)
    if mode == "double_pair_hit_root_sum":
        return (-double_pair, -root_sum, -double_pair_count, term_span, leaf)
    if mode == "double_pair_low_span_hit_root":
        return (-double_pair, term_span, -root_sum, -row_hit_sum, leaf)
    if mode == "support_compact_hit_root_sum":
        return (support, term_span, -root_sum, -row_hit_sum, leaf)
    if mode == "repeated_terms_hit_root_sum":
        return (-repeated_terms, -root_sum, -row_hit_sum, term_span, leaf)
    return (leaf,)


def best_accepted(selected_records: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        row
        for row in selected_records
        if row.get("accepted_relation_export") and row.get("below_rho")
    ]
    accepted.sort(
        key=lambda item: (
            as_float(item.get("common_leaf_ops_over_rho")) if as_float(item.get("common_leaf_ops_over_rho")) is not None else 999.0,
            as_int(item.get("leaf_index")),
        )
    )
    return accepted[0] if accepted else {}


def evaluate_selector(
    selector_index: int,
    mode: str,
    top_k: int,
    association_records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_transfer: dict[int, list[dict[str, Any]]] = {}
    for record in association_records:
        by_transfer.setdefault(as_int(record.get("transfer_index")), []).append(record)
    covered = []
    heldout_covered = []
    set_cost_below = []
    end_to_end_like = []
    heldout_end_to_end_like = []
    top1_end_to_end_like = []
    per_transfer_records = []
    for transfer_index in sorted(by_transfer):
        ranked = sorted(by_transfer[transfer_index], key=lambda item: score_key(item, mode))
        selected = ranked[:top_k]
        selected_leaves = [as_int(row.get("leaf_index")) for row in selected]
        selected_ops = sum(as_int(row.get("common_leaf_ops")) for row in selected)
        generic_rho_steps = max((as_int(row.get("generic_rho_steps")) for row in selected), default=0)
        selected_ops_over_rho = round_or_none(selected_ops / generic_rho_steps if generic_rho_steps else None)
        best = best_accepted(selected)
        known_positive = bool(selected[0].get("known_positive_transfer")) if selected else False
        if best:
            covered.append(transfer_index)
            if not known_positive:
                heldout_covered.append(transfer_index)
        below_cost = bool(generic_rho_steps and selected_ops < generic_rho_steps)
        if below_cost:
            set_cost_below.append(transfer_index)
            if best:
                end_to_end_like.append(transfer_index)
                if not known_positive:
                    heldout_end_to_end_like.append(transfer_index)
                if top_k == 1:
                    top1_end_to_end_like.append(transfer_index)
        per_transfer_records.append(
            {
                "accepted_below_rho": bool(best),
                "best_accepted_candidate_id": best.get("candidate_id") if best else None,
                "best_accepted_leaf_index": best.get("leaf_index") if best else None,
                "best_accepted_ops_over_rho": best.get("common_leaf_ops_over_rho") if best else None,
                "end_to_end_like": bool(below_cost and best),
                "known_positive": known_positive,
                "selected_leaves": selected_leaves,
                "selected_ops": selected_ops,
                "selected_ops_over_rho": selected_ops_over_rho,
                "set_cost_below_rho": below_cost,
                "top1_leaf_index": selected_leaves[0] if selected_leaves else None,
                "top1_score_key": list(score_key(selected[0], mode)) if selected else [],
                "transfer_index": transfer_index,
            }
        )
    selector_id = f"salt_assoc_{selector_index:03d}_{mode}_top{top_k}"
    return {
        "selector_id": selector_id,
        "selector_id_u64": min_transfer.digest_u64({"mode": mode, "top_k": top_k, "selector_index": selector_index}),
        "selector_index": selector_index,
        "mode": mode,
        "mode_code": MODE_CODES.get(mode, 0),
        "top_k": top_k,
        "selected_leaf_count": top_k,
        "covered_transfer_count": len(covered),
        "covered_transfers": sorted(covered),
        "heldout_covered_transfer_count": len(heldout_covered),
        "heldout_covered_transfers": sorted(heldout_covered),
        "set_cost_below_rho_transfer_count": len(set_cost_below),
        "set_cost_below_rho_transfers": sorted(set_cost_below),
        "end_to_end_like_transfer_count": len(end_to_end_like),
        "end_to_end_like_transfers": sorted(end_to_end_like),
        "heldout_end_to_end_like_transfer_count": len(heldout_end_to_end_like),
        "heldout_end_to_end_like_transfers": sorted(heldout_end_to_end_like),
        "top1_end_to_end_like_transfer_count": len(top1_end_to_end_like),
        "top1_end_to_end_like_transfers": sorted(top1_end_to_end_like),
        "full_sweep_equivalent": top_k >= FULL_SWEEP_TOP_K,
        "per_transfer_records": per_transfer_records,
    }


def choose_best(
    records: list[dict[str, Any]],
    *,
    allow_full: bool = False,
    max_top_k: int | None = None,
    prefer_end_to_end: bool = True,
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if (allow_full or not record.get("full_sweep_equivalent"))
        and (max_top_k is None or as_int(record.get("top_k")) <= max_top_k)
    ]
    if prefer_end_to_end:
        candidates.sort(
            key=lambda item: (
                -as_int(item.get("heldout_end_to_end_like_transfer_count")),
                -as_int(item.get("end_to_end_like_transfer_count")),
                -as_int(item.get("heldout_covered_transfer_count")),
                -as_int(item.get("covered_transfer_count")),
                as_int(item.get("top_k")),
                str(item.get("mode")),
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                -as_int(item.get("heldout_covered_transfer_count")),
                -as_int(item.get("covered_transfer_count")),
                as_int(item.get("top_k")),
                str(item.get("mode")),
            )
        )
    return candidates[0] if candidates else {}


def summarize(
    selector_records: list[dict[str, Any]],
    association_records: list[dict[str, Any]],
    materialized_targets: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    common_payload: dict[str, Any],
    fixed_predictor_payload: dict[str, Any],
) -> dict[str, Any]:
    best_e2e = choose_best(selector_records)
    best_coverage = choose_best(selector_records, prefer_end_to_end=False)
    compact_e2e = choose_best(selector_records, max_top_k=3)
    end_to_end_selectors = [
        record for record in selector_records if as_int(record.get("end_to_end_like_transfer_count")) > 0
    ]
    heldout_end_to_end_selectors = [
        record for record in selector_records if as_int(record.get("heldout_end_to_end_like_transfer_count")) > 0
    ]
    top1_end_to_end_selectors = [
        record for record in selector_records if as_int(record.get("top1_end_to_end_like_transfer_count")) > 0
    ]
    covered_selectors = [
        record for record in selector_records if as_int(record.get("heldout_covered_transfer_count")) > 0
    ]
    winning_records = [
        record
        for record in association_records
        if record.get("accepted_relation_export") and record.get("below_rho")
    ]
    transfer_count = len(transfer_indices(association_records))
    return {
        "association_leaf_record_count": len(association_records),
        "best_compact_selector_id": compact_e2e.get("selector_id"),
        "best_compact_top_k": compact_e2e.get("top_k"),
        "best_compact_heldout_end_to_end_like_transfer_count": compact_e2e.get("heldout_end_to_end_like_transfer_count"),
        "best_coverage_selector_id": best_coverage.get("selector_id"),
        "best_coverage_top_k": best_coverage.get("top_k"),
        "best_coverage_heldout_covered_transfer_count": best_coverage.get("heldout_covered_transfer_count"),
        "best_e2e_selector_id": best_e2e.get("selector_id"),
        "best_e2e_top_k": best_e2e.get("top_k"),
        "best_e2e_heldout_end_to_end_like_transfer_count": best_e2e.get("heldout_end_to_end_like_transfer_count"),
        "best_e2e_end_to_end_like_transfer_count": best_e2e.get("end_to_end_like_transfer_count"),
        "end_to_end_like_selector_count": len(end_to_end_selectors),
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "heldout_cover_selector_count": len(covered_selectors),
        "heldout_end_to_end_like_selector_count": len(heldout_end_to_end_selectors),
        "materialized_transfer_count": len(materialized_targets),
        "source_common_sweep_status": common_payload.get("claim_status"),
        "source_fixed_predictor_audit_status": fixed_predictor_payload.get("claim_status"),
        "target_transfer_count": transfer_count,
        "top1_end_to_end_like_selector_count": len(top1_end_to_end_selectors),
        "transfer_count": transfer_count,
        "verified": not failures,
        "winning_common_leaf_record_count": len(winning_records),
        "worker_interpretation": (
            "Salt-conditioned association rankers are charged as selected-set probes: "
            "the artifact is a candidate selector only when the selected leaves contain "
            "a relation-derived below-rho common leaf and their total common-leaf replay "
            "cost stays below generic rho for that transfer."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_SALT_CONDITIONED_ASSOCIATION_PREDICTOR_FAILED"
    if as_int(summary.get("heldout_end_to_end_like_selector_count")) > 0:
        return "SELECTED13_SALT_CONDITIONED_ASSOCIATION_PREDICTOR_HAS_HELDOUT_END_TO_END_CANDIDATE"
    if as_int(summary.get("end_to_end_like_selector_count")) > 0:
        return "SELECTED13_SALT_CONDITIONED_ASSOCIATION_PREDICTOR_HAS_KNOWN_END_TO_END_CANDIDATE"
    if as_int(summary.get("heldout_cover_selector_count")) > 0:
        return "SELECTED13_SALT_CONDITIONED_ASSOCIATION_PREDICTOR_COVERS_HELDOUT_NOT_BELOW_RHO_COST"
    return "SELECTED13_SALT_CONDITIONED_ASSOCIATION_PREDICTOR_NO_HELDOUT_COVERAGE"


def render_c_header(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = []
    for record in records:
        rows.append(
            "  {"
            f"{as_int(record.get('selector_index'))}ULL, "
            f"{as_int(record.get('selector_id_u64'))}ULL, "
            f"{as_int(record.get('mode_code'))}ULL, "
            f"{as_int(record.get('top_k'))}ULL, "
            f"{as_int(record.get('covered_transfer_count'))}ULL, "
            f"{as_int(record.get('heldout_covered_transfer_count'))}ULL, "
            f"{as_int(record.get('set_cost_below_rho_transfer_count'))}ULL, "
            f"{as_int(record.get('end_to_end_like_transfer_count'))}ULL, "
            f"{as_int(record.get('heldout_end_to_end_like_transfer_count'))}ULL, "
            f"{as_int(record.get('top1_end_to_end_like_transfer_count'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_ASSOCIATION_PREDICTOR_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_ASSOCIATION_PREDICTOR_PROBE_H

#include <stdint.h>

#define SELECTED13_SALT_ASSOC_SELECTOR_COUNT {len(records)}
#define SELECTED13_SALT_ASSOC_ASSOCIATION_LEAF_RECORD_COUNT {as_int(summary.get("association_leaf_record_count"))}
#define SELECTED13_SALT_ASSOC_MATERIALIZED_TRANSFER_COUNT {as_int(summary.get("materialized_transfer_count"))}
#define SELECTED13_SALT_ASSOC_HELDOUT_COVER_SELECTOR_COUNT {as_int(summary.get("heldout_cover_selector_count"))}
#define SELECTED13_SALT_ASSOC_END_TO_END_SELECTOR_COUNT {as_int(summary.get("end_to_end_like_selector_count"))}
#define SELECTED13_SALT_ASSOC_HELDOUT_END_TO_END_SELECTOR_COUNT {as_int(summary.get("heldout_end_to_end_like_selector_count"))}
#define SELECTED13_SALT_ASSOC_TOP1_END_TO_END_SELECTOR_COUNT {as_int(summary.get("top1_end_to_end_like_selector_count"))}

typedef struct {{
  uint64_t selector_index;
  uint64_t selector_id_u64;
  uint64_t mode_code;
  uint64_t top_k;
  uint64_t covered_transfer_count;
  uint64_t heldout_covered_transfer_count;
  uint64_t set_cost_below_rho_transfer_count;
  uint64_t end_to_end_like_transfer_count;
  uint64_t heldout_end_to_end_like_transfer_count;
  uint64_t top1_end_to_end_like_transfer_count;
}} selected13_salt_assoc_selector_t;

static const selected13_salt_assoc_selector_t SELECTED13_SALT_ASSOC_SELECTORS[] = {{
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
      sizeof(SELECTED13_SALT_ASSOC_SELECTORS) / sizeof(SELECTED13_SALT_ASSOC_SELECTORS[0]);
  uint64_t heldout_cover = 0;
  uint64_t end_to_end = 0;
  uint64_t heldout_end_to_end = 0;
  uint64_t top1_end_to_end = 0;

  if (selector_count != SELECTED13_SALT_ASSOC_SELECTOR_COUNT) failure_count++;
  if (selector_count == 0ULL) failure_count++;
  if (SELECTED13_SALT_ASSOC_ASSOCIATION_LEAF_RECORD_COUNT == 0ULL) failure_count++;
  if (SELECTED13_SALT_ASSOC_MATERIALIZED_TRANSFER_COUNT == 0ULL) failure_count++;

  for (size_t i = 0; i < selector_count; i++) {{
    const selected13_salt_assoc_selector_t *selector = &SELECTED13_SALT_ASSOC_SELECTORS[i];
    if (selector->selector_id_u64 == 0ULL) failure_count++;
    if (selector->mode_code == 0ULL) failure_count++;
    if (selector->top_k == 0ULL) failure_count++;
    if (selector->heldout_covered_transfer_count > 0ULL) heldout_cover++;
    if (selector->end_to_end_like_transfer_count > 0ULL) end_to_end++;
    if (selector->heldout_end_to_end_like_transfer_count > 0ULL) heldout_end_to_end++;
    if (selector->top1_end_to_end_like_transfer_count > 0ULL) top1_end_to_end++;
  }}

  if (heldout_cover != SELECTED13_SALT_ASSOC_HELDOUT_COVER_SELECTOR_COUNT) failure_count++;
  if (end_to_end != SELECTED13_SALT_ASSOC_END_TO_END_SELECTOR_COUNT) failure_count++;
  if (heldout_end_to_end != SELECTED13_SALT_ASSOC_HELDOUT_END_TO_END_SELECTOR_COUNT) failure_count++;
  if (top1_end_to_end != SELECTED13_SALT_ASSOC_TOP1_END_TO_END_SELECTOR_COUNT) failure_count++;

  printf("selected13_salt_assoc_predictor_preflight selectors=%llu leaf_records=%llu materialized_transfers=%llu heldout_cover=%llu end_to_end=%llu heldout_end_to_end=%llu top1_end_to_end=%llu failures=%llu\\n",
         (unsigned long long)selector_count,
         (unsigned long long)SELECTED13_SALT_ASSOC_ASSOCIATION_LEAF_RECORD_COUNT,
         (unsigned long long)SELECTED13_SALT_ASSOC_MATERIALIZED_TRANSFER_COUNT,
         (unsigned long long)heldout_cover,
         (unsigned long long)end_to_end,
         (unsigned long long)heldout_end_to_end,
         (unsigned long long)top1_end_to_end,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_salt_assoc_predictor_preflight_") as tmp:
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
    contract = load_json(Path(args.contract))
    fixed_predictor_payload = load_json(Path(args.fixed_predictor_audit))
    failures: list[dict[str, Any]] = []
    if common_payload.get("claim_status") != "SELECTED13_COMMON_LEAF_PAIR_SWEEP_HELDOUT_BELOW_RHO_CANDIDATES":
        failures.append({"code": "common_sweep_status_unexpected", "claim_status": common_payload.get("claim_status")})
    if contract.get("claim_status") != "FFE_SHARP_LANE_KERNEL_CONTRACT_READY":
        failures.append({"code": "contract_status_unexpected", "claim_status": contract.get("claim_status")})
    if fixed_predictor_payload.get("claim_status") != "SELECTED13_COMMON_LEAF_PREDICTOR_AUDIT_COVERS_HELDOUT_NOT_BELOW_RHO_COST":
        failures.append(
            {
                "code": "fixed_predictor_status_unexpected",
                "claim_status": fixed_predictor_payload.get("claim_status"),
            }
        )
    targets = common_sweep.contract_backfill_targets(contract)
    if not targets:
        failures.append({"code": "no_backfill_targets"})
    association_records, radius, materialize_failures, materialized_targets = materialize_association_records(
        args,
        targets,
        common_payload,
    )
    failures.extend(materialize_failures)
    selector_records = []
    index = 0
    for mode in args.scorer_modes:
        for top_k in args.top_ks:
            selector_records.append(evaluate_selector(index, mode, top_k, association_records))
            index += 1
    summary = summarize(
        selector_records,
        association_records,
        materialized_targets,
        failures,
        common_payload,
        fixed_predictor_payload,
    )
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
            "fixed_predictor_audit": str(Path(args.fixed_predictor_audit)),
            "leaf_limit": args.leaf_limit,
            "radius": radius,
            "scorer_modes": args.scorer_modes,
            "target": TARGET,
            "top_ks": args.top_ks,
            "transfer_source": str(Path(args.transfer_source)),
        },
        "materialized_targets": materialized_targets,
        "summary": summary,
        "selector_records": selector_records,
        "association_leaf_records": association_records,
        "failures": failures,
        "honesty_boundary": {
            "association_signals_only": True,
            "general_ecdlp_algorithm_claimed": False,
            "relation_labels_from_common_sweep": True,
            "selection_cost_note": "A selector is end-to-end-like only when the full selected leaf set for a transfer costs less than generic rho and contains a relation-derived below-rho common-leaf export.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-sweep", type=Path, default=DEFAULT_COMMON_SWEEP)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixed-predictor-audit", type=Path, default=DEFAULT_FIXED_PREDICTOR_AUDIT)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--leaf-limit", type=int)
    parser.add_argument("--scorer-modes", type=parse_csv, default=list(DEFAULT_SCORER_MODES))
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
