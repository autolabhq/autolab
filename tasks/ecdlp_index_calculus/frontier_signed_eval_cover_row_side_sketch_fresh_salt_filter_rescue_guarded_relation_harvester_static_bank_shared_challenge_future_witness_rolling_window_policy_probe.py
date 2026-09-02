#!/usr/bin/env python3
"""Calibrate a public rolling-window policy for compact ECDLP witnesses.

The future-witness stress probe found an immediate held-out positive on salts
228-235 and a disjoint negative on 236-243.  This driver materializes a longer
salt range once, scores rolling windows using public row/event features, and
then audits whether those public scores predict verifier-backed two-equation
witnesses.

The policy features intentionally stop before pair verification: event density,
same-term-signature multiplicity, and public filter cost are available after
row materialization, while derived secrets are used only as labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LIVE_TASK_DIR = Path("/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
if LIVE_TASK_DIR.exists():
    sys.path.insert(0, str(LIVE_TASK_DIR))

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_future_witness_stress_probe as stress_probe
import relation_probe


DEFAULT_OUT = (
    Path("ecdlp_index_calculus_state")
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_future_witness_rolling_window_policy_probe.json"
)


def shared_seed(prefix: str, target: str) -> str:
    return f"{prefix}:shared-challenge:{target}"


def materialize_candidate_row(
    verifier: Any,
    records: list[dict[str, Any]],
    config_source: dict[str, Any],
    spec: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not (
        args.challenge_seed_prefix
        or args.row_seed_prefix
        or args.scout_seed_prefix
        or args.filter_seed_prefix
    ):
        return stress_probe.materialize_candidate_row(verifier, records, config_source, spec, args)

    row = spec["row"]
    cfg = stress_probe.materialization_probe.cfg_for_row(config_source, row, args.row_pool)
    local_args = stress_probe.direct_witness_probe.shared_target_args(args, row)
    target = str(row.get("target"))
    if args.challenge_seed_prefix:
        local_args.challenge_seed = shared_seed(str(args.challenge_seed_prefix), target)
        local_args.seed = local_args.challenge_seed
    if args.row_seed_prefix:
        local_args.row_seed_prefix = str(args.row_seed_prefix)
    if args.scout_seed_prefix:
        local_args.scout_seed_prefix = shared_seed(str(args.scout_seed_prefix), target)
    elif args.challenge_seed_prefix:
        local_args.scout_seed_prefix = local_args.challenge_seed
    if args.filter_seed_prefix:
        local_args.filter_seed_prefix = shared_seed(str(args.filter_seed_prefix), target)
    elif args.challenge_seed_prefix:
        local_args.filter_seed_prefix = local_args.challenge_seed

    built = stress_probe.materialization_probe.build_eval_context_with_seed_control(
        verifier,
        records,
        cfg,
        local_args,
    )
    if built.get("error"):
        return (
            {
                "row_key": stress_probe.row_key(row),
                "target": spec.get("target"),
                "source_row_key": spec.get("source_row_key"),
                "error": built["error"],
            },
            built,
        )
    p = int(built["p"])
    scouts_by_pos = {int(scout["scout_pos"]): scout for scout in built["scouts"]}
    components = stress_probe.association_probe.build_components(built["scouts"], built["rows"], p)
    feature_rows = stress_probe.preassociation_filter_probe.leaf_public_features(components, scouts_by_pos, p)
    mode = str(spec.get("filter_mode"))
    top_k = int(spec.get("top_k") or 1)
    ranked = sorted(
        feature_rows,
        key=lambda item: stress_probe.preassociation_filter_probe.score_key(
            item,
            mode,
            f"{local_args.filter_seed_prefix}:{built['target']}:{mode}",
        ),
    )
    selected = {int(item["leaf_index"]) for item in ranked[: max(1, top_k)]}
    scan = stress_probe.direct_witness_probe.scan_selected(verifier, built, components, selected, local_args)
    compact_scan = stress_probe.direct_witness_probe.compact_scan(scan)
    event_summaries = [
        stress_probe.direct_witness_probe.dependency_circuit_probe.event_summary(index, event, int(built["order"]))
        for index, event in enumerate(scan.get("relation_events") or [])
    ]
    events = []
    for event, summary in zip(scan.get("relation_events") or [], event_summaries):
        events.append(
            {
                **summary,
                "row_key": stress_probe.row_key(row),
                "target": built["target"],
                "row_schedule_mode": row.get("row_schedule_mode"),
                "row_schedule_count": int(row.get("row_schedule_count") or 0),
                "row_schedule_salt": int(row.get("row_schedule_salt") or 0),
                "filter_mode": mode,
                "top_k": top_k,
                "preassociation_filter_ops": int(compact_scan.get("preassociation_filter_ops") or 0),
                "preassociation_filter_ops_over_rho": stress_probe.clean_ratio(
                    compact_scan.get("preassociation_filter_ops_over_rho")
                ),
                "_form": event["form"],
            }
        )
    result = {
        "row_key": stress_probe.row_key(row),
        "target": built["target"],
        "source_row_key": spec.get("source_row_key"),
        "challenge_seed": built.get("challenge_seed"),
        "row_seed_prefix": built.get("row_seed_prefix"),
        "scout_seed_prefix": built.get("scout_seed_prefix"),
        "filter_seed_prefix": getattr(local_args, "filter_seed_prefix", None),
        "row_schedule_mode": row.get("row_schedule_mode"),
        "row_schedule_count": int(row.get("row_schedule_count") or 0),
        "row_schedule_salt": int(row.get("row_schedule_salt") or 0),
        "filter_mode": mode,
        "top_k": top_k,
        "generic_rho_steps": int(built["generic_rho_steps"]),
        "selected_leaf_indices": sorted(selected),
        "ranked_prefix": compact_scan,
        "event_count": len(events),
        "events": events,
    }
    return result, built


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    event_signatures = [
        {
            "relation_index": event.get("relation_index"),
            "leaf_index": event.get("leaf_index"),
            "candidate_pos": event.get("candidate_pos"),
            "row_schedule_salt": row.get("row_schedule_salt"),
            "q_coeff": event.get("q_coeff"),
            "rhs": event.get("rhs"),
            "terms": event.get("terms"),
            "factor_support": event.get("factor_support"),
            "term_shape": event.get("term_shape"),
            "term_signature": list(stress_probe.term_signature(event)),
        }
        for event in row.get("events") or []
    ]
    return {
        "target": row.get("target"),
        "row_key": row.get("row_key"),
        "source_row_key": row.get("source_row_key"),
        "row_schedule_salt": row.get("row_schedule_salt"),
        "filter_mode": row.get("filter_mode"),
        "top_k": row.get("top_k"),
        "event_count": int(row.get("event_count") or 0),
        "generic_rho_steps": row.get("generic_rho_steps"),
        "challenge_seed": row.get("challenge_seed"),
        "row_seed_prefix": row.get("row_seed_prefix"),
        "scout_seed_prefix": row.get("scout_seed_prefix"),
        "filter_seed_prefix": row.get("filter_seed_prefix"),
        "ranked_prefix": {
            key: value
            for key, value in (row.get("ranked_prefix") or {}).items()
            if key
            in {
                "candidate_verifications",
                "preassociation_filter_ops",
                "preassociation_filter_ops_over_rho",
                "rank",
                "relation_count",
                "selected_hit_events",
                "selected_hit_roots",
                "selected_leaf_count",
            }
        },
        "error": row.get("error"),
        "event_signatures": event_signatures,
    }


def iter_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("error"):
            continue
        for event in row.get("events") or []:
            key = stress_probe.event_compact_key(event)
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
    return events


def same_signature_pairs(events: list[dict[str, Any]], generic_rho: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[stress_probe.term_signature(event)].append(event)
    pairs = []
    for signature, values in grouped.items():
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                if str(left.get("row_key")) == str(right.get("row_key")):
                    continue
                ops = stress_probe.pair_ops((left, right))
                pairs.append(
                    {
                        "signature": list(signature),
                        "row_keys": [left.get("row_key"), right.get("row_key")],
                        "salts": [
                            int(left.get("row_schedule_salt") or 0),
                            int(right.get("row_schedule_salt") or 0),
                        ],
                        "ops": ops,
                        "ops_over_rho": stress_probe.clean_ratio(ops / max(1, generic_rho)),
                    }
                )
    pairs.sort(key=lambda item: (item["ops_over_rho"] or 10**18, item["salts"], item["row_keys"]))
    return pairs


def public_features_for_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    materialized = [row for row in rows if not row.get("error")]
    events = iter_events(materialized)
    generic_rho = max((int(row.get("generic_rho_steps") or 0) for row in materialized), default=0)
    pairs = same_signature_pairs(events, generic_rho)
    signature_counts = Counter(stress_probe.term_signature(event) for event in events)
    row_event_counts = Counter(str(event.get("row_key")) for event in events)
    selected_hit_events = sum(int((row.get("ranked_prefix") or {}).get("selected_hit_events") or 0) for row in materialized)
    selected_hit_roots = sum(int((row.get("ranked_prefix") or {}).get("selected_hit_roots") or 0) for row in materialized)
    filter_ops = sum(int((row.get("ranked_prefix") or {}).get("preassociation_filter_ops") or 0) for row in materialized)
    return {
        "row_count": len(rows),
        "materialized_row_count": len(materialized),
        "error_count": sum(1 for row in rows if row.get("error")),
        "event_count": len(events),
        "event_row_count": len(row_event_counts),
        "max_events_on_one_row": max(row_event_counts.values(), default=0),
        "selected_hit_events": selected_hit_events,
        "selected_hit_roots": selected_hit_roots,
        "public_filter_ops": filter_ops,
        "public_filter_ops_over_rho": stress_probe.clean_ratio(filter_ops / max(1, generic_rho)),
        "same_signature_pair_count": len(pairs),
        "max_signature_multiplicity": max(signature_counts.values(), default=0),
        "min_same_signature_pair_ops_over_rho": pairs[0]["ops_over_rho"] if pairs else None,
        "top_same_signature_pairs": pairs[:5],
    }


def public_features(rows_by_target: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_target = {
        target: public_features_for_target(rows)
        for target, rows in sorted(rows_by_target.items())
    }
    pair_targets = [
        target
        for target, features in per_target.items()
        if int(features.get("same_signature_pair_count") or 0) > 0
    ]
    event_targets = [
        target
        for target, features in per_target.items()
        if int(features.get("event_count") or 0) > 0
    ]
    same_pair_ratios = [
        float(features["min_same_signature_pair_ops_over_rho"])
        for features in per_target.values()
        if features.get("min_same_signature_pair_ops_over_rho") is not None
    ]
    return {
        "target_features": per_target,
        "target_count": len(per_target),
        "targets_with_events": len(event_targets),
        "targets_with_same_signature_pair": len(pair_targets),
        "event_count": sum(int(features.get("event_count") or 0) for features in per_target.values()),
        "event_row_count": sum(int(features.get("event_row_count") or 0) for features in per_target.values()),
        "same_signature_pair_count": sum(
            int(features.get("same_signature_pair_count") or 0) for features in per_target.values()
        ),
        "max_signature_multiplicity": max(
            (int(features.get("max_signature_multiplicity") or 0) for features in per_target.values()),
            default=0,
        ),
        "min_same_signature_pair_ops_over_rho": min(same_pair_ratios) if same_pair_ratios else None,
        "public_filter_ops": sum(int(features.get("public_filter_ops") or 0) for features in per_target.values()),
    }


def public_policy_score(features: dict[str, Any]) -> tuple[Any, ...]:
    # Smaller is better; all fields are public pre-verification features.
    return (
        -int(features.get("targets_with_same_signature_pair") or 0),
        float(features.get("min_same_signature_pair_ops_over_rho") or 10**18),
        -int(features.get("same_signature_pair_count") or 0),
        -int(features.get("targets_with_events") or 0),
        -int(features.get("event_count") or 0),
        int(features.get("public_filter_ops") or 10**18),
    )


def policy_threshold_from_training(training_features: dict[str, Any]) -> dict[str, Any]:
    return {
        "min_targets_with_same_signature_pair": int(training_features.get("targets_with_same_signature_pair") or 0),
        "min_same_signature_pair_count": int(training_features.get("same_signature_pair_count") or 0),
        "max_min_same_signature_pair_ops_over_rho": training_features.get("min_same_signature_pair_ops_over_rho"),
    }


def threshold_selects(features: dict[str, Any], threshold: dict[str, Any]) -> bool:
    if int(features.get("targets_with_same_signature_pair") or 0) < int(
        threshold.get("min_targets_with_same_signature_pair") or 0
    ):
        return False
    if int(features.get("same_signature_pair_count") or 0) < int(threshold.get("min_same_signature_pair_count") or 0):
        return False
    max_ratio = threshold.get("max_min_same_signature_pair_ops_over_rho")
    if max_ratio is not None:
        ratio = features.get("min_same_signature_pair_ops_over_rho")
        if ratio is None or float(ratio) > float(max_ratio):
            return False
    return True


def analyze_window(
    verifier: Any,
    start: int,
    window_size: int,
    rows_by_target: dict[str, list[dict[str, Any]]],
    built_by_row: dict[str, dict[str, Any]],
    top_pairs_per_mode: int,
) -> dict[str, Any]:
    end = start + window_size - 1
    window_rows_by_target = {
        target: [
            row
            for row in rows
            if int(row.get("row_schedule_salt") or -1) >= start
            and int(row.get("row_schedule_salt") or -1) <= end
        ]
        for target, rows in rows_by_target.items()
    }
    features = public_features(window_rows_by_target)
    targets = [
        stress_probe.analyze_target(verifier, target, rows, built_by_row, top_pairs_per_mode)
        for target, rows in sorted(window_rows_by_target.items())
    ]
    verified_targets = [target for target in targets if target.get("has_direct_two_equation_witness")]
    verified_ops = [
        float(target["min_verified_pair_ops_over_rho"])
        for target in targets
        if target.get("min_verified_pair_ops_over_rho") is not None
    ]
    return {
        "salt_start": start,
        "salt_end": end,
        "public_features": features,
        "public_policy_score": list(public_policy_score(features)),
        "target_count": len(targets),
        "targets_with_direct_two_equation_witness": len(verified_targets),
        "all_targets_have_direct_two_equation_witness": bool(targets) and len(verified_targets) == len(targets),
        "total_verified_pairs": sum(int(target.get("verified_pair_count") or 0) for target in targets),
        "min_verified_pair_ops_over_rho": min(verified_ops) if verified_ops else None,
        "target_summaries": [
            {
                "target": target.get("target"),
                "event_count": target.get("event_count"),
                "pair_candidate_count": target.get("pair_candidate_count"),
                "verified_pair_count": target.get("verified_pair_count"),
                "has_direct_two_equation_witness": target.get("has_direct_two_equation_witness"),
                "min_verified_pair_ops_over_rho": target.get("min_verified_pair_ops_over_rho"),
                "first_verified_public_filter_rank": target.get("first_verified_public_filter_rank"),
                "first_verified_equation_shape_rank": target.get("first_verified_equation_shape_rank"),
            }
            for target in targets
        ],
    }


def summarize(windows: list[dict[str, Any]], training_window: dict[str, Any], threshold: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(windows, key=lambda window: (tuple(window["public_policy_score"]), window["salt_start"]))
    positives = [window for window in windows if int(window.get("targets_with_direct_two_equation_witness") or 0) > 0]
    all_target_positives = [window for window in windows if window.get("all_targets_have_direct_two_equation_witness")]
    selected = [window for window in windows if threshold_selects(window["public_features"], threshold)]
    selected_positive = [
        window
        for window in selected
        if int(window.get("targets_with_direct_two_equation_witness") or 0) > 0
    ]
    training_end = int(training_window.get("salt_end") or 0)

    def ranked_window(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "salt_start": row.get("salt_start"),
            "salt_end": row.get("salt_end"),
            "public_policy_rank": 1 + ranked.index(row),
            "public_policy_score": row.get("public_policy_score"),
            "public_features": row.get("public_features"),
            "targets_with_direct_two_equation_witness": row.get("targets_with_direct_two_equation_witness"),
            "all_targets_have_direct_two_equation_witness": row.get("all_targets_have_direct_two_equation_witness"),
            "min_verified_pair_ops_over_rho": row.get("min_verified_pair_ops_over_rho"),
        }

    first_unseen_positive = next(
        (
            row
            for row in ranked
            if int(row.get("salt_start") or 0) > training_end
            and int(row.get("targets_with_direct_two_equation_witness") or 0) > 0
        ),
        None,
    )
    first_unseen_all_target_positive = next(
        (
            row
            for row in ranked
            if int(row.get("salt_start") or 0) > training_end
            and row.get("all_targets_have_direct_two_equation_witness")
        ),
        None,
    )
    return {
        "window_count": len(windows),
        "positive_window_count": len(positives),
        "all_target_positive_window_count": len(all_target_positives),
        "public_threshold_selected_window_count": len(selected),
        "public_threshold_true_positive_count": len(selected_positive),
        "public_threshold_false_positive_count": len(selected) - len(selected_positive),
        "top8_public_policy_positive_count": sum(
            1 for row in ranked[:8] if int(row.get("targets_with_direct_two_equation_witness") or 0) > 0
        ),
        "top8_public_policy_all_target_positive_count": sum(
            1 for row in ranked[:8] if row.get("all_targets_have_direct_two_equation_witness")
        ),
        "top12_public_policy_positive_count": sum(
            1 for row in ranked[:12] if int(row.get("targets_with_direct_two_equation_witness") or 0) > 0
        ),
        "top12_public_policy_all_target_positive_count": sum(
            1 for row in ranked[:12] if row.get("all_targets_have_direct_two_equation_witness")
        ),
        "first_unseen_positive_window": ranked_window(first_unseen_positive) if first_unseen_positive else None,
        "first_unseen_all_target_positive_window": (
            ranked_window(first_unseen_all_target_positive) if first_unseen_all_target_positive else None
        ),
        "training_window": {
            "salt_start": training_window.get("salt_start"),
            "salt_end": training_window.get("salt_end"),
            "public_features": training_window.get("public_features"),
            "targets_with_direct_two_equation_witness": training_window.get("targets_with_direct_two_equation_witness"),
            "all_targets_have_direct_two_equation_witness": training_window.get("all_targets_have_direct_two_equation_witness"),
            "min_verified_pair_ops_over_rho": training_window.get("min_verified_pair_ops_over_rho"),
        },
        "threshold": threshold,
        "top_public_policy_windows": [
            ranked_window(window)
            for window in ranked[:12]
        ],
        "positive_windows": [
            ranked_window(window)
            for window in positives
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-source", type=Path, default=stress_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=stress_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=stress_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--salt-start", type=int, default=228)
    parser.add_argument("--salt-end", type=int, default=299)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--training-window-start", type=int, default=228)
    parser.add_argument("--top-pairs-per-mode", type=int, default=3)
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
    parser.add_argument("--allow-combined-coefficients", dest="require_unit_coefficients", action="store_false")
    parser.set_defaults(require_unit_coefficients=True)
    parser.add_argument("--row-factor", type=int, default=512)
    parser.add_argument("--product-factor", type=int, default=4096)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--challenge-seed-prefix")
    parser.add_argument("--row-seed-prefix")
    parser.add_argument("--scout-seed-prefix")
    parser.add_argument("--filter-seed-prefix")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.salt_end < args.salt_start:
        raise SystemExit("--salt-end must be >= --salt-start")
    if args.window_size <= 0:
        raise SystemExit("--window-size must be positive")
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")

    bank = stress_probe.load_json(args.bank_source)
    config_source = stress_probe.load_json(args.config_source)
    direct_source = stress_probe.load_json(args.direct_source)
    bank_rows = {
        stress_probe.row_key(row): row
        for row in bank.get("bank_rows") or []
        if isinstance(row, dict) and stress_probe.row_key(row)
    }
    specs = stress_probe.witness_specs(
        direct_source,
        bank_rows,
        args.salt_start,
        args.salt_end - args.salt_start + 1,
    )

    verifier = relation_probe.load_verifier_module()
    records = verifier.load_records()
    rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    built_by_row: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(specs, start=1):
        result, built = materialize_candidate_row(verifier, records, config_source, spec, args)
        rows_by_target[str(spec.get("target"))].append(result)
        if not result.get("error"):
            built_by_row[stress_probe.row_key(result)] = built
        print(
            json.dumps(
                {
                    "progress": index,
                    "total": len(specs),
                    "row_key": result.get("row_key"),
                    "event_count": result.get("event_count"),
                    "error": result.get("error"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    window_starts = range(
        args.training_window_start,
        args.salt_end - args.window_size + 2,
        args.stride,
    )
    windows = [
        analyze_window(
            verifier,
            start,
            args.window_size,
            rows_by_target,
            built_by_row,
            args.top_pairs_per_mode,
        )
        for start in window_starts
        if start >= args.salt_start and start + args.window_size - 1 <= args.salt_end
    ]
    training_window = next(
        (window for window in windows if int(window.get("salt_start") or 0) == int(args.training_window_start)),
        None,
    )
    if training_window is None:
        raise SystemExit("training window was not materialized")
    threshold = policy_threshold_from_training(training_window["public_features"])
    summary = summarize(windows, training_window, threshold)

    output = {
        "schema": "ecdlp_frontier_guarded_harvester_static_bank_shared_challenge_future_witness_rolling_window_policy_probe_v1",
        "method": "public_same_signature_event_density_policy_for_heldout_compact_witness_windows",
        "parameters": {
            "live_task_dir": str(LIVE_TASK_DIR),
            "bank_source": str(args.bank_source),
            "config_source": str(args.config_source),
            "direct_source": str(args.direct_source),
            "salt_start": args.salt_start,
            "salt_end": args.salt_end,
            "window_size": args.window_size,
            "stride": args.stride,
            "training_window_start": args.training_window_start,
            "top_pairs_per_mode": args.top_pairs_per_mode,
            "row_pool": args.row_pool,
            "scout_limit": args.scout_limit,
            "scout_mode": args.scout_mode,
            "scout_order": args.scout_order,
            "selected_limit": args.selected_limit,
            "factor_base_size": args.factor_base_size,
            "seed": args.seed,
            "challenge_seed_prefix": args.challenge_seed_prefix,
            "row_seed_prefix": args.row_seed_prefix,
            "scout_seed_prefix": args.scout_seed_prefix,
            "filter_seed_prefix": args.filter_seed_prefix,
        },
        "materialized_rows": [compact_row(row) for rows in rows_by_target.values() for row in rows],
        "summary": summary,
        "windows": windows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
