#!/usr/bin/env python3
"""Audit the public pre-surface FFE branch against full-source obligations."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.p1436_presurface_full_charge_preflight.v1"
EXPECTED_SCHEMAS = {
    "generator": "ecdlp_low_term_total2_ffe_public_factor_presurface_generator_audit_probe_v1",
    "pruning": "ecdlp_low_term_total2_ffe_public_factor_presurface_pruning_audit_probe_v1",
    "stage_guard": "ecdlp_low_term_total2_ffe_public_factor_stage_guard_charge_audit_probe_v1",
    "backfill": "ecdlp_low_term_total2_ffe_public_factor_presurface_backfill_audit_probe_v1",
    "prefactor": "ecdlp_low_term_total2_ffe_public_factor_prefactor_screen_audit_probe_v1",
    "screen_holdout": "ecdlp_low_term_total2_ffe_public_factor_screen_holdout_audit_probe_v1",
    "historical_prefactor": "ecdlp_low_term_total2_ffe_public_factor_prefactor_screen_audit_probe_v1",
    "p1324": "ecdlp.low_term_total2_p1324_public_prefix_slice_qroot_route_join_after_p1323.v1",
    "r68": "p1553.ffe_fixed_sum_information_conservation.r68.v1",
}
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
LIVE_ROOT = Path("/Volumes/Volume/git/autolab")
LIVE_STATE = LIVE_ROOT / "ecdlp_index_calculus_state"
LIVE_TASK = LIVE_ROOT / "tasks" / "ecdlp_index_calculus"
LONG_STEM = (
    "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_"
    "guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_"
    "low_term_total2_ffe_public_factor"
)
DEFAULT_PATHS = {
    "generator": LIVE_STATE
    / "public_factor_presurface_generator_audit_low_total2_816_975_probe.json",
    "pruning": LIVE_STATE
    / "public_factor_presurface_pruning_audit_low_total2_816_975_probe.json",
    "stage_guard": LIVE_STATE
    / "public_factor_stage_guard_charge_audit_low_total2_816_975_probe.json",
    "backfill": LIVE_STATE
    / "public_factor_presurface_backfill_audit_low_total2_816_975_probe.json",
    "prefactor": LIVE_STATE
    / "public_factor_prefactor_screen_audit_low_total2_816_975_probe.json",
    "screen_holdout": LIVE_STATE
    / "public_factor_screen_holdout_audit_low_total2_816_975_probe.json",
    "historical_prefactor": LIVE_STATE
    / f"{LONG_STEM}_prefactor_screen_audit_232_375_probe.json",
    "p1324": LIVE_STATE
    / "p1324_public_prefix_slice_qroot_route_join_after_p1323_probe.json",
    "r68": WORKTREE_ROOT
    / "p1553_ffe_fixed_sum_information_conservation_report_r68.json",
}
DEFAULT_CODE_PATHS = {
    "generator_source": LIVE_TASK
    / f"{LONG_STEM}_presurface_generator_audit_probe.py",
    "backfill_source": LIVE_TASK / "public_factor_presurface_backfill_audit_probe.py",
    "prefactor_source": LIVE_TASK / "public_factor_prefactor_screen_audit_probe.py",
    "screen_holdout_source": LIVE_TASK
    / "public_factor_screen_holdout_audit_probe.py",
}
DEFAULT_OUTPUT = (
    WORKTREE_ROOT
    / "ecdlp_index_calculus_state"
    / "p1436_presurface_full_charge_preflight.json"
)
REQUIRED_FORBIDDEN_SELECTION_LABELS = {
    "public_key_verified",
    "below_rho",
    "rank",
    "relation_count",
    "quadratic_preserves_selected_root_pairs",
    "public_factor_quadratic_root_beats_rho",
    "public_factor_quadratic_root_ops_over_rho",
}
ROUTE_NAMES = (
    "leave_one_target_route",
    "leave_one_target_window_route",
    "leave_one_window_route",
    "rolling_forward_window_route",
)
ROUTE_MODES = ("baseline_policy_only", "trained_screen_policy")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def validate_schemas(inputs: dict[str, dict[str, Any]]) -> None:
    mismatches = {
        name: {
            "expected": EXPECTED_SCHEMAS[name],
            "observed": value.get("schema"),
        }
        for name, value in inputs.items()
        if value.get("schema") != EXPECTED_SCHEMAS[name]
    }
    if mismatches:
        raise ValueError(f"input schema mismatch: {json.dumps(mismatches, sort_keys=True)}")


def heldout_rows(screen_holdout: dict[str, Any]) -> list[dict[str, Any]]:
    summary = screen_holdout.get("summary") or {}
    rows = []
    for route_name in ROUTE_NAMES:
        route = summary.get(route_name) or {}
        for mode in ROUTE_MODES:
            row = route.get(mode)
            if not isinstance(row, dict):
                continue
            rows.append({"route": route_name, "mode": mode, **row})
    return rows


def minimum_row(
    rows: list[dict[str, Any]],
    ratio_field: str,
    *,
    robust: bool,
) -> dict[str, Any] | None:
    eligible = []
    for row in rows:
        ratio = row.get(ratio_field)
        full_aggregate_recall = float(row.get("test_strict_recall") or 0.0) == 1.0
        robust_split = (
            row.get("all_splits_full_recall") is True
            and row.get("all_splits_positive") is True
        )
        if ratio is not None and full_aggregate_recall and (not robust or robust_split):
            eligible.append(row)
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row[ratio_field]))


def compact_route(row: dict[str, Any] | None, ratio_field: str) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "route": row["route"],
        "mode": row["mode"],
        "ratio": row[ratio_field],
        "test_strict_recall": row.get("test_strict_recall"),
        "all_splits_full_recall": row.get("all_splits_full_recall"),
        "all_splits_positive": row.get("all_splits_positive"),
        "split_count": row.get("split_count"),
        "test_strict_surface_count": row.get("test_strict_surface_count"),
        "test_recovered_strict_surface_count": row.get(
            "test_recovered_strict_surface_count"
        ),
        "strict_public_quadratic_root_ops": row.get(
            "strict_public_quadratic_root_ops"
        ),
        "all_selected_evaluated_factor_count": row.get(
            "all_selected_evaluated_factor_count"
        ),
        "all_selected_selector_eval_ops": row.get(
            "all_selected_selector_eval_ops"
        ),
        "sum_generic_rho_steps": row.get(
            "sum_generic_rho_steps_for_all_test_strict"
        ),
    }


def proxy_gap(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    root_ops = int(row.get("strict_public_quadratic_root_ops") or 0)
    evaluated = int(row.get("all_selected_evaluated_factor_count") or 0)
    selector = int(row.get("all_selected_selector_eval_ops") or 0)
    rho = int(row.get("sum_generic_rho_steps_for_all_test_strict") or 0)
    return {
        "root_plus_evaluated_factor_proxy_ops": root_ops + evaluated,
        "evaluated_factor_proxy_excess_over_rho": root_ops + evaluated - rho,
        "root_plus_selector_eval_ops": root_ops + selector,
        "selector_eval_excess_over_rho": root_ops + selector - rho,
        "sum_generic_rho_steps": rho,
        "factorization_cost_included": False,
    }


def historical_near_miss(prefactor: dict[str, Any]) -> dict[str, Any]:
    row = (prefactor.get("summary") or {}).get("best_single_policy_stream_proxy") or {}
    root_ops = int(row.get("strict_public_quadratic_root_ops") or 0)
    evaluated = int(row.get("all_proposal_evaluated_factor_count") or 0)
    selector = int(row.get("all_proposal_selector_eval_ops") or 0)
    rho = int(row.get("sum_generic_rho_steps_for_strict") or 0)
    return {
        "status": (prefactor.get("summary") or {}).get("status"),
        "proposal_surface_count": row.get("proposal_count"),
        "strict_surface_count": row.get("strict_surface_count"),
        "evaluated_factor_proxy_over_rho": row.get(
            "root_plus_evaluated_factor_proxy_over_strict_rho"
        ),
        "evaluated_factor_proxy_excess_over_rho": root_ops + evaluated - rho,
        "selector_eval_over_rho": row.get(
            "root_plus_selector_eval_ops_over_strict_rho"
        ),
        "selector_eval_excess_over_rho": root_ops + selector - rho,
        "factorization_cost_included": False,
        "interpretation": (
            "This superseded corpus is retained because its evaluated-factor proxy "
            "missed rho by only two units. The selector ledger and omitted "
            "factorization cost prevent a speedup claim."
        ),
    }


def evaluate(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validate_schemas(inputs)
    generator = inputs["generator"]
    pruning = inputs["pruning"]
    stage_guard = inputs["stage_guard"]
    backfill = inputs["backfill"]
    prefactor = inputs["prefactor"]
    screen = inputs["screen_holdout"]
    p1324 = inputs["p1324"]
    r68 = inputs["r68"]

    generator_summary = generator["summary"]["promoted_pre_surface_profile"]["summary"]
    pruning_summary = pruning["summary"]
    stage_summary = stage_guard["summary"]
    backfill_summary = backfill["summary"]
    prefactor_summary = prefactor["summary"]
    screen_summary = screen["summary"]
    p1324_summary = p1324["summary"]

    forbidden = set(generator.get("parameters", {}).get("selection_labels_forbidden") or [])
    public_generator_pass = (
        REQUIRED_FORBIDDEN_SELECTION_LABELS <= forbidden
        and float(generator_summary["row_leaf_recall"]) == 1.0
        and int(generator_summary["missed_materialized_surface_count"]) == 0
    )

    proposal_count = int(generator_summary["proposal_surface_count"])
    complete_materialization_pass = (
        backfill_summary.get("charge_ready_for_all_proposals") is True
        and float(backfill_summary.get("materialization_coverage") or 0.0) == 1.0
        and backfill_summary.get("unmaterialized_proposal_surface_count") == 0
        and int(backfill_summary.get("proposal_surface_count") or 0)
        == proposal_count
        and int(screen_summary.get("proposal_surface_count") or 0) == proposal_count
    )

    guard = backfill_summary["best_full_materialized_guard"]
    guard_summary = guard["summary"]
    guard_pass = (
        float(guard_summary["strict_recall_on_materialized"]) == 1.0
        and int(guard_summary["missed_strict_surface_count"]) == 0
        and int(guard_summary["false_positive_materialized_surface_count"]) == 0
        and float(guard_summary["max_charged_ops_over_rho"]) < 1.0
    )

    target_count = int(screen_summary.get("target_count") or 0)
    target_diversity_pass = target_count >= 4
    routes = heldout_rows(screen)
    evaluated_ratio_field = (
        "root_plus_evaluated_factor_proxy_over_all_test_strict_rho"
    )
    selector_ratio_field = "root_plus_selector_eval_ops_over_all_test_strict_rho"
    aggregate_evaluated = minimum_row(routes, evaluated_ratio_field, robust=False)
    robust_evaluated = minimum_row(routes, evaluated_ratio_field, robust=True)
    aggregate_selector = minimum_row(routes, selector_ratio_field, robust=False)
    robust_selector = minimum_row(routes, selector_ratio_field, robust=True)
    heldout_evaluated_pass = bool(
        robust_evaluated
        and float(robust_evaluated[evaluated_ratio_field]) < 1.0
    )
    heldout_selector_pass = bool(
        robust_selector and float(robust_selector[selector_ratio_field]) < 1.0
    )

    cost_note = str(prefactor.get("parameters", {}).get("charge_proxy_note") or "")
    full_factorization_cost_charged = (
        prefactor.get("parameters", {}).get("full_sage_factorization_cost_charged")
        is True
    )
    all_rejections_charged = (
        prefactor.get("parameters", {}).get("all_rejected_candidates_charged")
        is True
    )
    complete_source_cost_pass = (
        full_factorization_cost_charged
        and all_rejections_charged
        and heldout_selector_pass
    )

    new_rows = int(backfill_summary.get("new_independent_fixed_sum_row_count") or 0)
    fresh_rank_delta = int(backfill_summary.get("fresh_rank_delta") or 0)
    fresh_row_pass = (
        new_rows > 0
        and fresh_rank_delta > 0
        and p1324_summary.get("exact_route_join_gate_passed") is True
        and p1324_summary.get("additional_or_target_independent_gate_passed") is True
    )

    obligations = {
        "public_label_sequestered_presurface_generator": {
            "pass": public_generator_pass,
            "profile": generator["summary"]["promoted_pre_surface_profile"]["profile"],
            "proposal_surface_count": proposal_count,
            "strict_surface_count": generator_summary[
                "materialized_strict_surface_count"
            ],
            "row_leaf_recall": generator_summary["row_leaf_recall"],
            "proposal_precision": generator_summary[
                "proposal_to_materialized_precision"
            ],
            "required_forbidden_labels_present": sorted(
                REQUIRED_FORBIDDEN_SELECTION_LABELS & forbidden
            ),
        },
        "complete_public_proposal_materialization": {
            "pass": complete_materialization_pass,
            "prior_stage_charge_ready": stage_summary[
                "charge_ready_for_all_proposals"
            ],
            "prior_stage_materialization_coverage": stage_summary[
                "materialization_coverage"
            ],
            "backfill_charge_ready": backfill_summary[
                "charge_ready_for_all_proposals"
            ],
            "backfill_materialization_coverage": backfill_summary[
                "materialization_coverage"
            ],
            "unmaterialized_proposal_surface_count": backfill_summary[
                "unmaterialized_proposal_surface_count"
            ],
        },
        "exact_full_recall_materialized_factor_guard": {
            "pass": guard_pass,
            "guard": guard["guard"],
            **guard_summary,
            "scope": "materialized component only",
        },
        "frozen_disjoint_target_family_coverage": {
            "pass": target_diversity_pass,
            "target_count": target_count,
            "minimum_target_count": 4,
            "window_count": screen_summary.get("window_count"),
            "p1324_third_target": p1324_summary.get("target"),
            "p1324_third_target_strict_slice_gate": p1324_summary.get(
                "p1323_strict_slice_quadratic_gate_passed"
            ),
        },
        "full_recall_heldout_evaluated_factor_proxy_below_rho": {
            "pass": heldout_evaluated_pass,
            "best_aggregate_full_recall": compact_route(
                aggregate_evaluated, evaluated_ratio_field
            ),
            "best_every_split_positive_full_recall": compact_route(
                robust_evaluated, evaluated_ratio_field
            ),
            "aggregate_proxy_gap": proxy_gap(aggregate_evaluated),
            "proxy_boundary": (
                "Evaluated-factor counts assume an already available factor stream "
                "and omit full Sage factorization."
            ),
        },
        "full_recall_heldout_actual_selector_charge_below_rho": {
            "pass": heldout_selector_pass,
            "best_aggregate_full_recall": compact_route(
                aggregate_selector, selector_ratio_field
            ),
            "best_every_split_positive_full_recall": compact_route(
                robust_selector, selector_ratio_field
            ),
        },
        "complete_end_to_end_source_cost": {
            "pass": complete_source_cost_pass,
            "full_sage_factorization_cost_charged": full_factorization_cost_charged,
            "all_rejected_candidates_charged": all_rejections_charged,
            "charge_proxy_note": cost_note,
            "prefactor_method": prefactor.get("method"),
        },
        "fresh_independent_fixed_sum_rows_and_rank_delta": {
            "pass": fresh_row_pass,
            "new_independent_fixed_sum_row_count": new_rows,
            "fresh_rank_delta": fresh_rank_delta,
            "p1324_exact_route_join_gate_passed": p1324_summary[
                "exact_route_join_gate_passed"
            ],
            "p1324_target_independent_gate_passed": p1324_summary[
                "additional_or_target_independent_gate_passed"
            ],
            "r68_new_fixed_sum_rows_required": r68["result"][
                "pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows"
            ],
        },
    }
    failed = [name for name, gate in obligations.items() if not gate["pass"]]
    lane_admitted = not failed
    return {
        "classification": (
            "PRESURFACE_PUBLIC_SOURCE_LANE_ADMITTED"
            if lane_admitted
            else "PRESURFACE_PUBLIC_COMPONENT_ONLY"
        ),
        "obligations": obligations,
        "admission": {
            "lane_admitted": lane_admitted,
            "failed_obligations": failed,
            "passed_obligation_count": len(obligations) - len(failed),
            "obligation_count": len(obligations),
        },
        "negative_controls": {
            "public_pruning_status": pruning_summary["status"],
            "best_complete_recall_pruning_family": pruning_summary[
                "best_complete_recall_family"
            ],
            "best_pruning_recall": pruning_summary["best_recall_family"]["summary"][
                "row_leaf_recall"
            ],
            "screen_holdout_status": screen_summary["status"],
            "prefactor_status": prefactor_summary["status"],
        },
        "historical_two_operation_proxy_near_miss": historical_near_miss(
            inputs["historical_prefactor"]
        ),
        "claim_boundary": {
            "algorithm_breakthrough": False,
            "generic_prime_field_speedup": False,
            "shoup_bound_improvement": False,
            "component_below_rho_is_full_attack": False,
            "aggregate_proxy_crossing_is_speedup": False,
        },
        "fatal_obstruction": (
            "The public proposal generator and full backfill are valid component "
            "results, but the held-out factor proxy remains above rho, actual "
            "selector accounting is substantially above rho, full factorization is "
            "not charged, only two target families are represented, and no fresh "
            "independent fixed-sum rank delta is recorded."
        ),
        "next_action": (
            "Freeze one label-free algebraic factor-stage prefilter and its exact "
            "operation ledger on the existing two-target development corpus, then "
            "materialize it unchanged on at least two new generic-prime target "
            "families; require every split positive, 100% strict recall, full Sage "
            "factorization plus rejection charges below rho, and a positive fresh "
            "fixed-sum rank delta."
        ),
    }


def build_payload(
    paths: dict[str, Path],
    code_paths: dict[str, Path],
) -> dict[str, Any]:
    inputs = {name: read_json(path) for name, path in paths.items()}
    result = evaluate(inputs)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "public pre-surface FFE factor source with full charge",
        "source_bindings": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "schema": inputs[name]["schema"],
                "claim_status": inputs[name].get("claim_status"),
            }
            for name, path in sorted(paths.items())
        },
        "code_bindings": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(code_paths.items())
        },
        **result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in DEFAULT_PATHS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=path)
    for name, path in DEFAULT_CODE_PATHS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {name: getattr(args, name) for name in DEFAULT_PATHS}
    code_paths = {name: getattr(args, name) for name in DEFAULT_CODE_PATHS}
    payload = build_payload(paths, code_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    admission = payload["admission"]
    proxy = payload["obligations"][
        "full_recall_heldout_evaluated_factor_proxy_below_rho"
    ]["best_aggregate_full_recall"]
    print(
        f"output={args.output} admitted={admission['lane_admitted']} "
        f"passed={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} best_full_recall_proxy_rho="
        f"{proxy['ratio'] if proxy else None}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
