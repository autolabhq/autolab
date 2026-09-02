#!/usr/bin/env python3
"""Run exact legacy P1436 collision records at several curve sizes and fit cost."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import p1436_autoresearch_focus_harness as harness
import p1436_collision_record_emitter as emitter


SCHEMA = "ecdlp.p1436_collision_cost_scaling_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    WORKTREE_ROOT
    / "ecdlp_index_calculus_state/p1436_collision_cost_scaling_probe_20_22_24bit.json"
)


def one_curve_args(args: argparse.Namespace, curve_index: int) -> SimpleNamespace:
    return SimpleNamespace(
        mounted_root=args.mounted_root,
        collector=args.collector,
        curve_index=curve_index,
        policy=args.policy,
        shift_family=args.shift_family,
        source_family=args.source_family,
        mask_denominator=args.mask_denominator,
        target_descents=args.target_descents,
        output=Path("/dev/null"),
    )


def charged_cost_point(curve: dict[str, Any], policy: str) -> dict[str, Any]:
    cell = curve["policies"][policy]["full"]
    config_name, config = next(iter(cell["configurations"].items()))
    descents = cell["target_descents"]
    descent_costs = [
        int(row["total_field_operation_estimate"]) for row in descents
    ]
    preprocessing = int(config["total_field_operation_estimate"])
    phase_field_operations = {
        phase: sum(
            int(value)
            for key, value in config[field].items()
            if key.startswith("field_")
        )
        for phase, field in (
            ("setup", "setup_affine_field_operation_estimate"),
            ("source_attempts", "attempt_affine_field_operation_estimate"),
            ("factor_log_verification", "verify_affine_field_operation_estimate"),
        )
    }
    phase_field_operations["matrix"] = int(config["matrix_operation_estimate"])
    if sum(phase_field_operations.values()) != preprocessing:
        raise RuntimeError("collector phase costs do not sum to preprocessing total")
    single_target_max = preprocessing + max(descent_costs, default=0)
    rho_group = int(config["rho_group_operations"])
    rho_field_baseline = 11 * rho_group
    return {
        "split": curve["split"],
        "bits": int(curve["bits"]),
        "seed": curve["seed"],
        "group_order": int(curve["order"]),
        "policy": policy,
        "configuration": config_name,
        "factor_base_size_B": int(cell["factor_base_size_B"]),
        "attempts": int(config["attempts"]),
        "collision_edges": int(config["collision_edge_count"]),
        "relation_rows": int(config["relation_row_count"]),
        "relation_rank": int(config["relation_rank"]),
        "unknown_factor_count": int(config["unknown_factor_count"]),
        "preprocessing_field_operations": preprocessing,
        "preprocessing_phase_field_operations": phase_field_operations,
        "preprocessing_phase_fractions": {
            phase: round(cost / preprocessing, 10)
            for phase, cost in phase_field_operations.items()
        },
        "target_descent_field_operations": descent_costs,
        "single_target_field_operations_max": single_target_max,
        "rho_group_operations": rho_group,
        "rho_field_baseline_11x": rho_field_baseline,
        "preprocessing_ratio_vs_rho": round(
            preprocessing / rho_field_baseline * 11,
            10,
        ),
        "single_target_ratio_vs_rho": round(
            single_target_max / rho_field_baseline * 11,
            10,
        ),
        "target_descent_success_count": sum(row["recovered"] for row in descents),
        "target_descent_count": len(descents),
        "exact": all(config["validation"].values()),
        "verified_factor_logs": (
            config["factor_logs_available"]
            and config["factor_log_verification_failures"] == 0
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    child_payloads = [
        emitter.build_payload(one_curve_args(args, curve_index))
        for curve_index in args.curve_indices
    ]
    curves = [
        child["curve_records"][0]
        for child in child_payloads
    ]
    charged_costs = [
        charged_cost_point(curve, args.policy)
        for curve in curves
    ]
    fit = harness._log_scale_cost_fit(
        [
            (
                math.log(point["group_order"]),
                math.log(point["single_target_field_operations_max"]),
            )
            for point in charged_costs
        ]
    )
    exponent = fit["exponent_in_group_order"] if fit else None
    residual_ok = harness._residual_tolerance_passed(fit)
    scaling_gate_passed = bool(
        fit
        and exponent is not None
        and exponent < 0.5
        and residual_ok
        and all(point["single_target_ratio_vs_rho"] < 1 for point in charged_costs)
    )
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "DIAGNOSTIC_SCALING_PROBE_NO_ALGORITHM_CLAIM",
        "curve_records": curves,
        "charged_costs": charged_costs,
        "scaling_fit": {
            "fit": fit,
            "exponent_threshold": 0.5,
            "residual_tolerance": harness.SHOUP_SCALING_RESIDUAL_TOLERANCE,
            "all_single_target_points_below_rho": all(
                point["single_target_ratio_vs_rho"] < 1
                for point in charged_costs
            ),
            "gate_passed": scaling_gate_passed,
        },
        "summary": {
            "curve_count": len(curves),
            "exact_curve_count": sum(point["exact"] for point in charged_costs),
            "verified_log_curve_count": sum(
                point["verified_factor_logs"] for point in charged_costs
            ),
            "complete_target_descent_curve_count": sum(
                point["target_descent_success_count"]
                == point["target_descent_count"]
                for point in charged_costs
            ),
            "large_prime_breakthrough": False,
            "algorithm_breakthrough": False,
            "scaling_gate_passed": scaling_gate_passed,
        },
        "provenance": {
            "child_provenance": [child["provenance"] for child in child_payloads],
            "curve_indices": args.curve_indices,
            "collision_record_schema": emitter.ablation.COLLISION_RECORD_SCHEMA,
            "cost_model": (
                "single-target cost = reusable factor-base preprocessing plus the "
                "maximum measured exact target-descent cost; rho field baseline uses "
                "the collector's explicit 11-field-operations-per-group-operation model"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mounted-root",
        type=Path,
        default=emitter.DEFAULT_MOUNTED_ROOT,
    )
    parser.add_argument("--collector", type=Path, default=emitter.DEFAULT_COLLECTOR)
    parser.add_argument("--curve-indices", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--policy", default="two_map_union")
    parser.add_argument(
        "--shift-family",
        choices=("random", "consecutive", "mixed"),
        default="random",
    )
    parser.add_argument(
        "--source-family",
        choices=("canonical", "hash", "balanced_stride"),
        default="hash",
    )
    parser.add_argument("--mask-denominator", type=int, default=1)
    parser.add_argument("--target-descents", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(set(args.curve_indices)) < 2:
        raise ValueError("at least two distinct curve indices are required")
    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fit = payload["scaling_fit"]["fit"]
    print(
        f"output={args.output} exact={payload['summary']['exact_curve_count']}/"
        f"{payload['summary']['curve_count']} exponent="
        f"{fit['exponent_in_group_order'] if fit else None} "
        f"gate={payload['scaling_fit']['gate_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
