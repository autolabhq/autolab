#!/usr/bin/env python3
"""Emit exact collision records from the legacy P1436 collector without editing it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import p1436_collision_to_rank_routing_ablation as ablation


SCHEMA = "ecdlp.p1436_collision_record_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOUNTED_ROOT = Path("/Volumes/Volume/autolab")
DEFAULT_COLLECTOR = (
    DEFAULT_MOUNTED_ROOT
    / "tasks/ecdlp_index_calculus/p1436_large_prime_residual_collision_collector.py"
)
DEFAULT_OUTPUT = (
    WORKTREE_ROOT
    / "ecdlp_index_calculus_state/p1436_collision_record_probe_from_legacy_collector.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def load_collector(path: Path) -> ModuleType:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"collector not found: {path}")
    sys.dont_write_bytecode = True
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("p1436_legacy_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import collector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collision_admission_line(module: ModuleType) -> int:
    source, start = inspect.getsourcelines(module.run_collector)
    matches = [
        start + offset
        for offset, line in enumerate(source)
        if line.strip() == "if not any(unknown_row):"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "legacy collector admission boundary changed; refusing an ambiguous trace"
        )
    return matches[0]


def run_with_collision_trace(
    module: ModuleType,
    *args: Any,
    **kwargs: Any,
) -> tuple[list[int] | None, dict[str, Any]]:
    code = module.run_collector.__code__
    admission_line = collision_admission_line(module)
    records: list[dict[str, Any]] = []
    prior_trace = sys.gettrace()

    def trace(frame: Any, event_name: str, _arg: Any):
        if (
            frame.f_code is code
            and event_name == "line"
            and frame.f_lineno == admission_line
        ):
            local = frame.f_locals
            coefficients = [int(value) for value in local["unknown_row"]]
            edge_material = {
                "collision_index": len(records),
                "attempt": int(local["attempt"]),
                "left_shift": int(local["first"]["shift_index"]),
                "right_shift": int(local["event"]["shift_index"]),
                "left_source": list(local["first"]["source"]),
                "right_source": list(local["event"]["source"]),
                "residual": local["event"]["residual"],
            }
            records.append(
                {
                    "schema": ablation.COLLISION_RECORD_SCHEMA,
                    "edge_id": canonical_hash(edge_material),
                    "left_shift": edge_material["left_shift"],
                    "right_shift": edge_material["right_shift"],
                    "relation_coefficients": coefficients,
                    "relation_rhs": int(local["anchored_rhs"]),
                    "relation_admitted": any(coefficients),
                    "source_equation_exact": None,
                    "residual_equality_exact": (
                        local["first"]["residual"] == local["event"]["residual"]
                    ),
                }
            )
        return trace

    sys.settrace(trace)
    try:
        factor_logs, summary = module.run_collector(*args, **kwargs)
    finally:
        sys.settrace(prior_trace)

    if len(records) != int(summary["collision_edge_count"]):
        raise RuntimeError(
            "trace count does not match collector collision_edge_count: "
            f"{len(records)} != {summary['collision_edge_count']}"
        )
    sources_exact = bool(summary["validation"]["sources_exact"])
    for record in records:
        record["source_equation_exact"] = sources_exact
    summary["collision_records"] = records
    return factor_logs, summary


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    mounted_root = args.mounted_root.resolve()
    collector_path = args.collector.resolve()
    source_path = (
        mounted_root
        / "ecdlp_index_calculus_state/p1432_high_energy_factor_bases_after_p1431_probe.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_curve = source["curve_records"][args.curve_index]
    if args.policy not in source_curve["policies"]:
        raise KeyError(f"policy not present in source curve: {args.policy}")

    module = load_collector(collector_path)
    p = module.int_value(source_curve["p"])
    order = module.int_value(source_curve["order"])
    ainvs = [
        0,
        0,
        0,
        module.int_value(source_curve["curve_a"]),
        module.int_value(source_curve["curve_b"]),
    ]
    generator = tuple(source_curve["generator_point"])
    factor_base_size = module.int_value(source_curve["factor_base_size_B"])
    factors = module.anchored_factor_base(
        generator,
        [
            tuple(point)
            for point in source_curve["policies"][args.policy]["factor_base_points"]
        ],
        factor_base_size,
    )
    raw = module.p1398.relation_probe.load_verifier_module()
    label = (
        f"p1436-collision-record:{source_curve['seed']}:{args.policy}:full:"
        f"{args.shift_family}:{args.source_family}:mask{args.mask_denominator}"
    )
    factor_logs, config = run_with_collision_trace(
        module,
        raw,
        factors,
        generator,
        ainvs,
        p,
        order,
        module.int_value(source_curve["seed"]),
        args.shift_family,
        args.source_family,
        args.mask_denominator,
        label,
    )

    validation = ablation.validate_collision_source(config, modulus=order)
    if not validation["exact"]:
        raise RuntimeError(
            "emitted collision records fail the consumer ABI: "
            + ", ".join(validation["errors"])
        )

    target_descents: list[dict[str, Any]] = []
    target_schedule_sha256 = ""
    if factor_logs is not None and args.target_descents:
        schedule_verifier = module.p1408.OperationCountingVerifier(raw)
        target_schedule = module.p1433.generate_schedule(
            schedule_verifier,
            generator,
            ainvs,
            p,
            order,
            module.int_value(source_curve["seed"]),
            "target",
            4096,
        )
        target_schedule_sha256 = target_schedule[2]
        for target_index in range(args.target_descents):
            descent = module.target_descent(
                raw,
                factors,
                factor_logs,
                generator,
                target_schedule[1][target_index],
                target_schedule[0][target_index],
                ainvs,
                p,
                order,
                module.int_value(source_curve["seed"]),
                target_index,
            )
            descent["setup_affine_field_operation_estimate"] = (
                module.p1408.affine_field_operation_estimate(
                    descent["setup_point_operation_profile"]
                )
            )
            descent["attempt_affine_field_operation_estimate"] = (
                module.p1408.affine_field_operation_estimate(
                    descent["attempt_point_operation_profile"]
                )
            )
            descent["total_field_operation_estimate"] = sum(
                value
                for phase in (
                    descent["setup_affine_field_operation_estimate"],
                    descent["attempt_affine_field_operation_estimate"],
                )
                for key, value in phase.items()
                if key.startswith("field_")
            )
            descent["total_field_ratio_vs_11x_rho"] = module.ratio(
                descent["total_field_operation_estimate"],
                11 * descent["rho_group_operations"],
            )
            target_descents.append(descent)

    config_name = (
        f"{args.shift_family}_{args.source_family}_mask{args.mask_denominator}"
    )
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "EXACT_COLLISION_RECORDS_EMITTED_NO_ALGORITHM_CLAIM",
        "curve_records": [
            {
                **{
                    name: source_curve[name]
                    for name in (
                        "split",
                        "bits",
                        "seed",
                        "p",
                        "curve_a",
                        "curve_b",
                        "order",
                        "generator_point",
                    )
                },
                "policies": {
                    args.policy: {
                        "full": {
                            "factor_base_size_B": factor_base_size,
                            "configurations": {config_name: config},
                            "target_descents": target_descents,
                            "factor_logs_available": factor_logs is not None,
                        }
                    }
                },
            }
        ],
        "summary": {
            "curve_count": 1,
            "configuration_count": 1,
            "collision_record_count": len(config["collision_records"]),
            "relation_row_count": config["relation_row_count"],
            "relation_rank": config["relation_rank"],
            "augmented_rank": config["augmented_rank"],
            "target_descent_count": len(target_descents),
            "target_descent_success_count": sum(
                row["recovered"] for row in target_descents
            ),
            "target_descent_invalid_count": sum(
                row["invalid_candidate_count"] for row in target_descents
            ),
            "preprocessing_field_operation_estimate": config[
                "total_field_operation_estimate"
            ],
            "single_target_total_field_operation_estimate_max": (
                config["total_field_operation_estimate"]
                + max(
                    (
                        row["total_field_operation_estimate"]
                        for row in target_descents
                    ),
                    default=0,
                )
            ),
            "consumer_abi_exact": validation["exact"],
            "consumer_compiled_summary": validation["compiled_all_edge_summary"],
            "algorithm_breakthrough": False,
        },
        "provenance": {
            "collector_path": str(collector_path),
            "collector_sha256": sha256_file(collector_path),
            "collector_admission_line": collision_admission_line(module),
            "p1432_path": str(source_path),
            "p1432_sha256": sha256_file(source_path),
            "target_schedule_sha256": target_schedule_sha256,
            "emitter_path": str(Path(__file__).resolve()),
            "emitter_sha256": sha256_file(Path(__file__).resolve()),
            "collision_record_schema": ablation.COLLISION_RECORD_SCHEMA,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNTED_ROOT)
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--curve-index", type=int, default=0)
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
    if args.curve_index < 0:
        raise ValueError("curve-index must be nonnegative")
    if args.mask_denominator <= 0:
        raise ValueError("mask-denominator must be positive")
    if args.target_descents < 0 or args.target_descents > 32:
        raise ValueError("target-descents must be between 0 and 32")
    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = payload["summary"]
    print(
        f"output={args.output} collisions={summary['collision_record_count']} "
        f"rows={summary['relation_row_count']} rank={summary['relation_rank']} "
        f"augmented_rank={summary['augmented_rank']} "
        f"descents={summary['target_descent_success_count']}/"
        f"{summary['target_descent_count']} "
        f"abi_exact={summary['consumer_abi_exact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
