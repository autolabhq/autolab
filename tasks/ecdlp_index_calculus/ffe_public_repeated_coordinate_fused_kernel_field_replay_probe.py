#!/usr/bin/env python3
"""Replay fused first-pass candidate points as pure finite-field instructions.

The affine trace probe rematerializes public worklist contexts and records the
field values used by generalized-Weierstrass addition.  This probe consumes
that trace and independently replays it with plain modular arithmetic only.  It
does not import the verifier or call curve helper code.

The output is an instruction-level contract for a native/FFE kernel: the same
operand registers, field inversions, multiplications, additions/subtractions,
and candidate-point outputs must be reproduced before second-pass relation
predicates run.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_AFFINE_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_affine_trace_target67_672_744.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_field_replay.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_field(raw: Any) -> int:
    match = re.fullmatch(r"GF\((\d+)\)", str(raw or ""))
    if match is None:
        raise ValueError(f"field must be GF(p), got {raw!r}")
    return int(match.group(1))


def point_key(raw: Any) -> tuple[int, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    return (as_int(raw[0]), as_int(raw[1]))


def point_json(point: tuple[int, int] | None) -> list[int] | None:
    if point is None:
        return None
    return [int(point[0]), int(point[1])]


def add_failure(failures: list[dict[str, Any]], code: str, detail: dict[str, Any]) -> None:
    failures.append({"code": code, **detail})


def is_on_curve(point: tuple[int, int] | None, ainvs: list[int], p: int) -> bool:
    if point is None:
        return True
    a1, a2, a3, a4, a6 = [value % p for value in ainvs]
    x, y = point
    lhs = (y * y + a1 * x * y + a3 * y) % p
    rhs = (x * x * x + a2 * x * x + a4 * x + a6) % p
    return lhs == rhs


def field_op(
    instructions: list[dict[str, Any]],
    counts: Counter[str],
    op: str,
    out: str,
    *args: Any,
    p: int,
) -> int:
    if op == "add":
        value = (as_int(args[0]) + as_int(args[1])) % p
        counts["add"] += 1
    elif op == "sub":
        value = (as_int(args[0]) - as_int(args[1])) % p
        counts["sub"] += 1
    elif op == "mul":
        value = (as_int(args[0]) * as_int(args[1])) % p
        counts["mul"] += 1
    elif op == "neg":
        value = (-as_int(args[0])) % p
        counts["neg"] += 1
    elif op == "inv":
        value = pow(as_int(args[0]) % p, -1, p)
        counts["inv"] += 1
    else:
        raise ValueError(f"unsupported field op {op!r}")
    instructions.append(
        {
            "op": op,
            "out": out,
            "args": [int(arg) if isinstance(arg, int) else arg for arg in args],
            "value": int(value),
        }
    )
    return value


def replay_field_trace(trace: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    instructions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    p = parse_field(trace.get("field"))
    ainvs = [as_int(value) % p for value in trace.get("ainvs_mod_p") or []]
    if len(ainvs) != 5:
        add_failure(failures, "invalid_ainvs", {"ainvs_mod_p": trace.get("ainvs_mod_p")})
        return {
            "kernel_trace_status": "field_replay_failed_check",
            "field": trace.get("field"),
            "operation": trace.get("operation"),
            "result": None,
            "instruction_stream": instructions,
            "field_op_counts": dict(counts),
            "checks": {},
            "failures": failures,
        }
    a1, a2, a3, a4, a6 = ainvs
    operation = str(trace.get("operation") or "")
    left = point_key(trace.get("left"))
    right = point_key(trace.get("right"))
    recorded_result = point_key(trace.get("result"))

    if left is None and right is None:
        add_failure(failures, "missing_operands", {"left": trace.get("left"), "right": trace.get("right")})
        result = None
    elif operation == "identity_left":
        result = right
    elif operation == "identity_right":
        result = left
    elif operation in {"vertical_inverse", "vertical_tangent"}:
        result = None
    elif operation in {"addition", "doubling"} and left is not None and right is not None:
        x1, y1 = left
        x2, y2 = right
        if operation == "addition":
            slope_numerator = field_op(instructions, counts, "sub", "slope_numerator", y2, y1, p=p)
            slope_denominator = field_op(instructions, counts, "sub", "slope_denominator", x2, x1, p=p)
        else:
            x1_sq = field_op(instructions, counts, "mul", "x1_sq", x1, x1, p=p)
            three_x1_sq = field_op(instructions, counts, "mul", "three_x1_sq", 3, x1_sq, p=p)
            two_a2_x1 = field_op(instructions, counts, "mul", "two_a2_x1", (2 * a2) % p, x1, p=p)
            numerator_tmp = field_op(instructions, counts, "add", "numerator_tmp", three_x1_sq, two_a2_x1, p=p)
            numerator_tmp = field_op(instructions, counts, "add", "numerator_tmp_plus_a4", numerator_tmp, a4, p=p)
            a1_y1 = field_op(instructions, counts, "mul", "a1_y1", a1, y1, p=p)
            slope_numerator = field_op(instructions, counts, "sub", "slope_numerator", numerator_tmp, a1_y1, p=p)
            two_y1 = field_op(instructions, counts, "mul", "two_y1", 2, y1, p=p)
            a1_x1 = field_op(instructions, counts, "mul", "a1_x1", a1, x1, p=p)
            denominator_tmp = field_op(instructions, counts, "add", "denominator_tmp", two_y1, a1_x1, p=p)
            slope_denominator = field_op(instructions, counts, "add", "slope_denominator", denominator_tmp, a3, p=p)

        if slope_denominator % p == 0:
            add_failure(failures, "zero_slope_denominator", {"operation": operation})
            result = None
        else:
            slope_denominator_inverse = field_op(
                instructions,
                counts,
                "inv",
                "slope_denominator_inverse",
                slope_denominator,
                p=p,
            )
            slope = field_op(
                instructions,
                counts,
                "mul",
                "slope",
                slope_numerator,
                slope_denominator_inverse,
                p=p,
            )
            slope_x1 = field_op(instructions, counts, "mul", "slope_x1", slope, x1, p=p)
            intercept = field_op(instructions, counts, "sub", "intercept", y1, slope_x1, p=p)
            slope_sq = field_op(instructions, counts, "mul", "slope_sq", slope, slope, p=p)
            a1_slope = field_op(instructions, counts, "mul", "a1_slope", a1, slope, p=p)
            x3_tmp = field_op(instructions, counts, "add", "x3_tmp0", slope_sq, a1_slope, p=p)
            x3_tmp = field_op(instructions, counts, "sub", "x3_tmp1", x3_tmp, a2, p=p)
            x3_tmp = field_op(instructions, counts, "sub", "x3_tmp2", x3_tmp, x1, p=p)
            x3 = field_op(instructions, counts, "sub", "x3", x3_tmp, x2, p=p)
            slope_plus_a1 = field_op(instructions, counts, "add", "slope_plus_a1", slope, a1, p=p)
            y3_tmp = field_op(instructions, counts, "mul", "y3_tmp0", slope_plus_a1, x3, p=p)
            y3_tmp = field_op(instructions, counts, "neg", "y3_tmp1", y3_tmp, p=p)
            y3_tmp = field_op(instructions, counts, "sub", "y3_tmp2", y3_tmp, intercept, p=p)
            y3 = field_op(instructions, counts, "sub", "y3", y3_tmp, a3, p=p)
            result = (x3, y3)

            recorded_fields = {
                "slope_numerator": slope_numerator,
                "slope_denominator": slope_denominator,
                "slope_denominator_inverse": slope_denominator_inverse,
                "slope": slope,
                "intercept": intercept,
            }
            for field, expected in recorded_fields.items():
                if trace.get(field) is not None and as_int(trace.get(field), -1) % p != expected:
                    add_failure(
                        failures,
                        "recorded_intermediate_mismatch",
                        {"field": field, "recorded": trace.get(field), "replayed": expected},
                    )
    else:
        add_failure(failures, "unsupported_operation", {"operation": operation})
        result = None

    result_json = point_json(result)
    recorded_result_json = point_json(recorded_result)
    if result_json != recorded_result_json:
        add_failure(
            failures,
            "recorded_result_mismatch",
            {"recorded_result": recorded_result_json, "replayed_result": result_json},
        )

    checks = {
        "left_on_curve": is_on_curve(left, ainvs, p),
        "right_on_curve": is_on_curve(right, ainvs, p),
        "result_on_curve": is_on_curve(result, ainvs, p),
        "matches_recorded_result": result_json == recorded_result_json,
        "recorded_intermediates_match": not any(
            failure["code"] == "recorded_intermediate_mismatch" for failure in failures
        ),
        "uses_verifier_result": False,
    }
    for key in ("left_on_curve", "right_on_curve", "result_on_curve"):
        if not checks[key]:
            add_failure(failures, "curve_membership_check_failed", {"check": key})

    return {
        "kernel_trace_status": "field_replay_verified" if not failures else "field_replay_failed_check",
        "field": trace.get("field"),
        "operation": operation,
        "ainvs_mod_p": ainvs,
        "result": result_json,
        "recorded_result": recorded_result_json,
        "instruction_stream": instructions,
        "field_op_counts": {
            "add": int(counts["add"]),
            "sub": int(counts["sub"]),
            "mul": int(counts["mul"]),
            "neg": int(counts["neg"]),
            "inv": int(counts["inv"]),
            "total": int(sum(counts.values())),
        },
        "checks": checks,
        "failures": failures,
    }


def combine_counts(traces: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for trace in traces:
        for key, value in (trace.get("field_op_counts") or {}).items():
            counts[key] += as_int(value)
    return {key: int(counts[key]) for key in ("add", "sub", "mul", "neg", "inv", "total")}


def validate_group(group: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    representative = replay_field_trace(group.get("representative_affine_trace") or {})
    row_replays = []
    for row in group.get("row_instance_traces") or []:
        row_replay = replay_field_trace(row.get("affine_trace") or {})
        row_replays.append(
            {
                "row_key": row.get("row_key"),
                "scheduled_trial": as_int(row.get("scheduled_trial")),
                "candidate_pos": as_int(row.get("candidate_pos")),
                "scout_pos": as_int(row.get("scout_pos")),
                "original_trial": as_int(row.get("original_trial")),
                "result": row_replay.get("result"),
                "kernel_trace_status": row_replay.get("kernel_trace_status"),
                "field_op_counts": row_replay.get("field_op_counts"),
                "checks": row_replay.get("checks"),
                "failures": row_replay.get("failures"),
                "instruction_stream": row_replay.get("instruction_stream"),
            }
        )

    abi_point = point_key(group.get("abi_candidate_point"))
    representative_point = point_key(group.get("representative_candidate_point"))
    representative_result = point_key(representative.get("result"))
    row_results = [point_key(row.get("result")) for row in row_replays]
    local_consistent = all(result == representative_result for result in row_results)
    matches_abi = representative_result == abi_point
    matches_representative = representative_result == representative_point

    if representative.get("kernel_trace_status") != "field_replay_verified":
        add_failure(
            failures,
            "representative_field_replay_failed",
            {"event_key": group.get("event_key"), "failures": representative.get("failures")},
        )
    for row in row_replays:
        if row.get("kernel_trace_status") != "field_replay_verified":
            add_failure(
                failures,
                "row_field_replay_failed",
                {
                    "event_key": group.get("event_key"),
                    "row_key": row.get("row_key"),
                    "failures": row.get("failures"),
                },
            )
    if not local_consistent:
        add_failure(
            failures,
            "row_replay_outputs_not_consistent",
            {"event_key": group.get("event_key"), "row_results": [point_json(result) for result in row_results]},
        )
    if not matches_abi:
        add_failure(
            failures,
            "representative_result_not_abi_point",
            {
                "event_key": group.get("event_key"),
                "replayed_result": point_json(representative_result),
                "abi_candidate_point": point_json(abi_point),
            },
        )
    if not matches_representative:
        add_failure(
            failures,
            "representative_result_not_recorded_candidate",
            {
                "event_key": group.get("event_key"),
                "replayed_result": point_json(representative_result),
                "representative_candidate_point": point_json(representative_point),
            },
        )

    return {
        "event_key": group.get("event_key"),
        "fanout": as_int(group.get("fanout")),
        "candidate_point_reused": bool(group.get("candidate_point_reused")),
        "representative_candidate_point": group.get("representative_candidate_point"),
        "abi_candidate_point": group.get("abi_candidate_point"),
        "replayed_candidate_point": representative.get("result"),
        "field_replay_group_status": "field_replay_group_verified" if not failures else "field_replay_group_failed_check",
        "representative_kernel_replay": representative,
        "row_kernel_replays": row_replays,
        "field_op_counts_representative": representative.get("field_op_counts"),
        "field_op_counts_row_instances": combine_counts(row_replays),
        "checks": {
            "representative_replay_verified": representative.get("kernel_trace_status") == "field_replay_verified",
            "row_replay_verified_count": sum(
                1 for row in row_replays if row.get("kernel_trace_status") == "field_replay_verified"
            ),
            "row_instance_count": len(row_replays),
            "row_outputs_consistent": local_consistent,
            "matches_abi_candidate_point": matches_abi,
            "matches_representative_candidate_point": matches_representative,
        },
        "failures": failures,
    }


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    groups = [validate_group(group) for group in record.get("trace_groups") or []]
    checks = record.get("checks") or {}
    verified_groups = [group for group in groups if group.get("field_replay_group_status") == "field_replay_group_verified"]
    representative_counts = combine_counts(
        [group.get("representative_kernel_replay") or {} for group in groups]
    )
    row_counts = combine_counts(
        [
            row
            for group in groups
            for row in group.get("row_kernel_replays") or []
        ]
    )

    if len(verified_groups) != len(groups):
        add_failure(
            failures,
            "group_field_replay_failures",
            {
                "group_count": len(groups),
                "verified_group_count": len(verified_groups),
            },
        )
    if as_int(checks.get("first_pass_group_count"), -1) != len(groups):
        add_failure(
            failures,
            "source_first_pass_group_count_mismatch",
            {"source_count": checks.get("first_pass_group_count"), "field_replay_count": len(groups)},
        )
    row_replay_count = sum(as_int(group.get("checks", {}).get("row_instance_count")) for group in groups)
    if as_int(checks.get("row_instance_trace_count"), -1) != row_replay_count:
        add_failure(
            failures,
            "source_row_instance_count_mismatch",
            {"source_count": checks.get("row_instance_trace_count"), "field_replay_count": row_replay_count},
        )

    return {
        "source_name": record.get("source_name"),
        "public_group_key": record.get("public_group_key"),
        "field_replay_status": "fused_kernel_field_replay_verified" if not failures else "fused_kernel_field_replay_failed_check",
        "source_affine_trace_status": record.get("affine_trace_status"),
        "trace_groups": groups,
        "checks": {
            "first_pass_group_count": len(groups),
            "field_replay_group_verified_count": len(verified_groups),
            "row_instance_replay_count": row_replay_count,
            "all_groups_verified": len(verified_groups) == len(groups),
            "all_representative_outputs_match_abi": all(
                bool((group.get("checks") or {}).get("matches_abi_candidate_point")) for group in groups
            ),
            "all_row_outputs_consistent": all(
                bool((group.get("checks") or {}).get("row_outputs_consistent")) for group in groups
            ),
            "matches_source_first_pass_group_count": as_int(checks.get("first_pass_group_count"), -1) == len(groups),
            "matches_source_row_instance_count": as_int(checks.get("row_instance_trace_count"), -1) == row_replay_count,
            "matches_contract_below_rho": bool(checks.get("matches_contract_below_rho")),
            "matches_contract_event_reuse_target": bool(checks.get("matches_contract_event_reuse_target")),
        },
        "field_op_counts_first_pass_representatives": representative_counts,
        "field_op_counts_all_row_instances": row_counts,
        "failures": failures,
        "implementation_boundary": (
            "Verifier-independent finite-field replay of the first-pass fused "
            "candidate-point instruction stream.  A native/FFE kernel can be "
            "compared against these register outputs directly."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if record.get("field_replay_status") == "fused_kernel_field_replay_verified"]
    below = [
        record for record in verified if bool((record.get("checks") or {}).get("matches_contract_below_rho"))
    ]
    event_reuse = [
        record for record in below if bool((record.get("checks") or {}).get("matches_contract_event_reuse_target"))
    ]
    first_pass_counts = combine_counts(
        [
            {"field_op_counts": record.get("field_op_counts_first_pass_representatives") or {}}
            for record in records
        ]
    )
    row_counts = combine_counts(
        [
            {"field_op_counts": record.get("field_op_counts_all_row_instances") or {}}
            for record in records
        ]
    )
    return {
        "record_count": len(records),
        "field_replay_verified_count": len(verified),
        "below_rho_field_replay_verified_count": len(below),
        "event_reuse_below_rho_field_replay_verified_count": len(event_reuse),
        "total_first_pass_groups": sum(as_int((record.get("checks") or {}).get("first_pass_group_count")) for record in records),
        "total_row_instance_replays": sum(as_int((record.get("checks") or {}).get("row_instance_replay_count")) for record in records),
        "first_pass_representative_field_op_counts": first_pass_counts,
        "all_row_instance_field_op_counts": row_counts,
        "interpretation": (
            "The first-pass fused candidate-point layer now replays as a pure "
            "finite-field instruction stream with no verifier arithmetic imports."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affine-source", type=Path, default=DEFAULT_AFFINE_SOURCE)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = load_json(args.affine_source)
    wanted = set(args.source_name or [])
    records = []
    for record in source.get("records") or []:
        if not isinstance(record, dict):
            continue
        source_name = str(record.get("source_name") or "")
        if wanted and source_name not in wanted:
            continue
        records.append(validate_record(record))

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_field_replay_probe_v1",
        "method": "verifier_independent_finite_field_instruction_replay_for_first_pass_candidate_points",
        "parameters": {
            "affine_source": str(args.affine_source),
            "source_names": sorted(wanted),
            "verifier_dependency": "none",
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                record.get("field_replay_status") != "fused_kernel_field_replay_verified",
                not bool((record.get("checks") or {}).get("matches_contract_event_reuse_target")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This is a finite-field replay of first-pass candidate-point arithmetic, not a new relation search.",
            "It consumes an existing affine trace and does not yet replace worklist rematerialization.",
            "The second-pass relation predicates and modular linear algebra remain checked by the ABI/contract layers.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
