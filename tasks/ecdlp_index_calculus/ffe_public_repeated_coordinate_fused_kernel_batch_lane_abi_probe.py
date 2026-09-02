#!/usr/bin/env python3
"""Verify ABI and curve invariants for grouped batch-lane kernels.

The native batch-lane probe collapses repeated-coordinate candidate-point
arithmetic by shared denominator and left y-coordinate.  This guard promotes
that low-level shape back to the verifier boundary: each source-local lane must
emit a finite base-subgroup curve point, all event cases in that lane must
share the same affine registers, and the lane-amortized contract must still
preserve the original second-pass relation checks and derived secret.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

import relation_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_AFFINE_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_affine_trace_target67_672_744.json"
)
DEFAULT_BATCH_LANE_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_batch_lane_target67_672_744.json"
)
DEFAULT_LANE_CONTRACT_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_batch_lane_contract_target67_672_744.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_batch_lane_abi.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_target(raw: Any) -> tuple[str, int]:
    label, sep, prime = str(raw or "").partition("@")
    if not sep or not label:
        raise ValueError(f"target must be label@prime, got {raw!r}")
    return label, int(prime)


def parse_field(raw: Any) -> int:
    match = re.fullmatch(r"GF\((\d+)\)", str(raw or ""))
    if match is None:
        raise ValueError(f"field must be GF(p), got {raw!r}")
    return int(match.group(1))


def compact_event_key(raw: Any) -> tuple[str, int, int]:
    values = list(raw or [])
    if len(values) != 3:
        return ("", -1, -1)
    return (str(values[0]), as_int(values[1], -1), as_int(values[2], -1))


def compact_event_key_json(raw: Any) -> list[Any]:
    leaf, scout_pos, original_trial = compact_event_key(raw)
    return [leaf, scout_pos, original_trial]


def point_key(raw: Any) -> tuple[int, int] | None:
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    return (as_int(raw[0]), as_int(raw[1]))


def point_pair(raw: Any) -> tuple[int, int]:
    parsed = point_key(raw)
    if parsed is None:
        raise ValueError(f"point must be [x,y], got {raw!r}")
    return parsed


def sorted_point_json(points: set[tuple[int, int]]) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in sorted(points)]


def add_failure(failures: list[dict[str, Any]], code: str, detail: dict[str, Any] | None = None) -> None:
    failures.append({"code": code, **(detail or {})})


def load_target_contexts(verifier: Any, targets: set[str]) -> dict[str, dict[str, Any]]:
    records = verifier.load_records()
    by_label = {str(record["label"]): record for record in records}
    contexts: dict[str, dict[str, Any]] = {}
    for target in sorted(targets):
        label, p = parse_target(target)
        record = by_label.get(label)
        if record is None:
            raise KeyError(f"no verifier record for {label!r}")
        inv = verifier.reduction_invariants(record, p)
        base_order = int(inv["base_order"])
        contexts[target] = {
            "label": label,
            "p": int(p),
            "ainvs": [int(value) for value in record["ainvs"]],
            "group_order": int(inv["order"]),
            "base_order": base_order,
            "base": verifier.point_to_json(inv["base"]),
            "generic_rho_steps": math.ceil(math.sqrt(math.pi * base_order / 2.0)),
        }
    return contexts


def candidate_point_curve_check(verifier: Any, ctx: dict[str, Any], point_json: Any) -> dict[str, Any]:
    p = int(ctx["p"])
    base_order = int(ctx["base_order"])
    parsed_key = point_key(point_json)
    parsed = parsed_key is not None
    finite_point = False
    coords_in_field = False
    on_curve = False
    in_base_order_subgroup = False
    if parsed_key is not None:
        x, y = parsed_key
        coords_in_field = 0 <= x < p and 0 <= y < p
        point = verifier.point_from_json([x, y])
        finite_point = point is not verifier.POINT_AT_INFINITY
        on_curve = bool(verifier.is_on_curve(point, ctx["ainvs"], p))
        in_base_order_subgroup = bool(
            verifier.mul_point(base_order, point, ctx["ainvs"], p) is verifier.POINT_AT_INFINITY
        )
    return {
        "point": point_json,
        "parsed": parsed,
        "finite_point": finite_point,
        "coords_in_field": coords_in_field,
        "on_curve": on_curve,
        "in_base_order_subgroup": in_base_order_subgroup,
    }


def records_by_source(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("source_name") or ""): record
        for record in source.get("records") or []
        if isinstance(record, dict)
    }


def trace_register_key(trace: dict[str, Any]) -> tuple[int, int, int, int, int]:
    result_x, result_y = point_pair(trace.get("result"))
    return (
        as_int(trace.get("slope_numerator"), -1),
        as_int(trace.get("slope"), -1),
        as_int(trace.get("intercept"), -1),
        result_x,
        result_y,
    )


def collect_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for record in records:
        source_name = str(record.get("source_name") or "")
        public_group = record.get("public_group_key") or {}
        target = str(public_group.get("target") or "")
        for group_index, group in enumerate(record.get("trace_groups") or []):
            if not isinstance(group, dict):
                continue
            trace = group.get("representative_affine_trace") or {}
            left_x, left_y = point_pair(trace.get("left"))
            right_x, right_y = point_pair(trace.get("right"))
            result_x, result_y = point_pair(trace.get("result"))
            ainvs = [as_int(value) for value in trace.get("ainvs_mod_p") or []]
            if len(ainvs) != 5:
                raise ValueError(f"expected five curve coefficients, got {ainvs!r}")
            row_instance_traces = [
                item for item in group.get("row_instance_traces") or [] if isinstance(item, dict)
            ]
            cases.append(
                {
                    "source_name": source_name,
                    "target": target,
                    "trace_group_index": group_index,
                    "event_key": compact_event_key(group.get("event_key")),
                    "event_key_json": compact_event_key_json(group.get("event_key")),
                    "fanout": as_int(group.get("fanout")),
                    "row_instance_trace_count": as_int(group.get("row_instance_trace_count")),
                    "observed_row_instance_trace_count": len(row_instance_traces),
                    "candidate_point_reused": bool(group.get("candidate_point_reused")),
                    "matches_abi_candidate_point": bool(group.get("matches_abi_candidate_point")),
                    "local_affine_outputs_consistent": bool(group.get("local_affine_outputs_consistent")),
                    "operation": str(trace.get("operation") or ""),
                    "matches_verifier_add": bool(trace.get("matches_verifier_add")),
                    "p": parse_field(trace.get("field")),
                    "ainvs": ainvs,
                    "left_x": left_x,
                    "left_y": left_y,
                    "right_x": right_x,
                    "right_y": right_y,
                    "expected_slope_numerator": as_int(trace.get("slope_numerator"), -1),
                    "expected_slope_denominator": as_int(trace.get("slope_denominator"), -1),
                    "expected_slope_denominator_inverse": as_int(
                        trace.get("slope_denominator_inverse"), -1
                    ),
                    "expected_slope": as_int(trace.get("slope"), -1),
                    "expected_intercept": as_int(trace.get("intercept"), -1),
                    "expected_x": result_x,
                    "expected_y": result_y,
                    "row_register_keys": [
                        trace_register_key(item.get("affine_trace") or {})
                        for item in row_instance_traces
                        if isinstance(item.get("affine_trace"), dict)
                    ],
                }
            )
    return cases


def group_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        as_int(case["p"]),
        tuple(as_int(value) for value in case["ainvs"]),
        as_int(case["left_x"]),
        as_int(case["right_x"]),
        as_int(case["right_y"]),
        as_int(case["expected_slope_denominator"]),
        as_int(case["expected_slope_denominator_inverse"]),
    )


def lane_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        as_int(case["left_y"]),
        as_int(case["expected_slope_numerator"]),
        as_int(case["expected_slope"]),
        as_int(case["expected_intercept"]),
        as_int(case["expected_x"]),
        as_int(case["expected_y"]),
    )


def left_y_lane_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (*group_key(case), as_int(case["left_y"]))


def build_global_layout(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_group[group_key(case)].append(case)

    groups = []
    lanes = []
    event_cases = []
    for group_index, key in enumerate(sorted(by_group, key=lambda item: (-len(by_group[item]), item))):
        group_cases = sorted(by_group[key], key=lambda item: (item["source_name"], item["event_key"]))
        first = group_cases[0]
        lane_start = len(lanes)
        by_lane: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for case in group_cases:
            by_lane[lane_key(case)].append(case)

        lane_index_by_key = {}
        for local_lane_index, lkey in enumerate(sorted(by_lane, key=lambda item: (-len(by_lane[item]), item))):
            lane_cases = by_lane[lkey]
            lane = lane_cases[0]
            lane_index = len(lanes)
            lane_index_by_key[lkey] = lane_index
            lanes.append(
                {
                    "lane_index": lane_index,
                    "group_index": group_index,
                    "left_y": lane["left_y"],
                    "expected_slope_numerator": lane["expected_slope_numerator"],
                    "expected_slope": lane["expected_slope"],
                    "expected_intercept": lane["expected_intercept"],
                    "expected_x": lane["expected_x"],
                    "expected_y": lane["expected_y"],
                    "case_count": len(lane_cases),
                    "row_instance_trace_count": sum(
                        as_int(case.get("row_instance_trace_count")) for case in lane_cases
                    ),
                    "local_lane_index": local_lane_index,
                    "event_keys": [case["event_key_json"] for case in lane_cases],
                    "source_names": sorted({case["source_name"] for case in lane_cases}),
                }
            )

        for case in group_cases:
            event_cases.append({**case, "group_index": group_index, "lane_index": lane_index_by_key[lane_key(case)]})

        groups.append(
            {
                "group_index": group_index,
                "p": first["p"],
                "ainvs": first["ainvs"],
                "left_x": first["left_x"],
                "right_x": first["right_x"],
                "right_y": first["right_y"],
                "expected_slope_denominator": first["expected_slope_denominator"],
                "expected_slope_denominator_inverse": first["expected_slope_denominator_inverse"],
                "lane_start": lane_start,
                "lane_count": len(by_lane),
                "case_count": len(group_cases),
            }
        )
    return groups, lanes, event_cases


def validate_global_batch_artifact(
    batch_source: dict[str, Any],
    groups: list[dict[str, Any]],
    lanes: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    summary = batch_source.get("summary") or {}
    shared_groups = batch_source.get("shared_groups") or []
    batch_lanes = batch_source.get("batch_lanes") or []

    checks = {
        "global_batch_lane_verified": bool(summary.get("global_batch_lane_verified")),
        "native_failure_count_zero": as_int(summary.get("native_failure_count"), -1) == 0,
        "summary_group_count_matches": as_int(summary.get("shared_group_count"), -1) == len(groups),
        "summary_lane_count_matches": as_int(summary.get("batch_lane_count"), -1) == len(lanes),
        "summary_case_count_matches": as_int(summary.get("native_case_count"), -1) == len(cases),
        "shared_group_shape_matches": len(shared_groups) == len(groups),
        "batch_lane_shape_matches": len(batch_lanes) == len(lanes),
    }

    for name, passed in checks.items():
        if not passed:
            add_failure(failures, f"global_{name}_failed")

    for expected, observed in zip(groups, shared_groups, strict=False):
        observed_key = {
            "group_index": as_int(observed.get("group_index"), -1),
            "p": as_int(observed.get("p"), -1),
            "left_x": as_int(observed.get("left_x"), -1),
            "right": point_key(observed.get("right")),
            "slope_denominator": as_int(observed.get("slope_denominator"), -1),
            "slope_denominator_inverse": as_int(observed.get("slope_denominator_inverse"), -1),
            "case_count": as_int(observed.get("case_count"), -1),
            "lane_count": as_int(observed.get("lane_count"), -1),
        }
        expected_key = {
            "group_index": as_int(expected.get("group_index"), -1),
            "p": as_int(expected.get("p"), -1),
            "left_x": as_int(expected.get("left_x"), -1),
            "right": (as_int(expected.get("right_x"), -1), as_int(expected.get("right_y"), -1)),
            "slope_denominator": as_int(expected.get("expected_slope_denominator"), -1),
            "slope_denominator_inverse": as_int(expected.get("expected_slope_denominator_inverse"), -1),
            "case_count": as_int(expected.get("case_count"), -1),
            "lane_count": as_int(expected.get("lane_count"), -1),
        }
        if observed_key != expected_key:
            add_failure(
                failures,
                "global_shared_group_shape_mismatch",
                {"expected": expected_key, "observed": observed_key},
            )

    for expected, observed in zip(lanes, batch_lanes, strict=False):
        observed_key = {
            "lane_index": as_int(observed.get("lane_index"), -1),
            "group_index": as_int(observed.get("group_index"), -1),
            "left_y": as_int(observed.get("left_y"), -1),
            "candidate_point": point_key(observed.get("candidate_point")),
            "case_count": as_int(observed.get("case_count"), -1),
        }
        expected_key = {
            "lane_index": as_int(expected.get("lane_index"), -1),
            "group_index": as_int(expected.get("group_index"), -1),
            "left_y": as_int(expected.get("left_y"), -1),
            "candidate_point": (as_int(expected.get("expected_x"), -1), as_int(expected.get("expected_y"), -1)),
            "case_count": as_int(expected.get("case_count"), -1),
        }
        if observed_key != expected_key:
            add_failure(
                failures,
                "global_batch_lane_shape_mismatch",
                {"expected": expected_key, "observed": observed_key},
            )

    checks["global_batch_lane_artifact_matches"] = not failures
    return checks, failures


def validate_source_record(
    verifier: Any,
    record: dict[str, Any],
    source_cases: list[dict[str, Any]],
    global_lanes_by_index: dict[int, dict[str, Any]],
    ctx: dict[str, Any],
    batch_record: dict[str, Any] | None,
    contract_record: dict[str, Any] | None,
    global_checks: dict[str, Any],
    global_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    source_name = str(record.get("source_name") or "")
    public_group = record.get("public_group_key") or {}

    if record.get("affine_trace_status") != "fused_kernel_affine_trace_verified":
        add_failure(
            failures,
            "affine_trace_status_not_verified",
            {"affine_trace_status": record.get("affine_trace_status")},
        )

    if batch_record is None:
        add_failure(failures, "missing_batch_lane_record")
    elif batch_record.get("batch_lane_status") != "batch_lane_native_kernel_verified":
        add_failure(
            failures,
            "batch_lane_status_not_verified",
            {"batch_lane_status": batch_record.get("batch_lane_status")},
        )

    if contract_record is None:
        add_failure(failures, "missing_lane_contract_record")
        contract_checks: dict[str, Any] = {}
        relation_replay: dict[str, Any] = {}
        work_counters: dict[str, Any] = {}
    else:
        contract_checks = contract_record.get("checks") or {}
        relation_replay = contract_record.get("relation_form_replay") or {}
        work_counters = contract_record.get("lane_amortized_work_counters") or {}
        if contract_record.get("batch_lane_contract_status") != "batch_lane_contract_verified":
            add_failure(
                failures,
                "lane_contract_status_not_verified",
                {"batch_lane_contract_status": contract_record.get("batch_lane_contract_status")},
            )

    if global_failures:
        add_failure(
            failures,
            "global_batch_lane_artifact_mismatch",
            {"failure_count": len(global_failures)},
        )

    lane_keys_by_source: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    left_y_outputs: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    event_keys = set()
    for case in source_cases:
        event_keys.add(case["event_key"])
        lane_keys_by_source[(case["group_index"], *lane_key(case))].append(case)
        left_y_outputs[left_y_lane_key(case)].add(lane_key(case))

        if case["operation"] != "addition":
            add_failure(
                failures,
                "unsupported_affine_operation",
                {"event_key": case["event_key_json"], "operation": case["operation"]},
            )
        if not bool(case.get("matches_verifier_add")):
            add_failure(failures, "affine_trace_verifier_add_mismatch", {"event_key": case["event_key_json"]})
        if not bool(case.get("matches_abi_candidate_point")):
            add_failure(failures, "affine_trace_abi_candidate_mismatch", {"event_key": case["event_key_json"]})
        if not bool(case.get("local_affine_outputs_consistent")):
            add_failure(failures, "row_instance_affine_outputs_not_consistent", {"event_key": case["event_key_json"]})
        if as_int(case.get("fanout"), -1) != as_int(case.get("row_instance_trace_count"), -2):
            add_failure(
                failures,
                "fanout_row_trace_count_mismatch",
                {
                    "event_key": case["event_key_json"],
                    "fanout": case.get("fanout"),
                    "row_instance_trace_count": case.get("row_instance_trace_count"),
                },
            )
        if as_int(case.get("row_instance_trace_count"), -1) != as_int(
            case.get("observed_row_instance_trace_count"), -2
        ):
            add_failure(
                failures,
                "observed_row_trace_count_mismatch",
                {
                    "event_key": case["event_key_json"],
                    "row_instance_trace_count": case.get("row_instance_trace_count"),
                    "observed": case.get("observed_row_instance_trace_count"),
                },
            )
        expected_register = trace_register_key(
            {
                "slope_numerator": case["expected_slope_numerator"],
                "slope": case["expected_slope"],
                "intercept": case["expected_intercept"],
                "result": [case["expected_x"], case["expected_y"]],
            }
        )
        for row_register in case["row_register_keys"]:
            if row_register != expected_register:
                add_failure(
                    failures,
                    "row_trace_register_mismatch",
                    {"event_key": case["event_key_json"], "expected": list(expected_register), "observed": list(row_register)},
                )

    for ly_key, outputs in left_y_outputs.items():
        if len(outputs) != 1:
            add_failure(
                failures,
                "left_y_lane_output_not_unique",
                {"left_y_lane_key": list(ly_key), "output_count": len(outputs)},
            )

    lane_curve_checks = []
    for source_lane_index, lane_cases in enumerate(
        sorted(
            lane_keys_by_source.values(),
            key=lambda values: (
                min(as_int(case["lane_index"]) for case in values),
                values[0]["source_name"],
                values[0]["event_key"],
            ),
        )
    ):
        first = lane_cases[0]
        global_lane = global_lanes_by_index[as_int(first["lane_index"])]
        candidate_point = [as_int(global_lane["expected_x"]), as_int(global_lane["expected_y"])]
        curve_check = candidate_point_curve_check(verifier, ctx, candidate_point)
        curve_check.update(
            {
                "source_lane_index": source_lane_index,
                "global_lane_index": as_int(first["lane_index"]),
                "group_index": as_int(first["group_index"]),
                "left_y": as_int(first["left_y"]),
                "case_count": len(lane_cases),
                "row_instance_trace_count": sum(
                    as_int(case.get("row_instance_trace_count")) for case in lane_cases
                ),
                "event_keys": [case["event_key_json"] for case in lane_cases],
                "registers": {
                    "slope_numerator": as_int(first["expected_slope_numerator"]),
                    "slope": as_int(first["expected_slope"]),
                    "intercept": as_int(first["expected_intercept"]),
                },
            }
        )
        lane_curve_checks.append(curve_check)
        if not all(
            bool(curve_check.get(name))
            for name in ("parsed", "finite_point", "coords_in_field", "on_curve", "in_base_order_subgroup")
        ):
            add_failure(
                failures,
                "batch_lane_curve_check_failed",
                {
                    "global_lane_index": as_int(first["lane_index"]),
                    "curve_check": curve_check,
                },
            )

    event_case_count = len(source_cases)
    batch_lane_count = len(lane_curve_checks)
    second_pass_check_count = sum(as_int(case.get("row_instance_trace_count")) for case in source_cases)
    reused_case_count = sum(1 for case in source_cases if bool(case.get("candidate_point_reused")))
    batch_case_count = as_int(batch_record.get("case_count"), -1) if batch_record else -1
    recorded_batch_lane_count = as_int(batch_record.get("lane_count"), -1) if batch_record else -1

    checks = {
        "affine_trace_verified": record.get("affine_trace_status") == "fused_kernel_affine_trace_verified",
        "batch_lane_verified": bool(
            batch_record and batch_record.get("batch_lane_status") == "batch_lane_native_kernel_verified"
        ),
        "lane_contract_verified": bool(
            contract_record
            and contract_record.get("batch_lane_contract_status") == "batch_lane_contract_verified"
        ),
        "case_count_matches": batch_case_count == event_case_count
        and as_int(contract_checks.get("batch_case_count"), -1) == event_case_count,
        "batch_lane_count_matches": recorded_batch_lane_count == batch_lane_count
        and as_int(contract_checks.get("batch_lane_count"), -1) == batch_lane_count
        and as_int(work_counters.get("batch_lane_first_pass_ops"), -1) == batch_lane_count,
        "second_pass_check_count_matches": as_int(contract_checks.get("second_pass_check_count"), -1)
        == second_pass_check_count,
        "preserves_second_pass_fanout": bool(contract_checks.get("preserves_second_pass_fanout"))
        and as_int(contract_checks.get("first_pass_group_instance_count"), -1) == second_pass_check_count,
        "relation_replay_derived": bool(relation_replay.get("derived")),
        "relation_replay_matches_secret": bool(relation_replay.get("matches_recorded_secret")),
        "all_lanes_on_curve": all(bool(check.get("on_curve")) for check in lane_curve_checks),
        "all_lanes_in_subgroup": all(
            bool(check.get("in_base_order_subgroup")) for check in lane_curve_checks
        ),
        "all_lanes_finite": all(bool(check.get("finite_point")) for check in lane_curve_checks),
        "all_event_outputs_match_abi": all(bool(case.get("matches_abi_candidate_point")) for case in source_cases),
        "all_event_outputs_match_verifier_add": all(
            bool(case.get("matches_verifier_add")) for case in source_cases
        ),
        "all_event_outputs_locally_consistent": all(
            bool(case.get("local_affine_outputs_consistent")) for case in source_cases
        ),
        "left_y_lane_outputs_unique": all(len(outputs) == 1 for outputs in left_y_outputs.values()),
        "global_batch_lane_artifact_matches": bool(global_checks.get("global_batch_lane_artifact_matches")),
        "lane_contract_below_rho": bool(work_counters.get("lane_fused_below_rho")),
    }

    reported_only_checks = {"lane_contract_below_rho"}
    for name, passed in checks.items():
        if name in reported_only_checks:
            continue
        if not passed:
            add_failure(failures, f"{name}_failed")

    return {
        "source_name": source_name,
        "public_group_key": public_group,
        "batch_lane_abi_status": "batch_lane_abi_verified" if not failures else "batch_lane_abi_failed_check",
        "target_context": {
            "label": ctx["label"],
            "p": ctx["p"],
            "ainvs": ctx["ainvs"],
            "group_order": ctx["group_order"],
            "base_order": ctx["base_order"],
            "base": ctx["base"],
            "generic_rho_steps": ctx["generic_rho_steps"],
        },
        "source_lane_counts": {
            "event_case_count": event_case_count,
            "batch_lane_count": batch_lane_count,
            "second_pass_check_count": second_pass_check_count,
            "reused_case_count": reused_case_count,
            "distinct_candidate_points": sorted_point_json(
                {(as_int(check["point"][0]), as_int(check["point"][1])) for check in lane_curve_checks}
            ),
        },
        "lane_amortized_work_counters": work_counters,
        "relation_form_replay": relation_replay,
        "lane_curve_checks": lane_curve_checks,
        "checks": checks,
        "failures": failures,
        "implementation_boundary": (
            "Batch-lane ABI/curve guard: grouped-left-y native lanes produce "
            "verifier-curve base-subgroup points while the portable lane "
            "contract preserves second-pass fanout and relation-derived secret recovery."
        ),
    }


def summarize(records: list[dict[str, Any]], global_checks: dict[str, Any], global_lanes: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if record.get("batch_lane_abi_status") == "batch_lane_abi_verified"]
    curve_verified = [record for record in records if bool((record.get("checks") or {}).get("all_lanes_on_curve"))]
    subgroup_verified = [
        record for record in records if bool((record.get("checks") or {}).get("all_lanes_in_subgroup"))
    ]
    fanout_verified = [
        record for record in records if bool((record.get("checks") or {}).get("preserves_second_pass_fanout"))
    ]
    derived = [record for record in records if bool((record.get("checks") or {}).get("relation_replay_derived"))]
    matched_secret = [
        record for record in records if bool((record.get("checks") or {}).get("relation_replay_matches_secret"))
    ]
    below = [record for record in verified if bool((record.get("checks") or {}).get("lane_contract_below_rho"))]
    ratios = [
        (record.get("lane_amortized_work_counters") or {}).get("lane_fused_ops_over_rho")
        for record in below
        if (record.get("lane_amortized_work_counters") or {}).get("lane_fused_ops_over_rho") is not None
    ]
    return {
        "record_count": len(records),
        "batch_lane_abi_verified_count": len(verified),
        "lane_curve_verified_count": len(curve_verified),
        "lane_subgroup_verified_count": len(subgroup_verified),
        "second_pass_fanout_verified_count": len(fanout_verified),
        "relation_replay_derived_count": len(derived),
        "relation_replay_secret_match_count": len(matched_secret),
        "below_rho_lane_contract_count": len(below),
        "global_batch_lane_artifact_verified": bool(global_checks.get("global_batch_lane_artifact_matches")),
        "global_batch_lane_count": len(global_lanes),
        "total_batch_lanes": sum(as_int((record.get("source_lane_counts") or {}).get("batch_lane_count")) for record in records),
        "total_event_cases": sum(as_int((record.get("source_lane_counts") or {}).get("event_case_count")) for record in records),
        "total_second_pass_checks": sum(
            as_int((record.get("source_lane_counts") or {}).get("second_pass_check_count")) for record in records
        ),
        "mean_lane_contract_below_rho_ops_over_rho": (
            round(mean([float(value) for value in ratios]), 8) if ratios else None
        ),
        "interpretation": (
            "The batch-lane implementation boundary is ABI-stable at the curve layer: "
            "source-local lanes are valid base-subgroup points and the lane-amortized "
            "contract keeps the original second-pass relation derivation intact."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affine-source", type=Path, default=DEFAULT_AFFINE_SOURCE)
    parser.add_argument("--batch-lane-source", type=Path, default=DEFAULT_BATCH_LANE_SOURCE)
    parser.add_argument("--lane-contract-source", type=Path, default=DEFAULT_LANE_CONTRACT_SOURCE)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    affine_source = load_json(args.affine_source)
    batch_source = load_json(args.batch_lane_source)
    lane_contract_source = load_json(args.lane_contract_source)
    wanted = set(args.source_name or [])

    raw_records = [
        record
        for record in affine_source.get("records") or []
        if isinstance(record, dict) and (not wanted or str(record.get("source_name") or "") in wanted)
    ]
    raw_cases = collect_cases(raw_records)
    groups, global_lanes, event_cases = build_global_layout(raw_cases)
    global_checks, global_failures = validate_global_batch_artifact(batch_source, groups, global_lanes, event_cases)
    event_cases_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in event_cases:
        event_cases_by_source[str(case.get("source_name") or "")].append(case)

    targets = {
        str((record.get("public_group_key") or {}).get("target") or "")
        for record in raw_records
        if str((record.get("public_group_key") or {}).get("target") or "")
    }
    verifier = relation_probe.load_verifier_module()
    contexts = load_target_contexts(verifier, targets)
    batch_by_source = records_by_source(batch_source)
    contract_by_source = records_by_source(lane_contract_source)
    global_lanes_by_index = {as_int(lane.get("lane_index")): lane for lane in global_lanes}

    records = [
        validate_source_record(
            verifier,
            record,
            event_cases_by_source[str(record.get("source_name") or "")],
            global_lanes_by_index,
            contexts[str((record.get("public_group_key") or {}).get("target"))],
            batch_by_source.get(str(record.get("source_name") or "")),
            contract_by_source.get(str(record.get("source_name") or "")),
            global_checks,
            global_failures,
        )
        for record in raw_records
    ]

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_batch_lane_abi_probe_v1",
        "method": "verifier_curve_and_contract_guard_for_grouped_left_y_batch_lane_kernel",
        "parameters": {
            "affine_source": str(args.affine_source),
            "batch_lane_source": str(args.batch_lane_source),
            "lane_contract_source": str(args.lane_contract_source),
            "source_names": sorted(wanted),
            "campaign_task_dir": str(CAMPAIGN_TASK_DIR),
        },
        "summary": summarize(records, global_checks, global_lanes),
        "global_batch_lane_artifact_checks": global_checks,
        "global_batch_lane_artifact_failures": global_failures,
        "records": sorted(
            records,
            key=lambda record: (
                record.get("batch_lane_abi_status") != "batch_lane_abi_verified",
                not bool((record.get("checks") or {}).get("lane_contract_below_rho")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This validates an implementation boundary for existing relation systems; it is not a fresh relation search.",
            "The lane-amortized accounting remains conditional on the native batch-lane artifact and portable contract.",
            "Second-pass relation predicates and modular linear algebra are unchanged from the prior fused-kernel contract.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0 if output["summary"]["batch_lane_abi_verified_count"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
