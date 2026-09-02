#!/usr/bin/env python3
"""Verify ABI and curve invariants for fused repeated-coordinate contracts.

The portable fused-kernel contract is the boundary a lower-level
FFE/summation-polynomial implementation should target.  This probe checks that
boundary without replaying scanner internals: every first-pass candidate point
must be a finite point on the verifier-reduced curve, every row-specific
second-pass check must line up with a first-pass public event key, and the
compact relation summaries must match the accepted row checks.

It is an ABI/curve guard for promoting the Python contract into a lower-level
kernel, not a new relation harvester.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

import relation_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_CONTRACT_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_contract_target67_672_744.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_abi.json"


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


def compact_event_key(raw: Any) -> tuple[str, int, int]:
    values = list(raw or [])
    if len(values) != 3:
        return ("", -1, -1)
    return (str(values[0]), as_int(values[1], -1), as_int(values[2], -1))


def compact_instance_key(raw: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(raw.get("row_key") or ""),
        as_int(raw.get("scheduled_trial"), -1),
        as_int(raw.get("candidate_pos"), -1),
    )


def relation_summary_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(summary.get("row_key") or ""),
        as_int(summary.get("relation_index"), -1),
        as_int(summary.get("candidate_pos"), -1),
        as_int(summary.get("scheduled_trial"), -1),
        as_int(summary.get("original_trial"), -1),
        as_int(summary.get("q_coeff"), -1),
        as_int(summary.get("rhs"), -1),
        tuple(as_int(value, -1) for value in summary.get("terms") or []),
    )


def point_key(point_json: Any) -> tuple[int, int] | None:
    if not isinstance(point_json, list) or len(point_json) != 2:
        return None
    return (as_int(point_json[0]), as_int(point_json[1]))


def sorted_point_json(points: set[tuple[int, int]]) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in sorted(points)]


def add_failure(failures: list[dict[str, Any]], code: str, detail: dict[str, Any]) -> None:
    failures.append({"code": code, **detail})


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


def validate_group_shape(group: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    event_key = compact_event_key(group.get("event_key"))
    fanout = as_int(group.get("fanout"), -1)
    instances = group.get("row_instances") or []
    saved_ops = as_int(group.get("candidate_point_reuse_saved_ops"), -1)
    reused = bool(group.get("candidate_point_reused"))

    if str(group.get("leaf_signature") or "") != event_key[0]:
        add_failure(
            failures,
            "group_leaf_signature_mismatch",
            {"event_key": list(event_key), "leaf_signature": group.get("leaf_signature")},
        )
    if as_int(group.get("scout_pos"), -1) != event_key[1]:
        add_failure(
            failures,
            "group_scout_pos_mismatch",
            {"event_key": list(event_key), "scout_pos": group.get("scout_pos")},
        )
    if as_int(group.get("original_trial"), -1) != event_key[2]:
        add_failure(
            failures,
            "group_original_trial_mismatch",
            {"event_key": list(event_key), "original_trial": group.get("original_trial")},
        )
    if fanout != len(instances):
        add_failure(
            failures,
            "group_fanout_mismatch",
            {"event_key": list(event_key), "fanout": fanout, "row_instance_count": len(instances)},
        )
    if as_int(group.get("candidate_point_ops"), -1) != 1:
        add_failure(
            failures,
            "group_candidate_point_ops_not_one",
            {"event_key": list(event_key), "candidate_point_ops": group.get("candidate_point_ops")},
        )
    if saved_ops != max(0, fanout - 1):
        add_failure(
            failures,
            "group_saved_ops_mismatch",
            {"event_key": list(event_key), "saved_ops": saved_ops, "expected_saved_ops": max(0, fanout - 1)},
        )
    if reused != (fanout > 1):
        add_failure(
            failures,
            "group_reuse_flag_mismatch",
            {"event_key": list(event_key), "candidate_point_reused": reused, "fanout": fanout},
        )

    representative_key = (
        str(group.get("representative_row_key") or ""),
        as_int(group.get("representative_scheduled_trial"), -1),
    )
    if not any(
        str(instance.get("row_key") or "") == representative_key[0]
        and as_int(instance.get("scheduled_trial"), -1) == representative_key[1]
        for instance in instances
    ):
        add_failure(
            failures,
            "group_representative_not_in_instances",
            {"event_key": list(event_key), "representative": list(representative_key)},
        )


def validate_second_pass(
    groups_by_event: dict[tuple[str, int, int], dict[str, Any]],
    second_pass: list[dict[str, Any]],
    summaries_by_key: dict[tuple[Any, ...], dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[tuple[str, int, int], int]:
    accepted_by_event: Counter[tuple[str, int, int]] = Counter()
    seen_instances: set[tuple[str, int, int, tuple[str, int, int]]] = set()
    for check in second_pass:
        event_key = compact_event_key(check.get("event_key"))
        group = groups_by_event.get(event_key)
        if group is None:
            add_failure(failures, "second_pass_missing_first_pass_group", {"event_key": list(event_key)})
            continue

        instance_key = compact_instance_key(check)
        event_instance_key = (*instance_key, event_key)
        if event_instance_key in seen_instances:
            add_failure(
                failures,
                "second_pass_duplicate_instance",
                {"event_key": list(event_key), "instance_key": list(instance_key)},
            )
        seen_instances.add(event_instance_key)

        allowed_instances = {compact_instance_key(instance) for instance in group.get("row_instances") or []}
        if instance_key not in allowed_instances:
            add_failure(
                failures,
                "second_pass_instance_not_in_group_fanout",
                {"event_key": list(event_key), "instance_key": list(instance_key)},
            )
        if str(check.get("leaf_signature") or "") != event_key[0]:
            add_failure(
                failures,
                "second_pass_leaf_signature_mismatch",
                {"event_key": list(event_key), "leaf_signature": check.get("leaf_signature")},
            )
        if as_int(check.get("scout_pos"), -1) != event_key[1]:
            add_failure(
                failures,
                "second_pass_scout_pos_mismatch",
                {"event_key": list(event_key), "scout_pos": check.get("scout_pos")},
            )
        if as_int(check.get("original_trial"), -1) != event_key[2]:
            add_failure(
                failures,
                "second_pass_original_trial_mismatch",
                {"event_key": list(event_key), "original_trial": check.get("original_trial")},
            )
        if check.get("candidate_point_source") != "hot_path_representative":
            add_failure(
                failures,
                "second_pass_candidate_point_source_mismatch",
                {"event_key": list(event_key), "source": check.get("candidate_point_source")},
            )
        if str(check.get("representative_row_key") or "") != str(group.get("representative_row_key") or ""):
            add_failure(
                failures,
                "second_pass_representative_row_mismatch",
                {"event_key": list(event_key), "representative_row_key": check.get("representative_row_key")},
            )
        if as_int(check.get("representative_scheduled_trial"), -1) != as_int(
            group.get("representative_scheduled_trial"), -1
        ):
            add_failure(
                failures,
                "second_pass_representative_trial_mismatch",
                {
                    "event_key": list(event_key),
                    "representative_scheduled_trial": check.get("representative_scheduled_trial"),
                },
            )

        if bool(check.get("accepted_relation")):
            accepted_by_event[event_key] += 1
            summary = check.get("relation_summary")
            if not isinstance(summary, dict):
                add_failure(failures, "accepted_check_missing_summary", {"event_key": list(event_key)})
                continue
            summary_key = relation_summary_key(summary)
            if summary_key not in summaries_by_key:
                add_failure(
                    failures,
                    "accepted_summary_not_in_contract_summary_list",
                    {"event_key": list(event_key), "summary_key": list(summary_key[:5])},
                )
            expected_fields = {
                "row_key": str(check.get("row_key") or ""),
                "candidate_pos": as_int(check.get("candidate_pos"), -1),
                "scheduled_trial": as_int(check.get("scheduled_trial"), -1),
                "original_trial": as_int(check.get("original_trial"), -1),
                "relation_index": as_int(check.get("relation_index"), -1),
            }
            observed_fields = {
                "row_key": str(summary.get("row_key") or ""),
                "candidate_pos": as_int(summary.get("candidate_pos"), -1),
                "scheduled_trial": as_int(summary.get("scheduled_trial"), -1),
                "original_trial": as_int(summary.get("original_trial"), -1),
                "relation_index": as_int(summary.get("relation_index"), -1),
            }
            if observed_fields != expected_fields:
                add_failure(
                    failures,
                    "accepted_summary_field_mismatch",
                    {
                        "event_key": list(event_key),
                        "expected": expected_fields,
                        "observed": observed_fields,
                    },
                )
            if str(summary.get("leaf_index")) != event_key[0]:
                add_failure(
                    failures,
                    "accepted_summary_leaf_mismatch",
                    {"event_key": list(event_key), "leaf_index": summary.get("leaf_index")},
                )
        elif check.get("relation_summary") is not None:
            add_failure(failures, "rejected_check_has_summary", {"event_key": list(event_key)})
    return dict(accepted_by_event)


def exact_744_reuse_anchor(
    source_name: str,
    groups_by_event: dict[tuple[str, int, int], dict[str, Any]],
    second_pass: list[dict[str, Any]],
    distinct_points: set[tuple[int, int]],
) -> dict[str, Any]:
    if source_name != "coord1114_744":
        return {"applicable": False, "verified": None}

    expected_points = {(2705, 9753), (9251, 9216)}
    reuse_key = ("12", 174, 469)
    group = groups_by_event.get(reuse_key) or {}
    accepted_checks = [
        check
        for check in second_pass
        if compact_event_key(check.get("event_key")) == reuse_key and bool(check.get("accepted_relation"))
    ]
    verified = bool(
        distinct_points == expected_points
        and point_key(group.get("candidate_point")) == (9251, 9216)
        and as_int(group.get("fanout"), -1) == 2
        and bool(group.get("candidate_point_reused"))
        and as_int(group.get("accepted_relation_count"), -1) == 2
        and len(accepted_checks) == 2
    )
    return {
        "applicable": True,
        "verified": verified,
        "expected_distinct_candidate_points": sorted_point_json(expected_points),
        "observed_distinct_candidate_points": sorted_point_json(distinct_points),
        "reuse_event_key": list(reuse_key),
        "reuse_event_candidate_point": group.get("candidate_point"),
        "reuse_event_fanout": group.get("fanout"),
        "reuse_event_accepted_relation_count": group.get("accepted_relation_count"),
        "accepted_check_count_for_reuse_event": len(accepted_checks),
    }


def validate_record(verifier: Any, record: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    source_name = str(record.get("source_name") or "")
    public_group = record.get("public_group_key") or {}
    kernel_contract = record.get("kernel_contract") or {}
    first_pass = kernel_contract.get("first_pass_candidate_point_groups") or []
    second_pass = kernel_contract.get("second_pass_row_checks") or []
    summaries = kernel_contract.get("accepted_relation_summaries") or []
    relation_replay = record.get("relation_form_replay") or {}
    work_counters = record.get("work_counters") or {}
    recorded_checks = record.get("checks") or {}

    if record.get("contract_status") != "fused_kernel_contract_verified":
        add_failure(
            failures,
            "contract_status_not_verified",
            {"contract_status": record.get("contract_status")},
        )
    if as_int(relation_replay.get("target_order"), -1) != int(ctx["base_order"]):
        add_failure(
            failures,
            "target_order_mismatch",
            {"contract_target_order": relation_replay.get("target_order"), "verifier_base_order": ctx["base_order"]},
        )
    if as_int(work_counters.get("generic_rho_steps"), -1) != int(ctx["generic_rho_steps"]):
        add_failure(
            failures,
            "generic_rho_steps_mismatch",
            {
                "contract_generic_rho_steps": work_counters.get("generic_rho_steps"),
                "verifier_generic_rho_steps": ctx["generic_rho_steps"],
            },
        )

    groups_by_event: dict[tuple[str, int, int], dict[str, Any]] = {}
    curve_checks = []
    distinct_points: set[tuple[int, int]] = set()
    for group in first_pass:
        event_key = compact_event_key(group.get("event_key"))
        if event_key in groups_by_event:
            add_failure(failures, "duplicate_first_pass_event_key", {"event_key": list(event_key)})
        groups_by_event[event_key] = group
        validate_group_shape(group, failures)
        curve_check = candidate_point_curve_check(verifier, ctx, group.get("candidate_point"))
        curve_check["event_key"] = list(event_key)
        curve_checks.append(curve_check)
        if point_key(group.get("candidate_point")) is not None:
            distinct_points.add(point_key(group.get("candidate_point")))  # type: ignore[arg-type]
        if not all(
            bool(curve_check.get(name))
            for name in ("parsed", "finite_point", "coords_in_field", "on_curve", "in_base_order_subgroup")
        ):
            add_failure(
                failures,
                "candidate_point_curve_check_failed",
                {"event_key": list(event_key), "curve_check": curve_check},
            )

    summary_keys = [relation_summary_key(summary) for summary in summaries if isinstance(summary, dict)]
    summary_counts = Counter(summary_keys)
    for key, count in summary_counts.items():
        if count > 1:
            add_failure(
                failures,
                "duplicate_accepted_relation_summary",
                {"summary_key": list(key[:5]), "count": count},
            )
    summaries_by_key = {
        relation_summary_key(summary): summary
        for summary in summaries
        if isinstance(summary, dict)
    }
    accepted_by_event = validate_second_pass(groups_by_event, second_pass, summaries_by_key, failures)

    for event_key, group in groups_by_event.items():
        accepted_count = accepted_by_event.get(event_key, 0)
        if as_int(group.get("accepted_relation_count"), -1) != accepted_count:
            add_failure(
                failures,
                "group_accepted_relation_count_mismatch",
                {
                    "event_key": list(event_key),
                    "group_accepted_relation_count": group.get("accepted_relation_count"),
                    "second_pass_accepted_relation_count": accepted_count,
                },
            )

    first_pass_group_count = len(first_pass)
    first_pass_group_instance_count = sum(as_int(group.get("fanout"), 0) for group in first_pass)
    second_pass_check_count = len(second_pass)
    accepted_relation_count = sum(1 for check in second_pass if bool(check.get("accepted_relation")))
    shared_first_event_pass_ops = sum(as_int(group.get("candidate_point_ops"), 0) for group in first_pass)
    first_pass_saved_ops = sum(as_int(group.get("candidate_point_reuse_saved_ops"), 0) for group in first_pass)

    counter_expectations = {
        "first_pass_group_count": first_pass_group_count,
        "first_pass_group_instance_count": first_pass_group_instance_count,
        "second_pass_check_count": second_pass_check_count,
        "accepted_relation_count": accepted_relation_count,
    }
    for key, expected in counter_expectations.items():
        if as_int(recorded_checks.get(key), -1) != expected:
            add_failure(
                failures,
                "recorded_check_counter_mismatch",
                {"counter": key, "expected": expected, "recorded": recorded_checks.get(key)},
            )
    if as_int(work_counters.get("shared_first_event_pass_ops"), -1) != shared_first_event_pass_ops:
        add_failure(
            failures,
            "first_pass_ops_counter_mismatch",
            {
                "expected": shared_first_event_pass_ops,
                "work_counter": work_counters.get("shared_first_event_pass_ops"),
            },
        )
    if as_int(work_counters.get("second_event_pass_ops"), -1) != second_pass_check_count:
        add_failure(
            failures,
            "second_pass_ops_counter_mismatch",
            {"expected": second_pass_check_count, "work_counter": work_counters.get("second_event_pass_ops")},
        )
    if as_int(work_counters.get("first_pass_saved_ops"), -1) != first_pass_saved_ops:
        add_failure(
            failures,
            "first_pass_saved_ops_counter_mismatch",
            {"expected": first_pass_saved_ops, "work_counter": work_counters.get("first_pass_saved_ops")},
        )
    if len(summaries) != accepted_relation_count:
        add_failure(
            failures,
            "accepted_summary_count_mismatch",
            {"summary_count": len(summaries), "accepted_relation_count": accepted_relation_count},
        )
    if first_pass_group_instance_count != second_pass_check_count:
        add_failure(
            failures,
            "first_pass_second_pass_fanout_mismatch",
            {
                "first_pass_group_instance_count": first_pass_group_instance_count,
                "second_pass_check_count": second_pass_check_count,
            },
        )

    anchor = exact_744_reuse_anchor(source_name, groups_by_event, second_pass, distinct_points)
    if bool(anchor.get("applicable")) and not bool(anchor.get("verified")):
        add_failure(failures, "coord1114_744_reuse_anchor_mismatch", anchor)

    return {
        "source_name": source_name,
        "public_group_key": public_group,
        "abi_curve_status": "fused_kernel_abi_curve_verified" if not failures else "fused_kernel_abi_curve_failed_check",
        "target_context": {
            "label": ctx["label"],
            "p": ctx["p"],
            "ainvs": ctx["ainvs"],
            "group_order": ctx["group_order"],
            "base_order": ctx["base_order"],
            "base": ctx["base"],
            "generic_rho_steps": ctx["generic_rho_steps"],
        },
        "candidate_point_curve_checks": curve_checks,
        "distinct_candidate_points": sorted_point_json(distinct_points),
        "event_reuse_groups": [
            {
                "event_key": list(event_key),
                "candidate_point": group.get("candidate_point"),
                "fanout": as_int(group.get("fanout")),
                "accepted_relation_count": as_int(group.get("accepted_relation_count")),
            }
            for event_key, group in sorted(groups_by_event.items())
            if as_int(group.get("fanout")) > 1
        ],
        "coord1114_744_reuse_anchor": anchor,
        "checks": {
            "candidate_point_curve_verified": all(bool(item.get("on_curve")) for item in curve_checks),
            "candidate_point_subgroup_verified": all(
                bool(item.get("in_base_order_subgroup")) for item in curve_checks
            ),
            "candidate_point_count": len(curve_checks),
            "first_pass_group_count": first_pass_group_count,
            "first_pass_group_instance_count": first_pass_group_instance_count,
            "second_pass_check_count": second_pass_check_count,
            "accepted_relation_count": accepted_relation_count,
            "accepted_summary_count": len(summaries),
            "shared_first_event_pass_ops": shared_first_event_pass_ops,
            "first_pass_saved_ops": first_pass_saved_ops,
            "first_second_pass_fanout_match": first_pass_group_instance_count == second_pass_check_count,
            "summary_count_matches_accepted_checks": len(summaries) == accepted_relation_count,
            "matches_contract_below_rho": bool(work_counters.get("fused_below_rho")),
            "matches_contract_event_reuse_target": bool(record.get("event_reuse_target")),
        },
        "failures": failures,
        "implementation_boundary": (
            "ABI/curve invariant check for the portable fused kernel contract. "
            "The first pass emits curve-valid candidate points; the second pass "
            "consumes only event keys and row-specific relation predicates."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if record.get("abi_curve_status") == "fused_kernel_abi_curve_verified"]
    curve_verified = [record for record in records if bool((record.get("checks") or {}).get("candidate_point_curve_verified"))]
    subgroup_verified = [
        record for record in records if bool((record.get("checks") or {}).get("candidate_point_subgroup_verified"))
    ]
    fanout_verified = [
        record for record in records if bool((record.get("checks") or {}).get("first_second_pass_fanout_match"))
    ]
    summary_verified = [
        record for record in records if bool((record.get("checks") or {}).get("summary_count_matches_accepted_checks"))
    ]
    below = [
        record for record in verified if bool((record.get("checks") or {}).get("matches_contract_below_rho"))
    ]
    event_reuse = [
        record for record in below if bool((record.get("checks") or {}).get("matches_contract_event_reuse_target"))
    ]
    anchor = [
        record
        for record in records
        if bool((record.get("coord1114_744_reuse_anchor") or {}).get("applicable"))
    ]
    return {
        "record_count": len(records),
        "abi_curve_verified_count": len(verified),
        "candidate_point_curve_verified_count": len(curve_verified),
        "candidate_point_subgroup_verified_count": len(subgroup_verified),
        "first_second_pass_fanout_verified_count": len(fanout_verified),
        "relation_summary_abi_verified_count": len(summary_verified),
        "below_rho_verified_count": len(below),
        "event_reuse_below_rho_verified_count": len(event_reuse),
        "coord1114_744_reuse_anchor_verified": bool(anchor) and all(
            bool((record.get("coord1114_744_reuse_anchor") or {}).get("verified")) for record in anchor
        ),
        "total_candidate_point_checks": sum(as_int((record.get("checks") or {}).get("candidate_point_count")) for record in records),
        "total_second_pass_checks": sum(as_int((record.get("checks") or {}).get("second_pass_check_count")) for record in records),
        "total_accepted_relations": sum(as_int((record.get("checks") or {}).get("accepted_relation_count")) for record in records),
        "interpretation": (
            "The portable contract is ABI-stable at the curve boundary: first-pass "
            "candidate-point outputs are verifier-curve points in the base-order "
            "subgroup, and second-pass row checks consume those public event keys "
            "without scanner-side candidate equality recomputation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-source", type=Path, default=DEFAULT_CONTRACT_SOURCE)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = load_json(args.contract_source)
    wanted = set(args.source_name or [])
    raw_records = [
        record
        for record in source.get("records") or []
        if isinstance(record, dict) and (not wanted or str(record.get("source_name") or "") in wanted)
    ]
    targets = {
        str((record.get("public_group_key") or {}).get("target") or "")
        for record in raw_records
        if str((record.get("public_group_key") or {}).get("target") or "")
    }
    verifier = relation_probe.load_verifier_module()
    contexts = load_target_contexts(verifier, targets)
    records = [
        validate_record(verifier, record, contexts[str((record.get("public_group_key") or {}).get("target"))])
        for record in raw_records
    ]

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_abi_probe_v1",
        "method": "verifier_curve_and_second_pass_abi_guard_for_portable_fused_kernel_contract",
        "parameters": {
            "contract_source": str(args.contract_source),
            "source_names": sorted(wanted),
            "campaign_task_dir": str(CAMPAIGN_TASK_DIR),
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                record.get("abi_curve_status") != "fused_kernel_abi_curve_verified",
                not bool((record.get("checks") or {}).get("matches_contract_event_reuse_target")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This validates the ABI and curve boundary of the portable contract; it is not a new ECDLP solve.",
            "It does not search for additional relation systems or lower the contract into optimized native code.",
            "A promoted FFE/summation-polynomial kernel must emit the same first-pass candidate points and second-pass checks.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
