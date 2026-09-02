#!/usr/bin/env python3
"""Emit affine field-operation traces for fused kernel candidate points.

The ABI probe verifies that the portable contract is valid at the curve
boundary.  This probe moves one layer lower: it rematerializes the public
worklist contexts, emits the affine generalized-Weierstrass addition trace that
produces each first-pass candidate point, and compares those field-operation
outputs against the ABI artifact.

The result is a trace shape an optimized FFE/summation-polynomial kernel can
match before it tries to replace the Python verifier-path arithmetic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

WORKTREE_TASK_DIR = Path(__file__).resolve().parent
if str(WORKTREE_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(WORKTREE_TASK_DIR))

import ffe_public_repeated_coordinate_fused_candidate_point_probe as candidate_point_probe


scanner_probe = candidate_point_probe.scanner_probe
replay_probe = candidate_point_probe.replay_probe

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_WORKLIST_SOURCE = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_worklist_audit_target67_672_744.json"
DEFAULT_CHARGED_SOURCE = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_hit_event_charged_replay_target67_672_744.json"
DEFAULT_ABI_SOURCE = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_abi_target67_672_744.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_affine_trace.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    return scanner_probe.as_int(value, default)


def compact_event_key(raw: Any) -> tuple[str, int, int]:
    values = list(raw or [])
    if len(values) != 3:
        return ("", -1, -1)
    return (str(values[0]), as_int(values[1], -1), as_int(values[2], -1))


def compact_point(verifier: Any, point: Any) -> Any:
    return verifier.point_to_json(point)


def point_key(raw: Any) -> tuple[int, int] | None:
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    return (as_int(raw[0]), as_int(raw[1]))


def add_failure(failures: list[dict[str, Any]], code: str, detail: dict[str, Any]) -> None:
    failures.append({"code": code, **detail})


def sorted_group_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        instances,
        key=lambda item: (
            str(item.get("row_key") or ""),
            as_int(item.get("scheduled_trial"), -1),
            as_int(item.get("candidate_pos"), -1),
        ),
    )


def inv_mod(value: int, p: int) -> int:
    return pow(value % p, -1, p)


def affine_add_trace(verifier: Any, left: Any, right: Any, ainvs: list[int], p: int) -> dict[str, Any]:
    a1, a2, a3, a4, a6 = [int(value) % p for value in ainvs]
    trace: dict[str, Any] = {
        "field": f"GF({p})",
        "curve_model": "generalized_weierstrass",
        "ainvs_mod_p": [a1, a2, a3, a4, a6],
        "left": compact_point(verifier, left),
        "right": compact_point(verifier, right),
    }

    if left is verifier.POINT_AT_INFINITY:
        result = right
        trace.update(
            {
                "operation": "identity_left",
                "slope_numerator": None,
                "slope_denominator": None,
                "slope_denominator_inverse": None,
                "slope": None,
                "intercept": None,
                "result": compact_point(verifier, result),
                "verifier_result": compact_point(verifier, verifier.add_points(left, right, ainvs, p)),
                "matches_verifier_add": compact_point(verifier, result)
                == compact_point(verifier, verifier.add_points(left, right, ainvs, p)),
            }
        )
        return trace
    if right is verifier.POINT_AT_INFINITY:
        result = left
        trace.update(
            {
                "operation": "identity_right",
                "slope_numerator": None,
                "slope_denominator": None,
                "slope_denominator_inverse": None,
                "slope": None,
                "intercept": None,
                "result": compact_point(verifier, result),
                "verifier_result": compact_point(verifier, verifier.add_points(left, right, ainvs, p)),
                "matches_verifier_add": compact_point(verifier, result)
                == compact_point(verifier, verifier.add_points(left, right, ainvs, p)),
            }
        )
        return trace

    x1, y1 = [int(value) % p for value in left]
    x2, y2 = [int(value) % p for value in right]
    trace.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    if x1 == x2 and (y1 + y2 + a1 * x1 + a3) % p == 0:
        result = verifier.POINT_AT_INFINITY
        trace.update(
            {
                "operation": "vertical_inverse",
                "slope_numerator": None,
                "slope_denominator": 0,
                "slope_denominator_inverse": None,
                "slope": None,
                "intercept": None,
                "result": compact_point(verifier, result),
                "verifier_result": compact_point(verifier, verifier.add_points(left, right, ainvs, p)),
                "matches_verifier_add": compact_point(verifier, result)
                == compact_point(verifier, verifier.add_points(left, right, ainvs, p)),
            }
        )
        return trace

    if x1 == x2 and y1 == y2:
        operation = "doubling"
        slope_denominator = (2 * y1 + a1 * x1 + a3) % p
        if slope_denominator == 0:
            result = verifier.POINT_AT_INFINITY
            trace.update(
                {
                    "operation": "vertical_tangent",
                    "slope_numerator": None,
                    "slope_denominator": 0,
                    "slope_denominator_inverse": None,
                    "slope": None,
                    "intercept": None,
                    "result": compact_point(verifier, result),
                    "verifier_result": compact_point(verifier, verifier.add_points(left, right, ainvs, p)),
                    "matches_verifier_add": compact_point(verifier, result)
                    == compact_point(verifier, verifier.add_points(left, right, ainvs, p)),
                }
            )
            return trace
        slope_numerator = (3 * x1 * x1 + 2 * a2 * x1 + a4 - a1 * y1) % p
    else:
        operation = "addition"
        slope_denominator = (x2 - x1) % p
        slope_numerator = (y2 - y1) % p

    slope_denominator_inverse = inv_mod(slope_denominator, p)
    slope = (slope_numerator * slope_denominator_inverse) % p
    intercept = (y1 - slope * x1) % p
    x3 = (slope * slope + a1 * slope - a2 - x1 - x2) % p
    y3 = (-(slope + a1) * x3 - intercept - a3) % p
    result = (x3, y3)
    verifier_result = verifier.add_points(left, right, ainvs, p)
    trace.update(
        {
            "operation": operation,
            "slope_numerator": slope_numerator,
            "slope_denominator": slope_denominator,
            "slope_denominator_inverse": slope_denominator_inverse,
            "slope": slope,
            "intercept": intercept,
            "x3_formula": "(m^2 + a1*m - a2 - x1 - x2) mod p",
            "y3_formula": "(-(m + a1)*x3 - b - a3) mod p",
            "result": compact_point(verifier, result),
            "verifier_result": compact_point(verifier, verifier_result),
            "matches_verifier_add": compact_point(verifier, result) == compact_point(verifier, verifier_result),
        }
    )
    return trace


def abi_records_by_source(abi_source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("source_name") or ""): record
        for record in abi_source.get("records") or []
        if isinstance(record, dict)
    }


def abi_points_by_event(abi_record: dict[str, Any] | None) -> dict[tuple[str, int, int], Any]:
    if not isinstance(abi_record, dict):
        return {}
    points = {}
    for item in abi_record.get("candidate_point_curve_checks") or []:
        if not isinstance(item, dict):
            continue
        points[compact_event_key(item.get("event_key"))] = item.get("point")
    return points


def trace_groups_for_record(
    verifier: Any,
    records: list[dict[str, Any]],
    worklist_record: dict[str, Any],
    charged_record: dict[str, Any] | None,
    abi_record: dict[str, Any] | None,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
    bundle_cache: dict[tuple[Any, ...], dict[str, Any]],
    event_summary_limit: int,
) -> dict[str, Any]:
    source_name = str(worklist_record.get("source_name") or "")
    failures: list[dict[str, Any]] = []
    if charged_record is None:
        return {
            "source_name": source_name,
            "affine_trace_status": "fused_kernel_affine_trace_failed_check",
            "failures": [{"code": "missing_charged_locator_record"}],
        }
    source_path = scanner_probe.resolve_path(charged_record.get("source_path"))
    if not source_path.exists():
        return {
            "source_name": source_name,
            "affine_trace_status": "fused_kernel_affine_trace_failed_check",
            "source_path": str(source_path),
            "failures": [{"code": "missing_decomposition_source"}],
        }

    candidate_artifact, params, _decomposition = scanner_probe.candidate_source_parameters(source_path)
    if not params:
        return {
            "source_name": source_name,
            "affine_trace_status": "fused_kernel_affine_trace_failed_check",
            "source_path": str(source_path),
            "candidate_artifact": str(candidate_artifact) if candidate_artifact is not None else None,
            "failures": [{"code": "missing_candidate_source_parameters"}],
        }

    row_leaves = scanner_probe.row_leaves_from_worklist(worklist_record)
    allowed_by_row, _expected_instances = scanner_probe.expected_worklist_instances(worklist_record)
    public_group = worklist_record.get("public_group_key") or {}
    case = {
        "target": public_group.get("target"),
        "transfer_index": as_int(public_group.get("transfer_index")),
        "top_k": as_int(public_group.get("top_k")),
    }
    bundle = scanner_probe.build_input_bundle(params, event_summary_limit, bundle_cache)
    contexts, errors = replay_probe.materialize_contexts(
        verifier,
        records,
        bundle["config_source"],
        bundle["specs_by_target"],
        case,
        sorted(row_leaves),
        bundle["args"],
        context_cache,
    )
    for error in errors:
        add_failure(failures, "context_materialization_error", error)

    row_states = {
        row_key: candidate_point_probe.prepare_row_state(
            row_key,
            row_leaves[row_key],
            allowed_by_row.get(row_key, set()),
            context,
        )
        for row_key, context in contexts.items()
        if row_key in row_leaves
    }
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for state in row_states.values():
        for event_key, instances in state["keyed_instances"].items():
            groups[event_key].extend(instances)

    expected_points = abi_points_by_event(abi_record)
    trace_groups = []
    matching_abi_outputs = 0
    verifier_add_matches = 0
    row_instance_trace_count = 0
    reused_group_count = 0

    for event_key in sorted(groups, key=scanner_probe.event_sort_key):
        ordered_instances = sorted_group_instances(groups[event_key])
        if not ordered_instances:
            continue
        representative = ordered_instances[0]
        representative_state = row_states[representative["row_key"]]
        representative_built = representative_state["built"]
        p = int(representative_built["p"])
        ainvs = [int(value) for value in representative_built["ainvs"]]
        expected_point = expected_points.get(event_key)
        row_traces = []
        local_outputs = []

        for instance in ordered_instances:
            state = row_states[instance["row_key"]]
            built = state["built"]
            trace = affine_add_trace(
                verifier,
                instance["scout"]["left"]["point"],
                instance["scout"]["right"]["point"],
                [int(value) for value in built["ainvs"]],
                int(built["p"]),
            )
            local_outputs.append(trace.get("result"))
            row_instance_trace_count += 1
            if bool(trace.get("matches_verifier_add")):
                verifier_add_matches += 1
            row_traces.append(
                {
                    "row_key": instance["row_key"],
                    "scheduled_trial": int(instance["scheduled_trial"]),
                    "candidate_pos": int(instance["candidate_pos"]),
                    "scout_pos": int(instance["scout_pos"]),
                    "original_trial": int(instance["original_trial"]),
                    "leaf_index": int(instance["leaf_index"]),
                    "unsigned_indices": [int(value) for value in instance["scout"]["unsigned_indices"]],
                    "affine_trace": trace,
                }
            )

        representative_trace = affine_add_trace(
            verifier,
            representative["scout"]["left"]["point"],
            representative["scout"]["right"]["point"],
            ainvs,
            p,
        )
        representative_output = representative_trace.get("result")
        local_consistent = all(output == representative_output for output in local_outputs)
        matches_abi = representative_output == expected_point
        if matches_abi:
            matching_abi_outputs += 1
        if len(ordered_instances) > 1:
            reused_group_count += 1
        if not local_consistent:
            add_failure(
                failures,
                "local_affine_outputs_not_consistent",
                {"event_key": list(event_key), "outputs": local_outputs},
            )
        if not matches_abi:
            add_failure(
                failures,
                "representative_affine_output_mismatch",
                {
                    "event_key": list(event_key),
                    "representative_output": representative_output,
                    "abi_candidate_point": expected_point,
                },
            )
        if not all(bool(row.get("affine_trace", {}).get("matches_verifier_add")) for row in row_traces):
            add_failure(failures, "affine_formula_mismatch", {"event_key": list(event_key)})

        trace_groups.append(
            {
                "event_key": list(event_key),
                "fanout": len(ordered_instances),
                "candidate_point_reused": len(ordered_instances) > 1,
                "representative_row_key": representative["row_key"],
                "representative_scheduled_trial": int(representative["scheduled_trial"]),
                "representative_candidate_point": representative_output,
                "abi_candidate_point": expected_point,
                "matches_abi_candidate_point": matches_abi,
                "local_affine_outputs_consistent": local_consistent,
                "row_instance_trace_count": len(row_traces),
                "representative_affine_trace": representative_trace,
                "row_instance_traces": row_traces,
            }
        )

    abi_checks = (abi_record or {}).get("checks") or {}
    if as_int(abi_checks.get("first_pass_group_count"), -1) != len(trace_groups):
        add_failure(
            failures,
            "first_pass_group_count_mismatch",
            {
                "trace_group_count": len(trace_groups),
                "abi_first_pass_group_count": abi_checks.get("first_pass_group_count"),
            },
        )
    if as_int(abi_checks.get("first_pass_group_instance_count"), -1) != row_instance_trace_count:
        add_failure(
            failures,
            "row_instance_trace_count_mismatch",
            {
                "row_instance_trace_count": row_instance_trace_count,
                "abi_first_pass_group_instance_count": abi_checks.get("first_pass_group_instance_count"),
            },
        )
    if not abi_record:
        add_failure(failures, "missing_abi_record", {"source_name": source_name})
    elif abi_record.get("abi_curve_status") != "fused_kernel_abi_curve_verified":
        add_failure(
            failures,
            "abi_record_not_verified",
            {"abi_curve_status": abi_record.get("abi_curve_status")},
        )

    return {
        "source_name": source_name,
        "public_group_key": public_group,
        "source_path": str(source_path),
        "candidate_artifact": str(candidate_artifact) if candidate_artifact is not None else None,
        "selected_row_keys": sorted(row_leaves),
        "affine_trace_status": "fused_kernel_affine_trace_verified" if not failures else "fused_kernel_affine_trace_failed_check",
        "context_error_count": len(errors),
        "context_errors": errors,
        "trace_groups": trace_groups,
        "checks": {
            "first_pass_group_count": len(trace_groups),
            "row_instance_trace_count": row_instance_trace_count,
            "reused_group_count": reused_group_count,
            "matching_abi_candidate_point_count": matching_abi_outputs,
            "verifier_add_match_count": verifier_add_matches,
            "all_affine_formulas_match_verifier": verifier_add_matches == row_instance_trace_count,
            "all_representative_outputs_match_abi": matching_abi_outputs == len(trace_groups),
            "matches_abi_first_pass_group_count": as_int(abi_checks.get("first_pass_group_count"), -1)
            == len(trace_groups),
            "matches_abi_row_instance_count": as_int(abi_checks.get("first_pass_group_instance_count"), -1)
            == row_instance_trace_count,
            "matches_contract_below_rho": bool((abi_record or {}).get("checks", {}).get("matches_contract_below_rho")),
            "matches_contract_event_reuse_target": bool(
                (abi_record or {}).get("checks", {}).get("matches_contract_event_reuse_target")
            ),
        },
        "failures": failures,
        "implementation_boundary": (
            "Affine field-operation trace for first-pass candidate-point output. "
            "A lower-level FFE kernel should produce these same outputs before "
            "the second-pass relation predicates run."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if record.get("affine_trace_status") == "fused_kernel_affine_trace_verified"]
    below = [
        record for record in verified if bool((record.get("checks") or {}).get("matches_contract_below_rho"))
    ]
    event_reuse = [
        record for record in below if bool((record.get("checks") or {}).get("matches_contract_event_reuse_target"))
    ]
    return {
        "record_count": len(records),
        "affine_trace_verified_count": len(verified),
        "below_rho_affine_trace_verified_count": len(below),
        "event_reuse_below_rho_affine_trace_verified_count": len(event_reuse),
        "total_first_pass_groups": sum(as_int((record.get("checks") or {}).get("first_pass_group_count")) for record in records),
        "total_row_instance_traces": sum(as_int((record.get("checks") or {}).get("row_instance_trace_count")) for record in records),
        "total_reused_groups": sum(as_int((record.get("checks") or {}).get("reused_group_count")) for record in records),
        "total_matching_abi_candidate_points": sum(
            as_int((record.get("checks") or {}).get("matching_abi_candidate_point_count")) for record in records
        ),
        "total_verifier_add_matches": sum(as_int((record.get("checks") or {}).get("verifier_add_match_count")) for record in records),
        "interpretation": (
            "The first-pass candidate-point layer now has an explicit affine "
            "field-operation trace.  The trace outputs match the ABI artifact "
            "and the verifier add formula for every row instance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist-source", type=Path, default=DEFAULT_WORKLIST_SOURCE)
    parser.add_argument("--charged-source", type=Path, default=DEFAULT_CHARGED_SOURCE)
    parser.add_argument("--abi-source", type=Path, default=DEFAULT_ABI_SOURCE)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--event-summary-limit", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    worklist_source = load_json(args.worklist_source)
    charged_source = load_json(args.charged_source)
    abi_source = load_json(args.abi_source)
    charged_by_source = scanner_probe.charged_records_by_source(charged_source)
    abi_by_source = abi_records_by_source(abi_source)
    wanted = set(args.source_name or [])
    verifier = replay_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    bundle_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    trace_records = []
    for record in worklist_source.get("records") or []:
        if not isinstance(record, dict):
            continue
        source_name = str(record.get("source_name") or "")
        if wanted and source_name not in wanted:
            continue
        trace_records.append(
            trace_groups_for_record(
                verifier,
                records,
                record,
                charged_by_source.get(source_name),
                abi_by_source.get(source_name),
                context_cache,
                bundle_cache,
                int(args.event_summary_limit),
            )
        )

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_affine_trace_probe_v1",
        "method": "affine_weierstrass_field_trace_for_first_pass_candidate_points",
        "parameters": {
            "worklist_source": str(args.worklist_source),
            "charged_source": str(args.charged_source),
            "abi_source": str(args.abi_source),
            "source_names": sorted(wanted),
            "event_summary_limit": int(args.event_summary_limit),
            "campaign_task_dir": str(CAMPAIGN_TASK_DIR),
        },
        "summary": summarize(trace_records),
        "records": sorted(
            trace_records,
            key=lambda record: (
                record.get("affine_trace_status") != "fused_kernel_affine_trace_verified",
                not bool((record.get("checks") or {}).get("matches_contract_event_reuse_target")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This is a first-pass candidate-point field trace, not a new relation search.",
            "It does not replace the relation predicate or linear algebra layers.",
            "A native FFE/summation-polynomial kernel still has to emit the same trace without Python verifier arithmetic.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
