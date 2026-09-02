#!/usr/bin/env python3
"""Reduce the R117 nonlinear C6 exception to a compact C5 source index."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.m6_nonlinear_value_sensitive_c6_source_locator.r118.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
RHO_EXPONENT = Fraction(5, 2)
GROUP_ORDER_EXPONENT = Fraction(5)
A6_BATCH_EXPONENT = Fraction(1, 2)
C_ATOM_EXPONENT = Fraction(3, 4)
C3_EXPONENT = 3 * C_ATOM_EXPONENT
C5_EXPONENT = 5 * C_ATOM_EXPONENT
C6_EXPONENT = 6 * C_ATOM_EXPONENT

R117_PRODUCER = pathlib.Path(
    "p1553_m6_target_batched_c3_elliptic_transpose_probe_r117.py"
)
R117_PRODUCER_SHA256 = (
    "af3d7447763e16ee2a21491f3217d719bf57bdafc035d666df5e8f5859cc49f2"
)
R117_REPORT = pathlib.Path(
    "p1553_m6_target_batched_c3_elliptic_"
    "transpose_probe_report_r117.json"
)
R117_REPORT_SHA256 = (
    "449021dd58e6567bb827bcfe41336175d7b628acda4446e1b63d9a46cf48f824"
)
R117_FROZEN = pathlib.Path(
    "frozen_m6_target_batched_c3_elliptic_transpose.json"
)
R117_FROZEN_SHA256 = (
    "377b169ede2e1fcdab6875fbfa00cf12be4f8b5ab0b58cfe0457afb0d3e96f73"
)
R117_COST = pathlib.Path("m6_target_batched_c3_transpose_cost_ledger.json")
R117_COST_SHA256 = (
    "33ca990fe05e8e80bd4b557d57f4a292a84de0a5d6f1a836fa6bb54965cada8b"
)
R117_REPLAY = pathlib.Path("m6_target_batched_c3_source_adjoint_replay.json")
R117_REPLAY_SHA256 = (
    "6878a34e379ae515b9cc301d44441f5726a66df2d96074b01dfebe742f3ba4d1"
)
R117_CONTROLS = pathlib.Path("m6_target_batched_c3_exceptional_controls.json")
R117_CONTROLS_SHA256 = (
    "5b426b99955b1143484198c56b312ca12e9ffa15fe5a36c942f10e74aed6603e"
)
R117_LOGS = pathlib.Path("factor_logs_and_identical_descent_r117.json")
R117_LOGS_SHA256 = (
    "2fff1a05cc1f032dfb21c7ddb12ce1d72f4b851e6732e63d4c6d83f0b786fb64"
)
R117_GATE = pathlib.Path(
    "p1553_m6_target_batched_c3_elliptic_transpose_probe_gate_r117.md"
)
R117_GATE_SHA256 = (
    "014f74c772dfc77d8821f658b10ef9bd11a644fe8f0b356c24fda61ef81d9de5"
)
R117_PARENT = pathlib.Path(
    "p1553_m6_target_batched_c3_elliptic_"
    "transpose_probe_parent_report_r117.yaml"
)
R117_PARENT_SHA256 = (
    "8b276efff8f61f4278e78ab367ff4536c3bee8a1e172ed3ea26a12c1c12dbdae"
)
DINUR_GOLOVNEV = pathlib.Path(
    "references/dinur_golovnev_3sum_indexing_2512.04258v2.pdf"
)
DINUR_GOLOVNEV_SHA256 = (
    "e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c"
)
MULTIPOINT = pathlib.Path(
    "references/bhargava_ghosh_guo_kumar_umans_"
    "multipoint_2205.00342v1.pdf"
)
MULTIPOINT_SHA256 = (
    "14eddc304a7dd8995ebc1e24171571fd9dc0f1f837ca35a7f9e2e6fb21bfafa8"
)
TRUNCATED_RESULTANT = pathlib.Path(
    "references/moroz_schost_truncated_resultant_1609.04259v1.pdf"
)
TRUNCATED_RESULTANT_SHA256 = (
    "160c68cfbb413ca27352a064cbf2d27f7ad4ed6a210c3d6ead2770e00204b709"
)

Point = tuple[int, int] | None


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R117 = load_module("p1553_r117_for_r118", R117_PRODUCER)
R82 = R117.R82
R70 = R117.R70


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r117_producer", R117_PRODUCER, R117_PRODUCER_SHA256),
        ("r117_report", R117_REPORT, R117_REPORT_SHA256),
        ("r117_frozen", R117_FROZEN, R117_FROZEN_SHA256),
        ("r117_cost", R117_COST, R117_COST_SHA256),
        ("r117_replay", R117_REPLAY, R117_REPLAY_SHA256),
        ("r117_controls", R117_CONTROLS, R117_CONTROLS_SHA256),
        ("r117_logs", R117_LOGS, R117_LOGS_SHA256),
        ("r117_gate", R117_GATE, R117_GATE_SHA256),
        ("r117_parent", R117_PARENT, R117_PARENT_SHA256),
        ("dinur_golovnev_v2", DINUR_GOLOVNEV, DINUR_GOLOVNEV_SHA256),
        ("multipoint_v1", MULTIPOINT, MULTIPOINT_SHA256),
        (
            "truncated_resultant_v1",
            TRUNCATED_RESULTANT,
            TRUNCATED_RESULTANT_SHA256,
        ),
    )
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in rows
    }


def verify_source_bindings() -> dict[str, str]:
    bindings = source_binding_records()
    actual = {
        name: sha256_file(pathlib.Path(binding["path"]))
        for name, binding in bindings.items()
    }
    failures = [
        name
        for name, binding in bindings.items()
        if actual[name] != binding["sha256"]
    ]
    if failures:
        raise AssertionError(f"R118 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def add_many(points: Iterable[Point], curve: dict[str, Any]) -> Point:
    return R82.add_many(points, curve)


def replay_c_source(
    target: Point,
    source: Sequence[int],
    deck: Sequence[Point],
    curve: dict[str, Any],
) -> bool:
    return add_many((deck[index] for index in source), curve) == target


def first_empty_target(
    support: set[Point],
    curve: dict[str, Any],
) -> Point:
    generator = R117.R81.curve_generator(curve)
    for scalar in range(curve["subgroup_order"]):
        target = R70.scalar_mul(scalar, generator, curve)
        if target not in support:
            return target
    raise AssertionError("finite C6 support unexpectedly fills the group")


def branch_query(
    target: Point,
    deck: Sequence[Point],
    c5_counts: dict[Point, int],
    c5_first: dict[Point, tuple[int, ...]],
    curve: dict[str, Any],
) -> dict[str, Any]:
    count = 0
    source: tuple[int, ...] | None = None
    inspected_branches = 0
    positive_branches = 0
    for atom_index, atom in enumerate(deck):
        inspected_branches += 1
        complement = R70.add(target, R70.negate(atom, curve), curve)
        suffix_count = c5_counts.get(complement, 0)
        count += suffix_count
        if suffix_count:
            positive_branches += 1
            if source is None:
                source = (atom_index, *c5_first[complement])
    return {
        "target": point_json(target),
        "branch_count": count,
        "inspected_first_atom_branches": inspected_branches,
        "positive_first_atom_branches": positive_branches,
        "source": None if source is None else list(source),
        "source_replay_exact": (
            source is None
            or replay_c_source(target, source, deck, curve)
        ),
    }


def split_count(
    target: Point,
    left_counts: dict[Point, int],
    right_counts: dict[Point, int],
    curve: dict[str, Any],
) -> int:
    return sum(
        left_count
        * right_counts.get(
            R70.add(target, R70.negate(endpoint, curve), curve),
            0,
        )
        for endpoint, left_count in left_counts.items()
    )


def finite_branch_control(deck_size: int) -> dict[str, Any]:
    curve = R82.FAMILIES[0]
    generator = R117.R81.curve_generator(curve)
    validation = R82.validate_family(curve, generator)
    if not all(validation.values()):
        raise AssertionError("R118 finite curve validation failed")
    _, atoms_c, _, construction = R82.compact_factor_base(curve, 0)
    deck = atoms_c[:deck_size]
    endpoint_maps = {
        arity: R82.ordered_endpoint_map(deck, arity, curve)
        for arity in range(1, 7)
    }
    c5_counts, c5_first = endpoint_maps[5]
    c6_counts, c6_first = endpoint_maps[6]
    repeated_target = add_many([deck[0]] * 6, curve)
    positive_target = min(
        c6_counts,
        key=lambda point: (
            point is None,
            () if point is None else point,
        ),
    )
    empty_target = first_empty_target(set(c6_counts), curve)
    target_rows = []
    for kind, target in (
        ("repeated_atom_positive", repeated_target),
        ("positive", positive_target),
        ("empty", empty_target),
        ("identity", None),
    ):
        branch = branch_query(
            target,
            deck,
            c5_counts,
            c5_first,
            curve,
        )
        direct_count = c6_counts.get(target, 0)
        direct_source = c6_first.get(target)
        split_rows = []
        for left_arity in range(1, 6):
            left_counts, _ = endpoint_maps[left_arity]
            right_counts, _ = endpoint_maps[6 - left_arity]
            split_rows.append(
                {
                    "left_arity": left_arity,
                    "right_arity": 6 - left_arity,
                    "count": split_count(
                        target,
                        left_counts,
                        right_counts,
                        curve,
                    ),
                    "matches_direct": (
                        split_count(
                            target,
                            left_counts,
                            right_counts,
                            curve,
                        )
                        == direct_count
                    ),
                }
            )
        if branch["branch_count"] != direct_count:
            raise AssertionError("one-C plus C5 branch count drifted")
        if not all(row["matches_direct"] for row in split_rows):
            raise AssertionError("finite split convolution count drifted")
        if direct_source is not None and not replay_c_source(
            target,
            direct_source,
            deck,
            curve,
        ):
            raise AssertionError("direct C6 source failed replay")
        target_rows.append(
            {
                "kind": kind,
                "direct_c6_count": direct_count,
                "direct_source": (
                    None
                    if direct_source is None
                    else list(direct_source)
                ),
                "direct_source_replay_exact": (
                    direct_source is None
                    or replay_c_source(
                        target,
                        direct_source,
                        deck,
                        curve,
                    )
                ),
                "one_c_plus_c5": branch,
                "all_binary_split_counts": split_rows,
                "positive": direct_count > 0,
            }
        )
    return {
        "control_id": f"r82_q{curve['subgroup_order']}_c{deck_size}",
        "curve": {
            "field_prime": curve["field_prime"],
            "subgroup_order": curve["subgroup_order"],
            "cofactor": curve["cofactor"],
        },
        "validation": validation,
        "factor_base_construction": construction["construction"],
        "deck": [point_json(point) for point in deck],
        "deck_size": deck_size,
        "ordered_endpoint_counts": {
            f"c{arity}": sum(endpoint_maps[arity][0].values())
            for arity in range(1, 7)
        },
        "endpoint_support_sizes": {
            f"c{arity}": len(endpoint_maps[arity][0])
            for arity in range(1, 7)
        },
        "targets": target_rows,
        "one_c_plus_c5_count_exact_on_all_targets": all(
            row["one_c_plus_c5"]["branch_count"]
            == row["direct_c6_count"]
            for row in target_rows
        ),
        "all_split_counts_exact_on_all_targets": all(
            split["matches_direct"]
            for row in target_rows
            for split in row["all_binary_split_counts"]
        ),
        "all_reported_sources_replay": all(
            row["direct_source_replay_exact"]
            and row["one_c_plus_c5"]["source_replay_exact"]
            for row in target_rows
        ),
        "positive_repeated_empty_and_identity_controls_present": (
            {row["kind"] for row in target_rows}
            == {
                "repeated_atom_positive",
                "positive",
                "empty",
                "identity",
            }
            and any(row["positive"] for row in target_rows)
            and any(not row["positive"] for row in target_rows)
        ),
        "finite_enumeration_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        finite_branch_control(2),
        finite_branch_control(3),
    ]
    return {
        "schema": "p1553.m6_nonlinear_c6_exceptional_controls.r118.v1",
        "controls": controls,
        "all_one_c_plus_c5_counts_exact": all(
            row["one_c_plus_c5_count_exact_on_all_targets"]
            for row in controls
        ),
        "all_binary_split_counts_exact": all(
            row["all_split_counts_exact_on_all_targets"]
            for row in controls
        ),
        "all_sources_replay": all(
            row["all_reported_sources_replay"] for row in controls
        ),
        "positive_repeated_empty_and_identity_controls_present": all(
            row["positive_repeated_empty_and_identity_controls_present"]
            for row in controls
        ),
        "finite_enumeration_receives_asymptotic_credit": False,
    }


def split_table_ledger() -> list[dict[str, Any]]:
    rows = []
    for stored_arity in range(7):
        enumerated_arity = 6 - stored_arity
        setup = stored_arity * C_ATOM_EXPONENT
        query = enumerated_arity * C_ATOM_EXPONENT
        batch = A6_BATCH_EXPONENT + query
        rows.append(
            {
                "stored_c_arity": stored_arity,
                "enumerated_c_arity": enumerated_arity,
                "setup_exponent_B": fraction_record(setup),
                "per_target_query_exponent_B": fraction_record(query),
                "a6_batch_query_exponent_B": fraction_record(batch),
                "inside_setup_cap": setup <= SETUP_CAP,
                "inside_batch_cap": batch <= ONLINE_CAP,
                "meets_both_caps": (
                    setup <= SETUP_CAP and batch <= ONLINE_CAP
                ),
            }
        )
    return rows


def branch_and_ffe_cost_ledger() -> dict[str, Any]:
    split_rows = split_table_ledger()
    k6_delta = Fraction(0)
    k6_state = C_ATOM_EXPONENT * (
        Fraction(11, 2) - k6_delta
    )
    outer_branch = A6_BATCH_EXPONENT + C_ATOM_EXPONENT
    c2_scan = 2 * C_ATOM_EXPONENT
    rows = [
        {
            "route_id": "one_c_branch_plus_materialized_c5_index",
            "setup_exponent_B": fraction_record(C5_EXPONENT),
            "outer_query_batch_exponent_B": fraction_record(outer_branch),
            "inside_setup_cap": C5_EXPONENT <= SETUP_CAP,
            "inside_batch_cap_if_lookup_constant": (
                outer_branch <= ONLINE_CAP
            ),
            "exact_source_reporting": True,
        },
        {
            "route_id": "one_c_branch_plus_dinur_golovnev_k6_index",
            "delta": fraction_record(k6_delta),
            "setup_exponent_B": fraction_record(k6_state),
            "five_sum_query_exponent_B": fraction_record(Fraction(0)),
            "outer_query_batch_exponent_B": fraction_record(outer_branch),
            "inside_setup_cap": k6_state <= SETUP_CAP,
            "inside_batch_cap": outer_branch <= ONLINE_CAP,
            "integer_residue_transfer_to_prime_order_ec": False,
        },
        {
            "route_id": "one_c_branch_plus_c2_scan_against_c3_hash",
            "setup_exponent_B": fraction_record(C3_EXPONENT),
            "five_sum_query_exponent_B": fraction_record(c2_scan),
            "outer_query_batch_exponent_B": fraction_record(
                outer_branch + c2_scan
            ),
            "inside_setup_cap": True,
            "inside_batch_cap": (
                outer_branch + c2_scan <= ONLINE_CAP
            ),
            "exact_source_reporting": True,
        },
        {
            "route_id": "five_variable_deck_quotient_or_grid",
            "represented_dimension_exponent_B": fraction_record(
                C5_EXPONENT
            ),
            "inside_setup_cap": C5_EXPONENT <= SETUP_CAP,
            "inside_batch_cap": C5_EXPONENT <= ONLINE_CAP,
            "summation_polynomial": "S6(x1,x2,x3,x4,x5,x_target)",
            "degree_per_source_variable": 16,
            "x_only_s6_fixed_sign_source_biconditional": False,
            "fixed_sign_point_source_verification_required": True,
            "maximum_sign_branches": 32,
            "deck_domain_polynomial_degree_exponent_B": fraction_record(
                C_ATOM_EXPONENT
            ),
            "tensor_quotient_dimension": "dim(A_C^tensor5)=|C|^5",
        },
        {
            "route_id": "all_grid_multivariate_multipoint",
            "represented_coefficient_or_output_exponent_B": fraction_record(
                C5_EXPONENT
            ),
            "inside_caps": False,
            "theorem": (
                "(d^m+N)^(1+o(1))*poly(m,d,log|F|)"
            ),
            "source_reporting_without_all_outputs_supplied": False,
        },
        {
            "route_id": "constant_order_truncated_c2_c3_resultant_per_c5_query",
            "explicit_degree_body_exponent_B": fraction_record(C3_EXPONENT),
            "outer_query_batch_exponent_B": fraction_record(
                outer_branch + C3_EXPONENT
            ),
            "inside_caps": False,
            "theorem": "soft-O(kd) for order-k truncation and degree d",
            "source_cofactor_or_gcd_unit_cost": False,
        },
        {
            "route_id": "r117_direct_c3_translated_divisor_batch",
            "setup_exponent_B": fraction_record(C3_EXPONENT),
            "a6_batch_query_exponent_B": fraction_record(
                A6_BATCH_EXPONENT + C3_EXPONENT
            ),
            "inside_setup_cap": True,
            "inside_batch_cap": False,
        },
    ]
    return {
        "schema": (
            "p1553.m6_nonlinear_c6_branch_and_ffe_cost_ledger.r118.v1"
        ),
        "normalization": {
            "group_order": "q=B^(5+o(1)) prime",
            "a6_target_batch": "B^(1/2+o(1))",
            "c_atom_deck": "|C|=B^(3/4+o(1))",
            "c3_persistent_state": "B^(9/4+o(1))",
            "setup_cap": "B^(9/4+o(1))",
            "fresh_batch_cap": "B^(5/4+o(1))",
        },
        "explicit_occurrence_split_table_theorem": {
            "rows": split_rows,
            "any_row_meets_both_caps": any(
                row["meets_both_caps"] for row in split_rows
            ),
            "best_setup_compatible_stored_arity": 3,
            "best_setup_compatible_batch_exponent_B": fraction_record(
                A6_BATCH_EXPONENT + 3 * C_ATOM_EXPONENT
            ),
            "minimum_stored_arity_for_batch_cap": 5,
            "minimum_online_compatible_setup_exponent_B": fraction_record(
                5 * C_ATOM_EXPONENT
            ),
            "proof": (
                "For occurrence-materialized C^s tables, setup permits "
                "stored arity s<=3. Then the complementary enumerated "
                "arity 6-s>=3 makes the A6 batch exponent at least "
                "1/2+9/4=11/4. Conversely the batch cap requires 6-s<=1, "
                "hence s>=5 and occurrence state at least 15/4."
            ),
            "arbitrary_endpoint_compressed_index_lower_bound_claimed": False,
        },
        "one_atom_branch_reduction": {
            "outer_query_count_exponent_B": fraction_record(outer_branch),
            "outer_query_count_equals_batch_cap": (
                outer_branch == ONLINE_CAP
            ),
            "required_c5_query_exponent_B": fraction_record(Fraction(0)),
            "required_interface": (
                "target-dependent C5 membership plus one five-C source "
                "in polylogarithmic field work"
            ),
            "negative_queries_require_exact_rejection": True,
            "positive_source_can_be_dyadically_recovered_if_predicate_exists": (
                True
            ),
        },
        "dinur_golovnev_k6": {
            "k": 6,
            "n": "|C|=B^(3/4+o(1))",
            "theorem": (
                "S=soft-O(n^(k-1/2-delta)), T=soft-O(n^delta), "
                "0<=delta<=1"
            ),
            "zero_online_slack_forces_delta": fraction_record(k6_delta),
            "state_exponent_B": fraction_record(k6_state),
            "source": str(DINUR_GOLOVNEV),
            "sha256": DINUR_GOLOVNEV_SHA256,
        },
        "literature_controls": {
            "multipoint": {
                "source": str(MULTIPOINT),
                "sha256": MULTIPOINT_SHA256,
                "scope": (
                    "upper bound for represented all-output evaluation, "
                    "not an output-sensitive lower bound"
                ),
            },
            "truncated_resultant": {
                "source": str(TRUNCATED_RESULTANT),
                "sha256": TRUNCATED_RESULTANT_SHA256,
                "scope": (
                    "upper bound for explicit coefficient bodies, not an "
                    "arithmetic-circuit lower bound"
                ),
            },
        },
        "routes": rows,
        "any_scoped_route_meets_both_caps": any(
            row.get("inside_setup_cap", False)
            and (
                row.get("inside_batch_cap", False)
                or row.get("inside_batch_cap_if_lookup_constant", False)
            )
            for row in rows
        ),
        "output_sensitive_nonlinear_c5_index_lower_bound_claimed": False,
        "general_arithmetic_circuit_or_data_structure_lower_bound_claimed": (
            False
        ),
        "candidate_work_credit": False,
    }


def source_replay(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "p1553.m6_nonlinear_c6_source_replay.r118.v1",
        "requested_source": (
            "one distinguished C atom plus one ordered five-C backpointer"
        ),
        "finite_control_count": len(controls["controls"]),
        "one_c_plus_c5_count_exact": controls[
            "all_one_c_plus_c5_counts_exact"
        ],
        "all_binary_split_counts_exact": controls[
            "all_binary_split_counts_exact"
        ],
        "positive_repeated_empty_and_identity_controls_present": controls[
            "positive_repeated_empty_and_identity_controls_present"
        ],
        "all_finite_sources_replay": controls["all_sources_replay"],
        "inside_cap_c5_membership_predicate_constructed": False,
        "inside_cap_c5_source_index_constructed": False,
        "finite_enumeration_receives_asymptotic_credit": False,
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R117_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R117 nonclaim boundary drifted")
    controls = finite_controls()
    cost = branch_and_ffe_cost_ledger()
    replay = source_replay(controls)
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r117_scoped_boundary_inherited": (
            inherited["admission"]["scoped_negative_admitted"]
            and not inherited["admission"]["lane_admitted"]
        ),
        "explicit_occurrence_split_table_cap_theorem_complete": (
            not cost["explicit_occurrence_split_table_theorem"][
                "any_row_meets_both_caps"
            ]
        ),
        "one_atom_branch_reduction_complete": cost[
            "one_atom_branch_reduction"
        ]["outer_query_count_equals_batch_cap"],
        "finite_one_c_plus_c5_counts_exact": controls[
            "all_one_c_plus_c5_counts_exact"
        ],
        "finite_all_binary_split_counts_exact": controls[
            "all_binary_split_counts_exact"
        ],
        "finite_positive_repeated_empty_identity_controls_complete": controls[
            "positive_repeated_empty_and_identity_controls_present"
        ],
        "finite_c6_sources_replay": controls["all_sources_replay"],
        "current_k6_and_standard_ffe_costs_charged": (
            cost["dinur_golovnev_k6"]["state_exponent_B"]["exact"]
            == "33/8"
        ),
        "scope_excludes_output_sensitive_and_general_lower_bounds": (
            not cost[
                "output_sensitive_nonlinear_c5_index_lower_bound_claimed"
            ]
            and not cost[
                "general_arithmetic_circuit_or_data_structure_lower_bound_claimed"
            ]
        ),
        "inside_cap_nonlinear_c5_membership_predicate_complete": False,
        "inside_cap_c5_source_index_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one output-sensitive nonlinear C5 membership "
        "and source index over the original scalar-blind C deck. It may "
        "store B^(9/4+o(1)) C3 state but must answer each arbitrary "
        "five-C target in polylogarithmic field work, reject empty targets "
        "exactly, and return five occurrence backpointers. Freeze the S6 "
        "and deck-domain representation, quotient/remainder dimensions, "
        "target-dependent branches, exceptional projective charts, and "
        "reverse source path. Then compose with the one-C branch and A6 "
        "batch before rank, logs, identical descent, memory, and full cost."
    )
    frozen = {
        "schema": "p1553.frozen_m6_nonlinear_c6_source_locator.r118.v1",
        "source_bindings": source_binding_records(),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "fresh_batch_exponent_B": fraction_record(ONLINE_CAP),
        },
        "exact_reduction": {
            "identity": (
                "mu_C^*6(T)=sum_(c in C) mu_C^*5(T-c)"
            ),
            "a6_target_count_exponent_B": fraction_record(
                A6_BATCH_EXPONENT
            ),
            "first_c_branch_exponent_B": fraction_record(C_ATOM_EXPONENT),
            "combined_outer_branch_exponent_B": fraction_record(
                A6_BATCH_EXPONENT + C_ATOM_EXPONENT
            ),
            "remaining_query_slack_exponent_B": fraction_record(
                Fraction(0)
            ),
        },
        "closed_scoped_grammars": [
            "every occurrence-materialized one-side C^s sum-table split",
            "current bound k=6 indexing theorem at zero query exponent",
            "explicit C2 scan against the C3 hash table",
            "represented five-variable deck quotient and all-grid evaluation",
            "explicit constant-order C2/C3 truncated resultant per C5 query",
        ],
        "preserved_interface": (
            "output-sensitive nonlinear C5 membership/source index with "
            "B^(9/4) state and polylogarithmic exact query"
        ),
        "general_lower_bound_claimed": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r118.v1",
        "r117_scoped_transpose_audit_complete": True,
        "r118_one_atom_branch_reduction_complete": True,
        "inside_cap_nonlinear_c5_source_index_complete": False,
        "relation_independence_theorem_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_log_solve_complete": False,
        "factor_log_verification_complete": False,
        "fresh_target_descent_complete": False,
        "identical_algorithm_used_for_relation_and_descent": False,
        "full_source_to_target_cost_complete": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_BRANCH_REDUCTION_AND_SCOPED_STANDARD_ROUTE_NEGATIVE_"
            "ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "EXACT_ONE_C_BRANCH_REDUCES_ALLOWED_BATCH_TO_CONSTANT_QUERY_C5_"
            "SOURCE_INDEX__NO_EXPLICIT_OCCURRENCE_SPLIT_TABLE_MEETS_B9O4_SETUP_AND_"
            "B5O4_BATCH_CAPS__SETUP_COMPATIBLE_C3_SPLIT_BATCH_B11O4__"
            "ONLINE_COMPATIBLE_C5_TABLE_B15O4__DINUR_GOLOVNEV_K6_ZERO_"
            "SLACK_STATE_B33O8__STANDARD_C5_QUOTIENT_AND_GRID_B15O4__"
            "FINITE_PROJECTIVE_POSITIVE_REPEATED_EMPTY_BRANCH_REPLAY_EXACT__"
            "OUTPUT_SENSITIVE_NONLINEAR_C5_MEMBERSHIP_FFE_SOURCE_INDEX_OPEN__"
            "NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "source_bindings": source_binding_records(),
        "exact_reduction": frozen["exact_reduction"],
        "finite_evidence": {
            "control_count": len(controls["controls"]),
            "all_one_c_plus_c5_counts_exact": controls[
                "all_one_c_plus_c5_counts_exact"
            ],
            "all_binary_split_counts_exact": controls[
                "all_binary_split_counts_exact"
            ],
            "all_sources_replay": controls["all_sources_replay"],
            "positive_repeated_empty_and_identity_controls_present": controls[
                "positive_repeated_empty_and_identity_controls_present"
            ],
            "asymptotic_credit": False,
        },
        "cost_boundary": {
            "explicit_occurrence_split_table": cost[
                "explicit_occurrence_split_table_theorem"
            ],
            "one_atom_branch": cost["one_atom_branch_reduction"],
            "k6_indexing": cost["dinur_golovnev_k6"],
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "exact_reduction_admitted": True,
            "scoped_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_m6_nonlinear_c6_source_locator.json",
            "cost": "m6_nonlinear_c6_branch_and_ffe_cost_ledger.json",
            "source_replay": "m6_nonlinear_c6_source_replay.json",
            "exceptional_controls": (
                "m6_nonlinear_c6_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r118.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Finite enumeration receives no asymptotic credit.",
            "The split theorem applies only to occurrence-materialized one-side tables.",
            "No arbitrary endpoint-compressed C5 index lower bound is claimed.",
            "X-only S6 requires fixed-sign point-source verification.",
            "Multipoint and truncated-resultant controls are represented upper bounds.",
            "No output-sensitive C5 data-structure lower bound is claimed.",
            "No nonlinear C5 predicate, relation rank, logs, or descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_EXACT_ONE_C_PLUS_C5_BRANCH_AND_FINITE_SOURCE_REPLAY_ONLY__"
            "REJECT_ALL_EXPLICIT_OCCURRENCE_SPLIT_TABLES_CURRENT_K6_INDEX_AND_STANDARD_"
            "REPRESENTED_FFE_RESULTANT_ROUTES_AT_FROZEN_CAPS__PRESERVE_"
            "OUTPUT_SENSITIVE_NONLINEAR_C5_MEMBERSHIP_SOURCE_INDEX__NO_"
            "LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__"
            "NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_m6_nonlinear_value_sensitive_c6_"
            "source_locator_probe_report_r118.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_m6_nonlinear_c6_source_locator.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_nonlinear_c6_branch_and_ffe_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path("m6_nonlinear_c6_source_replay.json"),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_nonlinear_c6_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r118.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs_descent"])
    report = bundle["report"]
    admission = report["admission"]
    print(
        f"R118 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
