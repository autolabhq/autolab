#!/usr/bin/env python3
"""Audit rational C2 selectors for cap-tight torus C5 routing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
from collections import Counter
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_rational_selector_degree.r128.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
C2_EXPONENT = Fraction(3, 2)
C3_EXPONENT = Fraction(9, 4)
C5_EXPONENT = Fraction(15, 4)

R127_PRODUCER = pathlib.Path(
    "p1553_torus_c5_bucket_resultant_routing_tradeoff_probe_r127.py"
)
R127_PRODUCER_SHA256 = (
    "4d4d88bbd2f0f98f1eca9f2986d6ebbb50f5fef7a3af9cc8a5a9c4cd9e802a7b"
)
R127_REPORT = pathlib.Path(
    "p1553_torus_c5_bucket_resultant_routing_"
    "tradeoff_probe_report_r127.json"
)
R127_REPORT_SHA256 = (
    "373a69c5a6a0dcdb4b12f68d1a1bd71b74f4c8e1971329dba64f103fe4ae138f"
)
R127_FROZEN = pathlib.Path(
    "frozen_torus_c5_bucket_resultant_routing_tradeoff.json"
)
R127_FROZEN_SHA256 = (
    "c91e94d84a12e920d0b59619b730a9e81b4ff2aa9bc8fb46014a2f39594fad4a"
)
R127_COST = pathlib.Path(
    "torus_c5_bucket_resultant_routing_tradeoff_cost_ledger.json"
)
R127_COST_SHA256 = (
    "2c1cf3b67084f01265dfde52c9c32e328bb7e30be06385f4752af8f8b5d9d45f"
)
R127_REPLAY = pathlib.Path(
    "torus_c5_bucket_resultant_routing_tradeoff_replay.json"
)
R127_REPLAY_SHA256 = (
    "293fc34625268ea3cc34f7e6bdd28b8f46ced80f5f00855855639525f0be38e7"
)
R127_CONTROLS = pathlib.Path(
    "torus_c5_bucket_resultant_routing_tradeoff_controls.json"
)
R127_CONTROLS_SHA256 = (
    "536b8b366e6705ae4c7a7f5c8b46630b85a3b6dd827040202cb0c726d99ae758"
)
R127_LOGS = pathlib.Path("factor_logs_and_identical_descent_r127.json")
R127_LOGS_SHA256 = (
    "7d3b8d62d4b77dcdbd92e1a2ec84df10dd0a106832ec6ee08ef022967ad6ee13"
)
R127_GATE = pathlib.Path(
    "p1553_torus_c5_bucket_resultant_routing_tradeoff_probe_gate_r127.md"
)
R127_GATE_SHA256 = (
    "729bb912d2db0bd518375bdbce0f7cf0fa5d068af854224b8306a7c8be6db4d8"
)
R127_PARENT = pathlib.Path(
    "p1553_torus_c5_bucket_resultant_routing_"
    "tradeoff_probe_parent_report_r127.yaml"
)
R127_PARENT_SHA256 = (
    "2b052667b1ed2687ce048c4059c13ee9be4804fc096f310bb537eaf0a10776e1"
)
R121_GATE = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_gate_r121.md"
)
R121_GATE_SHA256 = (
    "9266e3655a3f4176280834ec91c897cc7d30274382df6e197628af37bae71309"
)
R119_GATE = pathlib.Path(
    "p1553_m6_output_sensitive_nonlinear_c5_source_index_probe_gate_r119.md"
)
R119_GATE_SHA256 = (
    "941e375918c4acd1be8293fcc40666879b4f5178b4454e28c21487cb9be9a9e9"
)
R114_GATE = pathlib.Path(
    "p1553_5a5c_transposed_nonuniform_c5_leaf_generator_probe_gate_r114.md"
)
R114_GATE_SHA256 = (
    "6574145a6d67ec4e55a8bf0cf90c3a0c9f858aa4cddbfd7e9a733c1033b3a3da"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R127 = load_module("p1553_r127_for_r128", R127_PRODUCER)
R126 = R127.R126
R121 = R127.R121
R82 = R121.R82
Field = R121.Field
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r127_producer", R127_PRODUCER, R127_PRODUCER_SHA256),
        ("r127_report", R127_REPORT, R127_REPORT_SHA256),
        ("r127_frozen", R127_FROZEN, R127_FROZEN_SHA256),
        ("r127_cost", R127_COST, R127_COST_SHA256),
        ("r127_replay", R127_REPLAY, R127_REPLAY_SHA256),
        ("r127_controls", R127_CONTROLS, R127_CONTROLS_SHA256),
        ("r127_logs", R127_LOGS, R127_LOGS_SHA256),
        ("r127_gate", R127_GATE, R127_GATE_SHA256),
        ("r127_parent", R127_PARENT, R127_PARENT_SHA256),
        ("r121_gate", R121_GATE, R121_GATE_SHA256),
        ("r119_gate", R119_GATE, R119_GATE_SHA256),
        ("r114_gate", R114_GATE, R114_GATE_SHA256),
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
        raise AssertionError(f"R128 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def newton_interpolate(
    field: Field,
    points: Iterable[Fp2],
    values: Iterable[Fp2],
) -> tuple[Fp2, ...]:
    nodes = tuple(points)
    coefficients = list(values)
    if len(nodes) != len(coefficients):
        raise ValueError("point/value lengths differ")
    if len(set(nodes)) != len(nodes):
        raise ValueError("interpolation points must be distinct")
    for order in range(1, len(nodes)):
        for index in range(len(nodes) - 1, order - 1, -1):
            coefficients[index] = field.div(
                field.sub(
                    coefficients[index],
                    coefficients[index - 1],
                ),
                field.sub(nodes[index], nodes[index - order]),
            )
    return tuple(coefficients)


def newton_evaluate(
    field: Field,
    nodes: Iterable[Fp2],
    coefficients: Iterable[Fp2],
    target: Fp2,
) -> Fp2:
    node_values = tuple(nodes)
    coefficient_values = tuple(coefficients)
    if not coefficient_values:
        return field.zero
    result = coefficient_values[-1]
    for index in range(len(coefficient_values) - 2, -1, -1):
        result = field.add(
            coefficient_values[index],
            field.mul(
                field.sub(target, node_values[index]),
                result,
            ),
        )
    return result


def rational_selector_control(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c2 = R126.source_products(field, deck, 2)
    c3 = R126.source_products(field, deck, 3)
    c5 = R126.source_products(field, deck, 5)
    selected: dict[
        Fp2,
        tuple[Fp2, tuple[int, ...], tuple[int, ...]],
    ] = {}
    for pair_source, pair_value in c2:
        for triple_source, triple_value in c3:
            target = field.mul(pair_value, triple_value)
            selected.setdefault(
                target,
                (pair_value, pair_source, triple_source),
            )
    c5_products = {product for _, product in c5}
    if set(selected) != c5_products:
        raise AssertionError("selector domain does not equal C5 support")
    nodes = tuple(selected)
    values = tuple(selected[target][0] for target in nodes)
    coefficients = newton_interpolate(field, nodes, values)
    degree = max(
        index
        for index, coefficient in enumerate(coefficients)
        if coefficient != field.zero
    )
    interpolation_exact = all(
        newton_evaluate(field, nodes, coefficients, target)
        == selected[target][0]
        for target in nodes
    )
    source_replay_exact = all(
        field.product(
            deck[index]
            for index in tuple(
                sorted(selected[target][1] + selected[target][2])
            )
        )
        == target
        for target in nodes
    )
    fibers = Counter(values)
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "deck_size": len(deck),
        "c2_source_count": len(c2),
        "c3_source_count": len(c3),
        "c5_source_count": len(c5),
        "selector_domain_size": len(nodes),
        "selector_domain_equals_distinct_c5_support": (
            len(nodes) == len(c5_products) == len(c5)
        ),
        "selected_c2_value_count": len(fibers),
        "maximum_selected_c2_fiber_size": max(fibers.values()),
        "maximum_possible_fixed_c2_fiber_size": len(c3),
        "all_selected_c2_fibers_within_c3_bound": (
            max(fibers.values()) <= len(c3)
        ),
        "rational_degree_counting_lower_bound": math.ceil(
            len(nodes) / len(c2)
        ),
        "unique_interpolation_polynomial_degree": degree,
        "interpolation_degree_is_domain_size_minus_one": (
            degree == len(nodes) - 1
        ),
        "interpolation_exact_on_all_positive_targets": interpolation_exact,
        "all_selected_c2_c3_sources_replay": source_replay_exact,
        "newton_coefficient_nonzero_count": sum(
            coefficient != field.zero for coefficient in coefficients
        ),
        "newton_coefficients_sha256": sha256_json(
            [field.json(coefficient) for coefficient in coefficients]
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        rational_selector_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    return {
        "schema": (
            "p1553.torus_c5_rational_selector_degree_controls.r128.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "all_selector_domains_equal_distinct_c5_support": all(
            row["selector_domain_equals_distinct_c5_support"]
            for row in controls
        ),
        "all_interpolation_polynomials_have_full_degree": all(
            row["interpolation_degree_is_domain_size_minus_one"]
            for row in controls
        ),
        "all_interpolations_exact": all(
            row["interpolation_exact_on_all_positive_targets"]
            for row in controls
        ),
        "all_sources_replay": all(
            row["all_selected_c2_c3_sources_replay"]
            for row in controls
        ),
        "all_fibers_respect_c3_bound": all(
            row["all_selected_c2_fibers_within_c3_bound"]
            for row in controls
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "selector_model": (
            "A rational function R(Y)=A(Y)/D(Y), defined on every positive "
            "C5 target, returns a C2 factor x such that Y/x is in C3."
        ),
        "degree_definition": "d=max(deg A,deg D)",
        "fiber_bound": (
            "For each x in C2, A(Y)-xD(Y) has degree at most d and is "
            "not identically zero for a nonconstant selector, so R(Y)=x "
            "on at most d targets."
        ),
        "constant_selector_boundary": (
            "A constant x covers at most |C3| distinct targets x*C3 and "
            "cannot cover |C5| when |C5|>|C3|."
        ),
        "degree_lower_bound": "d>=ceil(|C5|/|C2|)",
        "iid_exponent_lower_bound": (
            "d>=B^(15/4-3/2+o(1))=B^(9/4+o(1))"
        ),
        "dense_representation_state_exponent_B": fraction_record(
            C3_EXPONENT
        ),
        "dense_horner_query_exponent_B": fraction_record(C3_EXPONENT),
        "inside_setup_cap": True,
        "inside_polylog_query_cap": False,
        "scope_limits": [
            "one globally defined rational C2 selector",
            "dense represented numerator and denominator evaluation",
            "piecewise selectors charged by represented branch degrees",
        ],
        "not_covered": [
            "high-degree low-SLP rational selectors",
            "compact branch-selection circuits",
            "adaptive cell-probe selectors",
            "shared transposed selector evaluation",
            "general arithmetic circuits or data structures",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_rational_selector_degree_"
            "cost_ledger.r128.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
        },
        "theorem": theorem,
        "routes": [
            {
                "route_id": "dense_single_rational_c2_selector",
                "minimum_degree_exponent_B": fraction_record(C3_EXPONENT),
                "setup_coefficient_exponent_B": fraction_record(
                    C3_EXPONENT
                ),
                "query_exponent_B": fraction_record(C3_EXPONENT),
                "inside_setup_cap": True,
                "inside_polylog_query_cap": False,
            },
            {
                "route_id": "explicit_target_to_c2_selector_table",
                "state_exponent_B": fraction_record(C5_EXPONENT),
                "inside_setup_cap": False,
            },
            {
                "route_id": "high_degree_low_slp_rational_selector",
                "degree_lower_bound_applies": True,
                "circuit_size_lower_bound_proved": False,
                "exact_structure_constructed": False,
                "status": "open",
            },
            {
                "route_id": "compact_piecewise_rational_selector",
                "total_branch_degree_lower_bound_exponent_B": (
                    fraction_record(C3_EXPONENT)
                ),
                "compact_branch_router_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R127_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R127 nonclaim boundary drifted")
    controls = finite_controls()
    theorem = theorem_record()
    cost = cost_ledger(theorem)
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r127_singleton_c3_router_interface_inherited": (
            inherited["admission"][
                "scoped_resultant_tradeoff_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "eight_actual_selector_controls_complete": (
            controls["control_count"] == 8
        ),
        "selector_domains_equal_distinct_c5_support": controls[
            "all_selector_domains_equal_distinct_c5_support"
        ],
        "all_actual_interpolants_have_full_degree": controls[
            "all_interpolation_polynomials_have_full_degree"
        ],
        "all_actual_interpolations_exact": controls[
            "all_interpolations_exact"
        ],
        "all_selected_sources_replay": controls["all_sources_replay"],
        "all_selector_fibers_respect_c3_bound": controls[
            "all_fibers_respect_c3_bound"
        ],
        "rational_degree_lower_bound_explicit": (
            theorem["degree_lower_bound"]
            == "d>=ceil(|C5|/|C2|)"
        ),
        "iid_B9O4_degree_cost_charged": (
            theorem["dense_representation_state_exponent_B"]
            == fraction_record(C3_EXPONENT)
        ),
        "dense_query_cost_rejected": (
            theorem["inside_setup_cap"]
            and not theorem["inside_polylog_query_cap"]
        ),
        "low_slp_and_piecewise_routes_preserved": (
            "high-degree low-SLP rational selectors"
            in theorem["not_covered"]
            and "compact branch-selection circuits"
            in theorem["not_covered"]
        ),
        "inside_cap_low_slp_selector_complete": False,
        "inside_cap_five_source_recovery_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one high-degree low-SLP or compact piecewise "
        "rational C2 selector for the cap-tight singleton-C3 index. It must "
        "evaluate in polylogarithmic arbitrary-target work despite degree "
        "B^(9/4+o(1)), return a matching C2+C3 source or an exact empty "
        "certificate, use at most B^(9/4+o(1)) total state, avoid field DLP, "
        "and expose branch routing, rank, logs, identical descent, memory, "
        "field-operation, and bit costs."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_rational_selector_degree.r128.v1"
        ),
        "source_bindings": source_binding_records(),
        "required_interface": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
            "exact_empty_rejection_required": True,
            "five_projective_backpointers_required": True,
            "field_discrete_logarithms_allowed": False,
        },
        "closed_scoped_grammars": [
            "dense single rational C2 selectors",
            "explicit target-to-C2 selector tables",
            "densely represented piecewise rational branches",
        ],
        "preserved_interface": (
            "high-degree low-SLP or compact piecewise rational C2 selector "
            "with implicit branch routing"
        ),
        "general_circuit_or_data_structure_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_rational_selector_degree_replay.r128.v1"
        ),
        "actual_selector_control_count": controls["control_count"],
        "all_interpolations_exact": controls[
            "all_interpolations_exact"
        ],
        "all_interpolants_full_degree": controls[
            "all_interpolation_polynomials_have_full_degree"
        ],
        "all_c2_c3_sources_replay": controls["all_sources_replay"],
        "inside_cap_low_slp_selector_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r128.v1",
        "r127_bucket_resultant_tradeoff_audit_complete": True,
        "r128_rational_selector_degree_audit_complete": True,
        "inside_cap_target_specialized_source_index_complete": False,
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
    classification = (
        "RATIONAL_C2_SELECTOR_FIBER_COUNT_FORCES_DEGREE_AT_LEAST_C5_OVER_"
        "C2_B9O4__ALL_EIGHT_ACTUAL_CANONICAL_SELECTOR_INTERPOLANTS_HAVE_"
        "FULL_C5_MINUS1_DEGREE_AND_EXACT_SOURCES__DENSE_B9O4_SELECTOR_FITS_"
        "SETUP_BUT_MISSES_POLYLOG_QUERY__HIGH_DEGREE_LOW_SLP_OR_COMPACT_"
        "PIECEWISE_SELECTOR_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_RATIONAL_SELECTOR_CONTROLS_AND_SCOPED_DEGREE_BOUND_"
            "ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "all_actual_interpolants_have_full_degree": controls[
                "all_interpolation_polynomials_have_full_degree"
            ],
            "all_actual_interpolations_exact": controls[
                "all_interpolations_exact"
            ],
            "all_selected_sources_replay": controls[
                "all_sources_replay"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "rational_selector_semantics_admitted": True,
            "scoped_rational_degree_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_torus_c5_rational_selector_degree.json",
            "cost": "torus_c5_rational_selector_degree_cost_ledger.json",
            "source_replay": (
                "torus_c5_rational_selector_degree_replay.json"
            ),
            "controls": "torus_c5_rational_selector_degree_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r128.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Degree is not an arithmetic-circuit size lower bound.",
            "The route does not close high-degree low-SLP selectors.",
            "Compact piecewise branch routing remains open.",
            "Finite controls receive no asymptotic credit.",
            "No rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_ACTUAL_RATIONAL_SELECTOR_INTERPOLATION_AND_SOURCE_"
            "SEMANTICS_ONLY__REJECT_DENSE_SINGLE_SELECTOR_AND_EXPLICIT_"
            "TARGET_TABLE_AT_FROZEN_CAPS__PRESERVE_HIGH_DEGREE_LOW_SLP_AND_"
            "COMPACT_PIECEWISE_SELECTOR__NO_LOCATOR__NO_RANK__NO_LOGS__NO_"
            "DESCENT__NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_rational_selector_degree_"
            "probe_report_r128.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_rational_selector_degree.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_rational_selector_degree_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_rational_selector_degree_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_rational_selector_degree_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r128.json"),
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
        f"R128 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
