#!/usr/bin/env python3
"""Audit the all-nonzero path of sparse zero-test C5 decision trees."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.torus_c5_all_nonzero_path_product.r136.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
SOURCE_DEGREE = 5
DETERMINISTIC_PRODUCT_MODE_BOUND = 6
CONTROL_FACTOR_COUNT = 5

R135_PRODUCER = pathlib.Path(
    "p1553_torus_c5_khatri_rao_kruskal_amplification_probe_r135.py"
)
R135_PRODUCER_SHA256 = (
    "f5729fc48372e99cafebb3d8f0bd5eda510802d0a6a3e68d21595dabe9c09ea5"
)
R135_REPORT = pathlib.Path(
    "p1553_torus_c5_khatri_rao_kruskal_amplification_"
    "probe_report_r135.json"
)
R135_REPORT_SHA256 = (
    "27c91b9b7d11c09b6511002f29e1e77d083891aaf5e7222a069a0e76408ef278"
)
R135_FROZEN = pathlib.Path(
    "frozen_torus_c5_khatri_rao_kruskal_amplification.json"
)
R135_FROZEN_SHA256 = (
    "25d1398eb817761d6cb4d9f2fea7db08eeee62000d88acfbe1ee276937cd3098"
)
R135_COST = pathlib.Path(
    "torus_c5_khatri_rao_kruskal_amplification_cost_ledger.json"
)
R135_COST_SHA256 = (
    "7f0c37506437d58e2c04c0ee567193ce9c0822c980e21a5e011561f3312d5bdf"
)
R135_REPLAY = pathlib.Path(
    "torus_c5_khatri_rao_kruskal_amplification_replay.json"
)
R135_REPLAY_SHA256 = (
    "c6dbb5005e19b068f6a2bffa585ab7a9629855fecb2b98036ebd479ea1a329ef"
)
R135_CONTROLS = pathlib.Path(
    "torus_c5_khatri_rao_kruskal_amplification_controls.json"
)
R135_CONTROLS_SHA256 = (
    "bfec781c34fe40c1bea102f15ff6699c9265424ae37be5cadb3c71e2cc1991d6"
)
R135_LOGS = pathlib.Path("factor_logs_and_identical_descent_r135.json")
R135_LOGS_SHA256 = (
    "41754c11d3f5448df4ac96f514e3a13eced8f4c82286da54a4e7dfbedc7150e2"
)
R135_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_khatri_rao_kruskal_amplification_probe_r135.py"
)
R135_TEST_SHA256 = (
    "a85447be8f30789f95ba0a262b10e26cd3131b0dca78a4a53f29ed9bd0fd539b"
)
R135_GATE = pathlib.Path(
    "p1553_torus_c5_khatri_rao_kruskal_amplification_probe_gate_r135.md"
)
R135_GATE_SHA256 = (
    "c9bbf87b8b2c5621218823122f61151ed968b348fbb7497789757797c15b1c19"
)
R135_PARENT = pathlib.Path(
    "p1553_torus_c5_khatri_rao_kruskal_amplification_"
    "probe_parent_report_r135.yaml"
)
R135_PARENT_SHA256 = (
    "39d3ddc00947291cc6437e385bc142bb50a680ee283a37e1d4e2b39fe7ddb62e"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R135 = load_module("p1553_r135_for_r136", R135_PRODUCER)
R134 = R135.R134
R133 = R135.R133
R131 = R135.R131
R129 = R135.R129
R126 = R135.R126
R121 = R135.R121
R82 = R135.R82
Field = R135.Field
Fp2 = tuple[int, int]
Polynomial = tuple[Fp2, ...]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r135_producer", R135_PRODUCER, R135_PRODUCER_SHA256),
        ("r135_report", R135_REPORT, R135_REPORT_SHA256),
        ("r135_frozen", R135_FROZEN, R135_FROZEN_SHA256),
        ("r135_cost", R135_COST, R135_COST_SHA256),
        ("r135_replay", R135_REPLAY, R135_REPLAY_SHA256),
        ("r135_controls", R135_CONTROLS, R135_CONTROLS_SHA256),
        ("r135_logs", R135_LOGS, R135_LOGS_SHA256),
        ("r135_test", R135_TEST, R135_TEST_SHA256),
        ("r135_gate", R135_GATE, R135_GATE_SHA256),
        ("r135_parent", R135_PARENT, R135_PARENT_SHA256),
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
        raise AssertionError(f"R136 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def trim_polynomial(
    coefficients: Sequence[Fp2],
    field: Field,
) -> Polynomial:
    values = list(coefficients)
    while len(values) > 1 and values[-1] == field.zero:
        values.pop()
    return tuple(values)


def polynomial_mul(
    left: Sequence[Fp2],
    right: Sequence[Fp2],
    field: Field,
) -> Polynomial:
    result = [field.zero] * (len(left) + len(right) - 1)
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            index = left_degree + right_degree
            result[index] = field.add(
                result[index],
                field.mul(left_value, right_value),
            )
    return trim_polynomial(result, field)


def polynomial_product(
    factors: Iterable[Sequence[Fp2]],
    field: Field,
) -> Polynomial:
    result: Polynomial = (field.one,)
    for factor in factors:
        result = polynomial_mul(result, factor, field)
    return result


def polynomial_eval(
    coefficients: Sequence[Fp2],
    value: Fp2,
    field: Field,
) -> Fp2:
    result = field.zero
    for coefficient in reversed(coefficients):
        result = field.add(field.mul(result, value), coefficient)
    return result


def linear_factor(root: Fp2, field: Field) -> Polynomial:
    return (field.neg(root), field.one)


def represented_mode_count(
    coefficients: Sequence[Fp2],
    field: Field,
) -> int:
    return sum(coefficient != field.zero for coefficient in coefficients)


def factor_control(
    field: Field,
    deck: tuple[Fp2, ...],
    c5_by_source: dict[tuple[int, ...], Fp2],
    c5_values: set[Fp2],
    color: int,
    part: tuple[int, ...],
) -> dict[str, Any]:
    base = {
        "color": color,
        "part_indices": list(part),
        "part_size": len(part),
        "witness_available": len(part) >= 2,
        "finite_control_receives_asymptotic_credit": False,
    }
    if len(part) < 2:
        return {
            **base,
            "unavailable_reason": "finite_color_part_has_fewer_than_two_atoms",
        }
    left_index, right_index = part[:2]
    values = R134.progression_targets(
        deck[left_index],
        deck[right_index],
        field,
    )
    sources = R134.progression_sources(left_index, right_index)
    roots = values[:CONTROL_FACTOR_COUNT]
    factors = tuple(linear_factor(root, field) for root in roots)
    product = polynomial_product(factors, field)
    selected_positive = values[-1]
    inverse_empty = field.inv(selected_positive)
    factor_evaluations = tuple(
        tuple(polynomial_eval(factor, value, field) for factor in factors)
        for value in values
    )
    product_evaluations = tuple(
        polynomial_eval(product, value, field) for value in values
    )
    c5_sample = tuple(sorted(c5_values))
    union_identity_on_c5 = all(
        (
            polynomial_eval(product, value, field) == field.zero
        )
        == any(
            polynomial_eval(factor, value, field) == field.zero
            for factor in factors
        )
        for value in c5_sample
    )
    return {
        **base,
        "left_index": left_index,
        "right_index": right_index,
        "factor_count": len(factors),
        "factor_mode_counts": [
            represented_mode_count(factor, field) for factor in factors
        ],
        "product_degree": len(product) - 1,
        "product_mode_count": represented_mode_count(product, field),
        "product_mode_bound_respected": (
            represented_mode_count(product, field)
            <= DETERMINISTIC_PRODUCT_MODE_BOUND
        ),
        "roots": [field.json(value) for value in roots],
        "product_coefficients": [
            field.json(coefficient) for coefficient in product
        ],
        "factor_zero_masks_on_progression": [
            [evaluation == field.zero for evaluation in row]
            for row in factor_evaluations
        ],
        "product_zero_mask_on_progression": [
            evaluation == field.zero for evaluation in product_evaluations
        ],
        "product_vanishes_on_exactly_first_five_progression_targets": (
            all(value == field.zero for value in product_evaluations[:-1])
            and product_evaluations[-1] != field.zero
        ),
        "selected_positive": field.json(selected_positive),
        "selected_positive_in_c5": selected_positive in c5_values,
        "selected_positive_all_factor_values_nonzero": all(
            polynomial_eval(factor, selected_positive, field) != field.zero
            for factor in factors
        ),
        "inverse_empty": field.json(inverse_empty),
        "inverse_empty_not_in_c5": inverse_empty not in c5_values,
        "inverse_empty_all_factor_values_nonzero": all(
            polynomial_eval(factor, inverse_empty, field) != field.zero
            for factor in factors
        ),
        "selected_positive_and_inverse_empty_same_all_nonzero_path": (
            all(
                polynomial_eval(factor, selected_positive, field)
                != field.zero
                and polynomial_eval(factor, inverse_empty, field)
                != field.zero
                for factor in factors
            )
        ),
        "product_zero_iff_any_factor_zero_on_all_c5_values": (
            union_identity_on_c5
        ),
        "all_sources_in_c5_replay": all(
            source in c5_by_source for source in sources
        ),
        "all_source_values_match_progression": all(
            c5_by_source[source] == value
            for source, value in zip(sources, values)
        ),
        "all_sources_accepted_by_color": all(
            R131.source_color_multiplicity(source, color)
            >= R134.COLOR_ACCEPTANCE_THRESHOLD
            for source in sources
        ),
    }


def finite_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c5 = R126.source_products(field, deck, SOURCE_DEGREE)
    c5_by_source = {source: value for source, value in c5}
    c5_values = set(c5_by_source.values())
    colors = [
        factor_control(
            field,
            deck,
            c5_by_source,
            c5_values,
            color,
            part,
        )
        for color, part in enumerate(
            R129.balanced_four_parts(len(deck))
        )
        if part
    ]
    available = [row for row in colors if row["witness_available"]]
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "subgroup_order_probable_prime": R82.R70.is_prime(
            curve["subgroup_order"]
        ),
        "deck_size": len(deck),
        "c5_source_count": len(c5),
        "c5_value_count": len(c5_values),
        "active_color_count": len(colors),
        "available_path_control_count": len(available),
        "color_controls": colors,
        "all_available_product_mode_bounds_respected": all(
            row["product_mode_bound_respected"] for row in available
        ),
        "all_available_products_leave_one_positive": all(
            row[
                "product_vanishes_on_exactly_first_five_progression_targets"
            ]
            and row["selected_positive_in_c5"]
            and row["selected_positive_all_factor_values_nonzero"]
            for row in available
        ),
        "all_available_inverse_empties_share_all_nonzero_path": all(
            row["inverse_empty_not_in_c5"]
            and row["inverse_empty_all_factor_values_nonzero"]
            and row[
                "selected_positive_and_inverse_empty_same_all_nonzero_path"
            ]
            for row in available
        ),
        "all_available_union_identities_exact": all(
            row["product_zero_iff_any_factor_zero_on_all_c5_values"]
            for row in available
        ),
        "all_available_sources_replay_and_are_accepted": all(
            row["all_sources_in_c5_replay"]
            and row["all_source_values_match_progression"]
            and row["all_sources_accepted_by_color"]
            for row in available
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        finite_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    witness_count = sum(
        row["available_path_control_count"] for row in controls
    )
    return {
        "schema": (
            "p1553.torus_c5_all_nonzero_path_product_"
            "controls.r136.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "active_color_control_count": sum(
            row["active_color_count"] for row in controls
        ),
        "available_path_control_count": witness_count,
        "all_subgroup_orders_probable_prime": all(
            row["subgroup_order_probable_prime"] for row in controls
        ),
        "all_available_product_mode_bounds_respected": (
            witness_count > 0
            and all(
                row["all_available_product_mode_bounds_respected"]
                for row in controls
            )
        ),
        "all_available_products_leave_one_positive": (
            witness_count > 0
            and all(
                row["all_available_products_leave_one_positive"]
                for row in controls
            )
        ),
        "all_available_inverse_empties_share_all_nonzero_path": (
            witness_count > 0
            and all(
                row[
                    "all_available_inverse_empties_share_all_nonzero_path"
                ]
                for row in controls
            )
        ),
        "all_available_union_identities_exact": (
            witness_count > 0
            and all(
                row["all_available_union_identities_exact"]
                for row in controls
            )
        ),
        "all_available_sources_replay_and_are_accepted": (
            witness_count > 0
            and all(
                row["all_available_sources_replay_and_are_accepted"]
                for row in controls
            )
        ),
        "finite_controls_receive_asymptotic_credit": False,
        "candidate_discrete_logs_consumed": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "all_nonzero_path_product": (
            "Let a deterministic membership decision tree on H test "
            "nonzero represented polynomials f_1,...,f_d along its "
            "all-nonzero root-to-leaf path, and put P=product_i f_i. "
            "A point follows that path exactly when P is nonzero. If the "
            "leaf rejects, every accepted point must have left the path, "
            "so P vanishes on the full acceptance support. If the leaf "
            "accepts, every point outside the support must leave the path, "
            "so the complement is contained in the root set of P, equal "
            "to the union of the root sets of the f_i."
        ),
        "deterministic_structured_dichotomy": (
            "For one asymptotic C5 color support S, let M be the number "
            "of distinct represented modes of the nonzero product P. "
            "R135 excludes P vanishing on S when M<=6. Kelley-adapted "
            "root bounds give |Z(f_i)|<=2*q^(1-1/(t_i-1)) for t_i>=2 "
            "and zero roots for t_i=1. Therefore every exact tree obeys "
            "M>6 or q-|S|<=sum_i 2*q^(1-1/(t_i-1))."
        ),
        "fixed_node_support_consequence": (
            "If every node has at most fixed t>=2 modes and the path "
            "depth is polylogarithmic, the root-budget sum is o(q) while "
            "|S|=q^(3/4+o(1)). Hence an exact tree must have more than "
            "six distinct modes in its expanded all-nonzero path product."
        ),
        "random_deck_amplification": (
            "Under the frozen uniform-random-deck model, replace six by "
            "L(q)=(5-o(1))*log2(q) with overwhelming probability using "
            "R135. For fixed node support t and polylogarithmic depth, "
            "exactness forces expanded path-product support above L(q). "
            "Since product support is at most product_i t_i, this implies "
            "depth greater than log_t L(q) when every t_i<=t. The lower "
            "bound is only Omega(log log q), compatible with polylog query "
            "work and compact high-expansion straight-line programs."
        ),
        "zero_test_scope_only": True,
        "nonzero_value_tests_covered": False,
        "coordinate_comparisons_covered": False,
        "high_expansion_low_slp_covered": False,
        "maximum_deterministic_path_product_mode_count_excluded": 6,
        "random_model_path_product_mode_count_excluded": (
            "(5-o(1))*log2(q)"
        ),
        "not_covered": [
            "all-nonzero path products with seven or more structured modes",
            "compact circuits whose expanded path product has large support",
            "tests of nonzero field values rather than equality to zero",
            "Frobenius-coordinate comparisons and table probes",
            "general arithmetic-circuit, RAM, or cell-probe lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_all_nonzero_path_product_"
            "cost_ledger.r136.v1"
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
                "route_id": (
                    "structured_zero_test_dag_product_through_six_modes"
                ),
                "random_model_required": False,
                "structured_factor_base_theorem": True,
                "status": (
                    "rejected_by_all_nonzero_path_product_root_cover_"
                    "dichotomy"
                ),
            },
            {
                "route_id": (
                    "random_deck_zero_test_dag_product_through_near_five_log"
                ),
                "random_model_required": True,
                "structured_factor_base_theorem": False,
                "status": (
                    "rejected_with_overwhelming_probability_under_"
                    "uniform_random_deck_model_only"
                ),
            },
            {
                "route_id": "growing_expanded_support_low_slp_dag",
                "expanded_path_product_bound_applicable": False,
                "inside_cap_exact_dag_constructed": False,
                "status": "open",
            },
            {
                "route_id": "nonzero_value_frobenius_coordinate_dag",
                "zero_test_path_product_argument_applicable": False,
                "inside_cap_exact_dag_constructed": False,
                "status": "open",
            },
            {
                "route_id": "structured_seven_plus_mode_path_product",
                "structured_rank_above_six_proved": False,
                "inside_cap_exact_dag_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R135_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R135 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    routes = {row["route_id"]: row for row in cost["routes"]}
    obligations = {
        "ten_source_bindings_verified": len(source_hashes) == 10,
        "r135_rank_interface_inherited": (
            inherited["admission"][
                "deterministic_one_to_six_mode_negative_admitted"
            ]
            and inherited["admission"][
                "random_model_near_five_log_mode_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "all_nonzero_path_semantics_explicit": (
            "root-to-leaf path"
            in theorem["all_nonzero_path_product"]
        ),
        "rejecting_leaf_positive_cover_explicit": (
            "leaf rejects" in theorem["all_nonzero_path_product"]
        ),
        "accepting_leaf_complement_cover_explicit": (
            "leaf accepts" in theorem["all_nonzero_path_product"]
        ),
        "structured_six_mode_dichotomy_explicit": (
            "M>6" in theorem["deterministic_structured_dichotomy"]
        ),
        "random_near_five_log_dichotomy_explicit": (
            theorem["random_model_path_product_mode_count_excluded"]
            == "(5-o(1))*log2(q)"
        ),
        "low_slp_escape_preserved": (
            not theorem["high_expansion_low_slp_covered"]
        ),
        "eight_actual_controls_complete": controls["control_count"] == 8,
        "twelve_actual_path_controls_complete": (
            controls["available_path_control_count"] == 12
        ),
        "all_available_path_replays_exact": (
            controls["all_available_product_mode_bounds_respected"]
            and controls["all_available_products_leave_one_positive"]
            and controls[
                "all_available_inverse_empties_share_all_nonzero_path"
            ]
            and controls["all_available_union_identities_exact"]
            and controls[
                "all_available_sources_replay_and_are_accepted"
            ]
        ),
        "finite_controls_receive_no_asymptotic_credit": (
            not controls["finite_controls_receive_asymptotic_credit"]
        ),
        "growing_support_nonzero_and_seven_mode_routes_preserved": all(
            routes[route_id]["status"] == "open"
            for route_id in (
                "growing_expanded_support_low_slp_dag",
                "nonzero_value_frobenius_coordinate_dag",
                "structured_seven_plus_mode_path_product",
            )
        ),
        "inside_cap_asymmetric_predicate_complete": False,
        "inside_cap_five_source_index_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    next_action = (
        "Probe the two explicit escapes from the path-product theorem: "
        "construct or refute a growing-support low-SLP selector whose "
        "expanded all-nonzero product exceeds the rank threshold, and test "
        "nonzero-value Frobenius-coordinate branches that do not reduce to "
        "zero sets. Freeze every circuit node and branch, replay exact "
        "positive and inverse-empty targets and C2+C3 sources, fit "
        "B^(9/4+o(1)) state and polylogarithmic arbitrary-target work, avoid "
        "field DLP, and charge rank, logs, identical descent, memory, field "
        "operations, and bits."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_all_nonzero_path_product.r136.v1"
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
            (
                "structured sparse zero-test DAGs whose expanded "
                "all-nonzero path product has at most six modes"
            ),
            (
                "random-deck sparse zero-test DAGs whose expanded "
                "all-nonzero path product has at most "
                "(5-o(1))*log2(q) modes"
            ),
        ],
        "preserved_interface": (
            "growing-support low-SLP path product, nonzero-value "
            "Frobenius-coordinate DAG, or structured seven-plus-mode "
            "path product"
        ),
        "random_deck_model_required_above_six_modes": True,
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_all_nonzero_path_product_replay.r136.v1"
        ),
        "actual_control_count": controls["control_count"],
        "active_color_control_count": controls[
            "active_color_control_count"
        ],
        "available_path_control_count": controls[
            "available_path_control_count"
        ],
        "all_available_product_mode_bounds_respected": controls[
            "all_available_product_mode_bounds_respected"
        ],
        "all_available_products_leave_one_positive": controls[
            "all_available_products_leave_one_positive"
        ],
        "all_available_inverse_empties_share_all_nonzero_path": controls[
            "all_available_inverse_empties_share_all_nonzero_path"
        ],
        "inside_cap_surviving_selector_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r136.v1",
        "r135_khatri_rao_amplification_audit_complete": True,
        "r136_all_nonzero_path_product_audit_complete": True,
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
        "SPARSE_ZERO_TEST_TREE_ALL_NONZERO_PATH_COLLAPSES_TO_PRODUCT_"
        "POLYNOMIAL__REJECTING_LEAF_FORCES_PRODUCT_TO_COVER_POSITIVE_"
        "SUPPORT__ACCEPTING_LEAF_FORCES_NODE_ROOT_UNION_TO_COVER_SUBGROUP_"
        "COMPLEMENT__STRUCTURED_EXACT_TREE_REQUIRES_PRODUCT_ABOVE_SIX_"
        "MODES_OR_LINEAR_SCALE_ROOT_BUDGET__RANDOM_DECK_THRESHOLD_NEAR_"
        "FIVE_LOG2_Q__TWELVE_EXACT_FIVE_FACTOR_POSITIVE_INVERSE_EMPTY_"
        "REPLAYS__GROWING_SUPPORT_LOW_SLP_AND_NONZERO_VALUE_ROUTES_OPEN__"
        "NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "ALL_NONZERO_PATH_PRODUCT_ROOT_COVER_DICHOTOMY_ONLY_"
            "WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "active_color_control_count": controls[
                "active_color_control_count"
            ],
            "available_path_control_count": controls[
                "available_path_control_count"
            ],
            "all_available_products_leave_one_positive": controls[
                "all_available_products_leave_one_positive"
            ],
            "all_available_inverse_empties_share_all_nonzero_path": (
                controls[
                    "all_available_inverse_empties_share_all_nonzero_path"
                ]
            ),
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "structured_six_mode_path_product_negative_admitted": True,
            "random_model_near_five_log_path_product_negative_admitted": True,
            "growing_support_or_nonzero_value_selector_admitted": False,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_torus_c5_all_nonzero_path_product.json",
            "cost": (
                "torus_c5_all_nonzero_path_product_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_all_nonzero_path_product_replay.json"
            ),
            "controls": (
                "torus_c5_all_nonzero_path_product_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r136.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The theorem covers equality-to-zero decision nodes, not arbitrary field-value tests.",
            "Expanded path-product support is not a straight-line-program size lower bound.",
            "The random-deck threshold does not transfer to the structured factor base.",
            "The exact finite controls receive no probability or asymptotic credit.",
            "Growing-support low-SLP and nonzero-value Frobenius DAGs remain open.",
            "No general circuit, RAM, or cell-probe lower bound is claimed.",
            "No source index, rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_ALL_NONZERO_PATH_PRODUCT_ROOT_COVER_DICHOTOMY__REJECT_"
            "STRUCTURED_ZERO_TEST_DAGS_WITH_AT_MOST_SIX_EXPANDED_PATH_"
            "MODES__ADMIT_NEAR_FIVE_LOG2_Q_RANDOM_MODEL_EXTENSION_ONLY__"
            "ADMIT_TWELVE_EXACT_FIVE_FACTOR_POSITIVE_INVERSE_EMPTY_REPLAYS_"
            "WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_GROWING_SUPPORT_LOW_SLP_"
            "NONZERO_VALUE_AND_STRUCTURED_SEVEN_MODE_ROUTES__NO_LOCATOR__"
            "NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_all_nonzero_path_product_"
            "probe_report_r136.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_all_nonzero_path_product.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_all_nonzero_path_product_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_all_nonzero_path_product_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_all_nonzero_path_product_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r136.json"),
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
        f"R136 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
