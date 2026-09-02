#!/usr/bin/env python3
"""Audit nodewise root-union depth for sparse C5 zero-test trees."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from fractions import Fraction
from typing import Any, Sequence


SCHEMA = "p1553.torus_c5_binomial_node_union_depth.r137.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
SOURCE_DEGREE = 5
STRUCTURED_ATOM_EXPONENT_Q = Fraction(3, 20)
RANDOM_SUPPORT_EXPONENT_Q = Fraction(3, 4)

R136_PRODUCER = pathlib.Path(
    "p1553_torus_c5_all_nonzero_path_product_probe_r136.py"
)
R136_PRODUCER_SHA256 = (
    "a0761c05909996a80b3322aec859fcde97b23d77ce7cf512ddfa0c888e3017d9"
)
R136_REPORT = pathlib.Path(
    "p1553_torus_c5_all_nonzero_path_product_probe_report_r136.json"
)
R136_REPORT_SHA256 = (
    "530b696871917a69ddef1e6a0a14a44421176439c0ccf7b9031ecf5a8a1b64ce"
)
R136_FROZEN = pathlib.Path(
    "frozen_torus_c5_all_nonzero_path_product.json"
)
R136_FROZEN_SHA256 = (
    "3a8587a33e176a7174dacf88ced1da243b0be9b77f214a1b41bb2ef3db4ccef0"
)
R136_COST = pathlib.Path(
    "torus_c5_all_nonzero_path_product_cost_ledger.json"
)
R136_COST_SHA256 = (
    "30935ad0b297822f56dca0e69fe80952e51b92c61f32cdee79a43dde8d483d8e"
)
R136_REPLAY = pathlib.Path(
    "torus_c5_all_nonzero_path_product_replay.json"
)
R136_REPLAY_SHA256 = (
    "4c4414fa1472eaec54f9b71eeb6defc47b75c3b278cdad66b7d14e5fa0fb1d4d"
)
R136_CONTROLS = pathlib.Path(
    "torus_c5_all_nonzero_path_product_controls.json"
)
R136_CONTROLS_SHA256 = (
    "bcd98529f44223cb07510094de08c0f234da14830410b67847ec1aef75bf216b"
)
R136_LOGS = pathlib.Path("factor_logs_and_identical_descent_r136.json")
R136_LOGS_SHA256 = (
    "c38aaf5b7966f24273ffe76d6564f4e2cf9d845caa2d17715a535ce044209049"
)
R136_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_all_nonzero_path_product_probe_r136.py"
)
R136_TEST_SHA256 = (
    "c6dd14aeabda0f4b69b6d3f021b207d998214d63922178fab58220a8beac01e3"
)
R136_GATE = pathlib.Path(
    "p1553_torus_c5_all_nonzero_path_product_probe_gate_r136.md"
)
R136_GATE_SHA256 = (
    "1a7e631d374e9b49d8d7eaf3440bc31d95b8d28f8f73f5a3e520b1f461bcabc5"
)
R136_PARENT = pathlib.Path(
    "p1553_torus_c5_all_nonzero_path_product_"
    "probe_parent_report_r136.yaml"
)
R136_PARENT_SHA256 = (
    "a4d61b3f08e92de9b9276bc12d137a1ff1c1494e81d08d8432c6fea9ecaa72dc"
)
DEVOS_PDF = pathlib.Path(
    "references/devos_structure_critical_product_sets_1301.0096.pdf"
)
DEVOS_PDF_SHA256 = (
    "924bac0a6b5a1e9379e38a53f3ab78a6ad129beb6fddb12396c8275403b8c511"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R136 = load_module("p1553_r136_for_r137", R136_PRODUCER)
R135 = R136.R135
R134 = R136.R134
R133 = R136.R133
R131 = R136.R131
R129 = R136.R129
R126 = R136.R126
R121 = R136.R121
R82 = R136.R82
Field = R136.Field
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r136_producer", R136_PRODUCER, R136_PRODUCER_SHA256),
        ("r136_report", R136_REPORT, R136_REPORT_SHA256),
        ("r136_frozen", R136_FROZEN, R136_FROZEN_SHA256),
        ("r136_cost", R136_COST, R136_COST_SHA256),
        ("r136_replay", R136_REPLAY, R136_REPLAY_SHA256),
        ("r136_controls", R136_CONTROLS, R136_CONTROLS_SHA256),
        ("r136_logs", R136_LOGS, R136_LOGS_SHA256),
        ("r136_test", R136_TEST, R136_TEST_SHA256),
        ("r136_gate", R136_GATE, R136_GATE_SHA256),
        ("r136_parent", R136_PARENT, R136_PARENT_SHA256),
        ("devos_pdf", DEVOS_PDF, DEVOS_PDF_SHA256),
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
        raise AssertionError(f"R137 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def product_set(
    atoms: Sequence[Fp2],
    degree: int,
    field: Field,
) -> tuple[Fp2, ...]:
    return tuple(
        sorted(
            {
                field.product(source)
                for source in itertools.combinations_with_replacement(
                    atoms,
                    degree,
                )
            }
        )
    )


def cauchy_davenport_iterated_bound(
    group_order: int,
    atom_count: int,
    degree: int,
) -> int:
    if atom_count < 1 or degree < 1:
        raise ValueError("atom count and degree must be positive")
    return min(group_order, degree * atom_count - degree + 1)


def random_depth_exponent_q(maximum_node_modes: int) -> Fraction:
    if maximum_node_modes < 2:
        raise ValueError("node mode count must be at least two")
    root_exponent = Fraction(
        maximum_node_modes - 2,
        maximum_node_modes - 1,
    )
    return RANDOM_SUPPORT_EXPONENT_Q - root_exponent


def random_depth_table() -> list[dict[str, Any]]:
    rows = []
    for modes in (2, 3, 4):
        exponent_q = random_depth_exponent_q(modes)
        rows.append(
            {
                "maximum_node_mode_count": modes,
                "maximum_single_node_root_exponent_q": fraction_record(
                    Fraction(modes - 2, modes - 1)
                ),
                "minimum_depth_exponent_q": fraction_record(exponent_q),
                "minimum_depth_exponent_B": fraction_record(
                    5 * exponent_q
                ),
                "polylogarithmic_query_compatible": exponent_q <= 0,
            }
        )
    return rows


def color_control(
    field: Field,
    deck: tuple[Fp2, ...],
    c5_values: set[Fp2],
    color: int,
    part: tuple[int, ...],
    subgroup_order: int,
) -> dict[str, Any]:
    atoms = tuple(deck[index] for index in part)
    products = product_set(atoms, SOURCE_DEGREE, field)
    bound = cauchy_davenport_iterated_bound(
        subgroup_order,
        len(atoms),
        SOURCE_DEGREE,
    )
    selected_positive = products[-1]
    inverse_empty = field.inv(selected_positive)
    roots = products[:-1]
    factors = tuple(R136.linear_factor(root, field) for root in roots)
    selected_nonzero = all(
        R136.polynomial_eval(factor, selected_positive, field) != field.zero
        for factor in factors
    )
    inverse_nonzero = all(
        R136.polynomial_eval(factor, inverse_empty, field) != field.zero
        for factor in factors
    )
    return {
        "color": color,
        "part_indices": list(part),
        "part_size": len(part),
        "product_set_size": len(products),
        "cauchy_davenport_lower_bound": bound,
        "cauchy_davenport_bound_satisfied": len(products) >= bound,
        "cauchy_davenport_bound_is_exact": len(products) == bound,
        "product_values": [field.json(value) for value in products],
        "binomial_path_factor_count": len(factors),
        "every_factor_has_two_modes": all(
            R136.represented_mode_count(factor, field) == 2
            for factor in factors
        ),
        "selected_positive": field.json(selected_positive),
        "selected_positive_in_global_c5": selected_positive in c5_values,
        "selected_positive_all_factors_nonzero": selected_nonzero,
        "inverse_empty": field.json(inverse_empty),
        "inverse_empty_not_in_global_c5": inverse_empty not in c5_values,
        "inverse_empty_all_factors_nonzero": inverse_nonzero,
        "positive_and_inverse_empty_share_all_nonzero_path": (
            selected_nonzero and inverse_nonzero
        ),
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c5 = R126.source_products(field, deck, SOURCE_DEGREE)
    c5_values = {value for _, value in c5}
    colors = [
        color_control(
            field,
            deck,
            c5_values,
            color,
            part,
            curve["subgroup_order"],
        )
        for color, part in enumerate(
            R129.balanced_four_parts(len(deck))
        )
        if part
    ]
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
        "color_controls": colors,
        "all_color_product_sets_meet_cauchy_davenport": all(
            row["cauchy_davenport_bound_satisfied"] for row in colors
        ),
        "all_color_controls_are_bound_tight": all(
            row["cauchy_davenport_bound_is_exact"] for row in colors
        ),
        "all_selected_positives_in_global_c5": all(
            row["selected_positive_in_global_c5"] for row in colors
        ),
        "all_inverse_empties_outside_global_c5": all(
            row["inverse_empty_not_in_global_c5"] for row in colors
        ),
        "all_positive_inverse_pairs_share_all_nonzero_path": all(
            row["positive_and_inverse_empty_share_all_nonzero_path"]
            for row in colors
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
    return {
        "schema": (
            "p1553.torus_c5_binomial_node_union_depth_"
            "controls.r137.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "active_color_control_count": sum(
            row["active_color_count"] for row in controls
        ),
        "all_subgroup_orders_probable_prime": all(
            row["subgroup_order_probable_prime"] for row in controls
        ),
        "all_color_product_sets_meet_cauchy_davenport": all(
            row["all_color_product_sets_meet_cauchy_davenport"]
            for row in controls
        ),
        "all_color_controls_are_bound_tight": all(
            row["all_color_controls_are_bound_tight"]
            for row in controls
        ),
        "all_selected_positives_in_global_c5": all(
            row["all_selected_positives_in_global_c5"]
            for row in controls
        ),
        "all_inverse_empties_outside_global_c5": all(
            row["all_inverse_empties_outside_global_c5"]
            for row in controls
        ),
        "all_positive_inverse_pairs_share_all_nonzero_path": all(
            row["all_positive_inverse_pairs_share_all_nonzero_path"]
            for row in controls
        ),
        "finite_controls_receive_asymptotic_credit": False,
        "candidate_discrete_logs_consumed": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "primary_source": {
            "author": "Matt DeVos",
            "title": "The Structure of Critical Product Sets",
            "identifier": "arXiv:1301.0096",
            "source_result": (
                "Introduction statement of the multiplicative "
                "Cauchy-Davenport theorem for prime-order groups"
            ),
            "pinned_sha256": DEVOS_PDF_SHA256,
        },
        "iterated_product_set_bound": (
            "Let A be a nonempty subset of a cyclic group H of prime "
            "order q. Cauchy-Davenport in multiplicative notation gives "
            "|XY|>=min(q,|X|+|Y|-1). Iterating five times yields "
            "|A^5|>=min(q,5|A|-4). No discrete logarithm is computed; an "
            "abstract group isomorphism transfers the theorem."
        ),
        "structured_color_support_lower_bound": (
            "A balanced structured color has "
            "m=B^(3/4+o(1))=q^(3/20+o(1)) distinct atoms. Its accepted C5 "
            "support contains A^5, hence at least 5m-4 targets. The full "
            "C5 tuple count is B^(15/4)=q^(3/4), so the color complement "
            "has q-o(q) points."
        ),
        "binomial_root_bound": (
            "A nonzero represented binomial "
            "a*z^u+b*z^v with u distinct from v modulo prime q has at "
            "most one root in H: after division it is z^(u-v)=c, and the "
            "nonzero power map is a permutation of H."
        ),
        "deterministic_binomial_depth_lower_bound": (
            "On the all-nonzero path, exactness forces the union of node "
            "root sets to cover either the color support or its complement. "
            "For binomial nodes this union has at most d points. Therefore "
            "d>=min(|S|,q-|S|)>=5m-4=B^(3/4+o(1)). This exceeds "
            "polylogarithmic arbitrary-target work regardless of the "
            "expanded support of the path product."
        ),
        "uniform_random_support_extension": (
            "Under the frozen uniform-random-deck support model only, "
            "|S|=q^(3/4+o(1)). If every node has at most t modes, Kelley "
            "gives at most 2*q^(1-1/(t-1)) roots per node. Thus depth is "
            "at least q^(3/4-(1-1/(t-1))+o(1))/2. For t=2,3,4 the q "
            "exponents are 3/4, 1/4, and 1/12, respectively, all "
            "polynomial and incompatible with polylog query work."
        ),
        "random_depth_table": random_depth_table(),
        "maximum_deterministic_node_mode_count_closed": 2,
        "maximum_random_model_node_mode_count_closed": 4,
        "random_support_model_transferred_to_structured_factor_base": False,
        "nonzero_value_tests_covered": False,
        "not_covered": [
            "structured zero-test nodes with three or more modes",
            "uniform-random support trees with five or more modes per node",
            "tests of nonzero field values rather than equality to zero",
            "Frobenius-coordinate comparisons and table probes",
            "general arithmetic-circuit, RAM, or cell-probe lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_binomial_node_union_depth_"
            "cost_ledger.r137.v1"
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
                "route_id": "structured_binomial_zero_test_tree",
                "minimum_query_depth_exponent_B": fraction_record(
                    Fraction(3, 4)
                ),
                "random_model_required": False,
                "status": "rejected_deterministically_by_product_set_size",
            },
            {
                "route_id": "random_support_at_most_four_mode_zero_test_tree",
                "minimum_four_mode_depth_exponent_B": fraction_record(
                    Fraction(5, 12)
                ),
                "random_model_required": True,
                "structured_factor_base_theorem": False,
                "status": (
                    "rejected_with_high_probability_under_"
                    "uniform_random_support_model_only"
                ),
            },
            {
                "route_id": "structured_three_plus_mode_zero_test_tree",
                "structured_support_lower_bound_sufficient": False,
                "inside_cap_exact_tree_constructed": False,
                "status": "open",
            },
            {
                "route_id": "five_plus_mode_low_slp_zero_test_tree",
                "random_root_budget_polynomial_gap": False,
                "inside_cap_exact_tree_constructed": False,
                "status": "open",
            },
            {
                "route_id": "nonzero_value_frobenius_coordinate_dag",
                "zero_set_root_budget_applicable": False,
                "inside_cap_exact_dag_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R136_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R136 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    routes = {row["route_id"]: row for row in cost["routes"]}
    obligations = {
        "eleven_source_bindings_verified": len(source_hashes) == 11,
        "r136_path_product_interface_inherited": (
            inherited["admission"][
                "structured_six_mode_path_product_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "primary_product_set_source_pinned": (
            theorem["primary_source"]["pinned_sha256"]
            == DEVOS_PDF_SHA256
        ),
        "iterated_cauchy_davenport_bound_explicit": (
            "5|A|-4" in theorem["iterated_product_set_bound"]
        ),
        "structured_color_support_lower_bound_explicit": (
            "q^(3/20+o(1))"
            in theorem["structured_color_support_lower_bound"]
        ),
        "prime_order_binomial_root_bound_explicit": (
            "at most one root" in theorem["binomial_root_bound"]
        ),
        "deterministic_binomial_depth_B_three_quarters": (
            "B^(3/4+o(1))"
            in theorem["deterministic_binomial_depth_lower_bound"]
        ),
        "random_four_mode_depth_B_five_twelfths": (
            theorem["random_depth_table"][-1][
                "minimum_depth_exponent_B"
            ]["exact"]
            == "5/12"
        ),
        "random_support_receives_no_structured_credit": (
            not theorem[
                "random_support_model_transferred_to_structured_factor_base"
            ]
        ),
        "eight_actual_controls_complete": controls["control_count"] == 8,
        "thirty_actual_color_controls_complete": (
            controls["active_color_control_count"] == 30
        ),
        "all_actual_product_set_bounds_exact": (
            controls["all_color_product_sets_meet_cauchy_davenport"]
            and controls["all_color_controls_are_bound_tight"]
        ),
        "all_actual_positive_inverse_paths_exact": (
            controls["all_selected_positives_in_global_c5"]
            and controls["all_inverse_empties_outside_global_c5"]
            and controls[
                "all_positive_inverse_pairs_share_all_nonzero_path"
            ]
        ),
        "finite_controls_receive_no_asymptotic_credit": (
            not controls["finite_controls_receive_asymptotic_credit"]
        ),
        "three_plus_five_plus_and_nonzero_routes_preserved": all(
            routes[route_id]["status"] == "open"
            for route_id in (
                "structured_three_plus_mode_zero_test_tree",
                "five_plus_mode_low_slp_zero_test_tree",
                "nonzero_value_frobenius_coordinate_dag",
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
        "Probe the surviving node classes: structured three-plus-mode "
        "zero tests, five-plus-mode low-SLP zero-test circuits under the "
        "random support model, and nonzero-value Frobenius-coordinate "
        "branches. Freeze every circuit node and branch, replay exact "
        "positive and inverse-empty targets and C2+C3 sources, fit "
        "B^(9/4+o(1)) state and polylogarithmic arbitrary-target work, avoid "
        "field DLP, and charge rank, logs, identical descent, memory, field "
        "operations, and bits."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_binomial_node_union_depth.r137.v1"
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
                "structured zero-test decision trees whose every node is "
                "a represented binomial"
            ),
            (
                "uniform-random-support zero-test trees with at most four "
                "represented modes per node"
            ),
        ],
        "preserved_interface": (
            "structured three-plus-mode zero-test tree, five-plus-mode "
            "low-SLP tree, or nonzero-value Frobenius-coordinate DAG"
        ),
        "random_support_model_required_for_three_and_four_mode_depth": True,
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_binomial_node_union_depth_replay.r137.v1"
        ),
        "actual_control_count": controls["control_count"],
        "active_color_control_count": controls[
            "active_color_control_count"
        ],
        "all_color_product_sets_meet_cauchy_davenport": controls[
            "all_color_product_sets_meet_cauchy_davenport"
        ],
        "all_color_controls_are_bound_tight": controls[
            "all_color_controls_are_bound_tight"
        ],
        "all_positive_inverse_pairs_share_all_nonzero_path": controls[
            "all_positive_inverse_pairs_share_all_nonzero_path"
        ],
        "inside_cap_surviving_selector_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r137.v1",
        "r136_all_nonzero_path_product_audit_complete": True,
        "r137_binomial_node_union_depth_audit_complete": True,
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
        "CAUCHY_DAVENPORT_FORCES_EVERY_STRUCTURED_COLOR_FIVEFOLD_PRODUCT_"
        "SET_TO_AT_LEAST_FIVE_M_MINUS_FOUR__PRIME_ORDER_BINOMIAL_HAS_AT_"
        "MOST_ONE_ROOT__EXACT_BINOMIAL_ZERO_TEST_TREE_NEEDS_B_THREE_"
        "QUARTERS_ALL_NONZERO_PATH_DEPTH_REGARDLESS_OF_PRODUCT_EXPANSION__"
        "UNIFORM_RANDOM_SUPPORT_EXTENDS_POLYNOMIAL_DEPTH_TO_FOUR_MODE_"
        "NODES_ONLY__THIRTY_EXACT_PRODUCT_SET_AND_POSITIVE_INVERSE_EMPTY_"
        "CONTROLS__STRUCTURED_THREE_PLUS_FIVE_PLUS_LOW_SLP_NONZERO_VALUE_"
        "ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "BINOMIAL_NODE_UNION_DEPTH_AND_RANDOM_FOUR_MODE_NEGATIVE_"
            "ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "active_color_control_count": controls[
                "active_color_control_count"
            ],
            "all_color_product_sets_meet_cauchy_davenport": controls[
                "all_color_product_sets_meet_cauchy_davenport"
            ],
            "all_color_controls_are_bound_tight": controls[
                "all_color_controls_are_bound_tight"
            ],
            "all_positive_inverse_pairs_share_all_nonzero_path": controls[
                "all_positive_inverse_pairs_share_all_nonzero_path"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "structured_binomial_tree_negative_admitted": True,
            "random_model_four_mode_tree_negative_admitted": True,
            "structured_three_plus_mode_or_nonzero_selector_admitted": False,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_binomial_node_union_depth.json"
            ),
            "cost": (
                "torus_c5_binomial_node_union_depth_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_binomial_node_union_depth_replay.json"
            ),
            "controls": (
                "torus_c5_binomial_node_union_depth_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r137.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Cauchy-Davenport supplies only a linear structured product-set lower bound.",
            "The three- and four-mode polynomial depth bounds require the random support model.",
            "The random support model does not transfer to the structured factor base.",
            "Nonzero-value and coordinate tests are not covered.",
            "The exact small controls receive no probability or asymptotic credit.",
            "No general straight-line-program, circuit, RAM, or cell-probe lower bound is claimed.",
            "No source index, rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_CAUCHY_DAVENPORT_STRUCTURED_PRODUCT_SET_FLOOR__REJECT_"
            "ALL_STRUCTURED_BINOMIAL_ZERO_TEST_TREES_AT_POLYLOG_QUERY__"
            "ADMIT_RANDOM_SUPPORT_POLYNOMIAL_DEPTH_LOWER_BOUND_THROUGH_FOUR_"
            "MODES_ONLY__ADMIT_THIRTY_EXACT_PRODUCT_SET_AND_POSITIVE_"
            "INVERSE_EMPTY_CONTROLS_WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_"
            "STRUCTURED_THREE_PLUS_FIVE_PLUS_LOW_SLP_AND_NONZERO_VALUE_"
            "ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__"
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
            "p1553_torus_c5_binomial_node_union_depth_"
            "probe_report_r137.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_binomial_node_union_depth.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_binomial_node_union_depth_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_binomial_node_union_depth_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_binomial_node_union_depth_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r137.json"),
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
        f"R137 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
