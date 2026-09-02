#!/usr/bin/env python3
"""Audit a finite-field Chebotarev route to sparse C5 fiber covers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Sequence


SCHEMA = "p1553.torus_c5_chebotarev_fiber_cover.r138.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
SOURCE_DEGREE = 5
STRUCTURED_ATOM_EXPONENT_B = Fraction(3, 4)
SUBGROUP_ORDER_EXPONENT_B = Fraction(5)
WINDOW_MODE_COUNT = 16

R137_PRODUCER = pathlib.Path(
    "p1553_torus_c5_binomial_node_union_depth_probe_r137.py"
)
R137_PRODUCER_SHA256 = (
    "0120bb27e2fcc271d197f2eaae4b2adc5fd6ca65590973e9fce35c8f7221fbb1"
)
R137_REPORT = pathlib.Path(
    "p1553_torus_c5_binomial_node_union_depth_probe_report_r137.json"
)
R137_REPORT_SHA256 = (
    "9e52400ba4ff17fe8d07e2f48f00e59ab8dd977ae9cc21a9a09b119c6cc8588d"
)
R137_FROZEN = pathlib.Path(
    "frozen_torus_c5_binomial_node_union_depth.json"
)
R137_FROZEN_SHA256 = (
    "e8cf340e380f99d57d2db08e24bdaa507f902d7fd93df02b6773603e43f310e0"
)
R137_COST = pathlib.Path(
    "torus_c5_binomial_node_union_depth_cost_ledger.json"
)
R137_COST_SHA256 = (
    "45f178e841b3234607e7ae5706384f2553171be74de4a544e6eecc3973ae2ae5"
)
R137_REPLAY = pathlib.Path(
    "torus_c5_binomial_node_union_depth_replay.json"
)
R137_REPLAY_SHA256 = (
    "38cd8666637b5e7e3a6864a6177a160bc27d6b1395840960b423321db8fbd3d4"
)
R137_CONTROLS = pathlib.Path(
    "torus_c5_binomial_node_union_depth_controls.json"
)
R137_CONTROLS_SHA256 = (
    "9a3e9e71b2fd984d21feac2c1650adfd8cee3559a01ee62893f280c868ca1d31"
)
R137_LOGS = pathlib.Path("factor_logs_and_identical_descent_r137.json")
R137_LOGS_SHA256 = (
    "7e9d38cb8e3a6aec6d851837644ea6a813e9a0bae275bf4da6f0926140a3c6d6"
)
R137_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_binomial_node_union_depth_probe_r137.py"
)
R137_TEST_SHA256 = (
    "fb58d28c0d56125173b86c8eb2ce44aa11352fb76e271ae1a3b88487dc3d7a05"
)
R137_GATE = pathlib.Path(
    "p1553_torus_c5_binomial_node_union_depth_probe_gate_r137.md"
)
R137_GATE_SHA256 = (
    "88669b1d4a1177c747a3bddb2cbdcaa22352a3c705048417d07f0a80a6bb246e"
)
R137_PARENT = pathlib.Path(
    "p1553_torus_c5_binomial_node_union_depth_"
    "probe_parent_report_r137.yaml"
)
R137_PARENT_SHA256 = (
    "6000356936813d690d767e7952301a46c6738a460941ece6c6312a4a24fc5a47"
)
CHEBOTAREV_PDF = pathlib.Path(
    "references/emmrich_kunis_finite_field_chebotarev_2506.02947.pdf"
)
CHEBOTAREV_PDF_SHA256 = (
    "5a3b29fbaf5bd6f4833de1a63272bb7510179a92dc9b855d69b8b3dcb92cd627"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R137 = load_module("p1553_r137_for_r138", R137_PRODUCER)
R136 = R137.R136
R135 = R137.R135
R133 = R137.R133
R129 = R137.R129
R121 = R137.R121
R82 = R137.R82
Field = R137.Field
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r137_producer", R137_PRODUCER, R137_PRODUCER_SHA256),
        ("r137_report", R137_REPORT, R137_REPORT_SHA256),
        ("r137_frozen", R137_FROZEN, R137_FROZEN_SHA256),
        ("r137_cost", R137_COST, R137_COST_SHA256),
        ("r137_replay", R137_REPLAY, R137_REPLAY_SHA256),
        ("r137_controls", R137_CONTROLS, R137_CONTROLS_SHA256),
        ("r137_logs", R137_LOGS, R137_LOGS_SHA256),
        ("r137_test", R137_TEST, R137_TEST_SHA256),
        ("r137_gate", R137_GATE, R137_GATE_SHA256),
        ("r137_parent", R137_PARENT, R137_PARENT_SHA256),
        ("chebotarev_pdf", CHEBOTAREV_PDF, CHEBOTAREV_PDF_SHA256),
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
        raise AssertionError(f"R138 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def multiplicative_order_mod_prime(base: int, prime: int) -> int:
    if prime < 2 or not R82.R70.is_prime(prime):
        raise ValueError("modulus must be prime")
    reduced = base % prime
    if reduced == 0:
        raise ValueError("base must be nonzero modulo the prime")
    order = prime - 1
    residual = order
    factor = 2
    factors: list[int] = []
    while factor * factor <= residual:
        if residual % factor == 0:
            factors.append(factor)
            while residual % factor == 0:
                residual //= factor
        factor += 1 if factor == 2 else 2
    if residual > 1:
        factors.append(residual)
    for divisor in factors:
        while order % divisor == 0 and pow(
            reduced,
            order // divisor,
            prime,
        ) == 1:
            order //= divisor
    return order


def sparse_fiber_depth_lower_bound(
    atom_count: int,
    node_mode_count: int,
) -> int:
    if atom_count < 1:
        raise ValueError("atom count must be positive")
    if node_mode_count < 2:
        raise ValueError("node mode count must be at least two")
    return math.ceil(atom_count / (node_mode_count - 1))


def _projective_ratio(
    numerator: Fp2,
    denominator: Fp2,
    field: Field,
) -> Fp2:
    if denominator == field.zero:
        raise ZeroDivisionError("projective denominator is zero")
    return field.mul(numerator, field.inv(denominator))


def three_atom_all_mode_spark(
    atoms: Sequence[Fp2],
    subgroup_order: int,
    field: Field,
) -> dict[str, Any]:
    if len(atoms) != 3 or len(set(atoms)) != 3:
        raise ValueError("exactly three distinct atoms are required")
    x0, x1, x2 = atoms
    left_ratio = field.mul(x1, field.inv(x0))
    right_ratio = field.mul(x2, field.inv(x0))
    if (
        field.pow(left_ratio, subgroup_order) != field.one
        or field.pow(right_ratio, subgroup_order) != field.one
    ):
        raise AssertionError("normalized atom left the subgroup")
    left_power = field.one
    right_power = field.one
    seen: dict[Fp2, int] = {}
    collision: tuple[int, int] | None = None
    for exponent in range(1, subgroup_order):
        left_power = field.mul(left_power, left_ratio)
        right_power = field.mul(right_power, right_ratio)
        value = _projective_ratio(
            field.sub(left_power, field.one),
            field.sub(right_power, field.one),
            field,
        )
        previous = seen.get(value)
        if previous is not None:
            collision = (previous, exponent)
            break
        seen[value] = exponent
    return {
        "atom_count": 3,
        "subgroup_order": subgroup_order,
        "normalized_nonzero_mode_count": subgroup_order - 1,
        "normalized_projective_ratios_checked": len(seen),
        "collision_modes": list(collision) if collision else None,
        "all_three_by_three_character_minors_nonsingular": (
            collision is None
            and len(seen) == subgroup_order - 1
        ),
        "all_mode_triple_count": math.comb(subgroup_order, 3),
        "field_discrete_logarithms_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def window_three_minor_control(
    atoms: Sequence[Fp2],
    field: Field,
    mode_count: int = WINDOW_MODE_COUNT,
) -> dict[str, Any]:
    if mode_count < 3:
        raise ValueError("mode window must contain at least three modes")
    checked = 0
    first_singular: dict[str, list[int]] | None = None
    for rows in itertools.combinations(range(len(atoms)), 3):
        row_atoms = tuple(atoms[index] for index in rows)
        for modes in itertools.combinations(range(mode_count), 3):
            checked += 1
            matrix = R135.evaluation_matrix(row_atoms, modes, field)
            if R135.fp2_matrix_rank(matrix, field) < 3:
                first_singular = {
                    "rows": list(rows),
                    "modes": list(modes),
                }
                break
        if first_singular is not None:
            break
    return {
        "mode_window": [0, mode_count - 1],
        "row_triple_count": math.comb(len(atoms), 3),
        "mode_triple_count": math.comb(mode_count, 3),
        "minor_count_checked": checked,
        "first_singular_minor": first_singular,
        "all_checked_minors_nonsingular": first_singular is None,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    subgroup_order = curve["subgroup_order"]
    characteristic_order = multiplicative_order_mod_prime(
        field.p,
        subgroup_order,
    )
    color_sizes = [
        len(part) for part in R129.balanced_four_parts(len(deck))
    ]
    exact_all_mode = (
        three_atom_all_mode_spark(deck, subgroup_order, field)
        if len(deck) == 3
        else None
    )
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": subgroup_order,
        "field_prime_mod_subgroup_order": field.p % subgroup_order,
        "characteristic_order_mod_subgroup_order": characteristic_order,
        "chebotarev_primitive_order_required": subgroup_order - 1,
        "chebotarev_order_condition_satisfied": (
            characteristic_order == subgroup_order - 1
        ),
        "deck_size": len(deck),
        "balanced_color_sizes": color_sizes,
        "eligible_colored_atom_triple_count": sum(
            math.comb(size, 3) for size in color_sizes
        ),
        "window_three_minor_control": window_three_minor_control(
            deck,
            field,
        ),
        "all_mode_three_atom_control": exact_all_mode,
        "candidate_discrete_logs_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        finite_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    exact = [
        row["all_mode_three_atom_control"]
        for row in controls
        if row["all_mode_three_atom_control"] is not None
    ]
    return {
        "schema": (
            "p1553.torus_c5_chebotarev_fiber_cover_controls.r138.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "all_characteristic_orders_equal_two": all(
            row["characteristic_order_mod_subgroup_order"] == 2
            for row in controls
        ),
        "any_chebotarev_order_condition_satisfied": any(
            row["chebotarev_order_condition_satisfied"]
            for row in controls
        ),
        "window_minor_count_checked": sum(
            row["window_three_minor_control"]["minor_count_checked"]
            for row in controls
        ),
        "all_window_minors_nonsingular": all(
            row["window_three_minor_control"][
                "all_checked_minors_nonsingular"
            ]
            for row in controls
        ),
        "all_mode_three_atom_control_count": len(exact),
        "all_exact_three_atom_controls_full_spark": all(
            row["all_three_by_three_character_minors_nonsingular"]
            for row in exact
        ),
        "eligible_colored_atom_triple_count": sum(
            row["eligible_colored_atom_triple_count"]
            for row in controls
        ),
        "finite_controls_receive_asymptotic_credit": False,
        "candidate_discrete_logs_consumed": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "primary_source": {
            "authors": "Tarek Emmrich and Stefan Kunis",
            "title": "Real and finite field versions of Chebotarev's theorem",
            "identifier": "arXiv:2506.02947",
            "version_boundary": (
                "Corrected PDF: Theorem 16 and Corollary 17 assume that "
                "the characteristic is primitive modulo the prime Fourier "
                "size; Section 4.1 says no analogous explicit bound is "
                "proved for nonprimitive order."
            ),
            "pinned_sha256": CHEBOTAREV_PDF_SHA256,
        },
        "tuple_fiber_cover_lemma": (
            "Let A contain m nonzero atoms and let every t-column "
            "character-evaluation submatrix on A have full spark. A "
            "nonzero t-mode polynomial then has at most t-1 zeros on A. "
            "After fixing four coordinates of A^5, z=a1*a2*a3*a4*x "
            "only rescales its nonzero coefficients, so one node vanishes "
            "on at most (t-1)*m^4 ordered source tuples. If the all-nonzero "
            "leaf rejects, d node root sets cover all m^5 tuples only when "
            "d>=ceil(m/(t-1)). Product collisions do not weaken this "
            "pullback count."
        ),
        "conditional_trinomial_tree_bound": (
            "For t=3 and m=B^(3/4+o(1)), a rejecting all-nonzero leaf "
            "requires d>=ceil(m/2)=B^(3/4+o(1)). If that leaf accepts, "
            "the R133 global trinomial root bound forces polynomial depth "
            "to cover the q-o(q) complement. Thus atom-restricted "
            "three-column full spark would reject every polylog-query "
            "structured trinomial zero-test tree."
        ),
        "actual_order_obstruction": (
            "Every frozen norm-one family has field characteristic "
            "p=6q-1, hence p=-1 modulo prime subgroup order q and "
            "ord_q(p)=2. The corrected finite-field Chebotarev corollary "
            "requires ord_q(p)=q-1, so it does not certify these Fourier "
            "matrices or their atom-restricted submatrices."
        ),
        "extension_degree_charge": (
            "Forcing ord_q(p)=q-1 places primitive q-th roots first in "
            "F_(p^(q-1)). With q=B^(5+o(1)), merely representing one "
            "generic extension element uses q-1=B^(5+o(1)) base-field "
            "coordinates, already above the B^(9/4+o(1)) setup cap and "
            "incompatible with constant-degree FFE."
        ),
        "maximum_structured_node_mode_count_closed_unconditionally": 2,
        "maximum_structured_node_mode_count_closed_conditionally": 3,
        "chebotarev_transfer_to_actual_norm_one_families": False,
        "direct_atom_restricted_full_spark_theorem_complete": False,
        "not_covered": [
            "a characteristic-specific restricted-minor theorem for the actual atoms",
            "structured nodes with four or more represented modes",
            "five-plus-mode compact straight-line programs",
            "tests of nonzero field values or coordinate comparisons",
            "general arithmetic-circuit, RAM, or cell-probe lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_chebotarev_fiber_cover_cost_ledger.r138.v1"
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
                "route_id": "conditional_atom_full_spark_trinomial_tree",
                "minimum_rejecting_path_depth_exponent_B": fraction_record(
                    STRUCTURED_ATOM_EXPONENT_B
                ),
                "full_spark_hypothesis_certified": False,
                "status": "conditional_rejection_only",
            },
            {
                "route_id": "finite_field_chebotarev_generic_transfer",
                "actual_extension_degree": 2,
                "required_primitive_order_extension_exponent_B": (
                    fraction_record(SUBGROUP_ORDER_EXPONENT_B)
                ),
                "inside_setup_cap": False,
                "status": (
                    "rejected_as_generic_constant_degree_ffe_transfer"
                ),
            },
            {
                "route_id": "actual_atom_characteristic_specific_spark",
                "inside_cap_exact_tree_constructed": False,
                "status": "open",
            },
            {
                "route_id": "five_plus_mode_low_slp_zero_test_tree",
                "inside_cap_exact_tree_constructed": False,
                "status": "open",
            },
            {
                "route_id": "nonzero_value_frobenius_coordinate_dag",
                "inside_cap_exact_dag_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R137_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R137 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    routes = {row["route_id"]: row for row in cost["routes"]}
    obligations = {
        "eleven_source_bindings_verified": len(source_hashes) == 11,
        "r137_structured_binomial_boundary_inherited": (
            inherited["admission"]["structured_binomial_tree_negative_admitted"]
            and not inherited["admission"]["lane_admitted"]
        ),
        "corrected_primary_source_pinned": (
            theorem["primary_source"]["pinned_sha256"]
            == CHEBOTAREV_PDF_SHA256
        ),
        "primitive_order_requirement_explicit": (
            "primitive modulo" in theorem["primary_source"]["version_boundary"]
        ),
        "tuple_fiber_cover_lemma_explicit": (
            "(t-1)*m^4" in theorem["tuple_fiber_cover_lemma"]
            and "ceil(m/(t-1))" in theorem["tuple_fiber_cover_lemma"]
        ),
        "conditional_trinomial_B_three_quarters_explicit": (
            "B^(3/4+o(1))"
            in theorem["conditional_trinomial_tree_bound"]
        ),
        "all_actual_characteristic_orders_equal_two": controls[
            "all_characteristic_orders_equal_two"
        ],
        "actual_chebotarev_order_condition_rejected": (
            not controls["any_chebotarev_order_condition_satisfied"]
            and not theorem["chebotarev_transfer_to_actual_norm_one_families"]
        ),
        "primitive_extension_charge_B_five": (
            "B^(5+o(1))" in theorem["extension_degree_charge"]
            and not routes["finite_field_chebotarev_generic_transfer"][
                "inside_setup_cap"
            ]
        ),
        "eight_actual_window_controls_complete": (
            controls["control_count"] == 8
            and controls["window_minor_count_checked"] == 73920
        ),
        "all_actual_window_minors_nonsingular": controls[
            "all_window_minors_nonsingular"
        ],
        "two_all_mode_three_atom_controls_complete": (
            controls["all_mode_three_atom_control_count"] == 2
            and controls["all_exact_three_atom_controls_full_spark"]
        ),
        "finite_colors_too_small_for_three_atom_credit": (
            controls["eligible_colored_atom_triple_count"] == 0
            and not controls["finite_controls_receive_asymptotic_credit"]
        ),
        "direct_characteristic_specific_route_preserved": (
            routes["actual_atom_characteristic_specific_spark"]["status"]
            == "open"
        ),
        "five_plus_and_nonzero_routes_preserved": all(
            routes[route_id]["status"] == "open"
            for route_id in (
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
        "Probe a characteristic-specific three-column restricted-minor "
        "theorem for the actual norm-one atom set, or construct a surviving "
        "four-plus-mode or nonzero-value selector. Freeze every mode, "
        "coefficient, circuit node, branch, and reverse C2+C3 source pointer; "
        "replay positives and inverse empties; fit B^(9/4+o(1)) state and "
        "polylog arbitrary-target work; avoid field DLP; and charge rank, "
        "logs, identical descent, memory, field operations, extension degree, "
        "and bits."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_chebotarev_fiber_cover.r138.v1"
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
            "constant_extension_degree_required": True,
        },
        "closed_scoped_grammars": [
            (
                "generic transfer of the corrected primitive-order finite-"
                "field Chebotarev theorem to the actual degree-two norm-one "
                "families"
            )
        ],
        "conditional_closed_grammar": (
            "structured trinomial zero-test trees if actual atom-restricted "
            "three-column full spark is proved"
        ),
        "preserved_interface": (
            "characteristic-specific restricted-minor theorem, four-plus-"
            "mode low-SLP tree, or nonzero-value Frobenius-coordinate DAG"
        ),
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_chebotarev_fiber_cover_replay.r138.v1"
        ),
        "actual_control_count": controls["control_count"],
        "window_minor_count_checked": controls["window_minor_count_checked"],
        "all_window_minors_nonsingular": controls[
            "all_window_minors_nonsingular"
        ],
        "all_mode_three_atom_control_count": controls[
            "all_mode_three_atom_control_count"
        ],
        "all_exact_three_atom_controls_full_spark": controls[
            "all_exact_three_atom_controls_full_spark"
        ],
        "all_characteristic_orders_equal_two": controls[
            "all_characteristic_orders_equal_two"
        ],
        "inside_cap_surviving_selector_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r138.v1",
        "r137_binomial_node_union_depth_audit_complete": True,
        "r138_chebotarev_fiber_cover_audit_complete": True,
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
        "FULL_SPARK_T_MODE_ATOM_CODE_IMPLIES_NODE_FIBER_WEIGHT_AT_MOST_T_"
        "MINUS_ONE_TIMES_M_FOUR_AND_REJECTING_PATH_DEPTH_AT_LEAST_M_OVER_"
        "T_MINUS_ONE__TRINOMIAL_TREE_CONDITIONALLY_B_THREE_QUARTERS__"
        "CORRECTED_FINITE_FIELD_CHEBOTAREV_REQUIRES_PRIMITIVE_"
        "CHARACTERISTIC_ORDER_BUT_ACTUAL_NORM_ONE_FAMILIES_HAVE_ORDER_TWO__"
        "FORCING_PRIMITIVE_ORDER_COSTS_EXTENSION_DEGREE_B_FIVE__TWO_EXACT_"
        "ALL_MODE_AND_73920_WINDOW_MINOR_CONTROLS_WITHOUT_ASYMPTOTIC_"
        "CREDIT__DIRECT_CHARACTERISTIC_SPECIFIC_SPARK_FOUR_PLUS_NONZERO_"
        "ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "CONDITIONAL_FIBER_COVER_AND_CHEBOTAREV_TRANSFER_NEGATIVE_"
            "ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "window_minor_count_checked": controls[
                "window_minor_count_checked"
            ],
            "all_window_minors_nonsingular": controls[
                "all_window_minors_nonsingular"
            ],
            "all_mode_three_atom_control_count": controls[
                "all_mode_three_atom_control_count"
            ],
            "eligible_colored_atom_triple_count": controls[
                "eligible_colored_atom_triple_count"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "conditional_trinomial_tree_negative_admitted": True,
            "generic_chebotarev_transfer_negative_admitted": True,
            "actual_atom_restricted_spark_admitted": False,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_torus_c5_chebotarev_fiber_cover.json",
            "cost": "torus_c5_chebotarev_fiber_cover_cost_ledger.json",
            "source_replay": (
                "torus_c5_chebotarev_fiber_cover_replay.json"
            ),
            "controls": "torus_c5_chebotarev_fiber_cover_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r138.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The fiber-cover lower bound is conditional on atom-restricted full spark.",
            "The corrected primitive-order theorem does not apply to order-two norm-one FFE.",
            "Finite uncolored controls do not prove the asymptotic colored property.",
            "No characteristic-specific restricted-minor theorem is supplied.",
            "Four-plus-mode, low-SLP, nonzero-value, and coordinate tests remain open.",
            "No general straight-line-program, circuit, RAM, or cell-probe lower bound is claimed.",
            "No source index, rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_TUPLE_FIBER_COVER_LEMMA_AND_CONDITIONAL_TRINOMIAL_DEPTH__"
            "REJECT_GENERIC_PRIMITIVE_ORDER_CHEBOTAREV_TRANSFER_TO_DEGREE_"
            "TWO_NORM_ONE_FFE__ADMIT_TWO_EXACT_ALL_MODE_AND_73920_WINDOW_"
            "MINOR_CONTROLS_WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_DIRECT_"
            "CHARACTERISTIC_SPECIFIC_SPARK_FOUR_PLUS_LOW_SLP_AND_NONZERO_"
            "VALUE_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_"
            "SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_chebotarev_fiber_cover_probe_report_r138.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_chebotarev_fiber_cover.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_chebotarev_fiber_cover_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_chebotarev_fiber_cover_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_chebotarev_fiber_cover_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r138.json"),
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
        f"R138 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
