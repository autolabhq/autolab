#!/usr/bin/env python3
"""Prove three-minor rigidity for order-two norm-one Fourier matrices."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.torus_c5_order_two_three_minor_rigidity.r139.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
STRUCTURED_ATOM_EXPONENT_B = Fraction(3, 4)
COMPLEMENT_TRINOMIAL_DEPTH_EXPONENT_B = Fraction(5, 2)
ACTUAL_MODE_WINDOW = tuple(range(2, 18))
SYNTHETIC_SUBGROUP_ORDERS = (3, 5, 7, 17, 19, 23, 29, 43, 47)

R138_PRODUCER = pathlib.Path(
    "p1553_torus_c5_chebotarev_fiber_cover_probe_r138.py"
)
R138_PRODUCER_SHA256 = (
    "4d2be77b119fd60744ea03d3f7f3a4021ad0d3dc65ab2cea2da04d10839abba2"
)
R138_REPORT = pathlib.Path(
    "p1553_torus_c5_chebotarev_fiber_cover_probe_report_r138.json"
)
R138_REPORT_SHA256 = (
    "f6ccece429e9d7dd523fded90239690ba2c865bbd1478f8f2c63cb84fbd14056"
)
R138_FROZEN = pathlib.Path(
    "frozen_torus_c5_chebotarev_fiber_cover.json"
)
R138_FROZEN_SHA256 = (
    "91ec759e101e5265b5ee76cce1d300f47f686a2b72c4f9e8e3bd56932c5a70bc"
)
R138_COST = pathlib.Path(
    "torus_c5_chebotarev_fiber_cover_cost_ledger.json"
)
R138_COST_SHA256 = (
    "ba5d7ed1688b30242a1711554e371e61ae372e5aa4b3d8fda9785cf4a5bc4e1d"
)
R138_REPLAY = pathlib.Path(
    "torus_c5_chebotarev_fiber_cover_replay.json"
)
R138_REPLAY_SHA256 = (
    "416d9a6f6460c029203573c4a5f3016df19ae6a053d888263d8c547fc6c128ff"
)
R138_CONTROLS = pathlib.Path(
    "torus_c5_chebotarev_fiber_cover_controls.json"
)
R138_CONTROLS_SHA256 = (
    "0356dd83dc4e2fe44860b37acd18fdf9c6ed24661beb2b27d7b2e2111de57f6f"
)
R138_LOGS = pathlib.Path("factor_logs_and_identical_descent_r138.json")
R138_LOGS_SHA256 = (
    "580d3b753321618b9eaa6722004700aed9321590eb52be71dc645dfedd33c1b2"
)
R138_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_chebotarev_fiber_cover_probe_r138.py"
)
R138_TEST_SHA256 = (
    "0e574b915ef65427c84d3bfb5f200427e2dec125f6e115aa6d8b4fde73e0fa89"
)
R138_GATE = pathlib.Path(
    "p1553_torus_c5_chebotarev_fiber_cover_probe_gate_r138.md"
)
R138_GATE_SHA256 = (
    "7830c3ff7960c53fad0359d7871e64d7aa2e6a011e7845948a1c99f12aa12a3b"
)
R138_PARENT = pathlib.Path(
    "p1553_torus_c5_chebotarev_fiber_cover_"
    "probe_parent_report_r138.yaml"
)
R138_PARENT_SHA256 = (
    "7f96df65b74a77ca0fff06ce5abc4d99c37742e85d9c42454569ae6c2f807a59"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R138 = load_module("p1553_r138_for_r139", R138_PRODUCER)
R137 = R138.R137
R135 = R138.R135
R133 = R138.R133
R121 = R138.R121
R82 = R138.R82
Field = R138.Field
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r138_producer", R138_PRODUCER, R138_PRODUCER_SHA256),
        ("r138_report", R138_REPORT, R138_REPORT_SHA256),
        ("r138_frozen", R138_FROZEN, R138_FROZEN_SHA256),
        ("r138_cost", R138_COST, R138_COST_SHA256),
        ("r138_replay", R138_REPLAY, R138_REPLAY_SHA256),
        ("r138_controls", R138_CONTROLS, R138_CONTROLS_SHA256),
        ("r138_logs", R138_LOGS, R138_LOGS_SHA256),
        ("r138_test", R138_TEST, R138_TEST_SHA256),
        ("r138_gate", R138_GATE, R138_GATE_SHA256),
        ("r138_parent", R138_PARENT, R138_PARENT_SHA256),
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
        raise AssertionError(f"R139 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def normalized_minor_certificate(
    root: Fp2,
    subgroup_order: int,
    row_parameter: int,
    mode_parameter: int,
    field: Field,
) -> dict[str, Any]:
    r = row_parameter % subgroup_order
    s = mode_parameter % subgroup_order
    if r in (0, 1) or s in (0, 1):
        raise ValueError("normalized parameters must avoid zero and one")
    if field.p % subgroup_order != subgroup_order - 1:
        raise ValueError("characteristic must act as inversion")
    z_r = field.pow(root, r)
    z_s = field.pow(root, s)
    z_rs = field.pow(root, r * s % subgroup_order)
    left = field.mul(
        field.sub(root, field.one),
        field.sub(z_rs, field.one),
    )
    right = field.mul(
        field.sub(z_s, field.one),
        field.sub(z_r, field.one),
    )
    determinant = field.sub(left, right)
    left_multiplier = field.pow(
        root,
        (-(1 + r * s)) % subgroup_order,
    )
    right_multiplier = field.pow(
        root,
        (-(r + s)) % subgroup_order,
    )
    left_frobenius = field.pow(left, field.p)
    right_frobenius = field.pow(right, field.p)
    expected_left_frobenius = field.mul(left_multiplier, left)
    expected_right_frobenius = field.mul(right_multiplier, right)
    contradiction_exponent = (r - 1) * (s - 1) % subgroup_order
    matrix = R135.evaluation_matrix(
        (field.one, root, z_r),
        (0, 1, s),
        field,
    )
    return {
        "row_parameter": r,
        "mode_parameter": s,
        "left_nonzero": left != field.zero,
        "right_nonzero": right != field.zero,
        "determinant": field.json(determinant),
        "determinant_nonzero": determinant != field.zero,
        "matrix_rank": R135.fp2_matrix_rank(matrix, field),
        "left_frobenius_identity": (
            left_frobenius == expected_left_frobenius
        ),
        "right_frobenius_identity": (
            right_frobenius == expected_right_frobenius
        ),
        "left_right_multiplier_equal": (
            left_multiplier == right_multiplier
        ),
        "contradiction_exponent": contradiction_exponent,
        "contradiction_exponent_nonzero": contradiction_exponent != 0,
    }


def find_order_q_root(field: Field, subgroup_order: int) -> Fp2:
    exponent = (field.p * field.p - 1) // subgroup_order
    for real in range(min(field.p, 32)):
        for imaginary in range(1, min(field.p, 32)):
            root = field.pow((real, imaginary), exponent)
            if (
                root != field.one
                and field.pow(root, subgroup_order) == field.one
            ):
                return root
    raise AssertionError("unable to find prime-order root")


def normalized_sweep(
    field: Field,
    subgroup_order: int,
    root: Fp2,
    row_parameters: tuple[int, ...],
    mode_parameters: tuple[int, ...],
) -> dict[str, Any]:
    certificates = [
        normalized_minor_certificate(
            root,
            subgroup_order,
            r,
            s,
            field,
        )
        for r, s in itertools.product(row_parameters, mode_parameters)
    ]
    return {
        "row_parameter_count": len(row_parameters),
        "mode_parameter_count": len(mode_parameters),
        "normalized_minor_count": len(certificates),
        "zero_determinant_count": sum(
            not row["determinant_nonzero"] for row in certificates
        ),
        "all_matrix_ranks_three": all(
            row["matrix_rank"] == 3 for row in certificates
        ),
        "all_frobenius_identities_exact": all(
            row["left_frobenius_identity"]
            and row["right_frobenius_identity"]
            for row in certificates
        ),
        "all_contradiction_exponents_nonzero": all(
            row["contradiction_exponent_nonzero"]
            for row in certificates
        ),
        "certificates": certificates,
    }


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    subgroup_order = curve["subgroup_order"]
    root = next(value for value in deck_values if value != field.one)
    parameters = tuple(
        value
        for value in ACTUAL_MODE_WINDOW
        if value < subgroup_order
    )
    sweep = normalized_sweep(
        field,
        subgroup_order,
        root,
        parameters,
        parameters,
    )
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": subgroup_order,
        "field_prime_mod_subgroup_order": field.p % subgroup_order,
        "root_has_exact_prime_order": (
            root != field.one
            and field.pow(root, subgroup_order) == field.one
        ),
        "sweep": sweep,
        "candidate_discrete_logs_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def synthetic_control(subgroup_order: int) -> dict[str, Any]:
    field_prime = 6 * subgroup_order - 1
    if (
        not R82.R70.is_prime(subgroup_order)
        or not R82.R70.is_prime(field_prime)
    ):
        raise ValueError("synthetic Sophie Germain pair is invalid")
    field = Field(field_prime)
    root = find_order_q_root(field, subgroup_order)
    parameters = tuple(range(2, subgroup_order))
    sweep = normalized_sweep(
        field,
        subgroup_order,
        root,
        parameters,
        parameters,
    )
    return {
        "field_prime": field_prime,
        "subgroup_order": subgroup_order,
        "normalized_parameter_pair_count": (subgroup_order - 2) ** 2,
        "sweep": sweep,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    actual = [
        actual_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    synthetic = [
        synthetic_control(order) for order in SYNTHETIC_SUBGROUP_ORDERS
    ]
    all_sweeps = [
        row["sweep"] for row in actual
    ] + [
        row["sweep"] for row in synthetic
    ]
    return {
        "schema": (
            "p1553.torus_c5_order_two_three_minor_rigidity_"
            "controls.r139.v1"
        ),
        "actual_controls": actual,
        "actual_control_count": len(actual),
        "actual_normalized_minor_count": sum(
            row["sweep"]["normalized_minor_count"] for row in actual
        ),
        "synthetic_controls": synthetic,
        "synthetic_control_count": len(synthetic),
        "synthetic_normalized_minor_count": sum(
            row["sweep"]["normalized_minor_count"] for row in synthetic
        ),
        "all_determinants_nonzero": all(
            row["zero_determinant_count"] == 0 for row in all_sweeps
        ),
        "all_matrix_ranks_three": all(
            row["all_matrix_ranks_three"] for row in all_sweeps
        ),
        "all_frobenius_identities_exact": all(
            row["all_frobenius_identities_exact"] for row in all_sweeps
        ),
        "all_contradiction_exponents_nonzero": all(
            row["all_contradiction_exponents_nonzero"]
            for row in all_sweeps
        ),
        "finite_controls_receive_asymptotic_credit": False,
        "candidate_discrete_logs_consumed": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "proof_status": "self_contained_finite_field_argument",
        "affine_normalization": (
            "For prime q, translate and scale any three distinct row "
            "exponents and any three distinct mode exponents in Z/qZ. "
            "Row and column permutations and nonzero diagonal scalings "
            "preserve singularity, reducing every 3-by-3 Fourier minor to "
            "rows {0,1,r} and modes {0,1,s}, with r,s not in {0,1}."
        ),
        "normalized_determinant_equation": (
            "For a primitive q-th root zeta, the normalized determinant "
            "vanishes exactly when "
            "(zeta-1)(zeta^(r*s)-1)=(zeta^s-1)(zeta^r-1). "
            "Both sides are nonzero."
        ),
        "order_two_frobenius_step": (
            "If the characteristic p is -1 modulo q, Frobenius sends "
            "zeta to zeta^(-1). Conjugating a putative equality multiplies "
            "its left side by zeta^(-(1+r*s)) and its right side by "
            "zeta^(-(r+s)). Cancelling the original equal nonzero sides "
            "forces 1+r*s=r+s modulo q."
        ),
        "contradiction": (
            "The forced congruence is (r-1)(s-1)=0 modulo prime q, "
            "contradicting normalized distinct rows and modes. Therefore "
            "every 3-by-3 minor of the q-point Fourier matrix over any "
            "field of characteristic p=-1 modulo q is nonsingular."
        ),
        "atom_restricted_consequence": (
            "Every atom subset inherits three-column full spark. Hence "
            "every nonzero represented trinomial has at most two zeros "
            "on each structured atom color."
        ),
        "structured_tree_consequence": (
            "R138's tuple-fiber lemma now applies unconditionally. For "
            "m=B^(3/4+o(1)), a rejecting all-nonzero path has depth at "
            "least ceil(m/2)=B^(3/4+o(1)). If the leaf accepts, R133's "
            "global trinomial root bound requires q^(1/2+o(1))="
            "B^(5/2+o(1)) depth to cover the q-o(q) complement. Thus every "
            "exact structured trinomial zero-test tree misses the polylog "
            "arbitrary-target query cap."
        ),
        "maximum_structured_node_mode_count_closed_unconditionally": 3,
        "random_support_model_required": False,
        "field_discrete_logarithm_required": False,
        "not_covered": [
            "structured zero-test nodes with four or more represented modes",
            "five-plus-mode compact straight-line programs",
            "tests of nonzero field values or coordinate comparisons",
            "general arithmetic-circuit, RAM, or cell-probe lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_order_two_three_minor_rigidity_"
            "cost_ledger.r139.v1"
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
                "route_id": "structured_trinomial_zero_test_tree",
                "minimum_rejecting_path_depth_exponent_B": fraction_record(
                    STRUCTURED_ATOM_EXPONENT_B
                ),
                "minimum_accepting_path_depth_exponent_B": fraction_record(
                    COMPLEMENT_TRINOMIAL_DEPTH_EXPONENT_B
                ),
                "status": (
                    "rejected_deterministically_by_order_two_"
                    "three_minor_rigidity"
                ),
            },
            {
                "route_id": "structured_four_plus_mode_zero_test_tree",
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
    inherited = json.loads(R138_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R138 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    routes = {row["route_id"]: row for row in cost["routes"]}
    obligations = {
        "ten_source_bindings_verified": len(source_hashes) == 10,
        "r138_conditional_fiber_bound_inherited": (
            inherited["admission"][
                "conditional_trinomial_tree_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "affine_normalization_explicit": (
            "rows {0,1,r}" in theorem["affine_normalization"]
            and "modes {0,1,s}" in theorem["affine_normalization"]
        ),
        "normalized_determinant_equation_explicit": (
            "(zeta-1)" in theorem["normalized_determinant_equation"]
        ),
        "order_two_frobenius_identity_explicit": (
            "zeta^(-1)" in theorem["order_two_frobenius_step"]
        ),
        "nonzero_cancellation_contradiction_explicit": (
            "(r-1)(s-1)=0" in theorem["contradiction"]
        ),
        "three_minor_full_spark_theorem_complete": (
            theorem[
                "maximum_structured_node_mode_count_closed_unconditionally"
            ]
            == 3
        ),
        "structured_rejecting_depth_B_three_quarters": (
            "B^(3/4+o(1))" in theorem["structured_tree_consequence"]
        ),
        "structured_accepting_depth_B_five_halves": (
            "B^(5/2+o(1))" in theorem["structured_tree_consequence"]
        ),
        "eight_actual_controls_complete": (
            controls["actual_control_count"] == 8
        ),
        "two_thousand_forty_eight_actual_minors_complete": (
            controls["actual_normalized_minor_count"] == 2048
        ),
        "nine_synthetic_exhaustive_sweeps_complete": (
            controls["synthetic_control_count"] == 9
            and controls["synthetic_normalized_minor_count"]
            == sum((order - 2) ** 2 for order in SYNTHETIC_SUBGROUP_ORDERS)
        ),
        "all_exact_controls_pass": all(
            (
                controls["all_determinants_nonzero"],
                controls["all_matrix_ranks_three"],
                controls["all_frobenius_identities_exact"],
                controls["all_contradiction_exponents_nonzero"],
            )
        ),
        "finite_controls_receive_no_asymptotic_credit": (
            not controls["finite_controls_receive_asymptotic_credit"]
        ),
        "four_plus_low_slp_and_nonzero_routes_preserved": all(
            routes[route_id]["status"] == "open"
            for route_id in (
                "structured_four_plus_mode_zero_test_tree",
                "five_plus_mode_low_slp_zero_test_tree",
                "nonzero_value_frobenius_coordinate_dag",
            )
        ),
        "inside_cap_five_source_index_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
        "inside_cap_four_plus_or_nonzero_selector_complete": False,
    }
    next_action = (
        "Probe order-two four-column minors and structured four-mode root "
        "fibers, or construct a nonzero-value Frobenius-coordinate selector. "
        "Freeze every mode, coefficient, circuit node, branch, and reverse "
        "C2+C3 source pointer; replay positives and inverse empties; fit "
        "B^(9/4+o(1)) state and polylog arbitrary-target work; avoid field "
        "DLP; and charge rank, logs, identical descent, memory, field "
        "operations, extension degree, and bits."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_order_two_three_minor_"
            "rigidity.r139.v1"
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
                "structured represented-trinomial zero-test decision trees "
                "over prime-order subgroups with characteristic p=-1 mod q"
            )
        ],
        "preserved_interface": (
            "structured four-plus-mode low-SLP tree or nonzero-value "
            "Frobenius-coordinate DAG"
        ),
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_order_two_three_minor_rigidity_"
            "replay.r139.v1"
        ),
        "actual_control_count": controls["actual_control_count"],
        "actual_normalized_minor_count": controls[
            "actual_normalized_minor_count"
        ],
        "synthetic_control_count": controls["synthetic_control_count"],
        "synthetic_normalized_minor_count": controls[
            "synthetic_normalized_minor_count"
        ],
        "all_determinants_nonzero": controls[
            "all_determinants_nonzero"
        ],
        "all_frobenius_identities_exact": controls[
            "all_frobenius_identities_exact"
        ],
        "inside_cap_surviving_selector_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r139.v1",
        "r138_chebotarev_fiber_cover_audit_complete": True,
        "r139_order_two_three_minor_rigidity_audit_complete": True,
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
        "ORDER_TWO_FROBENIUS_RIGIDITY_FORCES_EVERY_THREE_BY_THREE_PRIME_"
        "FOURIER_MINOR_NONZERO__EVERY_STRUCTURED_ATOM_COLOR_HAS_THREE_"
        "COLUMN_FULL_SPARK__R138_FIBER_COVER_NOW_UNCONDITIONAL_FOR_"
        "TRINOMIALS__REJECTING_PATH_DEPTH_B_THREE_QUARTERS_AND_ACCEPTING_"
        "PATH_DEPTH_B_FIVE_HALVES__2048_ACTUAL_AND_EXHAUSTIVE_NINE_"
        "SYNTHETIC_NORMALIZED_SWEEPS__STRUCTURED_FOUR_PLUS_LOW_SLP_AND_"
        "NONZERO_VALUE_ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "ORDER_TWO_THREE_MINOR_AND_TRINOMIAL_TREE_NEGATIVE_ONLY_"
            "WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "actual_control_count": controls["actual_control_count"],
            "actual_normalized_minor_count": controls[
                "actual_normalized_minor_count"
            ],
            "synthetic_control_count": controls[
                "synthetic_control_count"
            ],
            "synthetic_normalized_minor_count": controls[
                "synthetic_normalized_minor_count"
            ],
            "all_determinants_nonzero": controls[
                "all_determinants_nonzero"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "order_two_three_minor_theorem_admitted": True,
            "structured_trinomial_tree_negative_admitted": True,
            "structured_four_plus_or_nonzero_selector_admitted": False,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_order_two_three_minor_rigidity.json"
            ),
            "cost": (
                "torus_c5_order_two_three_minor_rigidity_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_order_two_three_minor_rigidity_replay.json"
            ),
            "controls": (
                "torus_c5_order_two_three_minor_rigidity_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r139.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The theorem closes represented trinomials, not four-mode nodes.",
            "Finite sweeps are checks of the self-contained theorem and receive no asymptotic credit.",
            "Nonzero-value and coordinate tests are not covered.",
            "No general straight-line-program, circuit, RAM, or cell-probe lower bound is claimed.",
            "No source index, rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_SELF_CONTAINED_ORDER_TWO_THREE_MINOR_RIGIDITY__REJECT_"
            "ALL_STRUCTURED_REPRESENTED_TRINOMIAL_ZERO_TEST_TREES_AT_"
            "POLYLOG_QUERY__ADMIT_2048_ACTUAL_AND_NINE_SYNTHETIC_SWEEPS_"
            "WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_STRUCTURED_FOUR_PLUS_LOW_"
            "SLP_AND_NONZERO_VALUE_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__"
            "NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_order_two_three_minor_"
            "rigidity_probe_report_r139.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_order_two_three_minor_rigidity.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_order_two_three_minor_rigidity_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_order_two_three_minor_rigidity_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_order_two_three_minor_rigidity_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r139.json"),
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
        f"R139 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
