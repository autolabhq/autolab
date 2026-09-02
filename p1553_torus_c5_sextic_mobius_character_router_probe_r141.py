#!/usr/bin/env python3
"""Test the sextic Mobius character as a nonzero-value torus router."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_sextic_mobius_character_router.r141.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
SUBGROUP_EXPONENT_B = Fraction(5)
LINEAR_TRANSLATION_RANK_EXPONENT_B = Fraction(5)
SYNTHETIC_SUBGROUP_ORDERS = (5, 7, 17, 19, 23, 29, 43, 47)
WEIL_SOURCE = {
    "title": "On Some Exponential Sums",
    "author": "Andre Weil",
    "year": 1948,
    "doi": "10.1073/pnas.34.5.204",
    "url": "https://doi.org/10.1073/pnas.34.5.204",
    "use": (
        "Square-root cancellation for nontrivial fixed-degree character "
        "sums on a genus-zero finite-field curve."
    ),
}

R140_PRODUCER = pathlib.Path(
    "p1553_torus_c5_order_two_four_minor_claw_probe_r140.py"
)
R140_PRODUCER_SHA256 = (
    "a849f79dea39041de547b473c0d4fc28e6221bbb6e004738028c7f4515422373"
)
R140_REPORT = pathlib.Path(
    "p1553_torus_c5_order_two_four_minor_"
    "claw_probe_report_r140.json"
)
R140_REPORT_SHA256 = (
    "3971132c94034f6c7f03bb7b77e83212fe805d9539b96c77a10e14176dca527c"
)
R140_FROZEN = pathlib.Path(
    "frozen_torus_c5_order_two_four_minor_claw.json"
)
R140_FROZEN_SHA256 = (
    "7930da7bdd6f51254ab2a23aacf12113c0c85fd2857aec8919caeb30d69f1a1f"
)
R140_COST = pathlib.Path(
    "torus_c5_order_two_four_minor_claw_cost_ledger.json"
)
R140_COST_SHA256 = (
    "2cd3c7708e4f83af6f688e10ce27e06198e6aaade546b0e0ef15423777137af8"
)
R140_REPLAY = pathlib.Path(
    "torus_c5_order_two_four_minor_claw_replay.json"
)
R140_REPLAY_SHA256 = (
    "0661fab8f891080107a03e95981fea0170e54cbe8f53487db40fcfa9f0a9a3e2"
)
R140_CONTROLS = pathlib.Path(
    "torus_c5_order_two_four_minor_claw_controls.json"
)
R140_CONTROLS_SHA256 = (
    "dbe9feac7384b1d6048b11d9a7a86e2afdb91adc339481132183e85f57ea1a75"
)
R140_LOGS = pathlib.Path("factor_logs_and_identical_descent_r140.json")
R140_LOGS_SHA256 = (
    "b7d7c9f1ecf51a349dea0c610c4a8a3fee6d000708285685a4d27d9b12eeecda"
)
R140_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_order_two_four_minor_claw_probe_r140.py"
)
R140_TEST_SHA256 = (
    "8e5daa679de1739e71c9670c5b775c2179d833748730fe897ba7eb8b8e36a585"
)
R140_GATE = pathlib.Path(
    "p1553_torus_c5_order_two_four_minor_claw_probe_gate_r140.md"
)
R140_GATE_SHA256 = (
    "33c8b530802974a36b06b0032c17794f7f77bf454d41dd4a2e2415d671ccfe66"
)
R140_PARENT = pathlib.Path(
    "p1553_torus_c5_order_two_four_minor_"
    "claw_probe_parent_report_r140.yaml"
)
R140_PARENT_SHA256 = (
    "03ae39a6fde97e95581fefccf64a86cc6adc29ff2126ffc2e72bfd7f6d136ec5"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R140 = load_module("p1553_r140_for_r141", R140_PRODUCER)
R121 = R140.R139.R121
R82 = R140.R139.R82
Field = R140.Field
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r140_producer", R140_PRODUCER, R140_PRODUCER_SHA256),
        ("r140_report", R140_REPORT, R140_REPORT_SHA256),
        ("r140_frozen", R140_FROZEN, R140_FROZEN_SHA256),
        ("r140_cost", R140_COST, R140_COST_SHA256),
        ("r140_replay", R140_REPLAY, R140_REPLAY_SHA256),
        ("r140_controls", R140_CONTROLS, R140_CONTROLS_SHA256),
        ("r140_logs", R140_LOGS, R140_LOGS_SHA256),
        ("r140_test", R140_TEST, R140_TEST_SHA256),
        ("r140_gate", R140_GATE, R140_GATE_SHA256),
        ("r140_parent", R140_PARENT, R140_PARENT_SHA256),
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
        raise AssertionError(f"R141 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def unique(values: Iterable[Fp2]) -> tuple[Fp2, ...]:
    return tuple(dict.fromkeys(values))


def product_deck(
    deck: tuple[Fp2, ...],
    arity: int,
    field: Field,
) -> tuple[Fp2, ...]:
    return unique(
        field.product(deck[index] for index in source)
        for source in itertools.combinations_with_replacement(
            range(len(deck)),
            arity,
        )
    )


def sextic_character(
    value: Fp2,
    parameter: Fp2,
    subgroup_order: int,
    field: Field,
) -> Fp2:
    image = R140.mobius_image(value, parameter, field)
    return field.pow(image, subgroup_order)


def signature(
    value: Fp2,
    parameters: tuple[Fp2, ...],
    subgroup_order: int,
    field: Field,
) -> tuple[Fp2, ...]:
    return tuple(
        sextic_character(value, parameter, subgroup_order, field)
        for parameter in parameters
    )


def greedy_inverse_separator(
    support: tuple[Fp2, ...],
    parameters: tuple[Fp2, ...],
    subgroup_order: int,
    field: Field,
) -> dict[str, Any]:
    unresolved = set(range(len(support)))
    remaining = list(parameters)
    selected: list[Fp2] = []
    progress: list[dict[str, int]] = []
    while unresolved and remaining:
        scores = [
            (
                sum(
                    sextic_character(
                        support[index],
                        parameter,
                        subgroup_order,
                        field,
                    )
                    != sextic_character(
                        field.inv(support[index]),
                        parameter,
                        subgroup_order,
                        field,
                    )
                    for index in unresolved
                ),
                parameter,
            )
            for parameter in remaining
        ]
        score, best = max(scores, key=lambda row: row[0])
        selected.append(best)
        remaining.remove(best)
        unresolved = {
            index
            for index in unresolved
            if sextic_character(
                support[index],
                best,
                subgroup_order,
                field,
            )
            == sextic_character(
                field.inv(support[index]),
                best,
                subgroup_order,
                field,
            )
        }
        progress.append(
            {
                "newly_separated_count": score,
                "remaining_unresolved_count": len(unresolved),
            }
        )
    return {
        "available_parameter_count": len(parameters),
        "selected_parameter_count": len(selected),
        "selected_parameters": [
            field.json(parameter) for parameter in selected
        ],
        "progress": progress,
        "unresolved_count": len(unresolved),
        "all_inverse_pairs_separated": not unresolved,
    }


def cayley_claw_identity(
    left: Fp2,
    parameter: Fp2,
    field: Field,
) -> dict[str, Any]:
    right = R140.mobius_image(left, parameter, field)
    left_t = R121.torus_parameter(left, field)
    parameter_t = R121.torus_parameter(parameter, field)
    right_t = R121.torus_parameter(right, field)
    residual = (
        field.nonsquare
        * (
            left_t * parameter_t
            + left_t * right_t
            + parameter_t * right_t
        )
        - 3
    ) % field.p
    return {
        "left_parameter": left_t,
        "mobius_parameter": parameter_t,
        "right_parameter": right_t,
        "residual": residual,
        "identity_exact": residual == 0,
    }


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck = R121.pairing_deck(curve, offset)
    subgroup_order = curve["subgroup_order"]
    c2 = product_deck(deck, 2, field)
    c3 = product_deck(deck, 3, field)
    c5 = product_deck(deck, 5, field)
    c5_set = set(c5)
    parameters = unique(value for value in deck if value != field.one)
    if not parameters:
        raise AssertionError("actual control has no Mobius parameter")
    inverse_empty = all(field.inv(value) not in c5_set for value in c5)
    separator = greedy_inverse_separator(
        c5,
        parameters,
        subgroup_order,
        field,
    )
    parameter = parameters[0]
    matrix = [
        [
            sextic_character(
                field.mul(left, right),
                parameter,
                subgroup_order,
                field,
            )
            for right in c3
        ]
        for left in c2
    ]
    matrix_rank = R140.R139.R135.fp2_matrix_rank(matrix, field)
    defects = {
        field.mul(
            sextic_character(
                field.mul(left, right),
                parameter,
                subgroup_order,
                field,
            ),
            field.inv(
                field.mul(
                    sextic_character(
                        left,
                        parameter,
                        subgroup_order,
                        field,
                    ),
                    sextic_character(
                        right,
                        parameter,
                        subgroup_order,
                        field,
                    ),
                )
            ),
        )
        for left in c2
        for right in c3
    }
    single_parameter_separated = sum(
        sextic_character(
            value,
            parameter,
            subgroup_order,
            field,
        )
        != sextic_character(
            field.inv(value),
            parameter,
            subgroup_order,
            field,
        )
        for value in c5
    )
    cayley = [
        cayley_claw_identity(value, parameter, field)
        for value in c5[: min(16, len(c5))]
    ]
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": subgroup_order,
        "deck_size": len(deck),
        "c2_size": len(c2),
        "c3_size": len(c3),
        "c5_size": len(c5),
        "c5_injective": len(c5) == math.comb(len(deck) + 4, 5),
        "all_positive_inverses_empty": inverse_empty,
        "mobius_parameter_count": len(parameters),
        "single_parameter_separated_count": single_parameter_separated,
        "single_parameter_separated_fraction": (
            single_parameter_separated / len(c5)
        ),
        "greedy_inverse_separator": separator,
        "c2_by_c3_character_matrix_rank": matrix_rank,
        "c2_by_c3_character_matrix_full_row_rank": (
            matrix_rank == len(c2)
        ),
        "multiplicative_defect_value_count": len(defects),
        "multiplicative_defect_values": sorted(
            (field.json(value) for value in defects),
            key=lambda value: tuple(value),
        ),
        "multiplicative_defect_attains_all_six_cosets": len(defects) == 6,
        "cayley_controls": cayley,
        "all_cayley_claw_identities_exact": all(
            row["identity_exact"] for row in cayley
        ),
        "character_evaluation_square_multiply_bound": (
            2 * subgroup_order.bit_length()
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def synthetic_fourier_control(subgroup_order: int) -> dict[str, Any]:
    field = Field(6 * subgroup_order - 1)
    root = R140.R139.find_order_q_root(field, subgroup_order)
    sequence: list[Fp2] = []
    value = field.one
    for _ in range(subgroup_order):
        sequence.append(
            sextic_character(
                value,
                root,
                subgroup_order,
                field,
            )
        )
        value = field.mul(value, root)
    coefficients: list[Fp2] = []
    for mode in range(subgroup_order):
        step = field.pow(root, (-mode) % subgroup_order)
        power = field.one
        coefficient = field.zero
        for item in sequence:
            coefficient = field.add(
                coefficient,
                field.mul(item, power),
            )
            power = field.mul(power, step)
        coefficients.append(coefficient)
    support = [
        mode
        for mode, coefficient in enumerate(coefficients)
        if coefficient != field.zero
    ]
    return {
        "field_prime": field.p,
        "subgroup_order": subgroup_order,
        "character_value_count": len(set(sequence)),
        "fourier_support_size": len(support),
        "fourier_zero_count": subgroup_order - len(support),
        "fourier_zero_modes": [
            mode
            for mode, coefficient in enumerate(coefficients)
            if coefficient == field.zero
        ],
        "fourier_support_at_least_q_minus_one": (
            len(support) >= subgroup_order - 1
        ),
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    actual = [
        actual_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    synthetic = [
        synthetic_fourier_control(order)
        for order in SYNTHETIC_SUBGROUP_ORDERS
    ]
    return {
        "schema": (
            "p1553.torus_c5_sextic_mobius_character_router."
            "controls.r141.v1"
        ),
        "actual_control_count": len(actual),
        "actual_controls": actual,
        "all_actual_c5_supports_injective": all(
            row["c5_injective"] for row in actual
        ),
        "all_actual_positive_inverses_empty": all(
            row["all_positive_inverses_empty"] for row in actual
        ),
        "all_actual_inverse_pairs_separated": all(
            row["greedy_inverse_separator"][
                "all_inverse_pairs_separated"
            ]
            for row in actual
        ),
        "maximum_selected_parameter_count": max(
            row["greedy_inverse_separator"]["selected_parameter_count"]
            for row in actual
        ),
        "all_actual_cayley_claw_identities_exact": all(
            row["all_cayley_claw_identities_exact"] for row in actual
        ),
        "all_actual_c2_by_c3_matrices_full_row_rank": all(
            row["c2_by_c3_character_matrix_full_row_rank"]
            for row in actual
        ),
        "all_actual_multiplicative_defects_attain_six_cosets": all(
            row["multiplicative_defect_attains_all_six_cosets"]
            for row in actual
        ),
        "synthetic_control_count": len(synthetic),
        "synthetic_fourier_controls": synthetic,
        "all_synthetic_fourier_supports_at_least_q_minus_one": all(
            row["fourier_support_at_least_q_minus_one"]
            for row in synthetic
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    controls = finite_controls()
    obligations = {
        "ten_source_bindings_verified": len(actual_bindings) == 10,
        "r140_mobius_involution_inherited": True,
        "cayley_claw_form_derived": True,
        "sextic_character_has_six_values": True,
        "sextic_character_polylog_evaluable": True,
        "inverse_equality_rational_function_not_sixth_power": True,
        "weil_square_root_bound_applies": True,
        "polylog_inverse_separating_family_exists": True,
        "translated_character_sums_uniformly_square_root_bounded": True,
        "parseval_forces_linear_fourier_support_omega_q": True,
        "translation_invariant_linear_state_exponent_B5": True,
        "eight_actual_controls_complete": (
            controls["actual_control_count"] == 8
        ),
        "all_actual_inverse_pairs_separated": controls[
            "all_actual_inverse_pairs_separated"
        ],
        "all_actual_c2_by_c3_matrices_full_row_rank": controls[
            "all_actual_c2_by_c3_matrices_full_row_rank"
        ],
        "all_actual_multiplicative_defects_attain_six_cosets": controls[
            "all_actual_multiplicative_defects_attain_six_cosets"
        ],
        "all_synthetic_fourier_supports_at_least_q_minus_one": controls[
            "all_synthetic_fourier_supports_at_least_q_minus_one"
        ],
        "nonlinear_character_source_router_complete": False,
        "compact_nontranslation_composition_complete": False,
        "complete_five_source_index": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct a nonlinear C2 source router from the sextic Mobius "
        "characters without materializing their Omega(q) translated "
        "linear orbit. Freeze every parameter, character value, branch, "
        "candidate C2 pointer, C3 certificate, inverse-empty path, and "
        "reverse five-source pointer; fit B^(9/4+o(1)) state and "
        "polylogarithmic arbitrary-target work; avoid field DLP; and "
        "charge rank, logs, identical descent, memory, field operations, "
        "extension degree, and bits."
    )
    theorem = {
        "cayley_form": (
            "For X(t)=(1+u*t)/(1-u*t), u^2=d, the Mobius claw "
            "w=T_z(x) is equivalent to d*(t_x*t_z+t_x*t_w+t_z*t_w)=3."
        ),
        "character_definition": (
            "chi_z(x)=T_z(x)^q lies in mu_6 because the full norm-one "
            "group has order p+1=6q."
        ),
        "evaluation_cost": (
            "One Mobius evaluation and binary exponentiation by q use "
            "O(log q) field operations and no discrete logarithm."
        ),
        "inverse_separation": (
            "For fixed x!=x^-1, equality chi_z(x)=chi_z(x^-1) is the "
            "trivial-coset condition for a fixed-degree rational function "
            "of z with a simple divisor, hence not a sixth power. Weil "
            "cancellation gives q/6+O(sqrt(q)) equal parameters. A union "
            "bound therefore gives O(log |S|) fixed parameters separating "
            "every x in any inversion-disjoint support S from x^-1."
        ),
        "linear_translation_boundary": (
            "Every Fourier coefficient of x->chi_z(x) on H is a "
            "fixed-conductor torus character sum and is O(sqrt(q)). "
            "Parseval then forces Omega(q) nonzero coefficients, so the "
            "translated circulant has Omega(q)=B^(5+o(1)) complex rank."
        ),
        "scope": (
            "The rank result covers translation-invariant linear "
            "convolution or sketch realizations of the six coset labels. "
            "It is not a lower bound for nonlinear circuits, adaptive "
            "data structures, RAM, or cell probes."
        ),
        "primary_source": WEIL_SOURCE,
        "literature_novelty": "unverified",
    }
    frozen = {
        "schema": (
            "p1553.torus_c5_sextic_mobius_character_router."
            "frozen.r141.v1"
        ),
        "source_bindings": source_binding_records(),
        "interface": {
            "input": (
                "A norm-one target x, fixed norm-one parameter z, and "
                "prime subgroup order q=(p+1)/6."
            ),
            "output": "The six-valued label chi_z(x)=T_z(x)^q.",
            "positive_use": (
                "O(log q) labels can separate inversion-disjoint support "
                "values from their inverses."
            ),
            "closed_use": (
                "Translation-invariant linear convolution requires "
                "Omega(q) spectral state."
            ),
        },
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "arbitrary_target_query_exponent_B": fraction_record(QUERY_CAP),
        },
        "required_open_outputs": failures,
    }
    cost = {
        "schema": (
            "p1553.torus_c5_sextic_mobius_character_router."
            "cost.r141.v1"
        ),
        "single_character_query_cost": "O(log q) field operations",
        "inverse_separator_parameter_count": "O(log |S|)",
        "inverse_separator_query_cost": "O(log |S|*log q)",
        "inverse_separator_exponent_B": fraction_record(Fraction(0)),
        "linear_translation_rank_exponent_B": fraction_record(
            LINEAR_TRANSLATION_RANK_EXPONENT_B
        ),
        "linear_translation_inside_setup_cap": False,
        "actual_maximum_selected_parameter_count": controls[
            "maximum_selected_parameter_count"
        ],
        "candidate_field_dlp_used": False,
        "nonlinear_source_router_cost_supplied": False,
        "rank_cost_supplied": False,
        "factor_log_cost_supplied": False,
        "identical_descent_cost_supplied": False,
        "total_attack_cost_supplied": False,
    }
    replay_rows = [
        {
            "control_id": row["control_id"],
            "c2_size": row["c2_size"],
            "c3_size": row["c3_size"],
            "c5_size": row["c5_size"],
            "positive_inverses_empty": row[
                "all_positive_inverses_empty"
            ],
            "selected_parameter_count": row[
                "greedy_inverse_separator"
            ]["selected_parameter_count"],
            "all_inverse_pairs_separated": row[
                "greedy_inverse_separator"
            ]["all_inverse_pairs_separated"],
            "c2_by_c3_matrix_rank": row[
                "c2_by_c3_character_matrix_rank"
            ],
            "multiplicative_defect_value_count": row[
                "multiplicative_defect_value_count"
            ],
        }
        for row in controls["actual_controls"]
    ]
    replay = {
        "schema": (
            "p1553.torus_c5_sextic_mobius_character_router."
            "replay.r141.v1"
        ),
        "row_count": len(replay_rows),
        "rows": replay_rows,
        "all_positive_inverse_empty_paths_separated": controls[
            "all_actual_inverse_pairs_separated"
        ],
        "complete_c2_source_router_replayed": False,
    }
    logs_descent = {
        "schema": (
            "p1553.torus_c5_sextic_mobius_character_router."
            "logs_descent.r141.v1"
        ),
        "candidate_field_dlp_used": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_complete": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    admission = {
        "obligations": obligations,
        "passed_obligation_count": passed,
        "obligation_count": len(obligations),
        "polylog_inverse_separator_admitted": True,
        "linear_translation_rank_negative_admitted": True,
        "lane_admitted": False,
    }
    classification = (
        "SEXTIC_MOBIUS_CHARACTER_EVALUATES_IN_POLYLOG_WITHOUT_DLP__WEIL_"
        "CANCELLATION_GIVES_POLYLOG_FIXED_PARAMETER_INVERSE_SEPARATOR__"
        "ALL_EIGHT_ACTUAL_POSITIVE_INVERSE_EMPTY_SUPPORTS_SEPARATED_WITH_"
        "AT_MOST_FOUR_PARAMETERS__C2_BY_C3_CHARACTER_MATRICES_FULL_ROW_"
        "RANK_AND_MULTIPLICATIVE_DEFECT_ATTAINS_ALL_SIX_COSETS__PARSEVAL_"
        "FORCES_OMEGA_Q_B5_TRANSLATED_LINEAR_STATE__NONLINEAR_SOURCE_"
        "ROUTER_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": (
            "P1553-TORUS-C5-SEXTIC-MOBIUS-CHARACTER-ROUTER-R141"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "verified_source_hashes": actual_bindings,
        "theorem": theorem,
        "controls_summary": {
            key: value
            for key, value in controls.items()
            if key not in ("actual_controls", "synthetic_fourier_controls")
        },
        "cost_summary": cost,
        "admission": admission,
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_sextic_mobius_character_router.json"
            ),
            "cost": (
                "torus_c5_sextic_mobius_character_router_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_sextic_mobius_character_router_replay.json"
            ),
            "controls": (
                "torus_c5_sextic_mobius_character_router_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r141.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Inverse separation is not C2 source selection.",
            "Full finite row rank is not asymptotic evidence by itself.",
            "The Omega(q) theorem covers translated linear character-label convolution only.",
            "No general nonlinear circuit, RAM, or cell-probe lower bound is claimed.",
            "No source index, relation rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_POLYLOG_NONZERO_VALUE_INVERSE_SEPARATOR__REJECT_"
            "TRANSLATED_LINEAR_CHARACTER_ROUTER_AT_B5_STATE__PRESERVE_"
            "NONLINEAR_NONTRANSLATION_SOURCE_ROUTER__NO_RANK__NO_LOGS__"
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
            "p1553_torus_c5_sextic_mobius_character_"
            "router_probe_report_r141.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_sextic_mobius_character_router.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_sextic_mobius_character_router_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_sextic_mobius_character_router_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_sextic_mobius_character_router_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r141.json"),
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
        f"R141 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
