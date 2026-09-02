#!/usr/bin/env python3
"""Audit moment, torus, and subfunction routes for multiplicative C5."""

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


SCHEMA = "p1553.m6_small_k_multiplicative_c5_moment_torus.r121.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
C_ATOM_EXPONENT = Fraction(3, 4)
C2_EXPONENT = 2 * C_ATOM_EXPONENT
C3_EXPONENT = 3 * C_ATOM_EXPONENT
C5_EXPONENT = 5 * C_ATOM_EXPONENT

R120_PRODUCER = pathlib.Path(
    "p1553_m6_suboutput_implicit_c5_character_pairing_probe_r120.py"
)
R120_PRODUCER_SHA256 = (
    "e758a1fed3cb0a41d18d74fd0fd4ffd05f047bd4d37d2d9be15f821195a9a131"
)
R120_REPORT = pathlib.Path(
    "p1553_m6_suboutput_implicit_c5_character_"
    "pairing_probe_report_r120.json"
)
R120_REPORT_SHA256 = (
    "de3e063e1d11a61cea49896a9a1e5e252b0e167cd558e6b50affe733bdeb60e5"
)
R120_FROZEN = pathlib.Path(
    "frozen_m6_suboutput_implicit_c5_membership_circuit.json"
)
R120_FROZEN_SHA256 = (
    "1a4f30674052f3b6ab598b79fd06d6fed300c23f9ca6001a09826f7a514715c8"
)
R120_COST = pathlib.Path("m6_c5_implicit_circuit_cost_ledger.json")
R120_COST_SHA256 = (
    "0e12afd3f31209dd707c022280c3868eec5a7d5533afcd7692f5bb19c0f71a42"
)
R120_REPLAY = pathlib.Path("m6_c5_implicit_circuit_source_replay.json")
R120_REPLAY_SHA256 = (
    "ab50bec7273b718bcde1492df984d4dac054973a0706674d04b8b595b19aa600"
)
R120_CONTROLS = pathlib.Path(
    "m6_c5_implicit_circuit_exceptional_controls.json"
)
R120_CONTROLS_SHA256 = (
    "3dc277dab4854e9a4da6de31d7eb953ca228ac232d6cb7cffa586788988f2244"
)
R120_LOGS = pathlib.Path("factor_logs_and_identical_descent_r120.json")
R120_LOGS_SHA256 = (
    "3b79bac599448a55a16d43a85af68e1519a72dfcb8ac1d74f9e2716bf5f40cbf"
)
R120_GATE = pathlib.Path(
    "p1553_m6_suboutput_implicit_c5_character_pairing_probe_gate_r120.md"
)
R120_GATE_SHA256 = (
    "ab8efbe571f310a510f943e45f9fb65bac34fd37b2ee4d04337cdf3a5c9c8da1"
)
R120_PARENT = pathlib.Path(
    "p1553_m6_suboutput_implicit_c5_character_"
    "pairing_probe_parent_report_r120.yaml"
)
R120_PARENT_SHA256 = (
    "4e51a5d10a1debf66f7b4305f173cd818914e48e61512e37d11a6ea45e55169d"
)
R118_GATE = pathlib.Path(
    "p1553_m6_nonlinear_value_sensitive_c6_source_locator_probe_gate_r118.md"
)
R118_GATE_SHA256 = (
    "9ed7f59fb94dd1ecff54d8844068502183a7928c64e991c9a98b270ab8645b9d"
)
R90_GATE = pathlib.Path(
    "p1553_5a5c_nonlocal_moment_hankel_translation_probe_gate_r90.md"
)
R90_GATE_SHA256 = (
    "b1618f6a354b995db01fbbc7aeeb69df6ebb5248b5c558bc4c72d87ce523897b"
)
R91_GATE = pathlib.Path(
    "p1553_5a5c_unequal_list_subfunction_inversion_probe_gate_r91.md"
)
R91_GATE_SHA256 = (
    "57469654e0b535ccbf1d62edd9548cd827a09d244deaecb06b8aedbd410ac6d3"
)
TENSOR_TRACE_RED_TEAM = pathlib.Path("p1553_tensor_trace_minpoly_red_team.md")
TENSOR_TRACE_RED_TEAM_SHA256 = (
    "a534f6260a342bacc1aeba6becce8a35707644ed6fcd16eb2a75cfb7dbbe0c3c"
)
DINUR_GOLOVNEV_PDF = pathlib.Path(
    "references/dinur_golovnev_3sum_indexing_2512.04258v2.pdf"
)
DINUR_GOLOVNEV_PDF_SHA256 = (
    "e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c"
)

Fp2 = tuple[int, int]
Point = tuple[int, int] | None


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R120 = load_module("p1553_r120_for_r121", R120_PRODUCER)
R119 = R120.R119
R82 = R120.R82
Field = R120.QuadraticField


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r120_producer", R120_PRODUCER, R120_PRODUCER_SHA256),
        ("r120_report", R120_REPORT, R120_REPORT_SHA256),
        ("r120_frozen", R120_FROZEN, R120_FROZEN_SHA256),
        ("r120_cost", R120_COST, R120_COST_SHA256),
        ("r120_replay", R120_REPLAY, R120_REPLAY_SHA256),
        ("r120_controls", R120_CONTROLS, R120_CONTROLS_SHA256),
        ("r120_logs", R120_LOGS, R120_LOGS_SHA256),
        ("r120_gate", R120_GATE, R120_GATE_SHA256),
        ("r120_parent", R120_PARENT, R120_PARENT_SHA256),
        ("r118_gate", R118_GATE, R118_GATE_SHA256),
        ("r90_gate", R90_GATE, R90_GATE_SHA256),
        ("r91_gate", R91_GATE, R91_GATE_SHA256),
        (
            "tensor_trace_red_team",
            TENSOR_TRACE_RED_TEAM,
            TENSOR_TRACE_RED_TEAM_SHA256,
        ),
        (
            "dinur_golovnev_pdf",
            DINUR_GOLOVNEV_PDF,
            DINUR_GOLOVNEV_PDF_SHA256,
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
        raise AssertionError(f"R121 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def field_sum(field: Field, values: Iterable[Fp2]) -> Fp2:
    result = field.zero
    for value in values:
        result = field.add(result, value)
    return result


def torus_parameter(value: Fp2, field: Field) -> int:
    """Return t for z=(1+u*t)/(1-u*t); -1 is the omitted chart point."""
    a_coord, b_coord = value
    denominator = (a_coord + 1) % field.p
    if denominator == 0:
        raise ZeroDivisionError("the Cayley chart omits z=-1")
    return b_coord * pow(denominator, field.p - 2, field.p) % field.p


def torus_lift(parameter: int, field: Field) -> Fp2:
    square = parameter * parameter % field.p
    denominator = (1 - field.nonsquare * square) % field.p
    inverse = pow(denominator, field.p - 2, field.p)
    return (
        (1 + field.nonsquare * square) * inverse % field.p,
        2 * parameter * inverse % field.p,
    )


def elementary_symmetric(values: tuple[int, ...], prime: int) -> list[int]:
    result = [1] + [0] * len(values)
    degree = 0
    for value in values:
        degree += 1
        for index in range(degree, 0, -1):
            result[index] = (
                result[index] + value * result[index - 1]
            ) % prime
    return result


def torus_five_form(
    parameters: tuple[int, ...],
    target_parameter: int,
    field: Field,
) -> dict[str, int]:
    if len(parameters) != 5:
        raise ValueError("the torus C5 form requires five parameters")
    e = elementary_symmetric(parameters, field.p)
    d = field.nonsquare
    even = (1 + d * e[2] + d * d * e[4]) % field.p
    odd = (e[1] + d * e[3] + d * d * e[5]) % field.p
    return {
        "even": even,
        "odd": odd,
        "residual": (odd - target_parameter * even) % field.p,
    }


def complete_homogeneous_five(
    deck: tuple[Fp2, ...],
    exponent: int,
    field: Field,
) -> Fp2:
    powered = tuple(field.pow(value, exponent) for value in deck)
    power_sums = [field.zero]
    for degree in range(1, 6):
        power_sums.append(
            field_sum(
                field,
                (field.pow(value, degree) for value in powered),
            )
        )
    complete = [field.one]
    for degree in range(1, 6):
        numerator = field_sum(
            field,
            (
                field.mul(power_sums[index], complete[degree - index])
                for index in range(1, degree + 1)
            ),
        )
        complete.append(field.div(numerator, field.elt(degree)))
    return complete[5]


def berlekamp_massey(
    sequence: tuple[Fp2, ...],
    field: Field,
) -> tuple[int, tuple[Fp2, ...]]:
    connection = [field.one]
    previous = [field.one]
    order = 0
    shift = 1
    previous_discrepancy = field.one
    for position, value in enumerate(sequence):
        discrepancy = value
        for index in range(1, order + 1):
            discrepancy = field.add(
                discrepancy,
                field.mul(connection[index], sequence[position - index]),
            )
        if discrepancy == field.zero:
            shift += 1
            continue
        old_connection = connection[:]
        scale = field.div(discrepancy, previous_discrepancy)
        required = len(previous) + shift
        if len(connection) < required:
            connection.extend(
                [field.zero] * (required - len(connection))
            )
        for index, coefficient in enumerate(previous):
            location = index + shift
            connection[location] = field.sub(
                connection[location],
                field.mul(scale, coefficient),
            )
        if 2 * order <= position:
            order = position + 1 - order
            previous = old_connection
            previous_discrepancy = discrepancy
            shift = 1
        else:
            shift += 1
    return order, tuple(connection[: order + 1])


def root_annihilator(
    roots: tuple[Fp2, ...],
    field: Field,
) -> tuple[Fp2, ...]:
    """Return [1,a1,...,aM] for prod(T-root)=T^M+a1*T^(M-1)+..."""
    coefficients = [field.one]
    for root in roots:
        updated = [field.zero] * (len(coefficients) + 1)
        updated[0] = coefficients[0]
        for index in range(1, len(coefficients)):
            updated[index] = field.sub(
                coefficients[index],
                field.mul(root, coefficients[index - 1]),
            )
        updated[-1] = field.neg(field.mul(root, coefficients[-1]))
        coefficients = updated
    return tuple(coefficients)


def evaluate_monic(
    coefficients: tuple[Fp2, ...],
    value: Fp2,
    field: Field,
) -> Fp2:
    result = coefficients[0]
    for coefficient in coefficients[1:]:
        result = field.add(field.mul(result, value), coefficient)
    return result


def pairing_deck(
    curve: dict[str, Any],
    offset: int,
) -> tuple[Field, tuple[Point, ...], tuple[Fp2, ...]]:
    field = Field(curve["field_prime"])
    generator = R120.point2(R119.R117.R81.curve_generator(curve), field)
    distortion = R120.j_zero_distortion(generator, field)
    _, deck, _, _ = R82.compact_factor_base(curve, offset)
    images = tuple(
        R120.reduced_tate_pairing(
            R120.point2(point, field),
            distortion,
            curve["subgroup_order"],
            curve,
            field,
        )
        for point in deck
    )
    return field, tuple(deck), images


def split_index(
    deck: tuple[Fp2, ...],
    field: Field,
) -> tuple[dict[Fp2, tuple[int, ...]], tuple[tuple[int, ...], ...]]:
    triples: dict[Fp2, tuple[int, ...]] = {}
    for source in itertools.combinations_with_replacement(
        range(len(deck)),
        3,
    ):
        value = field.product(deck[index] for index in source)
        triples.setdefault(value, source)
    pairs = tuple(
        itertools.combinations_with_replacement(range(len(deck)), 2)
    )
    return triples, pairs


def split_query(
    target: Fp2,
    deck: tuple[Fp2, ...],
    triples: dict[Fp2, tuple[int, ...]],
    pairs: tuple[tuple[int, ...], ...],
    field: Field,
) -> tuple[int, ...] | None:
    for pair in pairs:
        pair_value = field.product(deck[index] for index in pair)
        complement = field.div(target, pair_value)
        if complement in triples:
            return tuple(sorted(pair + triples[complement]))
    return None


def finite_control(
    curve: dict[str, Any],
    offset: int,
    inherited: dict[str, Any],
) -> dict[str, Any]:
    field, point_deck, deck = pairing_deck(curve, offset)
    order = curve["subgroup_order"]
    sources = tuple(
        itertools.combinations_with_replacement(range(len(deck)), 5)
    )
    products = tuple(
        field.product(deck[index] for index in source)
        for source in sources
    )
    if len(set(products)) != len(products):
        raise AssertionError("R121 requires the inherited injective controls")

    norm_one = all(field.pow(value, field.p + 1) == field.one for value in deck)
    chart_complete = all(value != field.neg(field.one) for value in deck)
    parameters = tuple(torus_parameter(value, field) for value in deck)
    chart_roundtrips = all(
        torus_lift(parameter, field) == value
        for parameter, value in zip(parameters, deck)
    )
    binary_law_exact = True
    for left, right in itertools.product(range(len(deck)), repeat=2):
        product = field.mul(deck[left], deck[right])
        product_parameter = torus_parameter(product, field)
        denominator = (
            1
            + field.nonsquare * parameters[left] * parameters[right]
        ) % field.p
        expected = (
            (parameters[left] + parameters[right])
            * pow(denominator, field.p - 2, field.p)
        ) % field.p
        binary_law_exact &= product_parameter == expected

    all_five_forms_exact = True
    all_five_even_denominators_nonzero = True
    for source, product in zip(sources, products):
        source_parameters = tuple(parameters[index] for index in source)
        target_parameter = torus_parameter(product, field)
        form = torus_five_form(source_parameters, target_parameter, field)
        all_five_forms_exact &= form["residual"] == 0
        all_five_even_denominators_nonzero &= form["even"] != 0

    moment_count = 2 * len(products)
    direct_moments = tuple(
        field_sum(field, (field.pow(value, exponent) for value in products))
        for exponent in range(moment_count)
    )
    compact_moments = tuple(
        complete_homogeneous_five(deck, exponent, field)
        for exponent in range(moment_count)
    )
    moment_identity_exact = direct_moments == compact_moments
    recurrence_order, recurrence = berlekamp_massey(
        compact_moments,
        field,
    )
    annihilator = root_annihilator(products, field)
    recurrence_equals_annihilator = recurrence == annihilator
    all_positive_roots_zero = all(
        evaluate_monic(annihilator, product, field) == field.zero
        for product in products
    )
    empty_image = tuple(inherited["empty_target_image"])
    empty_annihilator_nonzero = (
        evaluate_monic(annihilator, empty_image, field) != field.zero
    )

    triples, pairs = split_index(deck, field)
    all_split_sources_exact = True
    all_split_point_sources_replay = True
    for source, product in zip(sources, products):
        recovered = split_query(product, deck, triples, pairs, field)
        all_split_sources_exact &= recovered == source
        if recovered is None:
            all_split_point_sources_replay = False
        else:
            all_split_point_sources_replay &= (
                R82.add_many(
                    (point_deck[index] for index in recovered),
                    curve,
                )
                == R82.add_many(
                    (point_deck[index] for index in source),
                    curve,
                )
            )
    empty_split_rejected = (
        split_query(empty_image, deck, triples, pairs, field) is None
    )

    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": order,
        "embedding_degree": 2,
        "deck_size": len(deck),
        "canonical_c5_source_count": len(sources),
        "distinct_product_count": len(set(products)),
        "product_map_injective": len(set(products)) == len(products),
        "all_pairing_images_norm_one": norm_one,
        "minus_one_absent_from_odd_q_subgroup_deck": chart_complete,
        "all_cayley_chart_roundtrips_exact": chart_roundtrips,
        "all_binary_torus_laws_exact": binary_law_exact,
        "all_c5_torus_forms_exact": all_five_forms_exact,
        "all_c5_even_denominators_nonzero": (
            all_five_even_denominators_nonzero
        ),
        "moment_prefix_length": moment_count,
        "direct_moment_sha256": sha256_json(
            [field.json(value) for value in direct_moments]
        ),
        "compact_h5_moment_sha256": sha256_json(
            [field.json(value) for value in compact_moments]
        ),
        "complete_homogeneous_moment_identity_exact": moment_identity_exact,
        "berlekamp_massey_order": recurrence_order,
        "berlekamp_massey_order_equals_distinct_support": (
            recurrence_order == len(products)
        ),
        "recurrence_coefficient_sha256": sha256_json(
            [field.json(value) for value in recurrence]
        ),
        "annihilator_coefficient_sha256": sha256_json(
            [field.json(value) for value in annihilator]
        ),
        "recurrence_equals_full_product_annihilator": (
            recurrence_equals_annihilator
        ),
        "all_positive_annihilator_evaluations_zero": all_positive_roots_zero,
        "empty_annihilator_evaluation_nonzero": empty_annihilator_nonzero,
        "c2_occurrence_count": len(pairs),
        "c3_distinct_product_count": len(triples),
        "all_c2_c3_split_sources_exact": all_split_sources_exact,
        "all_c2_c3_projective_sources_replay": (
            all_split_point_sources_replay
        ),
        "empty_c2_c3_split_query_rejected": empty_split_rejected,
        "candidate_scalar_labels_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    inherited_payload = json.loads(
        R120_CONTROLS.read_text(encoding="utf-8")
    )
    inherited = {
        row["control_id"]: row
        for row in inherited_payload["r82_pairing_controls"]
    }
    rows = []
    for curve in R82.FAMILIES:
        for offset in (0, 1):
            control_id = f"{curve['family_id']}_offset{offset}"
            rows.append(
                finite_control(curve, offset, inherited[control_id])
            )
    return {
        "schema": (
            "p1553.m6_small_k_multiplicative_c5_"
            "exceptional_controls.r121.v1"
        ),
        "r82_torus_moment_controls": rows,
        "control_count": len(rows),
        "all_product_maps_injective": all(
            row["product_map_injective"] for row in rows
        ),
        "all_pairing_images_norm_one_and_charted": all(
            row["all_pairing_images_norm_one"]
            and row["minus_one_absent_from_odd_q_subgroup_deck"]
            and row["all_cayley_chart_roundtrips_exact"]
            for row in rows
        ),
        "all_torus_laws_and_c5_forms_exact": all(
            row["all_binary_torus_laws_exact"]
            and row["all_c5_torus_forms_exact"]
            and row["all_c5_even_denominators_nonzero"]
            for row in rows
        ),
        "all_compact_moment_identities_exact": all(
            row["complete_homogeneous_moment_identity_exact"]
            for row in rows
        ),
        "all_bm_orders_equal_distinct_c5_support": all(
            row["berlekamp_massey_order_equals_distinct_support"]
            and row["recurrence_equals_full_product_annihilator"]
            for row in rows
        ),
        "all_annihilator_membership_controls_exact": all(
            row["all_positive_annihilator_evaluations_zero"]
            and row["empty_annihilator_evaluation_nonzero"]
            for row in rows
        ),
        "all_c2_c3_sources_and_empty_queries_exact": all(
            row["all_c2_c3_split_sources_exact"]
            and row["all_c2_c3_projective_sources_replay"]
            and row["empty_c2_c3_split_query_rejected"]
            for row in rows
        ),
        "candidate_scalar_labels_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def algebraic_identities() -> dict[str, Any]:
    return {
        "norm_one_cayley_chart": {
            "field": "Fp2=Fp[u]/(u^2-d), d a nonsquare",
            "chart": "z(t)=(1+u*t)/(1-u*t)",
            "inverse": "t=b/(a+1) for z=a+b*u",
            "omitted_point": "z=-1",
            "omitted_point_absent_from_mu_q_for_odd_q": True,
            "group_law": "t+s over 1+d*t*s",
        },
        "five_product_form": {
            "elementary_symmetric_variables": "e0,...,e5 in t1,...,t5",
            "even_part": "A=e0+d*e2+d^2*e4",
            "odd_part": "B=e1+d*e3+d^2*e5",
            "finite_target_equation": "B-t_target*A=0",
            "degree": 5,
            "multilinear_in_source_parameters": True,
            "requires_discrete_logarithms": False,
        },
        "complete_homogeneous_endpoint_moment": {
            "identity": (
                "sum_(i1<=...<=i5)(z_i1*...*z_i5)^m "
                "=h_5(z_1^m,...,z_n^m)"
            ),
            "newton_recurrence": (
                "r*h_r=sum_(j=1..r) p_j*h_(r-j), "
                "p_j=sum_i z_i^(j*m)"
            ),
            "validity_condition": "characteristic p>5",
            "single_moment_uses_five_deck_power_sums": True,
        },
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_small_k_multiplicative_c5_cost_ledger.r121.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_c5_query_exponent_B": fraction_record(QUERY_CAP),
            "c_atom_exponent_B": fraction_record(C_ATOM_EXPONENT),
        },
        "identities": algebraic_identities(),
        "routes": [
            {
                "route_id": "complete_homogeneous_full_moment_recurrence",
                "one_endpoint_moment_from_deck_exponent_B": fraction_record(
                    C_ATOM_EXPONENT
                ),
                "exact_recurrence_order_random_deck_exponent_B": (
                    fraction_record(C5_EXPONENT)
                ),
                "represented_state_or_output_exponent_B": fraction_record(
                    C5_EXPONENT
                ),
                "naive_all_moments_work_exponent_B": fraction_record(
                    C5_EXPONENT + C_ATOM_EXPONENT
                ),
                "inside_setup_cap": False,
            },
            {
                "route_id": "full_endpoint_annihilator_membership",
                "annihilator_degree_exponent_B": fraction_record(
                    C5_EXPONENT
                ),
                "inside_setup_cap": False,
            },
            {
                "route_id": "cayley_torus_degree_five_grid_evaluation",
                "predicate_degree": 5,
                "predicate_description_exponent_B": fraction_record(
                    Fraction(0)
                ),
                "source_grid_exponent_B": fraction_record(C5_EXPONENT),
                "inside_setup_cap_when_materialized": False,
                "compact_predicate_is_not_compact_membership_index": True,
            },
            {
                "route_id": "c2_table_c3_table_split_query",
                "c3_setup_exponent_B": fraction_record(C3_EXPONENT),
                "c2_query_exponent_B": fraction_record(C2_EXPONENT),
                "inside_setup_cap": True,
                "inside_polylog_query_cap": False,
            },
            {
                "route_id": "dinur_golovnev_balanced_k6_index_delta_zero",
                "state_exponent_B": fraction_record(Fraction(33, 8)),
                "query_exponent_B": fraction_record(Fraction(0)),
                "inside_setup_cap": False,
                "integer_addition_or_xor_theorem_directly_transferred": False,
            },
            {
                "route_id": (
                    "dinur_golovnev_theorem4p1_subfunction_template"
                ),
                "candidate_domain_exponent_B": fraction_record(C5_EXPONENT),
                "ambient_mu_q_universe_exponent_B": fraction_record(
                    Fraction(5)
                ),
                "paper_requires_output_universe_soft_O_domain": True,
                "ambient_universe_meets_paper_requirement": False,
                "injective_image_recoding_grant": (
                    "Even if an admissible image-range recoding is granted, "
                    "Items 4-5 cover at most D*L image points per sampled "
                    "decomposition, so 5/6 expected coverage forces "
                    "D*L>=5M/6."
                ),
                "delta_zero_space_term": "soft-O(D*L^(3/2))",
                "delta_zero_space_lower_bound_under_grant": "Omega(M)",
                "granted_state_exponent_B": fraction_record(C5_EXPONENT),
                "inside_setup_cap": False,
                "general_data_structure_lower_bound_claimed": False,
            },
            {
                "route_id": "pairing_image_exponent_linearization",
                "would_convert_products_to_integer_sums": True,
                "requires_finite_field_discrete_logarithms": True,
                "allowed": False,
            },
            {
                "route_id": (
                    "target_specialized_nonlinear_torus_c5_"
                    "membership_source_circuit"
                ),
                "exact_circuit_constructed": False,
                "general_lower_bound_claimed": False,
                "status": "open",
            },
        ],
        "random_deck_support_boundary": {
            "inherited_whp_distinct_support_exponent_B": (
                fraction_record(C5_EXPONENT)
            ),
            "transferred_to_every_filtered_deck": False,
        },
        "semantic_dedup": {
            "nearby_lanes": [
                "R90 additive exponential-moment Hankel translation",
                "R91 integer unequal-list subfunction inversion",
                "R118 scalar-blind elliptic C5 indexing",
                "R120 independent-torsion pairing character",
            ],
            "r121_distinct_scope": (
                "multiplicative complete-homogeneous moments and a complete "
                "odd-order norm-one torus chart after the R120 pairing"
            ),
            "new_idea_id_claimed": False,
        },
        "candidate_work_credit": False,
    }


def source_replay(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_small_k_multiplicative_c5_source_replay.r121.v1"
        ),
        "control_count": controls["control_count"],
        "all_torus_laws_and_c5_forms_exact": controls[
            "all_torus_laws_and_c5_forms_exact"
        ],
        "all_compact_moment_identities_exact": controls[
            "all_compact_moment_identities_exact"
        ],
        "all_full_annihilator_membership_controls_exact": controls[
            "all_annihilator_membership_controls_exact"
        ],
        "all_c2_c3_sources_and_empty_queries_exact": controls[
            "all_c2_c3_sources_and_empty_queries_exact"
        ],
        "candidate_scalar_labels_consumed": False,
        "inside_cap_polylog_membership_circuit_constructed": False,
        "inside_cap_polylog_source_recovery_constructed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R120_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R120 nonclaim boundary drifted")
    controls = finite_controls()
    cost = cost_ledger()
    replay = source_replay(controls)
    obligations = {
        "fourteen_source_bindings_verified": len(source_hashes) == 14,
        "r120_small_k_multiplicative_interface_inherited": (
            inherited["admission"]["pairing_character_control_admitted"]
            and not inherited["admission"]["lane_admitted"]
        ),
        "all_eight_r82_torus_moment_controls_complete": (
            controls["control_count"] == 8
        ),
        "all_pairing_images_norm_one_and_cayley_charted": controls[
            "all_pairing_images_norm_one_and_charted"
        ],
        "all_torus_laws_and_degree_five_forms_exact": controls[
            "all_torus_laws_and_c5_forms_exact"
        ],
        "all_complete_homogeneous_moment_identities_exact": controls[
            "all_compact_moment_identities_exact"
        ],
        "all_bm_orders_equal_distinct_c5_support": controls[
            "all_bm_orders_equal_distinct_c5_support"
        ],
        "all_full_annihilator_membership_controls_exact": controls[
            "all_annihilator_membership_controls_exact"
        ],
        "all_c2_c3_sources_and_empty_queries_exact": controls[
            "all_c2_c3_sources_and_empty_queries_exact"
        ],
        "moment_torus_split_cost_routes_charged": (
            not cost["candidate_work_credit"]
        ),
        "dinur_golovnev_template_scope_and_universe_charged": (
            not {
                row["route_id"]: row
                for row in cost["routes"]
            }[
                "dinur_golovnev_theorem4p1_subfunction_template"
            ]["ambient_universe_meets_paper_requirement"]
        ),
        "semantic_dedup_complete": (
            not cost["semantic_dedup"]["new_idea_id_claimed"]
        ),
        "candidate_scalar_labels_not_consumed": (
            not controls["candidate_scalar_labels_consumed"]
        ),
        "inside_cap_polylog_membership_circuit_complete": False,
        "inside_cap_polylog_source_recovery_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one target-specialized nonlinear torus C5 "
        "membership/source circuit outside the full moment/Newton/Hankel, "
        "explicit C2|C3 split, integer-residue kSUM, and Theorem 4.1 "
        "subfunction grammars. It may inject the target into the exact "
        "degree-five form B-t*A before expansion, but must use at most "
        "B^(9/4+o(1)) state, polylogarithmic arbitrary-target work, no "
        "pairing-image discrete logarithms, exact empty certification, and "
        "five projective occurrence backpointers. Charge pairing, extension, "
        "false-positive, relation, rank, log, and identical-descent work, "
        "and keep pairing-unfriendly inputs as a separate failed branch."
    )
    frozen = {
        "schema": (
            "p1553.frozen_m6_small_k_multiplicative_c5_"
            "subfunction_index.r121.v1"
        ),
        "source_bindings": source_binding_records(),
        "required_interface": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_c5_query_exponent_B": fraction_record(QUERY_CAP),
            "exact_empty_rejection_required": True,
            "five_projective_occurrence_backpointers_required": True,
            "pairing_image_discrete_logarithms_allowed": False,
        },
        "exact_available_representation": {
            "norm_one_torus_parameter": "t=b/(a+1)",
            "target_predicate": "B(t1,...,t5)-t_target*A(t1,...,t5)=0",
            "predicate_total_degree": 5,
            "complete_moment": "h_5(z_1^m,...,z_n^m)",
        },
        "closed_scoped_grammars": [
            "full endpoint moment sequence and BM/Newton annihilator",
            "materialized degree-five torus source grid",
            "explicit C2 table with C3 table and C2 query scan",
            "direct balanced integer kSUM indexing theorem",
            "Dinur-Golovnev Theorem 4.1 at delta zero on an injective image",
            "product-to-integer-sum conversion by field discrete logarithm",
        ],
        "preserved_interface": (
            "target-specialized nonlinear norm-one-torus C5 "
            "membership/source circuit outside the closed grammars"
        ),
        "pairing_unfriendly_inputs_solved": False,
        "general_arithmetic_circuit_or_data_structure_lower_bound_claimed": (
            False
        ),
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r121.v1",
        "r120_character_pairing_audit_complete": True,
        "r121_torus_and_moment_audit_complete": True,
        "inside_cap_polylog_multiplicative_c5_source_index_complete": False,
        "pairing_unfriendly_generic_branch_complete": False,
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
            "EXACT_TORUS_MOMENT_AND_SPLIT_CONTROLS_WITH_SCOPED_"
            "REPRESENTATION_NEGATIVES_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "PAIRING_IMAGES_ON_K2_CONTROLS_HAVE_COMPLETE_NORM_ONE_CAYLEY_"
            "CHART__FIVE_PRODUCT_IS_EXACT_DEGREE5_SYMMETRIC_TORUS_FORM__"
            "ENDPOINT_MOMENTS_ARE_COMPLETE_HOMOGENEOUS_DECK_MOMENTS__"
            "BM_ORDER_EQUALS_INJECTIVE_C5_SUPPORT__FULL_MOMENT_AND_"
            "ANNIHILATOR_STATE_B15O4__C2C3_SPLIT_SETUP_B9O4_QUERY_B3O2__"
            "DINUR_GOLOVNEV_DIRECT_K6_AND_THEOREM4P1_POLYLOG_ROUTES_OVER_"
            "CAP_OR_INAPPLICABLE__TARGET_SPECIALIZED_NONLINEAR_TORUS_C5_"
            "CIRCUIT_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "source_bindings": source_binding_records(),
        "algebraic_identities": algebraic_identities(),
        "finite_evidence": {
            "control_count": controls["control_count"],
            "all_pairing_images_norm_one_and_charted": controls[
                "all_pairing_images_norm_one_and_charted"
            ],
            "all_torus_laws_and_c5_forms_exact": controls[
                "all_torus_laws_and_c5_forms_exact"
            ],
            "all_compact_moment_identities_exact": controls[
                "all_compact_moment_identities_exact"
            ],
            "all_bm_orders_equal_distinct_c5_support": controls[
                "all_bm_orders_equal_distinct_c5_support"
            ],
            "all_annihilator_membership_controls_exact": controls[
                "all_annihilator_membership_controls_exact"
            ],
            "all_c2_c3_sources_and_empty_queries_exact": controls[
                "all_c2_c3_sources_and_empty_queries_exact"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "torus_and_moment_controls_admitted": True,
            "scoped_representation_negatives_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_m6_small_k_multiplicative_c5_subfunction_index.json"
            ),
            "cost": "m6_small_k_multiplicative_c5_cost_ledger.json",
            "source_replay": (
                "m6_small_k_multiplicative_c5_source_replay.json"
            ),
            "exceptional_controls": (
                "m6_small_k_multiplicative_c5_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r121.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The exact degree-five torus predicate is not a membership index.",
            "Finite embedding-degree-two controls receive no generic credit.",
            "The moment-order result is scoped to the represented moment grammar.",
            "No general arithmetic-circuit or data-structure lower bound is claimed.",
            "No field discrete logarithm receives unit-cost oracle credit.",
            "Pairing-unfriendly generic inputs remain unsolved.",
            "No relation rank, factor logs, or identical descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_EXACT_NORM_ONE_TORUS_COMPLETE_HOMOGENEOUS_MOMENT_BM_"
            "AND_SPLIT_SOURCE_CONTROLS_ONLY__REJECT_FULL_MOMENT_ANNIHILATOR_"
            "TORUS_GRID_C2C3_QUERY_DIRECT_KSUM_THEOREM4P1_AND_FIELD_DLP_"
            "ROUTES_AT_FROZEN_CAPS__PRESERVE_TARGET_SPECIALIZED_NONLINEAR_"
            "TORUS_C5_CIRCUIT_AND_PAIRING_UNFRIENDLY_GAP__NO_LOCATOR__NO_"
            "RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_m6_small_k_multiplicative_c5_"
            "moment_torus_probe_report_r121.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_m6_small_k_multiplicative_c5_subfunction_index.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_small_k_multiplicative_c5_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_small_k_multiplicative_c5_source_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_small_k_multiplicative_c5_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r121.json"
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
        f"R121 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
