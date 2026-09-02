#!/usr/bin/env python3
"""Test whether a target-local first norm jet composes through addition."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
from collections import Counter
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.5a5c_jet_preserving_addition_pushforward.r87.v1"
SETUP_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4

R86_REPORT = pathlib.Path(
    "p1553_5a5c_sparse_multihomogeneous_moment_recurrence_"
    "probe_report_r86.json"
)
R86_REPORT_SHA256 = (
    "b93c1e581a7953cf1a9a86d3ff4220f5c2a92b2ea3e0ed7076c57d8896ee1173"
)
R86_GATE = pathlib.Path(
    "p1553_5a5c_sparse_multihomogeneous_moment_recurrence_probe_gate_r86.md"
)
R86_GATE_SHA256 = (
    "21d52b3d5f04fa408d0d9a4ef229f9169f05f4a31aa858e21ff252ef497f9b44"
)
P1536_AUDIT = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-133/p1536_frobenius_projector_norm_jet_audit.md"
)
P1536_AUDIT_SHA256 = (
    "81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393"
)
P1515_NAVIGATOR = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-098/compressed_navigator_gate_v1.md"
)
P1515_NAVIGATOR_SHA256 = (
    "dadcadf45bdea910f0a12e904bdfe32c4a517b0756ef08148de75fb39929e3e5"
)
P1515_TRICHOTOMY = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-098/recursive_s3_local_separator_trichotomy_v1.md"
)
P1515_TRICHOTOMY_SHA256 = (
    "dec667b097bcaefdf4c54091b2a9fa7757db5a65efe5b36e6ac15a6ff11a435a"
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R86_REPORT: R86_REPORT_SHA256,
        R86_GATE: R86_GATE_SHA256,
        P1536_AUDIT: P1536_AUDIT_SHA256,
        P1515_NAVIGATOR: P1515_NAVIGATOR_SHA256,
        P1515_TRICHOTOMY: P1515_TRICHOTOMY_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R87 source binding mismatch: {failures}")
    return actual


def trim(poly: Sequence[int], modulus: int) -> list[int]:
    result = [coefficient % modulus for coefficient in poly]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result or [0]


def poly_add(
    left: Sequence[int],
    right: Sequence[int],
    modulus: int,
) -> list[int]:
    size = max(len(left), len(right))
    result = [0] * size
    for index in range(size):
        result[index] = (
            (left[index] if index < len(left) else 0)
            + (right[index] if index < len(right) else 0)
        ) % modulus
    return trim(result, modulus)


def poly_mul(
    left: Sequence[int],
    right: Sequence[int],
    modulus: int,
) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] = (
                result[left_index + right_index]
                + left_value * right_value
            ) % modulus
    return trim(result, modulus)


def poly_from_roots(roots: Iterable[int], modulus: int) -> list[int]:
    result = [1]
    for root in roots:
        result = poly_mul(result, [(-root) % modulus, 1], modulus)
    return result


def poly_eval(poly: Sequence[int], value: int, modulus: int) -> int:
    result = 0
    for coefficient in reversed(poly):
        result = (result * value + coefficient) % modulus
    return result


def poly_derivative(poly: Sequence[int], modulus: int) -> list[int]:
    if len(poly) <= 1:
        return [0]
    return trim(
        [
            index * coefficient % modulus
            for index, coefficient in enumerate(poly)
            if index
        ],
        modulus,
    )


def poly_divmod(
    numerator: Sequence[int],
    denominator: Sequence[int],
    modulus: int,
) -> tuple[list[int], list[int]]:
    remainder = trim(numerator, modulus)
    divisor = trim(denominator, modulus)
    if divisor == [0]:
        raise ZeroDivisionError("polynomial division by zero")
    quotient = [0] * max(1, len(remainder) - len(divisor) + 1)
    inverse_lead = pow(divisor[-1], modulus - 2, modulus)
    while remainder != [0] and len(remainder) >= len(divisor):
        shift = len(remainder) - len(divisor)
        scale = remainder[-1] * inverse_lead % modulus
        quotient[shift] = scale
        for index, coefficient in enumerate(divisor):
            remainder[index + shift] = (
                remainder[index + shift] - scale * coefficient
            ) % modulus
        remainder = trim(remainder, modulus)
    return trim(quotient, modulus), remainder


def poly_mod(
    numerator: Sequence[int],
    denominator: Sequence[int],
    modulus: int,
) -> list[int]:
    return poly_divmod(numerator, denominator, modulus)[1]


def poly_monic(poly: Sequence[int], modulus: int) -> list[int]:
    value = trim(poly, modulus)
    if value == [0]:
        return value
    inverse = pow(value[-1], modulus - 2, modulus)
    return [coefficient * inverse % modulus for coefficient in value]


def poly_gcd(
    left: Sequence[int],
    right: Sequence[int],
    modulus: int,
) -> list[int]:
    a = trim(left, modulus)
    b = trim(right, modulus)
    while b != [0]:
        a, b = b, poly_mod(a, b, modulus)
    return poly_monic(a, modulus)


def translated_poly(
    poly: Sequence[int],
    target: int,
    modulus: int,
) -> list[int]:
    """Return P(target-X), with ascending coefficients in X."""
    result = [0]
    for degree, coefficient in enumerate(poly):
        term = [0] * (degree + 1)
        for x_degree in range(degree + 1):
            term[x_degree] = (
                coefficient
                * math.comb(degree, x_degree)
                * pow(target, degree - x_degree, modulus)
                * (-1 if x_degree % 2 else 1)
            ) % modulus
        result = poly_add(result, term, modulus)
    return trim(result, modulus)


def first_jet(
    poly: Sequence[int],
    target: int,
    modulus: int,
) -> tuple[int, int]:
    return (
        poly_eval(poly, target, modulus),
        poly_eval(poly_derivative(poly, modulus), target, modulus),
    )


def convolution_first_jet(
    right_poly: Sequence[int],
    left_support: Sequence[int],
    target: int,
    modulus: int,
) -> tuple[int, int]:
    values = [
        poly_eval(right_poly, target - shift, modulus)
        for shift in left_support
    ]
    derivatives = [
        poly_eval(
            poly_derivative(right_poly, modulus),
            target - shift,
            modulus,
        )
        for shift in left_support
    ]
    norm = math.prod(values) % modulus
    derivative = 0
    for index, derivative_value in enumerate(derivatives):
        derivative = (
            derivative
            + derivative_value
            * math.prod(values[:index] + values[index + 1 :])
        ) % modulus
    return norm, derivative


def marker_deformation_poly(
    roots: Sequence[int],
    weights: Sequence[int],
    modulus: int,
) -> list[int]:
    if len(roots) != len(weights):
        raise AssertionError("one marker weight is required per root")
    result = [0]
    for root_index, weight in enumerate(weights):
        quotient = poly_from_roots(
            roots[:root_index] + roots[root_index + 1 :],
            modulus,
        )
        result = poly_add(
            result,
            [(-weight * coefficient) % modulus for coefficient in quotient],
            modulus,
        )
    return result


def weights_for_marker_poly(
    roots: Sequence[int],
    marker_poly: Sequence[int],
    modulus: int,
) -> list[int]:
    root_poly = poly_from_roots(roots, modulus)
    derivative = poly_derivative(root_poly, modulus)
    weights = []
    for root in roots:
        denominator = poly_eval(derivative, root, modulus)
        if denominator == 0:
            raise AssertionError("marker interpolation needs squarefree roots")
        weights.append(
            -poly_eval(marker_poly, root, modulus)
            * pow(denominator, modulus - 2, modulus)
            % modulus
        )
    return weights


def local_jet_nonfunctor_witness() -> dict[str, Any]:
    modulus = 101
    target = 0
    left_support = [0, 1]
    roots_p = [1, 18, 24]
    roots_q = [2, 5, 23]
    poly_p = poly_from_roots(roots_p, modulus)
    poly_q = poly_from_roots(roots_q, modulus)
    jet_p = first_jet(poly_p, target, modulus)
    jet_q = first_jet(poly_q, target, modulus)
    translated_p = first_jet(poly_p, target - 1, modulus)
    translated_q = first_jet(poly_q, target - 1, modulus)
    convolved_p = convolution_first_jet(
        poly_p,
        left_support,
        target,
        modulus,
    )
    convolved_q = convolution_first_jet(
        poly_q,
        left_support,
        target,
        modulus,
    )
    marker_polys = [[value] for value in range(1, 6)]
    marker_weights_p = [
        weights_for_marker_poly(roots_p, marker_poly, modulus)
        for marker_poly in marker_polys
    ]
    marker_weights_q = [
        weights_for_marker_poly(roots_q, marker_poly, modulus)
        for marker_poly in marker_polys
    ]
    reconstructed_markers_p = [
        marker_deformation_poly(roots_p, weights, modulus)
        for weights in marker_weights_p
    ]
    reconstructed_markers_q = [
        marker_deformation_poly(roots_q, weights, modulus)
        for weights in marker_weights_q
    ]
    marker_state_p = [
        poly_eval(marker_poly, target, modulus)
        for marker_poly in reconstructed_markers_p
    ]
    marker_state_q = [
        poly_eval(marker_poly, target, modulus)
        for marker_poly in reconstructed_markers_q
    ]
    full_local_state_p = [*jet_p, *marker_state_p]
    full_local_state_q = [*jet_q, *marker_state_q]
    return {
        "field_prime": modulus,
        "target": target,
        "left_support": left_support,
        "right_roots_p": roots_p,
        "right_roots_q": roots_q,
        "right_poly_p": poly_p,
        "right_poly_q": poly_q,
        "target_first_jet_p": list(jet_p),
        "target_first_jet_q": list(jet_q),
        "target_first_jets_equal": jet_p == jet_q,
        "shifted_first_jet_p": list(translated_p),
        "shifted_first_jet_q": list(translated_q),
        "shifted_first_jets_differ": translated_p != translated_q,
        "convolution_first_jet_p": list(convolved_p),
        "convolution_first_jet_q": list(convolved_q),
        "convolution_first_jets_differ": convolved_p != convolved_q,
        "marker_deformation_polynomials": marker_polys,
        "marker_root_weights_p": marker_weights_p,
        "marker_root_weights_q": marker_weights_q,
        "marker_polynomials_reconstructed_p": (
            reconstructed_markers_p == marker_polys
        ),
        "marker_polynomials_reconstructed_q": (
            reconstructed_markers_q == marker_polys
        ),
        "full_local_first_jet_p": full_local_state_p,
        "full_local_first_jet_q": full_local_state_q,
        "full_local_first_jets_equal": (
            full_local_state_p == full_local_state_q
        ),
        "fixed_factor_index_marker_weights": False,
        "theorem_witness": (
            "Two valid five-marker first-order split norm deformations have "
            "the same full local jet at T, but their addition-pushforward "
            "first jets differ."
        ),
    }


def matrix_rank(rows: Sequence[Sequence[int]], modulus: int) -> int:
    matrix = [
        [value % modulus for value in row]
        for row in rows
    ]
    if not matrix:
        return 0
    row_count = len(matrix)
    column_count = len(matrix[0])
    rank = 0
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(rank, row_count)
                if matrix[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inverse = pow(matrix[rank][column], modulus - 2, modulus)
        matrix[rank] = [
            value * inverse % modulus for value in matrix[rank]
        ]
        for row in range(row_count):
            if row == rank or matrix[row][column] == 0:
                continue
            scale = matrix[row][column]
            matrix[row] = [
                (value - scale * pivot_value) % modulus
                for value, pivot_value in zip(
                    matrix[row],
                    matrix[rank],
                )
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


def translation_orbit_control(
    modulus_degree: int,
    translated_degree: int,
    prime: int = 1009,
) -> dict[str, Any]:
    if not (prime > translated_degree >= modulus_degree - 1):
        raise AssertionError("translation rank theorem preconditions failed")
    modulus_roots = list(range(1, modulus_degree + 1))
    translated_roots = list(
        range(100, 100 + translated_degree)
    )
    modulus_poly = poly_from_roots(modulus_roots, prime)
    translated = poly_from_roots(translated_roots, prime)
    rows = []
    for target in range(translated_degree + 1):
        remainder = poly_mod(
            translated_poly(translated, target, prime),
            modulus_poly,
            prime,
        )
        rows.append(
            remainder + [0] * (modulus_degree - len(remainder))
        )
    rank = matrix_rank(rows, prime)
    return {
        "field_prime": prime,
        "modulus_degree_m": modulus_degree,
        "translated_polynomial_degree_n": translated_degree,
        "characteristic_greater_than_n": prime > translated_degree,
        "translation_sample_count": len(rows),
        "translated_remainder_rank": rank,
        "full_quotient_rank": rank == modulus_degree,
        "theorem": (
            "For monic P of degree n in characteristic p>n, its "
            "translations span all polynomials of degree at most n. "
            "Reduction modulo monic A of degree m<=n+1 therefore spans "
            "the full m-dimensional quotient."
        ),
    }


def deterministic_decks(
    prefix: str,
    deck_count: int,
    deck_size: int,
    modulus: int,
) -> list[list[int]]:
    decks = []
    used: set[int] = set()
    for deck_index in range(deck_count):
        deck = []
        counter = 0
        while len(deck) < deck_size:
            digest = hashlib.sha256(
                f"P1553-R87|{prefix}|{deck_index}|{counter}".encode("ascii")
            ).digest()
            counter += 1
            value = int.from_bytes(digest, "big") % modulus
            if value in used:
                continue
            used.add(value)
            deck.append(value)
        decks.append(deck)
    return decks


def endpoint_table(
    decks: Sequence[Sequence[int]],
    modulus: int,
) -> tuple[list[int], Counter[int], dict[int, tuple[int, ...]]]:
    endpoints = []
    histogram: Counter[int] = Counter()
    first_source: dict[int, tuple[int, ...]] = {}
    for source in itertools.product(*(range(len(deck)) for deck in decks)):
        endpoint = sum(
            decks[index][source[index]]
            for index in range(len(decks))
        ) % modulus
        endpoints.append(endpoint)
        histogram[endpoint] += 1
        first_source.setdefault(endpoint, source)
    return endpoints, histogram, first_source


def unique_target_replay() -> dict[str, Any]:
    prime = 1_000_003
    a_decks = deterministic_decks("A", 5, 2, prime)
    c_decks = deterministic_decks("C", 5, 3, prime)
    a_endpoints, a_histogram, a_sources = endpoint_table(a_decks, prime)
    c_endpoints, c_histogram, c_sources = endpoint_table(c_decks, prime)
    target_histogram: Counter[int] = Counter()
    target_first: dict[int, tuple[int, int]] = {}
    for a_endpoint, a_multiplicity in a_histogram.items():
        for c_endpoint, c_multiplicity in c_histogram.items():
            target = (a_endpoint + c_endpoint) % prime
            target_histogram[target] += a_multiplicity * c_multiplicity
            target_first.setdefault(target, (a_endpoint, c_endpoint))
    target = next(
        value
        for value in sorted(target_histogram)
        if target_histogram[value] == 1
    )
    expected_a, expected_c = target_first[target]
    poly_a = poly_from_roots(a_endpoints, prime)
    poly_c = poly_from_roots(c_endpoints, prime)
    shifted_c = translated_poly(poly_c, target, prime)
    remainder = poly_mod(shifted_c, poly_a, prime)
    common = poly_gcd(poly_a, remainder, prime)
    recovered_a_roots = [
        endpoint
        for endpoint in a_histogram
        if poly_eval(common, endpoint, prime) == 0
    ]
    if recovered_a_roots != [expected_a]:
        raise AssertionError("translated gcd failed unique A source")
    recovered_a = recovered_a_roots[0]
    recovered_c = (target - recovered_a) % prime
    a_source = a_sources[recovered_a]
    c_source = c_sources[recovered_c]
    replay = (
        sum(a_decks[index][choice] for index, choice in enumerate(a_source))
        + sum(c_decks[index][choice] for index, choice in enumerate(c_source))
    ) % prime
    final_norm_jet = convolution_first_jet(
        poly_c,
        a_endpoints,
        target,
        prime,
    )
    return {
        "field_prime": prime,
        "a_deck_sizes": [len(deck) for deck in a_decks],
        "c_deck_sizes": [len(deck) for deck in c_decks],
        "a_occurrence_count": len(a_endpoints),
        "c_occurrence_count": len(c_endpoints),
        "full_occurrence_count": len(a_endpoints) * len(c_endpoints),
        "target": target,
        "target_multiplicity": target_histogram[target],
        "a_characteristic_degree": len(poly_a) - 1,
        "c_characteristic_degree": len(poly_c) - 1,
        "translated_remainder_degree": len(remainder) - 1,
        "translated_gcd_degree": len(common) - 1,
        "expected_a_endpoint": expected_a,
        "expected_c_endpoint": expected_c,
        "recovered_a_endpoint": recovered_a,
        "recovered_c_endpoint": recovered_c,
        "a_source": list(a_source),
        "c_source": list(c_source),
        "joint_source_replay": replay == target,
        "final_norm_jet": list(final_norm_jet),
        "simple_final_norm_branch": (
            final_norm_jet[0] == 0 and final_norm_jet[1] != 0
        ),
        "source_dictionaries_enumerated": True,
        "candidate_credit": False,
    }


def exceptional_controls() -> dict[str, Any]:
    prime = 101
    empty_a = poly_from_roots([1, 2], prime)
    empty_c = poly_from_roots([4, 5], prime)
    empty_target = 20
    empty_gcd = poly_gcd(
        empty_a,
        translated_poly(empty_c, empty_target, prime),
        prime,
    )
    multiple_target = 6
    multiple_gcd = poly_gcd(
        empty_a,
        translated_poly(empty_c, multiple_target, prime),
        prime,
    )
    nonreduced_c = poly_from_roots([5, 5], prime)
    nonreduced_jet = convolution_first_jet(
        nonreduced_c,
        [1],
        6,
        prime,
    )
    nonreduced_gcd = poly_gcd(
        poly_from_roots([1], prime),
        translated_poly(nonreduced_c, 6, prime),
        prime,
    )
    return {
        "empty": {
            "target": empty_target,
            "gcd_degree": len(empty_gcd) - 1,
            "detected": len(empty_gcd) == 1,
        },
        "multiple_distinct": {
            "target": multiple_target,
            "gcd_degree": len(multiple_gcd) - 1,
            "rejected_by_unique_source_gate": len(multiple_gcd) - 1 == 2,
        },
        "nonreduced": {
            "target": 6,
            "gcd_degree": len(nonreduced_gcd) - 1,
            "norm_first_jet": list(nonreduced_jet),
            "gcd_alone_misses_multiplicity": len(nonreduced_gcd) - 1 == 1,
            "first_jet_rejects": nonreduced_jet == (0, 0),
        },
        "actual_projective_semaev_charts_supplied": False,
        "signed_and_infinity_replay_complete": False,
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "five_a_endpoint_degree_exponent_B": 2.0,
        "five_c_endpoint_degree_exponent_B": 3.0,
        "a_characteristic_polynomial_words_exponent_B": 2.0,
        "c_characteristic_polynomial_words_exponent_B": 3.0,
        "translated_c_mod_a_remainder_words_exponent_B": 2.0,
        "target_local_first_jet_words_exponent_B": 0.0,
        "a_polynomial_inside_setup_cap": 2.0 <= SETUP_CAP_EXPONENT,
        "c_polynomial_inside_setup_cap": 3.0 <= SETUP_CAP_EXPONENT,
        "explicit_remainder_inside_fresh_cap": 2.0 <= ONLINE_CAP_EXPONENT,
        "target_local_first_jet_is_exact_intertwiner": False,
        "standard_explicit_translated_remainder_route_inside_caps": False,
        "scope_exception": (
            "a black-box scalar resultant/gcd and exact source localizer "
            "that consumes compact atom divisors without emitting P_C, the "
            "B^2 translated remainder, or a translated evaluation vector"
        ),
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    witness = local_jet_nonfunctor_witness()
    rank_controls = [
        translation_orbit_control(m, n)
        for m, n in ((3, 5), (5, 8), (7, 11), (9, 14))
    ]
    replay = unique_target_replay()
    exceptional = exceptional_controls()
    costs = cost_ledger()
    all_ranks_full = all(row["full_quotient_rank"] for row in rank_controls)
    all_exceptional = (
        exceptional["empty"]["detected"]
        and exceptional["multiple_distinct"][
            "rejected_by_unique_source_gate"
        ]
        and exceptional["nonreduced"]["first_jet_rejects"]
    )
    frozen = {
        "schema": "p1553.frozen_5a5c_jet_pushforward_intertwiner.r87.v1",
        "operation": "Pontryagin addition of effective endpoint divisors",
        "right_norm": "P_C(T)=product_c(T-c)",
        "exact_composition": (
            "P_(A*C)(T)=product_a P_C(T-a)"
        ),
        "first_derivative": (
            "sum_a P_C'(T-a) product_(b!=a) P_C(T-b)"
        ),
        "proposed_local_state": "j^1_T(P_C)=(P_C(T),P_C'(T))",
        "local_state_is_congruence_for_addition": False,
        "standard_exact_state": "P_C(T-X) mod P_A(X)",
        "candidate_dlp_labels_forbidden": True,
        "caps": {
            "setup_state_exponent_B": SETUP_CAP_EXPONENT,
            "fresh_work_exponent_B": ONLINE_CAP_EXPONENT,
        },
    }
    composition = {
        "schema": (
            "p1553.slotwise_jet_composition_truncation_receipts.r87.v1"
        ),
        "target_local_nonfunctor_witness": witness,
        "translation_orbit_controls": rank_controls,
        "all_translation_orbits_full_quotient_rank": all_ranks_full,
        "theorem_scope": (
            "No operator receiving only a marked right norm's full "
            "first-order jet at the final target can exactly compose every "
            "split marked norm with every nontrivial left divisor. The "
            "witness allows arbitrary public marker weights; the narrower "
            "fixed factor-index marker family and richer implicit state are "
            "not universal lower bounds."
        ),
    }
    source_replay = {
        "schema": "p1553.public_input_target_jet_source_replay.r87.v1",
        "synthetic_scalar_control": replay,
        "translated_resultant_identity_exact": (
            replay["translated_gcd_degree"] == 1
            and replay["joint_source_replay"]
        ),
        "source_dictionaries_enumerated": True,
        "public_compact_divisor_constructor_inside_caps": False,
        "candidate_credit": False,
    }
    exceptional_receipt = {
        "schema": (
            "p1553.exceptional_chart_false_positive_controls.r87.v1"
        ),
        "controls": exceptional,
        "all_scalar_branches_exact": all_exceptional,
        "actual_elliptic_source_biconditional_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r87.v1",
        "target_local_first_jet_intertwiner_found": False,
        "explicit_translated_remainder_inside_caps": False,
        "black_box_scalar_resultant_source_localizer_supplied": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "five_source_bindings_verified": len(bindings) == 5,
        "exact_addition_norm_identity": True,
        "target_local_first_jet_nonfunctor_witness": (
            witness["full_local_first_jets_equal"]
            and witness["marker_polynomials_reconstructed_p"]
            and witness["marker_polynomials_reconstructed_q"]
            and witness["convolution_first_jets_differ"]
        ),
        "four_full_translation_rank_controls": (
            len(rank_controls) == 4 and all_ranks_full
        ),
        "synthetic_unique_source_resultant_replay": (
            replay["translated_gcd_degree"] == 1
            and replay["joint_source_replay"]
        ),
        "empty_multiple_nonreduced_controls": all_exceptional,
        "public_compact_c5_norm_inside_setup_cap": False,
        "fresh_translated_remainder_inside_online_cap": False,
        "black_box_scalar_resultant_source_localizer": False,
        "actual_semaev_projective_exceptional_charts": False,
        "signed_source_biconditional_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "TARGET_LOCAL_FIRST_NORM_JET_NONFUNCTORIAL__"
            "EXPLICIT_TRANSLATED_REMAINDER_FULL_B2"
        ),
        "source_bindings": {
            "r86_report": {
                "path": str(R86_REPORT),
                "sha256": R86_REPORT_SHA256,
            },
            "r86_gate": {
                "path": str(R86_GATE),
                "sha256": R86_GATE_SHA256,
            },
            "p1536_norm_jet_audit": {
                "path": str(P1536_AUDIT),
                "sha256": P1536_AUDIT_SHA256,
            },
            "p1515_compressed_navigator": {
                "path": str(P1515_NAVIGATOR),
                "sha256": P1515_NAVIGATOR_SHA256,
            },
            "p1515_local_separator_trichotomy": {
                "path": str(P1515_TRICHOTOMY),
                "sha256": P1515_TRICHOTOMY_SHA256,
            },
        },
        "novelty_scope": (
            "R87 narrows P1536's open jet intertwiner and P1515's open "
            "field-router class. It proves that a target-local first jet is "
            "not an addition-pushforward state and identifies the standard "
            "exact replacement as a full translated remainder orbit."
        ),
        "target_local_nonfunctor_witness": witness,
        "translation_orbit_controls": rank_controls,
        "synthetic_source_replay": replay,
        "exceptional_controls": exceptional,
        "cost_ledger": costs,
        "side_artifacts": {
            "frozen": "frozen_5a5c_jet_pushforward_intertwiner.json",
            "composition": (
                "slotwise_jet_composition_and_truncation_receipts.json"
            ),
            "source_replay": "public_input_target_jet_and_source_replay.json",
            "exceptional": (
                "exceptional_chart_and_false_positive_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r87.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": (
            "This closes target-local first-jet propagation and explicit "
            "translated-remainder/value-vector realizations only. It is not "
            "a lower bound for the fixed factor-index marker subfamily, "
            "black-box scalar resultants, implicit half-gcd/source "
            "localization, or unrestricted arithmetic circuits over the "
            "compact elliptic divisors."
        ),
        "next_action": (
            "Construct or refute one black-box translated resultant/gcd "
            "source localizer for P_A(X) and implicit P_C(T-X). It must "
            "consume compact five-slot D_A,D_C, avoid materializing the "
            "B^3 C polynomial and B^2 translated remainder/evaluation "
            "orbit, specialize one fresh target inside B^(5/4), return the "
            "unique jointly coupled source, and reject multiple, "
            "nonreduced, signed, infinity, and exceptional fibers."
        ),
        "disposition": (
            "REJECT_TARGET_LOCAL_FIRST_JET_AND_EXPLICIT_TRANSLATED_REMAINDER_"
            "ONLY__EXACT_PONTRYAGIN_NORM_REQUIRES_SHIFTED_JETS__SPLIT_"
            "POLYNOMIAL_COUNTEREXAMPLE__TRANSLATION_REMAINDER_ORBIT_FULL_"
            "B2__C5_CHARACTERISTIC_B3__SYNTHETIC_TRANSLATED_GCD_SOURCE_"
            "EXACT_BUT_ENUMERATED__BLACK_BOX_SCALAR_RESULTANT_GCD_OPEN__NO_"
            "RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_"
            "BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "composition": composition,
        "source_replay": source_replay,
        "exceptional": exceptional_receipt,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_jet_preserving_addition_pushforward_"
            "probe_report_r87.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_jet_pushforward_intertwiner.json"
        ),
    )
    parser.add_argument(
        "--composition-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "slotwise_jet_composition_and_truncation_receipts.json"
        ),
    )
    parser.add_argument(
        "--source-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "public_input_target_jet_and_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "exceptional_chart_and_false_positive_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r87.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.composition_output, bundle["composition"])
    write_json(args.source_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
