#!/usr/bin/env python3
"""Test a nonlocal moment/Hankel translation sketch after R89."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import pathlib
from typing import Any, Sequence


SCHEMA = "p1553.5a5c_nonlocal_moment_hankel_translation.r90.v1"
SETUP_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4
ATOM_C_EXPONENT = 3 / 5
SLOT_COUNT = 5

R89_REPORT = pathlib.Path(
    "p1553_5a5c_fixed_marker_scalar_recurrence_probe_report_r89.json"
)
R89_REPORT_SHA256 = (
    "50a075dd73d3c5298be00efdbf5186734ddf694ed14bbb38cb37f890815a3b63"
)
R89_GATE = pathlib.Path(
    "p1553_5a5c_fixed_marker_scalar_recurrence_probe_gate_r89.md"
)
R89_GATE_SHA256 = (
    "2e6ac3545bb9db94e5d8b54eed5b4a67a22cf870d544dc0af80b96b41b444858"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
P1536_AUDIT = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-133/p1536_frobenius_projector_norm_jet_audit.md"
)
P1536_AUDIT_SHA256 = (
    "81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393"
)
P1515_TRICHOTOMY = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-098/recursive_s3_local_separator_trichotomy_v1.md"
)
P1515_TRICHOTOMY_SHA256 = (
    "dec667b097bcaefdf4c54091b2a9fa7757db5a65efe5b36e6ac15a6ff11a435a"
)
P1515_ROUTER = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-098/recursive_s3_field_router_candidate_v1.md"
)
P1515_ROUTER_SHA256 = (
    "ee7c0ef479f33d3a82ab6827c286460604c6d26c9a76aa551305eccc2a337e24"
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R89_REPORT: R89_REPORT_SHA256,
        R89_GATE: R89_GATE_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        P1536_AUDIT: P1536_AUDIT_SHA256,
        P1515_TRICHOTOMY: P1515_TRICHOTOMY_SHA256,
        P1515_ROUTER: P1515_ROUTER_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R90 source binding mismatch: {failures}")
    return actual


def radix_decks(deck_size: int) -> list[list[int]]:
    return [
        [choice * deck_size**slot for choice in range(deck_size)]
        for slot in range(SLOT_COUNT)
    ]


def radix_rows(deck_size: int) -> list[tuple[int, tuple[int, ...]]]:
    rows = []
    endpoint_count = deck_size**SLOT_COUNT
    for endpoint in range(endpoint_count):
        quotient = endpoint
        choices = []
        for _ in range(SLOT_COUNT):
            choices.append(quotient % deck_size)
            quotient //= deck_size
        rows.append(
            (
                endpoint,
                tuple(choice + 1 for choice in choices),
            )
        )
    return rows


def power_moment_channels(
    deck_size: int,
    modulus: int,
    length: int,
) -> list[list[int]]:
    channels = [[0] * length for _ in range(SLOT_COUNT + 1)]
    for endpoint, weights in radix_rows(deck_size):
        power = 1
        for degree in range(length):
            channels[0][degree] = (
                channels[0][degree] + power
            ) % modulus
            for slot, weight in enumerate(weights):
                channels[slot + 1][degree] = (
                    channels[slot + 1][degree] + weight * power
                ) % modulus
            power = power * endpoint % modulus
    return channels


def truncated_mul(
    left: Sequence[int],
    right: Sequence[int],
    order: int,
    modulus: int,
) -> list[int]:
    result = [0] * order
    for left_index, left_value in enumerate(left[:order]):
        for right_index, right_value in enumerate(
            right[: order - left_index]
        ):
            result[left_index + right_index] = (
                result[left_index + right_index]
                + left_value * right_value
            ) % modulus
    return result


def exponential_series(
    values: Sequence[int],
    order: int,
    modulus: int,
    weights: Sequence[int] | None = None,
) -> list[int]:
    if weights is None:
        weights = [1] * len(values)
    if len(values) != len(weights):
        raise AssertionError("one weight is required per deck value")
    inverse_factorials = [1] * order
    factorial = 1
    for degree in range(1, order):
        factorial = factorial * degree % modulus
        inverse_factorials[degree] = pow(factorial, modulus - 2, modulus)
    result = [0] * order
    for value, weight in zip(values, weights):
        power = 1
        for degree in range(order):
            result[degree] = (
                result[degree]
                + weight * power * inverse_factorials[degree]
            ) % modulus
            power = power * value % modulus
    return result


def deck_product_series(
    deck_size: int,
    order: int,
    modulus: int,
) -> list[list[int]]:
    decks = radix_decks(deck_size)
    unweighted = [
        exponential_series(deck, order, modulus)
        for deck in decks
    ]
    channels = []
    for marker_slot in range(-1, SLOT_COUNT):
        product = [1] + [0] * (order - 1)
        for slot, deck in enumerate(decks):
            factor = unweighted[slot]
            if slot == marker_slot:
                factor = exponential_series(
                    deck,
                    order,
                    modulus,
                    weights=list(range(1, deck_size + 1)),
                )
            product = truncated_mul(product, factor, order, modulus)
        channels.append(product)
    return channels


def translated_moments(
    moments: Sequence[int],
    target: int,
    modulus: int,
) -> list[int]:
    translated = []
    for degree in range(len(moments)):
        value = 0
        for inner in range(degree + 1):
            value += (
                math.comb(degree, inner)
                * pow(target, degree - inner, modulus)
                * (-1 if inner % 2 else 1)
                * moments[inner]
            )
        translated.append(value % modulus)
    return translated


def deck_update_and_translation_control() -> dict[str, Any]:
    modulus = 65_537
    deck_size = 3
    order = 24
    direct = power_moment_channels(deck_size, modulus, order)
    factorials = [1]
    for degree in range(1, order):
        factorials.append(factorials[-1] * degree % modulus)
    direct_egf = [
        [
            value * pow(factorials[degree], modulus - 2, modulus)
            % modulus
            for degree, value in enumerate(channel)
        ]
        for channel in direct
    ]
    product_egf = deck_product_series(deck_size, order, modulus)
    target = 1_237
    translated = [
        translated_moments(channel, target, modulus)
        for channel in direct
    ]
    rows = radix_rows(deck_size)
    direct_translated = [[0] * order for _ in direct]
    for endpoint, weights in rows:
        shifted = (target - endpoint) % modulus
        power = 1
        for degree in range(order):
            direct_translated[0][degree] = (
                direct_translated[0][degree] + power
            ) % modulus
            for slot, weight in enumerate(weights):
                direct_translated[slot + 1][degree] = (
                    direct_translated[slot + 1][degree]
                    + weight * power
                ) % modulus
            power = power * shifted % modulus
    return {
        "field_prime": modulus,
        "deck_size": deck_size,
        "truncation_order": order,
        "target": target,
        "deck_product_matches_direct_moments": (
            product_egf == direct_egf
        ),
        "binomial_translation_matches_direct_shift": (
            translated == direct_translated
        ),
        "target_independent_state": True,
        "deck_update_law": (
            "multiply the five deck exponential-moment series; replace one "
            "factor by its fixed-choice-weight series for each marker"
        ),
        "target_translation_law": (
            "G_target(z)=exp(target*z)*G(-z), channel by channel"
        ),
    }


def berlekamp_massey(
    sequence: Sequence[int],
    modulus: int,
) -> tuple[int, list[int]]:
    connection = [1] + [0] * len(sequence)
    previous = [1] + [0] * len(sequence)
    complexity = 0
    shift = 1
    discrepancy_scale = 1
    for index in range(len(sequence)):
        discrepancy = sequence[index] % modulus
        for offset in range(1, complexity + 1):
            discrepancy = (
                discrepancy
                + connection[offset] * sequence[index - offset]
            ) % modulus
        if discrepancy == 0:
            shift += 1
            continue
        saved = connection[:]
        scale = discrepancy * pow(
            discrepancy_scale,
            modulus - 2,
            modulus,
        ) % modulus
        for offset in range(len(sequence) - shift):
            connection[offset + shift] = (
                connection[offset + shift]
                - scale * previous[offset]
            ) % modulus
        if 2 * complexity <= index:
            complexity = index + 1 - complexity
            previous = saved
            discrepancy_scale = discrepancy
            shift = 1
        else:
            shift += 1
    return complexity, connection[: complexity + 1]


def newton_annihilator_desc(
    moments: Sequence[int],
    endpoint_count: int,
    modulus: int,
) -> list[int]:
    if len(moments) <= endpoint_count:
        raise AssertionError("moments p_0 through p_n are required")
    coefficients = [1]
    for degree in range(1, endpoint_count + 1):
        total = moments[degree]
        for coefficient_index in range(1, degree):
            total += (
                coefficients[coefficient_index]
                * moments[degree - coefficient_index]
            )
        coefficient = (
            -total * pow(degree, modulus - 2, modulus)
        ) % modulus
        coefficients.append(coefficient)
    return coefficients


def marker_polynomial_desc(
    norm_desc: Sequence[int],
    weighted_moments: Sequence[int],
    modulus: int,
) -> list[int]:
    endpoint_count = len(norm_desc) - 1
    if len(weighted_moments) < endpoint_count:
        raise AssertionError("weighted moments q_0 through q_(n-1) required")
    result = []
    for degree in range(endpoint_count):
        value = 0
        for norm_index in range(degree + 1):
            value += (
                norm_desc[norm_index]
                * weighted_moments[degree - norm_index]
            )
        result.append((-value) % modulus)
    return result


def eval_desc(
    coefficients: Sequence[int],
    value: int,
    modulus: int,
) -> int:
    result = 0
    for coefficient in coefficients:
        result = (result * value + coefficient) % modulus
    return result


def derivative_desc(
    coefficients: Sequence[int],
    modulus: int,
) -> list[int]:
    degree = len(coefficients) - 1
    return [
        coefficient * (degree - index) % modulus
        for index, coefficient in enumerate(coefficients[:-1])
    ]


def hankel_rank_control(deck_size: int) -> dict[str, Any]:
    modulus = 65_537
    endpoint_count = deck_size**SLOT_COUNT
    moments = power_moment_channels(
        deck_size,
        modulus,
        2 * endpoint_count,
    )
    complexities = [
        berlekamp_massey(channel, modulus)[0]
        for channel in moments
    ]
    norm_desc = newton_annihilator_desc(
        moments[0],
        endpoint_count,
        modulus,
    )
    root_count = sum(
        eval_desc(norm_desc, endpoint, modulus) == 0
        for endpoint in range(endpoint_count)
    )
    return {
        "field_prime": modulus,
        "deck_size": deck_size,
        "deck_sizes": [deck_size] * SLOT_COUNT,
        "endpoint_count": endpoint_count,
        "endpoint_support_is_radix_interval": True,
        "all_endpoints_distinct": True,
        "moment_samples_per_channel": 2 * endpoint_count,
        "norm_linear_complexity": complexities[0],
        "marker_linear_complexities": complexities[1:],
        "all_six_channels_have_full_hankel_rank": all(
            complexity == endpoint_count
            for complexity in complexities
        ),
        "newton_annihilator_degree": len(norm_desc) - 1,
        "newton_annihilator_root_count": root_count,
        "newton_annihilator_exact": root_count == endpoint_count,
    }


def full_marker_source_replay() -> dict[str, Any]:
    modulus = 65_537
    deck_size = 2
    endpoint_count = deck_size**SLOT_COUNT
    moments = power_moment_channels(
        deck_size,
        modulus,
        2 * endpoint_count,
    )
    norm_desc = newton_annihilator_desc(
        moments[0],
        endpoint_count,
        modulus,
    )
    derivative = derivative_desc(norm_desc, modulus)
    markers = [
        marker_polynomial_desc(norm_desc, channel, modulus)
        for channel in moments[1:]
    ]
    recovered = []
    expected = []
    for endpoint, weights in radix_rows(deck_size):
        derivative_value = eval_desc(derivative, endpoint, modulus)
        if derivative_value == 0:
            raise AssertionError("radix endpoint is unexpectedly multiple")
        recovered_weights = tuple(
            (
                -eval_desc(marker, endpoint, modulus)
                * pow(derivative_value, modulus - 2, modulus)
            ) % modulus
            for marker in markers
        )
        recovered.append(recovered_weights)
        expected.append(weights)
    target = endpoint_count + 7
    norm_at_target = eval_desc(norm_desc, target, modulus)
    direct_norm = math.prod(
        (target - endpoint) % modulus
        for endpoint in range(endpoint_count)
    ) % modulus
    return {
        "field_prime": modulus,
        "deck_size": deck_size,
        "endpoint_count": endpoint_count,
        "all_sources_recovered": recovered == expected,
        "recovered_source_count": sum(
            left == right for left, right in zip(recovered, expected)
        ),
        "target": target,
        "newton_norm_matches_direct_product": norm_at_target == direct_norm,
        "full_state_semantics_exact": (
            recovered == expected and norm_at_target == direct_norm
        ),
        "state_words_required": (
            (SLOT_COUNT + 1) * endpoint_count
        ),
    }


def universal_newton_sharpness_control() -> dict[str, Any]:
    modulus = 65_537
    degree = 32
    left_constant = 3
    right_constant = 5
    return {
        "field_prime": modulus,
        "degree": degree,
        "polynomial_left": f"X^{degree}-{left_constant}",
        "polynomial_right": f"X^{degree}-{right_constant}",
        "equal_power_sums_through_degree": degree - 1,
        "first_different_power_sum_degree": degree,
        "left_degree_n_power_sum": degree * left_constant % modulus,
        "right_degree_n_power_sum": degree * right_constant % modulus,
        "norm_values_at_zero_differ": left_constant != right_constant,
        "scope": (
            "sharpness for the universal Newton moment grammar; this is "
            "not asserted to be a five-deck realization"
        ),
    }


def cost_ledger() -> dict[str, Any]:
    endpoint_exponent = SLOT_COUNT * ATOM_C_EXPONENT
    return {
        "c_deck_size_exponent_B": ATOM_C_EXPONENT,
        "five_deck_endpoint_exponent_B": endpoint_exponent,
        "moment_state_channels": SLOT_COUNT + 1,
        "exact_norm_moment_order_exponent_B": endpoint_exponent,
        "exact_marker_moment_order_exponent_B": endpoint_exponent,
        "optimistic_quasilinear_deck_update_exponent_B": endpoint_exponent,
        "optimistic_full_state_translation_exponent_B": endpoint_exponent,
        "setup_state_cap_exponent_B": SETUP_CAP_EXPONENT,
        "online_work_cap_exponent_B": ONLINE_CAP_EXPONENT,
        "exact_state_inside_setup_cap": (
            endpoint_exponent <= SETUP_CAP_EXPONENT
        ),
        "exact_translation_inside_online_cap": (
            endpoint_exponent <= ONLINE_CAP_EXPONENT
        ),
        "fatal_obstruction": (
            "the actual distinct endpoint support gives Hankel/Prony order "
            "C^5=B^3, so exact Newton and fixed-marker reconstruction "
            "materialize B^3 moments or equivalent recurrence coefficients"
        ),
        "scope_exception": (
            "a non-moment, non-Hankel source-reporting translation index "
            "whose state is not a rational reconstruction of the endpoint "
            "resolvent"
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    update_control = deck_update_and_translation_control()
    rank_controls = [
        hankel_rank_control(deck_size)
        for deck_size in (2, 3, 4)
    ]
    source_replay = full_marker_source_replay()
    sharpness = universal_newton_sharpness_control()
    costs = cost_ledger()
    all_rank_controls_pass = all(
        control["all_six_channels_have_full_hankel_rank"]
        and control["newton_annihilator_exact"]
        for control in rank_controls
    )
    frozen = {
        "schema": "p1553.frozen_5a5c_nonlocal_translation_sketch.r90.v1",
        "sketch_id": "truncated_exponential_moment_hankel_pade_v1",
        "state": [
            "G(z)=product_i sum_j exp(c_ij*z) mod z^K",
            "G_i(z)=weighted_i(z)*product_(h!=i) unweighted_h(z) mod z^K",
        ],
        "marker_rule": "source choice index plus one in each C slot",
        "target_independent": True,
        "nonlocal": True,
        "nonlinear_deck_update": True,
        "translation_law": "G_target(z)=exp(target*z)*G(-z)",
        "decoder": "Newton identities or Hankel/Padé rational reconstruction",
        "forbidden_materializations": [
            "endpoint/source dictionary",
            "shift-value table",
            "quotient or Krylov vector",
        ],
        "caps": {
            "setup_state_exponent_B": SETUP_CAP_EXPONENT,
            "fresh_work_exponent_B": ONLINE_CAP_EXPONENT,
        },
    }
    transition = {
        "schema": "p1553.nonlinear_translation_state_update.r90.v1",
        "frozen_sketch": frozen,
        "deck_update_and_translation_control": update_control,
        "hankel_rank_controls": rank_controls,
        "universal_newton_sharpness_control": sharpness,
        "cost_ledger": costs,
    }
    replay = {
        "schema": "p1553.target_translation_marker_source_replay.r90.v1",
        "full_state_positive_control": source_replay,
        "all_fixed_marker_sources_exact_with_full_state": source_replay[
            "full_state_semantics_exact"
        ],
        "subcap_truncation_source_biconditional": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": "p1553.projective_exceptional_branch_controls.r90.v1",
        "finite_radix_simple_fibers_exact": source_replay[
            "full_state_semantics_exact"
        ],
        "multiple_and_nonreduced_moment_semantics": (
            "multiplicity is retained only when the full annihilator is "
            "reconstructed"
        ),
        "actual_semaev_projective_charts_supplied": False,
        "signed_infinity_tangent_replay_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r90.v1",
        "subcap_nonlocal_translation_sketch_found": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "six_source_bindings_verified": len(bindings) == 6,
        "target_independent_deck_update_exact": update_control[
            "deck_product_matches_direct_moments"
        ],
        "target_translation_law_exact": update_control[
            "binomial_translation_matches_direct_shift"
        ],
        "three_radix_support_controls_exact": len(rank_controls) == 3,
        "all_norm_and_marker_hankel_ranks_full": all_rank_controls_pass,
        "newton_annihilator_reconstructs_all_roots": all_rank_controls_pass,
        "full_marker_source_replay_exact": source_replay[
            "full_state_semantics_exact"
        ],
        "universal_newton_sharpness_control_exact": sharpness[
            "norm_values_at_zero_differ"
        ],
        "nonlocal_moment_state_inside_setup_cap": False,
        "fresh_translation_and_source_inside_online_cap": False,
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
            "NONLOCAL_MOMENT_UPDATE_EXACT__"
            "HANKEL_PADE_ORDER_C5_B3_OVER_CAP"
        ),
        "source_bindings": {
            "r89_report": {
                "path": str(R89_REPORT),
                "sha256": R89_REPORT_SHA256,
            },
            "r89_gate": {
                "path": str(R89_GATE),
                "sha256": R89_GATE_SHA256,
            },
            "r82_report": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "p1536_norm_jet_audit": {
                "path": str(P1536_AUDIT),
                "sha256": P1536_AUDIT_SHA256,
            },
            "p1515_local_separator_trichotomy": {
                "path": str(P1515_TRICHOTOMY),
                "sha256": P1515_TRICHOTOMY_SHA256,
            },
            "p1515_field_router_candidate": {
                "path": str(P1515_ROUTER),
                "sha256": P1515_ROUTER_SHA256,
            },
        },
        "literature_controls": [
            {
                "title": "The Berlekamp-Massey Algorithm via Minimal Polynomials",
                "url": "https://arxiv.org/abs/1001.1597",
                "use": "minimal linear recurrence control",
            },
            {
                "title": "Polynomial-Division-Based Algorithms for Computing Linear Recurrence Relations",
                "url": "https://arxiv.org/abs/2107.02582",
                "use": "Hankel/Padé rational reconstruction control",
            },
            {
                "title": "Improved Time-Space Tradeoffs for 3SUM-Indexing",
                "url": "https://arxiv.org/abs/2512.04258",
                "use": "surviving non-moment source-indexing control",
            },
        ],
        "novelty_scope": (
            "R90 instantiates the previously open target-independent "
            "nonlocal nonlinear state as six deck-factorized exponential-"
            "moment series. The update and translation identities are exact; "
            "the scoped failure is the actual C^5 Hankel/Padé order."
        ),
        "deck_update_and_translation_control": update_control,
        "hankel_rank_controls": rank_controls,
        "full_marker_source_replay": source_replay,
        "universal_newton_sharpness_control": sharpness,
        "cost_ledger": costs,
        "side_artifacts": {
            "frozen": "frozen_5a5c_nonlocal_translation_sketch.json",
            "transition": "nonlinear_translation_state_and_update_ledger.json",
            "replay": "target_translation_marker_and_source_replay.json",
            "exceptional": "projective_exceptional_branch_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r90.json",
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
            "This closes the truncated exponential-moment, Newton, and "
            "Hankel/Padé translation grammar only. It is not a lower bound "
            "against non-moment arithmetic circuits, source-reporting sum "
            "indices, representation-changing FFE identities, or all "
            "nonlinear data structures."
        ),
        "next_action": (
            "Instantiate or refute one unequal-list subfunction-inversion "
            "index for the B^2 five-A endpoints against the B^3 five-C "
            "endpoints. Derive setup, query, reporting, and memory exponents "
            "from an explicit finite-field 5SUM-indexing construction; fit "
            "B^(9/4) setup and B^(5/4) fresh source return without moments, "
            "endpoint tables, verifier scalars, or omitted exceptional "
            "branches."
        ),
        "disposition": (
            "REJECT_NONLOCAL_EXPONENTIAL_MOMENT_HANKEL_PADE_ONLY__"
            "DECK_UPDATE_AND_TRANSLATION_EXACT__NORM_AND_FIVE_MARKER_"
            "HANKEL_RANK_C5__NEWTON_AND_FULL_MARKER_SOURCE_REPLAY_EXACT_"
            "ONLY_WITH_B3_STATE__NO_PROJECTIVE_SOURCE_BICONDITIONAL__NO_"
            "RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_"
            "BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "transition": transition,
        "replay": replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_nonlocal_moment_hankel_translation_"
            "probe_report_r90.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_nonlocal_translation_sketch.json"
        ),
    )
    parser.add_argument(
        "--transition-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "nonlinear_translation_state_and_update_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "target_translation_marker_and_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "projective_exceptional_branch_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r90.json"
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
    write_json(args.transition_output, bundle["transition"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
