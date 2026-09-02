#!/usr/bin/env python3
"""Audit digitized multi-edge equality projectors after R98."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.5a5c_multiedge_digitized_equality_projector.r99.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
SOURCE_EXPONENT = Fraction(12, 5)
FIELD_EXPONENT = Fraction(5, 1)

R98_PRODUCER = pathlib.Path(
    "p1553_5a5c_nonlinear_tensor_tower_trace_probe_r98.py"
)
R98_PRODUCER_SHA256 = (
    "41d98744ff051a6dd259323e7a1298aea77a14b2d12d7cf0149b6eb592297a1e"
)
R98_REPORT = pathlib.Path(
    "p1553_5a5c_nonlinear_tensor_tower_trace_probe_report_r98.json"
)
R98_REPORT_SHA256 = (
    "cb322e28620a4634ffac0c474d772c06a26010f10b3dc32c97c9e75637a90dfa"
)
R98_GATE = pathlib.Path(
    "p1553_5a5c_nonlinear_tensor_tower_trace_probe_gate_r98.md"
)
R98_GATE_SHA256 = (
    "7acd34a7e3286edd658089a7d258688f9e327ffce03a185c762eee4eade06f3c"
)
R97_REPORT = pathlib.Path(
    "p1553_5a5c_factored_transposed_projector_"
    "trace_probe_report_r97.json"
)
R97_REPORT_SHA256 = (
    "84217a15910b0ed15b6b69c3ec8875f7f2b913a17e4c70fc89cc905c25f492f6"
)
R97_GATE = pathlib.Path(
    "p1553_5a5c_factored_transposed_projector_trace_probe_gate_r97.md"
)
R97_GATE_SHA256 = (
    "ef91d8a15dc7a218c21355b5b2f2777db948978514ebed26d6884ce929c09182"
)
R94_REPORT = pathlib.Path(
    "p1553_5a5c_implicit_veronese_hyperplane_"
    "source_index_probe_report_r94.json"
)
R94_REPORT_SHA256 = (
    "a9966f1fb407e720ce1e9aaafc82ab4a38eade9f338cda287cb737445d099df0"
)
R94_GATE = pathlib.Path(
    "p1553_5a5c_implicit_veronese_hyperplane_"
    "source_index_probe_gate_r94.md"
)
R94_GATE_SHA256 = (
    "66d2004d1b4dfa63ac69f60397da45e379dfca4bb8db12ede0647f028aaf517d"
)
R85_GATE = pathlib.Path(
    "p1553_5a5c_target_uniform_precoefficient_circuit_probe_gate_r85.md"
)
R85_GATE_SHA256 = (
    "bb052ad21fde0f8f35dd56e014a6ab3a1ba1e148fd697fb80af230e4d3c4029a"
)
P1513_HANDOFF = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1513_idea121_direct_ku_handoff_v3_20260717.md"
)
P1513_HANDOFF_SHA256 = (
    "27c8f1f15fd0c3b81ebe2008aa96db12417c3f6612c5c151212206dcba388dcc"
)

Polynomial = list[int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R98_PRODUCER: R98_PRODUCER_SHA256,
        R98_REPORT: R98_REPORT_SHA256,
        R98_GATE: R98_GATE_SHA256,
        R97_REPORT: R97_REPORT_SHA256,
        R97_GATE: R97_GATE_SHA256,
        R94_REPORT: R94_REPORT_SHA256,
        R94_GATE: R94_GATE_SHA256,
        R85_GATE: R85_GATE_SHA256,
        P1513_HANDOFF: P1513_HANDOFF_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R99 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def radix_digits(value: int, bit_count: int) -> tuple[int, ...]:
    return tuple((value >> bit) & 1 for bit in range(bit_count))


def digit_equality(
    left: tuple[int, ...],
    right: tuple[int, ...],
    prime: int,
) -> int:
    if len(left) != len(right):
        raise ValueError("digit lengths must match")
    result = 1
    for left_digit, right_digit in zip(left, right):
        difference = (left_digit - right_digit) % prime
        result = result * (1 - difference * difference) % prime
    return result


def interpolate_values(values: list[int], prime: int) -> Polynomial:
    if len(values) != prime:
        raise ValueError("one value per field element is required")
    augmented = [
        [
            pow(x_value, degree, prime)
            for degree in range(prime)
        ]
        + [values[x_value] % prime]
        for x_value in range(prime)
    ]
    for column in range(prime):
        pivot = next(
            row
            for row in range(column, prime)
            if augmented[row][column]
        )
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        inverse = pow(augmented[column][column], -1, prime)
        augmented[column] = [
            value * inverse % prime
            for value in augmented[column]
        ]
        for row in range(prime):
            if row == column or not augmented[row][column]:
                continue
            multiplier = augmented[row][column]
            augmented[row] = [
                (left - multiplier * right) % prime
                for left, right in zip(
                    augmented[row], augmented[column]
                )
            ]
    return [augmented[row][-1] for row in range(prime)]


def polynomial_degree(polynomial: Polynomial, prime: int) -> int:
    for degree in range(len(polynomial) - 1, -1, -1):
        if polynomial[degree] % prime:
            return degree
    return -1


def polynomial_evaluate(
    polynomial: Polynomial,
    value: int,
    prime: int,
) -> int:
    result = 0
    for coefficient in reversed(polynomial):
        result = (result * value + coefficient) % prime
    return result


@functools.lru_cache(maxsize=1)
def digit_channel_controls() -> dict[str, Any]:
    sweep = []
    for prime in (5, 7, 11, 13, 17):
        bit_count = math.ceil(math.log2(prime))
        encodings = [
            radix_digits(value, bit_count) for value in range(prime)
        ]
        equality_exact = all(
            digit_equality(encodings[left], encodings[right], prime)
            == int(left == right)
            for left in range(prime)
            for right in range(prime)
        )
        fiber_rows = []
        for bit in range(bit_count):
            for symbol in (0, 1):
                values = [
                    int(encoding[bit] == symbol)
                    for encoding in encodings
                ]
                fiber_size = sum(values)
                if fiber_size == 0:
                    continue
                polynomial = interpolate_values(values, prime)
                exact = all(
                    polynomial_evaluate(polynomial, value, prime)
                    == values[value]
                    for value in range(prime)
                )
                fiber_rows.append(
                    {
                        "bit": bit,
                        "symbol": symbol,
                        "fiber_size": fiber_size,
                        "interpolation_degree": polynomial_degree(
                            polynomial, prime
                        ),
                        "expected_degree": prime - 1,
                        "interpolation_exact": exact,
                        "coefficient_words": len(polynomial),
                        "root_list_words": fiber_size,
                    }
                )
        sweep.append(
            {
                "prime": prime,
                "bit_count": bit_count,
                "channel_alphabet_sizes": [2] * bit_count,
                "alphabet_product": 2**bit_count,
                "alphabet_product_covers_field": (
                    2**bit_count >= prime
                ),
                "encoding_injective": len(set(encodings)) == prime,
                "digit_equality_exact": equality_exact,
                "flattened_cut_capacity": 2**bit_count,
                "fiber_rows": fiber_rows,
                "all_nonempty_fiber_indicators_degree_p_minus_1": all(
                    row["interpolation_degree"] == prime - 1
                    for row in fiber_rows
                ),
                "all_fiber_interpolations_exact": all(
                    row["interpolation_exact"] for row in fiber_rows
                ),
                "materialized_digit_table_words": prime * bit_count,
                "all_fiber_root_words": sum(
                    row["root_list_words"] for row in fiber_rows
                ),
                "all_fiber_coefficient_words": sum(
                    row["coefficient_words"] for row in fiber_rows
                ),
            }
        )
    return {
        "radix": 2,
        "sweep": sweep,
        "all_digit_encodings_injective": all(
            row["encoding_injective"] for row in sweep
        ),
        "all_digit_equalities_exact": all(
            row["digit_equality_exact"] for row in sweep
        ),
        "all_alphabet_products_cover_field": all(
            row["alphabet_product_covers_field"] for row in sweep
        ),
        "all_fiber_indicators_degree_p_minus_1": all(
            row["all_nonempty_fiber_indicators_degree_p_minus_1"]
            for row in sweep
        ),
        "all_fiber_interpolations_exact": all(
            row["all_fiber_interpolations_exact"] for row in sweep
        ),
        "degree_theorem": (
            "For a nonempty proper S subset F_p, the unique polynomial "
            "indicator 1_S has x^(p-1) coefficient -|S| != 0, hence "
            "degree p-1."
        ),
        "rank_compatibility": (
            "The product of channel alphabet sizes is at least p, so the "
            "flattened multi-edge cut capacity remains compatible with the "
            "rank-p equality kernel even though each supplied digit edge "
            "has width two."
        ),
        "scope": (
            "Exact canonical-radix positive control plus materialized digit "
            "tables, fiber interpolation tables, and fiber root lists. "
            "Degree p-1 is not by itself an arithmetic-circuit lower bound."
        ),
    }


def dyadic_source(
    occurrences: list[int],
    target: int,
) -> dict[str, Any]:
    lower = 0
    upper = len(occurrences)
    transcript = []
    queried_occurrences = 0
    while True:
        selected = occurrences[lower:upper]
        count = sum(value == target for value in selected)
        queried_occurrences += len(selected)
        transcript.append(
            {"range": [lower, upper], "match_count": count}
        )
        if count == 0:
            return {
                "source_index": None,
                "returned_bottom": True,
                "queried_occurrences": queried_occurrences,
                "transcript": transcript,
            }
        if upper - lower == 1:
            return {
                "source_index": lower,
                "source_value": occurrences[lower],
                "returned_bottom": False,
                "returned_source_matches_target": (
                    occurrences[lower] == target
                ),
                "queried_occurrences": queried_occurrences,
                "transcript": transcript,
            }
        middle = (lower + upper) // 2
        left_count = sum(
            value == target for value in occurrences[lower:middle]
        )
        queried_occurrences += middle - lower
        transcript.append(
            {"range": [lower, middle], "match_count": left_count}
        )
        if left_count:
            upper = middle
        else:
            lower = middle


@functools.lru_cache(maxsize=1)
def occurrence_digit_controls() -> dict[str, Any]:
    prime = 101
    bit_count = math.ceil(math.log2(prime))
    occurrences = [2, 2, 5, 7, 9, 12, 20, 31]
    encodings = [
        radix_digits(value, bit_count) for value in occurrences
    ]
    positive_target = 2
    positive_digits = radix_digits(positive_target, bit_count)
    blind_target = 3
    blind_digits = radix_digits(blind_target, bit_count)
    positive_count = sum(
        digit_equality(encoding, positive_digits, prime)
        for encoding in encodings
    )
    blind_count = sum(
        digit_equality(encoding, blind_digits, prime)
        for encoding in encodings
    )
    return {
        "field_prime": prime,
        "bit_count": bit_count,
        "occurrence_values": occurrences,
        "materialized_occurrence_digit_words": (
            len(occurrences) * bit_count
        ),
        "positive_target": positive_target,
        "positive_integer_count": positive_count,
        "positive_dyadic_source": dyadic_source(
            occurrences, positive_target
        ),
        "blind_target": blind_target,
        "blind_integer_count": blind_count,
        "blind_dyadic_source": dyadic_source(
            occurrences, blind_target
        ),
        "duplicate_source_indices": [
            index
            for index, value in enumerate(occurrences)
            if value == positive_target
        ],
        "integer_counts_below_prime": (
            positive_count < prime and blind_count < prime
        ),
    }


def asymptotic_cost_control() -> dict[str, Any]:
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "supplied_radix_digits": {
            "edge_count": "Theta(log p)=Theta(log B)",
            "edge_width": 2,
            "equality_work": "Theta(log p)",
            "flattened_cut_capacity": "2^ceil(log2 p) in [p,2p)",
            "representation_positive_control": True,
        },
        "full_field_materialized_constructors": {
            "digit_table_words": "Theta(p log p)",
            "fiber_root_words": "Theta(p log p)",
            "fiber_coefficient_words": "Theta(p log p)",
            "exponent_B": fraction_record(FIELD_EXPONENT),
            "inside_setup_cap": FIELD_EXPONENT <= SETUP_CAP,
        },
        "sourcewise_digit_extraction": {
            "source_message_count": "D=Theta(B^(12/5))",
            "digit_words_or_extractor_calls": "Theta(D log p)",
            "exponent_B": fraction_record(SOURCE_EXPONENT),
            "inside_setup_cap": SOURCE_EXPONENT <= SETUP_CAP,
            "inside_online_cap": SOURCE_EXPONENT <= ONLINE_CAP,
        },
        "canonical_integer_digit_access_is_free_field_operation": False,
        "succinct_aggregate_digit_index_from_compact_divisors_supplied": False,
        "low_circuit_fiber_indicator_family_supplied": False,
        "scope": (
            "Charges materialized canonical-radix tables and per-source "
            "digit extraction only. It does not lower-bound a succinct "
            "arithmetic digit extractor or an aggregate digit trie built "
            "directly from compact A/C divisor circuits."
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    channels = digit_channel_controls()
    occurrences = occurrence_digit_controls()
    costs = asymptotic_cost_control()
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_multiedge_digitized_"
            "equality_projector.r99.v1"
        ),
        "field_encoding": "canonical integer representative in [0,p)",
        "channels": "binary radix digits",
        "per_edge_projector": "1-(d_i(u)-d_i(v))^2",
        "combined_projector": "product over all digit edges",
        "caps": costs["caps"],
        "excluded_unfrozen_grammar": (
            "succinct arithmetic digit extractors and aggregate digit tries "
            "derived directly from compact A/C divisor circuits"
        ),
    }
    state_ledger = {
        "schema": (
            "p1553.digit_extractor_state_cut_capacity_ledger.r99.v1"
        ),
        "digit_channel_controls": channels,
        "asymptotic_cost": costs,
        "standard_digit_constructors_inside_caps": False,
    }
    source_replay = {
        "schema": (
            "p1553.digitized_projector_integer_source_replay.r99.v1"
        ),
        "occurrence_controls": occurrences,
        "actual_five_a_five_c_integer_lift_complete": False,
        "actual_five_a_five_c_source_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.digitized_projector_exceptional_controls.r99.v1"
        ),
        "blind_bottom_exact": (
            occurrences["blind_integer_count"] == 0
            and occurrences["blind_dyadic_source"]["returned_bottom"]
        ),
        "duplicate_occurrence_count_exact": (
            occurrences["positive_integer_count"] == 2
        ),
        "projective_infinity_complete": False,
        "proper_subsum_complete": False,
        "tangent_source_complete": False,
        "multiplicity_complete_actual_source_return": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r99.v1",
        "standard_digit_constructors_inside_caps": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "nine_source_bindings_verified": len(bindings) == 9,
        "canonical_radix_encoding_injective": channels[
            "all_digit_encodings_injective"
        ],
        "digitized_equality_projector_exact": channels[
            "all_digit_equalities_exact"
        ],
        "alphabet_product_covers_field": channels[
            "all_alphabet_products_cover_field"
        ],
        "flattened_cut_rank_compatibility_recorded": "at least p"
        in channels["rank_compatibility"],
        "fiber_indicator_interpolations_exact": channels[
            "all_fiber_interpolations_exact"
        ],
        "nonempty_fiber_indicators_degree_p_minus_1": channels[
            "all_fiber_indicators_degree_p_minus_1"
        ],
        "table_root_and_coefficient_state_charged": all(
            row["materialized_digit_table_words"] > 0
            and row["all_fiber_root_words"]
            == row["prime"] * row["bit_count"]
            for row in channels["sweep"]
        ),
        "duplicate_occurrence_count_exact": exceptional[
            "duplicate_occurrence_count_exact"
        ],
        "blind_bottom_exact": exceptional["blind_bottom_exact"],
        "toy_dyadic_source_exact": (
            occurrences["positive_dyadic_source"]["source_index"] == 0
            and occurrences["positive_dyadic_source"][
                "returned_source_matches_target"
            ]
        ),
        "integer_no_wrap_replayed": occurrences[
            "integer_counts_below_prime"
        ],
        "supplied_digit_representation_positive_control": costs[
            "supplied_radix_digits"
        ]["representation_positive_control"],
        "source_table_charged_as_B12O5": (
            costs["sourcewise_digit_extraction"]["exponent_B"]["exact"]
            == "12/5"
        ),
        "full_lookup_charged_as_B5": (
            costs["full_field_materialized_constructors"][
                "exponent_B"
            ]["exact"]
            == "5"
        ),
        "aggregate_compact_digit_index_supplied": False,
        "standard_digit_setup_inside_cap": False,
        "standard_digit_online_inside_cap": False,
        "field_operation_digit_extractor_proved": False,
        "actual_five_a_five_c_integer_lift_complete": False,
        "actual_five_a_five_c_source_unranking_complete": False,
        "projective_infinity_source_complete": False,
        "proper_subsum_source_complete": False,
        "tangent_source_complete": False,
        "multiplicity_complete_actual_source_return": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
        "succinct_low_circuit_fiber_indicator_family_supplied": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "SUPPLIED_RADIX_DIGITS_DECOMPOSE_EQUALITY_EXACTLY__"
            "FLATTENED_CAPACITY_STILL_P__STANDARD_DIGIT_TABLE_B5__"
            "SOURCEWISE_DIGIT_TRAFFIC_B12O5__AGGREGATE_DIGIT_INDEX_OPEN"
        ),
        "source_bindings": {
            "r98_producer": {
                "path": str(R98_PRODUCER),
                "sha256": R98_PRODUCER_SHA256,
            },
            "r98_report": {
                "path": str(R98_REPORT),
                "sha256": R98_REPORT_SHA256,
            },
            "r98_gate": {
                "path": str(R98_GATE),
                "sha256": R98_GATE_SHA256,
            },
            "r97_report": {
                "path": str(R97_REPORT),
                "sha256": R97_REPORT_SHA256,
            },
            "r97_gate": {
                "path": str(R97_GATE),
                "sha256": R97_GATE_SHA256,
            },
            "r94_report": {
                "path": str(R94_REPORT),
                "sha256": R94_REPORT_SHA256,
            },
            "r94_gate": {
                "path": str(R94_GATE),
                "sha256": R94_GATE_SHA256,
            },
            "r85_gate": {
                "path": str(R85_GATE),
                "sha256": R85_GATE_SHA256,
            },
            "p1513_handoff": {
                "path": str(P1513_HANDOFF),
                "sha256": P1513_HANDOFF_SHA256,
            },
        },
        "novelty_scope": (
            "R99 is the first campaign receipt to credit an exact small-edge "
            "digitized equality representation while separating supplied "
            "digits from the aggregate compact-source constructor."
        ),
        "digit_channel_controls": channels,
        "occurrence_digit_controls": occurrences,
        "asymptotic_cost_control": costs,
        "side_artifacts": {
            "frozen": (
                "frozen_5a5c_multiedge_digitized_"
                "equality_projector.json"
            ),
            "state_ledger": (
                "digit_extractor_state_and_cut_capacity_ledger.json"
            ),
            "source_replay": (
                "digitized_projector_integer_source_replay.json"
            ),
            "exceptional": (
                "digitized_projector_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r99.json",
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
            "R99 positively verifies supplied radix-digit equality, then "
            "closes materialized digit tables, fiber interpolation/root "
            "tables, and per-source extraction. Degree p-1 is not an "
            "arithmetic-circuit lower bound, and a succinct aggregate digit "
            "index from compact A/C divisor circuits remains open."
        ),
        "next_action": (
            "Construct or refute one succinct aggregate digit trie from the "
            "compact A/C divisor circuits. Freeze the arithmetic digit or "
            "fiber-indicator circuit and aggregation law before outcomes; "
            "require setup/state below B^(9/4), fresh count and complete "
            "source return below B^(5/4), no sourcewise D traffic or p-size "
            "advice, exact multiplicity and all exceptional branches, "
            "known-RHS rank, factor logs, and identical target descent."
        ),
        "disposition": (
            "REJECT_STANDARD_DIGIT_CONSTRUCTORS_ONLY__SUPPLIED_RADIX_"
            "DIGITS_GIVE_EXACT_SMALL_EDGE_EQUALITY__FLATTENED_CAPACITY_P__"
            "NONEMPTY_FIBER_INDICATORS_DEGREE_P_MINUS_1__MATERIALIZED_"
            "TABLE_ROOT_AND_COEFFICIENT_STATE_B5__SOURCEWISE_DIGIT_TRAFFIC_"
            "B12O5__DUPLICATE_COUNT_AND_TOY_SOURCE_EXACT__SUCCINCT_"
            "AGGREGATE_DIGIT_TRIE_OPEN__ACTUAL_5A5C_AND_PROJECTIVE_BRANCHES_"
            "INCOMPLETE__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_"
            "CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "state_ledger": state_ledger,
        "source_replay": source_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_multiedge_digitized_equality_"
            "projector_probe_report_r99.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_multiedge_digitized_"
            "equality_projector.json"
        ),
    )
    parser.add_argument(
        "--state-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "digit_extractor_state_and_cut_capacity_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "digitized_projector_integer_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "digitized_projector_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r99.json"
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
    write_json(args.state_ledger_output, bundle["state_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
