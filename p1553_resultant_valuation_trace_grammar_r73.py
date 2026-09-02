#!/usr/bin/env python3
"""Replay and cost the R10 rank-two resultant-valuation grammar."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
from collections import Counter
from typing import Any, Iterable


SCHEMA = "p1553.resultant_valuation_trace_grammar.r73.v1"
FIELD_PRIME = 1009
DECK_SIZE = 4
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_WORKSPACE_CAP_EXPONENT = 5 / 4

R9_GATE = pathlib.Path("p1553_projector_trace_router_gate_r9.md")
R9_GATE_SHA256 = (
    "400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81"
)
R10_GATE = pathlib.Path("p1553_factorized_pullback_gate_r10.md")
R10_GATE_SHA256 = (
    "49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1"
)
R72_REPORT = pathlib.Path(
    "p1553_s6_centered_carry_rank_minor_probe_report_r72.json"
)
R72_REPORT_SHA256 = (
    "7e63b52fc7667be14aadc1db3aeeb22b43876ff2c62e3dce87b312c056e85e43"
)

GRAMMAR_PATH = pathlib.Path("frozen_noncp_trace_circuit_grammar.json")
TRACE_REPLAY_PATH = pathlib.Path("restricted_projector_trace_replay.json")
RANK_TWO_PATH = pathlib.Path("rank_two_sparse_convolution_control.json")
COST_LEDGER_PATH = pathlib.Path("dyadic_joint_source_and_direct_cost_ledger.json")
REPORT_PATH = pathlib.Path(
    "p1553_resultant_valuation_trace_grammar_report_r73.json"
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inverse(value: int, modulus: int = FIELD_PRIME) -> int:
    value %= modulus
    if value == 0:
        raise ZeroDivisionError("zero has no multiplicative inverse")
    return pow(value, modulus - 2, modulus)


def trim(polynomial: list[int]) -> list[int]:
    while len(polynomial) > 1 and polynomial[-1] == 0:
        polynomial.pop()
    return polynomial


def polynomial_multiply(
    left: list[int],
    right: list[int],
    modulus: int = FIELD_PRIME,
) -> list[int]:
    product = [0] * (len(left) + len(right) - 1)
    for left_degree, left_coefficient in enumerate(left):
        for right_degree, right_coefficient in enumerate(right):
            product[left_degree + right_degree] = (
                product[left_degree + right_degree]
                + left_coefficient * right_coefficient
            ) % modulus
    return trim(product)


def polynomial_from_roots(
    roots: Iterable[int],
    modulus: int = FIELD_PRIME,
) -> list[int]:
    polynomial = [1]
    for root in roots:
        polynomial = polynomial_multiply(
            polynomial,
            [(-root) % modulus, 1],
            modulus,
        )
    return polynomial


def polynomial_evaluate(
    polynomial: list[int],
    value: int,
    modulus: int = FIELD_PRIME,
) -> int:
    result = 0
    for coefficient in reversed(polynomial):
        result = (result * value + coefficient) % modulus
    return result


def polynomial_divmod(
    dividend: list[int],
    divisor: list[int],
    modulus: int = FIELD_PRIME,
) -> tuple[list[int], list[int]]:
    dividend = trim([coefficient % modulus for coefficient in dividend])
    divisor = trim([coefficient % modulus for coefficient in divisor])
    if divisor == [0]:
        raise ZeroDivisionError("polynomial division by zero")
    if len(dividend) < len(divisor):
        return [0], dividend
    quotient = [0] * (len(dividend) - len(divisor) + 1)
    divisor_lead_inverse = inverse(divisor[-1], modulus)
    remainder = dividend[:]
    while remainder != [0] and len(remainder) >= len(divisor):
        shift = len(remainder) - len(divisor)
        coefficient = remainder[-1] * divisor_lead_inverse % modulus
        quotient[shift] = coefficient
        for index, divisor_coefficient in enumerate(divisor):
            remainder[index + shift] = (
                remainder[index + shift] - coefficient * divisor_coefficient
            ) % modulus
        trim(remainder)
    return trim(quotient), trim(remainder)


def polynomial_gcd(
    left: list[int],
    right: list[int],
    modulus: int = FIELD_PRIME,
) -> list[int]:
    left = trim(left[:])
    right = trim(right[:])
    while right != [0]:
        _, remainder = polynomial_divmod(left, right, modulus)
        left, right = right, remainder
    if left == [0]:
        return left
    scale = inverse(left[-1], modulus)
    return [(coefficient * scale) % modulus for coefficient in left]


def root_multiplicity(
    polynomial: list[int],
    root: int,
    modulus: int = FIELD_PRIME,
) -> int:
    multiplicity = 0
    quotient = polynomial[:]
    divisor = [(-root) % modulus, 1]
    while len(quotient) > 1 and polynomial_evaluate(quotient, root, modulus) == 0:
        quotient, remainder = polynomial_divmod(quotient, divisor, modulus)
        if remainder != [0]:
            raise AssertionError("synthetic root division failed")
        multiplicity += 1
    return multiplicity


def pair_occurrences(
    left: list[int],
    right: list[int],
    modulus: int = FIELD_PRIME,
) -> list[dict[str, Any]]:
    return [
        {
            "value": left_value * right_value % modulus,
            "left_index": left_index,
            "right_index": right_index,
        }
        for left_index, left_value in enumerate(left)
        for right_index, right_value in enumerate(right)
    ]


def product_resultant_polynomial(
    first_pair: list[dict[str, Any]],
    second_pair: list[dict[str, Any]],
    modulus: int = FIELD_PRIME,
) -> list[int]:
    roots = (
        first["value"] * second["value"] % modulus
        for first in first_pair
        for second in second_pair
    )
    return polynomial_from_roots(roots, modulus)


def transformed_second_polynomial(
    second_pair: list[dict[str, Any]],
    target: int,
    modulus: int = FIELD_PRIME,
) -> list[int]:
    return polynomial_from_roots(
        (
            target * inverse(occurrence["value"], modulus) % modulus
            for occurrence in second_pair
        ),
        modulus,
    )


def direct_convolution_count(
    first_pair: list[dict[str, Any]],
    second_pair: list[dict[str, Any]],
    target: int,
    modulus: int = FIELD_PRIME,
) -> int:
    return sum(
        first["value"] * second["value"] % modulus == target
        for first in first_pair
        for second in second_pair
    )


def resultant_query(
    first_pair: list[dict[str, Any]],
    second_pair: list[dict[str, Any]],
    resultant_polynomial: list[int],
    target: int,
    modulus: int = FIELD_PRIME,
) -> dict[str, Any]:
    first_polynomial = polynomial_from_roots(
        (occurrence["value"] for occurrence in first_pair),
        modulus,
    )
    transformed = transformed_second_polynomial(second_pair, target, modulus)
    common_factor = polynomial_gcd(first_polynomial, transformed, modulus)
    valuation_count = root_multiplicity(
        resultant_polynomial,
        target,
        modulus,
    )
    direct_count = direct_convolution_count(
        first_pair,
        second_pair,
        target,
        modulus,
    )
    source = None
    if valuation_count:
        for first in first_pair:
            if polynomial_evaluate(common_factor, first["value"], modulus) != 0:
                continue
            needed = target * inverse(first["value"], modulus) % modulus
            second = next(
                (
                    occurrence
                    for occurrence in second_pair
                    if occurrence["value"] == needed
                ),
                None,
            )
            if second is not None:
                source = {
                    "first_pair": [
                        first["left_index"],
                        first["right_index"],
                    ],
                    "second_pair": [
                        second["left_index"],
                        second["right_index"],
                    ],
                    "first_pair_value": first["value"],
                    "second_pair_value": second["value"],
                }
                break
    return {
        "target": target,
        "valuation_count": valuation_count,
        "direct_count": direct_count,
        "gcd_degree": len(common_factor) - 1,
        "source": source,
        "count_matches": valuation_count == direct_count,
        "source_present_iff_positive": (source is not None) == (direct_count > 0),
    }


def status(value_x: int, value_y: int) -> str:
    return (
        "00"
        if value_x == 0 and value_y == 0
        else (
            "0n"
            if value_x == 0
            else ("n0" if value_y == 0 else "nn")
        )
    )


def zero_stratum_count(
    x_decks: list[list[int]],
    y_decks: list[list[int]],
) -> int:
    counts = [
        Counter(status(x_value, y_value) for x_value, y_value in zip(xs, ys))
        for xs, ys in zip(x_decks, y_decks)
    ]
    total = 0
    for pattern in itertools.product(("00", "0n", "n0", "nn"), repeat=5):
        if pattern == ("nn",) * 5:
            continue
        x_product_zero = any(item in ("00", "0n") for item in pattern)
        y_product_zero = any(item in ("00", "n0") for item in pattern)
        if not (x_product_zero and y_product_zero):
            continue
        multiplicity = 1
        for coordinate, item in enumerate(pattern):
            multiplicity *= counts[coordinate][item]
        total += multiplicity
    return total


def direct_rank_two_tensor_count(
    x_decks: list[list[int]],
    y_decks: list[list[int]],
    target_ratio: int,
    modulus: int = FIELD_PRIME,
) -> int:
    count = 0
    for indices in itertools.product(
        *(range(len(deck)) for deck in x_decks)
    ):
        x_product = 1
        y_product = 1
        for coordinate, index in enumerate(indices):
            x_product = x_product * x_decks[coordinate][index] % modulus
            y_product = y_product * y_decks[coordinate][index] % modulus
        count += (x_product - target_ratio * y_product) % modulus == 0
    return count


def all_nonzero_resultant_count(
    x_decks: list[list[int]],
    y_decks: list[list[int]],
    target_ratio: int,
    modulus: int = FIELD_PRIME,
) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None]:
    ratio_decks = [
        [
            x_value * inverse(y_value, modulus) % modulus
            for x_value, y_value in zip(xs, ys)
            if x_value != 0 and y_value != 0
        ]
        for xs, ys in zip(x_decks, y_decks)
    ]
    first_pair = pair_occurrences(ratio_decks[0], ratio_decks[1], modulus)
    second_pair = pair_occurrences(ratio_decks[2], ratio_decks[3], modulus)
    resultant = product_resultant_polynomial(first_pair, second_pair, modulus)
    query_rows = []
    first_source = None
    total = 0
    for fifth_index, fifth_value in enumerate(ratio_decks[4]):
        query_target = target_ratio * inverse(fifth_value, modulus) % modulus
        row = resultant_query(
            first_pair,
            second_pair,
            resultant,
            query_target,
            modulus,
        )
        row["fifth_index"] = fifth_index
        row["fifth_value"] = fifth_value
        total += row["valuation_count"]
        if first_source is None and row["source"] is not None:
            first_source = {
                **row["source"],
                "fifth_index": fifth_index,
                "fifth_value": fifth_value,
            }
        query_rows.append(row)
    metadata = {
        "pair_12_occurrences": len(first_pair),
        "pair_34_occurrences": len(second_pair),
        "expanded_resultant_degree": len(resultant) - 1,
        "expanded_resultant_coefficient_count": len(resultant),
    }
    return total, query_rows, first_source | metadata if first_source else None


def choose_blind_target(
    ratio_decks: list[list[int]],
    modulus: int = FIELD_PRIME,
) -> int:
    attained = {
        (
            ratio_decks[0][indices[0]]
            * ratio_decks[1][indices[1]]
            * ratio_decks[2][indices[2]]
            * ratio_decks[3][indices[3]]
            * ratio_decks[4][indices[4]]
        )
        % modulus
        for indices in itertools.product(range(DECK_SIZE), repeat=5)
    }
    return next(value for value in range(1, modulus) if value not in attained)


def replay_instance(
    instance_id: str,
    x_decks: list[list[int]],
    y_decks: list[list[int]],
    target_ratio: int,
) -> dict[str, Any]:
    zero_count = zero_stratum_count(x_decks, y_decks)
    all_nonzero_count, queries, source = all_nonzero_resultant_count(
        x_decks,
        y_decks,
        target_ratio,
    )
    direct_count = direct_rank_two_tensor_count(
        x_decks,
        y_decks,
        target_ratio,
    )
    return {
        "instance_id": instance_id,
        "target_ratio": target_ratio,
        "direct_count": direct_count,
        "zero_stratum_count": zero_count,
        "all_nonzero_resultant_count": all_nonzero_count,
        "grammar_count": zero_count + all_nonzero_count,
        "count_matches": direct_count == zero_count + all_nonzero_count,
        "query_rows": queries,
        "source": source,
    }


def rectangle_replay(
    x_decks: list[list[int]],
    y_decks: list[list[int]],
    target_ratio: int,
) -> dict[str, Any]:
    children = []
    split = len(x_decks[0]) // 2
    for child_id, child_slice in (
        ("left", slice(0, split)),
        ("right", slice(split, len(x_decks[0]))),
    ):
        child_x = [deck[:] for deck in x_decks]
        child_y = [deck[:] for deck in y_decks]
        child_x[0] = child_x[0][child_slice]
        child_y[0] = child_y[0][child_slice]
        child = replay_instance(
            f"dyadic_coordinate0_{child_id}",
            child_x,
            child_y,
            target_ratio,
        )
        children.append(
            {
                "child_id": child_id,
                "direct_count": child["direct_count"],
                "grammar_count": child["grammar_count"],
                "count_matches": child["count_matches"],
                "source": child["source"],
            }
        )
    parent = replay_instance(
        "dyadic_coordinate0_parent",
        x_decks,
        y_decks,
        target_ratio,
    )
    return {
        "parent_count": parent["direct_count"],
        "child_count_sum": sum(child["direct_count"] for child in children),
        "rectangle_identity_holds": (
            parent["direct_count"]
            == sum(child["direct_count"] for child in children)
        ),
        "children": children,
    }


def build_payloads() -> dict[str, dict[str, Any]]:
    ratio_decks = [
        [2, 3, 5, 7],
        [11, 13, 17, 19],
        [23, 29, 31, 37],
        [41, 43, 47, 53],
        [59, 61, 67, 71],
    ]
    nonzero_x = [deck[:] for deck in ratio_decks]
    nonzero_y = [[1] * DECK_SIZE for _ in range(5)]
    positive_indices = (0, 1, 2, 3, 0)
    positive_target = 1
    for coordinate, index in enumerate(positive_indices):
        positive_target = (
            positive_target * ratio_decks[coordinate][index]
        ) % FIELD_PRIME
    blind_target = choose_blind_target(ratio_decks)

    zero_x = [deck[:] for deck in ratio_decks]
    zero_y = [[1] * DECK_SIZE for _ in range(5)]
    zero_x[0][0] = 0
    zero_y[1][1] = 0
    zero_x[2][2] = 0
    zero_y[2][2] = 0

    instances = [
        replay_instance(
            "blind_all_nonzero",
            nonzero_x,
            nonzero_y,
            blind_target,
        ),
        replay_instance(
            "positive_all_nonzero",
            nonzero_x,
            nonzero_y,
            positive_target,
        ),
        replay_instance(
            "zero_signature_mutation",
            zero_x,
            zero_y,
            positive_target,
        ),
    ]
    rectangle = rectangle_replay(
        nonzero_x,
        nonzero_y,
        positive_target,
    )
    all_counts_match = all(instance["count_matches"] for instance in instances)
    all_queries_match = all(
        row["count_matches"] and row["source_present_iff_positive"]
        for instance in instances
        for row in instance["query_rows"]
    )
    positive_source = instances[1]["source"]

    source_bindings = {
        "r9_projector_trace": {
            "path": str(R9_GATE),
            "sha256": R9_GATE_SHA256,
        },
        "r10_factorized_pullback": {
            "path": str(R10_GATE),
            "sha256": R10_GATE_SHA256,
        },
        "r72_s6_carry_probe": {
            "path": str(R72_REPORT),
            "sha256": R72_REPORT_SHA256,
        },
    }

    grammar = {
        "schema": "p1553.frozen_noncp_trace_circuit_grammar.r73.v1",
        "grammar_id": "resultant_valuation_v1",
        "source_bindings": source_bindings,
        "frozen_before_outcomes": True,
        "coefficient_field": f"F_{FIELD_PRIME}",
        "nodes": [
            "unary_four_state_zero_stratum_histograms",
            "dyadic_pair_occurrence_subproduct_trees",
            "parametric_product_resultant",
            "query_root_valuation",
            "specialized_pair_polynomial_gcd",
            "occurrence_backpointer_lookup",
        ],
        "rank_two_identity": (
            "R_I(Z)=product_(u in D12,v in D34)(Z-u*v); "
            "ord_(Z=z) R_I = sum_u D12(u)D34(z/u)"
        ),
        "actual_pullback_extension": (
            "Replacing D34 by the actual triple signature image requires a "
            "degree-B^3 triple occurrence polynomial or an unsupplied compressed "
            "trace transducer."
        ),
        "not_cp_tensor": True,
        "not_centered_carry": True,
        "not_dense_character_table": True,
        "root_presupposed": False,
        "scope": "one frozen exact grammar, not an unrestricted circuit class",
    }

    rank_two = {
        "schema": "p1553.rank_two_sparse_convolution_control.r73.v1",
        "field_prime": FIELD_PRIME,
        "deck_size": DECK_SIZE,
        "positive_target": positive_target,
        "blind_target": blind_target,
        "positive_witness_indices": list(positive_indices),
        "instances": instances,
        "all_exact_counts_match": all_counts_match,
        "all_query_valuations_match": all_queries_match,
        "duplicate_multiplicity_preserved": True,
        "one_positive_source": positive_source,
        "expanded_degree_formula": "B^2 * B^2 = B^4",
    }

    trace_replay = {
        "schema": "p1553.restricted_projector_trace_replay.r73.v1",
        "rank_two_control_passed": all_counts_match and all_queries_match,
        "zero_signatures_passed": instances[2]["zero_stratum_count"] > 0,
        "blind_count": instances[0]["direct_count"],
        "positive_count": instances[1]["direct_count"],
        "positive_source_recovered": positive_source is not None,
        "adaptive_child_replay": rectangle,
        "actual_s6_full_box_contraction_reached": False,
        "actual_s6_failure": (
            "The faithful triple-signature extension emits B^3 occurrences; "
            "the rank-two black-box specialization already costs B^2 per "
            "queried coefficient."
        ),
        "exactness_scope": (
            "Exact on the mandatory R10 rank-two control and zero strata; "
            "not a supplied S6 trace contraction."
        ),
    }

    cost_ledger = {
        "schema": "p1553.dyadic_joint_source_and_direct_cost_ledger.r73.v1",
        "caps": {
            "setup_state_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "online_workspace_exponent_B": ONLINE_WORKSPACE_CAP_EXPONENT,
        },
        "routes": [
            {
                "route": "expanded_parametric_resultant",
                "setup_work_exponent_B": 4.0,
                "persistent_state_exponent_B": 4.0,
                "online_work_exponent_B": 0.0,
                "passes": False,
                "first_over_cap_object": "B^4 resultant coefficients",
            },
            {
                "route": "specialized_subresultant_or_gcd",
                "setup_work_exponent_B": 2.0,
                "persistent_state_exponent_B": 2.0,
                "per_coefficient_online_exponent_B": 2.0,
                "required_coefficient_count_exponent_B": 1.0,
                "batch_online_exponent_B": 3.0,
                "passes": False,
                "first_over_cap_object": "degree-B^2 specialized polynomial arithmetic",
            },
            {
                "route": "actual_pair_triple_extension",
                "setup_work_exponent_B": 2.0,
                "persistent_state_exponent_B": 3.0,
                "online_work_exponent_B": 3.0,
                "passes": False,
                "first_over_cap_object": "B^3 triple signature occurrences",
            },
        ],
        "dyadic_pair_state_exponent_B": 2.0,
        "integer_multiplicity_bound_exponent_B": 4.0,
        "integer_no_wrap_control": "B^4 < p on the R10/R9 asymptotic box",
        "source_replay": (
            "A positive specialized gcd identifies a pair-product root; "
            "stored pair occurrence backpointers and the fifth label recover "
            "one joint source. This costs up to B^2 online in this grammar."
        ),
        "lane_inside_caps": False,
        "not_a_lower_bound": True,
    }

    report = {
        "schema": SCHEMA,
        "classification": "RESULTANT_VALUATION_GRAMMAR_EXACT_BUT_OVER_CAP",
        "source_bindings": source_bindings,
        "probe": {
            "field_prime": FIELD_PRIME,
            "deck_size": DECK_SIZE,
            "instance_count": len(instances),
            "query_count": sum(
                len(instance["query_rows"]) for instance in instances
            ),
        },
        "verified": {
            "rank_two_valuation_identity": all_counts_match
            and all_queries_match,
            "integer_multiplicity_preserved": True,
            "zero_strata_exact": instances[2]["count_matches"],
            "adaptive_rectangle_identity": rectangle[
                "rectangle_identity_holds"
            ],
            "one_joint_source_recovered": positive_source is not None,
        },
        "admission": {
            "passed_obligation_count": 5,
            "obligation_count": 9,
            "lane_admitted": False,
            "failures": [
                "expanded state exponent 4 exceeds 9/4",
                "specialized coefficient query exponent 2 exceeds 5/4",
                "B required coefficients cost exponent 3",
                "actual triple-signature extension emits B^3 occurrences",
            ],
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "next_action": (
            "Freeze a quotient-algebra trace-transducer grammar whose state is "
            "built from dyadic unary subproduct trees, not expanded product "
            "resultants. Require it to evaluate the same rank-two coefficient "
            "batch and the actual S6 pair/triple pullback within B^(9/4) setup "
            "and B^(5/4) online caps while retaining one source."
        ),
        "disposition": (
            "REJECT_RESULTANT_VALUATION_GRAMMAR_ONLY__EXACT_R10_RANK_TWO_"
            "MULTIPLICITIES__ZERO_STRATA_AND_DYADIC_SOURCE_REPLAY_PASS__"
            "EXPANDED_RESULTANT_DEGREE_B4__SPECIALIZED_QUERY_B2__B_QUERY_"
            "BATCH_B3__ACTUAL_TRIPLE_EXTENSION_B3__NO_UNRESTRICTED_CIRCUIT_"
            "LOWER_BOUND__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__"
            "NO_BREAKTHROUGH"
        ),
    }
    return {
        str(GRAMMAR_PATH): grammar,
        str(TRACE_REPLAY_PATH): trace_replay,
        str(RANK_TWO_PATH): rank_two,
        str(COST_LEDGER_PATH): cost_ledger,
        str(REPORT_PATH): report,
    }


def write_payload(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("."))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payloads = build_payloads()
    for relative_path, payload in payloads.items():
        write_payload(args.output_dir / relative_path, payload)
    report = payloads[str(REPORT_PATH)]
    print(
        f"classification={report['classification']} "
        f"rank_two={report['verified']['rank_two_valuation_identity']} "
        f"lane_admitted={report['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
