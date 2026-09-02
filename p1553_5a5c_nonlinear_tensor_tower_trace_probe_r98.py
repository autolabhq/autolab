#!/usr/bin/env python3
"""Audit one-bond nonlinear tensor projector traces after R97."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.5a5c_nonlinear_tensor_tower_trace.r98.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
SOURCE_EXPONENT = Fraction(12, 5)
FIELD_EXPONENT = Fraction(5, 1)

R97_PRODUCER = pathlib.Path(
    "p1553_5a5c_factored_transposed_projector_trace_probe_r97.py"
)
R97_PRODUCER_SHA256 = (
    "f9015647b49a246680258ce259dfa49a80a237172162e34f3e4390b87b726972"
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
R95_REPORT = pathlib.Path(
    "p1553_5a5c_aggregate_veronese_projector_"
    "recurrence_probe_report_r95.json"
)
R95_REPORT_SHA256 = (
    "50441deecd5fadcb05d30c09532d2b5fe6e8893ea2530243c20a52761e0a1742"
)
R95_GATE = pathlib.Path(
    "p1553_5a5c_aggregate_veronese_projector_"
    "recurrence_probe_gate_r95.md"
)
R95_GATE_SHA256 = (
    "6645a20b942f4c86abafed39fa14538b1bb80edee3ffa94a2e7d914e12c0c763"
)
R78_REPORT = pathlib.Path(
    "p1553_actual_s6_fermat_tensor_train_probe_report_r78.json"
)
R78_REPORT_SHA256 = (
    "e590002c433c6504725d5ab7ff1dba97ad8c15400bf0117846742da7359c5e60"
)
R78_GATE = pathlib.Path(
    "p1553_actual_s6_fermat_tensor_train_probe_gate_r78.md"
)
R78_GATE_SHA256 = (
    "9149c3b903d91a2156125a2c61896dc1111eaeffafdf0536c89a50b7e0e84ff9"
)
R80_REPORT = pathlib.Path(
    "p1553_batched_nested_norm_node_compiler_probe_report_r80.json"
)
R80_REPORT_SHA256 = (
    "936537fb78908dd2916bf6fa5b2091f336b9a47217a1ff787b068ae0491992c5"
)
R80_GATE = pathlib.Path(
    "p1553_batched_nested_norm_node_compiler_probe_gate_r80.md"
)
R80_GATE_SHA256 = (
    "1ff3641688f4f0e13fd64f83aa540ea429a4164e0d0741b64b38acd804d7fb01"
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

Matrix = list[list[int]]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R97_PRODUCER: R97_PRODUCER_SHA256,
        R97_REPORT: R97_REPORT_SHA256,
        R97_GATE: R97_GATE_SHA256,
        R95_REPORT: R95_REPORT_SHA256,
        R95_GATE: R95_GATE_SHA256,
        R78_REPORT: R78_REPORT_SHA256,
        R78_GATE: R78_GATE_SHA256,
        R80_REPORT: R80_REPORT_SHA256,
        R80_GATE: R80_GATE_SHA256,
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
        raise AssertionError(f"R98 source binding mismatch: {failures}")
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


def matrix_rank(matrix: Matrix, prime: int) -> int:
    work = [[value % prime for value in row] for row in matrix]
    rank = 0
    column_count = len(work[0]) if work else 0
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(rank, len(work))
                if work[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], -1, prime)
        work[rank] = [
            value * inverse % prime for value in work[rank]
        ]
        for row in range(len(work)):
            if row == rank or not work[row][column]:
                continue
            multiplier = work[row][column]
            work[row] = [
                (left - multiplier * right) % prime
                for left, right in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == len(work):
            break
    return rank


def zero_projector(value: int, prime: int) -> int:
    return (1 - pow(value % prime, prime - 1, prime)) % prime


def quadratic_resultant(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    prime: int,
) -> int:
    a, b, c = left
    d, e, f = right
    return (
        (a * f - c * d) ** 2
        - (a * e - b * d) * (b * f - c * e)
    ) % prime


def monic_quadratic_resultant(u: int, v: int, prime: int) -> int:
    return quadratic_resultant((1, 0, u), (1, 0, v), prime)


def equality_kernel(labels: list[int], prime: int) -> Matrix:
    return [
        [
            zero_projector(left - right, prime)
            for right in labels
        ]
        for left in labels
    ]


def resultant_projector_kernel(
    labels: list[int],
    prime: int,
) -> Matrix:
    return [
        [
            zero_projector(
                monic_quadratic_resultant(left, right, prime),
                prime,
            )
            for right in labels
        ]
        for left in labels
    ]


def one_hot_factorization(labels: list[int]) -> tuple[Matrix, Matrix]:
    dimension = len(labels)
    left = [
        [int(row == column) for column in range(dimension)]
        for row in range(dimension)
    ]
    right = [row[:] for row in left]
    return left, right


def matrix_mul_transpose(
    left: Matrix,
    right: Matrix,
    prime: int,
) -> Matrix:
    return [
        [
            sum(
                left[row][index] * right[column][index]
                for index in range(len(left[row]))
            )
            % prime
            for column in range(len(right))
        ]
        for row in range(len(left))
    ]


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
def separation_rank_controls() -> dict[str, Any]:
    full_field = []
    for prime in (3, 5, 7, 11, 13):
        labels = list(range(prime))
        equality = equality_kernel(labels, prime)
        resultant = resultant_projector_kernel(labels, prime)
        left, right = one_hot_factorization(labels)
        full_field.append(
            {
                "prime": prime,
                "label_count": len(labels),
                "resultant_projector_equals_equality": (
                    resultant == equality
                ),
                "equality_rank": matrix_rank(equality, prime),
                "resultant_projector_rank": matrix_rank(
                    resultant, prime
                ),
                "one_hot_width": len(labels),
                "one_hot_factorization_exact": (
                    matrix_mul_transpose(left, right, prime)
                    == equality
                ),
            }
        )
    restricted = []
    prime = 101
    for dimension in (4, 8, 16, 32):
        labels = list(range(dimension))
        kernel = resultant_projector_kernel(labels, prime)
        restricted.append(
            {
                "prime": prime,
                "distinct_label_count": dimension,
                "rank": matrix_rank(kernel, prime),
                "full_rank": matrix_rank(kernel, prime) == dimension,
            }
        )
    return {
        "monic_quadratic_subfamily": {
            "left": "A_u(z)=z^2+u",
            "right": "C_v(z)=z^2+v",
            "resultant_identity": "Res(A_u,C_v)=(v-u)^2",
            "projector_identity": (
                "delta_0(Res(A_u,C_v))=delta_0(u-v)"
            ),
        },
        "full_field_sweep": full_field,
        "restricted_distinct_message_sweep": restricted,
        "all_full_field_kernels_rank_p": all(
            row["equality_rank"] == row["prime"]
            and row["resultant_projector_rank"] == row["prime"]
            for row in full_field
        ),
        "all_restricted_kernels_full_rank": all(
            row["full_rank"] for row in restricted
        ),
        "one_bond_theorem": (
            "If K(u,v)=sum_(r=1)^w L_r(u)R_r(v), with arbitrary nonlinear "
            "local encoders L and R, then rank(K)<=w. The equality kernel "
            "is I_p, so exact one-bond width w>=p; on m distinct labels, "
            "w>=m."
        ),
        "scope": (
            "Exact separation-rank theorem for one A/C cut and the full "
            "monic-quadratic coefficient subfamily. It does not cover "
            "multi-edge digit encodings, nonalgebraic lookup, or a "
            "restriction to an unproved smaller image of the actual EC "
            "divisor circuits."
        ),
    }


@functools.lru_cache(maxsize=1)
def occurrence_source_controls() -> dict[str, Any]:
    prime = 101
    occurrences = [2, 2, 5, 7, 9, 12, 20, 31]
    positive_target = 2
    blind_target = 3
    positive_count = sum(
        zero_projector(
            monic_quadratic_resultant(value, positive_target, prime),
            prime,
        )
        for value in occurrences
    )
    blind_count = sum(
        zero_projector(
            monic_quadratic_resultant(value, blind_target, prime),
            prime,
        )
        for value in occurrences
    )
    positive_source = dyadic_source(occurrences, positive_target)
    blind_source = dyadic_source(occurrences, blind_target)
    unique_values = sorted(set(occurrences))
    return {
        "field_prime": prime,
        "occurrence_values": occurrences,
        "positive_target": positive_target,
        "positive_integer_count": positive_count,
        "positive_dyadic_source": positive_source,
        "blind_target": blind_target,
        "blind_integer_count": blind_count,
        "blind_dyadic_source": blind_source,
        "value_collapsed_positive_count": sum(
            value == positive_target for value in unique_values
        ),
        "occurrence_count_exceeds_value_collapsed_count": (
            positive_count
            > sum(value == positive_target for value in unique_values)
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
        "field_scaling": {
            "prime": "p=Theta(B^5)",
            "exponent_B": fraction_record(FIELD_EXPONENT),
        },
        "full_field_one_bond_tensor": {
            "minimum_bond_width": "p",
            "state_and_contraction_exponent_B": fraction_record(
                FIELD_EXPONENT
            ),
            "inside_setup_cap": FIELD_EXPONENT <= SETUP_CAP,
            "inside_online_cap": FIELD_EXPONENT <= ONLINE_CAP,
        },
        "restricted_distinct_source_messages": {
            "message_count": "D=Theta(B^(12/5))",
            "minimum_bond_width_if_messages_are_distinct": "D",
            "state_and_contraction_exponent_B": fraction_record(
                SOURCE_EXPONENT
            ),
            "inside_setup_cap": SOURCE_EXPONENT <= SETUP_CAP,
            "inside_online_cap": SOURCE_EXPONENT <= ONLINE_CAP,
            "actual_distinct_message_reachability_theorem_supplied": False,
        },
        "multi_edge_digitized_algebraic_encoding_supplied": False,
        "compact_digit_extractor_supplied": False,
        "scope": (
            "Charges one-bond separation width and a restricted distinct-"
            "message channel. It is not a lower bound on a multi-edge "
            "digitized encoding with a proved compact algebraic extractor."
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    ranks = separation_rank_controls()
    sources = occurrence_source_controls()
    costs = asymptotic_cost_control()
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_nonlinear_tensor_tower_trace.r98.v1"
        ),
        "left_subfamily": "A_u(z)=z^2+u",
        "right_subfamily": "C_v(z)=z^2+v",
        "projector_kernel": "K(u,v)=delta_0(Res(A_u,C_v))",
        "tensor_grammar": (
            "K(u,v)=sum_(r=1)^w L_r(u)R_r(v), arbitrary nonlinear "
            "local encoders, one A/C bond"
        ),
        "caps": costs["caps"],
        "excluded_unfrozen_grammar": (
            "multi-edge digitized encodings and compact algebraic digit "
            "extractors"
        ),
    }
    state_ledger = {
        "schema": (
            "p1553.nonlinear_tensor_state_transition_ledger.r98.v1"
        ),
        "separation_rank_controls": ranks,
        "asymptotic_cost": costs,
        "one_bond_tensor_inside_caps": False,
    }
    source_replay = {
        "schema": (
            "p1553.tensor_tower_integer_source_replay.r98.v1"
        ),
        "occurrence_controls": sources,
        "actual_five_a_five_c_integer_lift_complete": False,
        "actual_five_a_five_c_source_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.tensor_tower_exceptional_controls.r98.v1"
        ),
        "blind_bottom_exact": (
            sources["blind_integer_count"] == 0
            and sources["blind_dyadic_source"]["returned_bottom"]
        ),
        "duplicate_occurrence_count_exact": (
            sources["positive_integer_count"] == 2
        ),
        "value_collapse_loses_occurrence": sources[
            "occurrence_count_exceeds_value_collapsed_count"
        ],
        "projective_infinity_complete": False,
        "proper_subsum_complete": False,
        "tangent_source_complete": False,
        "multiplicity_complete_actual_source_return": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r98.v1",
        "one_bond_tensor_inside_caps": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "eleven_source_bindings_verified": len(bindings) == 11,
        "monic_quadratic_resultant_identity_exact": all(
            row["resultant_projector_equals_equality"]
            for row in ranks["full_field_sweep"]
        ),
        "fermat_projector_equality_kernel_exact": all(
            row["resultant_projector_equals_equality"]
            for row in ranks["full_field_sweep"]
        ),
        "full_field_equality_rank_p": ranks[
            "all_full_field_kernels_rank_p"
        ],
        "one_bond_width_theorem_recorded": "rank(K)<=w"
        in ranks["one_bond_theorem"],
        "width_p_factorization_replayed": all(
            row["one_hot_factorization_exact"]
            for row in ranks["full_field_sweep"]
        ),
        "restricted_distinct_message_rank_exact": ranks[
            "all_restricted_kernels_full_rank"
        ],
        "duplicate_occurrence_count_exact": exceptional[
            "duplicate_occurrence_count_exact"
        ],
        "blind_bottom_exact": exceptional["blind_bottom_exact"],
        "toy_dyadic_source_exact": (
            sources["positive_dyadic_source"]["source_index"] == 0
            and sources["positive_dyadic_source"][
                "returned_source_matches_target"
            ]
        ),
        "value_collapse_occurrence_loss_replayed": exceptional[
            "value_collapse_loses_occurrence"
        ],
        "integer_no_wrap_replayed": sources[
            "integer_counts_below_prime"
        ],
        "full_field_width_charged_as_B5": (
            costs["full_field_one_bond_tensor"][
                "state_and_contraction_exponent_B"
            ]["exact"]
            == "5"
        ),
        "restricted_width_charged_as_B12O5": (
            costs["restricted_distinct_source_messages"][
                "state_and_contraction_exponent_B"
            ]["exact"]
            == "12/5"
        ),
        "one_bond_tensor_inside_setup_cap": False,
        "one_bond_tensor_inside_online_cap": False,
        "actual_distinct_message_reachability_theorem": False,
        "multi_edge_digitized_encoding_inside_caps": False,
        "compact_algebraic_digit_extractor_supplied": False,
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
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "MONIC_QUADRATIC_RESULTANT_PROJECTOR_IS_EQUALITY_KERNEL__"
            "ONE_BOND_NONLINEAR_ENCODER_WIDTH_P__RESTRICTED_DISTINCT_"
            "WIDTH_D__DIGITIZED_MULTI_EDGE_OPEN"
        ),
        "source_bindings": {
            "r97_producer": {
                "path": str(R97_PRODUCER),
                "sha256": R97_PRODUCER_SHA256,
            },
            "r97_report": {
                "path": str(R97_REPORT),
                "sha256": R97_REPORT_SHA256,
            },
            "r97_gate": {
                "path": str(R97_GATE),
                "sha256": R97_GATE_SHA256,
            },
            "r95_report": {
                "path": str(R95_REPORT),
                "sha256": R95_REPORT_SHA256,
            },
            "r95_gate": {
                "path": str(R95_GATE),
                "sha256": R95_GATE_SHA256,
            },
            "r78_report": {
                "path": str(R78_REPORT),
                "sha256": R78_REPORT_SHA256,
            },
            "r78_gate": {
                "path": str(R78_GATE),
                "sha256": R78_GATE_SHA256,
            },
            "r80_report": {
                "path": str(R80_REPORT),
                "sha256": R80_REPORT_SHA256,
            },
            "r80_gate": {
                "path": str(R80_GATE),
                "sha256": R80_GATE_SHA256,
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
            "R98 is the first campaign receipt to prove an exact one-bond "
            "separation-rank theorem for the resultant projector while "
            "allowing arbitrary nonlinear encoders on both compact sides."
        ),
        "separation_rank_controls": ranks,
        "occurrence_source_controls": sources,
        "asymptotic_cost_control": costs,
        "side_artifacts": {
            "frozen": "frozen_5a5c_nonlinear_tensor_tower_trace.json",
            "state_ledger": (
                "nonlinear_tensor_state_and_transition_ledger.json"
            ),
            "source_replay": "tensor_tower_integer_source_replay.json",
            "exceptional": "tensor_tower_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r98.json",
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
            "This closes one-bond A/C tensor contractions over the full "
            "monic-quadratic coefficient subfamily and restricted distinct "
            "message sets. It is not a lower bound on multi-edge digitized "
            "encodings, nonalgebraic lookup, or an actual EC divisor image "
            "proved to omit this subfamily."
        ),
        "next_action": (
            "Construct or refute one multi-edge digitized algebraic equality "
            "projector over F_p. Freeze the digit or channel extractor from "
            "the compact A/C divisor circuits before outcomes; require "
            "total cut capacity, extractor state, and setup below B^(9/4), "
            "fresh evaluation and source return below B^(5/4), no p-size "
            "lookup/interpolation table, exact occurrence multiplicity and "
            "all exceptional branches, known-RHS rank, factor logs, and "
            "identical target descent."
        ),
        "disposition": (
            "REJECT_ONE_BOND_NONLINEAR_TENSOR_TOWER_ONLY__MONIC_"
            "QUADRATIC_RESULTANT_PROJECTOR_IS_EQUALITY__FULL_FIELD_"
            "SEPARATION_RANK_P_B5__RESTRICTED_DISTINCT_MESSAGES_RANK_D_"
            "B12O5__DUPLICATE_COUNT_AND_TOY_SOURCE_EXACT__VALUE_COLLAPSE_"
            "LOSES_OCCURRENCE__MULTI_EDGE_DIGITIZED_ENCODING_OPEN__"
            "ACTUAL_EC_IMAGE_AND_PROJECTIVE_BRANCHES_INCOMPLETE__NO_RANK__"
            "NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
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
            "p1553_5a5c_nonlinear_tensor_tower_"
            "trace_probe_report_r98.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_nonlinear_tensor_tower_trace.json"
        ),
    )
    parser.add_argument(
        "--state-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "nonlinear_tensor_state_and_transition_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "tensor_tower_integer_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "tensor_tower_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r98.json"
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
