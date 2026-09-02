#!/usr/bin/env python3
"""Audit pointwise transposed projector traces after R96."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.5a5c_factored_transposed_projector_trace.r97.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
SOURCE_EXPONENT = Fraction(12, 5)

R96_PRODUCER = pathlib.Path(
    "p1553_5a5c_modular_frobenius_trace_recurrence_probe_r96.py"
)
R96_PRODUCER_SHA256 = (
    "92f646378156476cf897729b6bca6d67bc47398663137aee1ab17d910511f960"
)
R96_REPORT = pathlib.Path(
    "p1553_5a5c_modular_frobenius_trace_recurrence_"
    "probe_report_r96.json"
)
R96_REPORT_SHA256 = (
    "e4e667703d22b96c5ef64b3bfbbd24119f128cd00f68d0d7d850ba41c0c0e163"
)
R96_GATE = pathlib.Path(
    "p1553_5a5c_modular_frobenius_trace_recurrence_probe_gate_r96.md"
)
R96_GATE_SHA256 = (
    "55655280c02bbef1e5f0f29bcad1beb1b21d6809e79b7de400a58997b0caa64d"
)
R95_REPORT = pathlib.Path(
    "p1553_5a5c_aggregate_veronese_projector_"
    "recurrence_probe_report_r95.json"
)
R95_REPORT_SHA256 = (
    "50441deecd5fadcb05d30c09532d2b5fe6e8893ea2530243c20a52761e0a1742"
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
R88_REPORT = pathlib.Path(
    "p1553_5a5c_black_box_translated_resultant_localizer_"
    "probe_report_r88.json"
)
R88_REPORT_SHA256 = (
    "d73e017c731a54c6913aeaa94e6b5c6d54ca3757f56be14e8ca8ee5524031de1"
)
R88_GATE = pathlib.Path(
    "p1553_5a5c_black_box_translated_resultant_localizer_probe_gate_r88.md"
)
R88_GATE_SHA256 = (
    "90f0980bdeb51f540cd233f207218361cc14beb8a899c246418410d36ef43d56"
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
        R96_PRODUCER: R96_PRODUCER_SHA256,
        R96_REPORT: R96_REPORT_SHA256,
        R96_GATE: R96_GATE_SHA256,
        R95_REPORT: R95_REPORT_SHA256,
        R78_REPORT: R78_REPORT_SHA256,
        R78_GATE: R78_GATE_SHA256,
        R88_REPORT: R88_REPORT_SHA256,
        R88_GATE: R88_GATE_SHA256,
        P1513_HANDOFF: P1513_HANDOFF_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R97 source binding mismatch: {failures}")
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


def zero_projector_derivative(value: int, prime: int) -> int:
    # d(1-h^(p-1))/dh = h^(p-2) in characteristic p.
    return pow(value % prime, prime - 2, prime)


def projector_count(values: list[int], prime: int) -> int:
    return sum(zero_projector(value, prime) for value in values)


def dyadic_intervals(dimension: int) -> list[tuple[int, int]]:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    intervals: list[tuple[int, int]] = []

    def visit(lower: int, upper: int) -> None:
        intervals.append((lower, upper))
        if upper - lower == 1:
            return
        middle = (lower + upper) // 2
        visit(lower, middle)
        visit(middle, upper)

    visit(0, dimension)
    return intervals


def dyadic_adjoint_matrix(values: list[int], prime: int) -> Matrix:
    derivatives = [
        zero_projector_derivative(value, prime) for value in values
    ]
    return [
        [
            derivatives[index] if lower <= index < upper else 0
            for index in range(len(values))
        ]
        for lower, upper in dyadic_intervals(len(values))
    ]


def dyadic_zero_source(
    values: list[int],
    prime: int,
) -> dict[str, Any]:
    lower = 0
    upper = len(values)
    transcript = []
    queried_values = 0
    while True:
        selected = values[lower:upper]
        count = projector_count(selected, prime)
        queried_values += len(selected)
        transcript.append(
            {"range": [lower, upper], "zero_count": count}
        )
        if count == 0:
            return {
                "source_index": None,
                "returned_bottom": True,
                "queried_value_occurrences": queried_values,
                "transcript": transcript,
            }
        if upper - lower == 1:
            return {
                "source_index": lower,
                "returned_bottom": False,
                "returned_source_is_zero": values[lower] % prime == 0,
                "queried_value_occurrences": queried_values,
                "transcript": transcript,
            }
        middle = (lower + upper) // 2
        left_count = projector_count(values[lower:middle], prime)
        queried_values += middle - lower
        transcript.append(
            {"range": [lower, middle], "zero_count": left_count}
        )
        if left_count:
            upper = middle
        else:
            lower = middle


def product_gradient(values: list[int], prime: int) -> list[int]:
    return [
        (
            1
            if len(values) == 1
            else functools.reduce(
                lambda left, right: left * right % prime,
                values[:index] + values[index + 1 :],
                1,
            )
        )
        for index in range(len(values))
    ]


def adjoint_rank_controls() -> dict[str, Any]:
    prime = 101
    sweep = []
    for dimension in (4, 8, 16, 32):
        values = list(range(1, dimension + 1))
        derivatives = [
            zero_projector_derivative(value, prime)
            for value in values
        ]
        jacobian = [
            [
                derivatives[row] if row == column else 0
                for column in range(dimension)
            ]
            for row in range(dimension)
        ]
        dyadic = dyadic_adjoint_matrix(values, prime)
        sweep.append(
            {
                "dimension": dimension,
                "dyadic_mask_count": len(dyadic),
                "expected_full_binary_mask_count": 2 * dimension - 1,
                "jacobian_rank": matrix_rank(jacobian, prime),
                "dyadic_adjoint_rank": matrix_rank(dyadic, prime),
                "all_projector_derivatives_nonzero": all(derivatives),
            }
        )
    return {
        "field_prime": prime,
        "projector_identity": "delta_0(h)=1-h^(p-1)",
        "projector_derivative_identity": (
            "delta_0'(h)=h^(p-2) in F_p"
        ),
        "blind_nonzero_sweep": sweep,
        "all_jacobians_full_rank": all(
            row["jacobian_rank"] == row["dimension"] for row in sweep
        ),
        "all_dyadic_adjoint_families_full_rank": all(
            row["dyadic_adjoint_rank"] == row["dimension"]
            for row in sweep
        ),
        "scope": (
            "Finite exact rank for the pointwise Jacobian and the complete "
            "dyadic mask family; not an asymptotic lower bound on arbitrary "
            "nonlinear compact trace circuits."
        ),
    }


def source_and_gradient_controls() -> dict[str, Any]:
    prime = 101
    unique_zero = [3, 5, 0, 7, 11, 13, 17, 19]
    two_zeros = [3, 0, 5, 7, 0, 11, 13, 17]
    blind = [3, 5, 7, 11, 13, 17, 19, 23]
    unique_projector_gradient = [
        zero_projector_derivative(value, prime)
        for value in unique_zero
    ]
    unique_product_gradient = product_gradient(unique_zero, prime)
    two_product_gradient = product_gradient(two_zeros, prime)
    source = dyadic_zero_source(unique_zero, prime)
    bottom = dyadic_zero_source(blind, prime)
    return {
        "field_prime": prime,
        "unique_zero": {
            "values": unique_zero,
            "integer_projector_count": projector_count(
                unique_zero, prime
            ),
            "projector_gradient": unique_projector_gradient,
            "zero_coordinate_has_nonzero_projector_gradient": (
                unique_projector_gradient[2] != 0
            ),
            "projector_gradient_materialized_words": len(unique_zero),
            "product_gradient": unique_product_gradient,
            "product_gradient_nonzero_indices": [
                index
                for index, value in enumerate(unique_product_gradient)
                if value
            ],
            "product_gradient_localizes_unique_zero": (
                [
                    index
                    for index, value in enumerate(unique_product_gradient)
                    if value
                ]
                == [2]
            ),
            "dyadic_source": source,
        },
        "two_zeros": {
            "values": two_zeros,
            "integer_projector_count": projector_count(
                two_zeros, prime
            ),
            "product_gradient": two_product_gradient,
            "product_gradient_is_zero_vector": not any(
                two_product_gradient
            ),
        },
        "blind_nonzero": {
            "values": blind,
            "integer_projector_count": projector_count(blind, prime),
            "dyadic_source": bottom,
        },
        "multiplicity_boundary": (
            "The projector sum counts two zero occurrences, while the first "
            "derivative of the scalar product is identically zero."
        ),
    }


def asymptotic_cost_control() -> dict[str, Any]:
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "source_side": {
            "value_count_exponent_B": fraction_record(SOURCE_EXPONENT),
            "source_count": "D=Theta(B^(12/5))",
        },
        "pointwise_fermat_projector": {
            "work": "Theta(D log p)=B^(12/5) polylog(B)",
            "state_or_streamed_source_values": "Theta(D)",
            "work_exponent_B": fraction_record(SOURCE_EXPONENT),
            "inside_setup_cap": SOURCE_EXPONENT <= SETUP_CAP,
            "inside_online_cap": SOURCE_EXPONENT <= ONLINE_CAP,
        },
        "full_transposed_adjoint": {
            "adjoint_words": "Theta(D)",
            "state_exponent_B": fraction_record(SOURCE_EXPONENT),
            "inside_setup_cap": SOURCE_EXPONENT <= SETUP_CAP,
        },
        "balanced_product_tree": {
            "leaf_values": "D",
            "forward_multiplications": "D-1",
            "full_forward_nodes": "2D-1",
            "reverse_adjoint_words": "Theta(D)",
            "work_and_state_exponent_B": fraction_record(
                SOURCE_EXPONENT
            ),
            "inside_setup_cap": SOURCE_EXPONENT <= SETUP_CAP,
            "inside_online_cap": SOURCE_EXPONENT <= ONLINE_CAP,
        },
        "dyadic_range_masks": {
            "mask_count": "2D-1 for power-of-two D",
            "linearized_span_rank": "D",
            "state_exponent_B": fraction_record(SOURCE_EXPONENT),
        },
        "compact_nonlinear_tensor_tower_trace_supplied": False,
        "scope": (
            "Charges pointwise Fermat evaluation, source-valued product "
            "trees, their reverse adjoints, and the full dyadic linearized "
            "mask family. It does not lower-bound a nonlinear tensor-tower "
            "trace whose node states are constructed directly from compact "
            "A/C divisor circuits."
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    ranks = adjoint_rank_controls()
    controls = source_and_gradient_controls()
    costs = asymptotic_cost_control()
    unique = controls["unique_zero"]
    duplicate = controls["two_zeros"]
    blind = controls["blind_nonzero"]
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_factored_transposed_projector_trace."
            "r97.v1"
        ),
        "projector": "delta_0(h)=1-h^(p-1)",
        "pointwise_adjoint": "delta_0'(h)=h^(p-2)",
        "range_family": "complete binary dyadic interval masks",
        "product_control": "P(h)=product_i h_i",
        "caps": costs["caps"],
        "excluded_unfrozen_grammar": (
            "nonlinear tensor-tower traces constructed directly from "
            "compact A/C divisor circuits"
        ),
    }
    state_ledger = {
        "schema": (
            "p1553.factored_trace_state_and_adjoint_ledger.r97.v1"
        ),
        "adjoint_rank_controls": ranks,
        "source_and_gradient_controls": controls,
        "asymptotic_cost": costs,
        "pointwise_transposed_trace_inside_caps": False,
    }
    source_replay = {
        "schema": (
            "p1553.range_idempotent_integer_source_replay.r97.v1"
        ),
        "unique_zero_integer_count": unique[
            "integer_projector_count"
        ],
        "unique_zero_dyadic_source": unique["dyadic_source"],
        "blind_bottom": blind["dyadic_source"],
        "duplicate_integer_count": duplicate[
            "integer_projector_count"
        ],
        "source_value_occurrences_charged": True,
        "actual_five_a_five_c_integer_lift_complete": False,
        "actual_five_a_five_c_source_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.transposed_trace_exceptional_controls.r97.v1"
        ),
        "blind_nonzero_bottom_exact": (
            blind["integer_projector_count"] == 0
            and blind["dyadic_source"]["returned_bottom"]
        ),
        "duplicate_projector_count_exact": (
            duplicate["integer_projector_count"] == 2
        ),
        "duplicate_product_gradient_collapse_replayed": duplicate[
            "product_gradient_is_zero_vector"
        ],
        "projective_infinity_complete": False,
        "proper_subsum_complete": False,
        "tangent_source_complete": False,
        "multiplicity_complete_source_return": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r97.v1",
        "pointwise_transposed_trace_inside_caps": False,
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
        "finite_projector_identity_exact": (
            unique["integer_projector_count"] == 1
            and duplicate["integer_projector_count"] == 2
        ),
        "toy_integer_counts_below_prime": 2 < controls["field_prime"],
        "unique_zero_dyadic_source_exact": (
            unique["dyadic_source"]["source_index"] == 2
            and unique["dyadic_source"]["returned_source_is_zero"]
        ),
        "duplicate_projector_count_exact": exceptional[
            "duplicate_projector_count_exact"
        ],
        "blind_nonzero_bottom_exact": exceptional[
            "blind_nonzero_bottom_exact"
        ],
        "projector_derivative_identity_replayed": all(
            row["all_projector_derivatives_nonzero"]
            for row in ranks["blind_nonzero_sweep"]
        ),
        "blind_projector_jacobians_full_rank": ranks[
            "all_jacobians_full_rank"
        ],
        "dyadic_adjoint_families_full_rank": ranks[
            "all_dyadic_adjoint_families_full_rank"
        ],
        "full_gradient_state_charged": (
            unique["projector_gradient_materialized_words"]
            == len(unique["values"])
        ),
        "unique_product_gradient_localizer_replayed": unique[
            "product_gradient_localizes_unique_zero"
        ],
        "two_zero_product_gradient_collapse_replayed": duplicate[
            "product_gradient_is_zero_vector"
        ],
        "standard_source_exponent_charged_as_B12O5": (
            costs["full_transposed_adjoint"]["state_exponent_B"][
                "exact"
            ]
            == "12/5"
        ),
        "pointwise_trace_inside_setup_cap": False,
        "pointwise_trace_inside_online_cap": False,
        "compact_range_idempotent_inside_caps": False,
        "actual_five_a_five_c_integer_lift_complete": False,
        "actual_five_a_five_c_source_unranking_complete": False,
        "projective_infinity_source_complete": False,
        "proper_subsum_source_complete": False,
        "tangent_source_complete": False,
        "multiplicity_complete_source_return": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
        "compact_nonlinear_tensor_tower_trace_supplied": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "POINTWISE_PROJECTOR_JACOBIAN_AND_DYADIC_ADJOINT_FULL_RANK__"
            "UNIQUE_PRODUCT_GRADIENT_LOCALIZER__MULTIZERO_GRADIENT_"
            "COLLAPSE__STANDARD_TRANSPOSE_B12O5"
        ),
        "source_bindings": {
            "r96_producer": {
                "path": str(R96_PRODUCER),
                "sha256": R96_PRODUCER_SHA256,
            },
            "r96_report": {
                "path": str(R96_REPORT),
                "sha256": R96_REPORT_SHA256,
            },
            "r96_gate": {
                "path": str(R96_GATE),
                "sha256": R96_GATE_SHA256,
            },
            "r95_report": {
                "path": str(R95_REPORT),
                "sha256": R95_REPORT_SHA256,
            },
            "r78_report": {
                "path": str(R78_REPORT),
                "sha256": R78_REPORT_SHA256,
            },
            "r78_gate": {
                "path": str(R78_GATE),
                "sha256": R78_GATE_SHA256,
            },
            "r88_report": {
                "path": str(R88_REPORT),
                "sha256": R88_REPORT_SHA256,
            },
            "r88_gate": {
                "path": str(R88_GATE),
                "sha256": R88_GATE_SHA256,
            },
            "p1513_handoff": {
                "path": str(P1513_HANDOFF),
                "sha256": P1513_HANDOFF_SHA256,
            },
        },
        "novelty_scope": (
            "R97 is the first campaign receipt to rank the complete "
            "pointwise projector Jacobian and dyadic adjoint family and to "
            "separate unique-zero product-gradient localization from its "
            "multi-zero collapse."
        ),
        "adjoint_rank_controls": ranks,
        "source_and_gradient_controls": controls,
        "asymptotic_cost_control": costs,
        "side_artifacts": {
            "frozen": (
                "frozen_5a5c_factored_transposed_projector_trace.json"
            ),
            "state_ledger": (
                "factored_trace_state_and_adjoint_ledger.json"
            ),
            "source_replay": (
                "range_idempotent_integer_source_replay.json"
            ),
            "exceptional": (
                "transposed_trace_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r97.json",
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
            "This closes pointwise Fermat powering, source-valued product "
            "trees, their reverse adjoints, and linearized dyadic masks. "
            "The finite full-rank result is not an asymptotic lower bound "
            "on a nonlinear tensor-tower trace built directly from compact "
            "A/C divisor circuits."
        ),
        "next_action": (
            "Construct or refute one nonlinear tensor-tower projector trace "
            "whose node states are derived directly from compact A/C "
            "divisor circuits rather than source leaves, quotient bases, "
            "moment vectors, or linearized dyadic masks. Freeze its tensor "
            "algebra and target transition before outcomes; require exact "
            "integer count, one complete coupled source, all exceptional "
            "branches, B^(9/4) setup/state, B^(5/4) fresh work/workspace, "
            "known-RHS rank, factor logs, and identical target descent."
        ),
        "disposition": (
            "REJECT_STANDARD_POINTWISE_TRANSPOSE_ONLY__PROJECTOR_COUNTS_"
            "AND_TOY_DYADIC_SOURCE_EXACT__BLIND_JACOBIAN_AND_DYADIC_"
            "ADJOINT_FULL_RANK__UNIQUE_PRODUCT_GRADIENT_LOCALIZES__"
            "MULTIZERO_PRODUCT_GRADIENT_COLLAPSES__SOURCE_VALUES_AND_"
            "ADJOINTS_B12O5__NONLINEAR_TENSOR_TOWER_OPEN__PROJECTIVE_"
            "AND_FULL_5A5C_SOURCE_INCOMPLETE__NO_RANK__NO_FACTOR_LOGS__"
            "NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
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
            "p1553_5a5c_factored_transposed_projector_"
            "trace_probe_report_r97.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_factored_transposed_projector_trace.json"
        ),
    )
    parser.add_argument(
        "--state-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factored_trace_state_and_adjoint_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "range_idempotent_integer_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "transposed_trace_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r97.json"
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
