#!/usr/bin/env python3
"""Audit an implicit Veronese hyperplane source index for the R84 split."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import itertools
import json
import pathlib
from collections import Counter
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.5a5c_implicit_veronese_hyperplane_index.r94.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
LEFT_ROOT_EXPONENT = Fraction(12, 5)
RIGHT_ROOT_EXPONENT = Fraction(13, 5)

R93_PRODUCER = pathlib.Path(
    "p1553_5a5c_shared_semilinear_incidence_"
    "correspondence_probe_r93.py"
)
R93_PRODUCER_SHA256 = (
    "9a82761a02cdb43fb0def4e959074eaa4e8e152dcea52901eec47bb1ab55c162"
)
R93_REPORT = pathlib.Path(
    "p1553_5a5c_shared_semilinear_incidence_"
    "correspondence_probe_report_r93.json"
)
R93_REPORT_SHA256 = (
    "f33c764ebbf491e55d7b8342fda9df8b29038beef8cdb06662c343ac415b7d4b"
)
R93_GATE = pathlib.Path(
    "p1553_5a5c_shared_semilinear_incidence_"
    "correspondence_probe_gate_r93.md"
)
R93_GATE_SHA256 = (
    "15d58598c875b8ffe89fb1c7d551cd2adfc15cc8f7215a5f3242987305a2b2ad"
)
R9_GATE = pathlib.Path("p1553_projector_trace_router_gate_r9.md")
R9_GATE_SHA256 = (
    "400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81"
)
R78_REPORT = pathlib.Path(
    "p1553_actual_s6_fermat_tensor_train_probe_report_r78.json"
)
R78_REPORT_SHA256 = (
    "e590002c433c6504725d5ab7ff1dba97ad8c15400bf0117846742da7359c5e60"
)
R84_REPORT = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_"
    "probe_report_r84.json"
)
R84_REPORT_SHA256 = (
    "c9b1c5fb0f58f2c5118562623fd5dfff5d55d7238b513892d4178a67af5ccf0b"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
P1515_TRICHOTOMY = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-098/recursive_s3_local_separator_trichotomy_v1.md"
)
P1515_TRICHOTOMY_SHA256 = (
    "dec667b097bcaefdf4c54091b2a9fa7757db5a65efe5b36e6ac15a6ff11a435a"
)
MULTIPOINT_PAPER = pathlib.Path(
    "references/"
    "bhargava_ghosh_guo_kumar_umans_multipoint_2205.00342v1.pdf"
)
MULTIPOINT_PAPER_SHA256 = (
    "14eddc304a7dd8995ebc1e24171571fd9dc0f1f837ca35a7f9e2e6fb21bfafa8"
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R93_PRODUCER: R93_PRODUCER_SHA256,
        R93_REPORT: R93_REPORT_SHA256,
        R93_GATE: R93_GATE_SHA256,
        R9_GATE: R9_GATE_SHA256,
        R78_REPORT: R78_REPORT_SHA256,
        R84_REPORT: R84_REPORT_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        P1515_TRICHOTOMY: P1515_TRICHOTOMY_SHA256,
        MULTIPOINT_PAPER: MULTIPOINT_PAPER_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R94 source binding mismatch: {failures}")
    return actual


@functools.lru_cache(maxsize=1)
def load_r93() -> Any:
    spec = importlib.util.spec_from_file_location("p1553_r93_bound", R93_PRODUCER)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load bound R93 producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def fermat_zero_indicator(value: int) -> int:
    r93 = load_r93()
    return (
        1 - pow(value % r93.FIELD_PRIME, r93.FIELD_PRIME - 1, r93.FIELD_PRIME)
    ) % r93.FIELD_PRIME


def direct_range_count(
    values: list[int],
    lower: int,
    upper: int,
) -> tuple[int, int]:
    indicators = [
        fermat_zero_indicator(value)
        for value in values[lower:upper]
    ]
    return sum(indicators), upper - lower


def dyadic_source_index(values: list[int]) -> dict[str, Any]:
    total_count, evaluations = direct_range_count(values, 0, len(values))
    if total_count == 0:
        return {
            "source_index": None,
            "exact_count": 0,
            "field_evaluation_count": evaluations,
            "query_count": 1,
        }
    lower = 0
    upper = len(values)
    query_count = 1
    while upper - lower > 1:
        midpoint = (lower + upper) // 2
        left_count, left_evaluations = direct_range_count(
            values,
            lower,
            midpoint,
        )
        evaluations += left_evaluations
        query_count += 1
        if left_count:
            upper = midpoint
        else:
            lower = midpoint
    return {
        "source_index": lower,
        "exact_count": total_count,
        "field_evaluation_count": evaluations,
        "query_count": query_count,
    }


def chart_index_control(deck_size: int) -> dict[str, Any]:
    r93 = load_r93()
    points = r93.canonical_affine_points()[:deck_size]
    pair_indices = list(
        itertools.combinations_with_replacement(range(deck_size), 2)
    )
    quadratics = [
        r93.s3_quadratic(points[left][0], points[right][0])
        for left, right in pair_indices
    ]
    projector_rows = []
    recovered_rows = []
    for left_index, left_quadratic in enumerate(quadratics):
        values = [
            r93.bilinear_resultant(left_quadratic, right_quadratic)
            for right_quadratic in quadratics
        ]
        projector = [
            fermat_zero_indicator(value) for value in values
        ]
        route = dyadic_source_index(values)
        source_index = route["source_index"]
        signed_source = None
        if source_index is not None:
            left_pair = pair_indices[left_index]
            right_pair = pair_indices[source_index]
            source_points = (
                points[left_pair[0]],
                points[left_pair[1]],
                points[right_pair[0]],
                points[right_pair[1]],
            )
            signed_source = r93.signed_relation(source_points)
        projector_rows.append(projector)
        recovered_rows.append(
            {
                "left_chart_index": left_index,
                "exact_zero_count": sum(projector),
                "dyadic_count_matches": (
                    route["exact_count"] == sum(projector)
                ),
                "returned_index_is_zero": (
                    source_index is not None
                    and projector[source_index] == 1
                ),
                "signed_source_returned": signed_source is not None,
                "field_evaluation_count": route[
                    "field_evaluation_count"
                ],
                "query_count": route["query_count"],
            }
        )
    pair_count = len(pair_indices)
    return {
        "deck_size": deck_size,
        "pair_chart_count": pair_count,
        "fermat_projector_degree": r93.FIELD_PRIME - 1,
        "every_projector_entry_boolean": all(
            value in (0, 1)
            for row in projector_rows
            for value in row
        ),
        "projector_equals_resultant_zero_predicate": all(
            projector_rows[left][right]
            == int(
                r93.bilinear_resultant(
                    quadratics[left],
                    quadratics[right],
                )
                == 0
            )
            for left in range(pair_count)
            for right in range(pair_count)
        ),
        "projector_matrix_rank": r93.matrix_rank_mod(projector_rows),
        "all_dyadic_counts_exact": all(
            row["dyadic_count_matches"] for row in recovered_rows
        ),
        "all_nonempty_rows_return_zero": all(
            row["returned_index_is_zero"]
            for row in recovered_rows
            if row["exact_zero_count"] > 0
        ),
        "all_returned_zeroes_have_signed_source": all(
            row["signed_source_returned"]
            for row in recovered_rows
            if row["exact_zero_count"] > 0
        ),
        "maximum_direct_dyadic_field_evaluations": max(
            row["field_evaluation_count"] for row in recovered_rows
        ),
        "less_than_twice_one_side_scan": max(
            row["field_evaluation_count"] for row in recovered_rows
        )
        < 2 * pair_count,
        "maximum_dyadic_query_count": max(
            row["query_count"] for row in recovered_rows
        ),
        "integer_count_no_mod_p_wrap": max(
            row["exact_zero_count"] for row in recovered_rows
        )
        < r93.FIELD_PRIME,
        "zero_count_histogram": {
            str(count): multiplicity
            for count, multiplicity in sorted(
                Counter(
                    row["exact_zero_count"] for row in recovered_rows
                ).items()
            )
        },
    }


def implicit_index_semantics_control() -> dict[str, Any]:
    rows = [
        chart_index_control(deck_size)
        for deck_size in (4, 6, 8, 10, 12)
    ]
    return {
        "identity": "delta_0(H)=1-H^(p-1) over F_p",
        "chart_rows": rows,
        "all_projector_predicates_exact": all(
            row["projector_equals_resultant_zero_predicate"]
            for row in rows
        ),
        "all_dyadic_source_routes_exact": all(
            row["all_dyadic_counts_exact"]
            and row["all_nonempty_rows_return_zero"]
            and row["all_returned_zeroes_have_signed_source"]
            for row in rows
        ),
        "largest_projector_rank": rows[-1]["projector_matrix_rank"],
        "largest_projector_full_rank": (
            rows[-1]["projector_matrix_rank"]
            == rows[-1]["pair_chart_count"]
        ),
        "semantic_equivalence": (
            "an exact hyperplane source query is the restricted Fermat "
            "projector existence/source router from the bound R9 gate"
        ),
    }


def minimal_projector_degree_control() -> dict[str, Any]:
    r93 = load_r93()
    degree = r93.FIELD_PRIME - 1
    values_exact = all(
        fermat_zero_indicator(value) == int(value == 0)
        for value in range(r93.FIELD_PRIME)
    )
    return {
        "field_prime": r93.FIELD_PRIME,
        "projector": "1-X^(p-1)",
        "degree": degree,
        "all_field_values_exact": values_exact,
        "minimal_degree_exact": True,
        "proof": (
            "a polynomial of degree below p-1 that is zero on all p-1 "
            "nonzero field elements must be the zero polynomial, "
            "contradicting value one at zero"
        ),
        "binary_powering_multiplication_count_upper_bound": (
            2 * degree.bit_length()
        ),
        "low_arithmetic_depth_implies_low_separated_rank": False,
    }


def multipoint_literature_control() -> dict[str, Any]:
    return {
        "paper": (
            "Bhargava-Ghosh-Guo-Kumar-Umans, Fast Multivariate "
            "Multipoint Evaluation Over All Finite Fields"
        ),
        "arxiv": "2205.00342v1",
        "bound": (
            "(d^m+N)^(1+o(1))*poly(m,d,log|F|) for N output points"
        ),
        "source_sha256": MULTIPOINT_PAPER_SHA256,
        "root_side_evaluation_count_exponent_B": fraction_record(
            LEFT_ROOT_EXPONENT
        ),
        "certified_output_all_values_cost_exponent_B": fraction_record(
            LEFT_ROOT_EXPONENT
        ),
        "inside_online_cap": LEFT_ROOT_EXPONENT <= ONLINE_CAP,
        "source_reporting_without_all_outputs_supplied": False,
        "scope": (
            "a matched upper-bound control; it does not lower-bound an "
            "output-sensitive zero/source algorithm"
        ),
    }


def asymptotic_cost_control() -> dict[str, Any]:
    literature = multipoint_literature_control()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "r84_root_source_sides": {
            "smaller_exponent_B": fraction_record(LEFT_ROOT_EXPONENT),
            "larger_exponent_B": fraction_record(RIGHT_ROOT_EXPONENT),
        },
        "materialized_smaller_feature_index": {
            "state_exponent_B": fraction_record(LEFT_ROOT_EXPONENT),
            "inside_setup_cap": LEFT_ROOT_EXPONENT <= SETUP_CAP,
        },
        "direct_implicit_projector_scan": {
            "fresh_work_exponent_B": fraction_record(
                LEFT_ROOT_EXPONENT
            ),
            "inside_online_cap": LEFT_ROOT_EXPONENT <= ONLINE_CAP,
            "dyadic_replay_changes_exponent": False,
        },
        "all-output_multipoint_evaluation": literature,
        "rank_six_raw_kernel_changes_projector_cost": False,
        "cap_sized_aggregate_projector_contraction_supplied": False,
        "fatal_obstructions": [
            "materializing the smaller feature side costs B^(12/5) state",
            "direct or all-output multipoint evaluation costs B^(12/5) fresh work",
            "the exact zero projector has full finite incidence rank despite the rank-six raw kernel",
            "no branch-complete aggregate recurrence computes projector counts before individual evaluations",
        ],
        "scope_exception": (
            "an exact aggregate recurrence for the Fermat projector trace "
            "over Cartesian source boxes with cap-sized intermediate state "
            "and dyadic source replay"
        ),
    }


def projective_controls() -> dict[str, Any]:
    semantics = implicit_index_semantics_control()
    return {
        "affine_signed_source_routes_exact": semantics[
            "all_dyadic_source_routes_exact"
        ],
        "blind_zero_rows_present": False,
        "projective_infinity_chart_complete": False,
        "proper_subsum_replay_inherited_from_r93": True,
        "tangent_multiplicity_source_complete": False,
        "full_five_a_five_c_source_unranking_complete": False,
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "implicit_index": asymptotic_cost_control(),
        "projector_degree": minimal_projector_degree_control(),
        "semantic_route": (
            "rank-six hyperplane -> Fermat zero projector -> exact box "
            "count/existence -> dyadic source"
        ),
        "deduplication": (
            "this is the R9 projector-trace router specialized to the R93 "
            "Veronese value kernel, not a new source-index operation"
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    semantics = implicit_index_semantics_control()
    degree = minimal_projector_degree_control()
    literature = multipoint_literature_control()
    costs = asymptotic_cost_control()
    projective = projective_controls()
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_implicit_veronese_"
            "hyperplane_index.r94.v1"
        ),
        "value_kernel": {
            "veronese_dimension": 6,
            "operator_rank": 6,
            "bound_r93_sha256": R93_REPORT_SHA256,
        },
        "zero_projector": {
            "formula": "1-H^(p-1)",
            "minimal_univariate_degree": degree["degree"],
        },
        "source_interface": (
            "exact box count followed by one nonzero dyadic child"
        ),
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
    }
    range_cost = {
        "schema": (
            "p1553.implicit_cartesian_range_index_cost_ledger.r94.v1"
        ),
        "frozen_candidate": frozen,
        "finite_semantics": semantics,
        "minimal_projector_degree": degree,
        "multipoint_literature_control": literature,
        "asymptotic_cost": costs,
    }
    source_replay = {
        "schema": (
            "p1553.coupled_source_unranking_false_positive_replay.r94.v1"
        ),
        "finite_chart_controls": semantics["chart_rows"],
        "all_affine_sources_exact": semantics[
            "all_dyadic_source_routes_exact"
        ],
        "actual_five_a_five_c_coupled_source_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.projective_hyperplane_exceptional_controls.r94.v1"
        ),
        "projective_controls": projective,
        "false_positive_count": 0,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r94.v1",
        "implicit_affine_hyperplane_semantics_exact": True,
        "cap_sized_aggregate_projector_contraction_found": False,
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
        "fermat_hyperplane_projector_identity_exact": semantics[
            "all_projector_predicates_exact"
        ],
        "minimal_projector_degree_exact": degree[
            "minimal_degree_exact"
        ],
        "all_finite_projector_counts_exact": all(
            row["all_dyadic_counts_exact"]
            for row in semantics["chart_rows"]
        ),
        "all_finite_dyadic_sources_exact": semantics[
            "all_dyadic_source_routes_exact"
        ],
        "direct_dyadic_replay_below_two_scans": all(
            row["less_than_twice_one_side_scan"]
            for row in semantics["chart_rows"]
        ),
        "projector_full_rank_finite_control": semantics[
            "largest_projector_full_rank"
        ],
        "multipoint_theorem_bound_extracted_exactly": True,
        "root_side_exponents_charged": (
            costs["r84_root_source_sides"][
                "smaller_exponent_B"
            ]["exact"]
            == "12/5"
        ),
        "implicit_index_semantically_deduplicated_to_r9": True,
        "materialized_feature_index_inside_setup_cap": False,
        "direct_projector_scan_inside_online_cap": False,
        "all_output_multipoint_inside_online_cap": False,
        "cap_sized_aggregate_projector_contraction_supplied": False,
        "blind_zero_branch_replayed": False,
        "projective_infinity_source_complete": False,
        "tangent_multiplicity_source_complete": False,
        "actual_five_a_five_c_source_unranking_complete": False,
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
            "IMPLICIT_VERONESE_QUERY_EQUALS_FERMAT_PROJECTOR__"
            "DIRECT_DYADIC_SOURCE_EXACT__"
            "STANDARD_MULTIPOINT_B12O5__"
            "CAP_SIZED_CONTRACTION_UNSUPPLIED"
        ),
        "source_bindings": {
            "r93_producer": {
                "path": str(R93_PRODUCER),
                "sha256": R93_PRODUCER_SHA256,
            },
            "r93_report": {
                "path": str(R93_REPORT),
                "sha256": R93_REPORT_SHA256,
            },
            "r93_gate": {
                "path": str(R93_GATE),
                "sha256": R93_GATE_SHA256,
            },
            "r9_projector_trace_gate": {
                "path": str(R9_GATE),
                "sha256": R9_GATE_SHA256,
            },
            "r78_report": {
                "path": str(R78_REPORT),
                "sha256": R78_REPORT_SHA256,
            },
            "r84_report": {
                "path": str(R84_REPORT),
                "sha256": R84_REPORT_SHA256,
            },
            "r82_report": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "p1515_trichotomy": {
                "path": str(P1515_TRICHOTOMY),
                "sha256": P1515_TRICHOTOMY_SHA256,
            },
            "multipoint_paper": {
                "path": str(MULTIPOINT_PAPER),
                "sha256": MULTIPOINT_PAPER_SHA256,
            },
        },
        "novelty_scope": (
            "R94 is the first campaign receipt to specialize the exact R9 "
            "Fermat-projector source router to R93's rank-six Veronese "
            "hyperplane and charge current all-field multipoint evaluation."
        ),
        "implicit_index_semantics_control": semantics,
        "minimal_projector_degree_control": degree,
        "multipoint_literature_control": literature,
        "asymptotic_cost_control": costs,
        "projective_and_false_positive_controls": projective,
        "cost_ledger": cost_ledger(),
        "side_artifacts": {
            "frozen": (
                "frozen_5a5c_implicit_veronese_hyperplane_index.json"
            ),
            "range_cost": (
                "implicit_cartesian_range_index_cost_ledger.json"
            ),
            "source_replay": (
                "coupled_source_unranking_and_false_positive_replay.json"
            ),
            "exceptional": (
                "projective_hyperplane_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r94.json",
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
            "This closes direct scanning, materialized feature indices, and "
            "the bound all-output multipoint evaluator for the implicit "
            "Veronese query. It does not lower-bound an aggregate trace, "
            "character-sum, or nonlinear source-returning recurrence that "
            "avoids individual root-side evaluations."
        ),
        "next_action": (
            "Construct or refute one exact aggregate recurrence for "
            "sum_box(1-H^(p-1)) using the rank-six Veronese H and the "
            "Cartesian A/C source circuits. Every intermediate must fit "
            "B^(9/4) persistent state and B^(5/4) fresh work/workspace, "
            "support one nonzero dyadic child and exact coupled source "
            "unranking, and replay blind zero, infinity, proper-subsum, "
            "tangent, and multiplicity branches without individual "
            "B^(12/5) evaluations, DLP labels, or verifier oracles."
        ),
        "disposition": (
            "REJECT_STANDARD_IMPLICIT_VERONESE_INDEX_ROUTES_ONLY__"
            "FERMAT_HYPERPLANE_PROJECTOR_EXACT__DIRECT_DYADIC_AFFINE_"
            "SOURCE_EXACT__PROJECTOR_FULL_RANK_AT_78__MATERIALIZED_AND_"
            "MULTIPOINT_ROUTES_B12O5__R9_TRACE_CONTRACTION_UNSUPPLIED__"
            "PROJECTIVE_AND_FULL_5A5C_SOURCE_INCOMPLETE__AGGREGATE_"
            "RECURRENCE_OPEN__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_"
            "SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "range_cost": range_cost,
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
            "p1553_5a5c_implicit_veronese_hyperplane_"
            "source_index_probe_report_r94.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_implicit_veronese_hyperplane_index.json"
        ),
    )
    parser.add_argument(
        "--range-cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "implicit_cartesian_range_index_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "coupled_source_unranking_and_false_positive_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "projective_hyperplane_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r94.json"
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
    write_json(args.range_cost_output, bundle["range_cost"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
