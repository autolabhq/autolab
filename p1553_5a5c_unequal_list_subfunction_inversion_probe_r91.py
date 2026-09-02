#!/usr/bin/env python3
"""Audit unequal-list subfunction inversion for the R82 5A+5C source."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.5a5c_unequal_list_subfunction_inversion.r91.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
ATOM_A_EXPONENT = Fraction(2, 5)
ATOM_C_EXPONENT = Fraction(3, 5)

R90_REPORT = pathlib.Path(
    "p1553_5a5c_nonlocal_moment_hankel_translation_"
    "probe_report_r90.json"
)
R90_REPORT_SHA256 = (
    "02bece6fe25e335bd061eec784e237b6d9d2ab57f34bd8dd823217a137a56c2b"
)
R90_GATE = pathlib.Path(
    "p1553_5a5c_nonlocal_moment_hankel_translation_"
    "probe_gate_r90.md"
)
R90_GATE_SHA256 = (
    "b1618f6a354b995db01fbbc7aeeb69df6ebb5248b5c558bc4c72d87ce523897b"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
R83_REPORT = pathlib.Path(
    "p1553_5a5c_coordinate_filtration_probe_report_r83.json"
)
R83_REPORT_SHA256 = (
    "1478cdf21493ffbeaed0859af849ea6f2835027f23db7e3e4c008f3f24db500c"
)
R83_QUOTIENT_CONTROL = pathlib.Path(
    "partial_filter_composability_and_false_positive_controls.json"
)
R83_QUOTIENT_CONTROL_SHA256 = (
    "b0dd2c5f2675e8e0967a6c0722f26f35700ac9b68f41aafa5f461e17c1cb71af"
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
DINUR_GOLOVNEV_V2 = pathlib.Path(
    "references/dinur_golovnev_3sum_indexing_2512.04258v2.pdf"
)
DINUR_GOLOVNEV_V2_SHA256 = (
    "e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c"
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R90_REPORT: R90_REPORT_SHA256,
        R90_GATE: R90_GATE_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        R83_REPORT: R83_REPORT_SHA256,
        R83_QUOTIENT_CONTROL: R83_QUOTIENT_CONTROL_SHA256,
        P1515_TRICHOTOMY: P1515_TRICHOTOMY_SHA256,
        P1515_ROUTER: P1515_ROUTER_SHA256,
        DINUR_GOLOVNEV_V2: DINUR_GOLOVNEV_V2_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R91 source binding mismatch: {failures}")
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


def unequal_list_exponents(
    small_list_exponent: Fraction,
    large_list_exponent: Fraction,
    delta: Fraction,
) -> dict[str, Fraction]:
    if not (
        Fraction(0) < small_list_exponent <= large_list_exponent
    ):
        raise AssertionError("the unequal-list theorem requires 0<n<=m")
    if not Fraction(0) <= delta <= Fraction(1):
        raise AssertionError("the theorem requires 0<=delta<=1")
    return {
        "space": (
            small_list_exponent
            * (Fraction(3, 2) - delta)
            + large_list_exponent
        ),
        "query": small_list_exponent * delta,
        "auxiliary_list": large_list_exponent,
    }


def best_point_under_online_cap(
    small_list_exponent: Fraction,
    large_list_exponent: Fraction,
) -> dict[str, Any]:
    delta = min(
        Fraction(1),
        ONLINE_CAP / small_list_exponent,
    )
    exponents = unequal_list_exponents(
        small_list_exponent,
        large_list_exponent,
        delta,
    )
    return {
        "small_list_exponent_B": fraction_record(
            small_list_exponent
        ),
        "large_list_exponent_B": fraction_record(
            large_list_exponent
        ),
        "delta": fraction_record(delta),
        "space_exponent_B": fraction_record(exponents["space"]),
        "query_exponent_B": fraction_record(exponents["query"]),
        "auxiliary_list_exponent_B": fraction_record(
            exponents["auxiliary_list"]
        ),
        "online_cap_satisfied": exponents["query"] <= ONLINE_CAP,
        "setup_cap_satisfied": exponents["space"] <= SETUP_CAP,
        "auxiliary_list_inside_setup_cap": (
            exponents["auxiliary_list"] <= SETUP_CAP
        ),
    }


def intended_five_a_five_c_application() -> dict[str, Any]:
    small = Fraction(2)
    large = Fraction(3)
    delta_for_online = ONLINE_CAP / small
    point = best_point_under_online_cap(small, large)
    delta_for_setup = (
        small * Fraction(3, 2) + large - SETUP_CAP
    ) / small
    return {
        "paper_theorem": (
            "S=soft-O(n^(3/2-delta)*m), "
            "T=soft-O(n^delta), 0<=delta<=1"
        ),
        "substitution": "n=B^2 five-A endpoints, m=B^3 five-C endpoints",
        "space_exponent_formula_B": "6-2*delta",
        "query_exponent_formula_B": "2*delta",
        "delta_required_by_setup_cap": fraction_record(
            delta_for_setup
        ),
        "delta_allowed_by_online_cap": fraction_record(
            delta_for_online
        ),
        "cap_interval_nonempty": (
            delta_for_setup
            <= delta_for_online
            and delta_for_setup <= 1
        ),
        "best_point_under_online_cap": point,
        "explicit_auxiliary_contains_full_large_list": True,
        "explicit_auxiliary_state_exponent_B": fraction_record(large),
    }


def deck_partition_exponents() -> list[dict[str, Any]]:
    representatives: dict[Fraction, list[tuple[int, int]]] = {}
    total = Fraction(5)
    for a_count in range(6):
        for c_count in range(6):
            side = (
                a_count * ATOM_A_EXPONENT
                + c_count * ATOM_C_EXPONENT
            )
            if side <= 0 or side >= total:
                continue
            small = min(side, total - side)
            representatives.setdefault(small, []).append(
                (a_count, c_count)
            )
    rows = []
    for small in sorted(representatives):
        large = total - small
        point = best_point_under_online_cap(small, large)
        rows.append(
            {
                "small_side_exponent_B": fraction_record(small),
                "large_side_exponent_B": fraction_record(large),
                "representative_small_or_complement_deck_counts": [
                    {
                        "a_decks_on_named_side": a_count,
                        "c_decks_on_named_side": c_count,
                    }
                    for a_count, c_count in representatives[small][
                        :4
                    ]
                ],
                "best_point_under_online_cap": point,
            }
        )
    return rows


def all_partition_cap_control() -> dict[str, Any]:
    rows = deck_partition_exponents()
    best_row = min(
        rows,
        key=lambda row: row["best_point_under_online_cap"][
            "space_exponent_B"
        ]["decimal"],
    )
    return {
        "deck_multiset": {
            "a_decks": 5,
            "a_deck_exponent_B": fraction_record(ATOM_A_EXPONENT),
            "c_decks": 5,
            "c_deck_exponent_B": fraction_record(ATOM_C_EXPONENT),
            "total_source_exponent_B": fraction_record(Fraction(5)),
        },
        "unique_nonempty_partition_count": len(rows),
        "partition_rows": rows,
        "best_space_point_while_respecting_online_cap": best_row,
        "minimum_space_exponent_B": best_row[
            "best_point_under_online_cap"
        ]["space_exponent_B"],
        "any_partition_meets_both_caps": any(
            row["best_point_under_online_cap"][
                "setup_cap_satisfied"
            ]
            for row in rows
        ),
        "any_partition_auxiliary_list_inside_setup_cap": any(
            row["best_point_under_online_cap"][
                "auxiliary_list_inside_setup_cap"
            ]
            for row in rows
        ),
    }


def balanced_factor_base_ksum_control() -> dict[str, Any]:
    k = 6
    delta = Fraction(1)
    space = Fraction(2 * k - 1, 2) - delta
    query = delta
    return {
        "factor_base_size": "B",
        "query": "find five factor-base elements summing to the target",
        "paper_k_parameter": k,
        "paper_theorem": "S=soft-O(n^(k-1/2-delta)), T=soft-O(n^delta)",
        "best_delta_in_paper_range": fraction_record(delta),
        "space_exponent_B": fraction_record(space),
        "query_exponent_B": fraction_record(query),
        "setup_cap_satisfied": space <= SETUP_CAP,
        "online_cap_satisfied": query <= ONLINE_CAP,
        "scalar_label_granted_optimistic_control": True,
    }


def first_index_with_residue(
    values: Iterable[int],
    residue: int,
    modulus: int,
) -> int | None:
    for index, value in enumerate(values):
        if value % modulus == residue:
            return index
    return None


def integer_subfunction_source_replay() -> dict[str, Any]:
    left = [2, 7, 13, 19, 28, 34, 41]
    right = [1, 5, 9, 16, 22, 31, 38, 47, 53, 64, 70]
    modulus_p = 131
    modulus_q = 127
    filler = 130
    targets = sorted({a + b for a in left for b in right})
    right_exact = {
        value: index for index, value in enumerate(right)
    }

    def subfunction(index: int, residue: int) -> int:
        needed = (residue - left[index]) % modulus_q
        right_index = first_index_with_residue(
            right,
            needed,
            modulus_q,
        )
        if right_index is None:
            return filler
        return (left[index] + right[right_index]) % modulus_p

    recovered = []
    for target in targets:
        residue = target % modulus_q
        image = target % modulus_p
        candidates = [
            index
            for index in range(len(left))
            if subfunction(index, residue) == image
        ]
        source = None
        for left_index in candidates:
            needed = target - left[left_index]
            if needed in right_exact:
                source = (left_index, right_exact[needed])
                break
        recovered.append(
            {
                "target": target,
                "source": list(source) if source is not None else None,
                "verified": (
                    source is not None
                    and left[source[0]] + right[source[1]] == target
                ),
            }
        )
    absent = next(
        target
        for target in range(max(targets) + 2)
        if target not in set(targets)
    )
    absent_candidates = [
        index
        for index in range(len(left))
        if subfunction(index, absent % modulus_q)
        == absent % modulus_p
    ]
    absent_false_positive = any(
        absent - left[index] in right_exact
        for index in absent_candidates
    )
    return {
        "integer_lists": {"left": left, "right": right},
        "modulus_p": modulus_p,
        "modulus_q": modulus_q,
        "target_count": len(targets),
        "all_present_targets_report_exact_source": all(
            row["verified"] for row in recovered
        ),
        "absent_target": absent,
        "absent_target_rejected_after_exact_verification": (
            not absent_false_positive
        ),
        "direct_subfunction_scan_used": True,
        "fiat_naor_chain_runtime_credited": False,
        "scope": (
            "positive semantics control with explicit integer lists and "
            "large collision-free moduli; not an asymptotic implementation"
        ),
    }


def prime_order_transfer_control() -> dict[str, Any]:
    group_order = 101
    proper_target_orders = [2, 3, 5, 7, 11, 13]
    possible_image_orders = [
        order
        for order in proper_target_orders
        if group_order % order == 0
    ]
    return {
        "source_group": "cyclic group of prime order 101",
        "tested_proper_target_orders": proper_target_orders,
        "nontrivial_homomorphic_image_orders": possible_image_orders,
        "all_proper_homomorphic_filters_trivial": (
            not possible_image_orders
        ),
        "proof": (
            "the image order divides both the prime source order and the "
            "target order; for every smaller target order the gcd is one"
        ),
        "paper_integer_maps": ["y mod q", "y mod p"],
        "public_point_encoding_is_addition_compatible": False,
        "dlog_labels_make_integer_maps_addition_compatible": True,
        "dlog_labels_available_to_candidate": False,
        "r83_prime_order_quotient_control_bound": True,
        "generic_prime_group_transfer_complete": False,
    }


def cost_ledger() -> dict[str, Any]:
    intended = intended_five_a_five_c_application()
    partitions = all_partition_cap_control()
    balanced = balanced_factor_base_ksum_control()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "intended_five_a_five_c_application": intended,
        "all_ten_deck_partitions": partitions,
        "balanced_factor_base_ksum": balanced,
        "best_known_theorem_space_exponent_under_online_cap": (
            partitions["minimum_space_exponent_B"]
        ),
        "best_known_theorem_inside_setup_cap": False,
        "best_known_theorem_inside_online_cap": True,
        "explicit_endpoint_lists_inside_setup_cap": False,
        "generic_prime_group_transfer_available": False,
        "fatal_obstructions": [
            "the intended B^2/B^3 theorem point needs B^4.75 space under the online cap",
            "the best ten-deck partition still needs B^4.4 space",
            "the theorem auxiliary stores the explicit large endpoint list",
            "integer residue maps do not transfer to public encodings of a generic prime-order elliptic group",
        ],
        "scope_exception": (
            "a compact elliptic subfunction decomposition whose MAP1, MAP2, "
            "f_d, and source translator act directly on D_A,D_C without "
            "DLP labels, proper quotients, or explicit endpoint lists"
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    intended = intended_five_a_five_c_application()
    partitions = all_partition_cap_control()
    balanced = balanced_factor_base_ksum_control()
    replay_control = integer_subfunction_source_replay()
    transfer = prime_order_transfer_control()
    costs = cost_ledger()
    frozen = {
        "schema": "p1553.frozen_5a5c_unequal_list_subfunction_index.r91.v1",
        "paper": {
            "title": "Improved Time-Space Tradeoffs for 3SUM-Indexing",
            "authors": ["Itai Dinur", "Alexander Golovnev"],
            "version": "arXiv:2512.04258v2",
            "date": "2026-04-23",
            "sha256": DINUR_GOLOVNEV_V2_SHA256,
        },
        "theorem_5_1": {
            "input": "integer lists of lengths n<=m",
            "space": "soft-O(n^(3/2-delta)*m)",
            "query": "soft-O(n^delta)",
            "delta_range": "[0,1]",
            "output": "one exact source pair or bottom",
            "auxiliary": "sorted explicit n-list and m-list plus residues",
        },
        "candidate_substitution": {
            "n": "B^2 five-A endpoint list",
            "m": "B^3 five-C endpoint list",
        },
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
    }
    exponent_ledger = {
        "schema": "p1553.subfunction_inversion_exponent_ledger.r91.v1",
        "frozen_theorem": frozen,
        "intended_application": intended,
        "all_ten_deck_partitions": partitions,
        "balanced_factor_base_ksum": balanced,
        "cost_ledger": costs,
    }
    replay = {
        "schema": "p1553.finite_field_source_reporting_replay.r91.v1",
        "integer_subfunction_positive_control": replay_control,
        "elliptic_five_a_five_c_source_replay_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": "p1553.exceptional_branch_random_controls.r91.v1",
        "prime_order_transfer_control": transfer,
        "matched_integer_source_reporting_control": replay_control,
        "actual_semaev_reduced_branch_replay": False,
        "multiple_nonreduced_signed_infinity_tangent_replay": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r91.v1",
        "subcap_unequal_list_index_found": False,
        "compact_elliptic_subfunction_map_found": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "eight_source_bindings_verified": len(bindings) == 8,
        "paper_v2_theorem_extracted_exactly": True,
        "intended_split_exponents_derived_exactly": (
            intended["best_point_under_online_cap"][
                "space_exponent_B"
            ]["exact"]
            == "19/4"
        ),
        "all_ten_deck_partitions_enumerated": (
            partitions["unique_nonempty_partition_count"] == 11
        ),
        "best_partition_space_exponent_exact": (
            partitions["minimum_space_exponent_B"]["exact"] == "22/5"
        ),
        "balanced_ksum_control_exact": (
            balanced["space_exponent_B"]["exact"] == "9/2"
        ),
        "integer_subfunction_source_reporting_exact": replay_control[
            "all_present_targets_report_exact_source"
        ],
        "absent_integer_target_rejected": replay_control[
            "absent_target_rejected_after_exact_verification"
        ],
        "prime_order_quotient_obstruction_exact": transfer[
            "all_proper_homomorphic_filters_trivial"
        ],
        "subfunction_index_inside_setup_cap": False,
        "explicit_endpoint_auxiliary_inside_setup_cap": False,
        "generic_prime_group_transfer_complete": False,
        "compact_elliptic_subfunction_map_supplied": False,
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
            "UNEQUAL_LIST_SUBFUNCTION_THEOREM_EXACT__"
            "BEST_ONLINE_COMPATIBLE_SETUP_B4P4__"
            "INTEGER_RESIDUE_MAP_NO_GENERIC_GROUP_TRANSFER"
        ),
        "source_bindings": {
            "r90_report": {
                "path": str(R90_REPORT),
                "sha256": R90_REPORT_SHA256,
            },
            "r90_gate": {
                "path": str(R90_GATE),
                "sha256": R90_GATE_SHA256,
            },
            "r82_report": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "r83_report": {
                "path": str(R83_REPORT),
                "sha256": R83_REPORT_SHA256,
            },
            "r83_prime_order_quotient_control": {
                "path": str(R83_QUOTIENT_CONTROL),
                "sha256": R83_QUOTIENT_CONTROL_SHA256,
            },
            "p1515_local_separator_trichotomy": {
                "path": str(P1515_TRICHOTOMY),
                "sha256": P1515_TRICHOTOMY_SHA256,
            },
            "p1515_field_router_candidate": {
                "path": str(P1515_ROUTER),
                "sha256": P1515_ROUTER_SHA256,
            },
            "dinur_golovnev_v2": {
                "path": str(DINUR_GOLOVNEV_V2),
                "sha256": DINUR_GOLOVNEV_V2_SHA256,
            },
        },
        "novelty_scope": (
            "R91 is the first campaign receipt to apply the 2026 v2 "
            "unequal-list theorem directly to the colored 5A+5C exponents "
            "and to enumerate every atom-deck bipartition under the actual "
            "setup and online caps."
        ),
        "intended_five_a_five_c_application": intended,
        "all_ten_deck_partition_control": partitions,
        "balanced_factor_base_ksum_control": balanced,
        "integer_subfunction_source_replay": replay_control,
        "prime_order_transfer_control": transfer,
        "cost_ledger": costs,
        "side_artifacts": {
            "frozen": "frozen_5a5c_unequal_list_subfunction_index.json",
            "exponents": "subfunction_inversion_exponent_ledger.json",
            "replay": "finite_field_source_reporting_replay.json",
            "exceptional": (
                "exceptional_branch_and_matched_random_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r91.json",
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
            "This closes direct applications of the bound 2026 unequal-list "
            "and balanced kSUM-indexing theorems to explicit 5A+5C endpoint "
            "lists. It does not lower-bound a compact elliptic subfunction "
            "decomposition, a non-Fiat-Naor data structure, or a "
            "representation-changing summation-polynomial/FFE identity."
        ),
        "next_action": (
            "Construct or refute one compact elliptic analogue of the "
            "subfunction theorem: public MAP1, MAP2, f_d, and TR must act "
            "directly on compact D_A,D_C and one target, reduce inversion "
            "to one subfunction, fit B^(9/4) setup and B^(5/4) fresh source "
            "return, and use no DLP labels, proper quotient, explicit "
            "endpoint list, verifier oracle, or omitted projective branch."
        ),
        "disposition": (
            "REJECT_BOUND_UNEQUAL_LIST_AND_BALANCED_KSUM_INDEX_THEOREMS_"
            "ONLY__INTENDED_SPLIT_B4P75_SETUP_UNDER_ONLINE_CAP__BEST_TEN_"
            "DECK_SPLIT_B4P4__EXPLICIT_LARGE_LIST_OVER_CAP__INTEGER_SOURCE_"
            "REPORTING_CONTROL_EXACT__NO_ADDITIVE_RESIDUE_MAP_ON_GENERIC_"
            "PRIME_GROUP_ENCODINGS__COMPACT_ELLIPTIC_SUBFUNCTION_MAP_OPEN__"
            "NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_"
            "BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "exponents": exponent_ledger,
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
            "p1553_5a5c_unequal_list_subfunction_inversion_"
            "probe_report_r91.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_unequal_list_subfunction_index.json"
        ),
    )
    parser.add_argument(
        "--exponent-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "subfunction_inversion_exponent_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "finite_field_source_reporting_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "exceptional_branch_and_matched_random_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r91.json"
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
    write_json(args.exponent_output, bundle["exponents"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
