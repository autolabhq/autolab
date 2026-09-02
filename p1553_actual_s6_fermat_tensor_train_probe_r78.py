#!/usr/bin/env python3
"""Probe a value-first tensor-train Fermat projector for actual S6."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.actual_s6_fermat_tensor_train_probe.r78.v1"
PREFIX_SIZES = (3, 4, 5, 6)
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_WORKSPACE_CAP_EXPONENT = 5 / 4

R72_REPORT = pathlib.Path(
    "p1553_s6_centered_carry_rank_minor_probe_report_r72.json"
)
R72_REPORT_SHA256 = (
    "7e63b52fc7667be14aadc1db3aeeb22b43876ff2c62e3dce87b312c056e85e43"
)
R75_REPORT = pathlib.Path(
    "p1553_s6_iterated_norm_support_probe_report_r75.json"
)
R75_REPORT_SHA256 = (
    "c5f41fbb7325fe3f4e85fe6084fbaed6ab2df9d9101bd031cd1c589a95a3ba8c"
)
R76_REPORT = pathlib.Path(
    "p1553_s6_subset_incidence_mobius_probe_report_r76.json"
)
R76_REPORT_SHA256 = (
    "de41d1618bc71c46f700bfead0ed72c5ac0b29a89da3b5c32534a15314ef4c93"
)
R2_REGISTRY = pathlib.Path("p1553_r31_artifact_index_README.md")
R2_REGISTRY_SHA256 = (
    "0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f"
)
R2_GATE_SHA256 = (
    "55acc1457e7fd5a740da57c2c1db957374c7c18561c67b1748176dc8c61fcda5"
)


def load_module(path: str, module_name: str) -> Any:
    module_path = pathlib.Path(__file__).with_name(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R72 = load_module(
    "p1553_s6_centered_carry_rank_minor_probe_r72.py",
    "p1553_r72_for_r78",
)
Point = tuple[int, int] | None


def tensor_values(
    curve: dict[str, Any],
    decks: list[list[Point]],
    target: Point,
    size: int,
) -> list[int]:
    if target is None:
        raise ValueError("target must be affine")
    prime = curve["field_prime"]
    values = []
    for indices in itertools.product(range(size), repeat=5):
        x_coordinates = tuple(
            decks[mode][index][0]
            for mode, index in enumerate(indices)
        ) + (target[0],)
        values.append(
            R72.semaev_s6_mod(
                x_coordinates,
                curve["curve_a"],
                curve["curve_b"],
                prime,
            )
        )
    return values


def unfolding_rows(
    values: Sequence[int],
    size: int,
    cut: int,
) -> list[list[int]]:
    if len(values) != size**5:
        raise ValueError("tensor does not have five equal modes")
    if not 1 <= cut <= 4:
        raise ValueError("cut must be between one and four")
    column_count = size ** (5 - cut)
    return [
        list(values[start : start + column_count])
        for start in range(0, len(values), column_count)
    ]


def tt_ranks(
    values: Sequence[int],
    size: int,
    modulus: int,
) -> list[int]:
    return [
        len(
            R72.R70.row_reduce(
                unfolding_rows(values, size, cut),
                size ** (5 - cut),
                modulus,
            )[1]
        )
        for cut in range(1, 5)
    ]


def tt_core_parameter_count(size: int, ranks: Sequence[int]) -> int:
    extended = [1, *ranks, 1]
    return sum(
        extended[index] * size * extended[index + 1]
        for index in range(5)
    )


def full_tt_ranks(size: int) -> list[int]:
    return [
        min(size**cut, size ** (5 - cut))
        for cut in range(1, 5)
    ]


def dyadic_child_replay(
    zero_mask: Sequence[int],
    size: int,
) -> dict[str, Any]:
    block = size**4
    leaf_counts = [
        sum(zero_mask[index * block : (index + 1) * block])
        for index in range(size)
    ]
    conservation = []

    def visit(start: int, stop: int) -> int:
        count = sum(leaf_counts[start:stop])
        if stop - start <= 1:
            return count
        middle = start + (stop - start) // 2
        left = visit(start, middle)
        right = visit(middle, stop)
        conservation.append(count == left + right)
        return count

    root_count = visit(0, size)
    return {
        "root_count": root_count,
        "internal_node_count": len(conservation),
        "all_parent_counts_equal_child_sums": all(conservation),
        "positive_leaf_indices": [
            index for index, count in enumerate(leaf_counts) if count
        ],
    }


def source_from_mask(
    zero_mask: Sequence[int],
    size: int,
) -> list[int] | None:
    try:
        flat_index = zero_mask.index(1)
    except ValueError:
        return None
    indices = [0] * 5
    for mode in range(4, -1, -1):
        indices[mode] = flat_index % size
        flat_index //= size
    return indices


def source_verifies(
    source: Sequence[int] | None,
    decks: list[list[Point]],
    target: Point,
    curve: dict[str, Any],
) -> bool | None:
    if source is None:
        return None
    points = tuple(
        decks[mode][index] for mode, index in enumerate(source)
    )
    return R72.signed_relation_exists(points, target, curve)


def fermat_occurrence_control() -> dict[str, Any]:
    prime = 101
    values = [0, 0, 7]
    zero_mask = [
        (1 - pow(value, prime - 1, prime)) % prime
        for value in values
    ]
    return {
        "prime": prime,
        "resultant_values": values,
        "zero_mask": zero_mask,
        "duplicate_zero_occurrence_count": sum(zero_mask),
        "duplicate_occurrences_preserved": sum(zero_mask) == 2,
        "multiple_common_root_boundary": (
            "One tuple contributes one indicator whenever its resultant is "
            "zero, independently of the degree of its internal polynomial gcd."
        ),
    }


def target_profile(
    curve: dict[str, Any],
    decks: list[list[Point]],
    target_spec: dict[str, Any],
    size: int,
) -> dict[str, Any]:
    prime = curve["field_prime"]
    values = tensor_values(
        curve,
        decks,
        target_spec["point"],
        size,
    )
    raw_ranks = tt_ranks(values, size, prime)
    squared = [value * value % prime for value in values]
    square_ranks = tt_ranks(squared, size, prime)
    nonzero_mask = [
        pow(value, prime - 1, prime) for value in values
    ]
    if any(value not in (0, 1) for value in nonzero_mask):
        raise AssertionError("Fermat projector left a non-Boolean value")
    zero_mask = [1 - value for value in nonzero_mask]
    projector_ranks = tt_ranks(nonzero_mask, size, prime)
    zero_ranks = tt_ranks(zero_mask, size, prime)
    source = source_from_mask(zero_mask, size)
    dyadic = dyadic_child_replay(zero_mask, size)
    full_ranks = full_tt_ranks(size)
    middle_full = raw_ranks[1:3] == full_ranks[1:3]
    return {
        "target_id": target_spec["target_id"],
        "deck_size": size,
        "tensor_entry_count": len(values),
        "raw_s6_zero_occurrence_count": sum(zero_mask),
        "raw_s6_tt_ranks": raw_ranks,
        "first_hadamard_square_tt_ranks": square_ranks,
        "fermat_nonzero_projector_tt_ranks": projector_ranks,
        "final_zero_mask_tt_ranks": zero_ranks,
        "full_tt_rank_bounds": full_ranks,
        "raw_middle_cuts_full_rank": middle_full,
        "raw_minimal_tt_core_parameter_count": (
            tt_core_parameter_count(size, raw_ranks)
        ),
        "raw_minimal_tt_center_core_entry_count": (
            raw_ranks[1] * size * raw_ranks[2]
        ),
        "raw_center_core_exponent_B": (
            math.log(raw_ranks[1] * size * raw_ranks[2], size)
            if size > 1
            else None
        ),
        "final_zero_mask_tt_core_parameter_count": (
            tt_core_parameter_count(size, zero_ranks)
        ),
        "source_indices": source,
        "source_relation_verified": source_verifies(
            source,
            decks,
            target_spec["point"],
            curve,
        ),
        "forced_witness_recognized": (
            None
            if target_spec["forced_witness"] is None
            or max(target_spec["forced_witness"]) >= size
            else bool(sum(zero_mask))
        ),
        "dyadic_child_replay": dyadic,
        "fermat_binary_schedule": {
            "squaring_count_per_entry": (prime - 1).bit_length() - 1,
            "multiply_count_per_entry": max(
                0,
                (prime - 1).bit_count() - 1,
            ),
            "explicit_value_first_field_operation_floor": (
                len(values)
                * (
                    (prime - 1).bit_length()
                    - 1
                    + max(0, (prime - 1).bit_count() - 1)
                )
            ),
        },
    }


def probe_curve(
    curve: dict[str, Any],
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, Any]:
    decks, targets = R72.public_decks_and_targets(curve)
    rows = [
        target_profile(curve, decks, target, size)
        for size in prefix_sizes
        for target in targets
    ]
    return {
        "family_id": curve["family_id"],
        "field_bits": curve["field_prime"].bit_length(),
        "scalar_labels_consumed": False,
        "targets": rows,
    }


def build_bundle(
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, dict[str, Any]]:
    prefix_sizes = tuple(prefix_sizes)
    families = [
        probe_curve(dict(curve), prefix_sizes=prefix_sizes)
        for curve in R72.CURVES
    ]
    rows = [row for family in families for row in family["targets"]]
    blind_rows = [
        row for row in rows if row["target_id"] == "blind_hash_target"
    ]
    forced_rows = [
        row for row in rows if row["target_id"] == "forced_positive_target"
    ]
    occurrence_control = fermat_occurrence_control()
    grammar = {
        "schema": "p1553.frozen_nonlinear_nested_resultant_functional.r78.v1",
        "grammar_id": "actual_s6_value_tensor_tt_fermat_binary_v1",
        "input_leaf": (
            "actual Res_z(S4(x1,x2,x3,z),S4(xR,x4,x5,z)) value"
        ),
        "target_specialized": True,
        "nonlinear_operation": "entrywise binary Fermat powering",
        "state_representation": "exact five-core tensor train after every operation",
        "output": "sum of 1-value^(p-1), plus TT-guided dyadic source descent",
        "group_characters_used": False,
        "suffix_roots_materialized": False,
        "prefix_values_materialized_in_tt": True,
        "frozen_before_target_outcomes": True,
    }
    specialization_replay = {
        "schema": "p1553.actual_s4_target_specialization_replay.r78.v1",
        "curve_families": families,
        "all_raw_middle_cuts_full_rank": all(
            row["raw_middle_cuts_full_rank"] for row in rows
        ),
        "all_fermat_outputs_boolean": True,
    }
    source_replay = {
        "schema": "p1553.r76_mobius_zero_source_child_replay.r78.v1",
        "occurrence_control": occurrence_control,
        "blind_rows": [
            {
                "family_id": family["family_id"],
                "targets": [
                    {
                        "deck_size": row["deck_size"],
                        "count": row["raw_s6_zero_occurrence_count"],
                        "dyadic": row["dyadic_child_replay"],
                    }
                    for row in family["targets"]
                    if row["target_id"] == "blind_hash_target"
                ],
            }
            for family in families
        ],
        "forced_rows": [
            {
                "family_id": family["family_id"],
                "targets": [
                    {
                        "deck_size": row["deck_size"],
                        "count": row["raw_s6_zero_occurrence_count"],
                        "source_indices": row["source_indices"],
                        "source_relation_verified": row[
                            "source_relation_verified"
                        ],
                        "dyadic": row["dyadic_child_replay"],
                    }
                    for row in family["targets"]
                    if row["target_id"] == "forced_positive_target"
                ],
            }
            for family in families
        ],
        "all_blind_counts_zero": all(
            row["raw_s6_zero_occurrence_count"] == 0
            for row in blind_rows
        ),
        "all_forced_counts_one": all(
            row["raw_s6_zero_occurrence_count"] == 1
            for row in forced_rows
        ),
        "all_forced_sources_verified": all(
            row["source_relation_verified"] for row in forced_rows
        ),
        "all_dyadic_counts_conserve": all(
            row["dyadic_child_replay"][
                "all_parent_counts_equal_child_sums"
            ]
            and row["dyadic_child_replay"]["root_count"]
            == row["raw_s6_zero_occurrence_count"]
            for row in rows
        ),
    }
    cost_ledger = {
        "schema": "p1553.nonlinear_functional_state_direct_cost.r78.v1",
        "setup_state_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
        "online_workspace_cap_exponent_B": ONLINE_WORKSPACE_CAP_EXPONENT,
        "raw_value_tensor_entry_exponent_B": 5.0,
        "observed_raw_middle_tt_center_core_exponent_B": 5.0,
        "binary_fermat_pointwise_work_exponent_B": 5.0,
        "field_bitlength_factor_suppressed": True,
        "final_sparse_mask_receives_no_constructor_credit": True,
        "lane_inside_caps": False,
        "maximum_observed_center_core_entry_count": max(
            row["raw_minimal_tt_center_core_entry_count"] for row in rows
        ),
        "maximum_final_zero_mask_tt_parameter_count": max(
            row["final_zero_mask_tt_core_parameter_count"] for row in rows
        ),
    }
    report = {
        "schema": SCHEMA,
        "classification": (
            "ACTUAL_S6_VALUE_FIRST_FERMAT_TT_HAS_B5_CENTER_CORE"
        ),
        "source_bindings": {
            "r72_actual_s6_probe": {
                "path": str(R72_REPORT),
                "sha256": R72_REPORT_SHA256,
            },
            "r75_expanded_norm_control": {
                "path": str(R75_REPORT),
                "sha256": R75_REPORT_SHA256,
            },
            "r76_exact_incidence_control": {
                "path": str(R76_REPORT),
                "sha256": R76_REPORT_SHA256,
            },
            "r2_finite_deck_registry": {
                "path": str(R2_REGISTRY),
                "sha256": R2_REGISTRY_SHA256,
                "bound_gate_sha256": R2_GATE_SHA256,
            },
        },
        "grammar": grammar,
        "families": families,
        "aggregate": {
            "family_count": len(families),
            "target_instance_count": len(rows),
            "all_raw_middle_cuts_full_rank": all(
                row["raw_middle_cuts_full_rank"] for row in rows
            ),
            "all_blind_counts_zero": source_replay[
                "all_blind_counts_zero"
            ],
            "all_forced_counts_one": source_replay[
                "all_forced_counts_one"
            ],
            "all_forced_sources_verified": source_replay[
                "all_forced_sources_verified"
            ],
            "all_dyadic_counts_conserve": source_replay[
                "all_dyadic_counts_conserve"
            ],
            "duplicate_occurrence_control_passed": occurrence_control[
                "duplicate_occurrences_preserved"
            ],
            "maximum_raw_center_core_entry_count": max(
                row["raw_minimal_tt_center_core_entry_count"]
                for row in rows
            ),
            "maximum_final_zero_mask_tt_parameter_count": max(
                row["final_zero_mask_tt_core_parameter_count"]
                for row in rows
            ),
        },
        "side_artifacts": {
            "frozen_grammar": (
                "frozen_nonlinear_nested_resultant_functional_grammar.json"
            ),
            "target_specialization_replay": (
                "actual_s4_target_specialization_replay.json"
            ),
            "source_child_replay": (
                "r76_mobius_zero_source_and_child_replay.json"
            ),
            "cost_ledger": (
                "nonlinear_functional_state_and_direct_cost_ledger.json"
            ),
        },
        "cost_ledger": cost_ledger,
        "deduplication": {
            "r75_overlap": (
                "R75 closes expanded coefficient norms. R78 instead freezes "
                "a value-first exact TT plus Hadamard-Fermat grammar."
            ),
            "r2_overlap": (
                "R2 measured determinant-channel TT ranks and endpoint "
                "traffic. R78 measures actual target-specialized S6 values "
                "and the constructor path to the Fermat zero mask."
            ),
            "new_idea_id": None,
        },
        "admission": {
            "passed_obligation_count": 7,
            "obligation_count": 11,
            "lane_admitted": False,
            "failures": [
                "the raw target-specialized TT has a B^5 center core",
                "binary Fermat powering touches B^5 tuple values",
                "the tiny final zero mask is outcome-conditioned",
                "factor-log solve and identical fresh-target descent are unsupplied",
            ],
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "scope_boundary": (
            "This rejects only the frozen value-first exact tensor-train "
            "binary-Fermat grammar. It is not a lower bound against a "
            "straight-line scalar resultant/norm circuit that never "
            "represents tuple values or TT cores."
        ),
        "next_action": (
            "Construct a straight-line black-box nested S4 norm functional "
            "whose every node is scalar or at most B^(9/4) persistent state "
            "and B^(5/4) fresh-target state/work. It may never materialize a "
            "tuple-value tensor, TT core, coefficient cube, quotient algebra, "
            "or suffix polynomial, and must retain R76 zero, count, source, "
            "multiplicity, and dyadic replay."
        ),
        "disposition": (
            "REJECT_ACTUAL_S6_VALUE_FIRST_TT_FERMAT_GRAMMAR_ONLY__"
            "FOUR_STANDARD_CURVES__B3_4_5_6__BLIND_ZERO__FORCED_COUNT_ONE__"
            "SOURCE_AND_DYADIC_REPLAY__ALL_RAW_MIDDLE_CUTS_FULL__"
            "CENTER_CORE_B5__FINAL_MASK_TINY_ONLY_AFTER_VALUE_CONSTRUCTION__"
            "R2_R75_BOUNDARIES_SHARPENED__SCALAR_ONLY_CIRCUITS_OPEN__"
            "NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "grammar": grammar,
        "specialization_replay": specialization_replay,
        "source_replay": source_replay,
        "cost_ledger": cost_ledger,
    }


def build_report(
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, Any]:
    return build_bundle(prefix_sizes=prefix_sizes)["report"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_actual_s6_fermat_tensor_train_probe_report_r78.json"
        ),
    )
    parser.add_argument(
        "--grammar-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_nonlinear_nested_resultant_functional_grammar.json"
        ),
    )
    parser.add_argument(
        "--specialization-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "actual_s4_target_specialization_replay.json"
        ),
    )
    parser.add_argument(
        "--source-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "r76_mobius_zero_source_and_child_replay.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "nonlinear_functional_state_and_direct_cost_ledger.json"
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
    write_json(args.grammar_output, bundle["grammar"])
    write_json(args.specialization_output, bundle["specialization_replay"])
    write_json(args.source_output, bundle["source_replay"])
    write_json(args.cost_output, bundle["cost_ledger"])
    aggregate = bundle["report"]["aggregate"]
    print(
        f"families={aggregate['family_count']} "
        f"middle_full={aggregate['all_raw_middle_cuts_full_rank']} "
        f"sources_verified={aggregate['all_forced_sources_verified']} "
        f"lane_admitted={bundle['report']['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
