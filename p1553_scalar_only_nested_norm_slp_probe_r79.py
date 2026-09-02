#!/usr/bin/env python3
"""Charge a scalar-only five-level actual-S4 nested norm program."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.scalar_only_nested_norm_slp_probe.r79.v1"
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_WORKSPACE_CAP_EXPONENT = 5 / 4

R78_REPORT = pathlib.Path(
    "p1553_actual_s6_fermat_tensor_train_probe_report_r78.json"
)
R78_REPORT_SHA256 = (
    "e590002c433c6504725d5ab7ff1dba97ad8c15400bf0117846742da7359c5e60"
)
R78_GRAMMAR = pathlib.Path(
    "frozen_nonlinear_nested_resultant_functional_grammar.json"
)
R78_GRAMMAR_SHA256 = (
    "320c0ae21cbb56451c1dcd4e9f96e53bfd1cb87d07d7a7fc6d6402e835d107f3"
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


R78 = load_module(
    "p1553_actual_s6_fermat_tensor_train_probe_r78.py",
    "p1553_r78_for_r79",
)
Point = tuple[int, int] | None


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bound_r78_report() -> dict[str, Any]:
    if sha256_file(R78_REPORT) != R78_REPORT_SHA256:
        raise AssertionError("R78 report hash mismatch")
    return json.loads(R78_REPORT.read_text(encoding="utf-8"))


def scalar_product(values: Sequence[int], modulus: int) -> int:
    product = 1
    for value in values:
        product = product * value % modulus
    return product


def fermat_zero_indicators(
    values: Sequence[int],
    modulus: int,
) -> list[int]:
    return [
        (1 - pow(value, modulus - 1, modulus)) % modulus
        for value in values
    ]


def flat_source(indicators: Sequence[int], size: int) -> list[int] | None:
    try:
        flat_index = indicators.index(1)
    except ValueError:
        return None
    output = [0] * 5
    for mode in range(4, -1, -1):
        output[mode] = flat_index % size
        flat_index //= size
    return output


def direct_scalar_control(
    curve: dict[str, Any],
    size: int = 3,
) -> dict[str, Any]:
    decks, targets = R78.R72.public_decks_and_targets(curve)
    target_rows = []
    for target_spec in targets:
        values = R78.tensor_values(
            curve,
            decks,
            target_spec["point"],
            size,
        )
        indicators = fermat_zero_indicators(
            values,
            curve["field_prime"],
        )
        source = flat_source(indicators, size)
        target_rows.append(
            {
                "target_id": target_spec["target_id"],
                "deck_size": size,
                "leaf_count": len(values),
                "root_scalar_product_is_zero": (
                    scalar_product(values, curve["field_prime"]) == 0
                ),
                "exact_fermat_zero_count": sum(indicators),
                "source_indices": source,
                "source_relation_verified": R78.source_verifies(
                    source,
                    decks,
                    target_spec["point"],
                    curve,
                ),
            }
        )
    return {
        "family_id": curve["family_id"],
        "targets": target_rows,
        "all_root_products_match_counts": all(
            row["root_scalar_product_is_zero"]
            == (row["exact_fermat_zero_count"] > 0)
            for row in target_rows
        ),
    }


def occurrence_semantics_control() -> dict[str, Any]:
    prime = 101
    records = [
        {"occurrence_id": "duplicate-a", "resultant": 0, "gcd_degree": 2},
        {"occurrence_id": "duplicate-b", "resultant": 0, "gcd_degree": 1},
        {"occurrence_id": "nonroot", "resultant": 7, "gcd_degree": 0},
    ]
    indicators = fermat_zero_indicators(
        [record["resultant"] for record in records],
        prime,
    )
    return {
        "records": records,
        "indicators": indicators,
        "tuple_occurrence_count": sum(indicators),
        "gcd_degree_sum": sum(record["gcd_degree"] for record in records),
        "duplicates_preserved": indicators[:2] == [1, 1],
        "multiple_common_roots_count_tuple_once": (
            indicators[0] == 1 and records[0]["gcd_degree"] == 2
        ),
    }


def scalar_tree_counts(size: int) -> dict[str, Any]:
    level_node_counts = {
        f"fixed_prefix_length_{prefix_length}": size**prefix_length
        for prefix_length in range(6)
    }
    leaf_count = size**5
    cached_node_count = sum(level_node_counts.values())
    return {
        "deck_size": size,
        "level_node_counts": level_node_counts,
        "leaf_s6_resultant_evaluation_count": leaf_count,
        "scalar_product_multiplication_count": leaf_count - 1,
        "cached_scalar_node_count": cached_node_count,
        "streaming_live_scalar_count_upper_bound": 6,
    }


def first_mode_dyadic_transcript(
    count: int,
    source: Sequence[int] | None,
    size: int,
) -> dict[str, Any]:
    leaf_counts = [0] * size
    if source is not None:
        leaf_counts[source[0]] = count
    conservation = []

    def visit(start: int, stop: int) -> dict[str, Any]:
        node_count = sum(leaf_counts[start:stop])
        row = {"interval": [start, stop], "count": node_count}
        if stop - start > 1:
            middle = start + (stop - start) // 2
            left = visit(start, middle)
            right = visit(middle, stop)
            row["children"] = [left, right]
            conservation.append(
                node_count == left["count"] + right["count"]
            )
        return row

    root = visit(0, size)
    return {
        "root": root,
        "all_parent_counts_equal_child_sums": all(conservation),
        "source_first_mode_index": None if source is None else source[0],
    }


def r78_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"family_id": family["family_id"], **row}
        for family in report["families"]
        for row in family["targets"]
    ]


def build_bundle() -> dict[str, dict[str, Any]]:
    report78 = read_bound_r78_report()
    rows78 = r78_rows(report78)
    direct_controls = [
        direct_scalar_control(dict(curve))
        for curve in R78.R72.CURVES
    ]
    occurrence_control = occurrence_semantics_control()
    grammar = {
        "schema": "p1553.frozen_scalar_only_nested_norm_slp.r79.v1",
        "grammar_id": "actual_s4_five_level_scalar_product_norm_v1",
        "leaf": (
            "scalar Res_z(S4(x1,x2,x3,z),S4(xR,x4,x5,z))"
        ),
        "recursive_node": (
            "scalar product over one frozen deck coordinate"
        ),
        "root": "one scalar product over all B^5 tuple occurrences",
        "exact_count": (
            "sum over leaves of 1-resultant^(p-1), valid as an integer "
            "on the frozen B^5<p range"
        ),
        "source": "first zero leaf plus five occurrence indices",
        "dyadic_children": "first-mode interval sums of leaf indicators",
        "all_nodes_scalar": True,
        "norm_is_not_a_primitive_oracle": True,
        "frozen_before_target_outcomes": True,
    }
    scalar_replay = {
        "schema": "p1553.actual_s4_scalar_zero_count_replay.r79.v1",
        "direct_controls": direct_controls,
        "bound_r78_instance_count": len(rows78),
        "all_bound_blind_counts_zero": all(
            row["raw_s6_zero_occurrence_count"] == 0
            for row in rows78
            if row["target_id"] == "blind_hash_target"
        ),
        "all_bound_forced_counts_one": all(
            row["raw_s6_zero_occurrence_count"] == 1
            for row in rows78
            if row["target_id"] == "forced_positive_target"
        ),
        "all_direct_root_products_match_counts": all(
            control["all_root_products_match_counts"]
            for control in direct_controls
        ),
        "occurrence_semantics_control": occurrence_control,
    }
    transcripts = [
        {
            "family_id": row["family_id"],
            "target_id": row["target_id"],
            "deck_size": row["deck_size"],
            "exact_count": row["raw_s6_zero_occurrence_count"],
            "source_indices": row["source_indices"],
            "source_relation_verified": row["source_relation_verified"],
            "dyadic": first_mode_dyadic_transcript(
                row["raw_s6_zero_occurrence_count"],
                row["source_indices"],
                row["deck_size"],
            ),
        }
        for row in rows78
    ]
    source_transcript = {
        "schema": "p1553.rectangle_source_dyadic_transcript.r79.v1",
        "rows": transcripts,
        "all_source_rows_verified": all(
            row["source_relation_verified"]
            for row in transcripts
            if row["source_indices"] is not None
        ),
        "all_dyadic_rows_conserve": all(
            row["dyadic"]["all_parent_counts_equal_child_sums"]
            for row in transcripts
        ),
    }
    per_size = []
    for size in sorted({row["deck_size"] for row in rows78}):
        counts = scalar_tree_counts(size)
        per_size.append(
            {
                **counts,
                "blind_failed_zero_leaf_floor": size**5,
                "forced_exact_count_leaf_floor": size**5,
                "streaming_workspace_exponent_B": 0.0,
                "cached_tree_state_exponent_B": 5.0,
                "exact_count_work_exponent_B": 5.0,
            }
        )
    cost_ledger = {
        "schema": "p1553.per_node_state_work_failed_zero.r79.v1",
        "setup_state_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
        "online_workspace_cap_exponent_B": ONLINE_WORKSPACE_CAP_EXPONENT,
        "per_size": per_size,
        "asymptotic": {
            "leaf_s6_resultant_work_exponent_B": 5.0,
            "scalar_product_work_exponent_B": 5.0,
            "fermat_exact_count_work_exponent_B": 5.0,
            "streaming_workspace_exponent_B": 0.0,
            "cached_source_tree_state_exponent_B": 5.0,
            "blind_failed_zero_certificate_work_exponent_B": 5.0,
            "forced_exact_count_work_exponent_B": 5.0,
            "field_bitlength_factor_suppressed": True,
            "lane_inside_caps": False,
        },
        "charging_rule": (
            "A NORM node expands to its scalar child calls and multiplications; "
            "it receives no unit-cost black-box credit."
        ),
    }
    report = {
        "schema": SCHEMA,
        "classification": (
            "SCALAR_ONLY_LEAF_NESTED_NORM_HAS_B5_FAILED_ZERO_WORK"
        ),
        "source_bindings": {
            "r78_actual_s6_tt_control": {
                "path": str(R78_REPORT),
                "sha256": R78_REPORT_SHA256,
            },
            "r78_frozen_grammar": {
                "path": str(R78_GRAMMAR),
                "sha256": R78_GRAMMAR_SHA256,
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
        "direct_controls": direct_controls,
        "aggregate": {
            "bound_instance_count": len(rows78),
            "direct_control_family_count": len(direct_controls),
            "all_direct_root_products_match_counts": scalar_replay[
                "all_direct_root_products_match_counts"
            ],
            "all_bound_blind_counts_zero": scalar_replay[
                "all_bound_blind_counts_zero"
            ],
            "all_bound_forced_counts_one": scalar_replay[
                "all_bound_forced_counts_one"
            ],
            "all_sources_verified": source_transcript[
                "all_source_rows_verified"
            ],
            "all_dyadic_counts_conserve": source_transcript[
                "all_dyadic_rows_conserve"
            ],
            "duplicate_and_multiple_root_semantics_pass": (
                occurrence_control["duplicates_preserved"]
                and occurrence_control[
                    "multiple_common_roots_count_tuple_once"
                ]
            ),
            "maximum_leaf_count": max(
                row["leaf_s6_resultant_evaluation_count"]
                for row in per_size
            ),
            "maximum_cached_scalar_node_count": max(
                row["cached_scalar_node_count"] for row in per_size
            ),
        },
        "side_artifacts": {
            "frozen_slp": "frozen_scalar_only_nested_norm_slp.json",
            "scalar_replay": "actual_s4_scalar_zero_count_replay.json",
            "source_transcript": (
                "rectangle_source_and_dyadic_transcript.json"
            ),
            "cost_ledger": (
                "per_node_state_work_and_failed_zero_ledger.json"
            ),
        },
        "cost_ledger": cost_ledger,
        "admission": {
            "passed_obligation_count": 8,
            "obligation_count": 12,
            "lane_admitted": False,
            "failures": [
                "blind zero certification evaluates B^5 actual S6 leaves",
                "exact counting evaluates and Fermat-projects B^5 leaves",
                "caching scalar norm nodes uses B^5 state",
                "factor-log solve and identical fresh-target descent are unsupplied",
            ],
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "scope_boundary": (
            "This rejects only the explicit scalar-leaf product/norm SLP. It "
            "does not reject a batched norm-node compiler that proves a "
            "sub-B^5 arithmetic implementation from compact deck polynomials "
            "without hiding forbidden intermediate state."
        ),
        "next_action": (
            "Freeze one batched nested-norm node compiler from unary deck "
            "polynomials to scalar output. Expand every resultant, "
            "subproduct, remainder, transposed, and source operation into "
            "field-operation and state receipts; require a strict improvement "
            "over B^5 while preserving R76 zero/count/source/children before "
            "testing the direct caps."
        ),
        "disposition": (
            "REJECT_SCALAR_ONLY_LEAF_PRODUCT_NORM_SLP_ONLY__"
            "ACTUAL_S4_BY_S4_LEAVES__FOUR_DIRECT_STANDARD_CURVE_CONTROLS__"
            "R78_32_INSTANCE_BINDING__BLIND_ZERO__FORCED_COUNT_SOURCE__"
            "DUPLICATE_AND_MULTIPLE_ROOT_SEMANTICS__DYADIC_CHILDREN__"
            "STREAMING_WORKSPACE_CONSTANT_BUT_FAILED_ZERO_WORK_B5__"
            "CACHED_STATE_B5__BATCHED_NORM_COMPILER_OPEN__NO_FACTOR_LOGS__"
            "NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "grammar": grammar,
        "scalar_replay": scalar_replay,
        "source_transcript": source_transcript,
        "cost_ledger": cost_ledger,
    }


def build_report() -> dict[str, Any]:
    return build_bundle()["report"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_scalar_only_nested_norm_slp_probe_report_r79.json"
        ),
    )
    parser.add_argument(
        "--slp-output",
        type=pathlib.Path,
        default=pathlib.Path("frozen_scalar_only_nested_norm_slp.json"),
    )
    parser.add_argument(
        "--scalar-replay-output",
        type=pathlib.Path,
        default=pathlib.Path("actual_s4_scalar_zero_count_replay.json"),
    )
    parser.add_argument(
        "--source-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "rectangle_source_and_dyadic_transcript.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "per_node_state_work_and_failed_zero_ledger.json"
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
    write_json(args.slp_output, bundle["grammar"])
    write_json(args.scalar_replay_output, bundle["scalar_replay"])
    write_json(args.source_output, bundle["source_transcript"])
    write_json(args.cost_output, bundle["cost_ledger"])
    aggregate = bundle["report"]["aggregate"]
    print(
        f"bound_instances={aggregate['bound_instance_count']} "
        f"direct_exact={aggregate['all_direct_root_products_match_counts']} "
        f"sources_verified={aggregate['all_sources_verified']} "
        f"lane_admitted={bundle['report']['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
