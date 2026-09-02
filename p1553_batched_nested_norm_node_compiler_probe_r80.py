#!/usr/bin/env python3
"""Compile actual-S4 leaf norms into batched product trees and one gcd."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.batched_nested_norm_node_compiler_probe.r80.v1"
PREFIX_SIZES = (3, 4, 5)
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_WORKSPACE_CAP_EXPONENT = 5 / 4

R79_REPORT = pathlib.Path(
    "p1553_scalar_only_nested_norm_slp_probe_report_r79.json"
)
R79_REPORT_SHA256 = (
    "b5bd22b7256d80f5fa475517699c42f2201a1c8c6209718ea38ab832146a8929"
)
R76_REPORT = pathlib.Path(
    "p1553_s6_subset_incidence_mobius_probe_report_r76.json"
)
R76_REPORT_SHA256 = (
    "de41d1618bc71c46f700bfead0ed72c5ac0b29a89da3b5c32534a15314ef4c93"
)
R3_REGISTRY = pathlib.Path("p1553_r31_artifact_index_README.md")
R3_REGISTRY_SHA256 = (
    "0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f"
)
R3_GATE_SHA256 = (
    "b2ee5934e295ab1f0d6b43452898e520d0cb18e718a8f5865694b25909b0df5e"
)


def load_module(path: str, module_name: str) -> Any:
    module_path = pathlib.Path(__file__).with_name(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R76 = load_module(
    "p1553_s6_subset_incidence_mobius_probe_r76.py",
    "p1553_r76_for_r80",
)
R73 = R76.R74.R73
Point = tuple[int, int] | None
Occurrence = R76.Occurrence


def factor_polynomial(
    endpoint: R76.R74.EndpointKey,
    modulus: int,
) -> list[int]:
    return R73.polynomial_from_roots(endpoint[1], modulus)


def occurrence_factors(
    occurrences: Sequence[Occurrence],
    modulus: int,
) -> list[dict[str, Any]]:
    return [
        {
            "indices": indices,
            "endpoint": endpoint,
            "polynomial": factor_polynomial(endpoint, modulus),
            "identity_count": int(endpoint[0]),
        }
        for indices, endpoint in occurrences
    ]


def build_product_tree(
    factors: Sequence[dict[str, Any]],
    modulus: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    receipts = {
        "leaf_count": len(factors),
        "internal_node_count": 0,
        "naive_coefficient_product_count": 0,
        "stored_coefficient_count": 0,
    }

    def build(start: int, stop: int) -> dict[str, Any]:
        if stop - start == 1:
            factor = factors[start]
            polynomial = factor["polynomial"]
            receipts["stored_coefficient_count"] += len(polynomial)
            return {
                "start": start,
                "stop": stop,
                "polynomial": polynomial,
                "identity_count": factor["identity_count"],
                "leaf_index": start,
            }
        middle = start + (stop - start) // 2
        left = build(start, middle)
        right = build(middle, stop)
        receipts["internal_node_count"] += 1
        receipts["naive_coefficient_product_count"] += (
            len(left["polynomial"]) * len(right["polynomial"])
        )
        polynomial = R73.polynomial_multiply(
            left["polynomial"],
            right["polynomial"],
            modulus,
        )
        receipts["stored_coefficient_count"] += len(polynomial)
        return {
            "start": start,
            "stop": stop,
            "polynomial": polynomial,
            "identity_count": (
                left["identity_count"] + right["identity_count"]
            ),
            "left": left,
            "right": right,
        }

    return build(0, len(factors)), receipts


def polynomial_degree(polynomial: Sequence[int]) -> int:
    return len(R73.trim(list(polynomial))) - 1


def common_support_degree(
    left: dict[str, Any],
    right: dict[str, Any],
    modulus: int,
) -> int:
    common = R73.polynomial_gcd(
        left["polynomial"],
        right["polynomial"],
        modulus,
    )
    identity = min(left["identity_count"], right["identity_count"])
    return polynomial_degree(common) + identity


def descend_intersecting_leaf(
    tree: dict[str, Any],
    other: dict[str, Any],
    modulus: int,
) -> int | None:
    if common_support_degree(tree, other, modulus) == 0:
        return None
    node = tree
    while "leaf_index" not in node:
        left = node["left"]
        if common_support_degree(left, other, modulus) > 0:
            node = left
        else:
            node = node["right"]
    return node["leaf_index"]


def leaf_as_tree(factor: dict[str, Any]) -> dict[str, Any]:
    return {
        "polynomial": factor["polynomial"],
        "identity_count": factor["identity_count"],
    }


def synthetic_gcd_count_control() -> dict[str, Any]:
    left_sets = [
        (("x", 1), ("x", 2)),
        (("x", 1), ("x", 2)),
    ]
    right_sets = [
        (("x", 1), ("x", 2)),
        (("x", 2), ("x", 3)),
    ]
    left_histogram = R76.generic_histogram(left_sets)
    right_histogram = R76.generic_histogram(right_sets)
    tuple_count = R76.mobius_incidence_count(
        left_histogram,
        right_histogram,
    )
    left_multiplicity = {1: 2, 2: 2}
    right_multiplicity = {1: 1, 2: 2, 3: 1}
    gcd_degree = sum(
        min(multiplicity, right_multiplicity.get(root, 0))
        for root, multiplicity in left_multiplicity.items()
    )
    return {
        "mobius_tuple_count": tuple_count,
        "product_polynomial_gcd_degree": gcd_degree,
        "naive_singleton_incidence_count": 6,
        "gcd_degree_is_not_tuple_count": gcd_degree != tuple_count,
        "mobius_correction_required": True,
    }


def target_profile(
    target_spec: dict[str, Any],
    prefix_occurrences: Sequence[Occurrence],
    prefix_factors: Sequence[dict[str, Any]],
    prefix_tree: dict[str, Any],
    prefix_histogram: R76.Counter[R76.SubsetKey],
    prefix_sources: dict[R76.SubsetKey, tuple[int, ...]],
    decks: list[list[Point]],
    curve: dict[str, Any],
    size: int,
) -> dict[str, Any]:
    modulus = curve["field_prime"]
    suffix_occurrences = R76.suffix_occurrences(
        decks,
        target_spec["point"],
        curve,
        size,
    )
    suffix_factors = occurrence_factors(suffix_occurrences, modulus)
    suffix_tree, suffix_receipts = build_product_tree(
        suffix_factors,
        modulus,
    )
    gcd_degree = common_support_degree(
        prefix_tree,
        suffix_tree,
        modulus,
    )
    suffix_histogram, suffix_sources, _ = R76.subset_histogram(
        suffix_occurrences
    )
    tuple_count = R76.mobius_incidence_count(
        prefix_histogram,
        suffix_histogram,
    )
    prefix_leaf = descend_intersecting_leaf(
        prefix_tree,
        suffix_tree,
        modulus,
    )
    suffix_leaf = None
    source_indices = None
    source_verified = None
    if prefix_leaf is not None:
        suffix_leaf = descend_intersecting_leaf(
            suffix_tree,
            leaf_as_tree(prefix_factors[prefix_leaf]),
            modulus,
        )
    if prefix_leaf is not None and suffix_leaf is not None:
        source_indices = [
            *prefix_factors[prefix_leaf]["indices"],
            *suffix_factors[suffix_leaf]["indices"],
        ]
        source_verified = R76.R74.R72.signed_relation_exists(
            tuple(
                decks[mode][index]
                for mode, index in enumerate(source_indices)
            ),
            target_spec["point"],
            curve,
        )
    histogram_source = R76.recover_source(
        prefix_histogram,
        prefix_sources,
        suffix_histogram,
        suffix_sources,
    )
    return {
        "target_id": target_spec["target_id"],
        "suffix_occurrence_count": len(suffix_occurrences),
        "suffix_root_product_degree": polynomial_degree(
            suffix_tree["polynomial"]
        ),
        "suffix_tree_receipts": suffix_receipts,
        "product_polynomial_gcd_degree": gcd_degree,
        "product_polynomial_detects_existence": (
            (gcd_degree > 0) == (tuple_count > 0)
        ),
        "mobius_exact_tuple_count": tuple_count,
        "gcd_degree_equals_tuple_count_on_frozen_instance": (
            gcd_degree == tuple_count
        ),
        "product_tree_source_indices": source_indices,
        "product_tree_source_relation_verified": source_verified,
        "histogram_source_present": histogram_source is not None,
    }


def probe_curve(
    curve: dict[str, Any],
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, Any]:
    decks, targets = R76.R74.R72.public_decks_and_targets(curve)
    rows = []
    for size in prefix_sizes:
        prefix_occurrences = R76.triple_occurrences(decks, curve, size)
        prefix_factors = occurrence_factors(
            prefix_occurrences,
            curve["field_prime"],
        )
        prefix_tree, prefix_receipts = build_product_tree(
            prefix_factors,
            curve["field_prime"],
        )
        prefix_histogram, prefix_sources, contributions = (
            R76.subset_histogram(prefix_occurrences)
        )
        rows.append(
            {
                "deck_size": size,
                "prefix_occurrence_count": len(prefix_occurrences),
                "prefix_root_product_degree": polynomial_degree(
                    prefix_tree["polynomial"]
                ),
                "prefix_tree_receipts": prefix_receipts,
                "prefix_subset_histogram_key_count": len(
                    prefix_histogram
                ),
                "prefix_subset_contribution_count": contributions,
                "targets": [
                    target_profile(
                        target,
                        prefix_occurrences,
                        prefix_factors,
                        prefix_tree,
                        prefix_histogram,
                        prefix_sources,
                        decks,
                        curve,
                        size,
                    )
                    for target in targets
                ],
            }
        )
    return {
        "family_id": curve["family_id"],
        "scalar_labels_consumed": False,
        "prefixes": rows,
    }


def build_bundle(
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, dict[str, Any]]:
    prefix_sizes = tuple(prefix_sizes)
    families = [
        probe_curve(dict(curve), prefix_sizes=prefix_sizes)
        for curve in R76.R74.R72.CURVES
    ]
    prefix_rows = [
        row for family in families for row in family["prefixes"]
    ]
    target_rows = [
        target
        for prefix in prefix_rows
        for target in prefix["targets"]
    ]
    synthetic = synthetic_gcd_count_control()
    compiler = {
        "schema": "p1553.frozen_batched_norm_node_compiler.r80.v1",
        "compiler_id": "s4_factor_subproduct_gcd_mobius_v1",
        "prefix": (
            "balanced product tree of B^3 squarefree S4 endpoint factors"
        ),
        "suffix": (
            "balanced product tree of B^2 target-specialized S4 endpoint factors"
        ),
        "existence": "one finite-root polynomial gcd plus identity tag",
        "source": "gcd-guided descent in prefix and suffix factor trees",
        "exact_tuple_count": (
            "R76 alternating endpoint-subset histograms; gcd degree alone "
            "is not occurrence count"
        ),
        "fast_multiplication_model": "M(n)=n^(1+o(1)) field operations",
        "norm_or_resultant_unit_cost": False,
        "frozen_before_target_outcomes": True,
    }
    receipts = {
        "schema": "p1553.resultant_subproduct_remainder_receipts.r80.v1",
        "families": families,
        "all_product_gcd_existence_checks_exact": all(
            row["product_polynomial_detects_existence"]
            for row in target_rows
        ),
        "all_positive_product_tree_sources_verified": all(
            row["product_tree_source_relation_verified"]
            for row in target_rows
            if row["mobius_exact_tuple_count"] > 0
        ),
    }
    replay = {
        "schema": "p1553.r76_batched_zero_count_source_replay.r80.v1",
        "synthetic_gcd_count_control": synthetic,
        "rows": [
            {
                "family_id": family["family_id"],
                "prefixes": [
                    {
                        "deck_size": prefix["deck_size"],
                        "targets": [
                            {
                                "target_id": target["target_id"],
                                "gcd_degree": target[
                                    "product_polynomial_gcd_degree"
                                ],
                                "tuple_count": target[
                                    "mobius_exact_tuple_count"
                                ],
                                "source_verified": target[
                                    "product_tree_source_relation_verified"
                                ],
                            }
                            for target in prefix["targets"]
                        ],
                    }
                    for prefix in family["prefixes"]
                ],
            }
            for family in families
        ],
        "all_blind_counts_zero": all(
            row["mobius_exact_tuple_count"] == 0
            for row in target_rows
            if row["target_id"] == "blind_hash_target"
        ),
        "all_forced_counts_one": all(
            row["mobius_exact_tuple_count"] == 1
            for row in target_rows
            if row["target_id"] == "forced_positive_target"
        ),
        "all_forced_sources_verified": all(
            row["product_tree_source_relation_verified"]
            for row in target_rows
            if row["target_id"] == "forced_positive_target"
        ),
    }
    cost_ledger = {
        "schema": "p1553.compiled_node_field_state_cost.r80.v1",
        "setup_state_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
        "online_work_cap_exponent_B": ONLINE_WORKSPACE_CAP_EXPONENT,
        "online_workspace_cap_exponent_B": ONLINE_WORKSPACE_CAP_EXPONENT,
        "compiled_fast_arithmetic": {
            "prefix_factor_enumeration_exponent_B": 3.0,
            "prefix_product_tree_work_exponent_B": 3.0,
            "prefix_product_tree_state_exponent_B": 3.0,
            "suffix_factor_enumeration_exponent_B": 2.0,
            "suffix_product_tree_work_exponent_B": 2.0,
            "fresh_gcd_and_remainder_work_exponent_B": 3.0,
            "fresh_source_backtracking_work_exponent_B": 3.0,
            "r76_prefix_correction_state_exponent_B": 3.0,
            "r76_fresh_target_correction_work_exponent_B": 2.0,
            "polylog_factors_suppressed": True,
            "strict_improvement_over_leaf_B5": True,
            "lane_inside_caps": False,
        },
        "best_exact_integrated_path": {
            "setup_state_exponent_B": 3.0,
            "setup_work_exponent_B": 3.0,
            "fresh_target_work_exponent_B": 3.0,
            "fresh_target_workspace_exponent_B": 2.0,
            "uses_product_gcd_for_existence_source": True,
            "uses_r76_histograms_for_exact_tuple_count": True,
        },
        "failure_reasons": [
            "prefix product and exact-correction advice exceed B^(9/4)",
            "fresh degree-B^3 remainder/gcd work exceeds B^(5/4)",
            "fresh suffix and correction workspace exceed B^(5/4)",
        ],
    }
    report = {
        "schema": SCHEMA,
        "classification": (
            "BATCHED_S4_PRODUCT_GCD_IMPROVES_B5_TO_B3_BUT_MISSES_CAPS"
        ),
        "source_bindings": {
            "r79_scalar_leaf_slp": {
                "path": str(R79_REPORT),
                "sha256": R79_REPORT_SHA256,
            },
            "r76_exact_incidence_control": {
                "path": str(R76_REPORT),
                "sha256": R76_REPORT_SHA256,
            },
            "r3_query2p1_registry": {
                "path": str(R3_REGISTRY),
                "sha256": R3_REGISTRY_SHA256,
                "bound_gate_sha256": R3_GATE_SHA256,
            },
        },
        "compiler": compiler,
        "families": families,
        "aggregate": {
            "family_count": len(families),
            "prefix_instance_count": len(prefix_rows),
            "target_instance_count": len(target_rows),
            "all_product_gcd_existence_checks_exact": receipts[
                "all_product_gcd_existence_checks_exact"
            ],
            "all_positive_product_tree_sources_verified": receipts[
                "all_positive_product_tree_sources_verified"
            ],
            "all_blind_counts_zero": replay["all_blind_counts_zero"],
            "all_forced_counts_one": replay["all_forced_counts_one"],
            "synthetic_gcd_degree_differs_from_tuple_count": synthetic[
                "gcd_degree_is_not_tuple_count"
            ],
            "maximum_prefix_product_degree": max(
                row["prefix_root_product_degree"] for row in prefix_rows
            ),
            "maximum_suffix_product_degree": max(
                target["suffix_root_product_degree"]
                for target in target_rows
            ),
        },
        "side_artifacts": {
            "compiler": "frozen_batched_norm_node_compiler.json",
            "arithmetic_receipts": (
                "resultant_subproduct_remainder_receipts.json"
            ),
            "exact_replay": "r76_batched_zero_count_source_replay.json",
            "cost_ledger": "compiled_node_field_state_cost_ledger.json",
        },
        "cost_ledger": cost_ledger,
        "admission": {
            "passed_obligation_count": 9,
            "obligation_count": 14,
            "lane_admitted": False,
            "failures": [
                "target-independent prefix product/correction state costs B^3",
                "fresh-target remainder/gcd work costs B^3",
                "fresh-target suffix/correction workspace costs B^2",
                "gcd degree alone is not exact tuple count",
                "factor-log solve and identical fresh-target descent are unsupplied",
            ],
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "scope_boundary": (
            "This implements the standard batched subproduct/gcd compiler and "
            "its R76 exact correction. It is not a lower bound against factor "
            "bases whose triple endpoint support has a new provable compressed "
            "representation below B^(9/4)."
        ),
        "next_action": (
            "Leave random hash decks and construct one scalar-blind structured "
            "factor-base geometry whose B^3 triple endpoint multiset has a "
            "proved representation of at most B^(9/4+o(1)), while fresh "
            "target-plus-pair queries and source return cost at most "
            "B^(5/4+o(1)). Require a prospective relation-density theorem, "
            "matched random controls, exact rank, factor logs, and identical "
            "target descent."
        ),
        "disposition": (
            "REJECT_STANDARD_BATCHED_S4_SUBPRODUCT_GCD_COMPILER_ONLY__"
            "FOUR_STANDARD_CURVES__B3_4_5__STRICT_B5_TO_B3_IMPROVEMENT__"
            "EXACT_EXISTENCE_AND_PRODUCT_TREE_SOURCE__GCD_DEGREE_NOT_TUPLE_"
            "COUNT__R76_MOBIUS_CORRECTION__SETUP_STATE_B3__FRESH_TARGET_WORK_"
            "B3_STATE_B2__"
            "STRUCTURED_FACTOR_BASE_GEOMETRY_OPEN__NO_FACTOR_LOGS__NO_DESCENT__"
            "NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "compiler": compiler,
        "receipts": receipts,
        "replay": replay,
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
            "p1553_batched_nested_norm_node_compiler_probe_report_r80.json"
        ),
    )
    parser.add_argument(
        "--compiler-output",
        type=pathlib.Path,
        default=pathlib.Path("frozen_batched_norm_node_compiler.json"),
    )
    parser.add_argument(
        "--receipts-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "resultant_subproduct_remainder_receipts.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path("r76_batched_zero_count_source_replay.json"),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path("compiled_node_field_state_cost_ledger.json"),
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
    write_json(args.compiler_output, bundle["compiler"])
    write_json(args.receipts_output, bundle["receipts"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.cost_output, bundle["cost_ledger"])
    aggregate = bundle["report"]["aggregate"]
    print(
        f"families={aggregate['family_count']} "
        f"gcd_exact={aggregate['all_product_gcd_existence_checks_exact']} "
        f"sources_verified="
        f"{aggregate['all_positive_product_tree_sources_verified']} "
        f"lane_admitted={bundle['report']['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
