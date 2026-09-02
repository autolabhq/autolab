#!/usr/bin/env python3
"""P1418 projective Kummer differential-state compiler and exact replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HOST_TASK_DIR = Path("/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
if str(HOST_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_TASK_DIR))

import low_term_total2_p1398_richer_rational_map_factor_base_generator_after_p1397 as p1398
import low_term_total2_p1407_coordinate_advice_compression_expansion_after_p1406 as p1407
import low_term_total2_p1408_rational_map_image_factor_bases_after_p1407 as p1408
import low_term_total2_p1416_coordinate_additive_energy_recursive_s3_after_p1415 as p1416


STATE_DIR = Path("ecdlp_index_calculus_state")
HOST_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
RESEARCH_DIR = Path("research")
SCHEMA = "ecdlp.p1418_projective_kummer_differential_state.v1"

DEFAULT_CONTRACT = STATE_DIR / "experiment_contract_p1418_projective_kummer_differential_state_after_p1417.md"
DEFAULT_P1416 = HOST_STATE_DIR / "p1416_coordinate_additive_energy_recursive_s3_after_p1415_probe.json"
DEFAULT_P1417 = HOST_STATE_DIR / "p1417_kummer_s3_quotient_resultant_after_p1416_probe.json"
DEFAULT_P1417_AUDIT = HOST_STATE_DIR / "p1417_kummer_s3_quotient_resultant_after_p1416_audit.json"
DEFAULT_OUT = STATE_DIR / "p1418_projective_kummer_differential_state_after_p1417_probe.json"
DEFAULT_NOTE = STATE_DIR / "p1418_projective_kummer_differential_state_after_p1417.md"
DEFAULT_RESULT = STATE_DIR / "p1418_projective_kummer_differential_state_after_p1417_restricted_result.md"

STRUCTURED_POLICIES = p1416.STRUCTURED_POLICIES
HASH_POLICIES = p1416.HASH_POLICIES
ALL_POLICIES = p1416.ALL_POLICIES


def int_value(value: Any) -> int:
    return p1416.int_value(value)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_preregistration_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    prefix = text.split("\n## Results\n", 1)[0].rstrip() + "\n"
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def ratio(value: int | float, denominator: int | float) -> float | None:
    return p1407.ratio(value, denominator)


def point_key(point: Any) -> Any:
    return p1416.point_key(point)


def point_json(point: Any) -> Any:
    return p1416.point_json(point)


def point_sort_key(point: Any) -> tuple[int, int, int]:
    return p1416.point_sort_key(point)


def x_key(point: Any) -> Any:
    return None if point is None else int_value(point[0])


def key_json(key: Any) -> Any:
    if isinstance(key, tuple):
        return [key_json(value) for value in key]
    return key


def stable_key(key: Any) -> str:
    return json.dumps(key_json(key), sort_keys=True, separators=(",", ":"))


def directory_hash(directory: dict[Any, list[tuple[int, int, int]]]) -> str:
    return canonical_hash([
        [key_json(key), [list(row) for row in sorted(set(witnesses))]]
        for key, witnesses in sorted(directory.items(), key=lambda item: stable_key(item[0]))
    ])


def kummer(point: Any) -> tuple[int, int]:
    return (1, 0) if point is None else (int_value(point[0]), 1)


def curve_rhs(x: int, a: int, b: int, p: int) -> int:
    return (x * x % p * x + a * x + b) % p


def projective_xadd(
    left: Any,
    right: Any,
    difference: Any,
    a: int,
    b: int,
    p: int,
) -> tuple[tuple[int, int], str, dict[str, int]]:
    if left is None:
        return kummer(right), "left_infinity", {"field_moves": 1}
    if right is None:
        return kummer(left), "right_infinity", {"field_moves": 1}
    x_left = int_value(left[0]) % p
    x_right = int_value(right[0]) % p
    if x_left == x_right:
        if difference is None:
            rhs = curve_rhs(x_left, a, b, p)
            numerator = ((3 * x_left * x_left + a) ** 2 - 8 * x_left * rhs) % p
            denominator = (4 * rhs) % p
            if denominator == 0:
                return (1, 0), "doubling_to_infinity", {
                    "field_multiplications": 4,
                    "field_squarings": 2,
                    "field_additions_or_subtractions": 5,
                }
            return (numerator, denominator), "doubling", {
                "field_multiplications": 4,
                "field_squarings": 2,
                "field_additions_or_subtractions": 5,
            }
        return (1, 0), "inverse_to_infinity", {
            "field_comparisons": 1,
        }

    x_difference, z_difference = kummer(difference)
    delta = (x_right - x_left) % p
    delta_squared = delta * delta % p
    left_rhs = curve_rhs(x_left, a, b, p)
    right_rhs = curve_rhs(x_right, a, b, p)
    middle = (
        2 * (left_rhs + right_rhs) - 2 * (x_left + x_right) * delta_squared
    ) % p
    numerator = (middle * z_difference - x_difference * delta_squared) % p
    denominator = delta_squared * z_difference % p
    return (numerator, denominator), "general", {
        "field_multiplications": 8,
        "field_squarings": 3,
        "field_additions_or_subtractions": 10,
    }


def add_operation_counts(total: Counter[str], profile: dict[str, int]) -> None:
    total.update({key: int_value(value) for key, value in profile.items()})


def batch_normalize(
    values: list[tuple[int, int]], p: int,
) -> tuple[list[int | None], dict[str, int]]:
    finite_indices = [index for index, (_, z_value) in enumerate(values) if z_value % p]
    output: list[int | None] = [None] * len(values)
    if not finite_indices:
        return output, {
            "batch_inversions": 0,
            "batch_multiplications": 0,
            "normalized_finite_count": 0,
            "projective_infinity_count": len(values),
        }
    prefixes = []
    accumulator = 1
    for index in finite_indices:
        prefixes.append(accumulator)
        accumulator = accumulator * (values[index][1] % p) % p
    inverse = pow(accumulator, -1, p)
    inverse_values: dict[int, int] = {}
    for offset in range(len(finite_indices) - 1, -1, -1):
        index = finite_indices[offset]
        z_value = values[index][1] % p
        inverse_values[index] = inverse * prefixes[offset] % p
        inverse = inverse * z_value % p
    for index in finite_indices:
        output[index] = values[index][0] * inverse_values[index] % p
    count = len(finite_indices)
    return output, {
        "batch_inversions": 1,
        "batch_multiplications": 4 * count,
        "normalized_finite_count": count,
        "projective_infinity_count": len(values) - count,
    }


def directory_memory(
    unique_keys: int,
    witness_count: int,
    key_field_elements: int,
    tag_bytes: int,
    field_bytes: int,
    index_bytes: int,
) -> dict[str, int]:
    pointer_bytes = max(1, math.ceil(max(1, witness_count).bit_length() / 8))
    key_bytes = unique_keys * (key_field_elements * field_bytes + tag_bytes + pointer_bytes)
    witness_bytes = witness_count * 3 * index_bytes
    return {
        "unique_key_count": unique_keys,
        "witness_count": witness_count,
        "key_logical_bytes": key_bytes,
        "witness_logical_bytes": witness_bytes,
        "lossless_logical_bytes": key_bytes + witness_bytes,
        "one_witness_per_key_logical_bytes": key_bytes + unique_keys * 3 * index_bytes,
    }


def build_geometry(
    raw: Any,
    factors: list[Any],
    ainvs: list[int],
    p: int,
    order: int,
) -> dict[str, Any]:
    verifier = p1408.OperationCountingVerifier(raw)
    states = p1416.build_states(verifier, factors, ainvs, p, order)
    pair_table = states["pair_table"]
    a = int_value(ainvs[3]) % p
    b = int_value(ainvs[4]) % p
    before = verifier.snapshot()
    cache: dict[tuple[Any, int], dict[str, Any]] = {}
    source_records = []
    for pair_point in sorted(pair_table, key=point_sort_key):
        for left, middle in sorted(pair_table[pair_point]):
            for factor_index in range(len(factors)):
                oriented_source = (left, middle, factor_index)
                normalized_source = tuple(sorted(oriented_source))
                cache_key = (pair_point, factor_index)
                if cache_key not in cache:
                    factor = factors[factor_index]
                    difference = point_key(
                        verifier.add_points(pair_point, p1416.negate(factor, p), ainvs, p)
                    )
                    total = point_key(verifier.add_points(pair_point, factor, ainvs, p))
                    projective, branch, operations = projective_xadd(
                        pair_point, factor, difference, a, b, p
                    )
                    cache[cache_key] = {
                        "pair_point": pair_point,
                        "factor_index": factor_index,
                        "difference": difference,
                        "total": total,
                        "projective": projective,
                        "branch": branch,
                        "xadd_operations": operations,
                    }
                source_records.append({
                    "oriented_source": oriented_source,
                    "normalized_source": normalized_source,
                    "cache_key": cache_key,
                })
    construction_profile = p1416.profile_delta(before, verifier.snapshot())

    cache_keys = sorted(cache, key=lambda row: (point_sort_key(row[0]), row[1]))
    normalized, normalization_operations = batch_normalize(
        [cache[key]["projective"] for key in cache_keys], p
    )
    xadd_operations: Counter[str] = Counter()
    branch_cache_histogram: Counter[str] = Counter()
    xadd_failure_count = 0
    for index, key in enumerate(cache_keys):
        row = cache[key]
        row["normalized_output_x"] = normalized[index]
        add_operation_counts(xadd_operations, row["xadd_operations"])
        branch_cache_histogram[row["branch"]] += 1
        if normalized[index] != x_key(row["total"]):
            xadd_failure_count += 1
    add_operation_counts(xadd_operations, normalization_operations)

    difference_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    augmented_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    raw_projective_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    output_x_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    full_output_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    no_difference_outputs: dict[Any, set[Any]] = defaultdict(set)
    branch_source_histogram: Counter[str] = Counter()
    orientation_label_histogram: Counter[str] = Counter()
    for source_record in source_records:
        oriented_source = source_record["oriented_source"]
        normalized_source = source_record["normalized_source"]
        row = cache[source_record["cache_key"]]
        pair_point = row["pair_point"]
        factor = factors[row["factor_index"]]
        difference = row["difference"]
        total = row["total"]
        branch = row["branch"]
        difference_key = x_key(difference)
        augmented_key = (x_key(pair_point), x_key(factor), difference_key, branch)
        raw_projective_key = (*row["projective"], branch)
        output_key = row["normalized_output_x"]
        full_key = total
        no_difference_key = (x_key(pair_point), x_key(factor))
        difference_directory[difference_key].append(oriented_source)
        augmented_directory[augmented_key].append(oriented_source)
        raw_projective_directory[raw_projective_key].append(oriented_source)
        output_x_directory[output_key].append(oriented_source)
        full_output_directory[full_key].append(normalized_source)
        no_difference_outputs[no_difference_key].add(output_key)
        branch_source_histogram[branch] += 1
        if total is not None:
            orientation_label_histogram[str(int_value(total[1]) > p // 2)] += 1
        else:
            orientation_label_histogram["infinity"] += 1

    full_table = {
        point: sorted(set(witnesses)) for point, witnesses in full_output_directory.items()
    }
    expected_triple_count = math.comb(len(factors) + 2, 3)
    expected_oriented_state_count = len(factors) * math.comb(len(factors) + 1, 2)
    field_bytes = max(1, math.ceil(p.bit_length() / 8))
    index_bytes = max(1, math.ceil(max(1, len(factors)).bit_length() / 8))
    normalized_witness_count = sum(len(witnesses) for witnesses in full_table.values())
    oriented_state_count = len(source_records)
    memories = {
        "difference_x": directory_memory(
            len(difference_directory), oriented_state_count, 1, 1, field_bytes, index_bytes
        ),
        "augmented_kummer": directory_memory(
            len(augmented_directory), oriented_state_count, 3, 1, field_bytes, index_bytes
        ),
        "raw_projective_output": directory_memory(
            len(raw_projective_directory), oriented_state_count, 2, 1, field_bytes, index_bytes
        ),
        "output_x": directory_memory(
            len(output_x_directory), oriented_state_count, 1, 1, field_bytes, index_bytes
        ),
        "full_output": directory_memory(
            len(full_output_directory), normalized_witness_count, 2, 1, field_bytes, index_bytes
        ),
    }
    field_estimate = p1408.affine_field_operation_estimate(construction_profile)
    projective_weighted = (
        xadd_operations["field_multiplications"]
        + xadd_operations["batch_multiplications"]
        + xadd_operations["field_squarings"]
        + 80 * xadd_operations["batch_inversions"]
        + xadd_operations["field_additions_or_subtractions"] / 10
    )
    affine_weighted = (
        int_value(field_estimate.get("field_multiplications", 0))
        + int_value(field_estimate.get("field_squarings", 0))
        + 80 * int_value(field_estimate.get("field_inversions", 0))
        + int_value(field_estimate.get("field_additions_or_subtractions", 0)) / 10
    )
    ambiguous_no_difference = {
        key: values for key, values in no_difference_outputs.items() if len(values) > 1
    }
    pair_scans = len(pair_table)
    storage_entries = len(augmented_directory) + oriented_state_count
    return {
        "states": states,
        "full_triple_table": full_table,
        "summary": {
            "factor_base_size_B": len(factors),
            "pair_full_support_count": len(pair_table),
            "source_triple_count": normalized_witness_count,
            "expected_source_triple_count": expected_triple_count,
            "source_triple_count_exact": normalized_witness_count == expected_triple_count,
            "oriented_source_state_count": oriented_state_count,
            "expected_oriented_source_state_count": expected_oriented_state_count,
            "oriented_source_state_count_exact": (
                oriented_state_count == expected_oriented_state_count
            ),
            "pair_factor_cache_entry_count": len(cache),
            "difference_x_support_count": len(difference_directory),
            "augmented_kummer_support_count": len(augmented_directory),
            "raw_projective_output_support_count": len(raw_projective_directory),
            "output_x_support_count": len(output_x_directory),
            "full_output_support_count": len(full_output_directory),
            "no_difference_key_count": len(no_difference_outputs),
            "no_difference_ambiguous_key_count": len(ambiguous_no_difference),
            "no_difference_ambiguous_key_fraction": ratio(
                len(ambiguous_no_difference), len(no_difference_outputs)
            ),
            "no_difference_max_output_multiplicity": max(
                (len(values) for values in no_difference_outputs.values()), default=0
            ),
            "xadd_failure_count": xadd_failure_count,
            "branch_cache_histogram": dict(sorted(branch_cache_histogram.items())),
            "branch_source_histogram": dict(sorted(branch_source_histogram.items())),
            "orientation_label_histogram": dict(sorted(orientation_label_histogram.items())),
            "difference_directory_sha256": directory_hash(difference_directory),
            "augmented_directory_sha256": directory_hash(augmented_directory),
            "raw_projective_directory_sha256": directory_hash(raw_projective_directory),
            "output_x_directory_sha256": directory_hash(output_x_directory),
            "full_triple_state_sha256": p1416.support_hash(full_table),
            "xadd_operation_counts": dict(xadd_operations),
            "difference_and_sum_point_operation_profile": construction_profile,
            "difference_and_sum_affine_field_operation_estimate": field_estimate,
            "projective_xadd_weighted_field_operation_units": round(projective_weighted, 10),
            "difference_and_sum_weighted_field_operation_units": round(affine_weighted, 10),
            "total_construction_weighted_field_operation_units": round(
                projective_weighted + affine_weighted, 10
            ),
            "directory_memory": memories,
            "lossless_augmented_logical_bytes": memories["augmented_kummer"]["lossless_logical_bytes"],
            "one_witness_augmented_logical_bytes": memories["augmented_kummer"]["one_witness_per_key_logical_bytes"],
            "lossless_output_x_logical_bytes": memories["output_x"]["lossless_logical_bytes"],
            "one_witness_output_x_logical_bytes": memories["output_x"]["one_witness_per_key_logical_bytes"],
            "source_witness_symbolic_exponent_in_B": 3.0,
            "source_witness_symbolic_exponent_in_r": 0.6,
            "pair_scan_online_symbolic_exponent_in_B": 2.0,
            "pair_scan_online_symbolic_exponent_in_r": 0.4,
            "S_entries_T2_over_r": ratio(storage_entries * pair_scans * pair_scans, order),
        },
    }


def query_replay(
    raw: Any,
    states: dict[str, Any],
    full_triple_table: dict[Any, list[tuple[int, int, int]]],
    factors: list[Any],
    ainvs: list[int],
    p: int,
    query: dict[str, Any],
) -> dict[str, Any]:
    row_point = None if query["public_point"] is None else tuple(query["public_point"])
    compiled_states = {
        "pair_table": states["pair_table"],
        "triple_table": full_triple_table,
    }
    result = p1416.decompose_five(
        p1408.OperationCountingVerifier(raw), row_point, compiled_states, factors,
        ainvs, p, cap=1_000_000,
    )
    expected_hash = query["quotient"]["exact_tuples_sha256"]
    return {
        "kind": query["kind"],
        "index": query["index"],
        "public_point": query["public_point"],
        "tuple_count": result["all_valid_tuple_count"],
        "tuple_sha256": result["all_valid_tuple_sha256"],
        "p1417_tuple_sha256": expected_hash,
        "exact_match": result["all_valid_tuple_sha256"] == expected_hash,
        "invalid_witness_count": result["invalid_witness_count"],
        "pair_support_scans": result["pair_support_scans"],
        "x_false_positive_hits": result["x_false_positive_hits"],
    }


FIT_METRICS = (
    "source_triple_count",
    "oriented_source_state_count",
    "pair_factor_cache_entry_count",
    "difference_x_support_count",
    "augmented_kummer_support_count",
    "output_x_support_count",
    "lossless_augmented_logical_bytes",
    "one_witness_augmented_logical_bytes",
    "total_construction_weighted_field_operation_units",
)


def fit_exponent_in_B(rows: list[dict[str, Any]], metric: str) -> float | None:
    points = []
    for row in rows:
        b_size = float(row["factor_base_size_B"])
        value = float(row["geometry"][metric])
        if b_size > 1 and value > 0:
            points.append((math.log(b_size), math.log(value)))
    if len({x for x, _ in points}) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return round(
        sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator,
        10,
    )


def build_policy_record(
    raw: Any,
    source_curve: dict[str, Any],
    p1417_curve: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    p = int_value(source_curve["p"])
    order = int_value(source_curve["order"])
    ainvs = [0, 0, 0, int_value(source_curve["curve_a"]), int_value(source_curve["curve_b"])]
    source = source_curve["policies"][policy]
    factors = [tuple(point) for point in source["factor_base_points"]]
    geometry = build_geometry(raw, factors, ainvs, p, order)
    summary = geometry["summary"]
    expected_geometry = source["state_geometry"]
    triple_hash_match = summary["full_triple_state_sha256"] == expected_geometry["triple_state_sha256"]
    triple_support_match = summary["full_output_support_count"] == expected_geometry["triple_full_support_count"]
    query_rows = [
        query_replay(
            raw, geometry["states"], geometry["full_triple_table"], factors,
            ainvs, p, query,
        )
        for query in p1417_curve["policies"][policy]["queries"]
    ]
    reversed_geometry = build_geometry(raw, list(reversed(factors)), ainvs, p, order)["summary"]
    shuffle_metrics = (
        "source_triple_count", "oriented_source_state_count",
        "difference_x_support_count",
        "augmented_kummer_support_count", "output_x_support_count",
        "full_output_support_count", "no_difference_ambiguous_key_count",
    )
    shuffle_invariant = all(summary[key] == reversed_geometry[key] for key in shuffle_metrics)
    invalid = summary["xadd_failure_count"] + sum(
        int_value(row["invalid_witness_count"]) for row in query_rows
    )
    return {
        "factor_base_size_B": len(factors),
        "factor_base_sha256": source["factor_base_sha256"],
        "geometry": summary,
        "triple_state_hash_matches_p1416": triple_hash_match,
        "triple_support_matches_p1416": triple_support_match,
        "query_replays": query_rows,
        "query_replay_count": len(query_rows),
        "all_query_tuple_sets_match_p1417": all(row["exact_match"] for row in query_rows),
        "shuffle_control": {
            "checked_metrics": list(shuffle_metrics),
            "label_invariant": shuffle_invariant,
            "reversed": {key: reversed_geometry[key] for key in shuffle_metrics},
        },
        "invalid_witness_count": invalid,
        "validation": {
            "source_triple_count_exact": summary["source_triple_count_exact"],
            "oriented_source_state_count_exact": summary["oriented_source_state_count_exact"],
            "xadd_exact": summary["xadd_failure_count"] == 0,
            "p1416_triple_hash_exact": triple_hash_match,
            "p1416_triple_support_exact": triple_support_match,
            "p1417_queries_exact": all(row["exact_match"] for row in query_rows),
            "shuffle_invariant": shuffle_invariant,
            "zero_invalid_witnesses": invalid == 0,
        },
    }


def attach_comparisons(curve: dict[str, Any]) -> None:
    metrics = (
        "difference_x_support_count", "augmented_kummer_support_count",
        "output_x_support_count", "lossless_augmented_logical_bytes",
        "one_witness_augmented_logical_bytes",
        "total_construction_weighted_field_operation_units",
    )
    hash_means = {
        metric: sum(
            float(curve["policies"][name]["geometry"][metric]) for name in HASH_POLICIES
        ) / len(HASH_POLICIES)
        for metric in metrics
    }
    for policy in STRUCTURED_POLICIES:
        curve["policies"][policy]["comparisons"] = {
            f"{metric}_ratio_vs_hash_mean": ratio(
                curve["policies"][policy]["geometry"][metric], hash_means[metric]
            )
            for metric in metrics
        }


def synthetic_controls(source_curve: dict[str, Any]) -> dict[str, Any]:
    raw = p1398.relation_probe.load_verifier_module()
    verifier = p1408.OperationCountingVerifier(raw)
    p = int_value(source_curve["p"])
    order = int_value(source_curve["order"])
    ainvs = [0, 0, 0, int_value(source_curve["curve_a"]), int_value(source_curve["curve_b"])]
    base = tuple(source_curve["generator_point"])
    multiples = [1, order - 1, 2, order - 2, 3]
    factors = [point_key(verifier.mul_point(value, base, ainvs, p)) for value in multiples]
    geometry = build_geometry(raw, factors, ainvs, p, order)["summary"]
    branches = geometry["branch_source_histogram"]
    required = ("general", "doubling", "inverse_to_infinity", "left_infinity")
    return {
        "non_promotable": True,
        "reason": "known public multiples expose source logarithms; branch instrumentation only",
        "multiples": multiples,
        "required_branches": list(required),
        "branch_source_histogram": branches,
        "all_required_branches_exercised": all(int_value(branches.get(key, 0)) > 0 for key in required),
        "xadd_failure_count": geometry["xadd_failure_count"],
        "source_triple_count_exact": geometry["source_triple_count_exact"],
        "oriented_source_state_count_exact": geometry["oriented_source_state_count_exact"],
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.p1416.read_text(encoding="utf-8"))
    p1417 = json.loads(args.p1417.read_text(encoding="utf-8"))
    p1417_audit = json.loads(args.p1417_audit.read_text(encoding="utf-8"))
    raw = p1398.relation_probe.load_verifier_module()
    source_map = {
        (row["split"], int_value(row["bits"]), int_value(row["seed"])): row
        for row in source["curve_records"]
    }
    p1417_curves = p1417["curve_records"]
    if args.split:
        p1417_curves = [row for row in p1417_curves if row["split"] in set(args.split)]
    if args.bits:
        p1417_curves = [row for row in p1417_curves if int_value(row["bits"]) in set(args.bits)]
    if args.max_curves is not None:
        p1417_curves = p1417_curves[:args.max_curves]
    policies = tuple(args.policy) if args.policy else ALL_POLICIES
    if not p1417_curves or not policies:
        raise ValueError("P1418 selector produced no curve-policy cells")

    curve_records = []
    for p1417_curve in p1417_curves:
        key = (
            p1417_curve["split"], int_value(p1417_curve["bits"]),
            int_value(p1417_curve["seed"]),
        )
        source_curve = source_map[key]
        record = {
            **{name: source_curve[name] for name in (
                "split", "bits", "seed", "p", "curve_a", "curve_b", "order",
                "ordinary", "generator_point", "family",
                "field_modulus_special_structure_selected",
                "curve_map_specific_special_structure_selected",
            )},
            "factor_base_size_B": source_curve["factor_base_size_B"],
            "source_p1416_curve_sha256": canonical_hash(source_curve),
            "source_p1417_curve_sha256": canonical_hash(p1417_curve),
            "policies": {
                policy: build_policy_record(raw, source_curve, p1417_curve, policy)
                for policy in policies
            },
        }
        if all(name in policies for name in HASH_POLICIES):
            attach_comparisons(record)
        curve_records.append(record)

    fit_rows = [
        {
            "split": curve["split"],
            "policy": policy,
            "factor_base_size_B": curve["factor_base_size_B"],
            "geometry": cell["geometry"],
        }
        for curve in curve_records
        for policy, cell in curve["policies"].items()
    ]
    fits = {
        split: {
            policy: {
                metric: fit_exponent_in_B([
                    row for row in fit_rows
                    if row["split"] == split and row["policy"] == policy
                ], metric)
                for metric in FIT_METRICS
            }
            for policy in policies
        }
        for split in ("calibration", "heldout")
    }
    heldout = [curve for curve in curve_records if curve["split"] == "heldout"]
    promotion = {}
    for policy in policies:
        if policy not in STRUCTURED_POLICIES:
            continue
        cells = [curve["policies"][policy] for curve in heldout]
        exact = bool(cells) and all(all(cell["validation"].values()) for cell in cells)
        measured = fits["heldout"][policy]
        below = (
            measured["augmented_kummer_support_count"] is not None
            and measured["lossless_augmented_logical_bytes"] is not None
            and measured["total_construction_weighted_field_operation_units"] is not None
            and measured["augmented_kummer_support_count"] < 2.5
            and measured["lossless_augmented_logical_bytes"] < 2.5
            and measured["total_construction_weighted_field_operation_units"] < 2.5
        )
        symbolic = all(
            cell["geometry"]["source_witness_symbolic_exponent_in_B"] < 2.5
            for cell in cells
        ) if cells else False
        hash_advantage = False
        if all(name in policies for name in HASH_POLICIES):
            candidate = measured["lossless_augmented_logical_bytes"]
            hashes = [
                fits["heldout"][name]["lossless_augmented_logical_bytes"]
                for name in HASH_POLICIES
            ]
            hash_advantage = (
                candidate is not None and all(value is not None for value in hashes)
                and min(float(value) for value in hashes) - float(candidate) >= 0.05
            )
        promotion[policy] = {
            "exact_triple_query_and_branch_gate": exact,
            "measured_below_B2_5_gate": below,
            "symbolic_source_witness_below_B2_5_gate": symbolic,
            "fitted_hash_advantage_gate": hash_advantage,
            "end_to_end_or_fixed_curve_frontier_gate": False,
            "promoted": False,
        }
    promoted = [name for name, row in promotion.items() if row["promoted"]]
    cells = [cell for curve in curve_records for cell in curve["policies"].values()]
    queries = [query for cell in cells for query in cell["query_replays"]]
    synthetic = synthetic_controls(source_map[
        (p1417_curves[0]["split"], int_value(p1417_curves[0]["bits"]), int_value(p1417_curves[0]["seed"]))
    ]) if not args.skip_synthetic_controls else None
    all_exact = all(all(cell["validation"].values()) for cell in cells)
    full_preregistered_matrix = (
        not args.split
        and not args.bits
        and args.max_curves is None
        and set(policies) == set(ALL_POLICIES)
        and len(curve_records) == len(p1417["curve_records"])
    )
    claim = (
        "NEGATIVE_RESULT_P1418_KNOWN_DIFFERENCE_STATE_BOUNDARY"
        if full_preregistered_matrix and all_exact and not promoted
        else "OPEN_P1418_DIFFERENTIAL_STATE_AUDIT"
    )
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": claim,
        "claim_taxonomy": ["NEGATIVE RESULT", "TOY-EVIDENCE", "MODEL-BOUND"] if claim.startswith("NEGATIVE") else ["OPEN"],
        "hypothesis": (
            "Projective Kummer known-difference states compress exact pair-plus-factor composition, "
            "source witnesses, and query work below B^2.5 on a coordinate factor-base family."
        ),
        "null_hypothesis": (
            "Constructing x(A-F), orientation, and source witnesses restores cubic state/advice, "
            "and projective xADD changes only constants."
        ),
        "parameters": {
            "curve_count": len(curve_records),
            "full_preregistered_matrix": full_preregistered_matrix,
            "policies": list(policies),
            "factor_base_rule": "B=ceil(r^(1/5)), inherited from P1416",
            "canonical_triples": "0 <= i <= j <= k < B",
            "xadd_input": "(x(A),x(F),x(A-F)) with every A-F construction charged",
            "query_source": "the same public relation, target-descent, and schedule rows executed by P1417",
        },
        "input_hashes": {
            "contract_preregistration_sha256": contract_preregistration_sha256(
                args.contract
            ),
            "p1416_sha256": sha256_file(args.p1416),
            "p1417_sha256": sha256_file(args.p1417),
            "p1417_audit_sha256": sha256_file(args.p1417_audit),
            "p1417_audit_passed": p1417_audit["summary"]["audit_passed"],
        },
        "curve_records": curve_records,
        "synthetic_controls": synthetic,
        "summary": {
            "curve_count": len(curve_records),
            "policy_curve_count": len(cells),
            "query_count": len(queries),
            "exact_query_tuple_set_match_count": sum(row["exact_match"] for row in queries),
            "invalid_witness_count": sum(cell["invalid_witness_count"] for cell in cells),
            "exact_xadd_policy_curve_count": sum(cell["geometry"]["xadd_failure_count"] == 0 for cell in cells),
            "exact_p1416_triple_hash_policy_curve_count": sum(cell["triple_state_hash_matches_p1416"] for cell in cells),
            "shuffle_invariant_policy_curve_count": sum(cell["shuffle_control"]["label_invariant"] for cell in cells),
            "fitted_exponents_in_B": fits,
            "promotion_audit": {
                "policies": promotion,
                "promoted_coordinate_policies": promoted,
                "strict_generic_prime_field_speedup": False,
                "fixed_curve_frontier_improvement": False,
            },
            "synthetic_controls_passed": bool(
                synthetic is None or (
                    synthetic["all_required_branches_exercised"]
                    and synthetic["xadd_failure_count"] == 0
                    and synthetic["source_triple_count_exact"]
                    and synthetic["oriented_source_state_count_exact"]
                )
            ),
        },
        "interpretation": (
            "P1418 charges the known difference required by projective xADD and preserves every normalized "
            "source triple. The experiment separates projective field-operation savings from state and witness "
            "complexity. If exact augmented and lossless directories retain cubic scaling like matched hashes, "
            "this is a scoped negative for the known-difference representation, not for other divisor, adaptive, "
            "or non-enumerative index-calculus constructions."
        ),
        "next_action": (
            "If the exact known-difference inventory remains cubic, build P1419 around a symmetric-square "
            "degree-two divisor representation and test whether its add-one-factor update preserves source "
            "columns with subcubic state, using P1418 as the exact projective control."
        ),
        "red_team_objections": [
            "Every full-point A-F construction and every source witness is charged.",
            "Output-x and difference-x support counts are reported separately from lossless witness bytes.",
            "Projective normalization receives batch inversion but is not treated as free.",
            "Known-multiple synthetic controls are instrumentation-only and forbidden from promotion.",
            "Toy fitted exponents do not prove a universal lower bound.",
        ],
        "public_selector_inputs": [
            "frozen P1416 public factors and construction order",
            "P1417 public executed query points and fixed canonical triple ordering",
        ],
        "forbidden_selector_inputs_used": [],
    }


def render_note(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join([
        "# P1418 projective Kummer differential-state compiler", "",
        f"- Claim status: `{payload['claim_status']}`",
        f"- Curves / policy-curves / queries: `{summary['curve_count']}/{summary['policy_curve_count']}/{summary['query_count']}`",
        f"- Exact xADD cells: `{summary['exact_xadd_policy_curve_count']}/{summary['policy_curve_count']}`",
        f"- Exact P1416 triple hashes: `{summary['exact_p1416_triple_hash_policy_curve_count']}/{summary['policy_curve_count']}`",
        f"- Exact query tuple sets: `{summary['exact_query_tuple_set_match_count']}/{summary['query_count']}`",
        f"- Invalid witnesses: `{summary['invalid_witness_count']}`",
        f"- Synthetic controls passed: `{summary['synthetic_controls_passed']}`",
        f"- Promoted policies: `{summary['promotion_audit']['promoted_coordinate_policies']}`",
        "", "## Interpretation", "", payload["interpretation"],
        "", "## Next action", "", payload["next_action"], "",
    ])


def render_result(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join([
        "# P1418 restricted known-difference state result", "", "## Status", "",
        "NEGATIVE RESULT / TOY-EVIDENCE / MODEL-BOUND", "", "## Statement", "",
        "For every executed frozen cell, P1418 constructs each known difference A-F, evaluates an exact",
        "projective Kummer xADD, preserves all canonical source triples, and replays the P1417 public queries.",
        "", "## Evidence", "",
        f"- Exact xADD cells: `{summary['exact_xadd_policy_curve_count']}/{summary['policy_curve_count']}`.",
        f"- Exact P1416 triple hashes: `{summary['exact_p1416_triple_hash_policy_curve_count']}/{summary['policy_curve_count']}`.",
        f"- Exact query tuple sets: `{summary['exact_query_tuple_set_match_count']}/{summary['query_count']}`.",
        f"- Invalid witnesses: `{summary['invalid_witness_count']}`.",
        f"- Promoted coordinate policies: `{summary['promotion_audit']['promoted_coordinate_policies']}`.",
        "", "## Scope", "",
        "The result covers this canonical known-difference and lossless-witness representation on frozen toy curves.",
        "It does not rule out symmetric-square divisors, adaptive zero-product trees, or other non-enumerative representations.",
        "", "## Next positive direction", "", payload["next_action"], "",
    ])


def append_contract_results(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "", "## Results",
        f"- Timestamp: `{payload['generated_at']}`.",
        f"- Claim status: `{payload['claim_status']}`.",
        f"- Curves / policy-curves / queries: `{summary['curve_count']}/{summary['policy_curve_count']}/{summary['query_count']}`.",
        f"- Exact xADD cells: `{summary['exact_xadd_policy_curve_count']}/{summary['policy_curve_count']}`.",
        f"- Exact P1416 triple hashes: `{summary['exact_p1416_triple_hash_policy_curve_count']}/{summary['policy_curve_count']}`.",
        f"- Exact query tuple sets: `{summary['exact_query_tuple_set_match_count']}/{summary['query_count']}`.",
        f"- Invalid witnesses: `{summary['invalid_witness_count']}`.",
        f"- Promotion audit: `{summary['promotion_audit']}`.",
        "", "## Interpretation", payload["interpretation"], "",
    ]
    text = path.read_text(encoding="utf-8")
    prefix = text.split("\n## Results\n", 1)[0].rstrip()
    path.write_text(prefix + "\n" + "\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--p1416", type=Path, default=DEFAULT_P1416)
    parser.add_argument("--p1417", type=Path, default=DEFAULT_P1417)
    parser.add_argument("--p1417-audit", type=Path, default=DEFAULT_P1417_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--split", action="append")
    parser.add_argument("--bits", type=int, action="append")
    parser.add_argument("--policy", action="append", choices=ALL_POLICIES)
    parser.add_argument("--max-curves", type=int)
    parser.add_argument("--skip-synthetic-controls", action="store_true")
    parser.add_argument("--no-contract-update", action="store_true")
    parser.add_argument("--no-handoff", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.contract, args.p1416, args.p1417, args.p1417_audit):
        if not path.exists():
            raise FileNotFoundError(path)
    payload = build_payload(args)
    note = render_note(payload)
    result = render_result(payload)
    for path, text in (
        (args.out, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
        (args.note, note),
        (args.result, result),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if not args.no_handoff:
        for path, text in (
            (RESEARCH_DIR / args.out.name, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
            (RESEARCH_DIR / args.note.name, note),
            (RESEARCH_DIR / args.result.name, result),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if not args.no_contract_update:
        append_contract_results(args.contract, payload)
        if not args.no_handoff:
            (RESEARCH_DIR / args.contract.name).write_text(
                args.contract.read_text(encoding="utf-8"), encoding="utf-8"
            )
    summary = payload["summary"]
    print(
        "claim={claim} cells={cells} queries={queries} xadd={xadd}/{cells} "
        "triples={triples}/{cells} exact={exact}/{queries} invalid={invalid} promoted={promoted}".format(
            claim=payload["claim_status"],
            cells=summary["policy_curve_count"],
            queries=summary["query_count"],
            xadd=summary["exact_xadd_policy_curve_count"],
            triples=summary["exact_p1416_triple_hash_policy_curve_count"],
            exact=summary["exact_query_tuple_set_match_count"],
            invalid=summary["invalid_witness_count"],
            promoted=summary["promotion_audit"]["promoted_coordinate_policies"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
