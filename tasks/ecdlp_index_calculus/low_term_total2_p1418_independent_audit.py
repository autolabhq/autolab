#!/usr/bin/env python3
"""Independent exhaustive audit for the P1418 differential-state probe."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
import low_term_total2_p1408_rational_map_image_factor_bases_after_p1407 as p1408
import low_term_total2_p1416_coordinate_additive_energy_recursive_s3_after_p1415 as p1416


STATE_DIR = Path("ecdlp_index_calculus_state")
HOST_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
RESEARCH_DIR = Path("research")
DEFAULT_P1418 = STATE_DIR / "p1418_projective_kummer_differential_state_after_p1417_probe.json"
DEFAULT_P1416 = HOST_STATE_DIR / "p1416_coordinate_additive_energy_recursive_s3_after_p1415_probe.json"
DEFAULT_P1417 = HOST_STATE_DIR / "p1417_kummer_s3_quotient_resultant_after_p1416_probe.json"
DEFAULT_P1417_AUDIT = HOST_STATE_DIR / "p1417_kummer_s3_quotient_resultant_after_p1416_audit.json"
DEFAULT_CONTRACT = STATE_DIR / "experiment_contract_p1418_projective_kummer_differential_state_after_p1417.md"
DEFAULT_OUT = STATE_DIR / "p1418_projective_kummer_differential_state_after_p1417_audit.json"
DEFAULT_NOTE = STATE_DIR / "p1418_projective_kummer_differential_state_after_p1417_audit.md"

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


def point_key(point: Any) -> Any:
    return None if point is None else (int_value(point[0]), int_value(point[1]))


def point_sort_key(point: Any) -> tuple[int, int, int]:
    return (-1, 0, 0) if point is None else (0, int_value(point[0]), int_value(point[1]))


def x_key(point: Any) -> Any:
    return None if point is None else int_value(point[0])


def json_key(key: Any) -> Any:
    if isinstance(key, tuple):
        return [json_key(value) for value in key]
    return key


def stable_key(key: Any) -> str:
    return json.dumps(json_key(key), sort_keys=True, separators=(",", ":"))


def directory_hash(directory: dict[Any, list[tuple[int, int, int]]]) -> str:
    return canonical_hash([
        [json_key(key), [list(row) for row in sorted(set(witnesses))]]
        for key, witnesses in sorted(directory.items(), key=lambda item: stable_key(item[0]))
    ])


def curve_rhs(x_value: int, a: int, b: int, p: int) -> int:
    return (pow(x_value, 3, p) + a * x_value + b) % p


def ratio(value: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else round(float(value) / float(denominator), 8)


def independent_xadd(
    left: Any, right: Any, difference: Any, a: int, b: int, p: int,
) -> tuple[tuple[int, int], str, dict[str, int]]:
    if left is None:
        return ((1, 0) if right is None else (int_value(right[0]), 1)), "left_infinity", {"field_moves": 1}
    if right is None:
        return (int_value(left[0]), 1), "right_infinity", {"field_moves": 1}
    x_left = int_value(left[0]) % p
    x_right = int_value(right[0]) % p
    if x_left == x_right:
        if difference is not None:
            return (1, 0), "inverse_to_infinity", {"field_comparisons": 1}
        rhs = curve_rhs(x_left, a, b, p)
        projective = (
            (pow(3 * x_left * x_left + a, 2, p) - 8 * x_left * rhs) % p,
            4 * rhs % p,
        )
        branch = "doubling_to_infinity" if projective[1] == 0 else "doubling"
        return projective, branch, {
            "field_multiplications": 4,
            "field_squarings": 2,
            "field_additions_or_subtractions": 5,
        }
    x_difference, z_difference = (
        (1, 0) if difference is None else (int_value(difference[0]), 1)
    )
    delta_squared = pow((x_right - x_left) % p, 2, p)
    numerator = (
        2 * (curve_rhs(x_left, a, b, p) + curve_rhs(x_right, a, b, p)) * z_difference
        - delta_squared * (2 * (x_left + x_right) * z_difference + x_difference)
    ) % p
    return (numerator, delta_squared * z_difference % p), "general", {
        "field_multiplications": 8,
        "field_squarings": 3,
        "field_additions_or_subtractions": 10,
    }


def directory_memory(
    unique_keys: int,
    witness_count: int,
    key_field_elements: int,
    field_bytes: int,
    index_bytes: int,
) -> dict[str, int]:
    pointer_bytes = max(1, math.ceil(max(1, witness_count).bit_length() / 8))
    key_bytes = unique_keys * (key_field_elements * field_bytes + 1 + pointer_bytes)
    witness_bytes = witness_count * 3 * index_bytes
    return {
        "unique_key_count": unique_keys,
        "witness_count": witness_count,
        "key_logical_bytes": key_bytes,
        "witness_logical_bytes": witness_bytes,
        "lossless_logical_bytes": key_bytes + witness_bytes,
        "one_witness_per_key_logical_bytes": key_bytes + unique_keys * 3 * index_bytes,
    }


def rebuild_geometry(
    raw: Any, factors: list[Any], ainvs: list[int], p: int, order: int,
) -> dict[str, Any]:
    verifier = p1408.OperationCountingVerifier(raw)
    pairs: list[tuple[int, int, Any]] = []
    pair_support = set()
    for left in range(len(factors)):
        for right in range(left, len(factors)):
            pair_point = point_key(verifier.add_points(factors[left], factors[right], ainvs, p))
            pairs.append((left, right, pair_point))
            pair_support.add(pair_point)

    before = verifier.snapshot()
    a = int_value(ainvs[3]) % p
    b = int_value(ainvs[4]) % p
    cache: dict[tuple[Any, int], dict[str, Any]] = {}
    records = []
    xadd_failures = 0
    xadd_operations: Counter[str] = Counter()
    branch_cache: Counter[str] = Counter()
    for left, right, pair_point in pairs:
        for factor_index, factor in enumerate(factors):
            cache_key = (pair_point, factor_index)
            if cache_key not in cache:
                difference = point_key(
                    verifier.add_points(pair_point, p1416.negate(factor, p), ainvs, p)
                )
                total = point_key(verifier.add_points(pair_point, factor, ainvs, p))
                projective, branch, operations = independent_xadd(
                    pair_point, factor, difference, a, b, p
                )
                x_value, z_value = projective
                exact = (
                    z_value % p == 0 if total is None
                    else z_value % p != 0 and x_value % p == int_value(total[0]) * z_value % p
                )
                xadd_failures += not exact
                xadd_operations.update(operations)
                branch_cache[branch] += 1
                cache[cache_key] = {
                    "difference": difference,
                    "total": total,
                    "projective": projective,
                    "branch": branch,
                }
            records.append({
                "oriented": (left, right, factor_index),
                "normalized": tuple(sorted((left, right, factor_index))),
                "pair_point": pair_point,
                "factor_index": factor_index,
                "cache_key": cache_key,
            })
    construction_profile = p1416.profile_delta(before, verifier.snapshot())

    finite_count = sum(row["projective"][1] % p != 0 for row in cache.values())
    xadd_operations.update({
        "batch_inversions": int(finite_count > 0),
        "batch_multiplications": 4 * finite_count,
        "normalized_finite_count": finite_count,
        "projective_infinity_count": len(cache) - finite_count,
    })
    difference_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    augmented_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    projective_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    output_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    full_directory: dict[Any, list[tuple[int, int, int]]] = defaultdict(list)
    no_difference: dict[Any, set[Any]] = defaultdict(set)
    branch_source: Counter[str] = Counter()
    orientation: Counter[str] = Counter()
    for record in records:
        row = cache[record["cache_key"]]
        factor = factors[record["factor_index"]]
        difference_x = x_key(row["difference"])
        output_x = x_key(row["total"])
        branch = row["branch"]
        augmented_key = (x_key(record["pair_point"]), x_key(factor), difference_x, branch)
        no_difference_key = (x_key(record["pair_point"]), x_key(factor))
        difference_directory[difference_x].append(record["oriented"])
        augmented_directory[augmented_key].append(record["oriented"])
        projective_directory[(*row["projective"], branch)].append(record["oriented"])
        output_directory[output_x].append(record["oriented"])
        full_directory[row["total"]].append(record["normalized"])
        no_difference[no_difference_key].add(output_x)
        branch_source[branch] += 1
        orientation[
            "infinity" if row["total"] is None
            else str(int_value(row["total"][1]) > p // 2)
        ] += 1

    full_table = {
        point: sorted(set(witnesses)) for point, witnesses in full_directory.items()
    }
    normalized_count = sum(len(witnesses) for witnesses in full_table.values())
    oriented_count = len(records)
    field_bytes = max(1, math.ceil(p.bit_length() / 8))
    index_bytes = max(1, math.ceil(max(1, len(factors)).bit_length() / 8))
    memories = {
        "difference_x": directory_memory(len(difference_directory), oriented_count, 1, field_bytes, index_bytes),
        "augmented_kummer": directory_memory(len(augmented_directory), oriented_count, 3, field_bytes, index_bytes),
        "raw_projective_output": directory_memory(len(projective_directory), oriented_count, 2, field_bytes, index_bytes),
        "output_x": directory_memory(len(output_directory), oriented_count, 1, field_bytes, index_bytes),
        "full_output": directory_memory(len(full_directory), normalized_count, 2, field_bytes, index_bytes),
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
    ambiguous = {key: values for key, values in no_difference.items() if len(values) > 1}
    pair_scans = len(pair_support)
    storage_entries = len(augmented_directory) + oriented_count
    return {
        "factor_base_size_B": len(factors),
        "pair_full_support_count": len(pair_support),
        "source_triple_count": normalized_count,
        "expected_source_triple_count": math.comb(len(factors) + 2, 3),
        "source_triple_count_exact": normalized_count == math.comb(len(factors) + 2, 3),
        "oriented_source_state_count": oriented_count,
        "expected_oriented_source_state_count": len(factors) * math.comb(len(factors) + 1, 2),
        "oriented_source_state_count_exact": oriented_count == len(factors) * math.comb(len(factors) + 1, 2),
        "pair_factor_cache_entry_count": len(cache),
        "difference_x_support_count": len(difference_directory),
        "augmented_kummer_support_count": len(augmented_directory),
        "raw_projective_output_support_count": len(projective_directory),
        "output_x_support_count": len(output_directory),
        "full_output_support_count": len(full_directory),
        "no_difference_key_count": len(no_difference),
        "no_difference_ambiguous_key_count": len(ambiguous),
        "no_difference_ambiguous_key_fraction": ratio(len(ambiguous), len(no_difference)),
        "no_difference_max_output_multiplicity": max((len(values) for values in no_difference.values()), default=0),
        "xadd_failure_count": xadd_failures,
        "branch_cache_histogram": dict(sorted(branch_cache.items())),
        "branch_source_histogram": dict(sorted(branch_source.items())),
        "orientation_label_histogram": dict(sorted(orientation.items())),
        "difference_directory_sha256": directory_hash(difference_directory),
        "augmented_directory_sha256": directory_hash(augmented_directory),
        "raw_projective_directory_sha256": directory_hash(projective_directory),
        "output_x_directory_sha256": directory_hash(output_directory),
        "full_triple_state_sha256": p1416.support_hash(full_table),
        "xadd_operation_counts": dict(xadd_operations),
        "difference_and_sum_point_operation_profile": construction_profile,
        "difference_and_sum_affine_field_operation_estimate": field_estimate,
        "projective_xadd_weighted_field_operation_units": round(projective_weighted, 10),
        "difference_and_sum_weighted_field_operation_units": round(affine_weighted, 10),
        "total_construction_weighted_field_operation_units": round(projective_weighted + affine_weighted, 10),
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
    }


def brute_tuples(
    raw: Any, row_point: Any, factors: list[Any], ainvs: list[int], p: int,
) -> list[tuple[int, ...]]:
    verifier = p1408.OperationCountingVerifier(raw)
    matches = []
    for indices in itertools.combinations_with_replacement(range(len(factors)), 5):
        total = None
        for index in indices:
            total = verifier.add_points(total, factors[index], ainvs, p)
        if point_key(total) == point_key(row_point):
            matches.append(indices)
    return matches


def fit_exponent(rows: list[dict[str, Any]], metric: str) -> float | None:
    points = [
        (math.log(float(row["B"])), math.log(float(row["geometry"][metric])))
        for row in rows if row["B"] > 1 and row["geometry"][metric] > 0
    ]
    if len({x_value for x_value, _ in points}) < 2:
        return None
    mean_x = sum(x_value for x_value, _ in points) / len(points)
    mean_y = sum(y_value for _, y_value in points) / len(points)
    denominator = sum((x_value - mean_x) ** 2 for x_value, _ in points)
    return round(
        sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in points)
        / denominator,
        10,
    )


def add_check(
    errors: list[str], checks: Counter[str], condition: bool, label: str,
) -> None:
    checks["total"] += 1
    if condition:
        checks["passed"] += 1
    else:
        errors.append(label)


def audit_payload(
    p1418: dict[str, Any], p1416: dict[str, Any], p1417_audit: dict[str, Any],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    raw = p1398.relation_probe.load_verifier_module()
    source_curves = {
        (row["split"], int_value(row["bits"]), int_value(row["seed"])): row
        for row in p1416["curve_records"]
    }
    checks: Counter[str] = Counter()
    errors: list[str] = []
    fit_rows = []
    geometry_rebuilds = 0
    shuffle_rebuilds = 0
    query_rebuilds = 0
    tuple_candidates = 0
    projective_checks = 0

    for curve in p1418["curve_records"]:
        key = (curve["split"], int_value(curve["bits"]), int_value(curve["seed"]))
        source_curve = source_curves[key]
        p = int_value(curve["p"])
        order = int_value(curve["order"])
        ainvs = [0, 0, 0, int_value(curve["curve_a"]), int_value(curve["curve_b"])]
        for policy, cell in curve["policies"].items():
            label = f"{key}:{policy}"
            factors = [tuple(point) for point in source_curve["policies"][policy]["factor_base_points"]]
            rebuilt = rebuild_geometry(raw, factors, ainvs, p, order)
            quoted = cell["geometry"]
            for metric, value in rebuilt.items():
                add_check(errors, checks, quoted[metric] == value, f"{label}:geometry:{metric}")
            add_check(
                errors,
                checks,
                rebuilt["full_triple_state_sha256"]
                == source_curve["policies"][policy]["state_geometry"]["triple_state_sha256"],
                f"{label}:p1416_triple_hash",
            )
            reversed_geometry = rebuild_geometry(raw, list(reversed(factors)), ainvs, p, order)
            for metric in cell["shuffle_control"]["checked_metrics"]:
                add_check(
                    errors, checks, rebuilt[metric] == reversed_geometry[metric],
                    f"{label}:shuffle:{metric}",
                )
            fit_rows.append({"split": curve["split"], "policy": policy, "B": len(factors), "geometry": rebuilt})
            geometry_rebuilds += 1
            shuffle_rebuilds += 1
            projective_checks += rebuilt["pair_factor_cache_entry_count"]

            for query in cell["query_replays"]:
                query_point = None if query["public_point"] is None else tuple(query["public_point"])
                tuples = brute_tuples(raw, query_point, factors, ainvs, p)
                tuple_hash = canonical_hash([list(row) for row in tuples])
                query_label = f"{label}:{query['kind']}:{query['index']}"
                add_check(errors, checks, tuple_hash == query["tuple_sha256"], query_label + ":p1418_hash")
                add_check(errors, checks, tuple_hash == query["p1417_tuple_sha256"], query_label + ":p1417_hash")
                add_check(errors, checks, len(tuples) == query["tuple_count"], query_label + ":count")
                query_rebuilds += 1
                tuple_candidates += math.comb(len(factors) + 4, 5)

    for split in ("calibration", "heldout"):
        for policy in p1418["parameters"]["policies"]:
            rows = [row for row in fit_rows if row["split"] == split and row["policy"] == policy]
            for metric in FIT_METRICS:
                add_check(
                    errors,
                    checks,
                    fit_exponent(rows, metric)
                    == p1418["summary"]["fitted_exponents_in_B"][split][policy][metric],
                    f"fit:{split}:{policy}:{metric}",
                )

    synthetic = p1418["synthetic_controls"]
    source_curve = p1416["curve_records"][0]
    p = int_value(source_curve["p"])
    order = int_value(source_curve["order"])
    ainvs = [0, 0, 0, int_value(source_curve["curve_a"]), int_value(source_curve["curve_b"])]
    verifier = p1408.OperationCountingVerifier(raw)
    base = tuple(source_curve["generator_point"])
    factors = [point_key(verifier.mul_point(value, base, ainvs, p)) for value in synthetic["multiples"]]
    synthetic_rebuilt = rebuild_geometry(raw, factors, ainvs, p, order)
    add_check(errors, checks, synthetic_rebuilt["branch_source_histogram"] == synthetic["branch_source_histogram"], "synthetic:branches")
    add_check(errors, checks, synthetic_rebuilt["xadd_failure_count"] == 0, "synthetic:xadd")
    add_check(errors, checks, all(synthetic_rebuilt["branch_source_histogram"].get(branch, 0) > 0 for branch in synthetic["required_branches"]), "synthetic:coverage")

    summary = p1418["summary"]
    add_check(errors, checks, p1417_audit["summary"]["audit_passed"], "input:p1417_audit")
    add_check(errors, checks, p1418["input_hashes"]["contract_preregistration_sha256"] == input_hashes["contract_preregistration_sha256"], "input:contract")
    add_check(errors, checks, p1418["input_hashes"]["p1416_sha256"] == input_hashes["p1416_sha256"], "input:p1416")
    add_check(errors, checks, p1418["input_hashes"]["p1417_sha256"] == input_hashes["p1417_sha256"], "input:p1417")
    add_check(errors, checks, p1418["input_hashes"]["p1417_audit_sha256"] == input_hashes["p1417_audit_sha256"], "input:p1417_audit_hash")
    add_check(errors, checks, summary["policy_curve_count"] == geometry_rebuilds, "summary:cells")
    add_check(errors, checks, summary["query_count"] == query_rebuilds, "summary:queries")
    add_check(errors, checks, summary["exact_xadd_policy_curve_count"] == geometry_rebuilds, "summary:xadd")
    add_check(errors, checks, summary["exact_p1416_triple_hash_policy_curve_count"] == geometry_rebuilds, "summary:triples")
    add_check(errors, checks, summary["shuffle_invariant_policy_curve_count"] == shuffle_rebuilds, "summary:shuffle")
    add_check(errors, checks, summary["invalid_witness_count"] == 0, "summary:invalid")
    add_check(errors, checks, summary["promotion_audit"]["promoted_coordinate_policies"] == [], "summary:no_promotion")
    add_check(errors, checks, all(not row["measured_below_B2_5_gate"] and not row["symbolic_source_witness_below_B2_5_gate"] and not row["fitted_hash_advantage_gate"] and not row["promoted"] for row in summary["promotion_audit"]["policies"].values()), "summary:promotion_gates")
    add_check(errors, checks, p1418["claim_status"] == "NEGATIVE_RESULT_P1418_KNOWN_DIFFERENCE_STATE_BOUNDARY", "summary:claim")

    return {
        "schema": "ecdlp.p1418_independent_audit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_hashes": input_hashes,
        "summary": {
            "audit_passed": not errors,
            "check_count": checks["total"],
            "passed_check_count": checks["passed"],
            "error_count": len(errors),
            "geometry_rebuild_count": geometry_rebuilds,
            "label_shuffle_rebuild_count": shuffle_rebuilds,
            "projective_xadd_cross_multiplication_check_count": projective_checks,
            "independent_five_tuple_query_rebuild_count": query_rebuilds,
            "independent_normalized_five_tuple_candidates_checked": tuple_candidates,
            "promoted_coordinate_policies": summary["promotion_audit"]["promoted_coordinate_policies"],
        },
        "errors": errors,
        "method": [
            "fresh pair-by-factor construction over every normalized pair witness and every factor",
            "independent projective xADD formula checked by cross multiplication against affine A+F",
            "fresh difference, augmented, projective, output, and normalized-triple witness directories",
            "independent exhaustive five-factor enumeration for every public query",
            "fresh label-reversal rebuilds and exponent fits from reconstructed raw rows",
        ],
        "interpretation": (
            "The audit validates the executed permutation-closed known-difference representation and its "
            "cost accounting. It is a scoped toy/model negative, not a lower bound for divisor, adaptive, "
            "or non-enumerative index-calculus representations."
        ),
    }


def render_note(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join([
        "# P1418 independent audit", "",
        f"- Audit passed: `{summary['audit_passed']}`",
        f"- Checks: `{summary['passed_check_count']}/{summary['check_count']}`",
        f"- Geometry / shuffle rebuilds: `{summary['geometry_rebuild_count']}/{summary['label_shuffle_rebuild_count']}`",
        f"- Projective xADD cross checks: `{summary['projective_xadd_cross_multiplication_check_count']}`",
        f"- Five-term queries / candidates: `{summary['independent_five_tuple_query_rebuild_count']}/{summary['independent_normalized_five_tuple_candidates_checked']}`",
        f"- Errors: `{summary['error_count']}`",
        f"- Promoted policies: `{summary['promoted_coordinate_policies']}`",
        "", "## Interpretation", "", payload["interpretation"], "",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1418", type=Path, default=DEFAULT_P1418)
    parser.add_argument("--p1416", type=Path, default=DEFAULT_P1416)
    parser.add_argument("--p1417", type=Path, default=DEFAULT_P1417)
    parser.add_argument("--p1417-audit", type=Path, default=DEFAULT_P1417_AUDIT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--no-handoff", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p1418 = json.loads(args.p1418.read_text(encoding="utf-8"))
    p1416 = json.loads(args.p1416.read_text(encoding="utf-8"))
    p1417_audit = json.loads(args.p1417_audit.read_text(encoding="utf-8"))
    input_hashes = {
        "p1418_sha256": sha256_file(args.p1418),
        "p1416_sha256": sha256_file(args.p1416),
        "p1417_sha256": sha256_file(args.p1417),
        "p1417_audit_sha256": sha256_file(args.p1417_audit),
        "contract_preregistration_sha256": contract_preregistration_sha256(args.contract),
    }
    payload = audit_payload(p1418, p1416, p1417_audit, input_hashes)
    note = render_note(payload)
    for path, text in (
        (args.out, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
        (args.note, note),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if not args.no_handoff:
        for path, text in (
            (RESEARCH_DIR / args.out.name, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
            (RESEARCH_DIR / args.note.name, note),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    summary = payload["summary"]
    print(
        f"audit_passed={summary['audit_passed']} "
        f"checks={summary['passed_check_count']}/{summary['check_count']} "
        f"xadd={summary['projective_xadd_cross_multiplication_check_count']} "
        f"queries={summary['independent_five_tuple_query_rebuild_count']} "
        f"errors={summary['error_count']}"
    )
    return 0 if summary["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
