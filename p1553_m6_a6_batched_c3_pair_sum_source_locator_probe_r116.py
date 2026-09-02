#!/usr/bin/env python3
"""Audit the exact A6-batched C3+C3 interface at the R115 m=6 vertex."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "p1553.m6_a6_batched_c3_pair_sum_source_locator.r116.v1"
LOG_B_GROUP_ORDER = Fraction(5)
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
RHO_EXPONENT = Fraction(5, 2)
ALPHA = Fraction(1, 12)
BETA = Fraction(3, 4)
A3_EXPONENT = 3 * ALPHA
A6_EXPONENT = 6 * ALPHA
C3_EXPONENT = 3 * BETA
C6_EXPONENT = 6 * BETA

R115_PRODUCER = pathlib.Path(
    "p1553_relation_arity_factor_base_transposed_"
    "interface_rebalance_probe_r115.py"
)
R115_PRODUCER_SHA256 = (
    "d3da38cdf54c39fca800fa451ad54a99bf0288529987b14ec39ee1367b3b615a"
)
R115_REPORT = pathlib.Path(
    "p1553_relation_arity_factor_base_transposed_"
    "interface_rebalance_probe_report_r115.json"
)
R115_REPORT_SHA256 = (
    "2b43c37f93e03c6ccdb675a34d48d7478deb3b0349026c65b2f6fe65695cb88e"
)
R115_FROZEN = pathlib.Path(
    "frozen_relation_arity_factor_base_exponent_model.json"
)
R115_FROZEN_SHA256 = (
    "8c6a589c1caa227f23adbf2c72d0800026d6cc49c34461fa99b72ac6a3f79209"
)
R115_LEDGER = pathlib.Path(
    "relation_arity_factor_base_feasibility_ledger.json"
)
R115_LEDGER_SHA256 = (
    "4936d19fa0a1024743d4aca462ddb14362495e1f52581913a6231586c7bc8ffe"
)
R115_CAP_TABLE = pathlib.Path("transposed_interface_cap_table.json")
R115_CAP_TABLE_SHA256 = (
    "4ceb78067425e998f17f562ad10b79891b561662da2c4078bb14e95117230891"
)
R115_LOGS = pathlib.Path("factor_logs_and_identical_descent_r115.json")
R115_LOGS_SHA256 = (
    "e8bfa068aae74130599656c1b9a2bfa2d91c5b5cd3c26cc1f2eae3cc4425f497"
)
R115_GATE = pathlib.Path(
    "p1553_relation_arity_factor_base_transposed_"
    "interface_rebalance_probe_gate_r115.md"
)
R115_GATE_SHA256 = (
    "11dd899c1ab6b89444cd9c9cc75709a64d9abf99d9370d2536e2f20e3d0fd1de"
)
R115_PARENT = pathlib.Path(
    "p1553_relation_arity_factor_base_transposed_"
    "interface_rebalance_probe_parent_report_r115.yaml"
)
R115_PARENT_SHA256 = (
    "78e92013af76848e4c925cbbe99e8b74d9c9e73726e2f9c7f542a03c2dd9b27b"
)
R82_PRODUCER = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_r82.py"
)
R82_PRODUCER_SHA256 = (
    "7380bff3175625016affee4703b0b0f2867a28113f72eef2d90614ed57ffef07"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
DINUR_GOLOVNEV = pathlib.Path(
    "references/dinur_golovnev_3sum_indexing_2512.04258v2.pdf"
)
DINUR_GOLOVNEV_SHA256 = (
    "e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c"
)
KIRKPATRICK_ET_AL = pathlib.Path(
    "references/kirkpatrick_et_al_preprocessed_3sum_"
    "unknown_universes_2602.11363v1.pdf"
)
KIRKPATRICK_ET_AL_SHA256 = (
    "6c676ae909461219b8d2f4480225aade0ab46633c989e6249c980dae79c203ca"
)

Point = tuple[int, int] | None
Counter = collections.Counter[Point]
IndexSource = tuple[int, ...]


def load_r82() -> Any:
    spec = importlib.util.spec_from_file_location("p1553_r82_for_r116", R82_PRODUCER)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R82 finite-curve controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R82 = load_r82()
R81 = R82.R81
R70 = R82.R70


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r115_producer", R115_PRODUCER, R115_PRODUCER_SHA256),
        ("r115_report", R115_REPORT, R115_REPORT_SHA256),
        ("r115_frozen", R115_FROZEN, R115_FROZEN_SHA256),
        ("r115_ledger", R115_LEDGER, R115_LEDGER_SHA256),
        ("r115_cap_table", R115_CAP_TABLE, R115_CAP_TABLE_SHA256),
        ("r115_logs", R115_LOGS, R115_LOGS_SHA256),
        ("r115_gate", R115_GATE, R115_GATE_SHA256),
        ("r115_parent", R115_PARENT, R115_PARENT_SHA256),
        ("r82_producer", R82_PRODUCER, R82_PRODUCER_SHA256),
        ("r82_report", R82_REPORT, R82_REPORT_SHA256),
        ("dinur_golovnev_v2", DINUR_GOLOVNEV, DINUR_GOLOVNEV_SHA256),
        (
            "kirkpatrick_et_al_v1",
            KIRKPATRICK_ET_AL,
            KIRKPATRICK_ET_AL_SHA256,
        ),
    )
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in rows
    }


def verify_source_bindings() -> dict[str, str]:
    bindings = source_binding_records()
    actual = {
        name: sha256_file(pathlib.Path(binding["path"]))
        for name, binding in bindings.items()
    }
    failures = [
        name
        for name, binding in bindings.items()
        if actual[name] != binding["sha256"]
    ]
    if failures:
        raise AssertionError(f"R116 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def ordered_endpoint_map(
    points: Sequence[Point],
    arity: int,
    curve: dict[str, Any],
) -> tuple[Counter, dict[Point, IndexSource]]:
    return R82.ordered_endpoint_map(points, arity, curve)


def convolve_counters(
    left: Mapping[Point, int],
    right: Mapping[Point, int],
    curve: dict[str, Any],
) -> Counter:
    result: Counter = collections.Counter()
    for left_endpoint, left_count in left.items():
        for right_endpoint, right_count in right.items():
            endpoint = R70.add(left_endpoint, right_endpoint, curve)
            result[endpoint] += left_count * right_count
    return result


def convolve_first_sources(
    left: Mapping[Point, Any],
    right: Mapping[Point, Any],
    curve: dict[str, Any],
) -> dict[Point, tuple[Any, Any]]:
    result: dict[Point, tuple[Any, Any]] = {}
    for left_endpoint, left_source in left.items():
        for right_endpoint, right_source in right.items():
            endpoint = R70.add(left_endpoint, right_endpoint, curve)
            result.setdefault(endpoint, (left_source, right_source))
    return result


def convolution_count(
    target: Point,
    left: Mapping[Point, int],
    right: Mapping[Point, int],
    curve: dict[str, Any],
) -> int:
    if len(left) > len(right):
        left, right = right, left
    total = 0
    for endpoint, multiplicity in left.items():
        complement = R70.add(target, R70.negate(endpoint, curve), curve)
        total += multiplicity * right.get(complement, 0)
    return total


def convolution_source(
    target: Point,
    left: Mapping[Point, Any],
    right: Mapping[Point, Any],
    curve: dict[str, Any],
) -> tuple[Any, Any] | None:
    if len(left) <= len(right):
        for endpoint, source in left.items():
            complement = R70.add(target, R70.negate(endpoint, curve), curve)
            if complement in right:
                return source, right[complement]
        return None
    reverse = convolution_source(target, right, left, curve)
    return None if reverse is None else (reverse[1], reverse[0])


def replay_atom_source(
    source_a: Sequence[int],
    source_c: Sequence[int],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R82.add_many(
        itertools.chain(
            (atoms_a[index] for index in source_a),
            (atoms_c[index] for index in source_c),
        ),
        curve,
    )


def replay_factor_source(
    source_a: Sequence[int],
    source_c: Sequence[int],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    if len(source_a) != len(source_c):
        raise AssertionError("factor source requires one A and one C per slot")
    factors = [
        R70.add(atoms_a[index_a], atoms_c[index_c], curve)
        for index_a, index_c in zip(source_a, source_c)
    ]
    return R82.add_many(factors, curve)


def s3_chain_certificate(
    factors: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    if not factors:
        raise ValueError("S3 chain requires at least one factor")
    current = factors[0]
    checks = []
    for factor in factors[1:]:
        endpoint = R70.add(current, factor, curve)
        if current is None or factor is None or endpoint is None:
            checks.append(
                {
                    "projective_exception": True,
                    "s3_zero": None,
                    "endpoint": point_json(endpoint),
                }
            )
        else:
            value = R70.semaev_s3(
                current[0],
                factor[0],
                R70.negate(endpoint, curve)[0],
                curve,
            )
            checks.append(
                {
                    "projective_exception": False,
                    "s3_zero": value == 0,
                    "endpoint": point_json(endpoint),
                }
            )
        current = endpoint
    return {
        "step_count": len(checks),
        "nonprojective_checks": sum(
            row["s3_zero"] is not None for row in checks
        ),
        "all_nonprojective_s3_zero": all(
            row["s3_zero"] in {None, True} for row in checks
        ),
        "final_endpoint": point_json(current),
        "steps": checks,
    }


def batched_c3_source(
    target: Point,
    a6_first: Mapping[Point, IndexSource],
    c3_first: Mapping[Point, IndexSource],
    curve: dict[str, Any],
) -> tuple[IndexSource, IndexSource, IndexSource] | None:
    for a6_endpoint, a6_source in a6_first.items():
        c_target = R70.add(
            target,
            R70.negate(a6_endpoint, curve),
            curve,
        )
        c_pair = convolution_source(c_target, c3_first, c3_first, curve)
        if c_pair is not None:
            return a6_source, c_pair[0], c_pair[1]
    return None


def target_control(
    target: Point,
    a6_hist: Mapping[Point, int],
    a6_first: Mapping[Point, IndexSource],
    c3_hist: Mapping[Point, int],
    c3_first: Mapping[Point, IndexSource],
    c6_hist: Mapping[Point, int],
    c6_first: Mapping[Point, IndexSource],
    f3_hist: Mapping[Point, int],
    f3_first: Mapping[
        Point,
        tuple[IndexSource, IndexSource],
    ],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    direct_count = convolution_count(target, a6_hist, c6_hist, curve)
    batched_count = 0
    for a6_endpoint, multiplicity in a6_hist.items():
        c_target = R70.add(
            target,
            R70.negate(a6_endpoint, curve),
            curve,
        )
        batched_count += multiplicity * convolution_count(
            c_target,
            c3_hist,
            c3_hist,
            curve,
        )
    self_count = convolution_count(target, f3_hist, f3_hist, curve)

    direct_source = convolution_source(
        target,
        a6_first,
        c6_first,
        curve,
    )
    batched_source = batched_c3_source(
        target,
        a6_first,
        c3_first,
        curve,
    )
    self_source = convolution_source(
        target,
        f3_first,
        f3_first,
        curve,
    )

    direct_replay = None
    if direct_source is not None:
        direct_replay = replay_atom_source(
            direct_source[0],
            direct_source[1],
            atoms_a,
            atoms_c,
            curve,
        )

    batched_replay = None
    batched_factor_replay = None
    batched_chain = None
    if batched_source is not None:
        source_a = batched_source[0]
        source_c = batched_source[1] + batched_source[2]
        batched_replay = replay_atom_source(
            source_a,
            source_c,
            atoms_a,
            atoms_c,
            curve,
        )
        batched_factor_replay = replay_factor_source(
            source_a,
            source_c,
            atoms_a,
            atoms_c,
            curve,
        )
        factors = [
            R70.add(atoms_a[index_a], atoms_c[index_c], curve)
            for index_a, index_c in zip(source_a, source_c)
        ]
        batched_chain = s3_chain_certificate(factors, curve)

    self_replay = None
    if self_source is not None:
        left, right = self_source
        source_a = left[0] + right[0]
        source_c = left[1] + right[1]
        self_replay = replay_atom_source(
            source_a,
            source_c,
            atoms_a,
            atoms_c,
            curve,
        )

    positive = direct_count > 0
    source_presence_exact = (
        (direct_source is not None)
        == (batched_source is not None)
        == (self_source is not None)
        == positive
    )
    return {
        "target": point_json(target),
        "direct_a6_c6_count": direct_count,
        "a6_batched_c3_pair_count": batched_count,
        "three_f_self_convolution_count": self_count,
        "all_three_counts_equal": (
            direct_count == batched_count == self_count
        ),
        "positive": positive,
        "source_presence_exact": source_presence_exact,
        "direct_source_replay_exact": (
            direct_replay == target if direct_source is not None else not positive
        ),
        "batched_atom_source_replay_exact": (
            batched_replay == target
            if batched_source is not None
            else not positive
        ),
        "batched_factor_source_replay_exact": (
            batched_factor_replay == target
            if batched_source is not None
            else not positive
        ),
        "self_convolution_source_replay_exact": (
            self_replay == target if self_source is not None else not positive
        ),
        "batched_source": (
            None
            if batched_source is None
            else {
                "a6_indices": list(batched_source[0]),
                "left_c3_indices": list(batched_source[1]),
                "right_c3_indices": list(batched_source[2]),
            }
        ),
        "factor_s3_chain": batched_chain,
    }


def first_absent_target(
    generator: Point,
    a6_hist: Mapping[Point, int],
    c6_hist: Mapping[Point, int],
    curve: dict[str, Any],
) -> Point | None:
    # The projective identity is encoded as None, so scan nonidentity points
    # first and reserve None as this helper's "not found" sentinel.
    for scalar in range(1, curve["subgroup_order"]):
        target = R70.scalar_mul(scalar, generator, curve)
        if convolution_count(target, a6_hist, c6_hist, curve) == 0:
            return target
    return None


def finite_instance(
    curve: dict[str, Any],
    *,
    include_absent: bool,
    deck_limits: tuple[int, int] | None = None,
) -> dict[str, Any]:
    generator = R81.curve_generator(curve)
    validation = R82.validate_family(curve, generator)
    if not all(validation.values()):
        raise AssertionError(f"invalid finite family {curve['family_id']}")
    atoms_a, atoms_c, factor_points, construction = R82.compact_factor_base(
        curve,
        0,
    )
    if deck_limits is not None:
        limit_a, limit_c = deck_limits
        atoms_a = atoms_a[:limit_a]
        atoms_c = atoms_c[:limit_c]
        factor_points = [
            R70.add(left, right, curve)
            for left in atoms_a
            for right in atoms_c
        ]
        construction = {
            **construction,
            "atom_a_size": len(atoms_a),
            "atom_c_size": len(atoms_c),
            "factor_base_size": len(factor_points),
            "finite_semantic_subdeck": True,
            "asymptotic_credit": False,
        }
    a3_hist, a3_first = ordered_endpoint_map(atoms_a, 3, curve)
    a6_hist, a6_first = ordered_endpoint_map(atoms_a, 6, curve)
    c3_hist, c3_first = ordered_endpoint_map(atoms_c, 3, curve)
    c6_hist, c6_first = ordered_endpoint_map(atoms_c, 6, curve)
    f3_hist = convolve_counters(a3_hist, c3_hist, curve)
    f3_first = convolve_first_sources(a3_first, c3_first, curve)

    if sum(a3_hist.values()) != len(atoms_a) ** 3:
        raise AssertionError("A3 occurrence mass drifted")
    if sum(a6_hist.values()) != len(atoms_a) ** 6:
        raise AssertionError("A6 occurrence mass drifted")
    if sum(c3_hist.values()) != len(atoms_c) ** 3:
        raise AssertionError("C3 occurrence mass drifted")
    if sum(c6_hist.values()) != len(atoms_c) ** 6:
        raise AssertionError("C6 occurrence mass drifted")
    if sum(f3_hist.values()) != len(factor_points) ** 3:
        raise AssertionError("3F occurrence mass drifted")

    targets: list[Point] = []
    for shift in range(3):
        source_a = tuple(
            (index + shift) % len(atoms_a) for index in range(6)
        )
        source_c = tuple(
            (2 * index + shift) % len(atoms_c) for index in range(6)
        )
        target = replay_atom_source(
            source_a,
            source_c,
            atoms_a,
            atoms_c,
            curve,
        )
        if target not in targets:
            targets.append(target)
    absent = (
        first_absent_target(generator, a6_hist, c6_hist, curve)
        if include_absent
        else None
    )
    if include_absent and absent is None:
        raise AssertionError("finite no-relation control unexpectedly absent")
    if absent is not None:
        targets.append(absent)

    controls = [
        target_control(
            target,
            a6_hist,
            a6_first,
            c3_hist,
            c3_first,
            c6_hist,
            c6_first,
            f3_hist,
            f3_first,
            atoms_a,
            atoms_c,
            curve,
        )
        for target in targets
    ]
    all_exact = all(
        row["all_three_counts_equal"]
        and row["source_presence_exact"]
        and row["direct_source_replay_exact"]
        and row["batched_atom_source_replay_exact"]
        and row["batched_factor_source_replay_exact"]
        and row["self_convolution_source_replay_exact"]
        and (
            row["factor_s3_chain"] is None
            or row["factor_s3_chain"]["all_nonprojective_s3_zero"]
        )
        for row in controls
    )
    if not all_exact:
        raise AssertionError(
            f"finite convolution identity failed for {curve['family_id']}"
        )
    return {
        "family_id": (
            curve["family_id"]
            if deck_limits is None
            else (
                f"{curve['family_id']}_subdeck_"
                f"u{len(atoms_a)}_v{len(atoms_c)}"
            )
        ),
        "curve": {
            "field_prime": curve["field_prime"],
            "curve_a": curve["curve_a"],
            "curve_b": curve["curve_b"],
            "subgroup_order": curve["subgroup_order"],
            "cofactor": curve["cofactor"],
        },
        "validation": validation,
        "factor_base": construction,
        "support_sizes": {
            "a3": len(a3_hist),
            "a6": len(a6_hist),
            "c3": len(c3_hist),
            "c6": len(c6_hist),
            "three_f": len(f3_hist),
        },
        "occurrence_masses": {
            "a3": sum(a3_hist.values()),
            "a6": sum(a6_hist.values()),
            "c3": sum(c3_hist.values()),
            "c6": sum(c6_hist.values()),
            "three_f": sum(f3_hist.values()),
            "six_f": len(factor_points) ** 6,
        },
        "target_controls": controls,
        "positive_control_count": sum(row["positive"] for row in controls),
        "negative_control_count": sum(not row["positive"] for row in controls),
        "all_exact": all_exact,
        "enumeration_receives_asymptotic_credit": False,
    }


def indexing_ledger() -> dict[str, Any]:
    dg_online_delta = Fraction(1, 3)
    dg_setup = C3_EXPONENT * (Fraction(5, 2) - dg_online_delta)
    dg_query = C3_EXPONENT * dg_online_delta
    dg_min_setup = C3_EXPONENT * Fraction(3, 2)
    fn_setup = C3_EXPONENT * (
        Fraction(2) - dg_online_delta / 3
    )
    preprocessed_min_space = C3_EXPONENT * Fraction(5, 3)
    preprocessed_min_query = C3_EXPONENT * Fraction(3, 2)
    rows = [
        {
            "route_id": "store_c3_scan_each_a6_target",
            "setup_exponent_B": fraction_record(C3_EXPONENT),
            "query_per_target_exponent_B": fraction_record(C3_EXPONENT),
            "batch_exponent_B": fraction_record(A6_EXPONENT + C3_EXPONENT),
            "inside_setup_cap": True,
            "inside_fresh_batch_cap": False,
            "exact_source_reporting": True,
            "transfer_to_prime_order_ec": True,
        },
        {
            "route_id": "store_all_c3_pair_sums",
            "setup_exponent_B": fraction_record(2 * C3_EXPONENT),
            "query_per_target_exponent_B": fraction_record(Fraction(0)),
            "batch_exponent_B": fraction_record(A6_EXPONENT),
            "inside_setup_cap": False,
            "inside_fresh_batch_cap": True,
            "exact_source_reporting": True,
            "transfer_to_prime_order_ec": True,
        },
        {
            "route_id": "dinur_golovnev_v2_online_compatible_boundary",
            "delta": fraction_record(dg_online_delta),
            "setup_exponent_B": fraction_record(dg_setup),
            "preprocessing_work_exponent_B": fraction_record(
                2 * C3_EXPONENT
            ),
            "query_per_target_exponent_B": fraction_record(dg_query),
            "batch_exponent_B": fraction_record(A6_EXPONENT + dg_query),
            "inside_setup_cap": dg_setup <= SETUP_CAP,
            "inside_fresh_batch_cap": (
                A6_EXPONENT + dg_query <= ONLINE_CAP
            ),
            "exact_source_reporting": True,
            "integer_residue_transfer_to_prime_order_ec": False,
        },
        {
            "route_id": "dinur_golovnev_v2_minimum_advice_endpoint",
            "delta": fraction_record(Fraction(1)),
            "setup_exponent_B": fraction_record(dg_min_setup),
            "query_per_target_exponent_B": fraction_record(C3_EXPONENT),
            "batch_exponent_B": fraction_record(A6_EXPONENT + C3_EXPONENT),
            "inside_setup_cap": False,
            "inside_fresh_batch_cap": False,
            "exact_source_reporting": True,
            "integer_residue_transfer_to_prime_order_ec": False,
        },
        {
            "route_id": "fiat_naor_2019_online_compatible_boundary",
            "delta": fraction_record(dg_online_delta),
            "setup_exponent_B": fraction_record(fn_setup),
            "query_per_target_exponent_B": fraction_record(dg_query),
            "batch_exponent_B": fraction_record(A6_EXPONENT + dg_query),
            "inside_setup_cap": fn_setup <= SETUP_CAP,
            "inside_fresh_batch_cap": True,
            "integer_residue_transfer_to_prime_order_ec": False,
        },
        {
            "route_id": "preprocessed_unknown_universe_2026_best_space",
            "epsilon": fraction_record(Fraction(1, 2)),
            "setup_exponent_B": fraction_record(preprocessed_min_space),
            "preprocessing_work_exponent_B": fraction_record(
                2 * C3_EXPONENT
            ),
            "query_exponent_B": fraction_record(2 * C3_EXPONENT),
            "inside_setup_cap": False,
            "inside_fresh_batch_cap": False,
            "model_matches_point_challenge": False,
            "integer_fft_transfer_to_prime_order_ec": False,
        },
        {
            "route_id": "preprocessed_unknown_universe_2026_best_query",
            "epsilon": fraction_record(Fraction(0)),
            "setup_exponent_B": fraction_record(2 * C3_EXPONENT),
            "preprocessing_work_exponent_B": fraction_record(
                2 * C3_EXPONENT
            ),
            "query_exponent_B": fraction_record(preprocessed_min_query),
            "inside_setup_cap": False,
            "inside_fresh_batch_cap": False,
            "model_matches_point_challenge": False,
            "integer_fft_transfer_to_prime_order_ec": False,
        },
    ]
    return {
        "schema": "p1553.c3_pair_sum_indexing_ledger.r116.v1",
        "normalization": {
            "c3_list_length_M": "B^(9/4+o(1))",
            "a6_target_batch_Q": "B^(1/2+o(1))",
            "allowed_average_query_work": "B^(3/4+o(1))",
            "setup_cap": "B^(9/4+o(1))",
            "fresh_batch_cap": "B^(5/4+o(1))",
        },
        "dinur_golovnev_v2": {
            "theorem": (
                "S=soft-O(n^(5/2-delta)), T=soft-O(n^delta), "
                "0<=delta<=1 for equal lists"
            ),
            "batch_cap_forces_delta_at_most": fraction_record(
                dg_online_delta
            ),
            "setup_cap_would_require_delta_at_least": fraction_record(
                Fraction(3, 2)
            ),
            "online_compatible_setup_exponent_B": fraction_record(dg_setup),
            "paper_preprocessing_exponent_B": fraction_record(
                2 * C3_EXPONENT
            ),
            "source": str(DINUR_GOLOVNEV),
            "sha256": DINUR_GOLOVNEV_SHA256,
        },
        "kirkpatrick_et_al_v1": {
            "theorem": (
                "for 0<=epsilon<=1/2, preprocessing n^2, space "
                "n^(2-2epsilon/3), query n^(3/2+epsilon)"
            ),
            "source": str(KIRKPATRICK_ET_AL),
            "sha256": KIRKPATRICK_ET_AL_SHA256,
            "stronger_different_problem": (
                "queries subsets A',B' and an unknown set C' of size O(n), "
                "not one point challenge"
            ),
        },
        "prime_order_transfer_boundary": {
            "generic_group_order": "prime q=B^(5+o(1))",
            "proper_small_residue_homomorphism_is_trivial": True,
            "public_ec_coordinate_encoding_is_addition_compatible": False,
            "dlog_labels_make_integer_residues_compatible": True,
            "dlog_labels_available_to_candidate": False,
        },
        "routes": rows,
        "any_bound_indexing_route_meets_both_caps": any(
            row.get("inside_setup_cap", False)
            and row.get("inside_fresh_batch_cap", False)
            and (
                row.get("transfer_to_prime_order_ec", False)
                or row.get(
                    "integer_residue_transfer_to_prime_order_ec",
                    False,
                )
            )
            for row in rows
        ),
        "polynomial_data_structure_lower_bound_claimed": False,
        "candidate_work_credit": False,
    }


def algebraic_cost_ledger() -> dict[str, Any]:
    rows = [
        {
            "route_id": "explicit_c3_endpoint_divisor",
            "state_exponent_B": fraction_record(C3_EXPONENT),
            "inside_setup_cap": True,
            "locator_complete": False,
        },
        {
            "route_id": "one_explicit_target_translated_c3_divisor_gcd",
            "fresh_work_exponent_B": fraction_record(C3_EXPONENT),
            "inside_fresh_cap": False,
            "source_dictionary_exponent_B": fraction_record(C3_EXPONENT),
        },
        {
            "route_id": "a6_batch_of_explicit_translated_divisor_gcds",
            "fresh_work_exponent_B": fraction_record(
                A6_EXPONENT + C3_EXPONENT
            ),
            "inside_fresh_cap": False,
        },
        {
            "route_id": "materialized_c3_pair_sum_resultant_or_chow_form",
            "occurrence_degree_exponent_B": fraction_record(
                2 * C3_EXPONENT
            ),
            "inside_setup_cap": False,
        },
        {
            "route_id": "materialized_c3_c3_a6_incidence_pushforward",
            "occurrence_degree_exponent_B": fraction_record(
                2 * C3_EXPONENT + A6_EXPONENT
            ),
            "inside_setup_cap": False,
            "equals_group_order_exponent": (
                2 * C3_EXPONENT + A6_EXPONENT
                == LOG_B_GROUP_ORDER
            ),
        },
        {
            "route_id": "sparse_group_algebra_c3_square",
            "output_occurrence_exponent_B": fraction_record(
                2 * C3_EXPONENT
            ),
            "inside_setup_cap": False,
            "requires_scalar_exponents_for_cyclic_polynomial_encoding": True,
        },
        {
            "route_id": "dense_group_fourier_transform",
            "mode_count_exponent_B": fraction_record(LOG_B_GROUP_ORDER),
            "inside_setup_cap": False,
            "requires_character_or_dlog_labels": True,
        },
        {
            "route_id": "balanced_three_f_meet_in_middle",
            "work_exponent_B": fraction_record(RHO_EXPONENT),
            "state_exponent_B": fraction_record(RHO_EXPONENT),
            "strictly_below_rho": False,
        },
    ]
    return {
        "schema": "p1553.c3_pair_sum_algebraic_cost_ledger.r116.v1",
        "summation_polynomial_degrees": {
            "factor_level_s7": {
                "arity": 7,
                "degree_per_variable": 32,
                "total_degree": 192,
            },
            "expanded_atom_level_s13": {
                "arity": 13,
                "degree_per_variable": 2048,
                "total_degree": 24576,
            },
            "fixed_degree_has_B_exponent_zero": True,
            "fixed_degree_is_not_a_source_locator": True,
        },
        "endpoint_bodies": {
            "a3_exponent_B": fraction_record(A3_EXPONENT),
            "a6_exponent_B": fraction_record(A6_EXPONENT),
            "c3_exponent_B": fraction_record(C3_EXPONENT),
            "c6_exponent_B": fraction_record(C6_EXPONENT),
            "c3_pair_occurrence_exponent_B": fraction_record(
                2 * C3_EXPONENT
            ),
            "full_source_occurrence_exponent_B": fraction_record(
                2 * C3_EXPONENT + A6_EXPONENT
            ),
        },
        "routes": rows,
        "scoped_route_boundary": (
            "standard explicit coefficient, divisor translation, resultant "
            "output, sparse group-algebra output, and dense Fourier routes"
        ),
        "not_excluded": [
            "a jointly transposed target-batched elliptic coefficient functional",
            "a source-returning FFE contraction that never emits C3+C3",
            "a representation-specific low-rank addition pushforward",
        ],
        "unconditional_lower_bound_claimed": False,
        "candidate_work_credit": False,
    }


def finite_replay_bundle() -> dict[str, Any]:
    instances = [
        finite_instance(R82.FAMILIES[0], include_absent=False),
        finite_instance(
            R82.FAMILIES[0],
            include_absent=True,
            deck_limits=(1, 2),
        ),
        finite_instance(R82.FAMILIES[1], include_absent=False),
    ]
    return {
        "schema": "p1553.m6_a6_c3_self_convolution_source_replay.r116.v1",
        "identity": (
            "mu_A^*6 * mu_C^*6 = "
            "(mu_A^*3 * mu_C^*3) * (mu_A^*3 * mu_C^*3)"
        ),
        "batched_form": (
            "sum_a6 mu_A6(a6) "
            "(mu_C3*mu_C3)(R-a6)"
        ),
        "source_rule": (
            "one A6 backpointer plus two C3 backpointers yields six "
            "ordered factors F_i=A_i+C_i"
        ),
        "instances": instances,
        "all_instances_exact": all(row["all_exact"] for row in instances),
        "negative_control_present": any(
            row["negative_control_count"] > 0 for row in instances
        ),
        "finite_enumeration_receives_asymptotic_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    replay = finite_replay_bundle()
    indexing = indexing_ledger()
    algebraic = algebraic_cost_ledger()
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r115_vertex_exactly_imported": (
            ALPHA == Fraction(1, 12)
            and BETA == Fraction(3, 4)
        ),
        "a6_and_c3_exponents_exact": (
            A6_EXPONENT == Fraction(1, 2)
            and C3_EXPONENT == Fraction(9, 4)
        ),
        "finite_three_form_count_identity_exact": replay[
            "all_instances_exact"
        ],
        "finite_negative_control_present": replay[
            "negative_control_present"
        ],
        "six_coupled_factor_source_replay_exact": replay[
            "all_instances_exact"
        ],
        "dinur_golovnev_online_boundary_exact": (
            indexing["dinur_golovnev_v2"][
                "online_compatible_setup_exponent_B"
            ]["exact"]
            == "39/8"
        ),
        "current_bound_indexing_routes_rejected": not indexing[
            "any_bound_indexing_route_meets_both_caps"
        ],
        "standard_algebraic_route_costs_charged": True,
        "implicit_target_batched_coefficient_functional_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "full_source_to_target_cost_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one exact target-batched elliptic coefficient "
        "functional for sum_{a6} mu_A6(a6) "
        "(mu_C3*mu_C3)(R-a6). It may store the B^(9/4) C3 endpoint "
        "divisor and enumerate the B^(1/2) A6 target batch, but must use "
        "at most B^(5/4+o(1)) fresh work and workspace, never emit the "
        "B^(9/2) C3 pair sum or B^(11/4) translated-divisor batch, use no "
        "DLP/residue labels or determinant oracle, and return one A6 plus "
        "two C3 occurrence backpointers on every positive projective branch. "
        "Then apply the same operator to relation collection and target "
        "descent and pass known-RHS rank and factor-log verification."
    )
    frozen = {
        "schema": "p1553.frozen_m6_c3_pair_sum_batch_interface.r116.v1",
        "source_bindings": source_binding_records(),
        "vertex": {
            "relation_arity_m": 6,
            "alpha_A_exponent_B": fraction_record(ALPHA),
            "beta_C_exponent_B": fraction_record(BETA),
            "group_order_exponent_B": fraction_record(LOG_B_GROUP_ORDER),
        },
        "exact_interface": {
            "preprocessed_multiset": "C3=mu_C^*3",
            "preprocessed_occurrence_exponent_B": fraction_record(
                C3_EXPONENT
            ),
            "fresh_target_batch": "{R-a6 : a6 in A6}",
            "fresh_target_batch_exponent_B": fraction_record(A6_EXPONENT),
            "average_query_allowance_exponent_B": fraction_record(
                ONLINE_CAP - A6_EXPONENT
            ),
            "requested_coefficient": (
                "sum_a6 mu_A6(a6)(mu_C3*mu_C3)(R-a6)"
            ),
            "requested_source": "one A6 and two C3 occurrence backpointers",
        },
        "transfer_boundary": indexing["prime_order_transfer_boundary"],
        "novelty_boundary": {
            "new_exact_reduction": (
                "R115 implicit 3F self-convolution is the A6-batched "
                "pair-sum indexing problem on the C3 endpoint multiset"
            ),
            "new_algorithm_claimed": False,
            "unconditional_lower_bound_claimed": False,
        },
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r116.v1",
        "exact_interface_reduction_complete": True,
        "finite_source_replay_complete": replay["all_instances_exact"],
        "implicit_target_batched_coefficient_functional_complete": False,
        "relation_independence_theorem_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_log_solve_complete": False,
        "factor_log_verification_complete": False,
        "fresh_target_descent_complete": False,
        "identical_algorithm_used_for_relation_and_descent": False,
        "full_source_to_target_cost_complete": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_INTERFACE_REDUCTION_AND_SCOPED_ROUTE_AUDIT_ONLY_"
            "WITHHOLD_PROMOTION"
        ),
        "classification": (
            "M6_SELF_CONVOLUTION_EXACTLY_REDUCES_TO_A6_BATCH_OF_C3_PAIR_"
            "SUM_QUERIES__C3_STATE_B9O4_AND_BATCH_B1O2__AVERAGE_QUERY_"
            "ALLOWANCE_B3O4__DINUR_GOLOVNEV_ONLINE_BOUNDARY_SETUP_B39O8_"
            "AND_INTEGER_RESIDUE_TRANSFER_ABSENT__PREPROCESSED_UNIVERSE_"
            "N2_SETUP_FAIL__STANDARD_TRANSLATED_DIVISOR_BATCH_B11O4__"
            "PAIR_RESULTANT_B9O2__DENSE_FOURIER_B5__FINITE_PROJECTIVE_"
            "COUNTS_AND_SOURCES_EXACT__TRANSPOSED_COEFFICIENT_FUNCTIONAL_"
            "RANK_LOGS_DESCENT_OPEN"
        ),
        "source_bindings": source_binding_records(),
        "exact_reduction": frozen["exact_interface"],
        "finite_evidence": {
            "instance_count": len(replay["instances"]),
            "all_instances_exact": replay["all_instances_exact"],
            "negative_control_present": replay["negative_control_present"],
            "asymptotic_credit": False,
        },
        "cost_boundary": {
            "setup_cap_exponent_B": fraction_record(SETUP_CAP),
            "fresh_cap_exponent_B": fraction_record(ONLINE_CAP),
            "rho_exponent_B": fraction_record(RHO_EXPONENT),
            "bound_indexing_route_meets_both_caps": indexing[
                "any_bound_indexing_route_meets_both_caps"
            ],
            "standard_algebraic_routes_scoped_only": True,
            "inside_cap_algorithm_constructed": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "exact_interface_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen_interface": (
                "frozen_m6_c3_pair_sum_batch_interface.json"
            ),
            "indexing_ledger": (
                "c3_pair_sum_indexing_and_algebraic_cost_ledger.json"
            ),
            "source_replay": (
                "m6_a6_c3_self_convolution_source_replay.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r116.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Finite enumeration proves semantics, not asymptotic runtime.",
            "Failure of current indexing curves is not a data-structure lower bound.",
            "Standard explicit algebraic route costs do not exclude a new transpose.",
            "Fixed S7 and S13 degrees do not supply a coefficient functional.",
            "No relation-rank theorem, factor logs, or identical descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_EXACT_A6_BATCHED_C3_PAIR_SUM_REDUCTION_AND_FINITE_SOURCE_"
            "REPLAY_ONLY__REJECT_CURRENT_BOUND_INDEXING_AND_STANDARD_EXPLICIT_"
            "DIVISOR_RESULTANT_GROUP_ALGEBRA_ROUTES_AT_FROZEN_CAPS__PRESERVE_"
            "TARGET_BATCHED_TRANSPOSED_ELLIPTIC_COEFFICIENT_FUNCTIONAL__NO_"
            "LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__"
            "NO_BREAKTHROUGH"
        ),
    }
    combined_cost = {
        "schema": "p1553.c3_pair_sum_combined_cost_ledger.r116.v1",
        "indexing": indexing,
        "algebraic": algebraic,
        "candidate_work_credit": False,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": combined_cost,
        "replay": replay,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_m6_a6_batched_c3_pair_sum_source_"
            "locator_probe_report_r116.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_m6_c3_pair_sum_batch_interface.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "c3_pair_sum_indexing_and_algebraic_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_a6_c3_self_convolution_source_replay.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r116.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.logs_output, bundle["logs_descent"])
    report = bundle["report"]
    admission = report["admission"]
    print(
        f"R116 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
