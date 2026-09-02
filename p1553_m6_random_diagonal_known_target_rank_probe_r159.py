#!/usr/bin/env python3
"""Prove direct-target coverage and random-diagonal M6 relation rank."""

from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_random_diagonal_known_target_rank.r159.v1"

R158_PRODUCER = ROOT / (
    "p1553_m6_short_relation_near_injectivity_supply_probe_r158.py"
)
R158_REPORT = ROOT / (
    "p1553_m6_short_relation_near_injectivity_supply_probe_report_r158.json"
)
R158_FROZEN = ROOT / (
    "frozen_m6_short_relation_near_injectivity_supply.json"
)
R158_COST = ROOT / (
    "m6_short_relation_near_injectivity_supply_cost_ledger.json"
)
R158_REPLAY = ROOT / (
    "m6_short_relation_near_injectivity_supply_replay.json"
)
R158_CONTROLS = ROOT / (
    "m6_short_relation_near_injectivity_supply_controls.json"
)
R158_LOGS = ROOT / "factor_logs_and_identical_descent_r158.json"
R158_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_short_relation_near_injectivity_supply_probe_r158.py"
)
R158_GATE = ROOT / (
    "p1553_m6_short_relation_near_injectivity_supply_probe_gate_r158.md"
)
R158_PARENT = ROOT / (
    "p1553_m6_short_relation_near_injectivity_supply_probe_parent_report_r158.yaml"
)

SOURCE_BINDINGS = (
    (
        "r158_producer",
        R158_PRODUCER,
        "c68c0a8644117133b7dfaa51564e93a44c2fa614e66c899524ae941a3673439b",
    ),
    (
        "r158_report",
        R158_REPORT,
        "bc0e0e864af69c8c03fd21b0bb360adceea7247daf940f5cc5cbf0e51ad30c6a",
    ),
    (
        "r158_frozen",
        R158_FROZEN,
        "e278b905703403e30d223d611a36e7eb5cb857c4ef4cf50acab24a81eab8c8e5",
    ),
    (
        "r158_cost",
        R158_COST,
        "eac09e1950a431ae71c675210eeda5bacb797bbc6aa83cc32ff7fc9b06497f22",
    ),
    (
        "r158_replay",
        R158_REPLAY,
        "055d2a9887db144eed8e4791270522aac82a4760d5fa96ed4dec4b67e9fcb968",
    ),
    (
        "r158_controls",
        R158_CONTROLS,
        "af9095b0b7cba03fc3b6dd454f99672a5e7999c4899a34b30d4f8d5dcc2e13de",
    ),
    (
        "r158_logs",
        R158_LOGS,
        "8e1f858b3a49cf92c92c33a73ea9fb1c5133b8cd3eaa8f9939d60e96d74127af",
    ),
    (
        "r158_test",
        R158_TEST,
        "1f6134c23a8013c66d24a1d672a0e68faab615823872b9878a2af60f87b49582",
    ),
    (
        "r158_gate",
        R158_GATE,
        "20077d3231e0c535af2fe7a4c435148de53fe1da300e1174d1959e89f12a203d",
    ),
    (
        "r158_parent",
        R158_PARENT,
        "6a88c2a881948d474248f15e91f826109bf97c46fc7c2cfe35f044f300530f89",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_random_diagonal_known_target_rank_probe_report_r159.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_random_diagonal_known_target_rank.json"
)
DEFAULT_COST = ROOT / (
    "m6_random_diagonal_known_target_rank_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_random_diagonal_known_target_rank_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_random_diagonal_known_target_rank_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r159.json"

ARITY = 6
QUERY_CONSTANT = 8
OFFSETS = (0, 1)
SEEDS = (15901, 15902)
FAMILY_COUNT = 3


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R158 = load_module("p1553_r158_for_r159", R158_PRODUCER)
R157 = R158.R157
R144 = R157.R144
R82 = R157.R82
R81 = R157.R81
R70 = R157.R70


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {
        name: sha256_file(path)
        for name, path, _ in SOURCE_BINDINGS
    }
    failures = [
        name
        for name, _, expected in SOURCE_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R159 source binding mismatch: {failures}")
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


def deterministic_scalar(
    role: str, modulus: int, *parts: Any
) -> int:
    payload = "|".join(("R159", role, *(str(part) for part in parts)))
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest(), "big"
    ) % modulus


def positive_six_source_count(dimension: int) -> int:
    return math.comb(dimension + ARITY - 1, ARITY)


def point_sum(
    points: Iterable[Any], curve: dict[str, Any]
):
    return R157.point_sum(points, curve)


def positive_endpoint_structure(
    representatives: tuple[Any, ...],
    curve: dict[str, Any],
) -> dict[str, Any]:
    dimension = len(representatives)
    endpoints: dict[Any, list[tuple[int, ...]]] = (
        collections.defaultdict(list)
    )
    for indices in itertools.combinations_with_replacement(
        range(dimension), ARITY
    ):
        counts = R157.R155.multiplicity_vector(indices, dimension)
        endpoint = point_sum(
            (representatives[index] for index in indices), curve
        )
        endpoints[endpoint].append(counts)

    unique = {
        endpoint: sources[0]
        for endpoint, sources in endpoints.items()
        if len(sources) == 1
    }
    collision_pairs = sum(
        math.comb(len(sources), 2)
        for sources in endpoints.values()
    )
    bad_sources = sum(
        len(sources)
        for sources in endpoints.values()
        if len(sources) > 1
    )
    source_count = sum(len(sources) for sources in endpoints.values())
    if source_count != positive_six_source_count(dimension):
        raise AssertionError("positive C6 source count changed")
    return {
        "endpoints": dict(endpoints),
        "unique_endpoints": unique,
        "source_count": source_count,
        "endpoint_count": len(endpoints),
        "unique_endpoint_count": len(unique),
        "collision_pair_count": collision_pairs,
        "bad_source_count": bad_sources,
        "bad_source_fraction": bad_sources / source_count,
        "multiplicity_histogram": dict(
            sorted(
                collections.Counter(
                    len(sources) for sources in endpoints.values()
                ).items()
            )
        ),
        "sha256": sha256_json(
            [
                {
                    "point": R157.point_record(endpoint),
                    "sources": sources,
                }
                for endpoint, sources in sorted(
                    endpoints.items(),
                    key=lambda item: R157.point_sort_key(item[0]),
                )
            ]
        ),
    }


def target_query_cap(
    subgroup_order: int, source_count: int, dimension: int
) -> int:
    return math.ceil(
        QUERY_CONSTANT
        * subgroup_order
        * math.log(max(dimension, 2))
        / source_count
    )


def relation_for_column(
    *,
    control_id: str,
    column: int,
    representatives: tuple[Any, ...],
    generator: Any,
    curve: dict[str, Any],
    unique_endpoints: dict[Any, tuple[int, ...]],
    query_cap: int,
) -> dict[str, Any]:
    order = int(curve["subgroup_order"])
    shift = deterministic_scalar(
        "diagonal-shift", order, control_id, column
    )
    shift_point = R70.scalar_mul(
        shift, representatives[column], curve
    )
    for query_index in range(query_cap):
        known_scalar = deterministic_scalar(
            "known-target", order, control_id, column, query_index
        )
        known_point = R70.scalar_mul(known_scalar, generator, curve)
        target = R70.add(known_point, shift_point, curve)
        source = unique_endpoints.get(target)
        if source is None:
            continue
        row = list(source)
        row[column] -= shift
        row_tuple = tuple(row)
        exact = (
            R157.row_point(row_tuple, representatives, curve)
            == known_point
        )
        return {
            "covered": True,
            "column": column,
            "diagonal_shift": shift,
            "known_rhs_scalar": known_scalar,
            "query_count": query_index + 1,
            "source": source,
            "row_integer": row_tuple,
            "row_mod_order": tuple(value % order for value in row),
            "public_relation_identity_exact": exact,
        }
    return {
        "covered": False,
        "column": column,
        "diagonal_shift": shift,
        "known_rhs_scalar": None,
        "query_count": query_cap,
        "source": None,
        "row_integer": None,
        "row_mod_order": None,
        "public_relation_identity_exact": False,
    }


def target_descent(
    *,
    control_id: str,
    representatives: tuple[Any, ...],
    generator: Any,
    curve: dict[str, Any],
    unique_endpoints: dict[Any, tuple[int, ...]],
    recovered_logs: list[int],
    query_cap: int,
) -> dict[str, Any]:
    order = int(curve["subgroup_order"])
    verifier_secret = deterministic_scalar(
        "descent-secret", order, control_id
    )
    target = R70.scalar_mul(verifier_secret, generator, curve)
    for query_index in range(query_cap):
        known_scalar = deterministic_scalar(
            "descent-shift", order, control_id, query_index
        )
        known_point = R70.scalar_mul(known_scalar, generator, curve)
        endpoint = R70.add(target, known_point, curve)
        source = unique_endpoints.get(endpoint)
        if source is None:
            continue
        candidate = (
            sum(
                count * logarithm
                for count, logarithm in zip(source, recovered_logs)
            )
            - known_scalar
        ) % order
        return {
            "success": (
                R70.scalar_mul(candidate, generator, curve) == target
            ),
            "query_count": query_index + 1,
            "known_shift_scalar": known_scalar,
            "source": source,
            "candidate_logarithm": candidate,
            "candidate_equals_verifier_secret": (
                candidate == verifier_secret
            ),
            "public_scalar_verification": (
                R70.scalar_mul(candidate, generator, curve) == target
            ),
            "verifier_secret_not_used_by_candidate": True,
        }
    return {
        "success": False,
        "query_count": query_cap,
        "known_shift_scalar": None,
        "source": None,
        "candidate_logarithm": None,
        "candidate_equals_verifier_secret": False,
        "public_scalar_verification": False,
        "verifier_secret_not_used_by_candidate": True,
    }


def direct_target_control(
    curve: dict[str, Any], offset: int, seed: int
) -> dict[str, Any]:
    order = int(curve["subgroup_order"])
    generator = R81.curve_generator(curve)
    pairs = R157.c_point_pairs(curve, offset)
    representatives = tuple(positive for positive, _ in pairs)
    dimension = len(representatives)
    control_id = (
        f"{curve['family_id']}_offset{offset}_seed{seed}_d{dimension}"
    )
    endpoints = positive_endpoint_structure(representatives, curve)
    source_count = endpoints["source_count"]
    query_cap = target_query_cap(order, source_count, dimension)

    relations = [
        relation_for_column(
            control_id=control_id,
            column=column,
            representatives=representatives,
            generator=generator,
            curve=curve,
            unique_endpoints=endpoints["unique_endpoints"],
            query_cap=query_cap,
        )
        for column in range(dimension)
    ]
    covered = all(record["covered"] for record in relations)
    rows = [
        list(record["row_mod_order"])
        for record in relations
        if record["covered"]
    ]
    rhs = [
        int(record["known_rhs_scalar"])
        for record in relations
        if record["covered"]
    ]
    rank = R81.rank_mod(rows, order) if rows else 0
    full_rank = covered and rank == dimension
    recovered_logs: list[int] | None = None
    logs_verify = False
    descent: dict[str, Any] = {
        "success": False,
        "query_count": 0,
        "verifier_secret_not_used_by_candidate": True,
    }
    if full_rank:
        recovered_logs = R144.solve_square_mod(rows, rhs, order)
        logs_verify = all(
            R70.scalar_mul(logarithm, generator, curve) == point
            for logarithm, point in zip(
                recovered_logs, representatives
            )
        )
        descent = target_descent(
            control_id=control_id,
            representatives=representatives,
            generator=generator,
            curve=curve,
            unique_endpoints=endpoints["unique_endpoints"],
            recovered_logs=recovered_logs,
            query_cap=query_cap,
        )

    unique_ratio = Fraction(
        endpoints["unique_endpoint_count"], source_count
    )
    expected_collision_pairs = Fraction(
        source_count * (source_count - 1), 2 * order
    )
    iid_bad_unique_support_bound = Fraction(
        2 * source_count, order
    )
    finite_no_coverage_bound = (
        dimension
        * math.exp(
            -endpoints["unique_endpoint_count"] * query_cap / order
        )
    )
    return {
        "control_id": control_id,
        "family_id": curve["family_id"],
        "offset": offset,
        "seed": seed,
        "field_prime": int(curve["field_prime"]),
        "subgroup_order": order,
        "factor_base_dimension": dimension,
        "positive_c6_source_count": source_count,
        "positive_c6_source_count_formula_exact": (
            source_count == math.comb(dimension + 5, 6)
        ),
        "positive_c6_endpoint_count": endpoints["endpoint_count"],
        "positive_c6_unique_endpoint_count": endpoints[
            "unique_endpoint_count"
        ],
        "positive_c6_unique_source_fraction": fraction_record(
            unique_ratio
        ),
        "positive_c6_collision_pair_count": endpoints[
            "collision_pair_count"
        ],
        "positive_c6_bad_source_count": endpoints["bad_source_count"],
        "positive_c6_bad_source_fraction": endpoints[
            "bad_source_fraction"
        ],
        "positive_c6_multiplicity_histogram": endpoints[
            "multiplicity_histogram"
        ],
        "positive_c6_endpoint_sha256": endpoints["sha256"],
        "iid_expected_collision_pair_count": fraction_record(
            expected_collision_pairs
        ),
        "iid_probability_unique_support_below_half_bound": (
            fraction_record(iid_bad_unique_support_bound)
        ),
        "query_constant": QUERY_CONSTANT,
        "known_target_query_cap_per_column": query_cap,
        "known_target_query_count_total": sum(
            record["query_count"] for record in relations
        ),
        "finite_union_no_coverage_bound": finite_no_coverage_bound,
        "covered_column_count": sum(
            record["covered"] for record in relations
        ),
        "uncovered_column_count": sum(
            not record["covered"] for record in relations
        ),
        "all_public_relation_identities_exact": all(
            record["public_relation_identity_exact"]
            for record in relations
        ),
        "relation_rows": relations,
        "relation_matrix_sha256": sha256_json(rows),
        "random_diagonal_shift_sha256": sha256_json(
            [record["diagonal_shift"] for record in relations]
        ),
        "relation_rank_mod_subgroup_order": rank,
        "full_rank": full_rank,
        "recovered_factor_logs": recovered_logs,
        "factor_logs_publicly_verified": logs_verify,
        "identical_positive_c6_target_descent": descent,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": (
            False
        ),
        "finite_explicit_endpoint_enumeration_receives_attack_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "positive_c6_source_universe": (
            "For d positive factor-base representatives, the unordered "
            "six-source universe has M=binomial(d+5,6)=Theta(d^6)."
        ),
        "unique_endpoint_support": (
            "For iid uniform cyclic labels, the expected number of "
            "distinct-source collision pairs is at most M(M-1)/(2q). "
            "Every non-singleton source belongs to at least one such pair, "
            "so Markov gives Pr[unique endpoints < M/2] <= 2M/q."
        ),
        "zero_uncovered_columns": (
            "Choose an independent uniform diagonal shift s_j and T "
            "independent known scalars t for every column j, and query the "
            "positive-C6 locator at tG+[s_j]C_j. Conditional on at least "
            "M/2 unique endpoints, T=ceil(8q log(d)/M) gives total "
            "uncovered-column probability at most d*exp(-MT/(2q)) "
            "<= d^(-3)."
        ),
        "source_shift_independence": (
            "For fixed C_j and s_j, uniform t makes t+s_j log_G(C_j) "
            "uniform. Hence the first successful unique source and the "
            "success event are independent of s_j."
        ),
        "random_diagonal_full_rank": (
            "The selected relation matrix is R=V-diag(s_1,...,s_d). "
            "Conditional on V and coverage, det(R) is a nonzero "
            "multilinear polynomial with leading monomial "
            "(-1)^d product_j s_j. The finite-field polynomial zero bound "
            "therefore gives Pr[rank(R)<d] <= d/q."
        ),
        "combined_iid_failure": (
            "With d=B^(3/4), M=B^(9/2), and q=B^5, coverage and rank "
            "fail with probability O(B^(-1/2))+O(B^(-9/4))"
            "+O(B^(-17/4))."
        ),
        "conditioned_sampler_transfer": (
            "Conditioning iid labels to be nonzero and pairwise distinct "
            "up to sign exactly gives the ideal rejection-sampled "
            "hash-to-curve factor-base law. Its conditioning event fails "
            "with probability O(d^2/q)=O(B^(-7/2)), so the iid theorem "
            "transfers with only the reciprocal conditioning-probability "
            "factor."
        ),
        "identical_descent": (
            "After solving factor logs, query the identical positive-C6 "
            "locator at Q+tG. A unique source v gives "
            "log_G(Q)=sum_i v_i log_G(C_i)-t with the same "
            "T=Theta(B^(1/2)log B) query bound."
        ),
        "scope": (
            "The probability theorem is exact for iid cyclic labels and "
            "the conditioned ideal random-oracle sampler. Deterministic "
            "hash-to-curve pseudorandomness and the required batched "
            "reverse FFE unique-source locator remain unproved."
        ),
        "novelty_status": (
            "random_diagonal_direct_known_target_rank_novelty_unverified"
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_random_diagonal_known_target_rank.cost.r159.v1"
        ),
        "factor_base_dimension_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "positive_c6_source_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "group_order_exponent_B": fraction_record(Fraction(5)),
        "known_targets_per_column_exponent_B": fraction_record(
            Fraction(1, 2)
        ),
        "known_targets_per_column_polylog_factor": "log(B)",
        "all_column_known_target_count_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "all_column_known_target_count_polylog_factor": "log(B)",
        "relation_matrix_row_count_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "dense_factor_log_solve_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "sparse_matrix_solve_exponent_B": fraction_record(
            Fraction(3, 2)
        ),
        "explicit_positive_c6_enumeration_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "conditional_batched_locator_work_cap_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "query_generation_and_output_fit_conditional_cap": True,
        "explicit_control_exceeds_pollard_rho": True,
        "reverse_only_signed_marker_operator_supplied": False,
        "batched_positive_c6_unique_source_locator_supplied": False,
        "deterministic_hash_to_curve_transfer_supplied": False,
        "unconditional_total_attack_cost_supplied": False,
        "finite_explicit_controls_receive_attack_credit": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    controls_list = [
        direct_target_control(curve, offset, seed)
        for curve in R82.FAMILIES[:FAMILY_COUNT]
        for offset in OFFSETS
        for seed in SEEDS
    ]
    control_count = len(controls_list)
    full_rank_count = sum(row["full_rank"] for row in controls_list)
    log_count = sum(
        row["factor_logs_publicly_verified"] for row in controls_list
    )
    descent_count = sum(
        row["identical_positive_c6_target_descent"]["success"]
        for row in controls_list
    )
    all_public_exact = all(
        row["all_public_relation_identities_exact"]
        for row in controls_list
    )
    all_covered = all(
        row["uncovered_column_count"] == 0 for row in controls_list
    )
    controls = {
        "schema": (
            "p1553.m6_random_diagonal_known_target_rank.controls.r159.v1"
        ),
        "query_constant": QUERY_CONSTANT,
        "family_count": FAMILY_COUNT,
        "offsets": list(OFFSETS),
        "seeds": list(SEEDS),
        "control_count": control_count,
        "all_columns_covered_control_count": sum(
            row["uncovered_column_count"] == 0
            for row in controls_list
        ),
        "full_rank_control_count": full_rank_count,
        "publicly_verified_factor_log_control_count": log_count,
        "successful_identical_descent_control_count": descent_count,
        "all_public_relation_identities_exact": all_public_exact,
        "finite_controls_receive_asymptotic_credit": False,
        "controls": controls_list,
    }
    cost = cost_record()
    theorem = theorem_record()
    obligations = {
        "ten_source_bindings_verified": len(actual_bindings) == 10,
        "positive_c6_source_formula_proved": True,
        "unique_endpoint_support_probability_bound_proved": True,
        "direct_known_target_query_bound_proved": True,
        "zero_uncovered_columns_theorem_complete": True,
        "source_shift_independence_proved": True,
        "random_diagonal_determinant_polynomial_proved": True,
        "full_rank_probability_bound_proved": True,
        "ideal_conditioned_hash_to_curve_transfer_complete": True,
        "deterministic_hash_to_curve_transfer_complete": False,
        "identical_target_descent_reduction_complete": True,
        "twelve_public_curve_controls_complete": control_count == 12,
        "all_public_relation_identities_exact": all_public_exact,
        "all_finite_columns_covered": all_covered,
        "all_finite_relation_matrices_full_rank": (
            full_rank_count == control_count
        ),
        "all_finite_factor_logs_publicly_verified": (
            log_count == control_count
        ),
        "all_finite_identical_descents_verified": (
            descent_count == control_count
        ),
        "candidate_discrete_log_oracle_avoided": all(
            not row["candidate_discrete_log_oracle_consumed"]
            for row in controls_list
        ),
        "all_column_target_count_B5O4_charged": True,
        "explicit_c6_enumeration_B9O2_charged": True,
        "finite_controls_scoped_without_attack_credit": True,
        "batched_positive_c6_unique_source_locator_complete": False,
        "reverse_only_signed_marker_operator_complete": False,
        "signed_weight_separable_ffe_dag_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    admission = {
        "obligations": obligations,
        "passed_obligation_count": passed,
        "obligation_count": len(obligations),
        "zero_coverage_and_full_rank_theorem_admitted": True,
        "ideal_conditioned_sampler_transfer_admitted": True,
        "identical_descent_reduction_admitted": True,
        "batched_positive_c6_unique_source_locator_admitted": False,
        "reverse_only_signed_marker_operator_admitted": False,
        "lane_admitted": False,
    }
    replay = {
        "schema": (
            "p1553.m6_random_diagonal_known_target_rank.replay.r159.v1"
        ),
        "source_bindings": source_binding_records(),
        "control_count": control_count,
        "control_records": [
            {
                "control_id": row["control_id"],
                "endpoint_sha256": row[
                    "positive_c6_endpoint_sha256"
                ],
                "relation_matrix_sha256": row[
                    "relation_matrix_sha256"
                ],
                "random_diagonal_shift_sha256": row[
                    "random_diagonal_shift_sha256"
                ],
                "covered_column_count": row["covered_column_count"],
                "rank": row["relation_rank_mod_subgroup_order"],
                "factor_logs_verified": row[
                    "factor_logs_publicly_verified"
                ],
                "identical_descent_verified": row[
                    "identical_positive_c6_target_descent"
                ]["success"],
            }
            for row in controls_list
        ],
        "all_replay_invariants_pass": (
            all_public_exact
            and all_covered
            and full_rank_count == control_count
            and log_count == control_count
            and descent_count == control_count
        ),
    }
    frozen = {
        "schema": (
            "p1553.m6_random_diagonal_known_target_rank.frozen.r159.v1"
        ),
        "source_bindings": source_binding_records(),
        "arity": ARITY,
        "query_constant": QUERY_CONSTANT,
        "factor_base_exponent_B": "3/4",
        "positive_c6_source_exponent_B": "9/2",
        "known_targets_per_column_exponent_B": "1/2",
        "all_column_known_target_exponent_B": "5/4",
        "relation_row_formula": "v-s_j*e_j",
        "relation_rhs_formula": "t_jr",
        "determinant_leading_monomial": (
            "(-1)^d*product_j(s_j)"
        ),
        "full_rank_failure_bound": "d/q",
        "uncovered_column_failure_bound_given_half_unique": "d^(-3)",
        "dominant_open_component": (
            "batched_positive_c6_unique_source_reverse_ffe_locator"
        ),
        "theorem": theorem,
        "cost": cost,
        "admission": admission,
    }
    logs = {
        "schema": (
            "p1553.m6_random_diagonal_known_target_rank.logs.r159.v1"
        ),
        "factor_log_control_count": log_count,
        "identical_descent_control_count": descent_count,
        "candidate_discrete_log_oracle_consumed": False,
        "controls": [
            {
                "control_id": row["control_id"],
                "subgroup_order": row["subgroup_order"],
                "recovered_factor_logs": row[
                    "recovered_factor_logs"
                ],
                "factor_logs_publicly_verified": row[
                    "factor_logs_publicly_verified"
                ],
                "identical_target_descent": row[
                    "identical_positive_c6_target_descent"
                ],
            }
            for row in controls_list
        ],
        "unconditional_algorithm_credit": False,
    }
    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "objective": (
            "Replace pairwise relation supply by direct known-target "
            "coverage and prove full rank using independent diagonal "
            "shifts."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "DIRECT_KNOWN_TARGET_ZERO_COVERAGE__RANDOM_DIAGONAL_"
            "FULL_RANK__IDEAL_CONDITIONED_SAMPLER_TRANSFER__IDENTICAL_"
            "POSITIVE_C6_DESCENT_REDUCTION__B5O4_TARGET_BATCH__BATCHED_"
            "REVERSE_FFE_SOURCE_LOCATOR_OPEN__NO_SHOUP_BREAKTHROUGH"
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "next_action": (
            "Construct a public batched reverse FFE locator that returns "
            "unique positive-C6 sources for B^(5/4)log(B) arbitrary targets "
            "using at most B^(9/4+o(1)) setup and B^(5/4+o(1)) work, then "
            "replay the admitted random-diagonal log solve and identical "
            "descent without explicit endpoint enumeration."
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "logs": logs,
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=pathlib.Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=pathlib.Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=pathlib.Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=pathlib.Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--logs-output", type=pathlib.Path, default=DEFAULT_LOGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs"])
    admission = bundle["report"]["admission"]
    controls = bundle["controls"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"coverage={controls['all_columns_covered_control_count']}/"
        f"{controls['control_count']} "
        f"rank={controls['full_rank_control_count']}/"
        f"{controls['control_count']} "
        f"logs={controls['publicly_verified_factor_log_control_count']} "
        f"descent={controls['successful_identical_descent_control_count']} "
        "lane=0 breakthrough=0"
    )


if __name__ == "__main__":
    main()
