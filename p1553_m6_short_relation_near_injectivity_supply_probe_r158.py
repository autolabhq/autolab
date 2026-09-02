#!/usr/bin/env python3
"""Prove near-injectivity and projective row supply for short M6 relations."""

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
SCHEMA = "p1553.m6_short_relation_near_injectivity_supply.r158.v1"

R157_PRODUCER = ROOT / (
    "p1553_m6_hash_to_curve_projective_rank_probe_r157.py"
)
R157_REPORT = ROOT / (
    "p1553_m6_hash_to_curve_projective_rank_probe_report_r157.json"
)
R157_FROZEN = ROOT / "frozen_m6_hash_to_curve_projective_rank.json"
R157_COST = ROOT / "m6_hash_to_curve_projective_rank_cost_ledger.json"
R157_REPLAY = ROOT / "m6_hash_to_curve_projective_rank_replay.json"
R157_CONTROLS = ROOT / "m6_hash_to_curve_projective_rank_controls.json"
R157_LOGS = ROOT / "factor_logs_and_identical_descent_r157.json"
R157_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_hash_to_curve_projective_rank_probe_r157.py"
)
R157_GATE = ROOT / (
    "p1553_m6_hash_to_curve_projective_rank_probe_gate_r157.md"
)
R157_PARENT = ROOT / (
    "p1553_m6_hash_to_curve_projective_rank_probe_parent_report_r157.yaml"
)

SOURCE_BINDINGS = (
    (
        "r157_producer",
        R157_PRODUCER,
        "0f022ee1c90e5a2feb61f213f71e3b1345015b89611bcf5d738dd2242426cba9",
    ),
    (
        "r157_report",
        R157_REPORT,
        "50cff20e88c953f76a373747a4f11f0dd21a36a0e526bf25c5bbfdaa8997b7d7",
    ),
    (
        "r157_frozen",
        R157_FROZEN,
        "efb0d428c915f93b7c03057486c91d4263d695a536af12204e04fd136f0ff708",
    ),
    (
        "r157_cost",
        R157_COST,
        "ac54b8756ce08ecdd3e659f1f60bb969ca7db07ba19f8b630e0b3719628466f9",
    ),
    (
        "r157_replay",
        R157_REPLAY,
        "cc1e0613c6c7f493400fd403e840a000fbde60770e6526e127e79d329dc8d3de",
    ),
    (
        "r157_controls",
        R157_CONTROLS,
        "4d78e859ed61f7e4ac7d65f381a027e32f89e8e89a0e673e27daac8702bb1deb",
    ),
    (
        "r157_logs",
        R157_LOGS,
        "e8ce99ff9582fe5b90fee2bebc12d4641a203b1df2828513c1db8232568c493f",
    ),
    (
        "r157_test",
        R157_TEST,
        "d39dcdff1ec2f809ea275a28066f6f5f2cb0420f578f3f8c13837e05f38836d2",
    ),
    (
        "r157_gate",
        R157_GATE,
        "ea7757fa1adfb06c8d80c5be9224419220a7f96570f069613e0631de609a38af",
    ),
    (
        "r157_parent",
        R157_PARENT,
        "200f0a17b190385aa6f7e5c3ef5610b77631f1ac6df8b048df4e06a7a0cb23f2",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_short_relation_near_injectivity_supply_"
    "probe_report_r158.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_short_relation_near_injectivity_supply.json"
)
DEFAULT_COST = ROOT / (
    "m6_short_relation_near_injectivity_supply_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_short_relation_near_injectivity_supply_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_short_relation_near_injectivity_supply_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r158.json"

ARITY = 6
A_PAIR_COUNTS = (2, 3)
C_PAIR_COUNTS = (3, 4, 5)
INDEPENDENT_A_BATCH_COUNTS = (1, 2, 4)
SEEDS = (15801, 15802)
BASE_EXPECTED_RELATIONS_PER_DIMENSION = 4


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R157 = load_module("p1553_r157_for_r158", R157_PRODUCER)
R156 = R157.R156
R154 = R157.R154
R81 = R157.R81


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
        raise AssertionError(f"R158 source binding mismatch: {failures}")
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


def positive_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for suffix in positive_compositions(total - first, parts - 1):
            yield (first, *suffix)


def signed_weight_vectors(
    dimension: int, weight: int
) -> tuple[tuple[int, ...], ...]:
    if weight == 0:
        return ((0,) * dimension,)
    rows: list[tuple[int, ...]] = []
    for support_size in range(1, min(dimension, weight) + 1):
        for support in itertools.combinations(
            range(dimension), support_size
        ):
            for magnitudes in positive_compositions(weight, support_size):
                for signs in itertools.product((-1, 1), repeat=support_size):
                    row = [0] * dimension
                    for index, magnitude, sign in zip(
                        support, magnitudes, signs
                    ):
                        row[index] = sign * magnitude
                    rows.append(tuple(row))
    return tuple(rows)


def feasible_six_vectors(
    dimension: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        row
        for weight in (0, 2, 4, 6)
        for row in signed_weight_vectors(dimension, weight)
    )


def singleton_source_vectors(
    dimension: int,
) -> tuple[tuple[int, ...], ...]:
    return signed_weight_vectors(dimension, 6)


def singleton_relation_rows(
    dimension: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        row
        for weight in (5, 7)
        for row in signed_weight_vectors(dimension, weight)
    )


def signed_weight_count(dimension: int, weight: int) -> int:
    if weight == 0:
        return 1
    return sum(
        math.comb(dimension, support)
        * math.comb(weight - 1, support - 1)
        * (2**support)
        for support in range(1, min(dimension, weight) + 1)
    )


def relation_row_count(dimension: int) -> int:
    return signed_weight_count(dimension, 5) + signed_weight_count(
        dimension, 7
    )


def incident_relation_row_count(dimension: int) -> int:
    return relation_row_count(dimension) - relation_row_count(
        dimension - 1
    )


def first_nonzero_positive(row: tuple[int, ...]) -> bool:
    return next(value for value in row if value) > 0


def canonical_nonzero_a_vectors(
    dimension: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        row
        for row in feasible_six_vectors(dimension)
        if any(row) and first_nonzero_positive(row)
    )


def deterministic_labels(
    role: str,
    width: int,
    modulus: int,
    seed: int,
) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(
                f"R158|{role}|{seed}|{index}".encode("utf-8")
            ).digest(),
            "big",
        )
        % modulus
        for index in range(width)
    )


def dot_mod(
    row: Iterable[int], labels: Iterable[int], modulus: int
) -> int:
    return sum(
        coefficient * label
        for coefficient, label in zip(row, labels)
    ) % modulus


def projective_normalize(
    row: tuple[int, ...], modulus: int
) -> tuple[int, ...]:
    pivot = next(value for value in row if value % modulus)
    inverse = pow(pivot, -1, modulus)
    return tuple(value * inverse % modulus for value in row)


def collision_record(
    vectors: tuple[tuple[int, ...], ...],
    labels: tuple[int, ...],
    modulus: int,
    distinguished: set[tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    buckets: dict[int, list[tuple[int, ...]]] = collections.defaultdict(
        list
    )
    for vector in vectors:
        buckets[dot_mod(vector, labels, modulus)].append(vector)
    collision_pairs = sum(
        math.comb(len(bucket), 2) for bucket in buckets.values()
    )
    collided_vectors = sum(
        len(bucket) for bucket in buckets.values() if len(bucket) > 1
    )
    bad_distinguished = (
        sum(
            vector in distinguished
            for bucket in buckets.values()
            if len(bucket) > 1
            for vector in bucket
        )
        if distinguished is not None
        else 0
    )
    return {
        "image_size": len(buckets),
        "collision_pair_count": collision_pairs,
        "collided_vector_count": collided_vectors,
        "bad_distinguished_vector_count": bad_distinguished,
        "bucket_multiplicity_histogram": dict(
            sorted(
                collections.Counter(
                    len(bucket) for bucket in buckets.values()
                ).items()
            )
        ),
        "image_sha256": sha256_json(
            [
                [endpoint, sorted(bucket)]
                for endpoint, bucket in sorted(buckets.items())
            ]
        ),
    }


def control_modulus(a_pair_count: int, c_pair_count: int) -> int:
    a_projective_count = len(
        canonical_nonzero_a_vectors(a_pair_count)
    )
    rows = relation_row_count(c_pair_count)
    denominator = (
        BASE_EXPECTED_RELATIONS_PER_DIMENSION * c_pair_count
    )
    return R154.next_prime(math.ceil(a_projective_count * rows / denominator))


def finite_control(
    a_pair_count: int,
    c_pair_count: int,
    batch_count: int,
    seed: int,
) -> dict[str, Any]:
    modulus = control_modulus(a_pair_count, c_pair_count)
    a_vectors = canonical_nonzero_a_vectors(a_pair_count)
    c_vectors = feasible_six_vectors(c_pair_count)
    c_singletons = set(singleton_source_vectors(c_pair_count))
    relation_rows = singleton_relation_rows(c_pair_count)
    c_labels = deterministic_labels(
        f"C|a{a_pair_count}|c{c_pair_count}",
        c_pair_count,
        modulus,
        seed,
    )
    a_label_batches = [
        deterministic_labels(
            (
                f"A{batch}|a{a_pair_count}|c{c_pair_count}|"
                f"k{batch_count}"
            ),
            a_pair_count,
            modulus,
            seed,
        )
        for batch in range(batch_count)
    ]

    c_collision = collision_record(
        c_vectors,
        c_labels,
        modulus,
        distinguished=c_singletons,
    )
    c_endpoint_multiplicities = collections.Counter(
        dot_mod(vector, c_labels, modulus) for vector in c_vectors
    )
    good_c_singletons = {
        vector
        for vector in c_singletons
        if c_endpoint_multiplicities[
            dot_mod(vector, c_labels, modulus)
        ]
        == 1
    }
    usable_relation_rows = {
        tuple(
            value - (sign if index == target_index else 0)
            for index, value in enumerate(vector)
        )
        for vector in good_c_singletons
        for target_index in range(c_pair_count)
        for sign in (-1, 1)
    }
    usable_relation_rows &= set(relation_rows)
    relation_rows = tuple(sorted(usable_relation_rows))
    a_collisions = [
        collision_record(
            tuple(
                row
                for row in feasible_six_vectors(a_pair_count)
                if any(row)
            ),
            labels,
            modulus,
        )
        for labels in a_label_batches
    ]

    a_endpoint_multiplicities: collections.Counter[int] = (
        collections.Counter()
    )
    for labels in a_label_batches:
        for vector in a_vectors:
            a_endpoint_multiplicities[
                dot_mod(vector, labels, modulus)
            ] += 1

    row_endpoints = [
        dot_mod(row, c_labels, modulus) for row in relation_rows
    ]
    zero_form_event_count = sum(
        a_endpoint_multiplicities[endpoint]
        for endpoint in row_endpoints
    )
    selected_rows = [
        row
        for row, endpoint in zip(relation_rows, row_endpoints)
        if a_endpoint_multiplicities[endpoint]
    ]
    projective_rows = {
        projective_normalize(
            tuple(value % modulus for value in row), modulus
        )
        for row in selected_rows
    }
    rank = R81.rank_mod(sorted(projective_rows), modulus)
    covered = {
        index
        for row in selected_rows
        for index, value in enumerate(row)
        if value
    }

    candidate_form_count = (
        batch_count * len(a_vectors) * len(relation_rows)
    )
    expected_events = Fraction(candidate_form_count, modulus)
    event_variance = expected_events * (
        1 - Fraction(1, modulus)
    )
    duplicate_event_count = (
        zero_form_event_count - len(selected_rows)
    )
    c_expected_collision_pairs = Fraction(
        math.comb(len(c_vectors), 2), modulus
    )
    c_bad_fraction_bound = Fraction(
        len(c_vectors) - 1, modulus
    )
    incident_row_counts = [
        sum(row[index] != 0 for row in relation_rows)
        for index in range(c_pair_count)
    ]
    expected_incident_events = [
        Fraction(batch_count * len(a_vectors) * count, modulus)
        for count in incident_row_counts
    ]
    expected_uncovered_fraction_chebyshev_bound = min(
        Fraction(1),
        sum(
            (
                min(Fraction(1), Fraction(1, 1) / expected)
                if expected
                else Fraction(1)
            )
            for expected in expected_incident_events
        )
        / c_pair_count,
    )

    return {
        "control_id": (
            f"a{a_pair_count}|c{c_pair_count}|"
            f"batches{batch_count}|seed{seed}"
        ),
        "a_pair_count": a_pair_count,
        "c_pair_count": c_pair_count,
        "independent_a_batch_count": batch_count,
        "seed": seed,
        "subgroup_order": modulus,
        "subgroup_order_exceeds_coefficient_minor_bound_84": modulus > 84,
        "feasible_a6_vector_count": (
            R154.signed_coefficient_vector_count(
                a_pair_count, ARITY
            )
        ),
        "canonical_nonzero_a6_vector_count": len(a_vectors),
        "feasible_c6_vector_count": len(c_vectors),
        "l1_six_c6_vector_count": len(c_singletons),
        "good_l1_six_c6_vector_count": len(good_c_singletons),
        "complete_relation_row_universe_count_l1_five_or_seven": (
            relation_row_count(c_pair_count)
        ),
        "usable_singleton_relation_row_count": len(relation_rows),
        "unusable_relation_row_count_due_c6_collisions": (
            relation_row_count(c_pair_count) - len(relation_rows)
        ),
        "relation_rows_match_exact_formula": (
            len(singleton_relation_rows(c_pair_count))
            == relation_row_count(c_pair_count)
        ),
        "complete_incident_relation_row_count_per_column": (
            incident_relation_row_count(c_pair_count)
        ),
        "usable_incident_relation_row_counts_by_column": (
            incident_row_counts
        ),
        "candidate_projective_form_count": candidate_form_count,
        "expected_relation_event_count": fraction_record(expected_events),
        "pairwise_independent_event_variance": fraction_record(
            event_variance
        ),
        "observed_relation_event_count": zero_form_event_count,
        "observed_distinct_relation_row_count": len(selected_rows),
        "observed_projectively_distinct_row_count": len(projective_rows),
        "duplicate_relation_event_count": duplicate_event_count,
        "signed_relation_rank": rank,
        "signed_full_rank": rank == c_pair_count,
        "covered_column_count": len(covered),
        "uncovered_column_count": c_pair_count - len(covered),
        "expected_incident_events_per_column": [
            fraction_record(value) for value in expected_incident_events
        ],
        "expected_uncovered_fraction_chebyshev_bound": fraction_record(
            expected_uncovered_fraction_chebyshev_bound
        ),
        "c6_actual_collision_pair_count": c_collision[
            "collision_pair_count"
        ],
        "c6_expected_collision_pair_count": fraction_record(
            c_expected_collision_pairs
        ),
        "c6_bad_l1_six_vector_count": c_collision[
            "bad_distinguished_vector_count"
        ],
        "c6_bad_l1_six_fraction": fraction_record(
            Fraction(
                c_collision["bad_distinguished_vector_count"],
                len(c_singletons),
            )
        ),
        "c6_expected_bad_l1_six_fraction_union_bound": fraction_record(
            c_bad_fraction_bound
        ),
        "c6_collision_record": c_collision,
        "a6_collision_pair_counts_by_batch": [
            record["collision_pair_count"] for record in a_collisions
        ],
        "a6_collision_records_sha256": sha256_json(a_collisions),
        "relation_rows_sha256": sha256_json(sorted(projective_rows)),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "global_injectivity_refuted": (
            "With d=B^(3/4), the feasible C6 coefficient universe has "
            "Theta(d^6)=B^(9/2) vectors in a group of order q=B^5. "
            "Its expected collision-pair count is Theta(d^12/q)=B^4, "
            "so global coefficient-map injectivity is not an asymptotically "
            "valid requirement."
        ),
        "near_injectivity": (
            "For distinct feasible vectors v,w and q>12, "
            "Pr[(v-w).C=0]=1/q. A fixed l1-six vector is bad with "
            "probability at most (|V|-1)/q=Theta(B^(-1/2)). Thus the "
            "expected bad singleton fraction is O(B^(-1/2)); Markov gives "
            "bad fraction at most B^(-1/4) except with probability "
            "O(B^(-1/4))."
        ),
        "exact_relation_row_universe": (
            "A singleton source vector v has l1 norm six. Subtracting a "
            "signed target unit changes l1 by one, so every relation row "
            "has l1 norm five or seven. Conversely every integer vector "
            "with l1 norm five or seven is v-sigma*e_j for some l1-six v. "
            "The row universe is therefore exact."
        ),
        "collision_pruning_preserves_row_scale": (
            "A bad l1-six source vector can invalidate at most 2d signed "
            "target rows. Since the complete row universe is Theta(d^7) "
            "and the l1-six source universe is Theta(d^6), an o(1) bad "
            "source fraction removes only an o(1) fraction of rows."
        ),
        "projective_form_nonproportionality": (
            "A nonzero A6 vector has l1 norm in {2,4,6}; a relation row "
            "has l1 norm in {5,7}. For q>84, modular proportionality "
            "implies rational proportionality. The possible norm ratios "
            "intersect only at one, and canonical A sign removes the "
            "negative copy. Distinct combined A/C forms are therefore "
            "nonproportional."
        ),
        "relation_supply_concentration": (
            "Distinct nonproportional linear forms on independent uniform "
            "A/C labels have pairwise-independent zero events. With "
            "O(log B) independent A batches, the projective candidate-form "
            "count is Theta(B^(23/4)log B), giving mean and variance "
            "Theta(B^(3/4)log B) relation events. Chebyshev gives relative "
            "failure O(B^(-3/4)/log B)."
        ),
        "duplicate_row_bound": (
            "The expected number of repeated event hits on one row, summed "
            "over the Theta(B^(21/4)) row universe, is "
            "O(B^(-15/4)log(B)^2). Thus event count and distinct "
            "projective row count agree with high probability."
        ),
        "coverage_boundary": (
            "A fixed column belongs to R_d-R_(d-1)=Theta(d^6) rows and "
            "therefore receives Theta(log B) expected events. Pairwise "
            "independence alone bounds its zero-event probability only by "
            "O(1/log B), proving vanishing uncovered fraction but not full "
            "coverage or full rank."
        ),
        "scope": (
            "The probability theorem is for independent uniform cyclic "
            "labels. Exact transfer through the conditioned hash-to-curve "
            "factor-base sampler, full coverage, residual-rank control, "
            "reverse FFE construction, and identical descent remain open."
        ),
        "novelty_status": (
            "short_relation_near_injectivity_and_pairwise_supply_"
            "novelty_unverified"
        ),
    }


def exponent_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_short_relation_near_injectivity_supply."
            "exponents.r158.v1"
        ),
        "group_order_exponent_B": fraction_record(Fraction(5)),
        "c_pair_count_exponent_B": fraction_record(Fraction(3, 4)),
        "a_pair_count_exponent_B": fraction_record(Fraction(1, 12)),
        "feasible_c6_vector_count_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "c6_collision_pair_count_exponent_B": fraction_record(
            Fraction(4)
        ),
        "expected_bad_singleton_fraction_exponent_B": fraction_record(
            Fraction(-1, 2)
        ),
        "high_probability_bad_fraction_threshold_exponent_B": (
            fraction_record(Fraction(-1, 4))
        ),
        "high_probability_bad_fraction_failure_exponent_B": (
            fraction_record(Fraction(-1, 4))
        ),
        "a6_vector_count_exponent_B": fraction_record(Fraction(1, 2)),
        "relation_row_universe_exponent_B": fraction_record(
            Fraction(21, 4)
        ),
        "projective_candidate_form_count_exponent_B": fraction_record(
            Fraction(23, 4)
        ),
        "projective_candidate_form_polylog_factor": "log(B)",
        "expected_relation_count_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "expected_relation_count_polylog_factor": "log(B)",
        "relative_supply_failure_exponent_B": fraction_record(
            Fraction(-3, 4)
        ),
        "duplicate_row_event_exponent_B": fraction_record(
            Fraction(-15, 4)
        ),
        "duplicate_row_event_polylog_factor": "log(B)^2",
        "expected_incident_events_per_column": "Theta(log(B))",
        "pairwise_independence_proves_full_coverage": False,
        "pairwise_independence_proves_full_rank": False,
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_short_relation_near_injectivity_supply."
            "cost.r158.v1"
        ),
        "explicit_c6_enumeration_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "explicit_control_exceeds_pollard_rho": True,
        "conditional_relation_batch_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "conditional_relation_batch_polylog_factor": "log(B)",
        "conditional_matrix_free_solve_exponent_B": fraction_record(
            Fraction(2)
        ),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "finite_explicit_controls_receive_attack_credit": False,
        "iid_label_theorem_receives_hash_to_curve_transfer_credit": False,
        "full_coverage_or_rank_theorem_supplied": False,
        "reverse_only_signed_marker_operator_supplied": False,
        "identical_target_descent_supplied": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    controls_list = [
        finite_control(a_pairs, c_pairs, batches, seed)
        for a_pairs in A_PAIR_COUNTS
        for c_pairs in C_PAIR_COUNTS
        for batches in INDEPENDENT_A_BATCH_COUNTS
        for seed in SEEDS
    ]
    all_row_formulas = all(
        row["relation_rows_match_exact_formula"] for row in controls_list
    )
    all_moduli_safe = all(
        row["subgroup_order_exceeds_coefficient_minor_bound_84"]
        for row in controls_list
    )
    full_count = sum(row["signed_full_rank"] for row in controls_list)
    full_by_batch = [
        sum(
            row["signed_full_rank"]
            for row in controls_list
            if row["independent_a_batch_count"] == batches
        )
        for batches in INDEPENDENT_A_BATCH_COUNTS
    ]
    event_counts_by_batch = [
        {
            "independent_a_batch_count": batches,
            "control_count": sum(
                row["independent_a_batch_count"] == batches
                for row in controls_list
            ),
            "total_observed_relation_event_count": sum(
                row["observed_relation_event_count"]
                for row in controls_list
                if row["independent_a_batch_count"] == batches
            ),
            "total_distinct_relation_row_count": sum(
                row["observed_distinct_relation_row_count"]
                for row in controls_list
                if row["independent_a_batch_count"] == batches
            ),
            "full_rank_control_count": full_by_batch[index],
        }
        for index, batches in enumerate(INDEPENDENT_A_BATCH_COUNTS)
    ]
    controls = {
        "schema": (
            "p1553.m6_short_relation_near_injectivity_supply."
            "controls.r158.v1"
        ),
        "control_count": len(controls_list),
        "a_pair_counts": list(A_PAIR_COUNTS),
        "c_pair_counts": list(C_PAIR_COUNTS),
        "independent_a_batch_counts": list(INDEPENDENT_A_BATCH_COUNTS),
        "seeds": list(SEEDS),
        "base_expected_relations_per_dimension": (
            BASE_EXPECTED_RELATIONS_PER_DIMENSION
        ),
        "all_relation_row_formulas_exact": all_row_formulas,
        "all_moduli_exceed_coefficient_minor_bound": all_moduli_safe,
        "full_rank_control_count": full_count,
        "full_rank_counts_by_independent_a_batch_count": full_by_batch,
        "event_counts_by_independent_a_batch_count": event_counts_by_batch,
        "finite_controls_receive_asymptotic_credit": False,
        "controls": controls_list,
    }
    theorem = theorem_record()
    exponents = exponent_ledger()
    cost = cost_ledger()
    frozen = {
        "schema": (
            "p1553.m6_short_relation_near_injectivity_supply."
            "frozen.r158.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "design": {
            "arity": ARITY,
            "a_pair_counts": list(A_PAIR_COUNTS),
            "c_pair_counts": list(C_PAIR_COUNTS),
            "independent_a_batch_counts": list(
                INDEPENDENT_A_BATCH_COUNTS
            ),
            "seeds": list(SEEDS),
            "base_expected_relations_per_dimension": (
                BASE_EXPECTED_RELATIONS_PER_DIMENSION
            ),
        },
        "theorem": theorem,
        "exponents": exponents,
        "cost": cost,
        "open_obligations": {
            "exact_conditioned_hash_to_curve_transfer": "open",
            "full_column_coverage_theorem": "open",
            "full_projective_rank_theorem": "open",
            "reverse_only_signed_marker_operator": "open",
            "signed_weight_separable_ffe_dag": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "pollard_rho_improvement": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
    }
    replay = {
        "schema": (
            "p1553.m6_short_relation_near_injectivity_supply."
            "replay.r158.v1"
        ),
        "controls": controls_list,
        "all_relation_row_formulas_exact": all_row_formulas,
        "all_moduli_exceed_coefficient_minor_bound": all_moduli_safe,
        "breakthrough": False,
    }
    logs = {
        "schema": (
            "p1553.m6_short_relation_near_injectivity_supply."
            "logs.r158.v1"
        ),
        "r157_finite_factor_log_controls_inherited": 23,
        "r158_finite_full_rank_control_count": full_count,
        "new_factor_logs_receive_attack_credit": False,
        "candidate_discrete_log_oracle_consumed": False,
        "exact_conditioned_hash_to_curve_transfer_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_transfer_complete": False,
    }

    obligations = {
        "r157_public_group_factor_log_transfer_inherited": True,
        "ten_source_bindings_verified": len(actual_bindings) == 10,
        "global_coefficient_map_injectivity_refuted_as_target": True,
        "near_injectivity_bad_fraction_bound_proved": True,
        "exact_l1_five_or_seven_row_universe_proved": True,
        "collision_pruning_preserves_relation_row_scale_proved": True,
        "projective_form_nonproportionality_proved": True,
        "pairwise_relation_event_independence_proved": True,
        "relation_supply_second_moment_concentration_proved": True,
        "duplicate_row_event_bound_proved": True,
        "vanishing_uncovered_fraction_boundary_derived": True,
        "thirty_six_finite_controls_complete": len(controls_list) == 36,
        "all_finite_row_universes_exact": all_row_formulas,
        "all_finite_moduli_above_minor_bound": all_moduli_safe,
        "finite_rank_transition_recorded_without_credit": full_count > 0,
        "explicit_c6_enumeration_B9O2_charged": True,
        "exact_conditioned_hash_to_curve_transfer_complete": False,
        "full_column_coverage_theorem_complete": False,
        "full_projective_rank_theorem_complete": False,
        "reverse_only_signed_marker_operator_complete": False,
        "signed_weight_separable_ffe_dag_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(bool(value) for value in obligations.values())
    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "objective": (
            "Replace the false global C6 injectivity target with a "
            "near-injectivity theorem and prove concentrated projective "
            "short-relation row supply."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "exponents": exponents,
        "controls": controls,
        "cost": cost,
        "classification": (
            "GLOBAL_C6_INJECTIVITY_ASYMPTOTICALLY_REFUTED__BAD_SINGLETON_"
            "FRACTION_O1__EXACT_L1_5_OR_7_RELATION_ROW_UNIVERSE__PROJECTIVE_"
            "COMBINED_FORMS_PAIRWISE_INDEPENDENT__B3O4_LOG_B_DISTINCT_ROW_"
            "SUPPLY_CONCENTRATES__FULL_COVERAGE_RANK_HASH_TRANSFER_REVERSE_"
            "FFE_AND_DESCENT_OPEN__NO_SHOUP_BREAKTHROUGH"
        ),
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "near_injectivity_and_relation_supply_theorem_admitted": True,
            "exact_conditioned_hash_to_curve_transfer_admitted": False,
            "full_coverage_or_rank_admitted": False,
            "reverse_only_signed_marker_operator_admitted": False,
            "lane_admitted": False,
        },
        "next_action": (
            "Upgrade pairwise relation-event concentration to full column "
            "coverage and full projective rank for the conditioned "
            "hash-to-curve sampler, then construct the reverse signed FFE "
            "operator and identical target descent within the frozen caps."
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        f"full={controls['full_rank_control_count']}/"
        f"{controls['control_count']} "
        f"by_batch={controls['full_rank_counts_by_independent_a_batch_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
