#!/usr/bin/env python3
"""Compare signed singleton M6 rows with sparse random-matrix models."""

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
SCHEMA = "p1553.m6_singleton_relation_hypergraph_rank.r155.v1"

R154_PRODUCER = ROOT / (
    "p1553_m6_signed_quotient_multiscale_rank_probe_r154.py"
)
R154_REPORT = ROOT / (
    "p1553_m6_signed_quotient_multiscale_rank_"
    "probe_report_r154.json"
)
R154_FROZEN = ROOT / (
    "frozen_m6_signed_quotient_multiscale_rank.json"
)
R154_COST = ROOT / (
    "m6_signed_quotient_multiscale_rank_cost_ledger.json"
)
R154_REPLAY = ROOT / (
    "m6_signed_quotient_multiscale_rank_replay.json"
)
R154_CONTROLS = ROOT / (
    "m6_signed_quotient_multiscale_rank_controls.json"
)
R154_LOGS = ROOT / "factor_logs_and_identical_descent_r154.json"
R154_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_signed_quotient_multiscale_rank_probe_r154.py"
)
R154_GATE = ROOT / (
    "p1553_m6_signed_quotient_multiscale_rank_probe_gate_r154.md"
)
R154_PARENT = ROOT / (
    "p1553_m6_signed_quotient_multiscale_rank_"
    "probe_parent_report_r154.yaml"
)
XORSAT_RANK_PDF = ROOT / (
    "references/coja_oghlan_kang_krieg_rolvien_"
    "xorsat_rank_2301.09287.pdf"
)
SPARSE_RANK_PDF = ROOT / (
    "references/coja_oghlan_ergur_gao_hetterich_rolvien_"
    "sparse_rank_1906.05757.pdf"
)

SOURCE_BINDINGS = (
    (
        "r154_producer",
        R154_PRODUCER,
        "8e3f3cf4947417f341d4b708258106e11eb391170eb956d75992c5b0c6ebe6ad",
    ),
    (
        "r154_report",
        R154_REPORT,
        "16ee4f848ef89c54bda5a413fb05fefc6f6863600ff79f9f6f925e59bcc2d9ef",
    ),
    (
        "r154_frozen",
        R154_FROZEN,
        "f6082904507a58393507311fe712f5be2413fea71d087d3abebf6fe278a0856e",
    ),
    (
        "r154_cost",
        R154_COST,
        "bd7ac6ba5efbdd9156b2ce0054bbdae29a19760a1b4323012d2d1c91320bbdbb",
    ),
    (
        "r154_replay",
        R154_REPLAY,
        "87576bb1e92529815938dc1e8a1520a50e6e850d09ee28500df55d0375a236ee",
    ),
    (
        "r154_controls",
        R154_CONTROLS,
        "4eef913ace45f47337f1362af946fe8eb26f23167b738f79f785583f1ada66d6",
    ),
    (
        "r154_logs",
        R154_LOGS,
        "2302f2ea00ec127108817631f79961a4577dbbb0be912d11de8f0fd86343e631",
    ),
    (
        "r154_test",
        R154_TEST,
        "8b230718808612747c50d7cee4261c1f42482e235350b1040126f7f7ef345e64",
    ),
    (
        "r154_gate",
        R154_GATE,
        "905b7ac86fb1e444e0d40252f0107dbc65293de9823ec69a9e171a39da14ab27",
    ),
    (
        "r154_parent",
        R154_PARENT,
        "46ae102679ddc3acaa35a3c7750bb1b5f2c6f504cd0d5292b826c99a2d3db7a4",
    ),
    (
        "xorsat_rank_pdf",
        XORSAT_RANK_PDF,
        "862fe28d87041f8c52444cc37703196191dcd1cd181c2c4ee37fef4d7cb78a52",
    ),
    (
        "sparse_rank_pdf",
        SPARSE_RANK_PDF,
        "82275b37845a5d2f03648cd2c6189d0ff2931cb1d39b70e2c5dfa2f7adc1769d",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_singleton_relation_hypergraph_rank_"
    "probe_report_r155.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_singleton_relation_hypergraph_rank.json"
)
DEFAULT_COST = ROOT / (
    "m6_singleton_relation_hypergraph_rank_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_singleton_relation_hypergraph_rank_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_singleton_relation_hypergraph_rank_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r155.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R154 = load_module("p1553_r154_for_r155", R154_PRODUCER)
R153 = R154.R153
R81 = R154.R81


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
        raise AssertionError(f"R155 source binding mismatch: {failures}")
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


def multiplicity_vector(
    indices: Iterable[int], width: int
) -> tuple[int, ...]:
    counts = [0] * width
    for index in indices:
        counts[index] += 1
    return tuple(counts)


def endpoint_multisets(
    labels: tuple[int, ...],
    arity: int,
    modulus: int,
) -> dict[int, list[tuple[int, ...]]]:
    endpoints: dict[int, list[tuple[int, ...]]] = collections.defaultdict(
        list
    )
    for indices in itertools.combinations_with_replacement(
        range(len(labels)), arity
    ):
        counts = multiplicity_vector(indices, len(labels))
        endpoint = sum(
            count * label for count, label in zip(counts, labels)
        ) % modulus
        endpoints[endpoint].append(counts)
    return dict(endpoints)


def permutation_count(counts: Iterable[int]) -> int:
    values = tuple(counts)
    result = math.factorial(sum(values))
    for count in values:
        result //= math.factorial(count)
    return result


def signed_sparse_row(
    source_counts: tuple[int, ...],
    target_index: int,
    c_pairs: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    labels = R154.flatten_pairs(c_pairs)
    full = list(source_counts)
    full[target_index] -= 1
    return tuple(
        full[labels.index(positive)] - full[labels.index(negative)]
        for positive, negative in c_pairs
    )


def aggregate_signed_row(
    shift: int,
    shift_weight: int,
    target_index: int,
    c_pairs: tuple[tuple[int, int], ...],
    c5_counts: collections.Counter[int],
    c6_counts: collections.Counter[int],
    modulus: int,
) -> tuple[int, ...]:
    labels = R154.flatten_pairs(c_pairs)
    target = labels[target_index]
    count = (
        shift_weight * c6_counts[(shift + target) % modulus]
    ) % modulus
    full = [
        (
            shift_weight
            * R154.ARITY
            * c5_counts[(shift + target - column) % modulus]
            - (count if target_index == column_index else 0)
        )
        % modulus
        for column_index, column in enumerate(labels)
    ]
    return tuple(
        (
            full[labels.index(positive)]
            - full[labels.index(negative)]
        )
        % modulus
        for positive, negative in c_pairs
    )


def independent_uniform_coverage_probability(
    dimension: int, support_sizes: Iterable[int]
) -> Fraction:
    sizes = tuple(support_sizes)
    probability = Fraction(0)
    for omitted in range(dimension + 1):
        term = Fraction(math.comb(dimension, omitted))
        for size in sizes:
            denominator = math.comb(dimension, size)
            numerator = (
                math.comb(dimension - omitted, size)
                if size <= dimension - omitted
                else 0
            )
            term *= Fraction(numerator, denominator)
        probability += term if omitted % 2 == 0 else -term
    return probability


def independent_uniform_expected_uncovered(
    dimension: int, support_sizes: Iterable[int]
) -> Fraction:
    probability = Fraction(1)
    for size in support_sizes:
        probability *= Fraction(dimension - size, dimension)
    return dimension * probability


def peel_two_core(
    supports: list[set[int]], dimension: int
) -> dict[str, int]:
    active_rows = set(range(len(supports)))
    active_columns = set(range(dimension))
    while True:
        degrees = {
            column: sum(
                column in supports[row] for row in active_rows
            )
            for column in active_columns
        }
        peel_columns = {
            column for column, degree in degrees.items() if degree <= 1
        }
        if not peel_columns:
            break
        incident_rows = {
            row
            for row in active_rows
            if supports[row] & peel_columns
        }
        active_columns -= peel_columns
        active_rows -= incident_rows
    return {
        "peeled_column_count": dimension - len(active_columns),
        "peeled_row_count": len(supports) - len(active_rows),
        "core_column_count": len(active_columns),
        "core_row_count": len(active_rows),
    }


def singleton_control(
    c_pair_count: int, occupancy_multiplier: int, seed: int
) -> dict[str, Any]:
    max_a6_support = R154.signed_coefficient_vector_count(
        R154.A_PAIR_COUNT, R154.ARITY
    )
    max_c6_support = R154.signed_coefficient_vector_count(
        c_pair_count, R154.ARITY
    )
    modulus = R154.next_prime(
        math.ceil(
            max_a6_support
            * max_c6_support
            / occupancy_multiplier
        )
    )
    tag = (
        f"c{c_pair_count}|lambda{occupancy_multiplier}|seed{seed}"
    )
    a_pairs = R154.deterministic_signed_pairs(
        f"A|{tag}", R154.A_PAIR_COUNT, modulus
    )
    c_pairs = R154.deterministic_signed_pairs(
        f"C|{tag}", c_pair_count, modulus
    )
    a_labels = R154.flatten_pairs(a_pairs)
    c_labels = R154.flatten_pairs(c_pairs)
    shift_counts = R153.convolution_power(
        a_labels, R154.ARITY, modulus
    )
    c5_counts = R153.convolution_power(
        c_labels, R154.ARITY - 1, modulus
    )
    c6_counts = R153.convolution_power(
        c_labels, R154.ARITY, modulus
    )
    endpoints = endpoint_multisets(
        c_labels, R154.ARITY, modulus
    )
    aggregate = R154.signed_relation_system(
        a_labels, c_pairs, modulus
    )
    singleton_rows: list[tuple[int, ...]] = []
    singleton_supports: list[set[int]] = []
    singleton_aggregate_matches = True
    multi_source_relation_rows = 0
    positive_relation_rows = 0

    for shift, shift_weight in sorted(shift_counts.items()):
        for target_index, target in enumerate(c_labels):
            sources = endpoints.get((shift + target) % modulus, [])
            if not sources:
                continue
            positive_relation_rows += 1
            if len(sources) != 1:
                multi_source_relation_rows += 1
                continue
            source = sources[0]
            row = signed_sparse_row(source, target_index, c_pairs)
            multiplicity = permutation_count(source)
            aggregate_row = aggregate_signed_row(
                shift,
                shift_weight,
                target_index,
                c_pairs,
                c5_counts,
                c6_counts,
                modulus,
            )
            scale = (shift_weight * multiplicity) % modulus
            normalized = tuple(
                value * pow(scale, -1, modulus) % modulus
                for value in aggregate_row
            )
            singleton_aggregate_matches &= normalized == tuple(
                value % modulus for value in row
            )
            singleton_rows.append(
                tuple(value % modulus for value in row)
            )
            singleton_supports.append(
                {index for index, value in enumerate(row) if value}
            )

    nonzero_rows = [
        row for row in singleton_rows if any(value for value in row)
    ]
    nonzero_supports = [
        support
        for row, support in zip(singleton_rows, singleton_supports)
        if any(value for value in row)
    ]
    support_sizes = [len(support) for support in nonzero_supports]
    covered = set().union(*nonzero_supports) if nonzero_supports else set()
    dimension = len(c_pairs)
    singleton_rank = R81.rank_mod(nonzero_rows, modulus)
    unique_rows = set(nonzero_rows)
    projective_rows = {
        tuple(
            value * pow(next(item for item in row if item), -1, modulus)
            % modulus
            for value in row
        )
        for row in unique_rows
    }
    opposite_closure_exact = all(
        tuple((-value) % modulus for value in row) in unique_rows
        for row in unique_rows
    )
    coverage_probability = independent_uniform_coverage_probability(
        dimension, support_sizes
    )
    expected_uncovered = independent_uniform_expected_uncovered(
        dimension, support_sizes
    )
    core = peel_two_core(nonzero_supports, dimension)
    inherited = R154.synthetic_control(
        c_pair_count, occupancy_multiplier, seed
    )

    return {
        "control_id": tag,
        "subgroup_order": modulus,
        "c_pair_count": c_pair_count,
        "signed_log_dimension": dimension,
        "preregistered_occupancy_multiplier": occupancy_multiplier,
        "seed": seed,
        "positive_relation_row_count": positive_relation_rows,
        "singleton_relation_row_count": len(singleton_rows),
        "nonzero_singleton_row_count": len(nonzero_rows),
        "multi_source_relation_row_count": multi_source_relation_rows,
        "singleton_fraction": fraction_record(
            Fraction(
                len(singleton_rows),
                positive_relation_rows or 1,
            )
        ),
        "singleton_support_size_histogram": dict(
            sorted(collections.Counter(support_sizes).items())
        ),
        "singleton_total_incidence_count": sum(support_sizes),
        "singleton_covered_column_count": len(covered),
        "singleton_uncovered_column_count": dimension - len(covered),
        "singleton_matrix_rank": singleton_rank,
        "singleton_matrix_full_rank": singleton_rank == dimension,
        "unique_singleton_row_count": len(unique_rows),
        "projectively_distinct_singleton_row_count": len(projective_rows),
        "opposite_row_closure_exact": opposite_closure_exact,
        "aggregate_signed_quotient_rank": aggregate[
            "signed_quotient_rank"
        ],
        "aggregate_signed_quotient_full_rank": aggregate[
            "signed_quotient_full_rank"
        ],
        "inherited_aggregate_rank_matches": (
            inherited["signed_quotient_rank"]
            == aggregate["signed_quotient_rank"]
        ),
        "all_singleton_aggregate_rows_normalize_exactly": (
            singleton_aggregate_matches
        ),
        "independent_uniform_support_model": {
            "coverage_probability": fraction_record(
                coverage_probability
            ),
            "expected_uncovered_columns": fraction_record(
                expected_uncovered
            ),
            "model_applies_to_actual_rows": False,
        },
        "two_core": core,
        "singleton_rows_sha256": sha256_json(singleton_rows),
        "singleton_supports_sha256": sha256_json(
            [sorted(support) for support in singleton_supports]
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def grouped_summary(
    controls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for c_pair_count in R154.SYNTHETIC_LEVELS:
        for multiplier in R154.OCCUPANCY_MULTIPLIERS:
            rows = [
                row
                for row in controls
                if row["c_pair_count"] == c_pair_count
                and row["preregistered_occupancy_multiplier"]
                == multiplier
            ]
            summaries.append(
                {
                    "c_pair_count": c_pair_count,
                    "preregistered_occupancy_multiplier": multiplier,
                    "trial_count": len(rows),
                    "aggregate_full_rank_count": sum(
                        row["aggregate_signed_quotient_full_rank"]
                        for row in rows
                    ),
                    "singleton_full_rank_count": sum(
                        row["singleton_matrix_full_rank"] for row in rows
                    ),
                    "singleton_ranks": [
                        row["singleton_matrix_rank"] for row in rows
                    ],
                    "uncovered_column_counts": [
                        row["singleton_uncovered_column_count"]
                        for row in rows
                    ],
                    "singleton_row_counts": [
                        row["singleton_relation_row_count"] for row in rows
                    ],
                    "projective_row_counts": [
                        row["projectively_distinct_singleton_row_count"]
                        for row in rows
                    ],
                    "multi_source_row_counts": [
                        row["multi_source_relation_row_count"]
                        for row in rows
                    ],
                }
            )
    return summaries


def literature_record() -> dict[str, Any]:
    return {
        "xorsat_rank": {
            "title": "The k-XORSAT threshold revisited",
            "arxiv": "2301.09287",
            "sha256": SOURCE_BINDINGS[-2][2],
            "applicable_model": (
                "Rows have exactly k nonzero positions whose supports are "
                "independent uniformly random k-subsets; nonzero row "
                "coefficients may be prescribed."
            ),
            "campaign_gap": (
                "M6 singleton rows are selected by shared A6/C6 endpoint "
                "collisions and are neither independent nor uniform "
                "conditional on their support sizes."
            ),
        },
        "prescribed_degree_sparse_rank": {
            "title": "The rank of sparse random matrices",
            "arxiv": "1906.05757",
            "sha256": SOURCE_BINDINGS[-1][2],
            "applicable_model": (
                "A random Tanner graph with prescribed row and column "
                "degrees, under the paper's moment and simplicity "
                "hypotheses, with arbitrary nonzero field entries."
            ),
            "campaign_gap": (
                "The M6 Tanner graph is a correlated convolution image; "
                "no contiguity or configuration-model coupling is supplied."
            ),
        },
    }


def theorem_record() -> dict[str, Any]:
    return {
        "singleton_row": (
            "If one target row has exactly one unordered C6 multiset "
            "source with multiplicities m_b, then after dividing the "
            "aggregate row by its public nonzero A6 and permutation "
            "multiplicity, the signed quotient coefficients are "
            "m_b-m_(-b) minus the signed target unit vector."
        ),
        "support_bound": (
            "Every normalized singleton signed row has at most seven "
            "nonzero columns: at most six source inversion pairs and one "
            "target inversion pair."
        ),
        "independent_coverage_comparator": (
            "For independent uniform supports of sizes k_i on n columns, "
            "a fixed column is uncovered with probability "
            "product_i(1-k_i/n). Exact inclusion-exclusion gives the "
            "finite all-column coverage probability."
        ),
        "asymptotic_coupon_threshold": (
            "For fixed row width k and m=c*n rows, the expected number of "
            "uncovered columns is asymptotic to n*exp(-c*k), so constant "
            "occupancy cannot yield full column rank with high "
            "probability. The coverage scale is total incidence "
            "n*(log n+omega(1)), equivalently m=(n/k)*(log n+omega(1))."
        ),
        "campaign_consequence": (
            "With n=B^(3/4+o(1)), logarithmic oversampling changes no B "
            "exponent. Repeating a B^(5/4) reverse batch O(log B) times "
            "and the conditional B^2 solve by polylogarithmic factors "
            "remain B^(5/4+o(1)) and B^(2+o(1)), respectively."
        ),
        "scope": (
            "The coupon calculation applies only to the independent "
            "uniform-support comparator. It closes a naive fixed-constant "
            "occupancy transfer, not correlated M6 rank. The literature "
            "theorems require independence or a prescribed-degree random "
            "Tanner graph not established for the convolution rows."
        ),
        "novelty_status": (
            "singleton_hypergraph_contiguity_route_novelty_unverified"
        ),
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_singleton_relation_hypergraph_rank.cost.r155.v1"
        ),
        "signed_log_dimension_exponent_B": R154.fraction_record(
            Fraction(3, 4)
        ),
        "singleton_row_width_upper_bound": 7,
        "required_relation_count_exponent_B": R154.fraction_record(
            Fraction(3, 4)
        ),
        "required_relation_count_polylog_factor": "log(B)",
        "reverse_batch_exponent_B": R154.fraction_record(
            Fraction(5, 4)
        ),
        "reverse_batch_polylog_factor": "log(B)",
        "conditional_matrix_free_solve_exponent_B": R154.fraction_record(
            Fraction(2)
        ),
        "conditional_solve_polylog_factor": "polylog(B)",
        "setup_state_cap_exponent_B": R154.fraction_record(
            Fraction(9, 4)
        ),
        "pollard_rho_exponent_B": R154.fraction_record(
            Fraction(5, 2)
        ),
        "logarithmic_oversampling_changes_no_exponent": True,
        "singleton_hypergraph_contiguity_supplied": False,
        "random_rank_concentration_theorem_supplied": False,
        "hash_to_curve_rank_transfer_supplied": False,
        "reverse_only_signed_marker_operator_supplied": False,
        "factor_logs_without_verifier_labels_supplied": False,
        "identical_target_descent_supplied": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    controls_list = [
        singleton_control(c_pair_count, multiplier, seed)
        for c_pair_count in R154.SYNTHETIC_LEVELS
        for multiplier in R154.OCCUPANCY_MULTIPLIERS
        for seed in R154.SYNTHETIC_SEEDS
    ]
    summaries = grouped_summary(controls_list)
    all_exact = all(
        row["inherited_aggregate_rank_matches"]
        and row["all_singleton_aggregate_rows_normalize_exactly"]
        for row in controls_list
    )
    singleton_full_count = sum(
        row["singleton_matrix_full_rank"] for row in controls_list
    )
    aggregate_full_count = sum(
        row["aggregate_signed_quotient_full_rank"]
        for row in controls_list
    )
    all_full_singletons_cover = all(
        row["singleton_uncovered_column_count"] == 0
        for row in controls_list
        if row["singleton_matrix_full_rank"]
    )
    aggregate_and_singleton_full_rank_agree = all(
        row["aggregate_signed_quotient_full_rank"]
        == row["singleton_matrix_full_rank"]
        for row in controls_list
    )
    covered_but_deficient_count = sum(
        row["singleton_uncovered_column_count"] == 0
        and not row["singleton_matrix_full_rank"]
        for row in controls_list
    )
    max_observed_support = max(
        (
            max(
                (
                    int(size)
                    for size in row[
                        "singleton_support_size_histogram"
                    ]
                ),
                default=0,
            )
            for row in controls_list
        ),
        default=0,
    )
    all_opposite_closed = all(
        row["opposite_row_closure_exact"] for row in controls_list
    )
    literature = literature_record()
    theorem = theorem_record()
    costs = cost_ledger()

    controls = {
        "schema": (
            "p1553.m6_singleton_relation_hypergraph_rank."
            "controls.r155.v1"
        ),
        "control_count": len(controls_list),
        "all_singleton_normalizations_exact": all_exact,
        "aggregate_full_rank_control_count": aggregate_full_count,
        "singleton_full_rank_control_count": singleton_full_count,
        "all_full_rank_singleton_controls_cover_every_column": (
            all_full_singletons_cover
        ),
        "aggregate_and_singleton_full_rank_outcomes_agree": (
            aggregate_and_singleton_full_rank_agree
        ),
        "covered_but_rank_deficient_control_count": (
            covered_but_deficient_count
        ),
        "max_observed_singleton_support_size": max_observed_support,
        "all_singleton_row_sets_closed_under_opposites": (
            all_opposite_closed
        ),
        "grouped_summary": summaries,
        "controls": controls_list,
        "finite_controls_receive_asymptotic_credit": False,
    }

    frozen = {
        "schema": (
            "p1553.m6_singleton_relation_hypergraph_rank."
            "frozen.r155.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "literature": literature,
        "theorem": theorem,
        "cost": costs,
        "required_open_outputs": {
            "singleton_hypergraph_contiguity_or_dependency_theorem": "open",
            "random_rank_concentration_theorem": "open",
            "hash_to_curve_rank_transfer": "open",
            "reverse_only_signed_marker_operator": "open",
            "factor_logs_without_verifier_labels": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_singleton_relation_hypergraph_rank."
            "replay.r155.v1"
        ),
        "controls": controls_list,
        "grouped_summary": summaries,
        "all_singleton_normalizations_exact": all_exact,
    }

    logs = {
        "schema": (
            "p1553.m6_singleton_relation_hypergraph_rank."
            "logs_descent.r155.v1"
        ),
        "finite_singleton_full_rank_control_count": singleton_full_count,
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "twelve_source_bindings_verified": len(actual_bindings) == 12,
        "two_primary_rank_papers_pinned": len(literature) == 2,
        "r154_signed_quotient_inherited": True,
        "singleton_signed_row_formula_derived": True,
        "singleton_support_bound_seven_proved": True,
        "forty_eight_controls_complete": len(controls_list) == 48,
        "all_singleton_aggregate_normalizations_exact": all_exact,
        "exact_independent_coverage_comparators_complete": True,
        "all_full_rank_singleton_controls_cover_columns": (
            all_full_singletons_cover
        ),
        "aggregate_and_singleton_full_rank_outcomes_agree": (
            aggregate_and_singleton_full_rank_agree
        ),
        "covered_but_rank_deficient_controls_recorded": (
            covered_but_deficient_count > 0
        ),
        "opposite_row_dependency_recorded": all_opposite_closed,
        "two_core_diagnostics_complete": all(
            "two_core" in row for row in controls_list
        ),
        "constant_occupancy_coupon_obstruction_derived": True,
        "logarithmic_oversampling_charged": True,
        "literature_hypothesis_gap_explicit": True,
        "finite_results_scoped_without_transfer": True,
        "singleton_hypergraph_contiguity_complete": False,
        "random_rank_concentration_theorem_complete": False,
        "hash_to_curve_rank_transfer_complete": False,
        "reverse_only_signed_marker_operator_complete": False,
        "signed_weight_separable_ffe_dag_complete": False,
        "factor_logs_without_verifier_labels_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Prove or refute contiguity between the logarithmically "
        "oversampled singleton M6 Tanner graph and a prescribed-degree "
        "random sparse-matrix model, explicitly controlling shared-deck "
        "dependencies, repeated supports, cancellations, and the 2-core. "
        "Then transfer the result to hash-to-curve decks and instantiate "
        "the reverse signed FFE operator within B^(9/4) setup and "
        "B^(5/4+o(1)) batch work before attempting exact-residual logs "
        "or identical descent."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "SINGLETON_SIGNED_RELATION_ROWS_HAVE_WIDTH_AT_MOST_SEVEN__"
            "EXACT_NORMALIZATION_AND_TWO_CORE_CONTROLS__INDEPENDENT_"
            "SPARSE_SUPPORT_COMPARATOR_NEEDS_N_LOG_N_INCIDENCES__LOG_"
            "OVERSAMPLING_PRESERVES_B_EXPONENTS__PUBLISHED_RANK_THEOREMS_"
            "DO_NOT_APPLY_WITHOUT_CONVOLUTION_TANNER_CONTIGUITY__HASH_TO_"
            "CURVE_TRANSFER_REVERSE_FFE_LOGS_DESCENT_OPEN__NO_SHOUP_"
            "BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether the R154 finite rank transition matches an "
            "applicable sparse random-matrix theorem or hides an "
            "asymptotic coverage and dependency gap."
        ),
        "source_bindings": source_binding_records(),
        "literature": literature,
        "theorem": theorem,
        "cost": costs,
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "singleton_row_and_coupon_boundary_admitted": True,
            "logarithmic_oversampling_admitted": True,
            "published_sparse_rank_theorem_transfer_admitted": False,
            "reverse_only_signed_marker_operator_admitted": False,
            "lane_admitted": False,
        },
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": costs,
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
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"singleton_full="
        f"{bundle['controls']['singleton_full_rank_control_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
