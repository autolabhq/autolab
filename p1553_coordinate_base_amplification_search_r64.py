#!/usr/bin/env python3
"""Preregistered search for public coordinate bases with smooth-closure gain."""

import argparse
import hashlib
import json
import math
import pathlib

import p1553_public_factor_base_closure_probe_r63 as r63


BASE_SIZE = 40
PENCILS_PER_FAMILY = 512
SEARCH_SEED = 1564
SIGNIFICANCE_LEVEL = 0.01
FAMILY_NAMES = (
    "hash_p1553_r64_a",
    "hash_p1553_r64_b",
    "x_ascending",
    "x_descending",
    "y_ascending",
    "y_descending",
    "x_plus_y_mod_p",
    "x_minus_y_mod_p",
    "two_x_plus_y_mod_p",
    "x_plus_two_y_mod_p",
    "x_squared_mod_p",
    "y_squared_mod_p",
)


def raw_subgroup_records():
    records = []
    for scalar in range(r63.SUBGROUP_ORDER):
        point = r63.r62.r61.r60.r58.r44.r38.scalar_mul(
            scalar,
            r63.r62.r61.r60.r58.r44.r38.GENERATOR,
        )
        encoding = "O" if point is None else f"{point[0]},{point[1]}"
        records.append(
            {
                "point": point,
                "public_encoding": encoding,
                "verification_scalar": scalar,
            }
        )
    return records


def affine_coordinates(record):
    if record["point"] is None:
        return -1, -1, -1
    x_coordinate, y_coordinate = record["point"]
    return 0, x_coordinate, y_coordinate


def family_score(name, record):
    infinity, x_coordinate, y_coordinate = affine_coordinates(record)
    encoding = record["public_encoding"]
    if name.startswith("hash_"):
        return (
            hashlib.sha256(f"{name}|{encoding}".encode()).digest(),
            encoding,
        )
    if infinity < 0:
        return (-1, -1, encoding)
    if name == "x_ascending":
        value = x_coordinate
    elif name == "x_descending":
        value = -x_coordinate
    elif name == "y_ascending":
        value = y_coordinate
    elif name == "y_descending":
        value = -y_coordinate
    elif name == "x_plus_y_mod_p":
        value = (x_coordinate + y_coordinate) % r63.PRIME
    elif name == "x_minus_y_mod_p":
        value = (x_coordinate - y_coordinate) % r63.PRIME
    elif name == "two_x_plus_y_mod_p":
        value = (2 * x_coordinate + y_coordinate) % r63.PRIME
    elif name == "x_plus_two_y_mod_p":
        value = (x_coordinate + 2 * y_coordinate) % r63.PRIME
    elif name == "x_squared_mod_p":
        value = x_coordinate * x_coordinate % r63.PRIME
    elif name == "y_squared_mod_p":
        value = y_coordinate * y_coordinate % r63.PRIME
    else:
        raise ValueError(name)
    return value, x_coordinate, y_coordinate, encoding


def public_base(records, family_name):
    ordered = sorted(records, key=lambda record: family_score(family_name, record))
    base = ordered[:BASE_SIZE]
    digest = hashlib.sha256(
        "|".join(record["public_encoding"] for record in base).encode()
    ).hexdigest()
    return base, digest


def binomial_upper_tail(successes, trials, probability):
    if successes <= 0:
        return 1.0
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 1.0
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(count + 1)
        - math.lgamma(trials - count + 1)
        + count * math.log(probability)
        + (trials - count) * math.log1p(-probability)
        for count in range(successes, trials + 1)
    ]
    maximum = max(logs)
    return min(1.0, math.exp(maximum) * sum(math.exp(value - maximum) for value in logs))


def run_family(
    family_index,
    family_name,
    records,
    factor_bases,
    multiplication_matrix,
    equation,
):
    base, base_digest = public_base(records, family_name)
    catalogs = [
        r63.public_factor_catalog(base, class_sum, basis)
        for class_sum, basis in zip(r63.CLASS_SUM_SCALARS, factor_bases)
    ]
    products, smooth_lookup, multiplicities = r63.smooth_products(
        catalogs,
        multiplication_matrix,
    )
    pairs, pair_draws = r63.accepted_pencil_pairs(
        products,
        SEARCH_SEED + family_index,
    )
    closure = r63.closure_probe(
        products,
        smooth_lookup,
        pairs,
        equation,
        multiplication_matrix,
    )
    projected_counts = [len(catalog) // 4 for catalog in catalogs]
    projective_factor_section_count = r63.PRIME * r63.PRIME + r63.PRIME + 1
    multiplicity_one_smooth_sections = multiplicities[
        "factorization_multiplicity_histogram"
    ].get("1", 0)
    unconditional_reference_intensity = multiplicity_one_smooth_sections / (
        projective_factor_section_count * projective_factor_section_count
    )
    trials = closure["new_rational_rank_one_product_sections"]
    successes = closure["new_smooth_product_roots"]
    raw_tail = binomial_upper_tail(
        successes,
        trials,
        unconditional_reference_intensity,
    )
    return {
        "family_index": family_index,
        "family_name": family_name,
        "base_size": BASE_SIZE,
        "base_digest": base_digest,
        "projected_class_triple_counts": projected_counts,
        "even_torsion_lifted_factor_section_counts": [
            len(catalog) for catalog in catalogs
        ],
        "distinct_smooth_product_sections": len(products),
        "smooth_product_factorizations": multiplicities,
        "source_construction_accounting": {
            "public_triple_candidates_checked": 2 * math.comb(BASE_SIZE, 3),
            "lifted_factor_pair_candidates": len(catalogs[0]) * len(catalogs[1]),
            "pair_candidates_or_random_draws": pair_draws,
        },
        "closure_probe": closure,
        "unconditional_factor_pair_reference_intensity": unconditional_reference_intensity,
        "reference_intensity_times_conditioned_unique_lift_count": (
            unconditional_reference_intensity * trials
        ),
        "observed_smooth_root_rate": successes / trials if trials else None,
        "uncalibrated_rate_to_reference_ratio": (
            successes / trials / unconditional_reference_intensity
            if trials and unconditional_reference_intensity
            else None
        ),
        "uncalibrated_one_sided_binomial_replay_statistic": raw_tail,
    }


def run():
    r61_path = pathlib.Path(
        "p1553_degree_six_product_hypersurface_locator_report_r61.json"
    )
    r61_bytes = r61_path.read_bytes()
    r61_report = json.loads(r61_bytes)
    r63_script_bytes = pathlib.Path(
        "p1553_public_factor_base_closure_probe_r63.py"
    ).read_bytes()
    multiplication_matrix = r61_report["multiplication_map"]["multiplication_matrix"]
    equation = r63.equation_from_report(r61_report)
    factor_bases = r63.r62.factor_basis_lines(r61_report)
    records = raw_subgroup_records()
    rows = [
        run_family(
            index,
            name,
            records,
            factor_bases,
            multiplication_matrix,
            equation,
        )
        for index, name in enumerate(FAMILY_NAMES)
    ]
    for row in rows:
        row["bonferroni_adjusted_tail"] = min(
            1.0,
            row["uncalibrated_one_sided_binomial_replay_statistic"] * len(rows),
        )
        row["uncalibrated_root_level_threshold_crossed"] = (
            row["bonferroni_adjusted_tail"] < SIGNIFICANCE_LEVEL
            and row["observed_smooth_root_rate"]
            > row["unconditional_factor_pair_reference_intensity"]
        )

    threshold_crossings = [
        row["family_name"]
        for row in rows
        if row["uncalibrated_root_level_threshold_crossed"]
    ]
    pooled_roots = sum(
        row["closure_probe"]["new_rational_rank_one_product_sections"] for row in rows
    )
    pooled_lift_attempts = sum(
        row["closure_probe"]["new_distinct_finite_hypersurface_roots"]
        for row in rows
    )
    pooled_smooth = sum(
        row["closure_probe"]["new_smooth_product_roots"] for row in rows
    )
    pooled_reference_product = sum(
        row["reference_intensity_times_conditioned_unique_lift_count"]
        for row in rows
    )
    independent_zero_probability = math.exp(
        sum(
            row["closure_probe"]["new_rational_rank_one_product_sections"]
            * math.log1p(-row["unconditional_factor_pair_reference_intensity"])
            for row in rows
        )
    )
    checks = {
        "direct_r61_report_digest_is_pinned": hashlib.sha256(r61_bytes).hexdigest()
        == "f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64",
        "imported_r63_implementation_digest_is_pinned": hashlib.sha256(
            r63_script_bytes
        ).hexdigest()
        == "7f5d061a539d62d2b3d362fed8252fead5b8e14dbc9250f5543e686f3a8d89cb",
        "family_count_is_preregistered_twelve": len(rows) == 12,
        "family_names_match_constant": tuple(row["family_name"] for row in rows)
        == FAMILY_NAMES,
        "all_bases_have_requested_size": all(row["base_size"] == BASE_SIZE for row in rows),
        "all_pencils_are_disjoint_and_nondegenerate": all(
            row["closure_probe"]["accepted_disjoint_source_pencils"]
            == row["closure_probe"]["nonzero_sextic_restrictions"]
            == PENCILS_PER_FAMILY
            for row in rows
        ),
        "bonferroni_factor_is_twelve": all(
            abs(
                row["bonferroni_adjusted_tail"]
                - min(
                    1.0,
                    12 * row["uncalibrated_one_sided_binomial_replay_statistic"],
                )
            )
            < 1e-15
            for row in rows
        ),
        "uncalibrated_threshold_decision_is_replayed": threshold_crossings
        == [
            row["family_name"]
            for row in rows
            if row["bonferroni_adjusted_tail"] < SIGNIFICANCE_LEVEL
            and row["observed_smooth_root_rate"]
            > row["unconditional_factor_pair_reference_intensity"]
        ],
    }
    if not all(checks.values()):
        raise AssertionError(f"R64 coordinate-base amplification search failed: {checks}")

    ranked = sorted(
        rows,
        key=lambda row: (
            row["bonferroni_adjusted_tail"],
            -(row["uncalibrated_rate_to_reference_ratio"] or 0),
            row["family_index"],
        ),
    )
    return {
        "schema": "p1553.coordinate_base_amplification_search.r64.v1",
        "classification": [
            "toy",
            "preregistered-multiple-hypothesis-search",
            "heuristic-cost-model",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r63.PRIME,
        "prime_subgroup_order": r63.SUBGROUP_ORDER,
        "preregistration": {
            "base_size": BASE_SIZE,
            "base_size_selection": "B=40 was selected after R63 identified it as the first positive base; this choice is data-dependent",
            "pencils_per_family": PENCILS_PER_FAMILY,
            "search_seed": SEARCH_SEED,
            "family_names": list(FAMILY_NAMES),
            "primary_endpoint": "uncalibrated one-sided root-level binomial replay statistic using an unconditional factor-pair reference intensity",
            "endpoint_valid_for_inference": False,
            "endpoint_failure": "the observed population is conditioned on unique rational lifts while the reference is unconditional, and multiple roots are clustered within each pencil",
            "multiple_testing": "Bonferroni across 12 families",
            "familywise_significance_level": SIGNIFICANCE_LEVEL,
        },
        "family_rows": rows,
        "ranking": [row["family_name"] for row in ranked],
        "uncalibrated_root_level_threshold_crossing_families": threshold_crossings,
        "post_hoc_suppression_diagnostic": {
            "status": "descriptive only; root and family independence is false or unverified",
            "pooled_rational_rank_one_product_sections": pooled_roots,
            "pooled_smooth_roots": pooled_smooth,
            "pooled_reference_intensity_times_conditioned_count": pooled_reference_product,
            "probability_of_zero_under_independent_root_model": independent_zero_probability,
            "interpretation": "the pooled zero is reported for replay only and is not a valid p-value for amplification or suppression",
        },
        "checks": checks,
        "cost_boundary": {
            "literal_base_materialization": "O(N log N): N separate double-and-add scalar multiplications plus per-family sorting; uncharged",
            "possible_optimized_base_materialization": "successive group addition could reduce point generation toward O(N), but is not implemented",
            "public_triple_tests_per_family": 2 * math.comb(BASE_SIZE, 3),
            "source_product_materialization": "all disjoint pairs of four even torsion lifts; charged as exact toy counts but not extrapolated",
            "pencil_locator_and_membership_tests": "512 fixed-degree sextic restrictions and factorizations per family",
            "constant_size_rank_one_groebner_lift_attempts": pooled_lift_attempts,
            "generic_prime_distribution_theorem": False,
            "relation_rank_charged": False,
            "projected_scalar_logs_charged": False,
            "linear_algebra_charged": False,
            "target_descent_charged": False,
        },
        "result": {
            "preregistered_coordinate_family_search_completed": True,
            "root_level_preregistered_endpoint_valid": False,
            "statistically_significant_closure_amplification": False,
            "smooth_closure_amplification_inference": "inconclusive",
            "asymptotic_relation_algorithm": False,
            "fresh_target_action": False,
            "r10_queried_coefficients": False,
            "independent_rank": False,
            "factor_base_logs": False,
            "scalar_blind_descent": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the twelve families and threshold are preregistered within this artifact but were selected after observing R63",
            "one toy field and 512 pencils per family cannot establish a generic-prime distribution law",
            "the root-level binomial diagnostic assumes independence inside pencils and is invalid for inference",
            "base setup, projected logs, relation rank, linear algebra, and fresh-target descent remain uncharged",
            "no Shoup-bound improvement or ECDLP breakthrough is claimed",
        ],
        "pass": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    payload = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        pathlib.Path(arguments.output).write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
