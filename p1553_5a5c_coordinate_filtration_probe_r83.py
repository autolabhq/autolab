#!/usr/bin/env python3
"""Audit staged coordinate buckets on the R82 colored 5A+5C source."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Callable, Iterable, Sequence


SCHEMA = "p1553.5a5c_coordinate_filtration_probe.r83.v1"
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4
BASELINE_EXPLICIT_WORK_EXPONENT = 13 / 5
BASELINE_EXPLICIT_STATE_EXPONENT = 12 / 5
BASELINE_GENERIC_WORK_EXPONENT = 5 / 2
PARTIAL_A_COUNT = 2
PARTIAL_C_COUNT = 3
BUCKET_MODULI = (2, 4, 8)
SOURCE_SAMPLE_COUNT = 64

R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
R82_GATE = pathlib.Path("p1553_cartesian_sum_compact_divisor_probe_gate_r82.md")
R82_GATE_SHA256 = (
    "7c34e1d905c95a756689d4ec0ea92c6bd47808bcb3407d858ce08cccf75fd55e"
)
R31_REGISTRY = pathlib.Path("p1553_r31_artifact_index_README.md")
R31_REGISTRY_SHA256 = (
    "0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f"
)
P1539_AUDIT_SHA256 = (
    "634e5a7d2847e849a2e46178f31500f19109e9a9d88a2bf8c70d1f0afe4d467a"
)

Point = tuple[int, int] | None
Bucket = Callable[[Point, int], int]


def load_r82() -> Any:
    path = pathlib.Path(__file__).with_name(
        "p1553_cartesian_sum_compact_divisor_probe_r82.py"
    )
    spec = importlib.util.spec_from_file_location("p1553_r82_for_r83", path)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R82 controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R82 = load_r82()
R81 = R82.R81
R70 = R82.R70


def point_bytes(point: Point) -> bytes:
    if point is None:
        return b"identity"
    return f"{point[0]}:{point[1]}".encode("ascii")


def x_bucket(point: Point, modulus: int) -> int:
    return 0 if point is None else point[0] % modulus


def y_bucket(point: Point, modulus: int) -> int:
    return 0 if point is None else point[1] % modulus


def hash_bucket(point: Point, modulus: int) -> int:
    return int.from_bytes(hashlib.sha256(point_bytes(point)).digest(), "big") % modulus


FILTERS: tuple[tuple[str, Bucket], ...] = (
    ("x_residue", x_bucket),
    ("y_residue", y_bucket),
    ("encoding_hash", hash_bucket),
)


def source_point(
    source: tuple[tuple[int, ...], tuple[int, ...]],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R82.add_many(
        [
            *(atoms_a[index] for index in source[0]),
            *(atoms_c[index] for index in source[1]),
        ],
        curve,
    )


def partial_point(
    source: tuple[tuple[int, ...], tuple[int, ...]],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R82.add_many(
        [
            *(
                atoms_a[index]
                for index in source[0][:PARTIAL_A_COUNT]
            ),
            *(
                atoms_c[index]
                for index in source[1][:PARTIAL_C_COUNT]
            ),
        ],
        curve,
    )


def exact_source_dictionary(
    curve: dict[str, Any],
    offset: int,
) -> tuple[
    list[Point],
    list[Point],
    dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    R81.BatchBsgsVerifier,
    Point,
]:
    generator = R81.curve_generator(curve)
    verifier = R81.BatchBsgsVerifier(generator, curve)
    atoms_a, atoms_c, _, _ = R82.compact_factor_base(curve, offset)
    labels_a = verifier.labels(atoms_a)
    labels_c = verifier.labels(atoms_c)
    left, left_first = R82.weighted_multiset_histogram(
        labels_a,
        R82.RELATION_ARITY,
        curve["subgroup_order"],
    )
    right, right_first = R82.weighted_multiset_histogram(
        labels_c,
        R82.RELATION_ARITY,
        curve["subgroup_order"],
    )
    _, sources = R82.convolve_label_histograms(
        left,
        right,
        left_first,
        right_first,
        curve["subgroup_order"],
    )
    for endpoint, source in sources.items():
        point = source_point(source, atoms_a, atoms_c, curve)
        expected = R70.scalar_mul(endpoint, generator, curve)
        if point != expected:
            raise AssertionError("canonical source missed verifier endpoint")
    return atoms_a, atoms_c, sources, verifier, generator


def homomorphism_profile(
    points: Sequence[Point],
    bucket: Bucket,
    modulus: int,
    curve: dict[str, Any],
) -> dict[str, Any]:
    checks = 0
    successes = 0
    for left in points[:64]:
        for right in points[:64]:
            checks += 1
            endpoint = R70.add(left, right, curve)
            successes += (
                bucket(endpoint, modulus)
                == (bucket(left, modulus) + bucket(right, modulus)) % modulus
            )
    return {
        "pair_count": checks,
        "additivity_success_count": successes,
        "additivity_success_rate": successes / checks if checks else 0.0,
        "is_additive_on_sample": successes == checks,
    }


def filtration_profile(
    sources: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    bucket_name: str,
    bucket: Bucket,
    modulus: int,
    curve: dict[str, Any],
    generator: Point,
) -> dict[str, Any]:
    static_counts: collections.Counter[int] = collections.Counter()
    coupled_offsets: collections.Counter[int] = collections.Counter()
    partial_points = []
    all_endpoint_scalars = sorted(sources)
    for endpoint in all_endpoint_scalars:
        source = sources[endpoint]
        partial = partial_point(source, atoms_a, atoms_c, curve)
        target = R70.scalar_mul(endpoint, generator, curve)
        partial_bucket = bucket(partial, modulus)
        target_bucket = bucket(target, modulus)
        static_counts[partial_bucket] += 1
        coupled_offsets[(partial_bucket - target_bucket) % modulus] += 1
        partial_points.append(partial)
    source_count = len(sources)
    static_best = max(static_counts.values(), default=0)
    coupled_best = max(coupled_offsets.values(), default=0)
    return {
        "bucket_name": bucket_name,
        "bucket_modulus": modulus,
        "canonical_target_source_count": source_count,
        "static_bucket_counts": [
            static_counts[index] for index in range(modulus)
        ],
        "target_coupled_offset_counts": [
            coupled_offsets[index] for index in range(modulus)
        ],
        "best_static_bucket_survival_count": static_best,
        "best_static_bucket_survival_fraction": (
            static_best / source_count if source_count else 0.0
        ),
        "best_target_coupled_offset_survival_count": coupled_best,
        "best_target_coupled_offset_survival_fraction": (
            coupled_best / source_count if source_count else 0.0
        ),
        "single_static_bucket_is_complete": static_best == source_count,
        "single_target_coupled_offset_is_complete": coupled_best == source_count,
        "replaying_all_static_buckets_recovers_every_source": (
            sum(static_counts.values()) == source_count
        ),
        "replaying_all_coupled_offsets_recovers_every_source": (
            sum(coupled_offsets.values()) == source_count
        ),
        "required_exact_bucket_replays": modulus,
        "homomorphism_control": homomorphism_profile(
            partial_points,
            bucket,
            modulus,
            curve,
        ),
    }


def select_sources(
    sources: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    count: int,
    salt: str,
) -> list[tuple[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
    endpoints = sorted(sources)
    if len(endpoints) <= count:
        return [(endpoint, sources[endpoint]) for endpoint in endpoints]
    selected = []
    used = set()
    cursor = 0
    while len(selected) < count:
        digest = hashlib.sha256(f"{salt}|{cursor}".encode()).digest()
        cursor += 1
        endpoint = endpoints[int.from_bytes(digest, "big") % len(endpoints)]
        if endpoint in used:
            continue
        used.add(endpoint)
        selected.append((endpoint, sources[endpoint]))
    return selected


def semaev_chain_replay(
    selected: Iterable[
        tuple[int, tuple[tuple[int, ...], tuple[int, ...]]]
    ],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
    generator: Point,
) -> dict[str, Any]:
    source_count = 0
    group_failures = 0
    s3_checks = 0
    s3_failures = 0
    exceptional_identity_steps = 0
    for endpoint, source in selected:
        source_count += 1
        points = [
            *(atoms_a[index] for index in source[0]),
            *(atoms_c[index] for index in source[1]),
        ]
        target = R70.scalar_mul(endpoint, generator, curve)
        if R82.add_many(points, curve) != target:
            group_failures += 1
            continue
        running = points[0]
        for point in points[1:]:
            following = R70.add(running, point, curve)
            if running is None or point is None or following is None:
                exceptional_identity_steps += 1
            else:
                s3_checks += 1
                if R70.semaev_s3(
                    running[0],
                    point[0],
                    following[0],
                    curve,
                ) != 0:
                    s3_failures += 1
            running = following
        if running != target:
            group_failures += 1
    return {
        "sampled_source_count": source_count,
        "group_endpoint_failure_count": group_failures,
        "s3_auxiliary_chain_check_count": s3_checks,
        "s3_auxiliary_chain_failure_count": s3_failures,
        "exceptional_identity_step_count": exceptional_identity_steps,
        "all_group_sources_exact": group_failures == 0,
        "all_regular_s3_chain_steps_vanish": s3_failures == 0,
        "chain_interface": (
            "ten atom x-coordinates plus eight cumulative-sum auxiliary "
            "x-coordinates; exceptional identity strata replayed by group law"
        ),
        "solver_supplied": False,
    }


def divisible_order_quotient_control(modulus: int = 8) -> dict[str, Any]:
    group_order = 8 * 101
    values = list(range(group_order))
    failures = 0
    for left in values[:64]:
        for right in values[:64]:
            failures += (
                ((left + right) % group_order) % modulus
                != (left % modulus + right % modulus) % modulus
            )
    return {
        "group": f"Z/{group_order}Z",
        "quotient": f"Z/{modulus}Z",
        "proper_kernel_size": group_order // modulus,
        "checked_pair_count": 64**2,
        "failure_count": failures,
        "exact_homomorphic_filter": failures == 0,
        "purpose": (
            "positive control showing what an addition-compatible staged "
            "bucket looks like when the group order has a proper divisor"
        ),
    }


def prime_order_homomorphism_theorem() -> dict[str, Any]:
    return {
        "statement": (
            "For G of prime order q, every homomorphism G->H has kernel G "
            "or {O}; therefore it is constant or injective."
        ),
        "proof": (
            "The kernel is a subgroup of G. A prime-order group has only "
            "{O} and G as subgroups. A nonconstant map has trivial kernel "
            "and image order q, so it cannot be a proper filtering quotient."
        ),
        "proper_nontrivial_quotient_exists": False,
        "encoding_or_coordinate_bucket_is_covered": False,
        "reason_not_covered": (
            "x, y, and encoding hashes are not homomorphisms; they are "
            "measured separately by source-survival controls"
        ),
    }


def entropy_replay_theorem() -> dict[str, Any]:
    return {
        "relation_scale": {
            "source_exponent_B": 5.0,
            "group_order_exponent_B": 5.0,
            "distinct_target_support_exponent_B": 5.0,
            "asymptotic_solution_entropy_exponent_B": 0.0,
        },
        "static_subset_bound": (
            "A frozen accepted subset containing M canonical sources covers "
            "at most M target endpoints. If M=B^(5-delta), its average "
            "target success density is at most B^(-delta+o(1))."
        ),
        "exact_replay_consequence": (
            "Restoring constant target success requires B^(delta-o(1)) "
            "disjoint masks, offsets, or retries unless a constraint is "
            "forced by the target equation or a compact selector covers one "
            "source for every attained endpoint."
        ),
        "explicit_join_to_online_cap": {
            "baseline_work_exponent_B": BASELINE_EXPLICIT_WORK_EXPONENT,
            "desired_per_filter_work_exponent_B": ONLINE_CAP_EXPONENT,
            "required_filter_exponent_delta_B": (
                BASELINE_EXPLICIT_WORK_EXPONENT - ONLINE_CAP_EXPONENT
            ),
            "required_replay_exponent_B": (
                BASELINE_EXPLICIT_WORK_EXPONENT - ONLINE_CAP_EXPONENT
            ),
            "restored_total_work_exponent_B": (
                BASELINE_EXPLICIT_WORK_EXPONENT
            ),
        },
        "generic_collision_to_online_cap": {
            "baseline_work_exponent_B": BASELINE_GENERIC_WORK_EXPONENT,
            "desired_per_filter_work_exponent_B": ONLINE_CAP_EXPONENT,
            "required_filter_exponent_delta_B": (
                BASELINE_GENERIC_WORK_EXPONENT - ONLINE_CAP_EXPONENT
            ),
            "required_replay_exponent_B": (
                BASELINE_GENERIC_WORK_EXPONENT - ONLINE_CAP_EXPONENT
            ),
            "restored_total_work_exponent_B": (
                BASELINE_GENERIC_WORK_EXPONENT
            ),
        },
        "exceptions": [
            "a target-forced algebraic constraint with no entropy loss",
            "a compact source section covering every attained endpoint",
            "a representation-specific solver whose work is not accepted-source enumeration",
        ],
        "claim_scope": (
            "explicit frozen bucket subsets and target-coupled offset replay; "
            "not an arithmetic-circuit or algebraic-elimination lower bound"
        ),
    }


def analyze_instance(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    atoms_a, atoms_c, sources, verifier, generator = exact_source_dictionary(
        curve,
        offset,
    )
    profiles = [
        filtration_profile(
            sources,
            atoms_a,
            atoms_c,
            filter_name,
            bucket,
            modulus,
            curve,
            generator,
        )
        for filter_name, bucket in FILTERS
        for modulus in BUCKET_MODULI
    ]
    selected = select_sources(
        sources,
        SOURCE_SAMPLE_COUNT,
        f"{curve['family_id']}|{offset}",
    )
    return {
        "family_id": curve["family_id"],
        "field_prime": curve["field_prime"],
        "subgroup_order": curve["subgroup_order"],
        "offset": offset,
        "factor_base_size": curve["atom_a_size"] * curve["atom_c_size"],
        "canonical_target_source_count": len(sources),
        "filters": profiles,
        "semaev_chain_replay": semaev_chain_replay(
            selected,
            atoms_a,
            atoms_c,
            curve,
            generator,
        ),
        "verifier_bsgs_receipt": verifier.receipt(),
    }


def build_bundle(
    families: Sequence[dict[str, Any]] = R82.FAMILIES,
    offsets: Sequence[int] = R82.INSTANCE_OFFSETS,
) -> dict[str, dict[str, Any]]:
    instances = [
        analyze_instance(dict(curve), offset)
        for curve in families
        for offset in offsets
    ]
    filters = [
        profile
        for instance in instances
        for profile in instance["filters"]
    ]
    all_group_sources = all(
        instance["semaev_chain_replay"]["all_group_sources_exact"]
        for instance in instances
    )
    all_s3_chains = all(
        instance["semaev_chain_replay"][
            "all_regular_s3_chain_steps_vanish"
        ]
        for instance in instances
    )
    no_single_filter_complete = all(
        not profile["single_static_bucket_is_complete"]
        and not profile["single_target_coupled_offset_is_complete"]
        for profile in filters
    )
    all_bucket_replays_complete = all(
        profile["replaying_all_static_buckets_recovers_every_source"]
        and profile["replaying_all_coupled_offsets_recovers_every_source"]
        for profile in filters
    )
    no_actual_filter_homomorphic = all(
        not profile["homomorphism_control"]["is_additive_on_sample"]
        for profile in filters
    )

    frozen = {
        "schema": "p1553.frozen_5a5c_coordinate_filtration.r83.v1",
        "source_geometry": "R82 F=A+C with u=B^(2/5), v=B^(3/5)",
        "canonical_source_rule": (
            "lexicographically first weighted multiset source per attained "
            "endpoint, frozen before filter outcomes"
        ),
        "partial_split": {
            "a_atom_count": PARTIAL_A_COUNT,
            "c_atom_count": PARTIAL_C_COUNT,
            "partial_exponent_B": (
                PARTIAL_A_COUNT * R82.ATOM_A_EXPONENT
                + PARTIAL_C_COUNT * R82.ATOM_C_EXPONENT
            ),
            "complement_exponent_B": (
                R82.RELATION_ARITY
                - PARTIAL_A_COUNT * R82.ATOM_A_EXPONENT
                - PARTIAL_C_COUNT * R82.ATOM_C_EXPONENT
            ),
        },
        "filters": [
            {
                "name": name,
                "definition": {
                    "x_residue": "x(P) mod m, identity mapped to 0",
                    "y_residue": "y(P) mod m, identity mapped to 0",
                    "encoding_hash": "SHA256(affine encoding of P) mod m",
                }[name],
                "moduli": list(BUCKET_MODULI),
            }
            for name, _ in FILTERS
        ],
        "target_coupling": "bucket(partial)-bucket(target) mod m",
        "candidate_scalar_labels_consumed": False,
        "verifier_labels_used_to_index_attained_targets_only": True,
    }
    controls = {
        "schema": (
            "p1553.partial_filter_composability_false_positive_controls.r83.v1"
        ),
        "prime_order_homomorphism_theorem": (
            prime_order_homomorphism_theorem()
        ),
        "divisible_order_positive_control": divisible_order_quotient_control(),
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "canonical_target_source_count": instance[
                    "canonical_target_source_count"
                ],
                "filters": instance["filters"],
            }
            for instance in instances
        ],
        "all_actual_filters_nonadditive_on_samples": (
            no_actual_filter_homomorphic
        ),
        "no_single_frozen_bucket_or_offset_complete": (
            no_single_filter_complete
        ),
        "all_bucket_or_offset_replays_complete": (
            all_bucket_replays_complete
        ),
        "minimum_best_static_survival_fraction": min(
            profile["best_static_bucket_survival_fraction"]
            for profile in filters
        ),
        "maximum_best_static_survival_fraction": max(
            profile["best_static_bucket_survival_fraction"]
            for profile in filters
        ),
        "minimum_best_coupled_survival_fraction": min(
            profile["best_target_coupled_offset_survival_fraction"]
            for profile in filters
        ),
        "maximum_best_coupled_survival_fraction": max(
            profile["best_target_coupled_offset_survival_fraction"]
            for profile in filters
        ),
    }
    ffe_replay = {
        "schema": "p1553.summation_polynomial_ffe_source_replay.r83.v1",
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                **instance["semaev_chain_replay"],
            }
            for instance in instances
        ],
        "all_group_sources_exact": all_group_sources,
        "all_regular_s3_auxiliary_chains_exact": all_s3_chains,
        "candidate_solver_supplied": False,
        "interpretation": (
            "The factorized S3 chain is an exact certificate for supplied "
            "sources, not a subcap constructor or selector."
        ),
    }
    cost_ledger = {
        "schema": "p1553.full_query_state_cost_ledger.r83.v1",
        "caps": {
            "setup_state_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "online_work_exponent_B": ONLINE_CAP_EXPONENT,
            "online_workspace_exponent_B": ONLINE_CAP_EXPONENT,
        },
        "entropy_replay_theorem": entropy_replay_theorem(),
        "r82_baseline": {
            "explicit_join_work_exponent_B": (
                BASELINE_EXPLICIT_WORK_EXPONENT
            ),
            "explicit_join_state_exponent_B": (
                BASELINE_EXPLICIT_STATE_EXPONENT
            ),
            "generic_collision_work_exponent_B": (
                BASELINE_GENERIC_WORK_EXPONENT
            ),
        },
        "fixed_coordinate_bucket_inside_caps_with_constant_success": False,
        "target_coupled_offset_inside_caps_with_constant_success": False,
        "all_offset_replay_improves_total_exponent": False,
        "forced_algebraic_selector_supplied": False,
        "full_query_inside_cap": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r83.v1",
        "algorithmic_known_rhs_source_complete": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "reason": (
            "Every tested coordinate bucket loses attained targets; exact "
            "offset replay restores the source-search exponent."
        ),
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }

    obligations = {
        "r82_geometry_replayed": len(instances) == len(families) * len(offsets),
        "canonical_source_per_attained_target": all(
            instance["canonical_target_source_count"] > 0
            for instance in instances
        ),
        "x_y_hash_filters_frozen": len(FILTERS) == 3,
        "static_and_target_coupled_offsets_measured": True,
        "prime_order_homomorphism_theorem_supplied": True,
        "divisible_order_positive_control_passes": (
            controls["divisible_order_positive_control"][
                "exact_homomorphic_filter"
            ]
        ),
        "all_actual_filters_nonadditive": no_actual_filter_homomorphic,
        "no_single_bucket_or_offset_complete": no_single_filter_complete,
        "all_bucket_replays_complete": all_bucket_replays_complete,
        "exact_group_source_replay": all_group_sources,
        "exact_s3_auxiliary_chain_replay": all_s3_chains,
        "full_query_inside_online_cap_with_constant_success": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_target_descent": False,
        "generic_prime_breakthrough": False,
        "shoup_improvement_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    report = {
        "schema": SCHEMA,
        "classification": (
            "COORDINATE_BUCKET_FILTRATION_ENTROPY_REPLAY_FAIL"
        ),
        "source_bindings": {
            "r82_report": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "r82_gate": {
                "path": str(R82_GATE),
                "sha256": R82_GATE_SHA256,
            },
            "r31_registry": {
                "path": str(R31_REGISTRY),
                "sha256": R31_REGISTRY_SHA256,
                "bound_p1539_audit_sha256": P1539_AUDIT_SHA256,
            },
        },
        "deduplication": (
            "P1539 already screened broad neutral-mask Wagner merges for "
            "colored elliptic 5SUM. R83 adds the R82-specific canonical "
            "source-survival theorem, target-coupled offset controls, exact "
            "bucket replay, and S3 auxiliary-chain receipts; it claims no "
            "new general kSUM lower bound."
        ),
        "instances": instances,
        "aggregate": {
            "instance_count": len(instances),
            "filter_profile_count": len(filters),
            "all_group_sources_exact": all_group_sources,
            "all_regular_s3_auxiliary_chains_exact": all_s3_chains,
            "all_actual_filters_nonadditive_on_samples": (
                no_actual_filter_homomorphic
            ),
            "no_single_frozen_bucket_or_offset_complete": (
                no_single_filter_complete
            ),
            "all_bucket_or_offset_replays_complete": (
                all_bucket_replays_complete
            ),
            "best_static_survival_fraction_range": [
                controls["minimum_best_static_survival_fraction"],
                controls["maximum_best_static_survival_fraction"],
            ],
            "best_target_coupled_survival_fraction_range": [
                controls["minimum_best_coupled_survival_fraction"],
                controls["maximum_best_coupled_survival_fraction"],
            ],
        },
        "side_artifacts": {
            "frozen_filtration": "frozen_5a5c_coordinate_filtration.json",
            "controls": (
                "partial_filter_composability_and_false_positive_controls.json"
            ),
            "ffe_replay": "summation_polynomial_ffe_source_replay.json",
            "cost_ledger": "full_query_state_cost_ledger.json",
            "logs_descent": "factor_logs_and_identical_descent.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": (
            "This rejects fixed x/y/hash buckets and target-coupled offset "
            "replay on the R82 canonical-source grammar. It does not reject "
            "a target-forced algebraic invariant, compact source section, "
            "marked resultant, or arbitrary FFE/elimination circuit."
        ),
        "next_action": (
            "Construct or refute one P1510-style marked-resultant source "
            "section for the R82 2A+3C versus 3A+2C split. It must cover "
            "every attained target without bucket replay, return one jointly "
            "coupled atom source, and receive full coefficient, state, "
            "fresh-query, rank, log, and identical-descent receipts."
        ),
        "disposition": (
            "REJECT_FIXED_COORDINATE_FILTRATION_ONLY__R82_CANONICAL_SOURCE_"
            "PER_ATTAINED_TARGET__X_Y_HASH_BUCKETS_M2_M4_M8__NO_ACTUAL_"
            "FILTER_ADDITIVE__NO_SINGLE_BUCKET_OR_TARGET_OFFSET_COMPLETE__"
            "ALL_BUCKET_REPLAY_EXACT__CONSTANT_RELATION_ENTROPY_RESTORES_"
            "FILTER_EXPONENT__DIVISIBLE_ORDER_QUOTIENT_POSITIVE_CONTROL__"
            "EXACT_S3_AUXILIARY_CHAIN__NO_SUBCAP_CONSTRUCTOR__NO_FACTOR_"
            "LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "controls": controls,
        "ffe_replay": ffe_replay,
        "cost_ledger": cost_ledger,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_coordinate_filtration_probe_report_r83.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path("frozen_5a5c_coordinate_filtration.json"),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "partial_filter_composability_and_false_positive_controls.json"
        ),
    )
    parser.add_argument(
        "--ffe-output",
        type=pathlib.Path,
        default=pathlib.Path("summation_polynomial_ffe_source_replay.json"),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path("full_query_state_cost_ledger.json"),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent.json"),
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
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.ffe_output, bundle["ffe_replay"])
    write_json(args.cost_output, bundle["cost_ledger"])
    write_json(args.logs_output, bundle["logs_descent"])
    aggregate = bundle["report"]["aggregate"]
    print(
        f"instances={aggregate['instance_count']} "
        f"filters={aggregate['filter_profile_count']} "
        f"replay={aggregate['all_bucket_or_offset_replays_complete']} "
        f"lane_admitted={bundle['report']['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
