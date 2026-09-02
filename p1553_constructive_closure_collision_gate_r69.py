#!/usr/bin/env python3
"""Audit rank conservation and collision cost in a scalar-blind closure source."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
from typing import Any

import p1553_scalar_blind_rank_descent_audit_r67 as r67


SCHEMA = "p1553.constructive_closure_collision_gate.r69.v1"
INITIAL_BASE_SIZE = 12
R67_SOURCE = pathlib.Path("p1553_scalar_blind_rank_descent_audit_r67.py")
R67_REPORT = pathlib.Path("p1553_scalar_blind_rank_descent_audit_report_r67.json")
R68_SOURCE = pathlib.Path("p1553_ffe_fixed_sum_information_conservation_r68.py")
R68_REPORT = pathlib.Path(
    "p1553_ffe_fixed_sum_information_conservation_report_r68.json"
)
PINNED_SHA256 = {
    str(R67_SOURCE): "60cf8dc258f2ea097a8581bd95b8002829be492a512cf336c270abda6876129e",
    str(R67_REPORT): "fc47d3d509597fe5b7a0928c5fcffc4cd6c60679ea7efc7d531f4b24b28f143b",
    str(R68_SOURCE): "b8aa38732e577d10c7e7326f01ed5d3a688627963f7d0b98dee3f6a2078deddc",
    str(R68_REPORT): "5c064a36630c052093ebb7bd3b4429d37f836982ba097eacbad2f2ebe1bb6b51",
}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point_to_json(point: tuple[int, int] | None) -> list[int] | None:
    return None if point is None else list(point)


def relation_row(indices: tuple[int, ...], column_count: int) -> list[int]:
    row = [0] * column_count
    for index in indices:
        row[index] += 1
    return row


def rank(rows: list[list[int]], column_count: int) -> int:
    return len(r67.row_reduce(rows, column_count)[1])


def public_base_prefix() -> tuple[list[dict[str, Any]], dict[str, int]]:
    records, _ = r67.public_factor_base()
    prefix = records[:INITIAL_BASE_SIZE]
    counters = [
        int(record["source"].removeprefix("sha256_counter_"))
        for record in prefix
        if str(record["source"]).startswith("sha256_counter_")
    ]
    return prefix, {
        "anchor_count": 2,
        "accepted_hash_point_count": INITIAL_BASE_SIZE - 2,
        "hash_counter_attempts_through_prefix": max(counters) + 1,
    }


def closure_replay() -> dict[str, Any]:
    curve = r67.r65.r63.r62.r61.r60.r58.r44.r38
    base, base_cost = public_base_prefix()
    atoms = [
        {
            "atom_index": index,
            "point": record["point"],
            "source": record["source"],
            "origin": "independent_public_seed",
        }
        for index, record in enumerate(base)
    ]
    atom_index = {record["point"]: index for index, record in enumerate(atoms)}
    events = []

    for first, second in itertools.combinations(range(INITIAL_BASE_SIZE), 2):
        residual = curve.negate(
            curve.add(atoms[first]["point"], atoms[second]["point"])
        )
        residual_index = atom_index.get(residual)
        if residual_index is None:
            residual_index = len(atoms)
            atom_index[residual] = residual_index
            atoms.append(
                {
                    "atom_index": residual_index,
                    "point": residual,
                    "source": f"residual_of_seed_pair_{first}_{second}",
                    "origin": "constructively_introduced_residual",
                }
            )
            outcome = "fresh_residual"
            collision_type = None
        else:
            outcome = "closure_collision"
            collision_type = (
                "independent_seed_hit"
                if residual_index < INITIAL_BASE_SIZE
                else "repeated_generated_residual"
            )
        if r67.r65.r63.add_points(
            (
                atoms[first]["point"],
                atoms[second]["point"],
                atoms[residual_index]["point"],
            )
        ) is not None:
            raise AssertionError("constructive closure row failed public group replay")
        events.append(
            {
                "proposal_index": len(events),
                "parent_indices": [first, second],
                "residual_index": residual_index,
                "residual_point": point_to_json(residual),
                "outcome": outcome,
                "collision_type": collision_type,
            }
        )

    column_count = len(atoms)
    fresh_events = [event for event in events if event["outcome"] == "fresh_residual"]
    collision_events = [
        event for event in events if event["outcome"] == "closure_collision"
    ]
    for event in events:
        event["row"] = relation_row(
            (*event["parent_indices"], event["residual_index"]),
            column_count,
        )

    fresh_rows = [event["row"] for event in fresh_events]
    collision_rows = [event["row"] for event in collision_events]
    fresh_rank = rank(fresh_rows, column_count)
    current_rows = list(fresh_rows)
    current_rank = fresh_rank
    independent_collision_count = 0
    for event in collision_events:
        next_rank = rank(current_rows + [event["row"]], column_count)
        event["independent_after_fresh_basis"] = next_rank > current_rank
        if next_rank > current_rank:
            independent_collision_count += 1
        current_rows.append(event["row"])
        current_rank = next_rank

    reduced, pivots = r67.row_reduce(current_rows, column_count)
    nullspace = r67.nullspace_basis(reduced, pivots, column_count)
    if len(nullspace) != 1:
        raise AssertionError("closure relation system did not leave one orientation")
    oriented_logs = list(nullspace[0])
    generator_coefficient = oriented_logs[1]
    if not generator_coefficient:
        raise AssertionError("generator anchor cannot orient closure nullspace")
    inverse = pow(generator_coefficient, -1, r67.r65.r63.SUBGROUP_ORDER)
    oriented_logs = [
        value * inverse % r67.r65.r63.SUBGROUP_ORDER for value in oriented_logs
    ]
    if oriented_logs[:2] != [0, 1]:
        raise AssertionError("identity/generator anchors failed closure orientation")
    verification_failures = [
        index
        for index, (record, scalar) in enumerate(zip(atoms, oriented_logs))
        if r67.raw_scalar_mul(
            scalar,
            curve.GENERATOR,
        )
        != record["point"]
    ]

    return {
        "initial_base": {
            "size": INITIAL_BASE_SIZE,
            "points": [point_to_json(record["point"]) for record in atoms[:INITIAL_BASE_SIZE]],
            "sources": [record["source"] for record in atoms[:INITIAL_BASE_SIZE]],
            "construction_cost": base_cost,
            "scalar_labels_consumed_by_source": False,
        },
        "proposal_stream": {
            "definition": "all unordered pairs of the frozen initial public base",
            "proposal_count": len(events),
            "expected_pair_count": math.comb(INITIAL_BASE_SIZE, 2),
            "events": events,
        },
        "rank_accounting": {
            "final_atom_count": column_count,
            "fresh_residual_count": len(fresh_events),
            "closure_collision_count": len(collision_events),
            "fresh_residual_row_rank": fresh_rank,
            "nullity_after_fresh_residual_rows": column_count - fresh_rank,
            "independent_closure_collision_count": independent_collision_count,
            "dependent_closure_collision_count": (
                len(collision_events) - independent_collision_count
            ),
            "final_relation_rank": current_rank,
            "final_nullity": column_count - current_rank,
            "fresh_rows_preserve_initial_nullity": (
                column_count - fresh_rank == INITIAL_BASE_SIZE
            ),
            "rank_reduction_equals_independent_collision_count": (
                current_rank - fresh_rank == independent_collision_count
            ),
        },
        "factor_log_recovery": {
            "oriented_by_public_identity_and_generator": True,
            "recovered_log_count": len(oriented_logs),
            "verification_failure_indices": verification_failures,
            "all_recovered_logs_verify_by_public_scalar_multiplication": (
                not verification_failures
            ),
            "logs": oriented_logs,
        },
        "atoms": [
            {
                **record,
                "point": point_to_json(record["point"]),
            }
            for record in atoms
        ],
        "_runtime": {
            "points": [record["point"] for record in atoms],
            "point_to_index": atom_index,
            "logs": oriented_logs,
        },
    }


def target_descent(replay: dict[str, Any], target: tuple[int, int]) -> dict[str, Any]:
    curve = r67.r65.r63.r62.r61.r60.r58.r44.r38
    points = replay["_runtime"]["points"]
    point_to_index = replay["_runtime"]["point_to_index"]
    logs = replay["_runtime"]["logs"]
    pairs = set()
    for first, point in enumerate(points):
        second_point = curve.negate(curve.add(point, target))
        second = point_to_index.get(second_point)
        if second is not None:
            pairs.add(tuple(sorted((first, second))))
    candidates = sorted(
        {
            (-logs[first] - logs[second]) % r67.r65.r63.SUBGROUP_ORDER
            for first, second in pairs
        }
    )
    verified = [
        scalar
        for scalar in candidates
        if r67.raw_scalar_mul(scalar, curve.GENERATOR) == target
    ]
    return {
        "target_point": point_to_json(target),
        "target_was_absent_from_precomputation": target not in point_to_index,
        "linear_complement_lookups": len(points),
        "distinct_decomposition_pairs": [list(pair) for pair in sorted(pairs)],
        "distinct_candidate_logs": candidates,
        "verified_candidate_logs": verified,
        "recovered": len(verified) == 1 and verified == candidates,
    }


def exponent_rows() -> list[dict[str, Any]]:
    rows = []
    for beta in (0.0, 0.2, 0.25, 1 / 3, 0.4, 0.5):
        proposal_exponent = (1 + beta) / 2
        rows.append(
            {
                "seed_exponent_beta": beta,
                "proposals_for_B_self_collisions_exponent": proposal_exponent,
                "rho_exponent": 0.5,
                "strictly_below_rho": proposal_exponent < 0.5,
                "complete_seed_pair_stream_exponent": 2 * beta,
                "complete_pair_stream_has_B_expected_collisions": beta >= 1 / 3,
            }
        )
    return rows


def run() -> dict[str, Any]:
    source_hashes = {
        path: sha256_file(pathlib.Path(path)) for path in PINNED_SHA256
    }
    if source_hashes != PINNED_SHA256:
        raise AssertionError("R69 dependency hash mismatch")
    r67_report = json.loads(R67_REPORT.read_text(encoding="utf-8"))
    r68_report = json.loads(R68_REPORT.read_text(encoding="utf-8"))
    replay = closure_replay()
    target = tuple(r67_report["target_input"]["point"])
    descent = target_descent(replay, target)
    runtime = replay.pop("_runtime")
    del runtime

    subgroup_order = r67.r65.r63.SUBGROUP_ORDER
    rho_operations = math.ceil(math.sqrt(math.pi * subgroup_order / 2))
    proposal_count = replay["proposal_stream"]["proposal_count"]
    online_lookups = descent["linear_complement_lookups"]
    rank_accounting = replay["rank_accounting"]
    obligations = {
        "source_is_scalar_blind_and_frozen_before_outcomes": replay["initial_base"][
            "scalar_labels_consumed_by_source"
        ]
        is False,
        "fresh_residual_rows_preserve_seed_nullity": rank_accounting[
            "fresh_rows_preserve_initial_nullity"
        ],
        "only_closure_collisions_reduce_nullity": rank_accounting[
            "rank_reduction_equals_independent_collision_count"
        ],
        "positive_independent_collision_rank_on_toy": rank_accounting[
            "independent_closure_collision_count"
        ]
        > 0,
        "source_cheaper_than_direct_initial_pair_enumeration": proposal_count
        < math.comb(INITIAL_BASE_SIZE, 2),
        "source_cheaper_than_toy_rho": proposal_count < rho_operations,
        "online_descent_cheaper_than_toy_rho": online_lookups < rho_operations,
        "prospective_four_family_replay": False,
    }
    admission = {
        "lane_admitted": all(obligations.values()),
        "passed_obligation_count": sum(obligations.values()),
        "obligation_count": len(obligations),
        "obligations": obligations,
    }
    checks = {
        "dependencies_are_hash_pinned": source_hashes == PINNED_SHA256,
        "r67_and_r68_pass": r67_report["pass"] and r68_report["pass"],
        "proposal_stream_is_complete_initial_pair_stream": proposal_count
        == math.comb(INITIAL_BASE_SIZE, 2),
        "fresh_residual_rows_are_independent": rank_accounting[
            "fresh_residual_row_rank"
        ]
        == rank_accounting["fresh_residual_count"],
        "fresh_residual_rows_preserve_initial_nullity": rank_accounting[
            "fresh_rows_preserve_initial_nullity"
        ],
        "collision_rank_accounting_is_exact": rank_accounting[
            "rank_reduction_equals_independent_collision_count"
        ],
        "final_nullity_is_one": rank_accounting["final_nullity"] == 1,
        "all_logs_verify": replay["factor_log_recovery"][
            "all_recovered_logs_verify_by_public_scalar_multiplication"
        ],
        "fresh_target_log_recovered": descent["recovered"],
        "uniform_collision_source_never_strictly_beats_rho_for_growing_base": all(
            not row["strictly_below_rho"] for row in exponent_rows()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R69 constructive-closure gate failed: {checks}")

    return {
        "schema": SCHEMA,
        "classification": (
            "SCALAR_BLIND_CLOSURE_COLLISIONS_RECOVER_TOY_LOGS_BUT_COST_ABOVE_RHO"
        ),
        "labels": [
            "exact-rank-nullity-control",
            "scalar-blind-toy-replay",
            "model-bound-collision-cost",
            "novelty-unverified",
        ],
        "field_prime": r67.r65.r63.PRIME,
        "prime_subgroup_order": subgroup_order,
        "general_rank_identity": {
            "statement": (
                "A relation that introduces f fresh atom variables and contributes "
                "at most t independent equations changes nullity by at least f-t."
            ),
            "single_residual_consequence": (
                "A constructive line or summation step with one fresh residual and "
                "one equation cannot reduce unresolved-log nullity."
            ),
            "collision_consequence": (
                "An independent rank reduction requires a residual already present "
                "from an independent seed or a different constructive path."
            ),
            "ffe_binding": (
                "R68 removes product-quotient information credit; R69 locates the "
                "remaining information event at new-factor closure collisions."
            ),
        },
        "toy_replay": replay,
        "fresh_target_descent": descent,
        "charged_cost_lower_bounds": {
            "rho_expected_group_operations_ceil_sqrt_pi_N_over_2": rho_operations,
            "precomputation_pair_proposals": proposal_count,
            "precomputation_group_additions_lower_bound": proposal_count,
            "precomputation_ratio_vs_rho_before_base_and_linear_algebra": (
                proposal_count / rho_operations
            ),
            "direct_initial_pair_enumeration_operations": math.comb(
                INITIAL_BASE_SIZE, 2
            ),
            "source_savings_vs_direct_initial_pair_enumeration": 0,
            "online_target_complement_lookups": online_lookups,
            "online_group_additions_lower_bound": online_lookups,
            "online_ratio_vs_rho": online_lookups / rho_operations,
            "omitted_costs_make_result_only_worse": [
                "hash-to-curve field arithmetic",
                "relation-matrix row reduction",
                "nullspace orientation",
                "public scalar-multiplication verification",
            ],
        },
        "uniform_collision_model": {
            "seed_count": "B",
            "proposal_count": "M",
            "expected_seed_hits": "Theta(M*B/N)",
            "expected_repeated_residual_collisions": "Theta(M^2/N)",
            "proposals_for_Theta_B_collision_rows": "M=Omega(sqrt(B*N))",
            "complete_seed_pair_stream": "M=Theta(B^2)",
            "complete_pair_threshold_for_B_collisions": "B=Omega(N^(1/3))",
            "complete_pair_work_at_threshold": "Omega(N^(2/3))",
            "conclusion": (
                "Under uniform residuals, a growing scalar-blind base cannot obtain "
                "B independent closure collisions below rho. A valid escape needs a "
                "public coordinate structure that concentrates independent collisions "
                "and locates them without materializing the pair graph."
            ),
            "exponent_rows": exponent_rows(),
        },
        "semantic_dedup": {
            "new_idea_created": False,
            "owner": "ECDLP-IDEA-195 constructive product-section frontier",
            "overlap": [
                "R68 fixed-sum information conservation",
                "ECDLP-IDEA-027 public Freiman chart",
                "ECDLP-IDEA-340 BSG energy source chart",
                "ECDLP-IDEA-389 Plunnecke source graph",
            ],
            "new_receipt": (
                "Exact separation of nullity-preserving constructive residual rows "
                "from rank-reducing closure collisions."
            ),
        },
        "source_bindings": {
            path: {"path": path, "sha256": digest}
            for path, digest in source_hashes.items()
        },
        "checks": checks,
        "admission": admission,
        "result": {
            "scalar_blind_toy_factor_logs_recovered": True,
            "fresh_target_log_recovered": descent["recovered"],
            "constructive_fresh_rows_add_log_information": False,
            "closure_collisions_add_log_information": True,
            "source_beats_direct_pair_enumeration": False,
            "source_beats_rho": False,
            "online_descent_beats_rho": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "next_action": (
            "Preregister one public coordinate-defined closure-collision locator "
            "that acts before pair materialization, then require superuniform "
            "independent collision rank, total source and descent below rho, and "
            "unchanged transfer across four generic-prime target families."
        ),
        "limits": [
            "The rank/nullity identity is general; the numerical replay is one size-103 toy.",
            "The collision count model is uniform and is not an unrestricted lower bound on structured coordinate bases.",
            "The toy source and target phases already exceed rho before omitted costs are charged.",
            "No Shoup-bound improvement or ECDLP breakthrough is claimed.",
        ],
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_constructive_closure_collision_gate_report_r69.json"
        ),
    )
    args = parser.parse_args()
    payload = run()
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rank_accounting = payload["toy_replay"]["rank_accounting"]
    print(
        f"output={args.output} proposals="
        f"{payload['toy_replay']['proposal_stream']['proposal_count']} "
        f"fresh={rank_accounting['fresh_residual_count']} "
        f"collisions={rank_accounting['closure_collision_count']} "
        f"collision_rank={rank_accounting['independent_closure_collision_count']} "
        f"final_nullity={rank_accounting['final_nullity']} "
        f"admitted={payload['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
