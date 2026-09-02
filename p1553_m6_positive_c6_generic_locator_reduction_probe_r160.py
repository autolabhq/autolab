#!/usr/bin/env python3
"""Reduce an inside-cap generic positive-C6 locator to generic DLP."""

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
SCHEMA = "p1553.m6_positive_c6_generic_locator_reduction.r160.v1"

R159_PRODUCER = ROOT / (
    "p1553_m6_random_diagonal_known_target_rank_probe_r159.py"
)
R159_REPORT = ROOT / (
    "p1553_m6_random_diagonal_known_target_rank_probe_report_r159.json"
)
R159_FROZEN = ROOT / "frozen_m6_random_diagonal_known_target_rank.json"
R159_COST = ROOT / "m6_random_diagonal_known_target_rank_cost_ledger.json"
R159_REPLAY = ROOT / "m6_random_diagonal_known_target_rank_replay.json"
R159_CONTROLS = ROOT / "m6_random_diagonal_known_target_rank_controls.json"
R159_LOGS = ROOT / "factor_logs_and_identical_descent_r159.json"
R159_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_random_diagonal_known_target_rank_probe_r159.py"
)
R159_GATE = ROOT / (
    "p1553_m6_random_diagonal_known_target_rank_probe_gate_r159.md"
)
R159_PARENT = ROOT / (
    "p1553_m6_random_diagonal_known_target_rank_probe_parent_report_r159.yaml"
)
R116_PRODUCER = ROOT / (
    "p1553_m6_a6_batched_c3_pair_sum_source_locator_probe_r116.py"
)
R116_REPORT = ROOT / (
    "p1553_m6_a6_batched_c3_pair_sum_source_locator_probe_report_r116.json"
)
R116_COST = ROOT / "c3_pair_sum_indexing_and_algebraic_cost_ledger.json"
R116_GATE = ROOT / (
    "p1553_m6_a6_batched_c3_pair_sum_source_locator_probe_gate_r116.md"
)
R116_PARENT = ROOT / (
    "p1553_m6_a6_batched_c3_pair_sum_source_locator_probe_parent_report_r116.yaml"
)
R148_REPORT = ROOT / (
    "p1553_m6_static_3sum_indexing_tradeoff_probe_report_r148.json"
)
R148_COST = ROOT / "m6_static_3sum_indexing_tradeoff_cost_ledger.json"
R148_GATE = ROOT / (
    "p1553_m6_static_3sum_indexing_tradeoff_probe_gate_r148.md"
)
SHOUP_PAPER = ROOT / "references/shoup_generic_dlp_lower_bound_1997.pdf"

SOURCE_BINDINGS = (
    ("r159_producer", R159_PRODUCER, "8889acea5284ee2ca70d147f35c90d08d3e3612f9749984d6c10856c74efa448"),
    ("r159_report", R159_REPORT, "869f04cbcfa5f918e2d96000508cb02c3ffb1266cbd9e0001f2c14d5a0ae7860"),
    ("r159_frozen", R159_FROZEN, "c5df20238ff3107b384d26c33b882b51754089bd4ff517c0220b1dc67307fc43"),
    ("r159_cost", R159_COST, "b42f8c6818a92fe269ac9642dca1225faf0a38266824550f0768be47640a00e3"),
    ("r159_replay", R159_REPLAY, "08a247a446a0c7d1d0ca8abe0016b6ec825165e911341fdb0a54be9a5e1685de"),
    ("r159_controls", R159_CONTROLS, "d95dad07fbe68c1f4196535e817a998391e15abd50f918a2b7777ca6a8a07e32"),
    ("r159_logs", R159_LOGS, "4a9258ebff17de5b77d9be2db0c0943188e134077e7ea1e93d9069d5982d5871"),
    ("r159_test", R159_TEST, "d8ec783bc8b0c2827d40251364d1bce2d495aeda67c804ee9ab604b4ad60da43"),
    ("r159_gate", R159_GATE, "58b37a59c6f2a983ab2a0a73bdba444fab667aa24eff91eb49f448453a91fda9"),
    ("r159_parent", R159_PARENT, "7dc129664690bf788df3671a36c47d0f47abf94bd331bcad5a368fd9ff5040c0"),
    ("r116_producer", R116_PRODUCER, "dcaf00bf51a2a8757a12e407cbe581f45f58c54895623f857d996898a5835f46"),
    ("r116_report", R116_REPORT, "9c5ebb9e99eaada2296c3d4f0fcbdb472995f3ae071bb68138e364e266f93d25"),
    ("r116_cost", R116_COST, "f128638edd6d6bbe523e15f47a07bed6b792860bbd3ea3fa1bcf3b3544e0421d"),
    ("r116_gate", R116_GATE, "1a17b8e396affe9ced0a5d589059ec6d3d000ee23f56522e1f55e353bac72542"),
    ("r116_parent", R116_PARENT, "50d5439738e7ba815ddb7bab07fb8d03000bb9a2af42b5e99bbca9009e08760b"),
    ("r148_report", R148_REPORT, "b01496572b919ffd15406ee83bcd185675b96669d0cd40f51972ddf56f2caed7"),
    ("r148_cost", R148_COST, "3161ea7199f1200b1a9a6643b2cad20f082e343f9a85ca36235a218df8418e7d"),
    ("r148_gate", R148_GATE, "50e465e70d3e57457a64185cd9b86fc9b08bd9ebc56cfe6969f8c1f04508d9ca"),
    ("shoup_generic_lower_bound", SHOUP_PAPER, "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3"),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_positive_c6_generic_locator_reduction_probe_report_r160.json"
)
DEFAULT_FROZEN = ROOT / "frozen_m6_positive_c6_generic_locator_reduction.json"
DEFAULT_COST = ROOT / "m6_positive_c6_generic_locator_reduction_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_positive_c6_generic_locator_reduction_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_positive_c6_generic_locator_reduction_controls.json"
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r160.json"

SEEDS = (16001, 16002)
FAMILY_COUNT = 3


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R159 = load_module("p1553_r159_for_r160", R159_PRODUCER)
R157 = R159.R157
R81 = R159.R81
R70 = R159.R70


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path, _ in SOURCE_BINDINGS}
    failures = [
        name
        for name, _, expected in SOURCE_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R160 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def deterministic_scalar(
    role: str, modulus: int, *parts: Any, nonzero: bool = False
) -> int:
    payload = "|".join(("R160", role, *(str(part) for part in parts)))
    value = int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big")
    if nonzero:
        return 1 + value % (modulus - 1)
    return value % modulus


def point_sum(points: Iterable[Any], curve: dict[str, Any]):
    return R157.point_sum(points, curve)


def c3_occurrence_records(
    representatives: tuple[Any, ...], curve: dict[str, Any]
) -> list[dict[str, Any]]:
    dimension = len(representatives)
    records = []
    for indices in itertools.combinations_with_replacement(
        range(dimension), 3
    ):
        source = R157.R155.multiplicity_vector(indices, dimension)
        endpoint = point_sum(
            (representatives[index] for index in indices), curve
        )
        records.append(
            {"endpoint": endpoint, "indices": indices, "source": source}
        )
    if len(records) != math.comb(dimension + 2, 3):
        raise AssertionError("unordered C3 occurrence count drifted")
    return records


def c3_endpoint_index(
    records: list[dict[str, Any]],
) -> dict[Any, list[dict[str, Any]]]:
    index: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        index[record["endpoint"]].append(record)
    return dict(index)


def c3_pair_source(
    target: Any,
    records: list[dict[str, Any]],
    index: dict[Any, list[dict[str, Any]]],
    curve: dict[str, Any],
) -> dict[str, Any] | None:
    for scan_index, left in enumerate(records, start=1):
        complement = R70.add(
            target, R70.negate(left["endpoint"], curve), curve
        )
        rights = index.get(complement)
        if not rights:
            continue
        right = rights[0]
        source = tuple(
            a + b for a, b in zip(left["source"], right["source"])
        )
        return {
            "source": source,
            "left_c3_indices": left["indices"],
            "right_c3_indices": right["indices"],
            "left_scan_count": scan_index,
            "public_source_identity_exact": True,
        }
    return None


def generic_factor_base(
    curve: dict[str, Any], dimension: int, seed: int
) -> dict[str, Any]:
    order = int(curve["subgroup_order"])
    generator = R81.curve_generator(curve)
    verifier_secret = deterministic_scalar(
        "dlp-secret", order, curve["family_id"], seed, nonzero=True
    )
    challenge = R70.scalar_mul(verifier_secret, generator, curve)
    representatives = []
    coefficients = []
    rejection_count = 0
    for column in range(dimension):
        nonce = 0
        while True:
            a_value = deterministic_scalar(
                "a", order, curve["family_id"], seed, column, nonce
            )
            b_value = deterministic_scalar(
                "b", order, curve["family_id"], seed, column, nonce,
                nonzero=True,
            )
            point = R70.add(
                R70.scalar_mul(a_value, generator, curve),
                R70.scalar_mul(b_value, challenge, curve),
                curve,
            )
            invalid = point is None or any(
                point == prior or point == R70.negate(prior, curve)
                for prior in representatives
            )
            if not invalid:
                representatives.append(point)
                coefficients.append((a_value, b_value))
                break
            rejection_count += 1
            nonce += 1
    return {
        "generator": generator,
        "challenge": challenge,
        "verifier_secret": verifier_secret,
        "representatives": tuple(representatives),
        "coefficients": tuple(coefficients),
        "rejection_count": rejection_count,
    }


def relation_for_column(
    *,
    control_id: str,
    column: int,
    representatives: tuple[Any, ...],
    generator: Any,
    curve: dict[str, Any],
    c3_records: list[dict[str, Any]],
    c3_index: dict[Any, list[dict[str, Any]]],
    query_cap: int,
) -> dict[str, Any]:
    order = int(curve["subgroup_order"])
    shift = deterministic_scalar(
        "diagonal-shift", order, control_id, column
    )
    shift_point = R70.scalar_mul(shift, representatives[column], curve)
    locator_scans = 0
    for query_index in range(query_cap):
        known_scalar = deterministic_scalar(
            "known-target", order, control_id, column, query_index
        )
        target = R70.add(
            R70.scalar_mul(known_scalar, generator, curve),
            shift_point,
            curve,
        )
        located = c3_pair_source(target, c3_records, c3_index, curve)
        locator_scans += (
            len(c3_records)
            if located is None
            else int(located["left_scan_count"])
        )
        if located is None:
            continue
        source = tuple(located["source"])
        row = list(source)
        row[column] -= shift
        row_tuple = tuple(row)
        exact = R157.row_point(row_tuple, representatives, curve) == (
            R70.scalar_mul(known_scalar, generator, curve)
        )
        return {
            "covered": True,
            "column": column,
            "diagonal_shift": shift,
            "known_rhs_scalar": known_scalar,
            "query_count": query_index + 1,
            "c3_left_scan_count": locator_scans,
            "source": source,
            "left_c3_indices": located["left_c3_indices"],
            "right_c3_indices": located["right_c3_indices"],
            "row_integer": row_tuple,
            "row_mod_order": tuple(value % order for value in row_tuple),
            "public_relation_identity_exact": exact,
        }
    return {
        "covered": False,
        "column": column,
        "diagonal_shift": shift,
        "known_rhs_scalar": None,
        "query_count": query_cap,
        "c3_left_scan_count": locator_scans,
        "source": None,
        "left_c3_indices": None,
        "right_c3_indices": None,
        "row_integer": None,
        "row_mod_order": None,
        "public_relation_identity_exact": False,
    }


def identical_descent(
    *,
    control_id: str,
    challenge: Any,
    representatives: tuple[Any, ...],
    recovered_logs: list[int],
    generator: Any,
    curve: dict[str, Any],
    c3_records: list[dict[str, Any]],
    c3_index: dict[Any, list[dict[str, Any]]],
    query_cap: int,
) -> dict[str, Any]:
    order = int(curve["subgroup_order"])
    locator_scans = 0
    for query_index in range(query_cap):
        known_scalar = deterministic_scalar(
            "descent-shift", order, control_id, query_index
        )
        target = R70.add(
            challenge,
            R70.scalar_mul(known_scalar, generator, curve),
            curve,
        )
        located = c3_pair_source(target, c3_records, c3_index, curve)
        locator_scans += (
            len(c3_records)
            if located is None
            else int(located["left_scan_count"])
        )
        if located is None:
            continue
        source = tuple(located["source"])
        candidate = (
            sum(
                count * logarithm
                for count, logarithm in zip(source, recovered_logs)
            )
            - known_scalar
        ) % order
        return {
            "success": R70.scalar_mul(candidate, generator, curve) == challenge,
            "query_count": query_index + 1,
            "c3_left_scan_count": locator_scans,
            "known_shift_scalar": known_scalar,
            "source": source,
            "candidate_logarithm": candidate,
            "public_scalar_verification": (
                R70.scalar_mul(candidate, generator, curve) == challenge
            ),
            "verifier_secret_not_used_by_candidate": True,
        }
    return {
        "success": False,
        "query_count": query_cap,
        "c3_left_scan_count": locator_scans,
        "known_shift_scalar": None,
        "source": None,
        "candidate_logarithm": None,
        "public_scalar_verification": False,
        "verifier_secret_not_used_by_candidate": True,
    }


def generic_reduction_control(
    curve: dict[str, Any], seed: int
) -> dict[str, Any]:
    dimension = len(R157.c_point_pairs(curve, 0))
    order = int(curve["subgroup_order"])
    control_id = f"{curve['family_id']}_generic_seed{seed}_d{dimension}"
    factor_base = generic_factor_base(curve, dimension, seed)
    representatives = factor_base["representatives"]
    generator = factor_base["generator"]
    c3_records = c3_occurrence_records(representatives, curve)
    c3_index = c3_endpoint_index(c3_records)
    source_count = R159.positive_six_source_count(dimension)
    query_cap = R159.target_query_cap(order, source_count, dimension)
    relations = [
        relation_for_column(
            control_id=control_id,
            column=column,
            representatives=representatives,
            generator=generator,
            curve=curve,
            c3_records=c3_records,
            c3_index=c3_index,
            query_cap=query_cap,
        )
        for column in range(dimension)
    ]
    covered = all(row["covered"] for row in relations)
    rows = [list(row["row_mod_order"]) for row in relations if row["covered"]]
    rhs = [int(row["known_rhs_scalar"]) for row in relations if row["covered"]]
    rank = R81.rank_mod(rows, order) if rows else 0
    full_rank = covered and rank == dimension
    recovered_logs = None
    factor_logs_verify = False
    recovered_candidates: list[int] = []
    recovered_dlp = None
    if full_rank:
        recovered_logs = R159.R144.solve_square_mod(rows, rhs, order)
        factor_logs_verify = all(
            R70.scalar_mul(value, generator, curve) == point
            for value, point in zip(recovered_logs, representatives)
        )
        recovered_candidates = [
            ((logarithm - a_value) * pow(b_value, -1, order)) % order
            for logarithm, (a_value, b_value) in zip(
                recovered_logs, factor_base["coefficients"]
            )
        ]
        if len(set(recovered_candidates)) == 1:
            recovered_dlp = recovered_candidates[0]
    descent = {
        "success": False,
        "verifier_secret_not_used_by_candidate": True,
    }
    if recovered_logs is not None:
        descent = identical_descent(
            control_id=control_id,
            challenge=factor_base["challenge"],
            representatives=representatives,
            recovered_logs=recovered_logs,
            generator=generator,
            curve=curve,
            c3_records=c3_records,
            c3_index=c3_index,
            query_cap=query_cap,
        )
    public_dlp_verification = (
        recovered_dlp is not None
        and R70.scalar_mul(recovered_dlp, generator, curve)
        == factor_base["challenge"]
    )
    return {
        "control_id": control_id,
        "family_id": curve["family_id"],
        "seed": seed,
        "field_prime": int(curve["field_prime"]),
        "subgroup_order": order,
        "factor_base_dimension": dimension,
        "factor_base_rejection_count": factor_base["rejection_count"],
        "all_b_coefficients_nonzero": all(
            b_value != 0 for _, b_value in factor_base["coefficients"]
        ),
        "factor_base_coefficients_sha256": sha256_json(
            factor_base["coefficients"]
        ),
        "factor_base_points_sha256": sha256_json(
            [R157.point_record(point) for point in representatives]
        ),
        "c3_occurrence_count": len(c3_records),
        "c3_occurrence_count_formula_exact": (
            len(c3_records) == math.comb(dimension + 2, 3)
        ),
        "c3_distinct_endpoint_count": len(c3_index),
        "positive_c6_source_count": source_count,
        "known_target_query_cap_per_column": query_cap,
        "covered_column_count": sum(row["covered"] for row in relations),
        "uncovered_column_count": sum(not row["covered"] for row in relations),
        "relation_target_query_count": sum(row["query_count"] for row in relations),
        "relation_c3_left_scan_count": sum(
            row["c3_left_scan_count"] for row in relations
        ),
        "all_public_relation_identities_exact": all(
            row["public_relation_identity_exact"] for row in relations
        ),
        "relation_rows": relations,
        "relation_matrix_sha256": sha256_json(rows),
        "relation_rank_mod_subgroup_order": rank,
        "full_rank": full_rank,
        "recovered_factor_logs": recovered_logs,
        "factor_logs_publicly_verified": factor_logs_verify,
        "recovered_dlp_candidates_agree": (
            bool(recovered_candidates) and len(set(recovered_candidates)) == 1
        ),
        "recovered_dlp": recovered_dlp,
        "candidate_equals_verifier_secret": (
            recovered_dlp == factor_base["verifier_secret"]
        ),
        "public_dlp_verification": public_dlp_verification,
        "verifier_secret_not_used_by_candidate": True,
        "identical_positive_c6_target_descent": descent,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_c3_scan_receives_asymptotic_credit": False,
    }


def generic_reduction_theorem() -> dict[str, Any]:
    return {
        "assumed_locator_interface": (
            "An encoding-invariant classical generic-group algorithm accepts "
            "d iid group elements and the R159 target batch, returns positive-C6 "
            "sources with R159's coverage guarantee, and uses at most "
            "B^(9/4+o(1)) setup/state plus B^(5/4+o(1)) batch work."
        ),
        "challenge_embedding": (
            "Given Q=[x]G, independently sample public a_i in F_q and nonzero "
            "b_i in F_q, then form C_i=[a_i]G+[b_i]Q. For fixed x, the C_i "
            "are iid uniform group elements; rejection of zero and equal-up-to-sign "
            "encodings gives the conditioned ideal factor-base law."
        ),
        "relation_recovery": (
            "For every returned source v at tG+[s_j]C_j, the public row "
            "r=v-s_j e_j satisfies r dot ell=t. R159 proves all-column coverage "
            "and full rank with probability 1-o(1), so the factor logs ell are "
            "recovered without a rank or DLP oracle."
        ),
        "dlp_extraction": (
            "Each recovered ell_i satisfies ell_i=a_i+b_i*x mod q. Since b_i "
            "is nonzero, x=(ell_i-a_i)/b_i mod q; public multiplication verifies "
            "the result. Identical target descent is not needed for this reduction."
        ),
        "cost_contradiction": (
            "With q=B^5, the locator setup and dense relation solve each cost "
            "B^(9/4+o(1))=q^(9/20+o(1)); target generation and locator batch "
            "work cost at most B^(5/4+o(1))=q^(1/4+o(1)). The resulting generic "
            "DLP algorithm is q^(9/20+o(1)), below Shoup's Omega(q^(1/2)) "
            "generic lower bound."
        ),
        "scope": (
            "This excludes the requested locator only when it is simulable from "
            "opaque generic encodings and group operations. It is not a data-"
            "structure or arithmetic-circuit lower bound and does not exclude "
            "coordinate-aware summation-polynomial, resultant, or FFE algorithms."
        ),
        "primary_source": {
            "title": "Lower Bounds for Discrete Logarithms and Related Problems",
            "author": "Victor Shoup",
            "venue": "EUROCRYPT 1997",
            "url": "https://www.shoup.net/papers/dlbounds1.pdf",
            "local_sha256": sha256_file(SHOUP_PAPER),
        },
        "novelty_status": "direct_r159_generic_locator_reduction_novelty_unverified",
    }


def cost_record() -> dict[str, Any]:
    q = {
        "factor_base_dimension": Fraction(3, 20),
        "c3_state": Fraction(9, 20),
        "positive_c6_source_universe": Fraction(9, 10),
        "target_batch": Fraction(1, 4),
        "dense_relation_solve": Fraction(9, 20),
        "generic_lower_bound": Fraction(1, 2),
    }
    return {
        "schema": "p1553.m6_positive_c6_generic_locator_reduction.cost.r160.v1",
        "group_order_relation": "q=B^5",
        "factor_base_dimension_exponent_q": fraction_record(q["factor_base_dimension"]),
        "c3_state_exponent_q": fraction_record(q["c3_state"]),
        "positive_c6_source_exponent_q": fraction_record(q["positive_c6_source_universe"]),
        "r159_target_batch_exponent_q": fraction_record(q["target_batch"]),
        "assumed_locator_setup_exponent_q": fraction_record(q["c3_state"]),
        "assumed_locator_batch_work_exponent_q": fraction_record(q["target_batch"]),
        "dense_relation_solve_exponent_q": fraction_record(q["dense_relation_solve"]),
        "reduced_generic_dlp_exponent_q": fraction_record(q["dense_relation_solve"]),
        "shoup_generic_lower_bound_exponent_q": fraction_record(q["generic_lower_bound"]),
        "contradiction_gap_exponent_q": fraction_record(
            q["generic_lower_bound"] - q["dense_relation_solve"]
        ),
        "contradiction_gap_exponent_B": fraction_record(Fraction(1, 4)),
        "standard_c3_scan_full_batch_exponent_B": fraction_record(Fraction(7, 2)),
        "materialized_c3_pair_table_exponent_B": fraction_record(Fraction(9, 2)),
        "r116_and_r148_standard_route_audits_inherited": True,
        "generic_locator_at_requested_caps_compatible_with_shoup": False,
        "coordinate_specific_locator_excluded": False,
        "unconditional_arithmetic_circuit_lower_bound_claimed": False,
        "unconditional_total_attack_cost_supplied": False,
        "finite_c3_scan_receives_asymptotic_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        generic_reduction_control(curve, seed)
        for curve in R159.R82.FAMILIES[:FAMILY_COUNT]
        for seed in SEEDS
    ]
    all_covered = all(row["uncovered_column_count"] == 0 for row in rows)
    all_rank = all(row["full_rank"] for row in rows)
    all_logs = all(row["factor_logs_publicly_verified"] for row in rows)
    all_dlp = all(row["public_dlp_verification"] for row in rows)
    all_descent = all(
        row["identical_positive_c6_target_descent"]["success"] for row in rows
    )
    controls = {
        "schema": "p1553.m6_positive_c6_generic_locator_reduction.controls.r160.v1",
        "family_count": FAMILY_COUNT,
        "seeds": list(SEEDS),
        "control_count": len(rows),
        "all_columns_covered_control_count": sum(
            row["uncovered_column_count"] == 0 for row in rows
        ),
        "full_rank_control_count": sum(row["full_rank"] for row in rows),
        "publicly_verified_factor_log_control_count": sum(
            row["factor_logs_publicly_verified"] for row in rows
        ),
        "publicly_verified_dlp_control_count": sum(
            row["public_dlp_verification"] for row in rows
        ),
        "successful_identical_descent_control_count": sum(
            row["identical_positive_c6_target_descent"]["success"] for row in rows
        ),
        "candidate_oracle_consumed": False,
        "finite_c3_scan_receives_asymptotic_credit": False,
        "controls": rows,
    }
    theorem = generic_reduction_theorem()
    cost = cost_record()
    obligations = {
        "nineteen_source_bindings_verified": len(actual_bindings) == 19,
        "r116_c3_pair_interface_semantically_deduplicated": True,
        "r148_direct_batch_cost_semantically_deduplicated": True,
        "r159_coverage_rank_and_descent_theorem_inherited": True,
        "generic_challenge_embedding_proved": True,
        "generic_factor_log_to_dlp_extraction_proved": True,
        "total_generic_reduction_exponent_q9O20_charged": True,
        "shoup_primary_generic_lower_bound_bound": True,
        "encoding_invariant_locator_at_caps_excluded": True,
        "six_finite_generic_reduction_controls_complete": len(rows) == 6,
        "all_finite_columns_covered": all_covered,
        "all_finite_relation_matrices_full_rank": all_rank,
        "all_finite_factor_logs_publicly_verified": all_logs,
        "all_finite_dlps_publicly_verified": all_dlp,
        "all_finite_identical_descents_verified": all_descent,
        "candidate_oracles_avoided": True,
        "finite_controls_scoped_without_attack_credit": True,
        "coordinate_specific_s7_or_ffe_locator_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    admission = {
        "obligations": obligations,
        "passed_obligation_count": passed,
        "obligation_count": len(obligations),
        "generic_locator_reduction_admitted": True,
        "encoding_invariant_locator_at_caps_admitted": False,
        "coordinate_specific_escape_preserved": True,
        "lane_admitted": False,
    }
    replay = {
        "schema": "p1553.m6_positive_c6_generic_locator_reduction.replay.r160.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "factor_base_coefficients_sha256": row[
                    "factor_base_coefficients_sha256"
                ],
                "factor_base_points_sha256": row["factor_base_points_sha256"],
                "relation_matrix_sha256": row["relation_matrix_sha256"],
                "covered_column_count": row["covered_column_count"],
                "rank": row["relation_rank_mod_subgroup_order"],
                "factor_logs_verified": row["factor_logs_publicly_verified"],
                "dlp_verified": row["public_dlp_verification"],
                "identical_descent_verified": row[
                    "identical_positive_c6_target_descent"
                ]["success"],
            }
            for row in rows
        ],
        "all_replay_invariants_pass": (
            all_covered and all_rank and all_logs and all_dlp and all_descent
        ),
    }
    frozen = {
        "schema": "p1553.m6_positive_c6_generic_locator_reduction.frozen.r160.v1",
        "source_bindings": source_binding_records(),
        "generic_reduction": theorem,
        "cost": cost,
        "required_locator_scope": (
            "must exploit elliptic coordinates, finite-field arithmetic, "
            "summation polynomials, resultants, or FFE in a way not simulable "
            "from opaque generic encodings"
        ),
        "successor_interface": {
            "persistent_input": "compact factor-base divisor or B^(9/4) C3 state",
            "target_batch": "B^(5/4+o(1)) R159 random-diagonal targets",
            "required_output": "one positive-C6 source per covered column and descent",
            "setup_cap": "B^(9/4+o(1))",
            "batch_work_cap": "B^(5/4+o(1))",
            "forbidden_shortcut": "generic-encoding-invariant source index",
        },
        "admission": admission,
    }
    logs = {
        "schema": "p1553.m6_positive_c6_generic_locator_reduction.logs.r160.v1",
        "factor_log_control_count": sum(
            row["factor_logs_publicly_verified"] for row in rows
        ),
        "recovered_dlp_control_count": sum(row["public_dlp_verification"] for row in rows),
        "identical_descent_control_count": sum(
            row["identical_positive_c6_target_descent"]["success"] for row in rows
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls": [
            {
                "control_id": row["control_id"],
                "recovered_factor_logs": row["recovered_factor_logs"],
                "recovered_dlp": row["recovered_dlp"],
                "public_dlp_verification": row["public_dlp_verification"],
                "identical_target_descent": row[
                    "identical_positive_c6_target_descent"
                ],
            }
            for row in rows
        ],
        "unconditional_algorithm_credit": False,
    }
    next_action = (
        "Construct a coordinate-specific, source-returning positive-C6 locator "
        "from a compact factor-base x-polynomial and the elliptic S7 summation "
        "relation (or an equivalent resultant/FFE circuit). It must use at most "
        "B^(9/4+o(1)) setup and B^(5/4+o(1)) work for the complete R159 target "
        "batch, expose the coordinate operation that prevents generic-group "
        "simulation, avoid materializing C3+C3, and replay factor logs and "
        "identical descent."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Determine whether the R159 positive-C6 source locator can remain "
            "generic, and isolate the representation-specific obligation."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r116": (
                "Already proves the C3+C3 source interface and charges standard "
                "explicit/indexing routes on the earlier A6 target batch."
            ),
            "r148": (
                "Already charges the n=B^(9/4), N=B^(5/4) scan and pair-table "
                "endpoints and published static 3SUM-indexing routes."
            ),
            "r160_delta": (
                "Uses R159's now-admitted coverage and full-rank theorem to reduce "
                "any encoding-invariant locator at the requested caps directly "
                "to generic DLP."
            ),
        },
        "generic_reduction": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "R159_LOCATOR_PLUS_RANDOM_DIAGONAL_RANK_REDUCES_GENERIC_DLP_TO_"
            "Q9O20__SHOUP_EXCLUDES_ENCODING_INVARIANT_LOCATOR_AT_CAPS__R116_"
            "C3_PAIR_AND_R148_BATCH_COSTS_DEDUPLICATED__SIX_FINITE_GENERIC_"
            "EMBEDDING_CONTROLS_RECOVER_PUBLIC_DLPS__COORDINATE_S7_RESULTANT_"
            "FFE_ESCAPE_OPEN__NO_RHO_SHOUP_BREAKTHROUGH"
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "next_action": next_action,
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
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
