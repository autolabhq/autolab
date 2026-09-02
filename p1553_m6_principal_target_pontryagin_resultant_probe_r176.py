#!/usr/bin/env python3
"""Test principal-target incidence on selected Pontryagin product cycles."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_principal_target_pontryagin_resultant.r176.v1"

R175_PRODUCER = ROOT / "p1553_m6_scalar_subset_incidence_group_testing_probe_r175.py"
R175_REPORT = ROOT / "p1553_m6_scalar_subset_incidence_group_testing_probe_report_r175.json"
R175_FROZEN = ROOT / "frozen_m6_scalar_subset_incidence_group_testing.json"
R175_COST = ROOT / "m6_scalar_subset_incidence_group_testing_cost_ledger.json"
R175_REPLAY = ROOT / "m6_scalar_subset_incidence_group_testing_replay.json"
R175_CONTROLS = ROOT / "m6_scalar_subset_incidence_group_testing_controls.json"
R175_TREE = ROOT / "scalar_subset_incidence_group_testing_r175.json"
R175_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_scalar_subset_incidence_group_testing_probe_r175.py"
R175_GATE = ROOT / "p1553_m6_scalar_subset_incidence_group_testing_probe_gate_r175.md"
R175_PARENT = ROOT / "p1553_m6_scalar_subset_incidence_group_testing_probe_parent_report_r175.yaml"

R167_PRODUCER = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_r167.py"
R167_REPORT = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_report_r167.json"
R167_FROZEN = ROOT / "frozen_m6_generalized_target_divisor_weil_reciprocity_swap.json"
R167_COST = ROOT / "m6_generalized_target_divisor_weil_reciprocity_swap_cost_ledger.json"
R167_REPLAY = ROOT / "m6_generalized_target_divisor_weil_reciprocity_swap_replay.json"
R167_CONTROLS = ROOT / "m6_generalized_target_divisor_weil_reciprocity_swap_controls.json"
R167_RESULTANT = ROOT / "generalized_miller_elliptic_resultant_swap_r167.json"
R167_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_r167.py"
R167_GATE = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_gate_r167.md"
R167_PARENT = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_parent_report_r167.yaml"

EAGEN_PAPER = ROOT / "references/eagen_ecip_weil_reciprocity_2022_596.pdf"
MILLER_PAPER = ROOT / "references/miller_weil_pairing_algorithm_1986.pdf"
R169_REPORT = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_report_r169.json"
R169_GATE = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_gate_r169.md"

SOURCE_BINDINGS = (
    ("r175_producer", R175_PRODUCER, "f525ecc47496485919b4f2397b505842eac43ea7c1b53b90c6beaff4fa07d536"),
    ("r175_report", R175_REPORT, "5541b18b9ce670ca95902e75cf9fc14c2a35f9725c9ee756c0502963cbafc543"),
    ("r175_frozen", R175_FROZEN, "1710f98bdb7d29d4c016abb0272915d6fdcdf471d35a6e7f79941b22cc1be7ad"),
    ("r175_cost", R175_COST, "0c8790183f1d05593308421e1cce0eabec21c3ee52fef2ed3c4f1dc84f0e5818"),
    ("r175_replay", R175_REPLAY, "6eaf07b4a9507ca96681c85f79b8d33763033d7bd71faae5e8417987969f084f"),
    ("r175_controls", R175_CONTROLS, "1ff598263a892c4a8d87c04799b66383b671ffca9ecdb05c005c45219447fb2d"),
    ("r175_tree", R175_TREE, "cff099668231e6b239bc1a651492938affc09c8531e61489b7384dff48258c00"),
    ("r175_test", R175_TEST, "9cab6bb5fe9baa27d1d16e2874a3ceb782d0b93f799ce04befbd2cfb8472704f"),
    ("r175_gate", R175_GATE, "ccad60c12a8951476915c864d249135a369d0832a0a29704ef1cbd445739d3a1"),
    ("r175_parent", R175_PARENT, "e46cbd385ba78e5e14fd10dd70414b3a990a94493b8d32064258c362e56399c8"),
    ("r167_producer", R167_PRODUCER, "ba57052082668daf38027b928d88b0eca210510e2f6a6784ce3e0e8b481e1cd3"),
    ("r167_report", R167_REPORT, "1d2f009525ef9d54a0f538d3a0a8cefe8451d4d97933b91b31ed1bb5c0b47e3c"),
    ("r167_frozen", R167_FROZEN, "47d3d39017faa9adc1e79abc939c658c51272097dcb8620e72db51ce7543f492"),
    ("r167_cost", R167_COST, "a84dc45c92d9a17ef9e9b73ddad426a8fe5111e72b6cd974fd5968841ebd10c3"),
    ("r167_replay", R167_REPLAY, "cf983cfdb3428503634d84a5a69305b71ac70e44338c37acb24e646e5593f6d0"),
    ("r167_controls", R167_CONTROLS, "f78db8e5e5195727087552c8d8a122037eb1f0a0c5d469594ef08a070c093f83"),
    ("r167_resultant", R167_RESULTANT, "95a42cb9d499012b676d96a77ca7c8a066e0cc6ade79e6cf9d0dd9ccfb674247"),
    ("r167_test", R167_TEST, "8c098e9a820ea7967cc700544aa64cc419750806c67e7c3dca395f271ae505e9"),
    ("r167_gate", R167_GATE, "41c4869bb231dfa2d0de1bb0d280aa34bd95903ed425792b80a84c762b845e3b"),
    ("r167_parent", R167_PARENT, "0952a627b3bd5acd95e622533627f54b339605f92e9187be68d72431e7931db7"),
    ("eagen_2022_596", EAGEN_PAPER, "5310b35d288a9462ff704eb77e7651d18f681a5b560cfb2919d1cfd0e01ae09e"),
    ("miller_1986", MILLER_PAPER, "39c76c7643278b87b3d8c24b9a07d0b4cbfb561cd13735548990848e0f0bd166"),
    ("r169_report", R169_REPORT, "6f9123923396d0c6478486c9a669cb73ff759db00fa8997954f5b4a35d11ce86"),
    ("r169_gate", R169_GATE, "7aba9b4c16a35dc7db1f1ad5953ef50e597a882644260c028d8b6d7a412d185f"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_principal_target_pontryagin_resultant_probe_report_r176.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_principal_target_pontryagin_resultant.json"
DEFAULT_COST = ROOT / "m6_principal_target_pontryagin_resultant_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_principal_target_pontryagin_resultant_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_principal_target_pontryagin_resultant_controls.json"
DEFAULT_RESULTANT = ROOT / "principal_target_pontryagin_resultant_r176.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R175 = load_module("p1553_r175_for_r176", R175_PRODUCER)
R167 = load_module("p1553_r167_for_r176", R167_PRODUCER)
R161 = R175.R161


def sha256_file(path: Path) -> str:
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
        name for name, _, expected in SOURCE_BINDINGS if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R176 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def modular_product(values: list[int], prime: int) -> int:
    result = 1
    for value in values:
        result = result * value % prime
    return result


def point_record(point: tuple[int, int] | None) -> list[int] | None:
    return None if point is None else [int(point[0]), int(point[1])]


def scalar_point(
    scalar: int,
    point: tuple[int, int] | None,
    curve: dict[str, Any],
) -> tuple[int, int] | None:
    if point is None:
        return None
    result = R167.R161.R70.scalar_mul(scalar, point, curve)
    return None if result is None else tuple(result)


def r175_control(family_id: str, seed: int) -> dict[str, Any]:
    report = json.loads(R175_REPORT.read_text())
    control_id = f"{family_id}_scalar_subset_tree_seed{seed}"
    matches = [
        row
        for row in report["controls"]["controls"]
        if row["control_id"] == control_id
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one R175 control for {control_id}")
    return matches[0]


MillerState = tuple[tuple[int, int] | None, list[int]]


def miller_merge(
    left: MillerState,
    right: MillerState,
    probes: list[tuple[int, int]],
    curve: dict[str, Any],
) -> MillerState:
    left_sum, left_values = left
    right_sum, right_values = right
    prime = int(curve["field_prime"])
    if left_sum is None:
        return right_sum, [
            a_value * b_value % prime
            for a_value, b_value in zip(left_values, right_values)
        ]
    if right_sum is None:
        return left_sum, [
            a_value * b_value % prime
            for a_value, b_value in zip(left_values, right_values)
        ]

    merged_sum = R167.point_add(left_sum, right_sum, curve)
    if merged_sum is None:
        return None, [
            a_value * b_value * (probe[0] - left_sum[0]) % prime
            for probe, a_value, b_value in zip(
                probes, left_values, right_values
            )
        ]

    if left_sum == right_sum:
        slope = (
            (3 * left_sum[0] * left_sum[0] + int(curve["curve_a"]))
            * pow(2 * left_sum[1], -1, prime)
            % prime
        )
    else:
        slope = (
            (right_sum[1] - left_sum[1])
            * pow((right_sum[0] - left_sum[0]) % prime, -1, prime)
            % prime
        )

    values: list[int] = []
    for probe, a_value, b_value in zip(probes, left_values, right_values):
        line = (
            probe[1]
            - left_sum[1]
            - slope * (probe[0] - left_sum[0])
        ) % prime
        vertical = (probe[0] - merged_sum[0]) % prime
        if vertical == 0:
            raise ZeroDivisionError("intermediate Miller vertical meets probe")
        values.append(
            a_value * b_value * line * pow(vertical, -1, prime) % prime
        )
    return merged_sum, values


def miller_principal_values(
    zero_points: list[tuple[int, int] | None],
    probes: list[tuple[int, int]],
    curve: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    for attempt in range(64):
        payload = f"R176|{label}|{attempt}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        shuffled = list(zero_points)
        random.Random(seed).shuffle(shuffled)
        states: list[MillerState] = [
            (point, [1] * len(probes)) for point in shuffled
        ]
        try:
            while len(states) > 1:
                states = [
                    states[index]
                    if index + 1 == len(states)
                    else miller_merge(
                        states[index], states[index + 1], probes, curve
                    )
                    for index in range(0, len(states), 2)
                ]
        except (ValueError, ZeroDivisionError):
            continue
        if states[0][0] is not None:
            raise AssertionError("completed Miller zero cycle does not sum to O")
        return {
            "values": states[0][1],
            "shuffle_attempt": attempt,
            "merge_count": len(zero_points) - 1,
        }
    raise AssertionError("no probe-safe deterministic Miller merge order found")


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    r167 = R167.finite_control(curve, seed)
    r175 = r175_control(curve["family_id"], seed)
    _, divisor, _ = R175.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    prime = int(curve["field_prime"])
    n = len(selected)
    common_zero = tuple(r167["common_zero"])
    target_points = [
        tuple(point)
        for point in r167["numerator_witness"]["prescribed_points"]
        if tuple(point) != common_zero
    ]
    pole_points = [
        tuple(point)
        for point in r167["denominator_witness"]["prescribed_points"]
        if tuple(point) != common_zero
    ]
    if len(target_points) != len(pole_points):
        raise AssertionError("principal target witness lost degree zero")
    probes = target_points + pole_points

    pair_points: list[list[tuple[int, int] | None]] = []
    pair_values: list[list[int]] = []
    pair_rows: list[dict[str, Any]] = []
    principal_leaf_values: list[int] = []
    denominator_unit_count = 0
    diagonal_zero_count = 0
    off_diagonal_zero_count = 0
    for left_index, left in enumerate(selected):
        row_points: list[tuple[int, int] | None] = []
        row_values: list[int] = []
        for right_index, right in enumerate(selected):
            pair_sum = R167.point_add(left, right, curve)
            numerator_value = R167.witness_value(
                r167["numerator_witness"], pair_sum, prime
            )
            denominator_value = R167.witness_value(
                r167["denominator_witness"], pair_sum, prime
            )
            if denominator_value == 0:
                raise AssertionError("principal witness pole meets selected pair sum")
            denominator_unit_count += 1
            value = numerator_value * pow(denominator_value, -1, prime) % prime
            diagonal_zero_count += int(left_index == right_index and value == 0)
            off_diagonal_zero_count += int(left_index != right_index and value == 0)
            row_points.append(pair_sum)
            row_values.append(value)
            pair_rows.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "pair_sum": point_record(pair_sum),
                    "numerator_value": numerator_value,
                    "denominator_value": denominator_value,
                    "principal_value": value,
                    "candidate_incidence": value == 0,
                }
            )
        pair_points.append(row_points)
        pair_values.append(row_values)
        principal_leaf_values.append(modular_product(row_values, prime))

    principal_roots = sorted(
        int(point[0])
        for point, value in zip(selected, principal_leaf_values)
        if value == 0
    )
    if principal_roots != r175["r174_candidate_roots"]:
        raise AssertionError("principal-target roots differ from R175")
    if principal_roots != r167["candidate_roots"]:
        raise AssertionError("principal-target roots differ from R167 controls")

    tree = R175.recover_zero_leaves(
        selected,
        divisor["v"],
        principal_leaf_values,
        principal_roots,
        prime,
    )
    r175_zero_projection = [
        {
            "path": row["path"],
            "start": row["start"],
            "end": row["end"],
            "zero": row["zero"],
            "candidate_count_in_subset": row["candidate_count_in_subset"],
        }
        for row in r175["tree"]["queries"]
    ]
    principal_zero_projection = [
        {
            "path": row["path"],
            "start": row["start"],
            "end": row["end"],
            "zero": row["zero"],
            "candidate_count_in_subset": row["candidate_count_in_subset"],
        }
        for row in tree["queries"]
    ]
    if principal_zero_projection != r175_zero_projection:
        raise AssertionError("principal tree zero pattern differs from R175")

    selected_sum = R167.point_sum(selected, curve)
    h_at_infinity = R167.rational_value(
        r167["numerator_witness"],
        r167["denominator_witness"],
        None,
        prime,
    )
    if h_at_infinity == 0:
        raise AssertionError("principal target witness vanishes at infinity")

    reciprocity_rows: list[dict[str, Any]] = []
    all_cycle_sum_identities_exact = True
    all_reciprocity_identities_exact = True
    all_completion_values_units = True
    all_pole_products_units = True
    for query in tree["queries"]:
        start = int(query["start"])
        end = int(query["end"])
        subset_size = end - start
        cycle_points = [
            point
            for row in pair_points[start:end]
            for point in row
        ]
        cycle_values = [
            value
            for row in pair_values[start:end]
            for value in row
        ]
        pair_cycle_degree = subset_size * n
        if len(cycle_points) != pair_cycle_degree:
            raise AssertionError("Pontryagin cycle degree mismatch")
        direct_value = modular_product(cycle_values, prime)
        if direct_value != int(query["scalar_product"]):
            raise AssertionError("tree scalar is not the principal cycle value")

        cycle_sum = R167.point_sum(cycle_points, curve)
        subset_sum = R167.point_sum(selected[start:end], curve)
        formula_sum = R167.point_add(
            scalar_point(n, subset_sum, curve),
            scalar_point(subset_size, selected_sum, curve),
            curve,
        )
        cycle_sum_exact = cycle_sum == formula_sum
        all_cycle_sum_identities_exact &= cycle_sum_exact
        if not cycle_sum_exact:
            raise AssertionError("Pontryagin cycle group-sum identity failed")

        completion = R167.point_negate(cycle_sum, curve)
        if completion is None:
            miller_zeros = cycle_points
            pole_order = pair_cycle_degree
            completion_value = 1
        else:
            miller_zeros = cycle_points + [completion]
            pole_order = pair_cycle_degree + 1
            completion_value = R167.rational_value(
                r167["numerator_witness"],
                r167["denominator_witness"],
                completion,
                prime,
            )
        all_completion_values_units &= completion_value != 0
        if completion_value == 0:
            raise AssertionError("principal completion is not a target-witness unit")

        miller = miller_principal_values(
            miller_zeros,
            probes,
            curve,
            f"{r175['control_id']}|{query['path']}",
        )
        values = miller["values"]
        target_product = modular_product(
            values[: len(target_points)], prime
        )
        pole_product = modular_product(values[len(target_points) :], prime)
        all_pole_products_units &= pole_product != 0
        if pole_product == 0:
            raise AssertionError("Miller completion vanishes on an auxiliary pole")
        reciprocity_value = (
            target_product
            * pow(pole_product, -1, prime)
            * pow(h_at_infinity, pole_order, prime)
            * pow(completion_value, -1, prime)
            % prime
        )
        exact = direct_value == reciprocity_value
        all_reciprocity_identities_exact &= exact
        if not exact:
            raise AssertionError("principal Pontryagin reciprocity failed")
        if (target_product == 0) != (direct_value == 0):
            raise AssertionError("reciprocal target product lost candidate zeros")
        reciprocity_rows.append(
            {
                "path": query["path"],
                "start": start,
                "end": end,
                "subset_size": subset_size,
                "pair_cycle_degree": pair_cycle_degree,
                "cycle_sum": point_record(cycle_sum),
                "cycle_sum_formula_exact": cycle_sum_exact,
                "completion": point_record(completion),
                "completion_value": completion_value,
                "miller_pole_order": pole_order,
                "miller_merge_count": miller["merge_count"],
                "miller_shuffle_attempt": miller["shuffle_attempt"],
                "direct_principal_value": direct_value,
                "target_product": target_product,
                "pole_product": pole_product,
                "reciprocity_value": reciprocity_value,
                "identity_exact": exact,
                "literal_disjoint_support_identity": direct_value != 0,
                "candidate_specialized_zero_identity": direct_value == 0,
            }
        )

    return {
        "control_id": f"{curve['family_id']}_principal_pontryagin_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "c3_divisor_degree": n,
        "target_witness_degree": len(target_points),
        "principal_target_divisor_degree_zero": True,
        "selected_pair_evaluation_count": n * n,
        "selected_pair_denominator_unit_count": denominator_unit_count,
        "principal_pair_incidence_count": diagonal_zero_count
        + off_diagonal_zero_count,
        "diagonal_pair_incidence_count": diagonal_zero_count,
        "off_diagonal_pair_incidence_count": off_diagonal_zero_count,
        "principal_leaf_roots": principal_roots,
        "r175_candidate_roots": r175["r174_candidate_roots"],
        "r167_candidate_roots": r167["candidate_roots"],
        "principal_leaf_roots_match_r175_and_r167": True,
        "principal_leaf_value_sha256": sha256_json(principal_leaf_values),
        "pair_transcript_sha256": sha256_json(pair_rows),
        "principal_tree_query_count": tree["query_count"],
        "principal_tree_zero_projection_sha256": sha256_json(
            principal_zero_projection
        ),
        "r175_tree_zero_projection_sha256": sha256_json(r175_zero_projection),
        "principal_tree_zero_pattern_matches_r175": True,
        "all_pair_cycle_group_sum_identities_exact": all_cycle_sum_identities_exact,
        "all_completion_values_units": all_completion_values_units,
        "all_miller_pole_products_units": all_pole_products_units,
        "all_principal_pontryagin_reciprocity_identities_exact": all_reciprocity_identities_exact,
        "reciprocity_identity_count": len(reciprocity_rows),
        "literal_disjoint_support_identity_count": sum(
            row["literal_disjoint_support_identity"] for row in reciprocity_rows
        ),
        "candidate_specialized_zero_identity_count": sum(
            row["candidate_specialized_zero_identity"] for row in reciprocity_rows
        ),
        "queried_pair_cycle_degree_sum": sum(
            row["pair_cycle_degree"] for row in reciprocity_rows
        ),
        "miller_merge_count": sum(
            row["miller_merge_count"] for row in reciprocity_rows
        ),
        "charged_miller_probe_evaluation_count": sum(
            row["miller_merge_count"] * len(probes) for row in reciprocity_rows
        ),
        "maximum_miller_shuffle_attempt": max(
            row["miller_shuffle_attempt"] for row in reciprocity_rows
        ),
        "reciprocity_transcript_sha256": sha256_json(reciprocity_rows),
        "reciprocity_rows": reciprocity_rows,
        "finite_pair_and_miller_enumeration_receives_asymptotic_credit": False,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "principal_target_signed_incidence": (
            "Let h=F_num/F_den be the R167 principal target witness with divisor "
            "sum_T[T]-sum_R[R], and require every auxiliary pole R to avoid the "
            "selected pair sums. Then h(P+Q)=0 if and only if P+Q is a retained "
            "target. This is sign-sensitive and treats P=Q through the elliptic "
            "group law rather than an interpolant derivative or separate tangent formula."
        ),
        "subset_pontryagin_norm": (
            "For a selected subset A and full selected divisor D, define the "
            "effective Pontryagin cycle A*D=sum_(P in A,Q in D)[P+Q]. Its degree "
            "is m*n and H(A)=h(A*D)=product_(P,Q)h(P+Q). H(A)=0 exactly when A "
            "contains an R175 candidate, so the same balanced group-testing tree applies."
        ),
        "principal_completion": (
            "The group sum of A*D is n*sum(A)+m*sum(D). If it is nonzero, append "
            "its negative C and choose f_(A*D) with divisor A*D+[C]-(mn+1)[O]; "
            "if it is zero, use divisor A*D-mn[O]. A generalized Miller tree has "
            "one merge per pair-cycle point up to constants."
        ),
        "weil_reciprocity_scalar": (
            "For disjoint support, Weil reciprocity gives h(div f)=f(div h), so "
            "h(A*D)=f(targets)/f(auxiliary_poles)*h(O)^d/h(C), with d=mn+1 "
            "and the h(C) factor omitted when C=O. Candidate intersections are "
            "the zero specialization of the same identity and must not be inverted."
        ),
        "represented_degree_boundary": (
            "Reciprocity removes the explicit target-factor loop only after "
            "f_(A*D) has been constructed. Its zero cycle and Miller program have "
            "Theta(mn) represented size. At the root m=n this is n^2=B^(9/2); "
            "summing mn over the balanced queried nodes remains softly Theta(n^2)."
        ),
        "open_factored_resultant": (
            "The surviving primitive is a factored trilinear elliptic resultant "
            "that evaluates h(A*D) directly from compact A, D, and target-divisor "
            "descriptors, with reusable D/target preprocessing and softly "
            "O(m+N) node work, without materializing the mn Pontryagin cycle."
        ),
        "scope": (
            "R176 proves an exact signed principal-function and reciprocity "
            "interface and closes standard represented pair-cycle routes. It "
            "does not prove a lower bound against factored circuits and does not "
            "supply an unconditional ECDLP, Pollard-rho, or Shoup improvement."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_principal_target_pontryagin_resultant.cost.r176.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "r163_candidate_output_exponent_B": fraction_record(Fraction(3, 4)),
        "compact_target_principal_witness_exponent_B": fraction_record(Fraction(5, 4)),
        "compact_selected_divisor_state_exponent_B": fraction_record(Fraction(9, 4)),
        "root_pontryagin_cycle_degree_exponent_B": fraction_record(Fraction(9, 2)),
        "balanced_queried_pair_cycle_volume_exponent_B": fraction_record(Fraction(9, 2)),
        "represented_principal_completion_miller_state_exponent_B": fraction_record(Fraction(9, 2)),
        "direct_target_slp_on_pair_cycle_exponent_B": fraction_record(Fraction(23, 4)),
        "represented_fast_evaluation_after_pair_enumeration_exponent_B": fraction_record(Fraction(9, 2)),
        "conditional_factored_trilinear_resultant_total_exponent_B": fraction_record(Fraction(9, 4)),
        "r163_label_and_backpointer_postprocessing_exponent_B": fraction_record(Fraction(2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "standard_represented_pair_cycle_route_inside_rho": False,
        "standard_reciprocity_route_inside_rho": False,
        "conditional_factored_trilinear_resultant_strictly_inside_rho": True,
        "factored_trilinear_elliptic_resultant_supplied": False,
        "represented_degree_observation_claimed_as_circuit_lower_bound": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["principal_leaf_roots_match_r175_and_r167"]
        and row["principal_tree_zero_pattern_matches_r175"]
        and row["all_pair_cycle_group_sum_identities_exact"]
        and row["all_completion_values_units"]
        and row["all_miller_pole_products_units"]
        and row["all_principal_pontryagin_reciprocity_identities_exact"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_principal_target_pontryagin_resultant.controls.r176.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_principal_leaf_roots_match_r175_and_r167": all(
            row["principal_leaf_roots_match_r175_and_r167"] for row in rows
        ),
        "all_principal_tree_zero_patterns_match_r175": all(
            row["principal_tree_zero_pattern_matches_r175"] for row in rows
        ),
        "all_pair_cycle_group_sum_identities_exact": all(
            row["all_pair_cycle_group_sum_identities_exact"] for row in rows
        ),
        "all_completion_values_units": all(
            row["all_completion_values_units"] for row in rows
        ),
        "all_miller_pole_products_units": all(
            row["all_miller_pole_products_units"] for row in rows
        ),
        "all_principal_pontryagin_reciprocity_identities_exact": all(
            row["all_principal_pontryagin_reciprocity_identities_exact"]
            for row in rows
        ),
        "selected_pair_evaluation_count": sum(
            row["selected_pair_evaluation_count"] for row in rows
        ),
        "selected_pair_denominator_unit_count": sum(
            row["selected_pair_denominator_unit_count"] for row in rows
        ),
        "principal_pair_incidence_count": sum(
            row["principal_pair_incidence_count"] for row in rows
        ),
        "diagonal_pair_incidence_count": sum(
            row["diagonal_pair_incidence_count"] for row in rows
        ),
        "off_diagonal_pair_incidence_count": sum(
            row["off_diagonal_pair_incidence_count"] for row in rows
        ),
        "candidate_root_count": sum(len(row["principal_leaf_roots"]) for row in rows),
        "principal_tree_query_count": sum(
            row["principal_tree_query_count"] for row in rows
        ),
        "reciprocity_identity_count": sum(
            row["reciprocity_identity_count"] for row in rows
        ),
        "literal_disjoint_support_identity_count": sum(
            row["literal_disjoint_support_identity_count"] for row in rows
        ),
        "candidate_specialized_zero_identity_count": sum(
            row["candidate_specialized_zero_identity_count"] for row in rows
        ),
        "queried_pair_cycle_degree_sum": sum(
            row["queried_pair_cycle_degree_sum"] for row in rows
        ),
        "miller_merge_count": sum(row["miller_merge_count"] for row in rows),
        "charged_miller_probe_evaluation_count": sum(
            row["charged_miller_probe_evaluation_count"] for row in rows
        ),
        "maximum_miller_shuffle_attempt": max(
            row["maximum_miller_shuffle_attempt"] for row in rows
        ),
        "finite_pair_and_miller_enumeration_receives_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "r175_r167_r169_primary_source_bindings_exact": True,
        "six_controls_replayed": len(rows) == 6,
        "principal_target_signed_incidence_complete": controls[
            "all_principal_leaf_roots_match_r175_and_r167"
        ],
        "all_selected_pair_denominators_units": controls[
            "selected_pair_evaluation_count"
        ]
        == controls["selected_pair_denominator_unit_count"],
        "diagonal_group_law_incidence_complete": controls[
            "diagonal_pair_incidence_count"
        ]
        > 0,
        "r175_tree_zero_pattern_replay_complete": controls[
            "all_principal_tree_zero_patterns_match_r175"
        ],
        "pontryagin_cycle_degree_and_sum_complete": controls[
            "all_pair_cycle_group_sum_identities_exact"
        ],
        "principal_completion_units_complete": controls[
            "all_completion_values_units"
        ],
        "miller_auxiliary_pole_units_complete": controls[
            "all_miller_pole_products_units"
        ],
        "literal_disjoint_weil_reciprocity_complete": controls[
            "literal_disjoint_support_identity_count"
        ]
        > 0,
        "candidate_specialized_zero_reciprocity_complete": controls[
            "candidate_specialized_zero_identity_count"
        ]
        > 0,
        "all_node_reciprocity_identities_complete": controls[
            "all_principal_pontryagin_reciprocity_identities_exact"
        ],
        "represented_pair_cycle_n2_cost_charged": True,
        "represented_miller_n2_cost_charged": True,
        "direct_pair_target_n2N_cost_charged": True,
        "finite_enumeration_scoped_without_asymptotic_credit": True,
        "factored_trilinear_elliptic_resultant_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    classification = (
        "ADMIT_PRINCIPAL_TARGET_SIGNED_INCIDENCE__H_OF_P_PLUS_Q_ZERO_IFF_"
        "POSITIVE_TARGET_SUM__DIAGONAL_BY_GROUP_LAW__358_R175_TREE_QUERIES_"
        "REPLAYED__358_COMPLETED_PONTRYAGIN_WEIL_IDENTITIES__PAIR_CYCLE_DEGREE_"
        "MN__STANDARD_ROOT_AND_TREE_STATE_N2_B9O2__RECIPROCITY_PRESERVES_MN_"
        "INPUT__FACTORED_TRILINEAR_RESULTANT_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_"
        "RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute a factored trilinear elliptic resultant that accepts "
        "compact U_A,V_A, the fixed U_D,V_D, and the degree-N principal target "
        "witness h, then returns h(A*D) with one softly O(n+N) reusable setup and "
        "softly O(m+N) work per balanced node. Reject explicit mn pair sums, a "
        "degree-mn function or Miller program, n^2 tensor or displacement state, "
        "target-dependent per-node setup, candidate inversions, and unit-cost "
        "resultant, norm, root, count, marginal, rank, source, or generic locator oracles."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Replace the R175 signed subset scalar by a principal target-function "
            "evaluation on a Pontryagin product cycle and test whether ordinary "
            "Weil reciprocity supplies the required reusable below-rho oracle."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "eagen_ecip_weil_reciprocity": {
                "source": "IACR ePrint 2022/596",
                "fit": (
                    "Supplies the principal-divisor, Mumford, elliptic-resultant, "
                    "and Weil-reciprocity interface. It does not state a factored "
                    "Pontryagin-product resultant with sub-mn input work."
                ),
            },
            "miller_1986": {
                "fit": (
                    "Supplies the line-function merge algorithm used to evaluate "
                    "the completed principal cycles. Its program has one merge per "
                    "represented pair-cycle point, so the standard route retains mn size."
                )
            },
            "r169_displacement_deduplication": {
                "fit": (
                    "R169 already closes the tested ordinary diagonal displacement "
                    "generators. R176 leaves only a new factored Pontryagin-resultant "
                    "circuit, not another full n-by-n kernel representation."
                )
            },
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "principal_signed_incidence_admitted": all_exact,
            "represented_weil_reciprocity_route_admitted": True,
            "factored_trilinear_elliptic_resultant_admitted": False,
            "lane_admitted": False,
        },
        "classification": classification,
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    frozen = {
        "schema": "p1553.m6_principal_target_pontryagin_resultant.frozen.r176.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "critical_experiment": {
            "hypothesis": (
                "The scalar h(A*D) admits a factored trilinear resultant with "
                "softly O(m+n+N) work and reusable fixed-D/target state."
            ),
            "decisive_test": next_action,
            "falsifier": (
                "The route materializes any mn pair-sum cycle, degree-mn principal "
                "function, n^2 tensor or displacement state, or performs target-"
                "dependent preprocessing for each node."
            ),
        },
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_principal_target_pontryagin_resultant.replay.r176.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "principal_leaf_value_sha256": row[
                    "principal_leaf_value_sha256"
                ],
                "pair_transcript_sha256": row["pair_transcript_sha256"],
                "principal_tree_zero_projection_sha256": row[
                    "principal_tree_zero_projection_sha256"
                ],
                "reciprocity_transcript_sha256": row[
                    "reciprocity_transcript_sha256"
                ],
                "principal_leaf_roots": row["principal_leaf_roots"],
            }
            for row in rows
        ],
    }
    resultant = {
        "schema": "p1553.m6_principal_target_pontryagin_resultant.resultant.r176.v1",
        "principal_target_signed_incidence": theorem[
            "principal_target_signed_incidence"
        ],
        "subset_pontryagin_norm": theorem["subset_pontryagin_norm"],
        "principal_completion": theorem["principal_completion"],
        "weil_reciprocity_scalar": theorem["weil_reciprocity_scalar"],
        "represented_degree_boundary": theorem["represented_degree_boundary"],
        "controls": [
            {
                "control_id": row["control_id"],
                "c3_divisor_degree": row["c3_divisor_degree"],
                "target_witness_degree": row["target_witness_degree"],
                "principal_pair_incidence_count": row[
                    "principal_pair_incidence_count"
                ],
                "principal_leaf_roots": row["principal_leaf_roots"],
                "queried_pair_cycle_degree_sum": row[
                    "queried_pair_cycle_degree_sum"
                ],
                "reciprocity_rows": row["reciprocity_rows"],
            }
            for row in rows
        ],
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "resultant": resultant,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--resultant-output", type=Path, default=DEFAULT_RESULTANT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    outputs = (
        (args.report_output, bundle["report"]),
        (args.frozen_output, bundle["frozen"]),
        (args.cost_output, bundle["cost"]),
        (args.replay_output, bundle["replay"]),
        (args.controls_output, bundle["controls"]),
        (args.resultant_output, bundle["resultant"]),
    )
    for path, value in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane=0 breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
