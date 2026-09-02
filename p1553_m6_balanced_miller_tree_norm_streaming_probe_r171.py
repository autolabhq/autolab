#!/usr/bin/env python3
"""Instantiate balanced generalized Miller trees and audit norm streaming."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_balanced_miller_tree_norm_streaming.r171.v1"

R170_PRODUCER = ROOT / "p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_r170.py"
R170_REPORT = ROOT / "p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_report_r170.json"
R170_FROZEN = ROOT / "frozen_m6_lambda_zero_fitting_target_norm_dedup.json"
R170_COST = ROOT / "m6_lambda_zero_fitting_target_norm_dedup_cost_ledger.json"
R170_REPLAY = ROOT / "m6_lambda_zero_fitting_target_norm_dedup_replay.json"
R170_CONTROLS = ROOT / "m6_lambda_zero_fitting_target_norm_dedup_controls.json"
R170_FITTING = ROOT / "lambda_zero_fitting_norm_and_density_r170.json"
R170_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_r170.py"
R170_GATE = ROOT / "p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_gate_r170.md"
R170_PARENT = ROOT / "p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_parent_report_r170.yaml"
MILLER_PAPER = ROOT / "references/miller_weil_pairing_algorithm_1986.pdf"
ENGE_PAPER = ROOT / "references/enge_bilinear_pairings_1301.5520v2.pdf"
TRUNCATED_RESULTANT_PAPER = ROOT / "references/moroz_schost_truncated_resultant_1609.04259.pdf"
MULTIPOINT_PAPER = ROOT / "references/bhargava_ghosh_guo_kumar_umans_multipoint_2205.00342v1.pdf"

SOURCE_BINDINGS = (
    ("r170_producer", R170_PRODUCER, "71f94407c05de7c0a4bcc77643c63e5f601b1c609639cbdba874a72c56f0091d"),
    ("r170_report", R170_REPORT, "a354427601094c8346253ac38bc0fd53f878e67d71962846561088ee6495162c"),
    ("r170_frozen", R170_FROZEN, "34735b27f7b9d9a018a2c158cccaab4c9d61942144636d72b4cdcdc95d9d9f97"),
    ("r170_cost", R170_COST, "97cf12e1011c10148588da6febba2edbcfaa445f12fcb4df12ca81dc11278113"),
    ("r170_replay", R170_REPLAY, "84a36d4ab086664e4a2382e6b057f45b741dd43c38c7acd19026c0ff8f51d3cd"),
    ("r170_controls", R170_CONTROLS, "1c00d68dcc15217ce1d3b7192e0992ccb9a1443450df3d100cb1d1beb2fd409f"),
    ("r170_fitting", R170_FITTING, "fa7b5427696e244c257fa95168e3b7f5646c9bdff5d10ff25fd0b8193d28bea1"),
    ("r170_test", R170_TEST, "51f82e0b8e762591669eec9aaa240e80c44b34ec1e7e56e80ca8ee83e955d2f3"),
    ("r170_gate", R170_GATE, "564ad5ffd47317768209244b1432f13cc2a0d1245a3d582e59c9bb0174375e02"),
    ("r170_parent", R170_PARENT, "a6bcb432f330f492669dbb597361d3cc9c7ceff468292a3b00bebe231965df91"),
    ("miller_1986", MILLER_PAPER, "39c76c7643278b87b3d8c24b9a07d0b4cbfb561cd13735548990848e0f0bd166"),
    ("enge_2013", ENGE_PAPER, "93b99fa2d13e09c1bc8282b58d472be6285ceea9b594e9b97d765212a3aaa8e4"),
    ("moroz_schost_2016", TRUNCATED_RESULTANT_PAPER, "160c68cfbb413ca27352a064cbf2d27f7ad4ed6a210c3d6ead2770e00204b709"),
    ("bhargava_et_al_2022", MULTIPOINT_PAPER, "14eddc304a7dd8995ebc1e24171571fd9dc0f1f837ca35a7f9e2e6fb21bfafa8"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_report_r171.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_balanced_miller_tree_norm_streaming.json"
DEFAULT_COST = ROOT / "m6_balanced_miller_tree_norm_streaming_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_balanced_miller_tree_norm_streaming_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_balanced_miller_tree_norm_streaming_controls.json"
DEFAULT_SLP = ROOT / "balanced_miller_tree_and_leaf_cancellation_r171.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R170 = load_module("p1553_r170_for_r171", R170_PRODUCER)
R167 = R170.R167
R161 = R170.R161


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
        raise AssertionError(f"R171 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def point_list(point: tuple[int, int] | None) -> list[int] | None:
    return None if point is None else [int(point[0]), int(point[1])]


def point_key(point: tuple[int, int] | None) -> str:
    return "O" if point is None else f"{int(point[0])}:{int(point[1])}"


def point_from_json(value: list[int] | None) -> tuple[int, int] | None:
    return None if value is None else (int(value[0]), int(value[1]))


def line_kind(
    left: tuple[int, int] | None,
    right: tuple[int, int] | None,
    result: tuple[int, int] | None,
) -> str:
    if left is None or right is None:
        return "identity"
    if result is None:
        return "inverse_vertical"
    if left == right:
        return "tangent_over_vertical"
    return "chord_over_vertical"


def merge_state(
    left: dict[str, Any],
    right: dict[str, Any],
    curve: dict[str, Any],
    level: int,
    ordinal: int,
) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    left_sum = left["sum_point"]
    right_sum = right["sum_point"]
    result = R167.point_add(left_sum, right_sum, curve)
    kind = line_kind(left_sum, right_sum, result)
    line_leading = prime - 1 if kind in {
        "tangent_over_vertical", "chord_over_vertical"
    } else 1
    merge = {
        "merge_index": ordinal,
        "level": level,
        "left_size": left["size"],
        "right_size": right["size"],
        "left_sum": point_list(left_sum),
        "right_sum": point_list(right_sum),
        "result_sum": point_list(result),
        "line_kind": kind,
        "line_local_leading_coefficient": line_leading,
    }
    return {
        "size": left["size"] + right["size"],
        "sum_point": result,
        "depth": max(left["depth"], right["depth"]) + 1,
        "leading_coefficient": (
            left["leading_coefficient"]
            * right["leading_coefficient"]
            * line_leading
        ) % prime,
        "merges": [*left["merges"], *right["merges"], merge],
    }


def build_balanced_tree(
    points: list[tuple[int, int]], curve: dict[str, Any]
) -> dict[str, Any]:
    if not points:
        raise ValueError("zero list must be nonempty")
    states = [
        {
            "size": 1,
            "sum_point": point,
            "depth": 0,
            "leading_coefficient": 1,
            "merges": [],
        }
        for point in points
    ]
    level = 0
    ordinal = 0
    while len(states) > 1:
        next_states = []
        for index in range(0, len(states), 2):
            if index + 1 == len(states):
                next_states.append(states[index])
                continue
            next_states.append(
                merge_state(
                    states[index], states[index + 1], curve, level, ordinal
                )
            )
            ordinal += 1
        states = next_states
        level += 1
    tree = states[0]
    tree["leaf_points"] = [point_list(point) for point in points]
    tree["merge_count"] = len(tree["merges"])
    tree["merge_transcript_sha256"] = sha256_json(tree["merges"])
    return tree


def line_quotient_value(
    merge: dict[str, Any],
    evaluation: tuple[int, int] | None,
    curve: dict[str, Any],
) -> int:
    if evaluation is None:
        raise ZeroDivisionError("individual Miller line has a pole at infinity")
    prime = int(curve["field_prime"])
    kind = merge["line_kind"]
    if kind == "identity":
        return 1
    left = point_from_json(merge["left_sum"])
    right = point_from_json(merge["right_sum"])
    result = point_from_json(merge["result_sum"])
    if left is None or right is None:
        raise AssertionError("nonidentity merge lost a finite endpoint")
    xq, yq = evaluation
    x1, y1 = left
    if kind == "inverse_vertical":
        return (xq - x1) % prime
    x2, y2 = right
    if kind == "tangent_over_vertical":
        numerator = (3 * x1 * x1 + int(curve["curve_a"])) % prime
        denominator = 2 * y1 % prime
    else:
        numerator = (y2 - y1) % prime
        denominator = (x2 - x1) % prime
    if denominator == 0:
        raise ZeroDivisionError("Miller slope denominator vanished")
    slope = numerator * pow(denominator, -1, prime) % prime
    line = (yq - y1 - slope * (xq - x1)) % prime
    if result is None:
        raise AssertionError("noninverse line lost its result")
    vertical = (xq - result[0]) % prime
    if vertical == 0:
        raise ZeroDivisionError("Miller vertical denominator vanished")
    return line * pow(vertical, -1, prime) % prime


def tree_value(
    tree: dict[str, Any],
    evaluation: tuple[int, int] | None,
    curve: dict[str, Any],
) -> int:
    if evaluation is None:
        if tree["sum_point"] is not None:
            raise ZeroDivisionError("nonprincipal tree has a residual point")
        return int(tree["leading_coefficient"])
    prime = int(curve["field_prime"])
    value = 1
    for merge in tree["merges"]:
        value = value * line_quotient_value(merge, evaluation, curve) % prime
    return value


def tree_quotient_value(
    numerator_tree: dict[str, Any],
    denominator_tree: dict[str, Any],
    evaluation: tuple[int, int] | None,
    curve: dict[str, Any],
) -> int:
    prime = int(curve["field_prime"])
    numerator = tree_value(numerator_tree, evaluation, curve)
    denominator = tree_value(denominator_tree, evaluation, curve)
    if denominator == 0:
        raise ZeroDivisionError("Miller denominator tree vanished")
    return numerator * pow(denominator, -1, prime) % prime


def tree_symbolic_ledger(tree: dict[str, Any]) -> Counter[str]:
    ledger: Counter[str] = Counter()
    for merge in tree["merges"]:
        ledger[point_key(point_from_json(merge["left_sum"]))] += 1
        ledger[point_key(point_from_json(merge["right_sum"]))] += 1
        ledger[point_key(point_from_json(merge["result_sum"]))] -= 1
        ledger["O"] -= 1
    return Counter({key: value for key, value in ledger.items() if value})


def expected_tree_ledger(tree: dict[str, Any]) -> Counter[str]:
    ledger = Counter(point_key(point_from_json(point)) for point in tree["leaf_points"])
    ledger[point_key(tree["sum_point"])] -= 1
    ledger["O"] -= tree["size"] - 1
    return Counter({key: value for key, value in ledger.items() if value})


def signed_counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"point_key": key, "exponent": counter[key]}
        for key in sorted(counter)
        if counter[key]
    ]


def safe_dense_tree_ratios(
    tree: dict[str, Any],
    witness: dict[str, Any],
    generator: tuple[int, int],
    curve: dict[str, Any],
    wanted: int = 16,
) -> list[int]:
    prime = int(curve["field_prime"])
    order = int(curve["subgroup_order"])
    ratios = []
    for scalar in range(1, min(order, 8193)):
        point = R161.R70.scalar_mul(scalar, generator, curve)
        if point is None:
            continue
        point = tuple(point)
        try:
            compact = tree_value(tree, point, curve)
        except ZeroDivisionError:
            continue
        dense = R167.witness_value(witness, point, prime)
        if compact == 0 or dense == 0:
            continue
        ratios.append(dense * pow(compact, -1, prime) % prime)
        if len(ratios) == wanted:
            break
    if len(ratios) < wanted:
        raise AssertionError("insufficient safe dense/tree comparison points")
    if len(set(ratios)) != 1:
        raise AssertionError("balanced Miller tree differs non-scalarly from witness")
    return ratios


def translated_point(
    point: tuple[int, int] | None,
    shift: tuple[int, int],
    curve: dict[str, Any],
) -> tuple[int, int] | None:
    return R167.point_add(point, R167.point_negate(shift, curve), curve)


def line_reciprocity_rows(
    tree_name: str,
    tree: dict[str, Any],
    shifts: list[tuple[int, int]],
    signed_support: list[tuple[int, int]],
    divisor: dict[str, Any],
    curve: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    prime = int(curve["field_prime"])
    degree = len(divisor["records"])
    rows = []
    skipped = 0
    for merge in tree["merges"]:
        left = point_from_json(merge["left_sum"])
        right = point_from_json(merge["right_sum"])
        result = point_from_json(merge["result_sum"])
        for shift_index, shift in enumerate(shifts):
            try:
                at_shift = line_quotient_value(merge, shift, curve)
                signed_product = 1
                for support_point in signed_support:
                    evaluation = R167.point_add(support_point, shift, curve)
                    signed_product = (
                        signed_product
                        * line_quotient_value(merge, evaluation, curve)
                        % prime
                    )
                translated = [
                    translated_point(endpoint, shift, curve)
                    for endpoint in (left, right, result, None)
                ]
                if any(point is None for point in translated):
                    raise ZeroDivisionError("f0 evaluated at infinity")
                f_left, f_right, f_result, f_origin = [
                    R167.kummer_value(divisor, point, prime)
                    for point in translated
                ]
                denominator = (
                    pow(at_shift, 2 * degree, prime)
                    * f_result
                    * f_origin
                    % prime
                )
                if denominator == 0:
                    raise ZeroDivisionError("line reciprocity denominator vanished")
                left_value = signed_product * pow(
                    pow(at_shift, 2 * degree, prime), -1, prime
                ) % prime
                right_value = (
                    f_left
                    * f_right
                    * pow(f_result * f_origin % prime, -1, prime)
                    % prime
                )
            except ZeroDivisionError:
                skipped += 1
                continue
            if left_value != right_value:
                raise AssertionError("line-level Weil reciprocity failed")
            rows.append(
                {
                    "tree": tree_name,
                    "merge_index": merge["merge_index"],
                    "shift_index": shift_index,
                    "line_kind": merge["line_kind"],
                    "normalized_signed_product": left_value,
                    "four_translate_ratio": right_value,
                    "identity_exact": True,
                }
            )
    return rows, skipped


def generic_line_test_shifts(
    generator: tuple[int, int],
    divisor: dict[str, Any],
    curve: dict[str, Any],
    wanted: int = 16,
) -> list[tuple[int, int]]:
    prime = int(curve["field_prime"])
    order = int(curve["subgroup_order"])
    shifts = []
    for scalar in range(1, min(order, 8193)):
        point = R161.R70.scalar_mul(scalar, generator, curve)
        if point is None:
            continue
        point = tuple(point)
        negative = R167.point_negate(point, curve)
        if negative is None or R167.kummer_value(divisor, negative, prime) == 0:
            continue
        shifts.append(point)
        if len(shifts) == wanted:
            break
    if len(shifts) < wanted:
        raise AssertionError("insufficient generic line-reciprocity shifts")
    return shifts


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    factor_base, divisor, target_records = R167.R166.R164.target_material(curve, seed)
    r167 = R167.finite_control(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    numerator_points = [
        point_from_json(point)
        for point in r167["numerator_witness"]["prescribed_points"]
    ]
    denominator_points = [
        point_from_json(point)
        for point in r167["denominator_witness"]["prescribed_points"]
    ]
    if any(point is None for point in [*numerator_points, *denominator_points]):
        raise AssertionError("zero lists must be finite")
    numerator_points = [point for point in numerator_points if point is not None]
    denominator_points = [point for point in denominator_points if point is not None]
    numerator_tree = build_balanced_tree(numerator_points, curve)
    denominator_tree = build_balanced_tree(denominator_points, curve)
    if numerator_tree["sum_point"] is not None or denominator_tree["sum_point"] is not None:
        raise AssertionError("principal-divisor tree did not close at infinity")
    if numerator_tree["merge_count"] != len(numerator_points) - 1:
        raise AssertionError("numerator merge count drifted")
    if denominator_tree["merge_count"] != len(denominator_points) - 1:
        raise AssertionError("denominator merge count drifted")

    generator = tuple(factor_base["generator"])
    numerator_ratios = safe_dense_tree_ratios(
        numerator_tree, r167["numerator_witness"], generator, curve
    )
    denominator_ratios = safe_dense_tree_ratios(
        denominator_tree, r167["denominator_witness"], generator, curve
    )
    prime = int(curve["field_prime"])
    signed_support = selected + [R167.point_negate(point, curve) for point in selected]
    if any(point is None for point in signed_support):
        raise AssertionError("signed support reached infinity")
    signed_support = [point for point in signed_support if point is not None]
    correction_points = [
        tuple(r167["anchor"]),
        *[tuple(point) for point in r167["auxiliary_points"]],
    ]
    generic_shifts = generic_line_test_shifts(generator, divisor, curve)

    replay_rows = []
    for left in selected:
        direct = 1
        for target in targets:
            point = R167.point_subtract(target, left, curve)
            if point is None:
                raise AssertionError("retained target reached the pole")
            direct = direct * R167.kummer_value(divisor, point, prime) % prime
        correction = 1
        for correction_point in correction_points:
            point = R167.point_subtract(correction_point, left, curve)
            if point is None:
                raise AssertionError("correction reached the pole")
            correction = correction * R167.kummer_value(divisor, point, prime) % prime
        swapped = 1
        for support_point in signed_support:
            evaluation = R167.point_add(support_point, left, curve)
            swapped = swapped * tree_quotient_value(
                numerator_tree, denominator_tree, evaluation, curve
            ) % prime
        h_at_left = tree_quotient_value(
            numerator_tree, denominator_tree, left, curve
        )
        if h_at_left == 0:
            raise AssertionError("tree quotient lost its selected-divisor unit")
        corrected = (
            correction
            * swapped
            * pow(pow(h_at_left, 2 * len(selected), prime), -1, prime)
            % prime
        )
        if corrected != direct:
            raise AssertionError("balanced Miller quotient replay failed")
        replay_rows.append(
            {
                "left_endpoint": point_list(left),
                "direct_target_norm": direct,
                "tree_corrected_norm": corrected,
                "candidate_zero": direct == 0,
                "identity_exact": True,
            }
        )

    numerator_line_rows, numerator_skipped = line_reciprocity_rows(
        "numerator", numerator_tree, generic_shifts, signed_support, divisor, curve
    )
    denominator_line_rows, denominator_skipped = line_reciprocity_rows(
        "denominator", denominator_tree, generic_shifts, signed_support, divisor, curve
    )
    line_rows = [*numerator_line_rows, *denominator_line_rows]
    if not line_rows:
        raise AssertionError("no admissible line-level reciprocity controls")

    numerator_ledger = tree_symbolic_ledger(numerator_tree)
    denominator_ledger = tree_symbolic_ledger(denominator_tree)
    if numerator_ledger != expected_tree_ledger(numerator_tree):
        raise AssertionError("numerator tree did not telescope to its leaves")
    if denominator_ledger != expected_tree_ledger(denominator_tree):
        raise AssertionError("denominator tree did not telescope to its leaves")
    quotient_ledger = numerator_ledger.copy()
    quotient_ledger.subtract(denominator_ledger)
    correction_ledger = Counter(point_key(point) for point in correction_points)
    corrected_ledger = quotient_ledger.copy()
    corrected_ledger.update(correction_ledger)
    expected_targets = Counter(point_key(point) for point in targets)
    corrected_ledger = Counter(
        {key: value for key, value in corrected_ledger.items() if value}
    )
    if corrected_ledger != expected_targets:
        raise AssertionError("tree cancellation did not leave exactly target leaves")

    candidate_roots = sorted(
        int(row["left_endpoint"][0])
        for row in replay_rows
        if row["candidate_zero"]
    )
    return {
        "control_id": f"{curve['family_id']}_balanced_miller_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "subgroup_order": int(curve["subgroup_order"]),
        "seed": seed,
        "c3_divisor_degree": len(selected),
        "retained_target_count": len(targets),
        "zero_list_size": len(numerator_points),
        "numerator_tree": numerator_tree,
        "denominator_tree": denominator_tree,
        "total_line_merge_count": (
            numerator_tree["merge_count"] + denominator_tree["merge_count"]
        ),
        "maximum_tree_depth": max(
            numerator_tree["depth"], denominator_tree["depth"]
        ),
        "numerator_dense_tree_scalar_ratio": numerator_ratios[0],
        "denominator_dense_tree_scalar_ratio": denominator_ratios[0],
        "dense_tree_comparison_count": len(numerator_ratios) + len(denominator_ratios),
        "all_dense_tree_ratios_constant": True,
        "tree_corrected_norm_replay_exact": True,
        "candidate_roots": candidate_roots,
        "r167_candidate_roots": r167["candidate_roots"],
        "candidate_roots_match_r167": candidate_roots == r167["candidate_roots"],
        "line_reciprocity_admissible_row_count": len(line_rows),
        "line_reciprocity_generic_shift_count": len(generic_shifts),
        "selected_endpoint_linewise_origin_factor_is_zero": all(
            R167.kummer_value(
                divisor, R167.point_negate(point, curve), prime
            ) == 0
            for point in selected
        ),
        "line_reciprocity_skipped_pole_or_nonunit_row_count": (
            numerator_skipped + denominator_skipped
        ),
        "all_admissible_line_reciprocity_rows_exact": True,
        "line_reciprocity_transcript_sha256": sha256_json(line_rows),
        "tree_norm_replay_transcript_sha256": sha256_json(replay_rows),
        "numerator_telescoped_ledger": signed_counter_rows(numerator_ledger),
        "denominator_telescoped_ledger": signed_counter_rows(denominator_ledger),
        "quotient_telescoped_ledger": signed_counter_rows(quotient_ledger),
        "corrected_residual_ledger": signed_counter_rows(corrected_ledger),
        "residual_leaf_factor_count": sum(corrected_ledger.values()),
        "residual_leaf_factors_equal_targets": corrected_ledger == expected_targets,
        "balanced_depth_reduces_total_leaf_factor_count": False,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_controls_receive_asymptotic_attack_credit": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "balanced_generalized_miller_tree": (
            "For a finite zero list Z=(Z_1,...,Z_m), a leaf has state "
            "(1,Z_i). Merging states (f_A,A) and (f_B,B) multiplies by "
            "l_(A,B)/v_(A+B) and replaces the residual point by A+B. The "
            "result has divisor sum_i[Z_i]-[sum_i Z_i]-(m-1)[O]. A balanced "
            "tree has exactly m-1 line merges and logarithmic depth. When the "
            "point sum is O it represents the required principal function."
        ),
        "line_norm_homomorphism": (
            "For f0(Q)=U(x(Q)), a line quotient g_(A,B) and shift P obey, on "
            "disjoint support, product_(Q in S union -S) g_(A,B)(Q+P) / "
            "g_(A,B)(P)^(2n) = f0(A-P)f0(B-P) / "
            "(f0(A+B-P)f0(-P)). This is the node-local norm homomorphism."
        ),
        "specialization_boundary": (
            "For every selected endpoint P, f0(-P)=U(x(P))=0, so individual "
            "line ratios are not units and cannot be specialized one node at a "
            "time. The equal-size numerator/denominator origin factors must be "
            "cancelled symbolically before evaluating selected or candidate-zero "
            "rows."
        ),
        "tree_telescoping": (
            "Multiplying the node-local identities cancels every internal "
            "partial sum. A size-m tree leaves product_i f0(Z_i-P) divided "
            "by f0(sum_i Z_i-P)f0(-P)^(m-1). For a principal zero list this "
            "is product_i f0(Z_i-P)/f0(-P)^m."
        ),
        "quotient_and_correction_cancellation": (
            "The R167 numerator and denominator trees have equal size, both "
            "sum to O, and share the completion zero C. Their origin factors "
            "and C cancel. Multiplying the public anchor/auxiliary correction "
            "cancels every denominator leaf, leaving exactly the N original "
            "target translates and no internal partial-sum factor."
        ),
        "scoped_streaming_boundary": (
            "Balanced line streaming reduces live memory and circuit depth, "
            "but the node-local homomorphic evaluation still performs "
            "Theta(nN) scalar translate work, and symbolic telescoping returns "
            "the same N target factors as R170. This closes only per-node or "
            "per-leaf streaming; it is not an arithmetic-circuit, RAM, "
            "cell-probe, elliptic-resultant, or generic-group lower bound."
        ),
        "surviving_interface": (
            "The surviving primitive is nonlocal: from compact selected and "
            "target divisors, batch all N elliptic translates against U and "
            "emit their product modulo U in softly O(n+N) work without "
            "visiting the n-by-N pair grid or materializing N dense quotient-"
            "ring elements."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_balanced_miller_tree_norm_streaming.cost.r171.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "balanced_tree_node_state_exponent_B": fraction_record(Fraction(5, 4)),
        "balanced_tree_depth_exponent_B": fraction_record(Fraction(0)),
        "live_endpoint_value_vector_exponent_B": fraction_record(Fraction(9, 4)),
        "node_local_line_norm_work_exponent_B": fraction_record(Fraction(7, 2)),
        "telescoped_leaf_translate_work_exponent_B": fraction_record(Fraction(7, 2)),
        "raw_signed_grid_tree_expansion_exponent_B": fraction_record(Fraction(23, 4)),
        "represented_aggregate_element_exponent_B": fraction_record(Fraction(9, 4)),
        "preferred_nonlocal_batch_work_exponent_B": fraction_record(Fraction(9, 4)),
        "expected_candidate_exponent_B": fraction_record(Fraction(3, 4)),
        "expected_signed_verification_exponent_B": fraction_record(Fraction(2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "node_local_streaming_rho_excess_exponent_B": fraction_record(Fraction(1)),
        "balanced_tree_state_inside_rho": True,
        "live_endpoint_value_vector_inside_rho": True,
        "node_local_line_norm_work_inside_rho": False,
        "telescoped_leaf_translate_work_inside_rho": False,
        "nonlocal_batched_leaf_translate_operator_supplied": False,
        "finite_tree_cancellation_receives_lower_bound_credit": False,
        "arithmetic_circuit_lower_bound_claimed": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["all_dense_tree_ratios_constant"]
        and row["tree_corrected_norm_replay_exact"]
        and row["candidate_roots_match_r167"]
        and row["all_admissible_line_reciprocity_rows_exact"]
        and row["residual_leaf_factors_equal_targets"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_balanced_miller_tree_norm_streaming.controls.r171.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_balanced_trees_close_at_infinity": all(
            row["numerator_tree"]["sum_point"] is None
            and row["denominator_tree"]["sum_point"] is None
            for row in rows
        ),
        "all_dense_tree_ratios_constant": all(
            row["all_dense_tree_ratios_constant"] for row in rows
        ),
        "all_tree_corrected_norm_replays_exact": all(
            row["tree_corrected_norm_replay_exact"] for row in rows
        ),
        "all_candidate_roots_match_r167": all(
            row["candidate_roots_match_r167"] for row in rows
        ),
        "all_admissible_line_reciprocity_rows_exact": all(
            row["all_admissible_line_reciprocity_rows_exact"] for row in rows
        ),
        "all_residual_leaf_factors_equal_targets": all(
            row["residual_leaf_factors_equal_targets"] for row in rows
        ),
        "all_selected_endpoint_linewise_origin_factors_zero": all(
            row["selected_endpoint_linewise_origin_factor_is_zero"] for row in rows
        ),
        "total_line_merge_count": sum(row["total_line_merge_count"] for row in rows),
        "maximum_tree_depth": max(row["maximum_tree_depth"] for row in rows),
        "dense_tree_comparison_count": sum(
            row["dense_tree_comparison_count"] for row in rows
        ),
        "line_reciprocity_admissible_row_count": sum(
            row["line_reciprocity_admissible_row_count"] for row in rows
        ),
        "line_reciprocity_skipped_pole_or_nonunit_row_count": sum(
            row["line_reciprocity_skipped_pole_or_nonunit_row_count"] for row in rows
        ),
        "residual_leaf_factor_count": sum(
            row["residual_leaf_factor_count"] for row in rows
        ),
        "candidate_root_count": sum(len(row["candidate_roots"]) for row in rows),
        "candidate_oracle_consumed": False,
        "finite_controls_receive_asymptotic_attack_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "fourteen_source_bindings_verified": len(actual_bindings) == 14,
        "miller_principal_divisor_line_interface_bound": True,
        "enge_factored_line_and_direct_evaluation_boundary_bound": True,
        "six_balanced_numerator_trees_constructed": len(rows) == 6,
        "six_balanced_denominator_trees_constructed": len(rows) == 6,
        "all_tree_merge_counts_exact": all(
            row["numerator_tree"]["merge_count"] == row["zero_list_size"] - 1
            and row["denominator_tree"]["merge_count"] == row["zero_list_size"] - 1
            for row in rows
        ),
        "all_tree_sums_close_at_infinity": controls[
            "all_balanced_trees_close_at_infinity"
        ],
        "dense_riemann_roch_witnesses_matched_up_to_scalar": controls[
            "all_dense_tree_ratios_constant"
        ],
        "tree_quotient_replays_r167_norm": controls[
            "all_tree_corrected_norm_replays_exact"
        ],
        "candidate_roots_preserved": controls["all_candidate_roots_match_r167"],
        "node_local_line_reciprocity_verified": controls[
            "all_admissible_line_reciprocity_rows_exact"
        ],
        "selected_endpoint_linewise_origin_zero_recorded": controls[
            "all_selected_endpoint_linewise_origin_factors_zero"
        ],
        "all_internal_partial_sums_telescope": True,
        "common_zero_and_origin_factors_cancel": True,
        "public_auxiliary_correction_leaves_cancel": True,
        "exactly_target_leaves_survive": controls[
            "all_residual_leaf_factors_equal_targets"
        ],
        "balanced_depth_and_live_memory_charged": True,
        "node_local_nN_work_B7O2_charged": True,
        "raw_signed_grid_tree_expansion_B23O4_charged": True,
        "finite_controls_scoped_without_lower_bound_credit": True,
        "moroz_schost_input_contract_deduplicated": True,
        "bhargava_multipoint_input_contract_deduplicated": True,
        "nonlocal_batched_leaf_translate_operator_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    classification = (
        "ADMIT_EXPLICIT_BALANCED_GENERALIZED_MILLER_TREES__DENSE_WITNESS_"
        "EQUALITY_UP_TO_SCALAR__EXACT_R167_NORM_REPLAY__NODE_LOCAL_WEIL_"
        "HOMOMORPHISM__INTERNAL_SUMS_AND_AUXILIARY_LEAVES_TELESCOPE_TO_"
        "EXACTLY_N_TARGET_FACTORS__BALANCING_REDUCES_DEPTH_AND_MEMORY_NOT_"
        "NN_B7O2_WORK__NONLOCAL_BATCHED_LEAF_TRANSLATE_OPERATOR_OPEN__NO_"
        "CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute a nonlocal batched elliptic leaf-translate product "
        "operator: from U and compact selected/target divisors, emit product_j "
        "U(x(T_j-P)) modulo U in softly O(n+N) work, preferably B^(9/4+o(1)), "
        "without visiting the n-by-N pair grid, materializing N dense elements "
        "of F_p[X]/U, or invoking an uncharged norm/resultant/multipoint oracle."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Instantiate R167's generalized Miller witness and determine whether "
            "balanced line-norm streaming removes the R170 nN work barrier."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "miller_and_enge": {
                "titles": [
                    "The Weil pairing, and its efficient calculation",
                    "Bilinear pairings on elliptic curves",
                ],
                "fit": (
                    "Miller supplies the tangent-and-cord line recurrence. Enge "
                    "states that factored lines avoid dense degree growth and "
                    "that direct divisor evaluation uses field elements; neither "
                    "claim batches a moving n-by-N translate grid."
                ),
            },
            "moroz_schost_truncated_resultant": {
                "arxiv": "1609.04259",
                "fit": (
                    "The soft-O(kd) theorem starts from represented bivariate "
                    "polynomials and returns a local coefficient truncation. An "
                    "optimistic k=n,d=N substitution is nN work and does not "
                    "supply the elliptic point-list-to-product compilation."
                ),
            },
            "bhargava_et_al_multipoint": {
                "arxiv": "2205.00342",
                "fit": (
                    "The nearly linear theorem evaluates a polynomial supplied "
                    "as a coefficient vector at explicit points. It does not "
                    "construct the target-dependent elliptic product from its N "
                    "divisor leaves or fuse the moving pair grid."
                ),
            },
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "balanced_generalized_miller_tree_admitted": all_exact,
            "node_local_norm_homomorphism_admitted": all_exact,
            "node_local_slp_streaming_below_rho_admitted": False,
            "nonlocal_batched_leaf_translate_operator_admitted": False,
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
        "schema": "p1553.m6_balanced_miller_tree_norm_streaming.frozen.r171.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "critical_experiment": {
            "hypothesis": (
                "A nonlocal elliptic batch operator can fuse the surviving N "
                "target leaves before quotient-ring or endpoint expansion."
            ),
            "decisive_test": next_action,
            "falsifier": (
                "The route visits nN point/leaf pairs, materializes N dense "
                "quotient-ring factors, expands a degree-nN body, uses candidate "
                "inverses, or treats norm/resultant/multipoint evaluation as a "
                "unit-cost oracle."
            ),
        },
        "promotion_allowed": False,
    }
    slp = {
        "schema": "p1553.m6_balanced_miller_tree_norm_streaming.slp.r171.v1",
        "tree_recurrence": theorem["balanced_generalized_miller_tree"],
        "line_norm_identity": theorem["line_norm_homomorphism"],
        "telescoping_identity": theorem["tree_telescoping"],
        "control_trees": [
            {
                "control_id": row["control_id"],
                "numerator_tree": row["numerator_tree"],
                "denominator_tree": row["denominator_tree"],
                "corrected_residual_ledger": row["corrected_residual_ledger"],
                "line_reciprocity_transcript_sha256": row[
                    "line_reciprocity_transcript_sha256"
                ],
            }
            for row in rows
        ],
    }
    replay = {
        "schema": "p1553.m6_balanced_miller_tree_norm_streaming.replay.r171.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "numerator_merge_sha256": row["numerator_tree"][
                    "merge_transcript_sha256"
                ],
                "denominator_merge_sha256": row["denominator_tree"][
                    "merge_transcript_sha256"
                ],
                "tree_norm_replay_transcript_sha256": row[
                    "tree_norm_replay_transcript_sha256"
                ],
                "line_reciprocity_transcript_sha256": row[
                    "line_reciprocity_transcript_sha256"
                ],
                "candidate_roots": row["candidate_roots"],
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
        "slp": slp,
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
    parser.add_argument("--slp-output", type=Path, default=DEFAULT_SLP)
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
        (args.slp_output, bundle["slp"]),
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
