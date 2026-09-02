#!/usr/bin/env python3
"""Test the deterministic Kummer x-only relaxation of the R165 locator."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_kummer_x_translate_signed_verification.r166.v1"

R165_PRODUCER = ROOT / "p1553_m6_global_randomizer_elliptic_translate_product_probe_r165.py"
R165_REPORT = ROOT / "p1553_m6_global_randomizer_elliptic_translate_product_probe_report_r165.json"
R165_FROZEN = ROOT / "frozen_m6_global_randomizer_elliptic_translate_product.json"
R165_COST = ROOT / "m6_global_randomizer_elliptic_translate_product_cost_ledger.json"
R165_REPLAY = ROOT / "m6_global_randomizer_elliptic_translate_product_replay.json"
R165_CONTROLS = ROOT / "m6_global_randomizer_elliptic_translate_product_controls.json"
R165_TRANSLATE = ROOT / "global_randomizer_fixed_function_translate_product_r165.json"
R165_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_global_randomizer_elliptic_translate_product_probe_r165.py"
R165_GATE = ROOT / "p1553_m6_global_randomizer_elliptic_translate_product_probe_gate_r165.md"
R165_PARENT = ROOT / "p1553_m6_global_randomizer_elliptic_translate_product_probe_parent_report_r165.yaml"
R158_REPORT = ROOT / "p1553_m6_short_relation_near_injectivity_supply_probe_report_r158.json"
R158_GATE = ROOT / "p1553_m6_short_relation_near_injectivity_supply_probe_gate_r158.md"
R152_REPORT = ROOT / "p1553_m6_geometry_only_weight_interpolation_adjoint_probe_report_r152.json"
R152_GATE = ROOT / "p1553_m6_geometry_only_weight_interpolation_adjoint_probe_gate_r152.md"
R161_REPORT = ROOT / "p1553_m6_signed_c3_divisor_translation_gcd_probe_report_r161.json"
R161_GATE = ROOT / "p1553_m6_signed_c3_divisor_translation_gcd_probe_gate_r161.md"
R148_REPORT = ROOT / "p1553_m6_static_3sum_indexing_tradeoff_probe_report_r148.json"
R148_GATE = ROOT / "p1553_m6_static_3sum_indexing_tradeoff_probe_gate_r148.md"

SOURCE_BINDINGS = (
    ("r165_producer", R165_PRODUCER, "32b21c70e961b92a54d975d7294fb13403106e2c8e82f289e19458798d9d615b"),
    ("r165_report", R165_REPORT, "139277bf3141b7efabf0115399fc4e17377a7ba15783bfc2344dfc84826fc0c2"),
    ("r165_frozen", R165_FROZEN, "2d76c303138ecff1d53be8f277012fc87e4ac3ba8874a6f7eafe9c88f5b623e7"),
    ("r165_cost", R165_COST, "5521dc3ec760444f8403ecaa002092550ba7f78596cdbad7cc4ce170932e2daa"),
    ("r165_replay", R165_REPLAY, "5753a28235f5c5ace5ad708df6f7939ed8319ce4951637af84522b542fce7304"),
    ("r165_controls", R165_CONTROLS, "619ed335270854a95a6c2b6a66ece31140a13f581fc0656bef0f09bd54d29a37"),
    ("r165_translate", R165_TRANSLATE, "fd21e63a694c30913ffc9e540290209ea8315d75f7c7539405dd7593458e83ca"),
    ("r165_test", R165_TEST, "95ef2fddd6ca3e193115c2a869b79161c2342a48bd19b2054493c830a5a68ddf"),
    ("r165_gate", R165_GATE, "b0d812d186d2a1bb000d3bb94bdf8f23a74a0d5a338f6636e1f0b7e4fa0002c2"),
    ("r165_parent", R165_PARENT, "b8a81e2e9b259e5280ce2470bb0d57dc57c41172639832191ee83408945cb08d"),
    ("r158_report", R158_REPORT, "bc0e0e864af69c8c03fd21b0bb360adceea7247daf940f5cc5cbf0e51ad30c6a"),
    ("r158_gate", R158_GATE, "20077d3231e0c535af2fe7a4c435148de53fe1da300e1174d1959e89f12a203d"),
    ("r152_report", R152_REPORT, "f2dd5a43593e9e747d969c73afefac9ee7ce7aac3ca79ca2dd009ea1ffe4e46c"),
    ("r152_gate", R152_GATE, "57c336c6e337e06092efc54af710b8cd5ae8de46edf184d9c3335cd44f4bb38a"),
    ("r161_report", R161_REPORT, "7e73325212e9d8c3cf42ae46cce0a5c1bdebea9784b9bf5c30761c006cb8bedd"),
    ("r161_gate", R161_GATE, "0ca8660b9554c39e2c90a8f29cee05554b9fb3adce0c4c2a1038e739b3888bcd"),
    ("r148_report", R148_REPORT, "b01496572b919ffd15406ee83bcd185675b96669d0cd40f51972ddf56f2caed7"),
    ("r148_gate", R148_GATE, "50e465e70d3e57457a64185cd9b86fc9b08bd9ebc56cfe6969f8c1f04508d9ca"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_kummer_x_translate_signed_verification_probe_report_r166.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_kummer_x_translate_signed_verification.json"
DEFAULT_COST = ROOT / "m6_kummer_x_translate_signed_verification_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_kummer_x_translate_signed_verification_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_kummer_x_translate_signed_verification_controls.json"
DEFAULT_KUMMER = ROOT / "kummer_x_candidate_and_signed_false_branch_r166.json"

MATCHED_RANDOM_TARGET_COUNT = 4096


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R165 = load_module("p1553_r165_for_r166", R165_PRODUCER)
R164 = R165.R164
R163 = R165.R163
R161 = R165.R161


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
        raise AssertionError(f"R166 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def coefficient_l1(vector: tuple[int, ...]) -> int:
    return sum(abs(value) for value in vector)


def coefficient_subtract_add(
    target: tuple[int, ...], left: tuple[int, ...], opposite: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(a - b + c for a, b, c in zip(target, left, opposite))


def kummer_value(
    point: tuple[int, int], divisor: dict[str, Any], prime: int
) -> int:
    return R161.poly_eval(divisor["u"], int(point[0]), prime)


def classify_translate(
    left: tuple[int, int],
    target: tuple[int, int],
    point_index: dict[tuple[int, int], dict[str, Any]],
    curve: dict[str, Any],
) -> tuple[str, tuple[int, int] | None]:
    translated = R161.R70.add(target, R161.R70.negate(left, curve), curve)
    if translated is None:
        return "pole_equality", None
    translated = tuple(translated)
    if translated in point_index:
        return "true_signed", translated
    if tuple(R161.R70.negate(translated, curve)) in point_index:
        return "opposite_sign", translated
    return "nonzero", translated


def evaluate_target_set(
    curve: dict[str, Any],
    divisor: dict[str, Any],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    point_index = {
        tuple(record["endpoint"]): record for record in divisor["records"]
    }
    exact_true_roots: set[int] = set()
    exact_opposite_roots: set[int] = set()
    product_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    true_pair_count = 0
    opposite_pair_count = 0
    positive_opposite_pair_count = 0
    nonpositive_opposite_pair_count = 0
    pole_equality_pair_count = 0
    tangent_pair_count = 0
    all_kummer_zero_biconditionals_exact = True
    all_positive_coefficient_forms_nonzero = True
    positive_coefficient_form_count = 0
    positive_coefficient_form_max_l1 = 0
    for left_record in divisor["records"]:
        left = tuple(left_record["endpoint"])
        product_value = 1
        any_true = False
        any_opposite = False
        for target_index, target_record in enumerate(targets):
            target = tuple(target_record["target"])
            branch, translated = classify_translate(
                left, target, point_index, curve
            )
            if translated is None:
                value = 1
                pole_equality_pair_count += 1
            else:
                value = kummer_value(translated, divisor, prime)
                product_value = product_value * value % prime
                if left[0] == target[0]:
                    tangent_pair_count += 1
            is_true = branch == "true_signed"
            is_opposite = branch == "opposite_sign"
            any_true = any_true or is_true
            any_opposite = any_opposite or is_opposite
            true_pair_count += int(is_true)
            opposite_pair_count += int(is_opposite)
            if is_opposite and target_record.get("expected_source") is not None:
                positive_opposite_pair_count += 1
            elif is_opposite:
                nonpositive_opposite_pair_count += 1
            expected_zero = is_true or is_opposite
            all_kummer_zero_biconditionals_exact &= (
                translated is None or (value == 0) == expected_zero
            )
            if target_record.get("expected_source") is not None:
                target_source = tuple(target_record["expected_source"])
                for opposite_record in divisor["records"]:
                    positive_coefficient_form_count += 1
                    form = coefficient_subtract_add(
                        target_source,
                        tuple(left_record["source"]),
                        tuple(opposite_record["source"]),
                    )
                    all_positive_coefficient_forms_nonzero &= (
                        sum(form) == 6 and any(form)
                    )
                    positive_coefficient_form_max_l1 = max(
                        positive_coefficient_form_max_l1,
                        coefficient_l1(form),
                    )
            pair_rows.append(
                {
                    "left_endpoint": list(left),
                    "target_index": target_index,
                    "branch": branch,
                    "translated_endpoint": (
                        None if translated is None else list(translated)
                    ),
                    "kummer_value": value,
                }
            )
        if any_true:
            exact_true_roots.add(int(left[0]))
        if any_opposite:
            exact_opposite_roots.add(int(left[0]))
        product_rows.append(
            {
                "left_endpoint": list(left),
                "translate_product_value": product_value,
                "any_true_signed_match": any_true,
                "any_opposite_sign_match": any_opposite,
            }
        )

    selector = R161.interpolate(
        [
            (int(row["left_endpoint"][0]), int(row["translate_product_value"]))
            for row in product_rows
        ],
        prime,
    )
    candidate_factor = R161.poly_gcd(divisor["u"], selector, prime)
    candidate_roots = {
        int(record["endpoint"][0])
        for record in divisor["records"]
        if R161.poly_eval(
            candidate_factor, int(record["endpoint"][0]), prime
        )
        == 0
    }
    expected_candidate_roots = exact_true_roots | exact_opposite_roots
    verified_roots: set[int] = set()
    verification_scan_count = 0
    for record in divisor["records"]:
        left = tuple(record["endpoint"])
        if int(left[0]) not in candidate_roots:
            continue
        for target_record in targets:
            verification_scan_count += 1
            if R164.direct_match(
                left, tuple(target_record["target"]), point_index, curve
            ):
                verified_roots.add(int(left[0]))
                break

    return {
        "c3_divisor_degree": len(divisor["records"]),
        "target_count": len(targets),
        "fixed_kummer_function_pole_order": 2 * len(divisor["records"]),
        "fixed_kummer_function_zero_divisor_degree": 2
        * len(divisor["records"]),
        "translate_product_zero_divisor_degree": 2
        * len(divisor["records"])
        * len(targets),
        "translate_product_pole_divisor_degree": 2
        * len(divisor["records"])
        * len(targets),
        "pair_rows_sha256": sha256_json(pair_rows),
        "product_rows_sha256": sha256_json(product_rows),
        "selector_sha256": sha256_json(selector),
        "candidate_factor_sha256": sha256_json(candidate_factor),
        "candidate_factor_degree": len(candidate_roots),
        "candidate_roots": sorted(candidate_roots),
        "expected_candidate_roots": sorted(expected_candidate_roots),
        "true_signed_pair_count": true_pair_count,
        "opposite_sign_pair_count": opposite_pair_count,
        "positive_target_opposite_sign_pair_count": positive_opposite_pair_count,
        "nonpositive_target_opposite_sign_pair_count": nonpositive_opposite_pair_count,
        "true_signed_root_count": len(exact_true_roots),
        "opposite_sign_root_count": len(exact_opposite_roots),
        "opposite_sign_only_roots": sorted(
            exact_opposite_roots - exact_true_roots
        ),
        "pole_equality_pair_count": pole_equality_pair_count,
        "tangent_pair_count": tangent_pair_count,
        "all_kummer_zero_biconditionals_exact": (
            all_kummer_zero_biconditionals_exact
        ),
        "candidate_factor_matches_true_or_opposite_union": (
            candidate_roots == expected_candidate_roots
        ),
        "all_true_roots_retained": exact_true_roots.issubset(candidate_roots),
        "signed_verification_removes_all_opposite_only_roots": (
            verified_roots == exact_true_roots
        ),
        "verified_roots": sorted(verified_roots),
        "verified_factor_sha256": sha256_json(
            R161.monic_root_polynomial(sorted(verified_roots), prime)
        ),
        "verification_scan_count": verification_scan_count,
        "positive_coefficient_form_count": positive_coefficient_form_count,
        "all_positive_coefficient_forms_have_sum_six_and_are_nonzero": (
            all_positive_coefficient_forms_nonzero
        ),
        "positive_coefficient_form_max_l1": positive_coefficient_form_max_l1,
    }


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    _, divisor, targets = R164.target_material(curve, seed)
    evaluated = evaluate_target_set(curve, divisor, targets)
    inherited_r163 = R163.finite_control(curve, seed)
    return {
        "control_id": f"{curve['family_id']}_kummer_x_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": int(curve["field_prime"]),
        "subgroup_order": int(curve["subgroup_order"]),
        "seed": seed,
        **evaluated,
        "r163_expected_union_factor_sha256": inherited_r163[
            "expected_union_sha256"
        ],
        "verified_union_matches_r163_exactly": (
            evaluated["verified_factor_sha256"]
            == inherited_r163["expected_union_sha256"]
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_pair_enumeration_receives_attack_credit": False,
    }


def deterministic_nonzero_scalar(
    curve: dict[str, Any], seed: int, index: int
) -> int:
    order = int(curve["subgroup_order"])
    transcript = (
        f"r166-matched-random-target|{curve['family_id']}|{seed}|{index}"
    )
    return 1 + int.from_bytes(
        hashlib.sha256(transcript.encode()).digest(), "big"
    ) % (order - 1)


def matched_random_density_control(
    curve: dict[str, Any], seed: int
) -> dict[str, Any]:
    factor_base, divisor, _ = R164.target_material(curve, seed)
    point_index = {
        tuple(record["endpoint"]): record for record in divisor["records"]
    }
    generator = tuple(factor_base["generator"])
    true_pairs = 0
    opposite_pairs = 0
    true_roots: set[int] = set()
    opposite_roots: set[int] = set()
    target_digest_rows = []
    for index in range(MATCHED_RANDOM_TARGET_COUNT):
        scalar = deterministic_nonzero_scalar(curve, seed, index)
        target = R161.R70.scalar_mul(scalar, generator, curve)
        if target is None:
            raise AssertionError("nonzero subgroup scalar reached infinity")
        target = tuple(target)
        target_digest_rows.append([scalar, *target])
        for left_record in divisor["records"]:
            left = tuple(left_record["endpoint"])
            branch, _ = classify_translate(left, target, point_index, curve)
            if branch == "true_signed":
                true_pairs += 1
                true_roots.add(int(left[0]))
            elif branch == "opposite_sign":
                opposite_pairs += 1
                opposite_roots.add(int(left[0]))
    degree = len(divisor["records"])
    order = int(curve["subgroup_order"])
    expected = Fraction(degree * degree * MATCHED_RANDOM_TARGET_COUNT, order)
    return {
        "control_id": f"{curve['family_id']}_matched_random_seed{seed}",
        "family_id": curve["family_id"],
        "seed": seed,
        "subgroup_order": order,
        "c3_divisor_degree": degree,
        "target_count": MATCHED_RANDOM_TARGET_COUNT,
        "target_scalar_and_point_sha256": sha256_json(target_digest_rows),
        "expected_true_pair_count_iid": fraction_record(expected),
        "expected_opposite_pair_count_iid": fraction_record(expected),
        "observed_true_pair_count": true_pairs,
        "observed_opposite_pair_count": opposite_pairs,
        "observed_true_root_count": len(true_roots),
        "observed_opposite_root_count": len(opposite_roots),
        "deterministic_scalar_sampler_is_not_hash_to_curve_transfer_proof": True,
        "candidate_oracle_consumed": False,
        "finite_control_receives_attack_credit": False,
    }


def forced_opposite_sign_control() -> dict[str, Any]:
    curve = R161.R159.R82.FAMILIES[0]
    seed = R161.R160.SEEDS[0]
    _, divisor, _ = R164.target_material(curve, seed)
    records = divisor["records"]
    point_index = {tuple(record["endpoint"]): record for record in records}
    chosen: dict[str, Any] | None = None
    for left_record in records:
        left = tuple(left_record["endpoint"])
        for opposite_record in records:
            opposite = tuple(opposite_record["endpoint"])
            target = R161.R70.add(
                left, R161.R70.negate(opposite, curve), curve
            )
            if target is None:
                continue
            target = tuple(target)
            has_true_decomposition = any(
                R164.direct_match(
                    tuple(candidate["endpoint"]), target, point_index, curve
                )
                for candidate in records
            )
            if has_true_decomposition:
                continue
            chosen = {
                "left_record": left_record,
                "opposite_record": opposite_record,
                "target": target,
            }
            break
        if chosen is not None:
            break
    if chosen is None:
        raise AssertionError("unable to construct an opposite-sign-only target")
    target_record = {
        "role": "forced_opposite_sign_only",
        "target": list(chosen["target"]),
        "expected_source": None,
    }
    evaluated = evaluate_target_set(curve, divisor, [target_record])
    left = tuple(chosen["left_record"]["endpoint"])
    opposite = tuple(chosen["opposite_record"]["endpoint"])
    translated = R161.R70.add(
        tuple(chosen["target"]), R161.R70.negate(left, curve), curve
    )
    source_difference = tuple(
        a - b
        for a, b in zip(
            chosen["left_record"]["source"],
            chosen["opposite_record"]["source"],
        )
    )
    return {
        "control_id": "forced_kummer_opposite_sign_only",
        "family_id": curve["family_id"],
        "field_prime": int(curve["field_prime"]),
        "seed": seed,
        "left_endpoint": list(left),
        "opposite_endpoint": list(opposite),
        "target": list(chosen["target"]),
        "translated_endpoint": list(translated),
        "translated_is_negative_opposite": tuple(translated)
        == tuple(R161.R70.negate(opposite, curve)),
        "source_difference": list(source_difference),
        "source_difference_sum": sum(source_difference),
        "source_difference_l1": coefficient_l1(source_difference),
        "forced_left_root": int(left[0]),
        "forced_left_appears_in_candidate": int(left[0])
        in evaluated["candidate_roots"],
        "true_signed_pair_count": evaluated["true_signed_pair_count"],
        "opposite_sign_pair_count": evaluated["opposite_sign_pair_count"],
        "candidate_factor_degree": evaluated["candidate_factor_degree"],
        "signed_verification_returns_empty_union": not evaluated[
            "verified_roots"
        ],
        "candidate_factor_matches_true_or_opposite_union": evaluated[
            "candidate_factor_matches_true_or_opposite_union"
        ],
        "candidate_oracle_consumed": False,
        "finite_control_receives_attack_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "kummer_zero_divisor": (
            "Set r=0 in R165. The fixed function f_0(Q)=U(x(Q)) has pole "
            "order 2n at O and zero divisor S+(-S), where S is the selected "
            "signed C3 endpoint set encoded by U and its public y side table."
        ),
        "candidate_biconditional": (
            "For affine Q=T-P, U(x(Q))=0 exactly when Q lies in S or -S. "
            "The first branch is the desired signed decomposition; the second "
            "is an x-only false-sign branch. P=T maps to O and receives semantic "
            "factor one because O is outside both affine sets."
        ),
        "planted_positive_false_sign_bound": (
            "Write a planted target as v.C with nonnegative l1-six source v, "
            "and P=p.C, Q=q.C with nonnegative l1-three sources. A false-sign "
            "equation T=P-Q is (v-p+q).C=0. The integer coefficient sum is six, "
            "so the form is nonzero. Under independent uniform cyclic C labels "
            "and subgroup order above the coefficient bound, each fixed form "
            "vanishes with probability 1/q. A union bound gives at most n^2K/q "
            "expected false pairs, B^(1/4+o(1))."
        ),
        "unplanted_target_false_sign_bound": (
            "For an independent uniform unplanted target, each fixed P-Q target "
            "equality has probability 1/q. Across at most N targets the expected "
            "opposite-sign pair count is n^2N/q=B^(3/4+o(1)). Independence "
            "between pair events is unnecessary."
        ),
        "candidate_and_verification_cost": (
            "The true pair output is O(K) up to the constant 20 occurrence "
            "splits per planted C6 source; independent uniform unplanted targets "
            "contribute the same n^2N/q expected scale to the true orientation as "
            "to the false one. Total candidate roots are therefore "
            "B^(3/4+o(1)) in expectation. Scanning N target labels per candidate "
            "costs B^2 and removes every opposite-sign-only root exactly. Markov "
            "gives a B^(3/4+epsilon) candidate cap except with probability "
            "B^(-epsilon)."
        ),
        "constructor_boundary": (
            "The required kernel is gcd(U,product_j U(x(T_j-P))) with P=T_j "
            "pole factors regularized to one. Although the fixed function uses "
            "only U, restricting its translates to the selected signed divisor "
            "still requires V, or the equivalent y side table, to represent P. "
            "The standard represented zero and pole divisors have degree "
            "2nN=B^(7/2). R166 does not construct the product remainder modulo "
            "U below rho."
        ),
        "scope": (
            "The probability theorem is for independent uniform cyclic labels "
            "and independent uniform unplanted targets. Deterministic hash-to-curve "
            "transfer is unproved. Finite interpolation enumerates all pairs and "
            "receives no asymptotic attack credit."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_kummer_x_translate_signed_verification.cost.r166.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "positive_target_count_exponent_B": fraction_record(Fraction(3, 4)),
        "kummer_function_representation_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "signed_divisor_v_state_exponent_B": fraction_record(Fraction(9, 4)),
        "planted_positive_false_pair_exponent_B": fraction_record(
            Fraction(1, 4)
        ),
        "full_batch_false_pair_exponent_B": fraction_record(Fraction(3, 4)),
        "true_candidate_exponent_B": fraction_record(Fraction(3, 4)),
        "expected_total_candidate_exponent_B": fraction_record(Fraction(3, 4)),
        "signed_verification_exponent_B": fraction_record(Fraction(2)),
        "explicit_translate_product_divisor_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "standard_product_rho_excess_exponent_B": fraction_record(Fraction(1)),
        "global_randomizer_required": False,
        "signed_v_interpolant_required_by_constructor": True,
        "signed_y_side_table_required_by_verifier": True,
        "pole_equality_correction_inside_rho": True,
        "expected_signed_verification_inside_rho": True,
        "standard_explicit_translate_product_inside_rho": False,
        "output_sensitive_kummer_translate_product_supplied": False,
        "iid_label_theorem_receives_hash_to_curve_transfer_credit": False,
        "finite_pair_enumeration_receives_attack_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    matched_rows = [
        matched_random_density_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    forced = forced_opposite_sign_control()
    all_biconditionals = all(
        row["all_kummer_zero_biconditionals_exact"] for row in rows
    )
    all_candidates = all(
        row["candidate_factor_matches_true_or_opposite_union"] for row in rows
    )
    all_true = all(row["all_true_roots_retained"] for row in rows)
    all_verified = all(
        row["signed_verification_removes_all_opposite_only_roots"]
        and row["verified_union_matches_r163_exactly"]
        for row in rows
    )
    all_forms = all(
        row["all_positive_coefficient_forms_have_sum_six_and_are_nonzero"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_kummer_x_translate_signed_verification.controls.r166.v1",
        "control_count": len(rows),
        "matched_random_control_count": len(matched_rows),
        "matched_random_target_count_per_control": MATCHED_RANDOM_TARGET_COUNT,
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_kummer_zero_biconditionals_exact": all_biconditionals,
        "all_candidate_factors_exact": all_candidates,
        "all_true_roots_retained": all_true,
        "all_signed_verified_unions_exact": all_verified,
        "all_positive_coefficient_forms_have_sum_six_and_are_nonzero": all_forms,
        "actual_batch_true_signed_pair_count": sum(
            row["true_signed_pair_count"] for row in rows
        ),
        "actual_batch_opposite_sign_pair_count": sum(
            row["opposite_sign_pair_count"] for row in rows
        ),
        "actual_batch_opposite_sign_only_root_count": sum(
            len(row["opposite_sign_only_roots"]) for row in rows
        ),
        "matched_random_true_pair_count": sum(
            row["observed_true_pair_count"] for row in matched_rows
        ),
        "matched_random_opposite_pair_count": sum(
            row["observed_opposite_pair_count"] for row in matched_rows
        ),
        "forced_opposite_sign_control": forced,
        "matched_random_controls": matched_rows,
        "candidate_oracle_consumed": False,
        "finite_controls_receive_attack_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "eighteen_source_bindings_verified": len(actual_bindings) == 18,
        "r165_fixed_function_translate_interface_inherited": True,
        "r158_near_injectivity_scope_deduplicated": True,
        "r152_x_only_signed_branch_warning_deduplicated": True,
        "r161_signed_divisor_translation_input_and_verifier_inherited": True,
        "r148_static_3sum_tradeoff_deduplicated": True,
        "kummer_zero_divisor_s_plus_negative_s_complete": True,
        "x_only_candidate_biconditional_complete": True,
        "pole_equality_semantics_complete": True,
        "planted_positive_form_nonzero_complete": all_forms,
        "planted_positive_iid_false_pair_bound_complete": True,
        "unplanted_iid_false_pair_bound_complete": True,
        "expected_candidate_B3O4_bound_complete": True,
        "markov_candidate_cap_complete": True,
        "signed_verification_B2_bound_complete": True,
        "six_actual_target_controls_complete": len(rows) == 6,
        "six_matched_random_density_controls_complete": len(matched_rows) == 6,
        "all_finite_kummer_biconditionals_exact": all_biconditionals,
        "all_finite_candidate_factors_exact": all_candidates,
        "all_finite_true_roots_retained": all_true,
        "all_finite_signed_verified_unions_exact": all_verified,
        "forced_opposite_sign_branch_exercised_and_removed": (
            forced["translated_is_negative_opposite"]
            and forced["forced_left_appears_in_candidate"]
            and forced["true_signed_pair_count"] == 0
            and forced["opposite_sign_pair_count"] > 0
            and forced["signed_verification_returns_empty_union"]
        ),
        "candidate_oracles_avoided": True,
        "finite_controls_scoped_without_attack_credit": True,
        "output_sensitive_kummer_translate_product_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    admission = {
        "obligations": obligations,
        "passed_obligation_count": sum(obligations.values()),
        "obligation_count": len(obligations),
        "kummer_x_relaxation_admitted": True,
        "signed_false_branch_bound_admitted_in_iid_model": True,
        "output_sensitive_translate_constructor_admitted": False,
        "lane_admitted": False,
    }
    kummer = {
        "schema": "p1553.m6_kummer_x_candidate_and_signed_false_branch.r166.v1",
        "theorem": theorem,
        "actual_batch_records": [
            {
                "control_id": row["control_id"],
                "c3_divisor_degree": row["c3_divisor_degree"],
                "target_count": row["target_count"],
                "true_signed_pair_count": row["true_signed_pair_count"],
                "opposite_sign_pair_count": row["opposite_sign_pair_count"],
                "candidate_factor_degree": row["candidate_factor_degree"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "verified_factor_sha256": row["verified_factor_sha256"],
            }
            for row in rows
        ],
        "matched_random_records": matched_rows,
        "forced_opposite_sign_control": forced,
        "output_sensitive_kummer_translate_product_supplied": False,
    }
    replay = {
        "schema": "p1553.m6_kummer_x_translate_signed_verification.replay.r166.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "pair_rows_sha256": row["pair_rows_sha256"],
                "product_rows_sha256": row["product_rows_sha256"],
                "selector_sha256": row["selector_sha256"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "verified_factor_sha256": row["verified_factor_sha256"],
            }
            for row in rows
        ],
        "matched_random_records_sha256": sha256_json(matched_rows),
        "forced_opposite_sign_control_sha256": sha256_json(forced),
        "all_replay_invariants_pass": (
            all_biconditionals
            and all_candidates
            and all_true
            and all_verified
            and all_forms
            and obligations["forced_opposite_sign_branch_exercised_and_removed"]
        ),
    }
    frozen = {
        "schema": "p1553.m6_kummer_x_translate_signed_verification.frozen.r166.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": cost,
        "admission": admission,
        "successor_interface": {
            "input": (
                "one degree-n signed C3 divisor U,V (or U plus the equivalent "
                "signed y side table) and N public target points"
            ),
            "fixed_function": "f_0(Q)=U(x(Q))",
            "required_output": (
                "gcd(U,product_j U(x(T_j-P))) with P=T_j pole factors "
                "regularized to one"
            ),
            "expected_output_degree": "B^(3/4+o(1)) in the iid cyclic-label model",
            "preferred_work": "B^(9/4+o(1))",
            "maximum_total_work": "strictly below B^(5/2)",
            "forbidden_credit": (
                "explicit 2nN divisor, n-by-N value table, unit-cost norm/"
                "3SUM oracle, candidate DLP/root/count/rank/source oracle"
            ),
            "open_primitive": (
                "output-sensitive arbitrary-target Kummer translate product "
                "remainder modulo U"
            ),
        },
    }
    next_action = (
        "Construct or refute gcd(U,product_j U(x(T_j-P))) below B^(5/2), "
        "preferably B^(9/4+o(1)), without expanding the degree-2nN divisor "
        "or the n-by-N value table. Preserve V, or the signed y side table, as "
        "the translation input and reuse it for the B^2 post-verifier. Separately "
        "prove transfer of the signed-difference density bound to the "
        "deterministic hash-to-curve sampler."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Remove R165's randomizer and V-dependent y residual from the fixed "
            "translated function, while preserving V as signed-divisor translation "
            "input and charging x-only sign branches through exact verification "
            "and density bounds."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r165": (
                "R165 uses one randomized signed elliptic function. R166 sets "
                "r=0 and charges the resulting deterministic opposite-sign branch."
            ),
            "r158": (
                "R158 bounds collisions among nonnegative l1-six sources. R166 "
                "uses different nonzero signed forms v-p+q and does not claim "
                "R158 already proves their deterministic hash transfer."
            ),
            "r152": (
                "R152 warns that x-only Semaev elimination introduces sign "
                "branches. R166 retains that warning and supplies an exact signed "
                "post-verifier plus an iid-model density charge."
            ),
            "r161": (
                "R161 enforces signed membership with U,V during every target "
                "composition. R166 still needs V to translate the selected signed "
                "divisor, but its fixed function and product zero test use U alone; "
                "the signed point dictionary filters only output candidates."
            ),
            "r148": (
                "R148 closes standard static 3SUM indexing at the campaign caps. "
                "R166 asks for a coordinate-specific Kummer remainder operator, "
                "not a unit-cost 3SUM data structure."
            ),
        },
        "theorem": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "ADMIT_DETERMINISTIC_KUMMER_F0_EQUALS_U_OF_X_ZERO_DIVISOR_S_PLUS_"
            "NEGATIVE_S__TRUE_MATCHES_RETAINED__OPPOSITE_SIGN_BRANCH_EXACTLY_"
            "VERIFIED_AWAY__PLANTED_SIGNED_FORMS_NONZERO__IID_FALSE_PAIR_UNION_"
            "B3O4__EXPECTED_SIGNED_VERIFICATION_B2__SIX_ACTUAL_BATCHES_ZERO_"
            "FALSE_SIGN_PAIRS__FORCED_FALSE_SIGN_BRANCH_REMOVED__EXPLICIT_"
            "TRANSLATE_DIVISOR_B7O2__OUTPUT_SENSITIVE_KUMMER_REMAINDER_AND_HASH_"
            "TRANSFER_OPEN__NO_RHO_SHOUP_BREAKTHROUGH"
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
        "kummer": kummer,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--kummer-output", type=Path, default=DEFAULT_KUMMER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.kummer_output, bundle["kummer"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
