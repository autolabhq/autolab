#!/usr/bin/env python3
"""Audit linear and regular-section realizations of the R116 transpose."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Mapping, Sequence


SCHEMA = "p1553.m6_target_batched_c3_elliptic_transpose.r117.v1"
LOG_B_GROUP_ORDER = Fraction(5)
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
RHO_EXPONENT = Fraction(5, 2)
ALPHA = Fraction(1, 12)
BETA = Fraction(3, 4)
A6_EXPONENT = 6 * ALPHA
C3_EXPONENT = 3 * BETA
C_DECK_EXPONENT = BETA
FACTORIAL_6 = math.factorial(6)
CANONICAL_SOURCE_CONSTANT_DENOMINATOR = FACTORIAL_6**2

R116_PRODUCER = pathlib.Path(
    "p1553_m6_a6_batched_c3_pair_sum_source_locator_probe_r116.py"
)
R116_PRODUCER_SHA256 = (
    "dcaf00bf51a2a8757a12e407cbe581f45f58c54895623f857d996898a5835f46"
)
R116_REPORT = pathlib.Path(
    "p1553_m6_a6_batched_c3_pair_sum_source_"
    "locator_probe_report_r116.json"
)
R116_REPORT_SHA256 = (
    "9c5ebb9e99eaada2296c3d4f0fcbdb472995f3ae071bb68138e364e266f93d25"
)
R116_FROZEN = pathlib.Path("frozen_m6_c3_pair_sum_batch_interface.json")
R116_FROZEN_SHA256 = (
    "d2e8038408ddc46223b2fb725403791ef448b2ae51b60c0076b0d39c84ae1302"
)
R116_COST = pathlib.Path(
    "c3_pair_sum_indexing_and_algebraic_cost_ledger.json"
)
R116_COST_SHA256 = (
    "f128638edd6d6bbe523e15f47a07bed6b792860bbd3ea3fa1bcf3b3544e0421d"
)
R116_REPLAY = pathlib.Path("m6_a6_c3_self_convolution_source_replay.json")
R116_REPLAY_SHA256 = (
    "eba70be7b0ca9829fb18c9646e5a10c303b8d22452e3d0098633d43fe7400ae1"
)
R116_LOGS = pathlib.Path("factor_logs_and_identical_descent_r116.json")
R116_LOGS_SHA256 = (
    "1d3bfc1455bf0f82b02d4875d6278ab50501e0fef25617672f58427ac36deec4"
)
R116_GATE = pathlib.Path(
    "p1553_m6_a6_batched_c3_pair_sum_source_locator_probe_gate_r116.md"
)
R116_GATE_SHA256 = (
    "1a17b8e396affe9ced0a5d589059ec6d3d000ee23f56522e1f55e353bac72542"
)
R116_PARENT = pathlib.Path(
    "p1553_m6_a6_batched_c3_pair_sum_"
    "source_locator_probe_parent_report_r116.yaml"
)
R116_PARENT_SHA256 = (
    "50d5439738e7ba815ddb7bab07fb8d03000bb9a2af42b5e99bbca9009e08760b"
)
R82_PRODUCER = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_r82.py"
)
R82_PRODUCER_SHA256 = (
    "7380bff3175625016affee4703b0b0f2867a28113f72eef2d90614ed57ffef07"
)
R77_REPORT = pathlib.Path(
    "p1553_target_translated_frequency_orbit_probe_report_r77.json"
)
R77_REPORT_SHA256 = (
    "73f66184fa53a2d43397a915c4249a41c5687cbcd1d5d16cc3e4ff47cf254787"
)
R77_GATE = pathlib.Path(
    "p1553_target_translated_frequency_orbit_probe_gate_r77.md"
)
R77_GATE_SHA256 = (
    "45324f816cc159032cb6c2ac97c0a2f52a618c409e276616521b4e894cb46b55"
)
DINUR_GOLOVNEV = pathlib.Path(
    "references/dinur_golovnev_3sum_indexing_2512.04258v2.pdf"
)
DINUR_GOLOVNEV_SHA256 = (
    "e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c"
)

Point = tuple[int, int] | None


def load_r82() -> Any:
    spec = importlib.util.spec_from_file_location("p1553_r82_for_r117", R82_PRODUCER)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R82 finite-curve controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R82 = load_r82()
R81 = R82.R81
R70 = R82.R70


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r116_producer", R116_PRODUCER, R116_PRODUCER_SHA256),
        ("r116_report", R116_REPORT, R116_REPORT_SHA256),
        ("r116_frozen", R116_FROZEN, R116_FROZEN_SHA256),
        ("r116_cost", R116_COST, R116_COST_SHA256),
        ("r116_replay", R116_REPLAY, R116_REPLAY_SHA256),
        ("r116_logs", R116_LOGS, R116_LOGS_SHA256),
        ("r116_gate", R116_GATE, R116_GATE_SHA256),
        ("r116_parent", R116_PARENT, R116_PARENT_SHA256),
        ("r82_producer", R82_PRODUCER, R82_PRODUCER_SHA256),
        ("r77_report", R77_REPORT, R77_REPORT_SHA256),
        ("r77_gate", R77_GATE, R77_GATE_SHA256),
        ("dinur_golovnev_v2", DINUR_GOLOVNEV, DINUR_GOLOVNEV_SHA256),
    )
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in rows
    }


def verify_source_bindings() -> dict[str, str]:
    bindings = source_binding_records()
    actual = {
        name: sha256_file(pathlib.Path(binding["path"]))
        for name, binding in bindings.items()
    }
    failures = [
        name
        for name, binding in bindings.items()
        if actual[name] != binding["sha256"]
    ]
    if failures:
        raise AssertionError(f"R117 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def scalar_label_map(
    generator: Point,
    curve: dict[str, Any],
) -> dict[Point, int]:
    labels: dict[Point, int] = {}
    for scalar in range(curve["subgroup_order"]):
        point = R70.scalar_mul(scalar, generator, curve)
        if point in labels:
            raise AssertionError("subgroup walk repeated before its stated order")
        labels[point] = scalar
    if len(labels) != curve["subgroup_order"]:
        raise AssertionError("subgroup label map is incomplete")
    return labels


def convolve_by_deck(
    values: Sequence[int],
    deck_labels: Sequence[int],
) -> list[int]:
    modulus = len(values)
    result = [0] * modulus
    for index, value in enumerate(values):
        if value == 0:
            continue
        for label in deck_labels:
            result[(index + label) % modulus] += value
    return result


def ordered_six_plus_six_counts(
    order: int,
    labels_a: Sequence[int],
    labels_c: Sequence[int],
) -> list[int]:
    values = [0] * order
    values[0] = 1
    for _ in range(6):
        values = convolve_by_deck(values, labels_a)
    for _ in range(6):
        values = convolve_by_deck(values, labels_c)
    expected_mass = len(labels_a) ** 6 * len(labels_c) ** 6
    if sum(values) != expected_mass:
        raise AssertionError("ordered six-plus-six mass drifted")
    return values


def canonical_endpoint_counts(
    order: int,
    labels_a: Sequence[int],
    labels_c: Sequence[int],
) -> collections.Counter[int]:
    a6 = collections.Counter(
        sum(labels_a[index] for index in source) % order
        for source in itertools.combinations_with_replacement(
            range(len(labels_a)),
            6,
        )
    )
    c6 = collections.Counter(
        sum(labels_c[index] for index in source) % order
        for source in itertools.combinations_with_replacement(
            range(len(labels_c)),
            6,
        )
    )
    result: collections.Counter[int] = collections.Counter()
    for left, left_count in a6.items():
        for right, right_count in c6.items():
            result[(left + right) % order] += left_count * right_count
    expected_mass = math.comb(len(labels_a) + 5, 6) * math.comb(
        len(labels_c) + 5,
        6,
    )
    if sum(result.values()) != expected_mass:
        raise AssertionError("canonical six-plus-six mass drifted")
    return result


def dft_value(
    values: Mapping[int, int],
    frequency: int,
    root: int,
    modulus: int,
) -> int:
    base = pow(root, frequency, modulus)
    return sum(
        (value % modulus) * pow(base, index, modulus)
        for index, value in values.items()
    ) % modulus


def deck_dft_value(
    labels: Sequence[int],
    frequency: int,
    root: int,
    modulus: int,
) -> int:
    base = pow(root, frequency, modulus)
    return sum(pow(base, label, modulus) for label in labels) % modulus


def finite_translation_control(
    *,
    limit_a: int,
    limit_c: int,
) -> dict[str, Any]:
    curve = R82.FAMILIES[0]
    generator = R81.curve_generator(curve)
    validation = R82.validate_family(curve, generator)
    if not all(validation.values()):
        raise AssertionError("R117 finite curve validation failed")
    labels = scalar_label_map(generator, curve)
    atoms_a, atoms_c, _, construction = R82.compact_factor_base(curve, 0)
    atoms_a = atoms_a[:limit_a]
    atoms_c = atoms_c[:limit_c]
    labels_a = [labels[point] for point in atoms_a]
    labels_c = [labels[point] for point in atoms_c]
    order = curve["subgroup_order"]

    auxiliary_prime = 6 * order + 1
    primitive_root = 2
    root = pow(
        primitive_root,
        (auxiliary_prime - 1) // order,
        auxiliary_prime,
    )
    if (
        not R70.is_prime(auxiliary_prime)
        or pow(root, order, auxiliary_prime) != 1
        or root == 1
    ):
        raise AssertionError("invalid auxiliary DFT field")

    ordered = ordered_six_plus_six_counts(order, labels_a, labels_c)
    ordered_sparse = {
        index: value for index, value in enumerate(ordered) if value
    }
    canonical = canonical_endpoint_counts(order, labels_a, labels_c)
    if set(ordered_sparse) != set(canonical):
        raise AssertionError("ordered and canonical endpoint supports differ")

    deck_a_zero_frequencies = []
    deck_c_zero_frequencies = []
    factorization_failures = []
    direct_zero_frequencies = []
    for frequency in range(order):
        a_hat = deck_dft_value(
            labels_a,
            frequency,
            root,
            auxiliary_prime,
        )
        c_hat = deck_dft_value(
            labels_c,
            frequency,
            root,
            auxiliary_prime,
        )
        direct = dft_value(
            ordered_sparse,
            frequency,
            root,
            auxiliary_prime,
        )
        factored = (
            pow(a_hat, 6, auxiliary_prime)
            * pow(c_hat, 6, auxiliary_prime)
        ) % auxiliary_prime
        if a_hat == 0:
            deck_a_zero_frequencies.append(frequency)
        if c_hat == 0:
            deck_c_zero_frequencies.append(frequency)
        if direct == 0:
            direct_zero_frequencies.append(frequency)
        if direct != factored:
            factorization_failures.append(frequency)

    support_size = len(canonical)
    zero_target_count = order - support_size
    canonical_source_count = sum(canonical.values())
    expected_canonical_sources = math.comb(limit_a + 5, 6) * math.comb(
        limit_c + 5,
        6,
    )
    if canonical_source_count != expected_canonical_sources:
        raise AssertionError("canonical source count mismatch")
    if factorization_failures:
        raise AssertionError("finite convolution DFT factorization failed")
    if direct_zero_frequencies:
        raise AssertionError("finite target orbit lost a character mode")

    return {
        "control_id": f"r82_q{order}_u{limit_a}_v{limit_c}",
        "curve": {
            "field_prime": curve["field_prime"],
            "subgroup_order": order,
            "cofactor": curve["cofactor"],
        },
        "validation": validation,
        "factor_base_construction": construction["construction"],
        "deck_labels_for_verifier_only": {
            "a": labels_a,
            "c": labels_c,
            "candidate_algorithm_receives_labels": False,
        },
        "auxiliary_dft_field": {
            "prime": auxiliary_prime,
            "primitive_root": primitive_root,
            "qth_root": root,
            "qth_root_exact_order": True,
            "integer_mass_below_modulus": sum(ordered) < auxiliary_prime,
        },
        "ordered_source_count": sum(ordered),
        "canonical_source_count": canonical_source_count,
        "canonical_source_formula": (
            f"binom({limit_a}+5,6)*binom({limit_c}+5,6)"
        ),
        "target_support_size": support_size,
        "zero_target_count": zero_target_count,
        "regular_boolean_section_minimum_zero_divisor_degree": (
            zero_target_count
        ),
        "regular_boolean_section_minimum_pole_degree": zero_target_count,
        "ordered_and_canonical_support_equal": True,
        "deck_a_zero_frequency_count": len(deck_a_zero_frequencies),
        "deck_c_zero_frequency_count": len(deck_c_zero_frequencies),
        "target_count_zero_frequency_count": len(direct_zero_frequencies),
        "dft_factorization_failure_count": len(factorization_failures),
        "translation_orbit_rank_over_auxiliary_field": (
            order - len(direct_zero_frequencies)
        ),
        "full_translation_orbit_rank": not direct_zero_frequencies,
        "finite_labels_and_enumeration_receive_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        finite_translation_control(limit_a=1, limit_c=2),
        finite_translation_control(limit_a=2, limit_c=3),
    ]
    return {
        "schema": (
            "p1553.m6_target_batched_c3_exceptional_controls.r117.v1"
        ),
        "controls": controls,
        "all_dft_factorizations_exact": all(
            row["dft_factorization_failure_count"] == 0
            for row in controls
        ),
        "all_translation_orbits_full_rank": all(
            row["full_translation_orbit_rank"] for row in controls
        ),
        "sparse_support_full_rank_control_present": any(
            row["target_support_size"] * 10 < row["curve"]["subgroup_order"]
            and row["full_translation_orbit_rank"]
            for row in controls
        ),
        "finite_labels_and_enumeration_receive_asymptotic_credit": False,
    }


def indexing_and_section_ledger() -> dict[str, Any]:
    n_exponent = C_DECK_EXPONENT
    ksum_delta = Fraction(1)
    ksum_state = n_exponent * (
        Fraction(13, 2) - ksum_delta
    )
    ksum_query = n_exponent * ksum_delta
    ksum_batch = A6_EXPONENT + ksum_query
    trivial_state = 5 * n_exponent
    canonical_ratio = Fraction(
        1,
        CANONICAL_SOURCE_CONSTANT_DENOMINATOR,
    )
    rows = [
        {
            "route_id": "universal_characteristic_zero_linear_shift_sketch",
            "state_exponent_B": fraction_record(LOG_B_GROUP_ORDER),
            "inside_setup_cap": False,
            "theorem": "prime-cyclotomic deck DFT nonvanishing",
            "scope": "exact characteristic-zero linear shift-equivariant state",
        },
        {
            "route_id": "regular_rational_boolean_target_section",
            "zero_divisor_degree_exponent_B": fraction_record(
                LOG_B_GROUP_ORDER
            ),
            "pole_degree_exponent_B": fraction_record(LOG_B_GROUP_ORDER),
            "inside_setup_if_materialized": False,
            "arithmetic_circuit_lower_bound_claimed": False,
        },
        {
            "route_id": "dinur_golovnev_k7_index_on_original_c_deck",
            "k": 7,
            "delta": fraction_record(ksum_delta),
            "state_exponent_B": fraction_record(ksum_state),
            "query_exponent_B": fraction_record(ksum_query),
            "a6_batch_exponent_B": fraction_record(A6_EXPONENT),
            "fresh_batch_exponent_B": fraction_record(ksum_batch),
            "inside_setup_cap": ksum_state <= SETUP_CAP,
            "inside_fresh_batch_cap": ksum_batch <= ONLINE_CAP,
            "integer_residue_transfer_to_prime_order_ec": False,
        },
        {
            "route_id": "trivial_k7_store_five_c_sums",
            "state_exponent_B": fraction_record(trivial_state),
            "query_exponent_B": fraction_record(n_exponent),
            "fresh_batch_exponent_B": fraction_record(
                A6_EXPONENT + n_exponent
            ),
            "inside_setup_cap": trivial_state <= SETUP_CAP,
            "inside_fresh_batch_cap": (
                A6_EXPONENT + n_exponent <= ONLINE_CAP
            ),
            "exact_source_reporting": True,
        },
        {
            "route_id": "r116_equal_c3_index_online_boundary",
            "state_exponent_B": fraction_record(Fraction(39, 8)),
            "query_exponent_B": fraction_record(Fraction(3, 4)),
            "inside_setup_cap": False,
            "inside_fresh_batch_cap": True,
        },
        {
            "route_id": "r116_explicit_target_translated_divisor_batch",
            "fresh_batch_exponent_B": fraction_record(Fraction(11, 4)),
            "inside_fresh_batch_cap": False,
        },
    ]
    return {
        "schema": (
            "p1553.m6_target_batched_c3_transpose_cost_ledger.r117.v1"
        ),
        "normalization": {
            "group_order": "q=B^(5+o(1)) prime",
            "a_deck": "|A|=B^(1/12+o(1))",
            "c_deck": "|C|=B^(3/4+o(1))",
            "a6_batch": "B^(1/2+o(1))",
            "c3_state": "B^(9/4+o(1))",
        },
        "prime_cyclotomic_translation_rank_theorem": {
            "statement": (
                "For prime q and every nonempty proper D subset Z/qZ, "
                "sum_(d in D) zeta^(r*d) is nonzero for every qth "
                "character r over characteristic zero."
            ),
            "proof": (
                "A vanishing proper 0/1 deck polynomial at a primitive "
                "qth root would be divisible by Phi_q=1+X+...+X^(q-1), "
                "forcing all q coefficients equal."
            ),
            "six_factor_fourier_identity": (
                "hat(h)(chi)=hat(mu_A)(chi)^6*hat(mu_C)(chi)^6"
            ),
            "translation_orbit_rank": "q",
            "state_exponent_B": fraction_record(LOG_B_GROUP_ORDER),
            "extends_r77_finite_controls": True,
            "base_field_linear_rank_theorem_claimed": False,
        },
        "canonical_source_support_theorem": {
            "canonical_source_count": (
                "binom(|A|+5,6)*binom(|C|+5,6)"
            ),
            "leading_ratio_to_group_order": fraction_record(canonical_ratio),
            "leading_denominator": CANONICAL_SOURCE_CONSTANT_DENOMINATOR,
            "support_at_most_canonical_source_count": True,
            "zero_target_fraction_lower_bound": (
                "1-1/(6!)^2+o(1)"
            ),
            "regular_nonzero_boolean_section_zero_degree": (
                "Omega(q)"
            ),
            "regular_section_pole_degree_exponent_B": fraction_record(
                LOG_B_GROUP_ORDER
            ),
            "straight_line_circuit_lower_bound_claimed": False,
        },
        "dinur_golovnev_k7": {
            "theorem": (
                "S=soft-O(n^(k-1/2-delta)), T=soft-O(n^delta), "
                "0<=delta<=1"
            ),
            "n": "|C|=B^(3/4+o(1))",
            "k": 7,
            "online_compatible_delta": fraction_record(ksum_delta),
            "state_exponent_B": fraction_record(ksum_state),
            "source": str(DINUR_GOLOVNEV),
            "sha256": DINUR_GOLOVNEV_SHA256,
        },
        "routes": rows,
        "any_scoped_route_meets_both_caps": any(
            row.get("inside_setup_cap", False)
            and row.get("inside_fresh_batch_cap", False)
            for row in rows
        ),
        "unconditional_data_structure_lower_bound_claimed": False,
        "candidate_work_credit": False,
    }


def source_adjoint_replay() -> dict[str, Any]:
    inherited = json.loads(R116_REPLAY.read_text(encoding="utf-8"))
    if not inherited.get("all_instances_exact"):
        raise AssertionError("R116 source replay is not exact")
    return {
        "schema": (
            "p1553.m6_target_batched_c3_source_adjoint_replay.r117.v1"
        ),
        "inherited_r116_replay": {
            "path": str(R116_REPLAY),
            "sha256": R116_REPLAY_SHA256,
            "instance_count": len(inherited["instances"]),
            "all_instances_exact": inherited["all_instances_exact"],
            "negative_control_present": inherited["negative_control_present"],
        },
        "requested_source": "one A6 and two C3 occurrence backpointers",
        "linear_translation_sketch_source_adjoint_constructed": False,
        "regular_rational_section_source_adjoint_constructed": False,
        "nonlinear_value_sensitive_source_locator_constructed": False,
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    controls = finite_controls()
    cost = indexing_and_section_ledger()
    replay = source_adjoint_replay()
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r116_exact_source_semantics_inherited": replay[
            "inherited_r116_replay"
        ]["all_instances_exact"],
        "finite_integer_convolution_dft_factorization_exact": controls[
            "all_dft_factorizations_exact"
        ],
        "finite_full_translation_rank_exact": controls[
            "all_translation_orbits_full_rank"
        ],
        "finite_sparse_support_full_rank_control_present": controls[
            "sparse_support_full_rank_control_present"
        ],
        "prime_cyclotomic_nonvanishing_theorem_complete": True,
        "canonical_source_support_bound_complete": True,
        "regular_section_zero_pole_degree_bound_complete": True,
        "original_c_deck_k7_indexing_cost_charged": (
            cost["dinur_golovnev_k7"]["state_exponent_B"]["exact"]
            == "33/8"
        ),
        "scope_excludes_general_circuit_and_data_structure_lower_bound": (
            not cost["unconditional_data_structure_lower_bound_claimed"]
        ),
        "nonlinear_value_sensitive_source_locator_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one nonlinear value-sensitive six-C source "
        "locator on the original C deck. It may retain the B^(9/4) C3 "
        "preprocessing and spend B^(3/4+o(1)) per A6 target, but it must "
        "not be a universal linear shift sketch, a single regular "
        "target-section materialization, the B^(33/8) bound k=7 index, "
        "or the B^(11/4) translated-divisor batch. Freeze every branch, "
        "S7/FFE remainder or subresultant dimension, false-positive "
        "verification, and one six-C source backpointer; then compose with "
        "A6 and pass the same relation/rank/log/descent pipeline."
    )
    frozen = {
        "schema": (
            "p1553.frozen_m6_target_batched_c3_elliptic_transpose.r117.v1"
        ),
        "source_bindings": source_binding_records(),
        "r116_interface": {
            "coefficient": (
                "sum_a6 mu_A6(a6)(mu_C3*mu_C3)(R-a6)"
            ),
            "setup_exponent_B": fraction_record(C3_EXPONENT),
            "a6_batch_exponent_B": fraction_record(A6_EXPONENT),
            "average_query_allowance_exponent_B": fraction_record(
                ONLINE_CAP - A6_EXPONENT
            ),
        },
        "closed_scoped_grammars": [
            "universal exact characteristic-zero linear shift sketch",
            "single regular rational boolean target section by pole degree",
            "bound integer k=7 indexing theorem on the original C deck",
            "standard explicit translated C3 divisor batch",
        ],
        "preserved_interface": (
            "nonlinear value-sensitive target-specialized six-C source "
            "locator with branching and exact backpointers"
        ),
        "general_lower_bound_claimed": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r117.v1",
        "r116_exact_interface_reduction_complete": True,
        "r117_scoped_linear_and_regular_section_audit_complete": True,
        "nonlinear_value_sensitive_source_locator_complete": False,
        "relation_independence_theorem_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_log_solve_complete": False,
        "factor_log_verification_complete": False,
        "fresh_target_descent_complete": False,
        "identical_algorithm_used_for_relation_and_descent": False,
        "full_source_to_target_cost_complete": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "SCOPED_LINEAR_AND_REGULAR_SECTION_NEGATIVE_ONLY_"
            "WITHHOLD_PROMOTION"
        ),
        "classification": (
            "PRIME_CYCLOTOMIC_THEOREM_FORCES_FULL_B5_CHARACTERISTIC_ZERO_"
            "TRANSLATION_RANK_FOR_SIX_FACTOR_COUNT__CANONICAL_SOURCE_SUPPORT_"
            "LEAVES_1_MINUS_1O518400_ZERO_FRACTION_AND_REGULAR_SECTION_POLE_"
            "DEGREE_B5__FINITE_R82_AUXILIARY_FIELD_DFT_AND_SPARSE_FULL_RANK_"
            "CONTROLS_EXACT__ORIGINAL_C_DECK_K7_INDEX_AT_ONLINE_DELTA1_NEEDS_"
            "B33O8_STATE__GENERAL_BASE_FIELD_NONLINEAR_DATA_STRUCTURE_AND_"
            "CIRCUIT_LOWER_BOUND_NOT_CLAIMED__VALUE_SENSITIVE_SOURCE_LOCATOR_"
            "RANK_LOGS_DESCENT_OPEN"
        ),
        "source_bindings": source_binding_records(),
        "theorem_boundary": {
            "translation_rank": cost[
                "prime_cyclotomic_translation_rank_theorem"
            ],
            "canonical_support": cost[
                "canonical_source_support_theorem"
            ],
            "k7_indexing": cost["dinur_golovnev_k7"],
        },
        "finite_evidence": {
            "control_count": len(controls["controls"]),
            "all_dft_factorizations_exact": controls[
                "all_dft_factorizations_exact"
            ],
            "all_translation_orbits_full_rank": controls[
                "all_translation_orbits_full_rank"
            ],
            "sparse_support_full_rank_control_present": controls[
                "sparse_support_full_rank_control_present"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "scoped_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_m6_target_batched_c3_elliptic_transpose.json",
            "cost": "m6_target_batched_c3_transpose_cost_ledger.json",
            "source_replay": (
                "m6_target_batched_c3_source_adjoint_replay.json"
            ),
            "exceptional_controls": (
                "m6_target_batched_c3_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r117.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Auxiliary-field scalar labels are verifier-only.",
            "Characteristic-zero translation rank is not a base-field circuit lower bound.",
            "Pole degree is not straight-line-program size.",
            "Current kSUM indexing failure is not a data-structure lower bound.",
            "No nonlinear source locator, relation rank, logs, or descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "REJECT_UNIVERSAL_CHARACTERISTIC_ZERO_LINEAR_SHIFT_SKETCH_AND_"
            "SINGLE_REGULAR_RATIONAL_TARGET_SECTION_ONLY__FINITE_AUXILIARY_"
            "DFT_FULL_RANK_AND_SOURCE_SUPPORT_CONTROLS_EXACT__REJECT_BOUND_"
            "K7_INTEGER_INDEX_AND_STANDARD_TRANSLATED_DIVISOR_AT_FROZEN_"
            "CAPS__PRESERVE_NONLINEAR_VALUE_SENSITIVE_SIX_C_SOURCE_LOCATOR__"
            "NO_LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_"
            "CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_m6_target_batched_c3_elliptic_"
            "transpose_probe_report_r117.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_m6_target_batched_c3_elliptic_transpose.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_target_batched_c3_transpose_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_target_batched_c3_source_adjoint_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_target_batched_c3_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r117.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs_descent"])
    report = bundle["report"]
    admission = report["admission"]
    print(
        f"R117 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
