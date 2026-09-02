#!/usr/bin/env python3
"""Audit algebraic-character and pairing routes for implicit C5 membership."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.m6_suboutput_implicit_c5_character_pairing.r120.v1"
SETUP_CAP = Fraction(9, 4)
OUTER_BATCH_EXPONENT = Fraction(5, 4)
C_ATOM_EXPONENT = Fraction(3, 4)
C5_SUPPORT_EXPONENT = 5 * C_ATOM_EXPONENT

R119_PRODUCER = pathlib.Path(
    "p1553_m6_output_sensitive_nonlinear_c5_source_index_probe_r119.py"
)
R119_PRODUCER_SHA256 = (
    "99bd176b07eb119edad2ddbb6a2ff890b1109f1ef0a8cc26af061279e7bca578"
)
R119_REPORT = pathlib.Path(
    "p1553_m6_output_sensitive_nonlinear_c5_"
    "source_index_probe_report_r119.json"
)
R119_REPORT_SHA256 = (
    "1ff46d9c7a66ade7e617fc5523bf69d125ba1a102cc360b3cdee2ed0df1f75a9"
)
R119_FROZEN = pathlib.Path(
    "frozen_m6_output_sensitive_c5_source_index.json"
)
R119_FROZEN_SHA256 = (
    "9ba25b63d0dfb3dd2a4be31aeb3b2d0fba3624e5b3d3fcdfdd2c4be9b2abccda"
)
R119_COST = pathlib.Path("m6_c5_membership_ffe_cost_ledger.json")
R119_COST_SHA256 = (
    "933a3ad858ddb68b89a4328a2d404bca4df3a20b21fbac9f41a3514f75308c5c"
)
R119_REPLAY = pathlib.Path("m6_c5_membership_source_replay.json")
R119_REPLAY_SHA256 = (
    "408066047d2401dce4f57382b3231cda04667fd80599e65a5aa106b8c2af867e"
)
R119_CONTROLS = pathlib.Path(
    "m6_c5_membership_exceptional_controls.json"
)
R119_CONTROLS_SHA256 = (
    "06df2f774602cf9f467cfd08db68d4df73edb227cb3e55aa12914d8d0132afae"
)
R119_LOGS = pathlib.Path("factor_logs_and_identical_descent_r119.json")
R119_LOGS_SHA256 = (
    "e1b6a3e891b8ad5e1ce636326992ed5d0a9fa0a0c9ab6d4bda1c0971a1b0ed3a"
)
R119_GATE = pathlib.Path(
    "p1553_m6_output_sensitive_nonlinear_c5_source_index_probe_gate_r119.md"
)
R119_GATE_SHA256 = (
    "941e375918c4acd1be8293fcc40666879b4f5178b4454e28c21487cb9be9a9e9"
)
R119_PARENT = pathlib.Path(
    "p1553_m6_output_sensitive_nonlinear_c5_"
    "source_index_probe_parent_report_r119.yaml"
)
R119_PARENT_SHA256 = (
    "f1aceb3511e275dcfd6b3d43d76d24eab3de78b61ff42de7e5e11aee00645eea"
)
R82_PRODUCER = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_r82.py"
)
R82_PRODUCER_SHA256 = (
    "7380bff3175625016affee4703b0b0f2867a28113f72eef2d90614ed57ffef07"
)
PAIRING_REGISTRY = pathlib.Path("p1553_r23_artifact_index_README.md")
PAIRING_REGISTRY_SHA256 = (
    "8efff2f8ae6a85fdf7a2dd6ae364e0e8f86f6e165d6a4c7fb4edbabed9c223a5"
)
ENGE_PAIRING_PDF = pathlib.Path(
    "references/enge_bilinear_pairings_1301.5520v2.pdf"
)
ENGE_PAIRING_PDF_SHA256 = (
    "93b99fa2d13e09c1bc8282b58d472be6285ceea9b594e9b97d765212a3aaa8e4"
)
MILLER_PAIRING_PDF = pathlib.Path(
    "references/miller_weil_pairing_algorithm_1986.pdf"
)
MILLER_PAIRING_PDF_SHA256 = (
    "39c76c7643278b87b3d8c24b9a07d0b4cbfb561cd13735548990848e0f0bd166"
)

Fp2 = tuple[int, int]
Point = tuple[int, int] | None
Point2 = tuple[Fp2, Fp2] | None


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R119 = load_module("p1553_r119_for_r120", R119_PRODUCER)
R118 = R119.R118
R117 = R119.R117
R82 = R119.R82
R70 = R119.R70


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r119_producer", R119_PRODUCER, R119_PRODUCER_SHA256),
        ("r119_report", R119_REPORT, R119_REPORT_SHA256),
        ("r119_frozen", R119_FROZEN, R119_FROZEN_SHA256),
        ("r119_cost", R119_COST, R119_COST_SHA256),
        ("r119_replay", R119_REPLAY, R119_REPLAY_SHA256),
        ("r119_controls", R119_CONTROLS, R119_CONTROLS_SHA256),
        ("r119_logs", R119_LOGS, R119_LOGS_SHA256),
        ("r119_gate", R119_GATE, R119_GATE_SHA256),
        ("r119_parent", R119_PARENT, R119_PARENT_SHA256),
        ("r82_producer", R82_PRODUCER, R82_PRODUCER_SHA256),
        ("pairing_registry", PAIRING_REGISTRY, PAIRING_REGISTRY_SHA256),
        ("enge_pairing_pdf", ENGE_PAIRING_PDF, ENGE_PAIRING_PDF_SHA256),
        ("miller_pairing_pdf", MILLER_PAIRING_PDF, MILLER_PAIRING_PDF_SHA256),
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
        raise AssertionError(f"R120 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def multiplicative_order(value: int, modulus: int) -> int:
    if math.gcd(value, modulus) != 1:
        raise ValueError("multiplicative order requires a unit")
    current = 1
    for order in range(1, modulus):
        current = current * value % modulus
        if current == 1:
            return order
    raise AssertionError("unit order did not divide modulus-1")


class QuadraticField:
    """Fp2 with u^2=-3 for the j=0 distortion controls."""

    def __init__(self, prime: int) -> None:
        self.p = prime
        self.nonsquare = (-3) % prime
        if pow(self.nonsquare, (prime - 1) // 2, prime) != prime - 1:
            raise AssertionError("-3 must be nonsquare in the R82 fields")
        self.zero = (0, 0)
        self.one = (1, 0)

    def elt(self, value: int | Fp2) -> Fp2:
        if isinstance(value, int):
            return (value % self.p, 0)
        return (value[0] % self.p, value[1] % self.p)

    def add(self, left: Fp2, right: Fp2) -> Fp2:
        return (
            (left[0] + right[0]) % self.p,
            (left[1] + right[1]) % self.p,
        )

    def neg(self, value: Fp2) -> Fp2:
        return ((-value[0]) % self.p, (-value[1]) % self.p)

    def sub(self, left: Fp2, right: Fp2) -> Fp2:
        return self.add(left, self.neg(right))

    def mul(self, left: Fp2, right: Fp2) -> Fp2:
        a, b = left
        c, d = right
        return (
            (a * c + b * d * self.nonsquare) % self.p,
            (a * d + b * c) % self.p,
        )

    def square(self, value: Fp2) -> Fp2:
        return self.mul(value, value)

    def inv(self, value: Fp2) -> Fp2:
        a, b = value
        denominator = (
            a * a - self.nonsquare * b * b
        ) % self.p
        if denominator == 0:
            raise ZeroDivisionError("Fp2 inverse of zero")
        inverse = pow(denominator, self.p - 2, self.p)
        return (a * inverse % self.p, -b * inverse % self.p)

    def div(self, left: Fp2, right: Fp2) -> Fp2:
        return self.mul(left, self.inv(right))

    def pow(self, value: Fp2, exponent: int) -> Fp2:
        result = self.one
        base = value
        power = exponent
        while power:
            if power & 1:
                result = self.mul(result, base)
            base = self.square(base)
            power >>= 1
        return result

    def product(self, values: Iterable[Fp2]) -> Fp2:
        result = self.one
        for value in values:
            result = self.mul(result, value)
        return result

    def json(self, value: Fp2) -> list[int]:
        return [value[0], value[1]]


def point2(point: Point, field: QuadraticField) -> Point2:
    if point is None:
        return None
    return field.elt(point[0]), field.elt(point[1])


def curve2_add(
    left: Point2,
    right: Point2,
    curve: dict[str, Any],
    field: QuadraticField,
) -> Point2:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2 and field.add(y1, y2) == field.zero:
        return None
    if left == right:
        numerator = field.add(
            field.mul(field.elt(3), field.square(x1)),
            field.elt(curve["curve_a"]),
        )
        denominator = field.mul(field.elt(2), y1)
    else:
        numerator = field.sub(y2, y1)
        denominator = field.sub(x2, x1)
    slope = field.div(numerator, denominator)
    x3 = field.sub(field.sub(field.square(slope), x1), x2)
    y3 = field.sub(field.mul(slope, field.sub(x1, x3)), y1)
    return x3, y3


def curve2_scalar_mul(
    scalar: int,
    point: Point2,
    curve: dict[str, Any],
    field: QuadraticField,
) -> Point2:
    result: Point2 = None
    addend = point
    value = scalar
    while value:
        if value & 1:
            result = curve2_add(result, addend, curve, field)
        addend = curve2_add(addend, addend, curve, field)
        value >>= 1
    return result


def curve2_on_curve(
    point: Point2,
    curve: dict[str, Any],
    field: QuadraticField,
) -> bool:
    if point is None:
        return True
    x_coord, y_coord = point
    left = field.square(y_coord)
    right = field.add(
        field.add(
            field.mul(field.square(x_coord), x_coord),
            field.mul(field.elt(curve["curve_a"]), x_coord),
        ),
        field.elt(curve["curve_b"]),
    )
    return left == right


def j_zero_distortion(
    point: Point2,
    field: QuadraticField,
) -> Point2:
    if point is None:
        return None
    inverse_two = pow(2, field.p - 2, field.p)
    zeta = ((-inverse_two) % field.p, inverse_two)
    if field.add(
        field.add(field.square(zeta), zeta),
        field.one,
    ) != field.zero:
        raise AssertionError("distortion cube root drifted")
    return field.mul(zeta, point[0]), point[1]


def miller_line_quotient(
    left: Point2,
    right: Point2,
    evaluation: Point2,
    curve: dict[str, Any],
    field: QuadraticField,
) -> Fp2:
    if left is None or right is None or evaluation is None:
        raise AssertionError("unexpected identity in Miller line")
    x1, y1 = left
    x2, y2 = right
    xq, yq = evaluation
    if x1 == x2 and field.add(y1, y2) == field.zero:
        return field.sub(xq, x1)
    if left == right:
        numerator = field.add(
            field.mul(field.elt(3), field.square(x1)),
            field.elt(curve["curve_a"]),
        )
        denominator = field.mul(field.elt(2), y1)
    else:
        numerator = field.sub(y2, y1)
        denominator = field.sub(x2, x1)
    slope = field.div(numerator, denominator)
    line = field.sub(
        field.sub(yq, y1),
        field.mul(slope, field.sub(xq, x1)),
    )
    endpoint = curve2_add(left, right, curve, field)
    if endpoint is None:
        return line
    vertical = field.sub(xq, endpoint[0])
    return field.div(line, vertical)


def reduced_tate_pairing(
    point: Point2,
    evaluator: Point2,
    order: int,
    curve: dict[str, Any],
    field: QuadraticField,
) -> Fp2:
    if point is None:
        return field.one
    value = field.one
    running = point
    for bit in bin(order)[3:]:
        quotient = miller_line_quotient(
            running,
            running,
            evaluator,
            curve,
            field,
        )
        value = field.mul(field.square(value), quotient)
        running = curve2_add(running, running, curve, field)
        if bit == "1":
            quotient = miller_line_quotient(
                running,
                point,
                evaluator,
                curve,
                field,
            )
            value = field.mul(value, quotient)
            running = curve2_add(running, point, curve, field)
    if running is not None:
        raise AssertionError("Miller loop did not terminate at qP")
    exponent = (field.p * field.p - 1) // order
    return field.pow(value, exponent)


def point_from_json(value: list[int] | None) -> Point:
    if value is None:
        return None
    return value[0], value[1]


def pairing_control(
    curve: dict[str, Any],
    offset: int,
    r119_control: dict[str, Any],
) -> dict[str, Any]:
    prime = curve["field_prime"]
    order = curve["subgroup_order"]
    embedding_degree = multiplicative_order(prime % order, order)
    field = QuadraticField(prime)
    generator = R117.R81.curve_generator(curve)
    generator2 = point2(generator, field)
    distortion = j_zero_distortion(generator2, field)
    if not curve2_on_curve(distortion, curve, field):
        raise AssertionError("distortion point left the curve")
    if curve2_scalar_mul(order, distortion, curve, field) is not None:
        raise AssertionError("distortion point does not have q-torsion")
    character_generator = reduced_tate_pairing(
        generator2,
        distortion,
        order,
        curve,
        field,
    )
    if (
        character_generator == field.one
        or field.pow(character_generator, order) != field.one
    ):
        raise AssertionError("pairing character is not a nontrivial q-root")

    _, deck, _, construction = R82.compact_factor_base(curve, offset)
    deck_images = [
        reduced_tate_pairing(
            point2(point, field),
            distortion,
            order,
            curve,
            field,
        )
        for point in deck
    ]
    product_support: dict[Fp2, tuple[int, ...]] = {}
    all_endpoint_images_match = True
    all_sources_replay = True
    for source in itertools.combinations_with_replacement(
        range(len(deck)),
        5,
    ):
        endpoint = R82.add_many((deck[index] for index in source), curve)
        product_image = field.product(deck_images[index] for index in source)
        endpoint_image = reduced_tate_pairing(
            point2(endpoint, field),
            distortion,
            order,
            curve,
            field,
        )
        all_endpoint_images_match &= product_image == endpoint_image
        product_support.setdefault(product_image, source)
        all_sources_replay &= (
            R82.add_many((deck[index] for index in source), curve)
            == endpoint
        )
    empty = point_from_json(r119_control["empty_target"])
    empty_image = reduced_tate_pairing(
        point2(empty, field),
        distortion,
        order,
        curve,
        field,
    )
    expected_sources = math.comb(len(deck) + 4, 5)
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": prime,
        "subgroup_order": order,
        "cofactor": curve["cofactor"],
        "field_prime_mod_subgroup_order": prime % order,
        "embedding_degree": embedding_degree,
        "embedding_degree_is_two": embedding_degree == 2,
        "supersingular_j_zero_fixture": (
            prime % 3 == 2
            and curve["curve_a"] == 0
            and prime + 1 == curve["cofactor"] * order
        ),
        "minus_three_is_quadratic_nonsquare": True,
        "distortion_point_on_curve": True,
        "distortion_point_has_q_torsion": True,
        "pairing_character_nontrivial": character_generator != field.one,
        "pairing_character_has_order_q": (
            field.pow(character_generator, order) == field.one
        ),
        "pairing_character_generator": field.json(character_generator),
        "deck_size": len(deck),
        "factor_base_construction": construction["construction"],
        "canonical_c5_source_count": expected_sources,
        "distinct_product_image_count": len(product_support),
        "product_image_map_injective": len(product_support) == expected_sources,
        "all_c5_endpoint_images_equal_source_products": (
            all_endpoint_images_match
        ),
        "all_projective_sources_replay": all_sources_replay,
        "empty_target": r119_control["empty_target"],
        "empty_target_image": field.json(empty_image),
        "empty_target_image_absent_from_c5_product_support": (
            empty_image not in product_support
        ),
        "candidate_scalar_labels_consumed": False,
        "finite_pairing_control_receives_asymptotic_credit": False,
    }


MAXIMAL_EMBEDDING_CURVES = (
    {
        "field_prime": 7,
        "curve_a": 1,
        "curve_b": 6,
        "subgroup_order": 11,
    },
    {
        "field_prime": 7,
        "curve_a": 0,
        "curve_b": 3,
        "subgroup_order": 13,
    },
    {
        "field_prime": 11,
        "curve_a": 2,
        "curve_b": 4,
        "subgroup_order": 17,
    },
    {
        "field_prime": 13,
        "curve_a": 0,
        "curve_b": 2,
        "subgroup_order": 19,
    },
)


def curve_order(curve: dict[str, int]) -> int:
    prime = curve["field_prime"]
    total = 1
    for x_coord in range(prime):
        value = (
            x_coord**3
            + curve["curve_a"] * x_coord
            + curve["curve_b"]
        ) % prime
        if value == 0:
            total += 1
        else:
            symbol = pow(value, (prime - 1) // 2, prime)
            total += 2 if symbol == 1 else 0
    return total


def maximal_embedding_controls() -> list[dict[str, Any]]:
    rows = []
    for curve in MAXIMAL_EMBEDDING_CURVES:
        prime = curve["field_prime"]
        order = curve["subgroup_order"]
        discriminant = (
            4 * curve["curve_a"] ** 3 + 27 * curve["curve_b"] ** 2
        ) % prime
        observed_order = curve_order(curve)
        embedding_degree = multiplicative_order(prime % order, order)
        rows.append(
            {
                **curve,
                "curve_discriminant_nonzero": discriminant != 0,
                "observed_curve_order": observed_order,
                "curve_has_prime_order_q": (
                    observed_order == order and R70.is_prime(order)
                ),
                "embedding_degree": embedding_degree,
                "maximal_embedding_degree_q_minus_one": (
                    embedding_degree == order - 1
                ),
                "finite_control_receives_asymptotic_credit": False,
            }
        )
    return rows


def finite_controls() -> dict[str, Any]:
    inherited = json.loads(R119_CONTROLS.read_text(encoding="utf-8"))
    inherited_by_id = {
        row["control_id"]: row
        for row in inherited["actual_r82_controls"]
    }
    actual = []
    for curve in R82.FAMILIES:
        for offset in (0, 1):
            control_id = f"{curve['family_id']}_offset{offset}"
            actual.append(
                pairing_control(
                    curve,
                    offset,
                    inherited_by_id[control_id],
                )
            )
    maximal = maximal_embedding_controls()
    return {
        "schema": (
            "p1553.m6_c5_implicit_circuit_exceptional_controls.r120.v1"
        ),
        "r82_pairing_controls": actual,
        "r82_control_count": len(actual),
        "all_r82_embedding_degrees_two": all(
            row["embedding_degree_is_two"] for row in actual
        ),
        "all_r82_pairing_characters_nontrivial_q_roots": all(
            row["pairing_character_nontrivial"]
            and row["pairing_character_has_order_q"]
            for row in actual
        ),
        "all_r82_c5_endpoint_product_identities_exact": all(
            row["all_c5_endpoint_images_equal_source_products"]
            for row in actual
        ),
        "all_r82_product_maps_injective": all(
            row["product_image_map_injective"] for row in actual
        ),
        "all_r82_sources_and_empty_images_replay": all(
            row["all_projective_sources_replay"]
            and row[
                "empty_target_image_absent_from_c5_product_support"
            ]
            for row in actual
        ),
        "maximal_embedding_degree_prime_order_curve_controls": maximal,
        "maximal_embedding_control_count": len(maximal),
        "all_maximal_embedding_controls_exact": all(
            row["curve_discriminant_nonzero"]
            and row["curve_has_prime_order_q"]
            and row["maximal_embedding_degree_q_minus_one"]
            for row in maximal
        ),
        "finite_controls_receive_asymptotic_credit": False,
    }


def character_theorems() -> dict[str, Any]:
    return {
        "base_field_algebraic_group_character": {
            "statement": "Every algebraic-group morphism E -> G_m is trivial.",
            "proof": (
                "A morphism E -> G_m is a nowhere-zero global regular "
                "function on the proper geometrically connected curve E, "
                "hence is constant; a group morphism is therefore the "
                "identity constant."
            ),
            "nonconstant_base_field_algebraic_character_exists": False,
        },
        "self_weil_pairing": {
            "statement": (
                "For a cyclic subgroup G=<P>, e_q(aP,bP)=1 for all a,b."
            ),
            "proof": "The Weil pairing is bilinear and alternating.",
            "nontrivial_character_from_two_inputs_in_G": False,
        },
        "independent_torsion_pairing_character": {
            "statement": (
                "If R in E[q] satisfies e_q(P,R)!=1, then "
                "chi_R(X)=e_q(X,R) is an injective homomorphism "
                "G -> mu_q."
            ),
            "five_sum_equivalence": (
                "X1+...+X5=T iff "
                "chi_R(X1)*...*chi_R(X5)=chi_R(T)"
            ),
            "source_backpointer_preserved": True,
            "requires_independent_q_torsion_or_equivalent_pairing_input": True,
        },
        "embedding_degree": {
            "definition": "k=ord_q(p), the least k with q dividing p^k-1",
            "qth_root_of_unity_field": "mu_q is contained in F_(p^k)",
            "independent_torsion_field": (
                "may require F_(p^k) or a further extension; k remains a "
                "lower-bound extension degree for pairing values"
            ),
            "coordinate_representation_base_field_words_at_least": "k",
            "one_extension_field_operation_base_field_cost_at_least": (
                "Omega(k)"
            ),
            "uniform_subpolynomial_embedding_degree_assumed": False,
        },
    }


def cost_ledger() -> dict[str, Any]:
    theorem = character_theorems()
    kappa = "log_B(k)"
    return {
        "schema": "p1553.m6_c5_implicit_circuit_cost_ledger.r120.v1",
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_c5_query_exponent_B": fraction_record(Fraction(0)),
            "r118_outer_batch_exponent_B": fraction_record(
                OUTER_BATCH_EXPONENT
            ),
        },
        "character_theorems": theorem,
        "pairing_cost_parameter": {
            "embedding_degree": "k=ord_q(p)",
            "embedding_degree_exponent_B": kappa,
            "miller_loop_extension_operations": "O(log q)",
            "one_target_character_base_field_exponent_B": kappa,
            "r118_full_batch_base_field_exponent_B": f"5/4+{kappa}",
            "polylog_query_requires": "k=B^(o(1))",
            "uniform_generic_prime_requirement_proved": False,
        },
        "routes": [
            {
                "route_id": "base_field_algebraic_group_character",
                "nontrivial_route_exists": False,
                "reason": "Hom_alg_groups(E,G_m)=0",
            },
            {
                "route_id": "self_weil_pairing_inside_cyclic_subgroup",
                "nontrivial_route_exists": False,
                "reason": "alternation makes e_q(G,G)=1",
            },
            {
                "route_id": "independent_q_torsion_pairing_character",
                "exact_c5_product_encoding": True,
                "uniform_inside_query_cap": False,
                "inside_query_cap_condition": "k=B^(o(1))",
                "independent_torsion_setup_and_extension_charged": True,
            },
            {
                "route_id": "small_embedding_degree_pairing_then_explicit_c5",
                "state_exponent_B": fraction_record(C5_SUPPORT_EXPONENT),
                "inside_setup_cap": False,
                "random_deck_output_support_inherited": True,
            },
            {
                "route_id": "small_embedding_degree_pairing_then_current_k6_index",
                "state_exponent_B": fraction_record(Fraction(33, 8)),
                "inside_setup_cap": False,
            },
            {
                "route_id": "pairing_image_to_scalar_exponents",
                "requires_finite_field_discrete_logarithm": True,
                "unit_cost_discrete_log_oracle_allowed": False,
                "mov_transfer_not_membership_index": True,
            },
            {
                "route_id": (
                    "small_embedding_degree_suboutput_multiplicative_"
                    "c5_membership_source_circuit"
                ),
                "exact_circuit_constructed": False,
                "general_lower_bound_claimed": False,
                "status": "open",
            },
        ],
        "r82_fixture_boundary": {
            "all_finite_r82_embedding_degrees": 2,
            "all_r82_curves_supersingular_j_zero": True,
            "generic_asymptotic_credit": False,
        },
        "semantic_dedup": {
            "nearby_idea": "ECDLP-IDEA-008 / P1542 pairing lift-return",
            "nearby_scope": (
                "pairing inversion, distortion lift, and return geometry"
            ),
            "r120_distinct_scope": (
                "forward pairing character for exact C5 membership and "
                "source recovery"
            ),
            "new_idea_id_claimed": False,
        },
        "candidate_work_credit": False,
    }


def source_replay(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "p1553.m6_c5_implicit_circuit_source_replay.r120.v1",
        "pairing_control_count": controls["r82_control_count"],
        "all_pairing_characters_nontrivial_q_roots": controls[
            "all_r82_pairing_characters_nontrivial_q_roots"
        ],
        "all_c5_endpoint_product_identities_exact": controls[
            "all_r82_c5_endpoint_product_identities_exact"
        ],
        "all_product_maps_injective": controls[
            "all_r82_product_maps_injective"
        ],
        "all_projective_sources_and_empty_images_replay": controls[
            "all_r82_sources_and_empty_images_replay"
        ],
        "candidate_scalar_labels_consumed": False,
        "small_k_pairing_character_constructed": True,
        "inside_cap_multiplicative_c5_membership_circuit_constructed": False,
        "inside_cap_multiplicative_c5_source_recovery_constructed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R119_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R119 nonclaim boundary drifted")
    controls = finite_controls()
    cost = cost_ledger()
    replay = source_replay(controls)
    obligations = {
        "thirteen_source_bindings_verified": len(source_hashes) == 13,
        "r119_implicit_circuit_interface_inherited": (
            inherited["admission"]["random_model_theorem_admitted"]
            and not inherited["admission"]["lane_admitted"]
        ),
        "base_field_algebraic_character_triviality_theorem_complete": (
            not cost["character_theorems"][
                "base_field_algebraic_group_character"
            ]["nonconstant_base_field_algebraic_character_exists"]
        ),
        "self_weil_pairing_triviality_theorem_complete": (
            not cost["character_theorems"]["self_weil_pairing"][
                "nontrivial_character_from_two_inputs_in_G"
            ]
        ),
        "independent_torsion_pairing_character_theorem_complete": (
            cost["character_theorems"][
                "independent_torsion_pairing_character"
            ]["source_backpointer_preserved"]
        ),
        "embedding_degree_and_extension_cost_charged": (
            not cost["pairing_cost_parameter"][
                "uniform_generic_prime_requirement_proved"
            ]
        ),
        "all_eight_r82_pairing_controls_complete": (
            controls["r82_control_count"] == 8
        ),
        "all_r82_embedding_degrees_two_and_pairings_nontrivial": (
            controls["all_r82_embedding_degrees_two"]
            and controls[
                "all_r82_pairing_characters_nontrivial_q_roots"
            ]
        ),
        "all_r82_c5_product_identities_exact": controls[
            "all_r82_c5_endpoint_product_identities_exact"
        ],
        "all_r82_sources_and_empty_images_replay": controls[
            "all_r82_sources_and_empty_images_replay"
        ],
        "four_maximal_embedding_degree_curve_controls_complete": (
            controls["maximal_embedding_control_count"] == 4
            and controls["all_maximal_embedding_controls_exact"]
        ),
        "semantic_dedup_and_small_k_residual_scope_complete": (
            not cost["semantic_dedup"]["new_idea_id_claimed"]
            and not cost["candidate_work_credit"]
        ),
        "inside_cap_multiplicative_c5_membership_circuit_complete": False,
        "inside_cap_multiplicative_c5_source_recovery_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "On the only surviving pairing branch k=B^(o(1)), construct or "
        "refute an exact sub-output multiplicative five-product membership/"
        "source circuit over mu_q with B^(9/4+o(1)) state and polylogarithmic "
        "query. It must consume pairing images as field elements without "
        "taking their discrete logs, avoid the B^(15/4) product support and "
        "B^(33/8) current index, provide exact empty certification and five "
        "occurrence backpointers, and charge extension construction, "
        "independent q-torsion, every Miller/final-exponentiation operation, "
        "multiplicative FFE branch, relation use, and identical descent. "
        "Pairing-unfriendly inputs remain a separate failed branch."
    )
    frozen = {
        "schema": (
            "p1553.frozen_m6_suboutput_implicit_c5_"
            "membership_circuit.r120.v1"
        ),
        "source_bindings": source_binding_records(),
        "r119_required_interface": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_c5_query_exponent_B": fraction_record(Fraction(0)),
            "exact_empty_rejection_required": True,
            "five_occurrence_backpointers_required": True,
        },
        "closed_scoped_grammars": [
            "nonconstant base-field algebraic-group character E to G_m",
            "self Weil pairing restricted to the cyclic subgroup",
            "uniform pairing route without embedding-degree charge",
            "pairing followed by explicit random-deck C5 product support",
            "pairing followed by current bound k=6 indexing",
            "pairing image converted to scalar by a unit-cost field DLP",
        ],
        "preserved_interface": (
            "small-embedding-degree sub-output multiplicative C5 "
            "membership/source circuit over mu_q"
        ),
        "pairing_unfriendly_inputs_solved": False,
        "general_lower_bound_claimed": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r120.v1",
        "r119_random_deck_output_support_audit_complete": True,
        "r120_character_pairing_audit_complete": True,
        "small_k_pairing_character_constructed": True,
        "inside_cap_multiplicative_c5_source_index_complete": False,
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
            "EXACT_PAIRING_CHARACTER_CONTROL_AND_SCOPED_CHARACTER_"
            "NEGATIVE_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "BASE_FIELD_ALGEBRAIC_CHARACTERS_TRIVIAL__SELF_WEIL_PAIRING_"
            "TRIVIAL_ON_CYCLIC_G__INDEPENDENT_TORSION_PAIRING_GIVES_"
            "INJECTIVE_C5_PRODUCT_ENCODING_OVER_FPK__PAIRING_COST_SCALES_"
            "WITH_EMBEDDING_DEGREE_AND_R82_CONTROLS_ARE_SUPERSINGULAR_K2_"
            "EXCEPTIONS__FOUR_PRIME_ORDER_CURVES_REALIZE_MAXIMAL_K_QMINUS1_"
            "FINITE_CONTROLS__SMALL_K_PAIRING_ONLY_REENCODES_EXACT_"
            "MULTIPLICATIVE_FIVE_SUM__SUBOUTPUT_MULTIPLICATIVE_C5_CIRCUIT_"
            "OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "source_bindings": source_binding_records(),
        "character_theorems": cost["character_theorems"],
        "finite_evidence": {
            "r82_pairing_control_count": controls["r82_control_count"],
            "all_r82_embedding_degrees_two": controls[
                "all_r82_embedding_degrees_two"
            ],
            "all_r82_pairing_characters_nontrivial_q_roots": controls[
                "all_r82_pairing_characters_nontrivial_q_roots"
            ],
            "all_r82_c5_endpoint_product_identities_exact": controls[
                "all_r82_c5_endpoint_product_identities_exact"
            ],
            "all_r82_sources_and_empty_images_replay": controls[
                "all_r82_sources_and_empty_images_replay"
            ],
            "maximal_embedding_degree_curve_control_count": controls[
                "maximal_embedding_control_count"
            ],
            "all_maximal_embedding_controls_exact": controls[
                "all_maximal_embedding_controls_exact"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "pairing_character_control_admitted": True,
            "scoped_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_m6_suboutput_implicit_c5_membership_circuit.json"
            ),
            "cost": "m6_c5_implicit_circuit_cost_ledger.json",
            "source_replay": "m6_c5_implicit_circuit_source_replay.json",
            "exceptional_controls": (
                "m6_c5_implicit_circuit_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r120.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Finite R82 controls are supersingular embedding-degree-two fixtures.",
            "No uniform small embedding degree for generic inputs is claimed.",
            "A pairing character re-encodes but does not solve five-sum membership.",
            "No field discrete logarithm receives unit-cost oracle credit.",
            "No lower bound for the surviving multiplicative circuit is claimed.",
            "No relation rank, factor logs, or identical descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_EXACT_SMALL_K_PAIRING_CHARACTER_AND_C5_PRODUCT_REPLAY_"
            "ONLY__REJECT_BASE_FIELD_CHARACTER_SELF_PAIRING_UNCHARGED_"
            "EMBEDDING_DEGREE_EXPLICIT_PRODUCT_SUPPORT_CURRENT_K6_AND_UNIT_"
            "FIELD_DLP_ROUTES__PRESERVE_SMALL_K_SUBOUTPUT_MULTIPLICATIVE_C5_"
            "CIRCUIT_AND_PAIRING_UNFRIENDLY_INPUT_GAP__NO_LOCATOR__NO_RANK__"
            "NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
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
            "p1553_m6_suboutput_implicit_c5_character_"
            "pairing_probe_report_r120.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_m6_suboutput_implicit_c5_membership_circuit.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path("m6_c5_implicit_circuit_cost_ledger.json"),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path("m6_c5_implicit_circuit_source_replay.json"),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_c5_implicit_circuit_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r120.json"
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
        f"R120 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
