from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_s6_centered_carry_rank_minor_probe_r72.py"
SPEC = importlib.util.spec_from_file_location("p1553_r72", MODULE_PATH)
assert SPEC and SPEC.loader
R72 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R72)


class S6CenteredCarryRankMinorProbeTests(unittest.TestCase):
    def test_s4_polynomial_matches_r71_value(self) -> None:
        curve = dict(R72.R71.CURVES[0])
        x_values = (11, 29, 47, 83)
        polynomial = R72.s4_polynomial_last(
            x_values[0],
            x_values[1],
            x_values[2],
            curve["curve_a"],
            curve["curve_b"],
            curve["field_prime"],
        )
        evaluated = sum(
            coefficient * pow(x_values[3], degree, curve["field_prime"])
            for degree, coefficient in enumerate(polynomial)
        ) % curve["field_prime"]
        direct = R72.R71.semaev_s4_integer(
            *x_values,
            curve,
        ) % curve["field_prime"]
        self.assertEqual(evaluated, direct)

    def test_s6_vanishes_on_forced_large_curve_relation(self) -> None:
        curve = dict(R72.CURVES[0])
        decks, targets = R72.public_decks_and_targets(curve)
        target = next(
            row for row in targets if row["target_id"] == "forced_positive_target"
        )
        indices = target["forced_witness"]
        points = tuple(decks[mode][indices[mode]] for mode in range(5))
        self.assertTrue(
            R72.signed_relation_exists(points, target["point"], curve)
        )
        x_coordinates = tuple(point[0] for point in points) + (
            target["point"][0],
        )
        self.assertEqual(
            R72.semaev_s6_mod(
                x_coordinates,
                curve["curve_a"],
                curve["curve_b"],
                curve["field_prime"],
            ),
            0,
        )

    def test_curve_parameter_controls(self) -> None:
        for curve in R72.CURVES:
            self.assertTrue(R72.is_probable_prime(curve["field_prime"]))
            self.assertTrue(R72.is_probable_prime(curve["subgroup_order"]))
            self.assertTrue(R72.curve_discriminant_nonzero(curve))
            self.assertTrue(R72.point_is_on_curve(curve["generator"], curve))
            self.assertIsNone(
                R72.R70.scalar_mul(
                    curve["subgroup_order"],
                    curve["generator"],
                    curve,
                )
            )


if __name__ == "__main__":
    unittest.main()
