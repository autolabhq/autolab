from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_s4_centered_carry_rank_probe_r71.py"
SPEC = importlib.util.spec_from_file_location("p1553_r71", MODULE_PATH)
assert SPEC and SPEC.loader
R71 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R71)


class S4CenteredCarryRankProbeTests(unittest.TestCase):
    def test_quadratic_resultant_implementations_agree(self) -> None:
        left = (1, -3, 2)
        right = (1, -4, 3)
        self.assertEqual(R71.quadratic_resultant_closed(left, right), 0)
        self.assertEqual(
            R71.quadratic_resultant_closed((2, 5, 7), (3, 11, 13)),
            R71.quadratic_resultant_bareiss((2, 5, 7), (3, 11, 13)),
        )

    def test_s4_vanishes_on_forced_relation(self) -> None:
        curve = dict(R71.CURVES[0])
        decks, targets = R71.public_decks_and_targets(curve)
        target = next(
            row for row in targets if row["target_id"] == "forced_positive_target"
        )
        indices = target["forced_witness"]
        self.assertIsNotNone(indices)
        points = tuple(decks[mode][indices[mode]] for mode in range(3))
        self.assertTrue(
            R71.signed_relation_exists(points, target["point"], curve)
        )
        self.assertEqual(
            R71.semaev_s4_integer(
                points[0][0],
                points[1][0],
                points[2][0],
                target["point"][0],
                curve,
            )
            % curve["field_prime"],
            0,
        )

    def test_four_family_probe_preserves_nonclaim(self) -> None:
        payload = R71.run()
        self.assertEqual(payload["aggregate"]["family_count"], 4)
        self.assertEqual(payload["aggregate"]["target_instance_count"], 8)
        self.assertTrue(
            payload["checks"]["all_s4_predicates_match_signed_group_relations"]
        )
        self.assertTrue(payload["checks"]["all_carries_are_exactly_divisible"])
        self.assertFalse(payload["admission"]["lane_admitted"])
        self.assertFalse(payload["result"]["shoup_bound_improvement"])
        self.assertFalse(payload["result"]["breakthrough"])


if __name__ == "__main__":
    unittest.main()
