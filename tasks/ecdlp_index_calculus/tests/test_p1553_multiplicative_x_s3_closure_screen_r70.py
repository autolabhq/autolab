from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_multiplicative_x_s3_closure_screen_r70.py"
SPEC = importlib.util.spec_from_file_location("p1553_r70", MODULE_PATH)
assert SPEC and SPEC.loader
R70 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R70)


class MultiplicativeXS3ClosureScreenTests(unittest.TestCase):
    def test_semaev_s3_vanishes_on_exact_relation(self) -> None:
        curve = dict(R70.CURVES[0])
        points, _ = R70.multiplicative_x_base(curve)
        residual = R70.negate(R70.add(points[0], points[1], curve), curve)
        self.assertIsNotNone(residual)
        self.assertEqual(
            R70.semaev_s3(
                points[0][0],
                points[1][0],
                residual[0],
                curve,
            ),
            0,
        )

    def test_four_family_screen_rejects_untransferred_rank(self) -> None:
        payload = R70.run()
        self.assertEqual(payload["aggregate"]["family_count"], 4)
        self.assertEqual(
            payload["aggregate"]["families_with_preregistered_rank_excess"], 0
        )
        self.assertEqual(
            [
                row["candidate_profile"]["independent_collision_rank"]
                for row in payload["family_results"]
            ],
            [3, 1, 4, 0],
        )
        self.assertTrue(payload["checks"]["all_subgroup_orders_are_prime"])
        self.assertFalse(payload["admission"]["lane_admitted"])
        self.assertFalse(payload["result"]["breakthrough"])


if __name__ == "__main__":
    unittest.main()
