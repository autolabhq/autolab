from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_s6_residual_decision_diagram_probe_r74.py"
SPEC = importlib.util.spec_from_file_location("p1553_r74", MODULE_PATH)
assert SPEC and SPEC.loader
R74 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R74)


class S6ResidualDecisionDiagramProbeTests(unittest.TestCase):
    def test_endpoint_support_matches_s4_squarefree_radical(self) -> None:
        curve = dict(R74.R72.CURVES[0])
        decks, _ = R74.R72.public_decks_and_targets(curve)
        points = (decks[0][0], decks[1][1], decks[2][2])
        endpoint = R74.endpoint_polynomial_key(
            R74.endpoint_key(points, curve),
            curve["field_prime"],
        )
        radical = R74.s4_radical_key(points, curve)
        self.assertEqual(endpoint, radical)

    def test_endpoint_intersection_recognizes_forced_relation(self) -> None:
        curve = dict(R74.R72.CURVES[0])
        decks, targets = R74.R72.public_decks_and_targets(curve)
        target = next(
            row for row in targets if row["target_id"] == "forced_positive_target"
        )
        indices = target["forced_witness"]
        left = R74.endpoint_key(
            (
                decks[0][indices[0]],
                decks[1][indices[1]],
                decks[2][indices[2]],
            ),
            curve,
        )
        right = R74.endpoint_key(
            (
                target["point"],
                decks[3][indices[3]],
                decks[4][indices[4]],
            ),
            curve,
        )
        self.assertTrue(R74.keys_intersect(left, right))

    def test_small_prefix_keeps_nonclaim_boundary(self) -> None:
        curve = dict(R74.R72.CURVES[0])
        result = R74.probe_curve(curve, prefix_sizes=(4,))
        prefix = result["prefixes"][0]
        self.assertTrue(prefix["radical_sample_replay"]["all_match"])
        self.assertEqual(
            prefix["triple_occurrence_count"],
            4**3,
        )


if __name__ == "__main__":
    unittest.main()
