from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_s6_iterated_norm_support_probe_r75.py"
SPEC = importlib.util.spec_from_file_location("p1553_r75", MODULE_PATH)
assert SPEC and SPEC.loader
R75 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R75)


class S6IteratedNormSupportProbeTests(unittest.TestCase):
    def test_symbolic_factor_specializes_to_r72(self) -> None:
        curve = dict(R75.R72.CURVES[0])
        decks, _ = R75.R72.public_decks_and_targets(curve)
        replay = R75.factor_specialization_replay(curve, decks)
        self.assertTrue(replay["specialization_matches_r72"])
        self.assertEqual(replay["symbolic_support_count"], 125)

    def test_size_two_first_norm_has_full_cube(self) -> None:
        curve = dict(R75.R72.CURVES[0])
        result = R75.probe_curve(curve, prefix_sizes=(2,))
        prefix = result["prefixes"][0]
        self.assertTrue(prefix["support_is_full_cube"])
        self.assertEqual(prefix["support_count"], 9**3)

    def test_suffix_reduction_is_inactive_after_size_four(self) -> None:
        curve = dict(R75.R72.CURVES[0])
        result = R75.probe_curve(curve, prefix_sizes=(6,))
        prefix = result["prefixes"][0]
        self.assertFalse(prefix["z_reduction_active"])
        self.assertEqual(prefix["product_degree_bounds"]["z"], 24)
        self.assertEqual(prefix["suffix_modulus_degree_lower_bound"], 36)


if __name__ == "__main__":
    unittest.main()
