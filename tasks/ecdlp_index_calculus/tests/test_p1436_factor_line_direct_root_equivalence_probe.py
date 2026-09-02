from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "p1436_factor_line_direct_root_equivalence_probe.py"
)
SPEC = importlib.util.spec_from_file_location("p1436_factor_line", MODULE_PATH)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FactorLineDirectRootEquivalenceTests(unittest.TestCase):
    def test_recognizes_normalized_root_line(self) -> None:
        p = 101
        root = 17
        fingerprint = [
            [0, 0, root * root % p],
            [0, 1, 1],
            [1, 0, root],
        ]

        self.assertEqual(PROBE.encoded_root_from_line(fingerprint, p), root)

    def test_rejects_non_root_line(self) -> None:
        fingerprint = [[0, 0, 3], [0, 1, 1], [1, 0, 2]]

        self.assertIsNone(PROBE.encoded_root_from_line(fingerprint, 101))

    def test_zero_equivalence_matches_monic_quadratic_root(self) -> None:
        p = 101
        root = 17
        b_value = 9
        c_value = (-root * root - b_value * root) % p
        fingerprint = [
            [0, 0, root * root % p],
            [0, 1, 1],
            [1, 0, root],
        ]

        self.assertEqual(
            PROBE.evaluate_fingerprint(fingerprint, b_value, c_value, p),
            0,
        )
        self.assertEqual(
            (root * root + b_value * root + c_value) % p,
            0,
        )

    def test_parses_only_sage_factor_candidate_indices(self) -> None:
        self.assertEqual(
            PROBE.candidate_index({"candidate_name": "sage_resultant_factor_12"}),
            12,
        )
        self.assertIsNone(PROBE.candidate_index({"candidate_name": "other"}))


if __name__ == "__main__":
    unittest.main()
