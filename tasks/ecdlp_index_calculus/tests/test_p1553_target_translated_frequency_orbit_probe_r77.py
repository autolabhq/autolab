from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_target_translated_frequency_orbit_probe_r77.py"
SPEC = importlib.util.spec_from_file_location("p1553_r77", MODULE_PATH)
assert SPEC and SPEC.loader
R77 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R77)


class TargetTranslatedFrequencyOrbitProbeTests(unittest.TestCase):
    def test_cyclic_convolution_matches_fourier_product(self) -> None:
        order = 7
        modulus, root = R77.fourier_field(order, 100)
        left = [1, 1, 0, 0, 0, 0, 0]
        right = [0, 1, 0, 1, 0, 0, 0]
        direct = R77.cyclic_convolution(left, right)
        transformed = R77.dft(direct, root, modulus)
        product = R77.pointwise_product(
            [
                R77.dft(left, root, modulus),
                R77.dft(right, root, modulus),
            ],
            modulus,
        )
        self.assertEqual(transformed, product)

    def test_circulant_rank_equals_fourier_support(self) -> None:
        control = R77.small_circulant_rank_control()
        self.assertTrue(control["rank_equals_fourier_support"])

    def test_small_four_family_report_preserves_nonclaim(self) -> None:
        report = R77.build_report(prefix_sizes=(2,))
        aggregate = report["aggregate"]
        self.assertTrue(aggregate["all_convolution_theorem_checks_exact"])
        self.assertTrue(aggregate["all_branch_masses_conserved"])
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
