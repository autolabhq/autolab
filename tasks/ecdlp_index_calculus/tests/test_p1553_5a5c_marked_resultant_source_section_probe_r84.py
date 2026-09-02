from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "p1553_5a5c_marked_resultant_source_section_probe_r84.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r84", MODULE_PATH)
assert SPEC and SPEC.loader
R84 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R84)


class MarkedResultantSourceSectionProbeTests(unittest.TestCase):
    def test_fp2_root_polynomial_and_gcd(self) -> None:
        prime = 101
        nonsquare = R84.least_nonsquare(prime)
        roots = [(1, 2), (3, 4), (7, 9)]
        polynomial = R84.poly_from_roots(roots, prime, nonsquare)
        self.assertTrue(
            all(
                R84.poly_eval(polynomial, root, prime, nonsquare)
                == R84.f2_zero()
                for root in roots
            )
        )
        other = R84.poly_from_roots(
            [(3, 4), (8, 5)],
            prime,
            nonsquare,
        )
        gcd = R84.poly_gcd_monic(
            polynomial,
            other,
            prime,
            nonsquare,
        )
        self.assertEqual(len(gcd) - 1, 1)
        self.assertEqual(
            R84.poly_eval(gcd, (3, 4), prime, nonsquare),
            R84.f2_zero(),
        )

    def test_packed_source_roundtrip_is_joint(self) -> None:
        source = ((0, 2, 2), (1, 3))
        code = R84.encode_source(source, 3, 5)
        self.assertEqual(
            R84.decode_source(code, 3, 5, 3, 2),
            source,
        )

    def test_small_bundle_replays_every_target_and_rejects_caps(self) -> None:
        bundle = R84.build_bundle(
            families=R84.R82.FAMILIES[:1],
            offsets=(0,),
        )
        report = bundle["report"]
        self.assertTrue(report["aggregate"]["all_side_sections_exact"])
        self.assertTrue(
            report["aggregate"]["all_attained_target_sources_exact"]
        )
        self.assertTrue(
            report["aggregate"]["all_sampled_coefficient_gcds_exact"]
        )
        self.assertFalse(
            bundle["cost_ledger"][
                "explicit_radical_endpoint_polynomial"
            ]["smaller_side_inside_setup_cap"]
        )
        self.assertTrue(
            bundle["cost_ledger"]["p1510_control"][
                "output_sensitive_exception_preserved"
            ]
        )
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["factor_log_solve_complete"])
        self.assertFalse(report["fresh_target_descent_complete"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
