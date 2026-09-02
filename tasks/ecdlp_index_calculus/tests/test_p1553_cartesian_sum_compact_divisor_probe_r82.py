from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_cartesian_sum_compact_divisor_probe_r82.py"
SPEC = importlib.util.spec_from_file_location("p1553_r82", MODULE_PATH)
assert SPEC and SPEC.loader
R82 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R82)


class CartesianSumCompactDivisorProbeTests(unittest.TestCase):
    def test_factor_base_is_injective_and_scalar_blind(self) -> None:
        curve = dict(R82.FAMILIES[0])
        atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
            curve,
            0,
        )
        self.assertEqual(len(factors), len(atoms_a) * len(atoms_c))
        self.assertEqual(len(set(factors)), len(factors))
        self.assertTrue(geometry["factor_base_injective"])
        self.assertFalse(
            geometry["source_rules"]["candidate_scalar_labels_consumed"]
        )

    def test_triple_compiler_returns_exact_source(self) -> None:
        curve = dict(R82.FAMILIES[0])
        atoms_a, atoms_c, factors, _ = R82.compact_factor_base(curve, 0)
        left, left_first = R82.ordered_endpoint_map(atoms_a, 3, curve)
        right, right_first = R82.ordered_endpoint_map(atoms_c, 3, curve)
        target = R82.add_many(factors[:3], curve)
        source = R82.sumset_source(target, left_first, right_first, curve)
        self.assertIsNotNone(source)
        assert source is not None
        self.assertTrue(
            R82.source_replay(target, source, atoms_a, atoms_c, curve)
        )
        self.assertLessEqual(min(len(left), len(right)), len(atoms_a) ** 3)

    def test_rectangle_identities_have_expected_rank(self) -> None:
        size_a, size_c = 3, 5
        rows = R82.rectangle_identity_rows(size_a, size_c)
        self.assertEqual(len(rows), (size_a - 1) * (size_c - 1))
        self.assertEqual(
            R82.R81.rank_mod(rows, 524_683),
            size_a * size_c - size_a - size_c + 1,
        )

    def test_full_source_split_returns_to_rho_scale(self) -> None:
        ledger = R82.source_split_ledger()
        best = ledger["best_explicit_equality_join"]
        self.assertAlmostEqual(best["maximum_exponent_B"], 2.6)
        self.assertAlmostEqual(best["minimum_exponent_B"], 2.4)
        self.assertEqual(
            ledger["generic_collision_baseline"][
                "equivalent_group_order_exponent"
            ],
            0.5,
        )

    def test_small_bundle_passes_s4_but_withholds_pipeline(self) -> None:
        bundle = R82.build_bundle(
            families=R82.FAMILIES[:1],
            offsets=(0,),
            control_count=1,
        )
        report = bundle["report"]
        self.assertTrue(report["aggregate"]["local_s4_compiler_passes"])
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["factor_log_solve_complete"])
        self.assertFalse(report["fresh_target_descent_complete"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
