from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_5a5c_coordinate_filtration_probe_r83.py"
SPEC = importlib.util.spec_from_file_location("p1553_r83", MODULE_PATH)
assert SPEC and SPEC.loader
R83 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R83)


class CoordinateFiltrationProbeTests(unittest.TestCase):
    def test_divisible_order_quotient_is_exact_positive_control(self) -> None:
        control = R83.divisible_order_quotient_control(8)
        self.assertTrue(control["exact_homomorphic_filter"])
        self.assertEqual(control["failure_count"], 0)
        self.assertEqual(control["proper_kernel_size"], 101)

    def test_prime_order_theorem_has_no_proper_quotient(self) -> None:
        theorem = R83.prime_order_homomorphism_theorem()
        self.assertFalse(theorem["proper_nontrivial_quotient_exists"])
        self.assertIn("kernel", theorem["proof"])

    def test_entropy_replay_restores_baseline_exponents(self) -> None:
        theorem = R83.entropy_replay_theorem()
        explicit = theorem["explicit_join_to_online_cap"]
        generic = theorem["generic_collision_to_online_cap"]
        self.assertAlmostEqual(explicit["required_replay_exponent_B"], 1.35)
        self.assertEqual(explicit["restored_total_work_exponent_B"], 2.6)
        self.assertEqual(generic["required_replay_exponent_B"], 1.25)
        self.assertEqual(generic["restored_total_work_exponent_B"], 2.5)

    def test_small_filter_profile_requires_bucket_replay(self) -> None:
        curve = dict(R83.R82.FAMILIES[0])
        atoms_a, atoms_c, sources, _, generator = (
            R83.exact_source_dictionary(curve, 0)
        )
        profile = R83.filtration_profile(
            sources,
            atoms_a,
            atoms_c,
            "x_residue",
            R83.x_bucket,
            4,
            curve,
            generator,
        )
        self.assertFalse(profile["single_static_bucket_is_complete"])
        self.assertFalse(
            profile["single_target_coupled_offset_is_complete"]
        )
        self.assertTrue(
            profile["replaying_all_static_buckets_recovers_every_source"]
        )
        self.assertTrue(
            profile["replaying_all_coupled_offsets_recovers_every_source"]
        )

    def test_small_bundle_replays_s3_and_withholds_pipeline(self) -> None:
        bundle = R83.build_bundle(
            families=R83.R82.FAMILIES[:1],
            offsets=(0,),
        )
        report = bundle["report"]
        self.assertTrue(report["aggregate"]["all_group_sources_exact"])
        self.assertTrue(
            report["aggregate"]["all_regular_s3_auxiliary_chains_exact"]
        )
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["factor_log_solve_complete"])
        self.assertFalse(report["fresh_target_descent_complete"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
