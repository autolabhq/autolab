from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_full_multiplicative_x_coset_endpoint_probe_r81.py"
SPEC = importlib.util.spec_from_file_location("p1553_r81", MODULE_PATH)
assert SPEC and SPEC.loader
R81 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R81)


class FullMultiplicativeXCosetEndpointProbeTests(unittest.TestCase):
    def test_complete_coset_satisfies_sparse_domain_polynomial(self) -> None:
        curve = dict(R81.FAMILIES[0])
        points, geometry = R81.full_coset_factor_base(curve, 0)
        self.assertTrue(points)
        self.assertEqual(
            geometry["enumerated_coordinate_count"],
            geometry["coordinate_subgroup_order"],
        )
        self.assertEqual(
            geometry["sparse_domain_polynomial"][
                "nonzero_coefficient_count"
            ],
            2,
        )
        self.assertFalse(geometry["scalar_labels_consumed"])

    def test_bsgs_verifier_recovers_every_coset_point(self) -> None:
        curve = dict(R81.FAMILIES[0])
        generator = R81.curve_generator(curve)
        points, _ = R81.full_coset_factor_base(curve, 0)
        verifier = R81.BatchBsgsVerifier(generator, curve)
        labels = verifier.labels(points)
        self.assertEqual(len(labels), len(points))
        self.assertTrue(
            all(
                R81.R70.scalar_mul(label, generator, curve) == point
                for point, label in zip(points, labels)
            )
        )
        self.assertTrue(
            verifier.receipt()["excluded_from_algorithmic_credit"]
        )

    def test_endpoint_profile_preserves_ordered_occurrences(self) -> None:
        profile = R81.triple_endpoint_profile([1, 4, 9], 101)
        self.assertEqual(profile["ordered_triple_occurrence_count"], 27)
        self.assertGreater(profile["distinct_kummer_root_count"], 0)
        self.assertGreater(profile["distinct_endpoint_set_count"], 0)

    def test_small_bundle_keeps_nonclaim_and_cost_boundary(self) -> None:
        bundle = R81.build_bundle(
            families=R81.FAMILIES[:1],
            offsets=(0,),
            control_count=1,
            target_count_value=8,
        )
        report = bundle["report"]
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertTrue(
            report["aggregate"]["all_group_endpoint_samples_exact"]
        )


if __name__ == "__main__":
    unittest.main()
