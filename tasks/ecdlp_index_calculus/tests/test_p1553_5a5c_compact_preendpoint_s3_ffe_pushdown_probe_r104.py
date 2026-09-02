from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "p1553_5a5c_compact_preendpoint_s3_ffe_pushdown_probe_r104.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r104_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load R104 probe")
R104 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R104)


class CompactPreendpointS3FfePushdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R104.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["actual_controls"]
        cls.theorem = cls.report["mergeable_state_separation_theorem"]
        cls.costs = cls.report["cost_ledger"]

    def test_complete_homogeneous_group_algebra_is_exact(self) -> None:
        self.assertEqual(self.controls["instance_count"], 8)
        self.assertTrue(self.controls["all_group_algebra_histograms_exact"])
        self.assertTrue(self.controls["all_group_algebra_sources_replay"])

    def test_reverse_residual_queries_replay_all_controls(self) -> None:
        self.assertEqual(self.controls["query_control_count"], 32)
        self.assertTrue(self.controls["all_residual_queries_exact"])
        self.assertTrue(self.controls["all_blind_queries_return_bottom"])
        self.assertTrue(self.controls["all_projective_identity_queries_exact"])
        self.assertGreater(
            self.controls["repeated_multiplicity_instance_count"],
            0,
        )

    def test_actual_suffix_contexts_are_full_rank(self) -> None:
        self.assertTrue(self.controls["all_prefix_endpoint_maps_injective"])
        self.assertTrue(self.controls["all_suffix_endpoint_maps_injective"])
        self.assertTrue(
            self.controls["all_suffix_context_matrices_full_rank"]
        )
        for instance in self.controls["instances"]:
            self.assertEqual(
                instance["suffix_translation_context_matrix_rank"],
                instance["suffix_occurrence_count"],
            )

    def test_sign_and_projective_controls_are_exact(self) -> None:
        controls = self.report["projective_and_exceptional_controls"]
        self.assertTrue(controls["all_group_charts_exact"])
        self.assertTrue(controls["four_sign_branch_control_exact"])
        self.assertTrue(
            controls["identity_and_degree_drop_covered_before_affine_s3"]
        )
        self.assertTrue(controls["tangent_doubling_covered"])

    def test_mergeable_state_separation_requires_suffix_body(self) -> None:
        self.assertIn(
            "singleton context [T-P]",
            self.theorem["fixed_target_separation"],
        )
        self.assertEqual(
            self.theorem["required_state_exponent_B"]["exact"],
            "14/5",
        )
        self.assertFalse(self.theorem["inside_setup_cap"])
        self.assertFalse(self.theorem["inside_online_cap"])
        self.assertIn(
            "extension of scalars preserves",
            self.theorem["ffe_scalar_extension"],
        )

    def test_explicit_group_algebra_and_reverse_routes_miss_caps(self) -> None:
        group = self.costs["complete_homogeneous_group_algebra"]
        reverse = self.costs[
            "target_specialized_reverse_residual_recurrence"
        ]
        self.assertEqual(
            group["h5_a_occurrence_exponent_B"]["exact"],
            "2",
        )
        self.assertEqual(
            group["h5_c_occurrence_exponent_B"]["exact"],
            "3",
        )
        self.assertEqual(
            group["full_pair_convolution_exponent_B"]["exact"],
            "5",
        )
        self.assertTrue(group["h5_a_inside_setup_cap"])
        self.assertFalse(group["h5_c_inside_setup_cap"])
        self.assertFalse(group["full_pair_convolution_inside_online_cap"])
        self.assertEqual(
            reverse["prefix_state_exponent_B"]["exact"],
            "11/5",
        )
        self.assertEqual(
            reverse["suffix_residual_exponent_B"]["exact"],
            "14/5",
        )
        self.assertFalse(reverse["suffix_inside_online_cap"])

    def test_bundle_closes_mergeable_class_only(self) -> None:
        admission = self.report["admission"]
        self.assertFalse(admission["lane_admitted"])
        self.assertTrue(self.report["mergeable_preendpoint_pushdown_closed"])
        self.assertTrue(
            self.report["actual_deck_specific_nonmergeable_circuit_open"]
        )
        self.assertFalse(self.report["factor_log_solve_complete"])
        self.assertFalse(self.report["fresh_target_descent_complete"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "non-mergeable",
            self.report["scope_boundary"],
        )


if __name__ == "__main__":
    unittest.main()
