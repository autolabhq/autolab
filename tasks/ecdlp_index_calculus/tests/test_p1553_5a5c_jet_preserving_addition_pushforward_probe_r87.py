from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "p1553_5a5c_jet_preserving_addition_pushforward_probe_r87.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r87", MODULE_PATH)
assert SPEC and SPEC.loader
R87 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R87)


class JetPreservingAdditionPushforwardTests(unittest.TestCase):
    def test_target_local_first_jet_is_not_compositional(self) -> None:
        witness = R87.local_jet_nonfunctor_witness()
        self.assertTrue(witness["target_first_jets_equal"])
        self.assertTrue(witness["marker_polynomials_reconstructed_p"])
        self.assertTrue(witness["marker_polynomials_reconstructed_q"])
        self.assertTrue(witness["full_local_first_jets_equal"])
        self.assertTrue(witness["shifted_first_jets_differ"])
        self.assertTrue(witness["convolution_first_jets_differ"])

    def test_translation_remainder_orbit_has_full_rank(self) -> None:
        for modulus_degree, translated_degree in ((3, 5), (5, 8), (7, 11)):
            control = R87.translation_orbit_control(
                modulus_degree,
                translated_degree,
            )
            self.assertEqual(
                control["translated_remainder_rank"],
                modulus_degree,
            )
            self.assertTrue(control["full_quotient_rank"])

    def test_translated_gcd_recovers_unique_joint_source(self) -> None:
        replay = R87.unique_target_replay()
        self.assertEqual(replay["target_multiplicity"], 1)
        self.assertEqual(replay["translated_gcd_degree"], 1)
        self.assertTrue(replay["joint_source_replay"])
        self.assertTrue(replay["simple_final_norm_branch"])
        self.assertFalse(replay["candidate_credit"])

    def test_nonreduced_branch_needs_first_jet(self) -> None:
        controls = R87.exceptional_controls()
        self.assertTrue(controls["empty"]["detected"])
        self.assertTrue(
            controls["multiple_distinct"][
                "rejected_by_unique_source_gate"
            ]
        )
        self.assertTrue(
            controls["nonreduced"]["gcd_alone_misses_multiplicity"]
        )
        self.assertTrue(controls["nonreduced"]["first_jet_rejects"])

    def test_bundle_preserves_open_black_box_exception(self) -> None:
        bundle = R87.build_bundle()
        report = bundle["report"]
        self.assertEqual(
            report["classification"],
            (
                "TARGET_LOCAL_FIRST_NORM_JET_NONFUNCTORIAL__"
                "EXPLICIT_TRANSLATED_REMAINDER_FULL_B2"
            ),
        )
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertIn("black-box", report["next_action"])
        self.assertFalse(report["factor_log_solve_complete"])
        self.assertFalse(report["fresh_target_descent_complete"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
