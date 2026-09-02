from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "p1553_constructive_closure_collision_gate_r69.py"
SPEC = importlib.util.spec_from_file_location("p1553_r69", MODULE_PATH)
assert SPEC and SPEC.loader
R69 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R69)


class ConstructiveClosureCollisionGateTests(unittest.TestCase):
    def test_relation_row_preserves_repeated_atom_multiplicity(self) -> None:
        self.assertEqual(R69.relation_row((0, 2, 2), 4), [1, 0, 2, 0])

    def test_fresh_rows_preserve_nullity_and_collisions_reduce_it(self) -> None:
        replay = R69.closure_replay()
        rank = replay["rank_accounting"]
        self.assertEqual(rank["fresh_residual_count"], 48)
        self.assertEqual(rank["fresh_residual_row_rank"], 48)
        self.assertEqual(rank["nullity_after_fresh_residual_rows"], 12)
        self.assertEqual(rank["independent_closure_collision_count"], 11)
        self.assertEqual(rank["final_relation_rank"], 59)
        self.assertEqual(rank["final_nullity"], 1)
        self.assertTrue(
            replay["factor_log_recovery"][
                "all_recovered_logs_verify_by_public_scalar_multiplication"
            ]
        )

    def test_full_gate_recovers_target_but_rejects_cost_claim(self) -> None:
        payload = R69.run()
        self.assertTrue(payload["fresh_target_descent"]["recovered"])
        self.assertEqual(
            payload["fresh_target_descent"]["verified_candidate_logs"], [53]
        )
        self.assertFalse(payload["admission"]["lane_admitted"])
        self.assertFalse(payload["result"]["source_beats_rho"])
        self.assertFalse(payload["result"]["online_descent_beats_rho"])
        self.assertFalse(payload["result"]["breakthrough"])


if __name__ == "__main__":
    unittest.main()
