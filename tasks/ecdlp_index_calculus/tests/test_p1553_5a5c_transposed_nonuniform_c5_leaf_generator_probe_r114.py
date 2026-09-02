from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_5a5c_transposed_nonuniform_c5_leaf_generator_probe_r114.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r114", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R114 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R114)


class TransposedNonuniformC5LeafGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R114.build_bundle()
        cls.report = cls.bundle["report"]
        cls.replay = cls.bundle["source_replay"]
        cls.rows = [
            *cls.replay["actual"],
            *cls.replay["matched_random_decks"],
        ]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R114.verify_source_bindings()), 10)

    def test_first_and_second_adjoint_sources_replay_r113(self) -> None:
        summary = self.replay["summary"]
        self.assertEqual(summary["unique_zero_instance_count"], 14)
        self.assertEqual(summary["double_zero_instance_count"], 2)
        self.assertTrue(summary["all_unique_gradients_exact"])
        self.assertTrue(summary["all_double_hessian_vectors_exact"])
        self.assertTrue(summary["all_sources_match_r113"])

    def test_sources_preserve_markers_and_weight(self) -> None:
        self.assertEqual(self.replay["summary"]["source_count"], 18)
        for row in self.rows:
            for source in row["sources"]:
                self.assertEqual(
                    source["canonical_cycle_weight"],
                    R114.R108.FULL_CYCLE_SCALE,
                )
                self.assertEqual(
                    len(source["marker"]),
                    R114.R105.MARKER_DIMENSION,
                )

    def test_reverse_checkpointing_does_not_reduce_leaf_work(self) -> None:
        product = self.report["cost_ledger"][
            "product_forward_and_adjoint"
        ]
        self.assertEqual(product["primal_leaf_exponent_B"]["exact"], "3")
        self.assertEqual(
            product["reverse_product_work_exponent_B"]["exact"],
            "3",
        )
        self.assertEqual(
            product["checkpointed_reverse_state_exponent_B"]["exact"],
            "0",
        )
        self.assertEqual(
            product["checkpointed_reverse_work_exponent_B"]["exact"],
            "3",
        )
        self.assertFalse(
            product["higher_adjoint_order_reduces_primal_leaf_work"]
        )

    def test_first_preleaf_transposed_state_exceeds_both_caps(self) -> None:
        transpose = self.report["cost_ledger"][
            "transpose_before_leaf_formation"
        ]
        self.assertEqual(
            transpose["one_c_atom_expansion_exponent_B"]["exact"],
            "13/5",
        )
        self.assertFalse(transpose["first_state_inside_setup_cap"])
        self.assertFalse(transpose["first_state_inside_online_cap"])

    def test_typed_profiles_cover_all_c5_prefix_depths(self) -> None:
        for row in self.rows:
            profile = row["typed_transposed_state_profile"]
            self.assertEqual(
                [entry["c_prefix_depth"] for entry in profile],
                list(range(6)),
            )
            state_counts = [
                entry["distinct_boundary_typed_state_count"]
                for entry in profile
            ]
            self.assertEqual(state_counts, sorted(state_counts))
            self.assertGreater(state_counts[1], state_counts[0])

    def test_scoped_negative_routes_to_exponent_rebalance(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(admission["scoped_negative_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "relation-arity and asymmetric factor-base exponent space",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
