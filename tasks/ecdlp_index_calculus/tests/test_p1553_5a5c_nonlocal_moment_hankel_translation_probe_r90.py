from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "p1553_5a5c_nonlocal_moment_hankel_translation_probe_r90.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r90", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R90 probe")
R90 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R90)


class NonlocalMomentHankelTranslationTests(unittest.TestCase):
    def test_radix_decks_have_exact_distinct_support(self) -> None:
        rows = R90.radix_rows(3)
        self.assertEqual(len(rows), 3**5)
        self.assertEqual(
            [endpoint for endpoint, _ in rows],
            list(range(3**5)),
        )

    def test_deck_update_and_translation_laws_are_exact(self) -> None:
        control = R90.deck_update_and_translation_control()
        self.assertTrue(control["deck_product_matches_direct_moments"])
        self.assertTrue(
            control["binomial_translation_matches_direct_shift"]
        )

    def test_all_six_hankel_channels_have_full_support_rank(self) -> None:
        for deck_size in (2, 3):
            control = R90.hankel_rank_control(deck_size)
            self.assertTrue(
                control["all_six_channels_have_full_hankel_rank"]
            )
            self.assertTrue(control["newton_annihilator_exact"])
            self.assertEqual(
                control["norm_linear_complexity"],
                deck_size**5,
            )

    def test_full_moment_state_recovers_every_fixed_marker_source(self) -> None:
        replay = R90.full_marker_source_replay()
        self.assertTrue(replay["all_sources_recovered"])
        self.assertTrue(replay["newton_norm_matches_direct_product"])
        self.assertEqual(
            replay["recovered_source_count"],
            replay["endpoint_count"],
        )

    def test_bundle_closes_only_moment_hankel_grammar(self) -> None:
        bundle = R90.build_bundle()
        report = bundle["report"]
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertEqual(
            report["cost_ledger"]["exact_norm_moment_order_exponent_B"],
            3.0,
        )
        self.assertIn("non-moment", report["scope_boundary"])
        self.assertIn(
            "unequal-list subfunction-inversion",
            report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
