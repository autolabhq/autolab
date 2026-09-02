#!/usr/bin/env python3
"""Validate the mined selected13 second-stage rule on disjoint transfers.

The second-stage miner found a no-leaf public decision list that improves the
active-scout top-5 shortlist in-sample. This probe freezes that rule and tests it
against selected13-like nonadjacent validation transfers that are not in the
mining contract. It replays only the single leaf chosen by the frozen rule, so
the measured cost is the policy cost rather than the top-5 diagnostic scan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import ffe_single_hit_root_relation_replay_probe as replay_probe
import low_term_total2_selected13_common_leaf_pair_sweep_probe as common_sweep
import low_term_total2_selected13_public_prefix_min_transfer_probe as min_transfer
import low_term_total2_selected13_salt_conditioned_direct_replay_probe as direct_replay
import low_term_total2_selected13_salt_conditioned_second_stage_rule_miner as second_stage


SCHEMA = "ecdlp.low_term_total2_selected13_second_stage_disjoint_validation_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_TRAINING_CONTRACT = DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9999_probe.json"
DEFAULT_VALIDATION_MANIFEST = (
    DEFAULT_STATE_DIR / "low_term_total2_public_lane_validation_manifest_selected13_nonadjacent_9696_9999_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_second_stage_disjoint_validation_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_second_stage_disjoint_validation_probe.h"

TARGET = min_transfer.TARGET
DEFAULT_SHORTLIST_MODE = "active_scout_sum_desc"
DEFAULT_SHORTLIST_K = 5
DEFAULT_MAX_TARGETS = 13
DEFAULT_RULE_POLICY = "original_public_depth3"
SCOUT_TRIGGER = 6
SPAN_TRIGGER = 2
SPAN_POSITION_TRIGGER = 6
ROWHIT_TRIGGER = 6
ROWHIT_SPAN_TRIGGER = 3
ROWHIT_SPAN_LE_TRIGGER = 3
ROOT_TRIGGER = 1
ROOT_SPAN_A = 4
ROOT_SPAN_B = 9
ROOT1_ROW_HIT_TRIGGER = 4
ROOT1_SPAN_C = 1
ROOT2_TRIGGER = 2
ROOT2_ROW_HIT_TRIGGER = 6
ROOT2_SPAN_HIGH = 9
ROOT3_TRIGGER = 3
ROOT3_ROW_HIT_TRIGGER = 8
ROOT3_SPAN_HIGH = 9
ROOT2_SCOUT_TRIGGER = 6
ROOT2_SPAN_MED = 4
ROOT3_SCOUT_LOW_TRIGGER = 4
SCOUT_HIGH_TRIGGER = 8
SPAN_HIGH_TRIGGER = 9
SELECTOR_IDS = {
    "original_public_depth3": "second_stage_public_depth3_frozen_disjoint",
    "public_depth1_span_root": "second_stage_public_depth1_span_root_frozen_disjoint",
    "public_span_le6_position": "second_stage_public_span_le6_position_frozen_full147",
    "public_rowhit6_span3_then_span_le6_position": "second_stage_public_rowhit6_span3_then_span_le6_position_frozen_full147",
    "public_root1_span49_rowhit6_spanle3_then_span_le6_position": (
        "second_stage_public_root1_span49_rowhit6_spanle3_then_span_le6_position_frozen_full147"
    ),
    "public_noregret94_then_span_le6_position": (
        "second_stage_public_noregret94_then_span_le6_position_frozen_full147"
    ),
    "public_noregret96_then_span_le6_position": (
        "second_stage_public_noregret96_then_span_le6_position_frozen_full147"
    ),
    "public_position_guard104_then_span_le6_position": (
        "second_stage_public_position_guard104_then_span_le6_position_frozen_full147"
    ),
    "public_position_guard106_then_span_le6_position": (
        "second_stage_public_position_guard106_then_span_le6_position_frozen_full147"
    ),
    "public_guarded_span110_then_span_le6_position": (
        "second_stage_public_guarded_span110_then_span_le6_position_frozen_full147"
    ),
    "public_root2_span_le6_active111_then_span_le6_position": (
        "second_stage_public_root2_span_le6_active111_then_span_le6_position_frozen_full147"
    ),
}
SOURCE_SELECTOR_IDS = {
    "original_public_depth3": "second_stage_public_depth3",
    "public_depth1_span_root": "second_stage_public_depth1",
    "public_span_le6_position": "second_stage_public_depth1_full147_span_le6_position",
    "public_rowhit6_span3_then_span_le6_position": "second_stage_public_residual_rowhit6_span3_full147",
    "public_root1_span49_rowhit6_spanle3_then_span_le6_position": (
        "second_stage_public_residual_root1_span49_rowhit6_spanle3_full147"
    ),
    "public_noregret94_then_span_le6_position": "second_stage_public_residual_noregret94_full147",
    "public_noregret96_then_span_le6_position": "second_stage_public_residual_noregret96_full147",
    "public_position_guard104_then_span_le6_position": "second_stage_public_position_guard104_full147",
    "public_position_guard106_then_span_le6_position": "second_stage_public_position_guard106_full147",
    "public_guarded_span110_then_span_le6_position": "second_stage_public_guarded_span110_full147",
    "public_root2_span_le6_active111_then_span_le6_position": "second_stage_public_root2_span_le6_active111_full147",
}

PREDICATE_CODES = {
    "fallback": 1,
    "scout>=6": 2,
    "span==2": 3,
    "span<=6": 4,
    "rowhit>=6&span==3": 5,
    "root==1&span==4": 6,
    "root==1&span==9": 7,
    "rowhit>=6&span<=3": 8,
    "root==1&rowhit==4&span==1": 9,
    "root==2&rowhit==6&span==9": 10,
    "root==3&rowhit==8&span==9": 11,
    "root==2&scout==6&span==4": 12,
    "root==1&scout==4&span==1": 13,
    "root==2&scout==6&span==9": 14,
    "root==3&scout==4&span>=9": 15,
    "scout>=8&span>=9": 16,
    "position==0&span==2": 17,
    "position==1&span==3": 18,
    "position==2&span==4": 19,
    "position==0&span==1": 20,
    "position==1&span==2": 21,
    "position==1&span==13": 22,
    "position==0&root==3&rowhit==6&span==4": 23,
    "position==0&root==3&rowhit==6&span==8": 24,
    "active>=4&span<=4": 25,
    "root==2&span<=6": 26,
}
ORDER_CODES = {
    "span": 1,
    "span_root": 2,
    "position": 3,
    "position_desc": 4,
    "span_desc": 5,
    "active": 6,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: Any, default: int = 0) -> int:
    return min_transfer.as_int(value, default)


def as_float(value: Any) -> float | None:
    return min_transfer.as_float(value)


def round_or_none(value: Any, digits: int = 8) -> float | None:
    return min_transfer.round_or_none(value, digits)


def parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def training_transfers(contract: dict[str, Any]) -> list[int]:
    return sorted(as_int(target.get("transfer_index")) for target in common_sweep.contract_backfill_targets(contract))


def row_to_target(row: dict[str, Any], queue_position: int) -> dict[str, Any]:
    return {
        "first_pass_id": row.get("range"),
        "full_family_row_id": None,
        "group_id": row.get("range"),
        "known_positive": as_int(row.get("transfer_index")) in common_sweep.KNOWN_POSITIVE_TRANSFERS,
        "manifest_queue_position": queue_position,
        "manifest_range": row.get("range"),
        "manifest_selector": row.get("selector"),
        "min_direct_ops_over_rho": round_or_none(row.get("direct_ops_over_rho")),
        "row_check_hash": None,
        "row_keys": [str(row_key) for row_key in row.get("row_keys") or []],
        "salts": row.get("salts") or [],
        "target": str(row.get("target") or TARGET),
        "transfer_index": as_int(row.get("transfer_index")),
    }


def select_validation_targets(
    manifest: dict[str, Any],
    excluded_transfers: set[int],
    *,
    max_targets: int | None,
    requested_transfers: list[int] | None,
    skip_targets: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    requested = set(requested_transfers or [])
    effective_skip = 0 if requested else max(skip_targets, 0)
    queue = manifest.get("direct_rank_transfer_queue") or []
    targets: list[dict[str, Any]] = []
    seen: set[int] = set()
    skipped = 0
    for queue_position, item in enumerate(queue):
        transfer = as_int(item.get("transfer_index"))
        if requested and transfer not in requested:
            continue
        if transfer in seen:
            continue
        seen.add(transfer)
        if transfer in excluded_transfers:
            continue
        row = item.get("best_row") or {}
        row_keys = row.get("row_keys") or []
        if len(row_keys) != 2:
            failures.append(
                {
                    "code": "validation_row_key_count_unexpected",
                    "observed": len(row_keys),
                    "transfer_index": transfer,
                }
            )
            continue
        if skipped < effective_skip:
            skipped += 1
            continue
        targets.append(row_to_target(row, queue_position))
        if max_targets is not None and len(targets) >= max_targets:
            break
    if requested:
        selected = {as_int(target.get("transfer_index")) for target in targets}
        missing = sorted(requested - selected - excluded_transfers)
        excluded_requested = sorted(requested & excluded_transfers)
        if missing:
            failures.append({"code": "requested_transfers_missing_from_validation_manifest", "transfers": missing})
        if excluded_requested:
            failures.append({"code": "requested_transfers_overlap_training_contract", "transfers": excluded_requested})
    return targets, failures


def fixed_rule_choice(shortlist: list[dict[str, Any]], policy: str) -> tuple[dict[str, Any], str, str]:
    scout_matches = [record for record in shortlist if as_int(record.get("scout_hit_total_sum")) >= SCOUT_TRIGGER]
    if policy == "public_depth1_span_root":
        if scout_matches:
            return (
                sorted(scout_matches, key=lambda record: second_stage.order_key(compact_for_rule(record, 0), "span_root"))[0],
                "scout>=6",
                "span_root",
            )
        return (
            sorted(shortlist, key=lambda record: second_stage.order_key(compact_for_rule(record, 0), "span_root"))[0],
            "fallback",
            "span_root",
        )
    if policy == "public_span_le6_position":
        span_matches = [
            (position, record)
            for position, record in enumerate(shortlist)
            if as_int(record.get("min_term_span")) <= SPAN_POSITION_TRIGGER
        ]
        if span_matches:
            _, chosen = sorted(
                span_matches,
                key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
            )[0]
            return (
                chosen,
                "span<=6",
                "position",
            )
        return (
            sorted(shortlist, key=lambda record: second_stage.order_key(compact_for_rule(record, 0), "span_root"))[0],
            "fallback",
            "span_root",
        )
    if policy == "public_rowhit6_span3_then_span_le6_position":
        rowhit_span_matches = [
            (position, record)
            for position, record in enumerate(shortlist)
            if as_int(record.get("row_hit_total_sum")) >= ROWHIT_TRIGGER
            and as_int(record.get("min_term_span")) == ROWHIT_SPAN_TRIGGER
        ]
        if rowhit_span_matches:
            _, chosen = sorted(
                rowhit_span_matches,
                key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
            )[0]
            return (
                chosen,
                "rowhit>=6&span==3",
                "position",
            )
        span_matches = [
            (position, record)
            for position, record in enumerate(shortlist)
            if as_int(record.get("min_term_span")) <= SPAN_POSITION_TRIGGER
        ]
        if span_matches:
            _, chosen = sorted(
                span_matches,
                key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
            )[0]
            return (
                chosen,
                "span<=6",
                "position",
            )
        return (
            sorted(shortlist, key=lambda record: second_stage.order_key(compact_for_rule(record, 0), "span_root"))[0],
            "fallback",
            "span_root",
        )
    if policy in {
        "public_root1_span49_rowhit6_spanle3_then_span_le6_position",
        "public_noregret94_then_span_le6_position",
        "public_noregret96_then_span_le6_position",
        "public_position_guard104_then_span_le6_position",
        "public_position_guard106_then_span_le6_position",
        "public_guarded_span110_then_span_le6_position",
        "public_root2_span_le6_active111_then_span_le6_position",
    }:
        root_span_a_matches = [
            (position, record)
            for position, record in enumerate(shortlist)
            if as_int(record.get("hit_root_count_sum")) == ROOT_TRIGGER
            and as_int(record.get("min_term_span")) == ROOT_SPAN_A
        ]
        if root_span_a_matches:
            _, chosen = sorted(
                root_span_a_matches,
                key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
            )[0]
            return (
                chosen,
                "root==1&span==4",
                "position",
            )
        root_span_b_matches = [
            (position, record)
            for position, record in enumerate(shortlist)
            if as_int(record.get("hit_root_count_sum")) == ROOT_TRIGGER
            and as_int(record.get("min_term_span")) == ROOT_SPAN_B
        ]
        if root_span_b_matches:
            _, chosen = sorted(
                root_span_b_matches,
                key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
            )[0]
            return (
                chosen,
                "root==1&span==9",
                "position",
            )
        if policy in {
            "public_noregret96_then_span_le6_position",
            "public_position_guard104_then_span_le6_position",
            "public_position_guard106_then_span_le6_position",
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            root2_scout_span_med_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("hit_root_count_sum")) == ROOT2_TRIGGER
                and as_int(record.get("scout_hit_total_sum")) == ROOT2_SCOUT_TRIGGER
                and as_int(record.get("min_term_span")) == ROOT2_SPAN_MED
            ]
            if root2_scout_span_med_matches:
                _, chosen = sorted(
                    root2_scout_span_med_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "root==2&scout==6&span==4",
                    "span_root",
                )
        if policy == "public_noregret94_then_span_le6_position":
            root1_span_c_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("hit_root_count_sum")) == ROOT_TRIGGER
                and as_int(record.get("row_hit_total_sum")) == ROOT1_ROW_HIT_TRIGGER
                and as_int(record.get("min_term_span")) == ROOT1_SPAN_C
            ]
            if root1_span_c_matches:
                _, chosen = sorted(
                    root1_span_c_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
                )[0]
                return (
                    chosen,
                    "root==1&rowhit==4&span==1",
                    "position",
                )
            root2_high_span_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("hit_root_count_sum")) == ROOT2_TRIGGER
                and as_int(record.get("row_hit_total_sum")) == ROOT2_ROW_HIT_TRIGGER
                and as_int(record.get("min_term_span")) == ROOT2_SPAN_HIGH
            ]
            if root2_high_span_matches:
                _, chosen = sorted(
                    root2_high_span_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "root==2&rowhit==6&span==9",
                    "span_root",
                )
            root3_high_span_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("hit_root_count_sum")) == ROOT3_TRIGGER
                and as_int(record.get("row_hit_total_sum")) == ROOT3_ROW_HIT_TRIGGER
                and as_int(record.get("min_term_span")) == ROOT3_SPAN_HIGH
            ]
            if root3_high_span_matches:
                _, chosen = sorted(
                    root3_high_span_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "root==3&rowhit==8&span==9",
                    "span_root",
                )
        rowhit_span_le_matches = [
            (position, record)
            for position, record in enumerate(shortlist)
            if as_int(record.get("row_hit_total_sum")) >= ROWHIT_TRIGGER
            and as_int(record.get("min_term_span")) <= ROWHIT_SPAN_LE_TRIGGER
        ]
        if rowhit_span_le_matches:
            _, chosen = sorted(
                rowhit_span_le_matches,
                key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position_desc"),
            )[0]
            return (
                chosen,
                "rowhit>=6&span<=3",
                "position_desc",
            )
        if policy in {
            "public_noregret96_then_span_le6_position",
            "public_position_guard104_then_span_le6_position",
            "public_position_guard106_then_span_le6_position",
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            root1_scout_span_c_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("hit_root_count_sum")) == ROOT_TRIGGER
                and as_int(record.get("scout_hit_total_sum")) == ROOT1_ROW_HIT_TRIGGER
                and as_int(record.get("min_term_span")) == ROOT1_SPAN_C
            ]
            if root1_scout_span_c_matches:
                _, chosen = sorted(
                    root1_scout_span_c_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "root==1&scout==4&span==1",
                    "span_root",
                )
            root2_scout_high_span_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("hit_root_count_sum")) == ROOT2_TRIGGER
                and as_int(record.get("scout_hit_total_sum")) == ROOT2_SCOUT_TRIGGER
                and as_int(record.get("min_term_span")) == ROOT2_SPAN_HIGH
            ]
            if root2_scout_high_span_matches:
                _, chosen = sorted(
                    root2_scout_high_span_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "root==2&scout==6&span==9",
                    "span_root",
                )
            root3_scout_high_span_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("hit_root_count_sum")) == ROOT3_TRIGGER
                and as_int(record.get("scout_hit_total_sum")) == ROOT3_SCOUT_LOW_TRIGGER
                and as_int(record.get("min_term_span")) >= SPAN_HIGH_TRIGGER
            ]
            if root3_scout_high_span_matches:
                _, chosen = sorted(
                    root3_scout_high_span_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "root==3&scout==4&span>=9",
                    "span_root",
                )
            scout_high_span_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("scout_hit_total_sum")) >= SCOUT_HIGH_TRIGGER
                and as_int(record.get("min_term_span")) >= SPAN_HIGH_TRIGGER
            ]
            if scout_high_span_matches:
                _, chosen = sorted(
                    scout_high_span_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "scout>=8&span>=9",
                    "span_root",
                )
        if policy in {
            "public_position_guard104_then_span_le6_position",
            "public_position_guard106_then_span_le6_position",
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            position0_span2_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if position == 0
                and as_int(record.get("min_term_span")) == 2
            ]
            if position0_span2_matches:
                _, chosen = sorted(
                    position0_span2_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
                )[0]
                return (
                    chosen,
                    "position==0&span==2",
                    "position",
                )
            position1_span3_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if position == 1
                and as_int(record.get("min_term_span")) == 3
            ]
            if position1_span3_matches:
                _, chosen = sorted(
                    position1_span3_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "position==1&span==3",
                    "span_root",
                )
            position2_span4_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if position == 2
                and as_int(record.get("min_term_span")) == 4
            ]
            if position2_span4_matches:
                _, chosen = sorted(
                    position2_span4_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "position==2&span==4",
                    "span_root",
                )
        if policy in {
            "public_position_guard106_then_span_le6_position",
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            position0_span1_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if position == 0
                and as_int(record.get("min_term_span")) == 1
            ]
            if position0_span1_matches:
                _, chosen = sorted(
                    position0_span1_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
                )[0]
                return (
                    chosen,
                    "position==0&span==1",
                    "position",
                )
            position1_span2_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if position == 1
                and as_int(record.get("min_term_span")) == 2
            ]
            if position1_span2_matches:
                _, chosen = sorted(
                    position1_span2_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "position==1&span==2",
                    "span_root",
                )
            position1_span13_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if position == 1
                and as_int(record.get("min_term_span")) == 13
            ]
            if position1_span13_matches:
                _, chosen = sorted(
                    position1_span13_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_root"),
                )[0]
                return (
                    chosen,
                    "position==1&span==13",
                    "span_root",
                )
        if policy in {
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            position0_root3_span4_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if position == 0
                and as_int(record.get("hit_root_count_sum")) == 3
                and as_int(record.get("row_hit_total_sum")) == 6
                and as_int(record.get("min_term_span")) == 4
            ]
            if position0_root3_span4_matches:
                _, chosen = sorted(
                    position0_root3_span4_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
                )[0]
                return (
                    chosen,
                    "position==0&root==3&rowhit==6&span==4",
                    "position",
                )
            position0_root3_span8_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if position == 0
                and as_int(record.get("hit_root_count_sum")) == 3
                and as_int(record.get("row_hit_total_sum")) == 6
                and as_int(record.get("min_term_span")) == 8
            ]
            if position0_root3_span8_matches:
                _, chosen = sorted(
                    position0_root3_span8_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
                )[0]
                return (
                    chosen,
                    "position==0&root==3&rowhit==6&span==8",
                    "position",
                )
            active_span_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("active_scout_count_sum")) >= 4
                and as_int(record.get("min_term_span")) <= 4
            ]
            if active_span_matches:
                _, chosen = sorted(
                    active_span_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "span_desc"),
                )[0]
                return (
                    chosen,
                    "active>=4&span<=4",
                    "span_desc",
                )
        if policy == "public_root2_span_le6_active111_then_span_le6_position":
            root2_span_le6_matches = [
                (position, record)
                for position, record in enumerate(shortlist)
                if as_int(record.get("hit_root_count_sum")) == ROOT2_TRIGGER
                and as_int(record.get("min_term_span")) <= SPAN_POSITION_TRIGGER
            ]
            if root2_span_le6_matches:
                _, chosen = sorted(
                    root2_span_le6_matches,
                    key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "active"),
                )[0]
                return (
                    chosen,
                    "root==2&span<=6",
                    "active",
                )
        span_matches = [
            (position, record)
            for position, record in enumerate(shortlist)
            if as_int(record.get("min_term_span")) <= SPAN_POSITION_TRIGGER
        ]
        if span_matches:
            _, chosen = sorted(
                span_matches,
                key=lambda item: second_stage.order_key(compact_for_rule(item[1], item[0]), "position"),
            )[0]
            return (
                chosen,
                "span<=6",
                "position",
            )
        return (
            sorted(shortlist, key=lambda record: second_stage.order_key(compact_for_rule(record, 0), "span_root"))[0],
            "fallback",
            "span_root",
        )
    if policy != "original_public_depth3":
        raise ValueError(f"unsupported rule policy: {policy}")
    if scout_matches:
        return sorted(scout_matches, key=lambda record: second_stage.order_key(compact_for_rule(record, 0), "span"))[0], "scout>=6", "span"
    span_matches = [record for record in shortlist if as_int(record.get("min_term_span")) == SPAN_TRIGGER]
    if span_matches:
        return (
            sorted(span_matches, key=lambda record: second_stage.order_key(compact_for_rule(record, 0), "span_root"))[0],
            "span==2",
            "span_root",
        )
    return (
        sorted(shortlist, key=lambda record: second_stage.order_key(compact_for_rule(record, 0), "span_root"))[0],
        "fallback",
        "span_root",
    )


def frozen_rule_description(policy: str) -> list[dict[str, Any]]:
    if policy == "public_depth1_span_root":
        return [
            {"order_mode": "span_root", "predicate": {"field": "scout", "op": ">=", "value": SCOUT_TRIGGER}},
            {"fallback_order_mode": "span_root", "predicate": {"kind": "fallback"}},
        ]
    if policy == "public_span_le6_position":
        return [
            {"order_mode": "position", "predicate": {"field": "span", "op": "<=", "value": SPAN_POSITION_TRIGGER}},
            {"fallback_order_mode": "span_root", "predicate": {"kind": "fallback"}},
        ]
    if policy == "public_rowhit6_span3_then_span_le6_position":
        return [
            {
                "order_mode": "position",
                "predicate": {
                    "all": [
                        {"field": "rowhit", "op": ">=", "value": ROWHIT_TRIGGER},
                        {"field": "span", "op": "==", "value": ROWHIT_SPAN_TRIGGER},
                    ],
                    "kind": "all_of",
                },
            },
            {"order_mode": "position", "predicate": {"field": "span", "op": "<=", "value": SPAN_POSITION_TRIGGER}},
            {"fallback_order_mode": "span_root", "predicate": {"kind": "fallback"}},
        ]
    if policy in {
        "public_root1_span49_rowhit6_spanle3_then_span_le6_position",
        "public_noregret94_then_span_le6_position",
        "public_noregret96_then_span_le6_position",
        "public_position_guard104_then_span_le6_position",
        "public_position_guard106_then_span_le6_position",
        "public_guarded_span110_then_span_le6_position",
        "public_root2_span_le6_active111_then_span_le6_position",
    }:
        rules = [
            {
                "order_mode": "position",
                "predicate": {
                    "all": [
                        {"field": "root", "op": "==", "value": ROOT_TRIGGER},
                        {"field": "span", "op": "==", "value": ROOT_SPAN_A},
                    ],
                    "kind": "all_of",
                },
            },
            {
                "order_mode": "position",
                "predicate": {
                    "all": [
                        {"field": "root", "op": "==", "value": ROOT_TRIGGER},
                        {"field": "span", "op": "==", "value": ROOT_SPAN_B},
                    ],
                    "kind": "all_of",
                },
            },
        ]
        if policy == "public_noregret94_then_span_le6_position":
            rules.extend(
                [
                    {
                        "order_mode": "position",
                        "predicate": {
                            "all": [
                                {"field": "root", "op": "==", "value": ROOT_TRIGGER},
                                {"field": "rowhit", "op": "==", "value": ROOT1_ROW_HIT_TRIGGER},
                                {"field": "span", "op": "==", "value": ROOT1_SPAN_C},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "root", "op": "==", "value": ROOT2_TRIGGER},
                                {"field": "rowhit", "op": "==", "value": ROOT2_ROW_HIT_TRIGGER},
                                {"field": "span", "op": "==", "value": ROOT2_SPAN_HIGH},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "root", "op": "==", "value": ROOT3_TRIGGER},
                                {"field": "rowhit", "op": "==", "value": ROOT3_ROW_HIT_TRIGGER},
                                {"field": "span", "op": "==", "value": ROOT3_SPAN_HIGH},
                            ],
                            "kind": "all_of",
                        },
                    },
                ]
            )
        if policy in {
            "public_noregret96_then_span_le6_position",
            "public_position_guard104_then_span_le6_position",
            "public_position_guard106_then_span_le6_position",
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            rules.append(
                {
                    "order_mode": "span_root",
                    "predicate": {
                        "all": [
                            {"field": "root", "op": "==", "value": ROOT2_TRIGGER},
                            {"field": "scout", "op": "==", "value": ROOT2_SCOUT_TRIGGER},
                            {"field": "span", "op": "==", "value": ROOT2_SPAN_MED},
                        ],
                        "kind": "all_of",
                    },
                }
            )
        rules.extend(
            [
                {
                    "order_mode": "position_desc",
                    "predicate": {
                        "all": [
                            {"field": "rowhit", "op": ">=", "value": ROWHIT_TRIGGER},
                            {"field": "span", "op": "<=", "value": ROWHIT_SPAN_LE_TRIGGER},
                        ],
                        "kind": "all_of",
                    },
                },
            ]
        )
        if policy in {
            "public_noregret96_then_span_le6_position",
            "public_position_guard104_then_span_le6_position",
            "public_position_guard106_then_span_le6_position",
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            rules.extend(
                [
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "root", "op": "==", "value": ROOT_TRIGGER},
                                {"field": "scout", "op": "==", "value": ROOT1_ROW_HIT_TRIGGER},
                                {"field": "span", "op": "==", "value": ROOT1_SPAN_C},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "root", "op": "==", "value": ROOT2_TRIGGER},
                                {"field": "scout", "op": "==", "value": ROOT2_SCOUT_TRIGGER},
                                {"field": "span", "op": "==", "value": ROOT2_SPAN_HIGH},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "root", "op": "==", "value": ROOT3_TRIGGER},
                                {"field": "scout", "op": "==", "value": ROOT3_SCOUT_LOW_TRIGGER},
                                {"field": "span", "op": ">=", "value": SPAN_HIGH_TRIGGER},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "scout", "op": ">=", "value": SCOUT_HIGH_TRIGGER},
                                {"field": "span", "op": ">=", "value": SPAN_HIGH_TRIGGER},
                            ],
                            "kind": "all_of",
                        },
                    },
                ]
            )
        if policy in {
            "public_position_guard104_then_span_le6_position",
            "public_position_guard106_then_span_le6_position",
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            rules.extend(
                [
                    {
                        "order_mode": "position",
                        "predicate": {
                            "all": [
                                {"field": "position", "op": "==", "value": 0},
                                {"field": "span", "op": "==", "value": 2},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "position", "op": "==", "value": 1},
                                {"field": "span", "op": "==", "value": 3},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "position", "op": "==", "value": 2},
                                {"field": "span", "op": "==", "value": 4},
                            ],
                            "kind": "all_of",
                        },
                    },
                ]
            )
        if policy in {
            "public_position_guard106_then_span_le6_position",
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            rules.extend(
                [
                    {
                        "order_mode": "position",
                        "predicate": {
                            "all": [
                                {"field": "position", "op": "==", "value": 0},
                                {"field": "span", "op": "==", "value": 1},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "position", "op": "==", "value": 1},
                                {"field": "span", "op": "==", "value": 2},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_root",
                        "predicate": {
                            "all": [
                                {"field": "position", "op": "==", "value": 1},
                                {"field": "span", "op": "==", "value": 13},
                            ],
                            "kind": "all_of",
                        },
                    },
                ]
            )
        if policy in {
            "public_guarded_span110_then_span_le6_position",
            "public_root2_span_le6_active111_then_span_le6_position",
        }:
            rules.extend(
                [
                    {
                        "order_mode": "position",
                        "predicate": {
                            "all": [
                                {"field": "position", "op": "==", "value": 0},
                                {"field": "root", "op": "==", "value": 3},
                                {"field": "rowhit", "op": "==", "value": 6},
                                {"field": "span", "op": "==", "value": 4},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "position",
                        "predicate": {
                            "all": [
                                {"field": "position", "op": "==", "value": 0},
                                {"field": "root", "op": "==", "value": 3},
                                {"field": "rowhit", "op": "==", "value": 6},
                                {"field": "span", "op": "==", "value": 8},
                            ],
                            "kind": "all_of",
                        },
                    },
                    {
                        "order_mode": "span_desc",
                        "predicate": {
                            "all": [
                                {"field": "active", "op": ">=", "value": 4},
                                {"field": "span", "op": "<=", "value": 4},
                            ],
                            "kind": "all_of",
                        },
                    },
                ]
            )
        if policy == "public_root2_span_le6_active111_then_span_le6_position":
            rules.append(
                {
                    "order_mode": "active",
                    "predicate": {
                        "all": [
                            {"field": "root", "op": "==", "value": ROOT2_TRIGGER},
                            {"field": "span", "op": "<=", "value": SPAN_POSITION_TRIGGER},
                        ],
                        "kind": "all_of",
                    },
                }
            )
        rules.extend(
            [
                {"order_mode": "position", "predicate": {"field": "span", "op": "<=", "value": SPAN_POSITION_TRIGGER}},
                {"fallback_order_mode": "span_root", "predicate": {"kind": "fallback"}},
            ]
        )
        return rules
    return [
        {"order_mode": "span", "predicate": {"field": "scout", "op": ">=", "value": SCOUT_TRIGGER}},
        {"order_mode": "span_root", "predicate": {"field": "span", "op": "==", "value": SPAN_TRIGGER}},
        {"fallback_order_mode": "span_root", "predicate": {"kind": "fallback"}},
    ]


def compact_for_rule(record: dict[str, Any], position: int) -> dict[str, Any]:
    return {
        "active": as_int(record.get("active_scout_count_sum")),
        "leaf": as_int(record.get("leaf_index")),
        "position": position,
        "root": as_int(record.get("hit_root_count_sum")),
        "rowhit": as_int(record.get("row_hit_total_sum")),
        "scout": as_int(record.get("scout_hit_total_sum")),
        "span": as_int(record.get("min_term_span")),
    }


def shortlist_entry(record: dict[str, Any], position: int, chosen: bool, mode: str) -> dict[str, Any]:
    out = direct_replay.compact_association_leaf(record, mode)
    out["chosen_by_frozen_rule"] = chosen
    out["position"] = position
    return out


def build_replay_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        row_pool=args.row_pool,
        row_count=args.row_count,
        scout_limit=args.scout_limit,
        scout_mode=args.scout_mode,
        scout_order=args.scout_order,
        selected_limit=args.selected_limit,
        factor_base_size=args.factor_base_size,
        max_relations=args.max_relations,
        min_distinct_indices=args.min_distinct_indices,
        min_unsigned_distinct_indices=args.min_unsigned_distinct_indices,
        require_unit_coefficients=args.require_unit_coefficients,
        row_factor=args.row_factor,
        product_factor=args.product_factor,
        seed=args.seed,
        event_summary_limit=args.event_summary_limit,
        context_top_k=args.context_top_k,
    )


def replay_fixed_rule(
    args: argparse.Namespace,
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    bank_source = replay_probe.load_json(Path(args.bank_source))
    config_source = replay_probe.load_json(Path(args.config_source))
    direct_source = replay_probe.load_json(Path(args.direct_source))
    transfer_source = replay_probe.load_json(Path(args.transfer_source))
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    if not isinstance(params, dict):
        params = {}
    radius = as_int(args.radius if args.radius is not None else params.get("radius"), 4)
    specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
    verifier = replay_probe.relation_probe.load_verifier_module()
    verifier_records = verifier.load_records()
    replay_args = build_replay_args(args)
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    records_out: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        contexts, errors = common_sweep.materialize_contexts_for_target(
            verifier,
            verifier_records,
            config_source,
            specs_by_target,
            target,
            replay_args,
            context_cache,
        )
        for error in errors:
            failures.append(
                {
                    "code": "context_materialization_error",
                    "error": error,
                    "transfer_index": target.get("transfer_index"),
                }
            )
        if errors or len(contexts) != 2:
            continue
        ranked = direct_replay.ranked_leaf_records(target, contexts, args.shortlist_mode, args.leaf_limit)
        shortlist = ranked[: max(1, args.shortlist_k)]
        if not shortlist:
            failures.append({"code": "empty_shortlist", "transfer_index": target.get("transfer_index")})
            continue
        chosen, predicate, order_mode = fixed_rule_choice(shortlist, args.rule_policy)
        chosen_leaf = as_int(chosen.get("leaf_index"))
        row_leaves = {str(row_key): {chosen_leaf} for row_key in target.get("row_keys") or []}
        result, row_events = replay_probe.replay_selection(
            verifier,
            row_leaves,
            contexts,
            scan_cache,
            args.event_summary_limit,
        )
        replay_record = direct_replay.direct_replay_record(
            index,
            target,
            [chosen],
            result,
            row_events,
            args.event_summary_limit,
            args.shortlist_mode,
            1,
        )
        replay_record["manifest_queue_position"] = as_int(target.get("manifest_queue_position"))
        replay_record["manifest_range"] = target.get("manifest_range")
        replay_record["manifest_selector"] = target.get("manifest_selector")
        replay_record["second_stage_order_mode"] = order_mode
        replay_record["second_stage_predicate"] = predicate
        replay_record["second_stage_predicate_code"] = PREDICATE_CODES[predicate]
        replay_record["second_stage_selector_id"] = SELECTOR_IDS[args.rule_policy]
        replay_record["shortlist"] = [
            shortlist_entry(record, position, as_int(record.get("leaf_index")) == chosen_leaf, args.shortlist_mode)
            for position, record in enumerate(shortlist)
        ]
        replay_record["shortlist_position"] = next(
            (position for position, record in enumerate(shortlist) if as_int(record.get("leaf_index")) == chosen_leaf),
            0,
        )
        records_out.append(replay_record)
    return records_out, radius, failures


def best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if record.get("accepted_relation_export") and record.get("below_rho")
    ]
    candidates.sort(
        key=lambda item: (
            as_float(item.get("ops_over_rho")) if as_float(item.get("ops_over_rho")) is not None else 999.0,
            as_int(item.get("transfer_index")),
        )
    )
    return candidates[0] if candidates else {}


def summarize(
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    selected_targets: list[dict[str, Any]],
    excluded_transfers: set[int],
) -> dict[str, Any]:
    accepted = [record for record in records if record.get("accepted_relation_export")]
    below = [record for record in accepted if record.get("below_rho")]
    statuses = Counter(str(record.get("status")) for record in records)
    selected_transfers = sorted(as_int(target.get("transfer_index")) for target in selected_targets)
    overlap = sorted(set(selected_transfers) & excluded_transfers)
    best = best_record(records)
    missing = [
        as_int(record.get("transfer_index"))
        for record in records
        if not (record.get("accepted_relation_export") and record.get("below_rho"))
    ]
    return {
        "accepted_relation_export_count": len(accepted),
        "below_rho_accepted_relation_export_count": len(below),
        "below_rho_accepted_relation_export_transfers": sorted(as_int(record.get("transfer_index")) for record in below),
        "best_candidate_id": best.get("candidate_id"),
        "best_derived_secret": best.get("derived_secret"),
        "best_ops_over_rho": best.get("ops_over_rho"),
        "best_transfer_index": best.get("transfer_index"),
        "disjoint_from_training_contract": not overlap,
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "missing_below_rho_transfers": sorted(missing),
        "selected_validation_transfer_count": len(selected_targets),
        "selected_validation_transfers": selected_transfers,
        "status_counts": dict(sorted(statuses.items())),
        "training_overlap_count": len(overlap),
        "training_overlap_transfers": overlap,
        "verified": not failures,
        "worker_interpretation": (
            "This freezes the mined no-leaf decision list and applies it to selected13-like "
            "nonadjacent validation transfers that are disjoint from the mining contract. "
            "It is disjoint transfer validation on the same target family, not a promoted "
            "general ECDLP speedup."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_FAILED"
    if as_int(summary.get("training_overlap_count")) > 0:
        return "SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_HAS_OVERLAP"
    if as_int(summary.get("below_rho_accepted_relation_export_count")) > 0:
        return "SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_BELOW_RHO_EXPORT"
    return "SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_NO_EXPORT"


def render_c_header(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = []
    for record in records:
        ops_scaled = 0
        ops_over_rho = as_float(record.get("ops_over_rho"))
        if ops_over_rho is not None:
            ops_scaled = int(round(ops_over_rho * 1_000_000))
        selected = record.get("selected_leaf_indices") or []
        order_mode = str(record.get("second_stage_order_mode"))
        rows.append(
            "  {"
            f"{as_int(record.get('candidate_index'))}ULL, "
            f"{as_int(record.get('candidate_id_u64'))}ULL, "
            f"{as_int(record.get('transfer_index'))}ULL, "
            f"{as_int(record.get('manifest_queue_position'))}ULL, "
            f"{as_int(selected[0] if selected else 0)}ULL, "
            f"{as_int(record.get('shortlist_position'))}ULL, "
            f"{as_int(record.get('second_stage_predicate_code'))}ULL, "
            f"{ORDER_CODES.get(order_mode, 0)}ULL, "
            f"{1 if record.get('known_positive_transfer') else 0}ULL, "
            f"{1 if record.get('below_rho') else 0}ULL, "
            f"{1 if record.get('public_key_verified') else 0}ULL, "
            f"{1 if record.get('relation_derived_ecdlp') else 0}ULL, "
            f"{as_int(record.get('rank'))}ULL, "
            f"{as_int(record.get('relation_count'))}ULL, "
            f"{as_int(record.get('derived_secret'))}ULL, "
            f"{ops_scaled}ULL, "
            f"{as_int(record.get('status_code'))}ULL"
            "},"
        )
    accepted_count = sum(1 for record in records if record.get("relation_derived_ecdlp"))
    accepted_below_count = sum(1 for record in records if record.get("relation_derived_ecdlp") and record.get("below_rho"))
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_PROBE_H

#include <stdint.h>

#define SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_RECORD_COUNT {len(records)}
#define SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_SELECTED_TRANSFER_COUNT {as_int(summary.get("selected_validation_transfer_count"))}
#define SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_TRAINING_OVERLAP_COUNT {as_int(summary.get("training_overlap_count"))}
#define SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_ACCEPTED_COUNT {accepted_count}
#define SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_ACCEPTED_BELOW_COUNT {accepted_below_count}

typedef struct {{
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t transfer_index;
  uint64_t manifest_queue_position;
  uint64_t selected_leaf_index;
  uint64_t shortlist_position;
  uint64_t predicate_code;
  uint64_t order_mode_code;
  uint64_t known_positive_transfer;
  uint64_t below_rho;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t rank;
  uint64_t relation_count;
  uint64_t derived_secret;
  uint64_t ops_over_rho_scaled_1e6;
  uint64_t status_code;
}} selected13_second_stage_disjoint_validation_record_t;

static const selected13_second_stage_disjoint_validation_record_t SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_RECORDS[] = {{
{chr(10).join(rows)}
}};

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  uint64_t failure_count = 0;
  uint64_t record_count =
      sizeof(SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_RECORDS) /
      sizeof(SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_RECORDS[0]);
  uint64_t accepted = 0;
  uint64_t accepted_below = 0;

  if (record_count != SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_RECORD_COUNT) failure_count++;
  if (record_count == 0ULL) failure_count++;
  if (record_count != SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_SELECTED_TRANSFER_COUNT) failure_count++;
  if (SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_TRAINING_OVERLAP_COUNT != 0ULL) failure_count++;

  for (size_t i = 0; i < record_count; i++) {{
    const selected13_second_stage_disjoint_validation_record_t *record =
        &SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_RECORDS[i];
    if (record->candidate_id_u64 == 0ULL) failure_count++;
    if (record->predicate_code == 0ULL) failure_count++;
    if (record->order_mode_code == 0ULL) failure_count++;
    if (record->status_code == 0ULL) failure_count++;
    if (record->relation_derived_ecdlp && !record->public_key_verified) failure_count++;
    if (record->relation_derived_ecdlp && record->derived_secret == 0ULL) failure_count++;
    if (record->relation_derived_ecdlp) {{
      accepted++;
      if (record->below_rho) accepted_below++;
    }}
  }}

  if (accepted != SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_ACCEPTED_COUNT) failure_count++;
  if (accepted_below != SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_ACCEPTED_BELOW_COUNT) failure_count++;

  printf("selected13_second_stage_disjoint_validation_preflight records=%llu accepted=%llu accepted_below=%llu overlap=%llu failures=%llu\\n",
         (unsigned long long)record_count,
         (unsigned long long)accepted,
         (unsigned long long)accepted_below,
         (unsigned long long)SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_TRAINING_OVERLAP_COUNT,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_second_stage_disjoint_validation_preflight_") as tmp:
        tmp_path = Path(tmp)
        c_path = tmp_path / "preflight.c"
        exe_path = tmp_path / "preflight"
        local_header = tmp_path / header_path.name
        c_path.write_text(source)
        local_header.write_text(header_path.read_text())
        compile_cmd = ["cc", "-std=c99", "-Wall", "-Wextra", "-O2", str(c_path), "-o", str(exe_path)]
        compile_run = subprocess.run(compile_cmd, text=True, capture_output=True, check=False)
        if compile_run.returncode != 0:
            return {
                "compile_command": compile_cmd,
                "compile_returncode": compile_run.returncode,
                "compile_stderr": compile_run.stderr,
                "verified": False,
            }
        preflight_run = subprocess.run([str(exe_path)], text=True, capture_output=True, check=False)
        return {
            "compile_command": compile_cmd,
            "compile_returncode": compile_run.returncode,
            "preflight_returncode": preflight_run.returncode,
            "preflight_stdout": preflight_run.stdout.strip(),
            "preflight_stderr": preflight_run.stderr.strip(),
            "verified": preflight_run.returncode == 0,
        }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    training_contract = load_json(Path(args.training_contract))
    validation_manifest = load_json(Path(args.validation_manifest))
    failures: list[dict[str, Any]] = []
    train_transfers = set(training_transfers(training_contract))
    if not train_transfers:
        failures.append({"code": "empty_training_transfer_set"})
    requested_transfers = parse_int_csv(args.transfer_list) if args.transfer_list else None
    targets, target_failures = select_validation_targets(
        validation_manifest,
        train_transfers,
        max_targets=args.max_targets,
        requested_transfers=requested_transfers,
        skip_targets=args.skip_targets,
    )
    failures.extend(target_failures)
    if not targets:
        failures.append({"code": "no_disjoint_validation_targets"})
    records, radius, replay_failures = replay_fixed_rule(args, targets)
    failures.extend(replay_failures)
    summary = summarize(records, failures, targets, train_transfers)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, summary),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "config_source": str(Path(args.config_source)),
            "context_top_k": args.context_top_k,
            "leaf_limit": args.leaf_limit,
            "max_targets": args.max_targets,
            "radius": radius,
            "rule_policy": args.rule_policy,
            "skip_targets": args.skip_targets,
            "shortlist_k": args.shortlist_k,
            "shortlist_mode": args.shortlist_mode,
            "target": TARGET,
            "training_contract": str(Path(args.training_contract)),
            "training_contract_transfer_count": len(train_transfers),
            "training_contract_transfers": sorted(train_transfers),
            "transfer_list": requested_transfers,
            "transfer_source": str(Path(args.transfer_source)),
            "validation_manifest": str(Path(args.validation_manifest)),
        },
        "frozen_rule": {
            "selector_id": SELECTOR_IDS[args.rule_policy],
            "source_selector_id": SOURCE_SELECTOR_IDS[args.rule_policy],
            "rules": frozen_rule_description(args.rule_policy),
            "uses_leaf_predicate": False,
            "uses_leaf_order_mode": False,
        },
        "summary": summary,
        "selected_validation_targets": targets,
        "direct_replay_records": records,
        "failures": failures,
        "honesty_boundary": {
            "common_sweep_labels_used": False,
            "direct_shortlist_labels_used_for_selection": False,
            "general_ecdlp_algorithm_claimed": False,
            "same_target_family_as_training": True,
            "training_transfer_overlap_count": summary.get("training_overlap_count"),
            "validation_shape": "disjoint transfer validation inside selected13-like nonadjacent manifest",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-contract", type=Path, default=DEFAULT_TRAINING_CONTRACT)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--shortlist-mode", default=DEFAULT_SHORTLIST_MODE)
    parser.add_argument("--shortlist-k", type=int, default=DEFAULT_SHORTLIST_K)
    parser.add_argument("--rule-policy", choices=sorted(SELECTOR_IDS), default=DEFAULT_RULE_POLICY)
    parser.add_argument("--max-targets", type=int, default=DEFAULT_MAX_TARGETS)
    parser.add_argument("--skip-targets", type=int, default=0)
    parser.add_argument("--transfer-list")
    parser.add_argument("--leaf-limit", type=int)
    parser.add_argument("--context-top-k", type=int, default=16)
    parser.add_argument("--row-pool", type=int, default=512)
    parser.add_argument("--row-count", type=int, default=128)
    parser.add_argument("--scout-limit", type=int, default=192)
    parser.add_argument("--scout-mode", default="s3_coeff_spread")
    parser.add_argument("--scout-order", default="eval_cover_hits_high")
    parser.add_argument("--selected-limit", type=int, default=64)
    parser.add_argument("--factor-base-size", type=int, default=16)
    parser.add_argument("--max-relations", type=int, default=96)
    parser.add_argument("--min-distinct-indices", type=int, default=4)
    parser.add_argument("--min-unsigned-distinct-indices", type=int, default=2)
    parser.add_argument(
        "--allow-combined-coefficients",
        dest="require_unit_coefficients",
        action="store_false",
    )
    parser.set_defaults(require_unit_coefficients=True)
    parser.add_argument("--row-factor", type=int, default=512)
    parser.add_argument("--product-factor", type=int, default=4096)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--event-summary-limit", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["direct_replay_records"], payload["summary"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = claim_status(payload["failures"], payload["summary"])
    write_json(Path(args.out), payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "out": str(args.out),
                "summary": payload["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
