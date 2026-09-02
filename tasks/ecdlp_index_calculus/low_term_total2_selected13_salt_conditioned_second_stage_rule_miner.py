#!/usr/bin/env python3
"""Mine second-stage public decision lists for selected13 top-5 leaves.

The top-5 replay artifact proves that many held-out wins exist inside the
active-scout shortlist, but a single static tie-breaker only reaches six
held-out exports. This miner consumes those direct replay records and searches
small public decision lists over association features.

The output is intentionally honest: replay labels are used for mining, so the
in-sample decision list is a candidate rule family, not a generalized ECDLP
speedup. Leave-one-transfer-out replay-label validation is reported beside the
in-sample result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import low_term_total2_selected13_public_prefix_min_transfer_probe as min_transfer


SCHEMA = "ecdlp.low_term_total2_selected13_salt_conditioned_second_stage_rule_miner.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_TOP5_REPLAY = DEFAULT_STATE_DIR / "low_term_total2_selected13_salt_conditioned_top5_tiebreaker_replay_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_salt_conditioned_second_stage_rule_miner.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_salt_conditioned_second_stage_rule_miner.h"

ACCEPTED_SOURCE_STATUSES = {
    "SELECTED13_SALT_CONDITIONED_TOP5_TIEBREAKER_REPLAY_IMPROVES_HELDOUT",
    "SELECTED13_SECOND_STAGE_DISJOINT_MISS_DIAGNOSTIC_FOUND_RULE_SELECTION_MISSES",
    "SELECTED13_SECOND_STAGE_DISJOINT_ALL_TOP5_DIAGNOSTIC_BELOW_RHO_COVERAGE",
    "SELECTED13_SECOND_STAGE_MERGED_TOP5_DIAGNOSTIC_BELOW_RHO_COVERAGE",
}
ORDER_MODES = (
    "span",
    "span_root",
    "root",
    "active",
    "position",
    "position_desc",
    "leaf",
    "leaf_desc",
    "span_desc",
)
PUBLIC_FIELDS = ("position", "active", "root", "rowhit", "scout", "span")
LEAF_FIELDS = PUBLIC_FIELDS + ("leaf",)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: Any, default: int = 0) -> int:
    return min_transfer.as_int(value, default)


def merge_counter_dicts(payloads: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for payload in payloads:
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        counts = summary.get(key) if isinstance(summary, dict) else {}
        if not isinstance(counts, dict):
            continue
        for name, value in counts.items():
            out[str(name)] = out.get(str(name), 0) + as_int(value)
    return dict(sorted(out.items()))


def merge_top5_sources(paths: list[Path]) -> dict[str, Any]:
    payloads = [load_json(path) for path in paths]
    replay_records = []
    per_transfer = []
    failures: list[dict[str, Any]] = []
    seen_transfer_sources: dict[int, str] = {}
    for path, payload in zip(paths, payloads):
        status = payload.get("claim_status")
        if status not in ACCEPTED_SOURCE_STATUSES:
            failures.append({"code": "top5_replay_status_unexpected", "claim_status": status, "path": str(path)})
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        if isinstance(summary, dict) and as_int(summary.get("failure_count")):
            failures.append(
                {
                    "code": "top5_replay_source_has_failures",
                    "failure_count": as_int(summary.get("failure_count")),
                    "path": str(path),
                }
            )
        if isinstance(summary, dict) and as_int(summary.get("training_overlap_count")):
            failures.append(
                {
                    "code": "top5_replay_source_has_training_overlap",
                    "overlap_count": as_int(summary.get("training_overlap_count")),
                    "path": str(path),
                }
            )
        for row in payload.get("per_transfer_diagnostics") or []:
            transfer = as_int(row.get("transfer_index"))
            if transfer in seen_transfer_sources:
                failures.append(
                    {
                        "code": "duplicate_transfer_in_merged_top5_sources",
                        "first_path": seen_transfer_sources[transfer],
                        "path": str(path),
                        "transfer_index": transfer,
                    }
                )
            seen_transfer_sources[transfer] = str(path)
            per_transfer.append(row)
        replay_records.extend(payload.get("shortlist_leaf_replays") or [])

    source_transfers = sorted({as_int(row.get("transfer_index")) for row in per_transfer})
    accepted_records = [
        row
        for row in replay_records
        if bool(row.get("accepted_relation_export")) and bool(row.get("below_rho"))
    ]
    rescue_transfers = sorted({as_int(row.get("transfer_index")) for row in accepted_records})
    missing_top5 = sorted(set(source_transfers) - set(rescue_transfers))
    source_missing = sorted(
        {
            as_int(transfer)
            for payload in payloads
            for transfer in ((payload.get("summary") or {}).get("source_missing_transfers") or [])
        }
    )
    return {
        "claim_status": "SELECTED13_SECOND_STAGE_MERGED_TOP5_DIAGNOSTIC_BELOW_RHO_COVERAGE",
        "artifacts": {
            "merged_top5_replay_sources": [str(path) for path in paths],
        },
        "failures": failures,
        "per_transfer_diagnostics": per_transfer,
        "shortlist_leaf_replays": replay_records,
        "summary": {
            "all_source_misses_diagnosed": all(
                bool((payload.get("summary") or {}).get("all_source_misses_diagnosed"))
                for payload in payloads
            ),
            "diagnostic_source_mode": "merged_all_source_transfers",
            "diagnostic_transfer_count": len(source_transfers),
            "failure_count": len(failures),
            "general_ecdlp_algorithm_claimed": False,
            "missing_top5_below_rho_transfers": missing_top5,
            "shortlist_below_rho_replay_count": len(accepted_records),
            "shortlist_below_rho_transfer_count": len(rescue_transfers),
            "shortlist_below_rho_transfers": rescue_transfers,
            "shortlist_replay_count": len(replay_records),
            "source_diagnostic_transfers": source_transfers,
            "source_missing_transfers": source_missing,
            "status_counts": merge_counter_dicts(payloads, "status_counts"),
            "training_overlap_count": sum(
                as_int((payload.get("summary") or {}).get("training_overlap_count"))
                for payload in payloads
            ),
            "training_overlap_transfers": sorted(
                {
                    as_int(transfer)
                    for payload in payloads
                    for transfer in ((payload.get("summary") or {}).get("training_overlap_transfers") or [])
                }
            ),
            "verified": not failures,
            "worker_interpretation": (
                "This merged source combines disjoint top-5 diagnostic chunks so "
                "the rule miner can test cross-chunk selector stability."
            ),
        },
    }


def load_top5_source(paths: list[Path]) -> dict[str, Any]:
    if len(paths) == 1:
        return load_json(paths[0])
    return merge_top5_sources(paths)


def parse_record(row: dict[str, Any]) -> dict[str, Any]:
    association = (row.get("association_top_records") or [{}])[0]
    return {
        "accepted_below_rho": bool(row.get("accepted_relation_export")) and bool(row.get("below_rho")),
        "active": as_int(association.get("active_scout_count_sum")),
        "derived_secret": row.get("derived_secret"),
        "known_positive": bool(row.get("known_positive_transfer")),
        "leaf": as_int((row.get("selected_leaf_indices") or [0])[0]),
        "ops_over_rho": row.get("ops_over_rho"),
        "position": as_int(row.get("shortlist_position")),
        "root": as_int(association.get("hit_root_count_sum")),
        "rowhit": as_int(association.get("row_hit_total_sum")),
        "scout": as_int(association.get("scout_hit_total_sum")),
        "span": as_int(association.get("min_term_span")),
        "status": row.get("status"),
        "transfer_index": as_int(row.get("transfer_index")),
    }


def grouped_by_transfer(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        out.setdefault(as_int(record.get("transfer_index")), []).append(record)
    return out


def order_key(record: dict[str, Any], mode: str) -> tuple[Any, ...]:
    if mode == "span":
        return (as_int(record.get("span")), as_int(record.get("leaf")))
    if mode == "span_root":
        return (
            as_int(record.get("span")),
            -as_int(record.get("root")),
            -as_int(record.get("rowhit")),
            as_int(record.get("leaf")),
        )
    if mode == "root":
        return (
            -as_int(record.get("root")),
            -as_int(record.get("rowhit")),
            as_int(record.get("span")),
            as_int(record.get("leaf")),
        )
    if mode == "active":
        return (
            -as_int(record.get("active")),
            -as_int(record.get("scout")),
            -as_int(record.get("root")),
            as_int(record.get("span")),
            as_int(record.get("leaf")),
        )
    if mode == "position":
        return (as_int(record.get("position")), as_int(record.get("leaf")))
    if mode == "position_desc":
        return (-as_int(record.get("position")), as_int(record.get("leaf")))
    if mode == "leaf":
        return (as_int(record.get("leaf")),)
    if mode == "leaf_desc":
        return (-as_int(record.get("leaf")),)
    if mode == "span_desc":
        return (-as_int(record.get("span")), as_int(record.get("leaf")))
    return (as_int(record.get("position")), as_int(record.get("leaf")))


def predicate_match(record: dict[str, Any], predicate: dict[str, Any]) -> bool:
    if predicate.get("kind") == "all":
        return True
    value = as_int(record.get(str(predicate.get("field"))))
    target = as_int(predicate.get("value"))
    op = str(predicate.get("op"))
    if op == "==":
        return value == target
    if op == "<=":
        return value <= target
    if op == ">=":
        return value >= target
    return False


def predicate_text(predicate: dict[str, Any]) -> str:
    if predicate.get("kind") == "all":
        return "all"
    return f"{predicate.get('field')}{predicate.get('op')}{predicate.get('value')}"


def candidate_rules(
    records: list[dict[str, Any]],
    transfer_indices: list[int],
    *,
    allow_leaf_field: bool,
) -> list[dict[str, Any]]:
    train = [record for record in records if as_int(record.get("transfer_index")) in transfer_indices]
    fields = LEAF_FIELDS if allow_leaf_field else PUBLIC_FIELDS
    predicates = []
    for field in fields:
        values = sorted({as_int(record.get(field)) for record in train})
        for value in values:
            for op in ("==", "<=", ">="):
                predicates.append({"field": field, "kind": "comparison", "op": op, "value": value})
    rules = []
    for predicate in predicates:
        for mode in ORDER_MODES:
            if not allow_leaf_field and mode in {"leaf", "leaf_desc"}:
                continue
            rules.append({"order_mode": mode, "predicate": predicate})
    return rules


def choose_record(
    records_by_transfer: dict[int, list[dict[str, Any]]],
    transfer_index: int,
    rules: list[dict[str, Any]],
    fallback_order_mode: str,
) -> tuple[dict[str, Any], str, str]:
    candidates = records_by_transfer.get(transfer_index, [])
    for rule in rules:
        matching = [record for record in candidates if predicate_match(record, rule["predicate"])]
        if matching:
            return (
                sorted(matching, key=lambda record: order_key(record, str(rule["order_mode"])))[0],
                predicate_text(rule["predicate"]),
                str(rule["order_mode"]),
            )
    return (
        sorted(candidates, key=lambda record: order_key(record, fallback_order_mode))[0],
        "fallback",
        fallback_order_mode,
    )


def evaluate_rules(
    records: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    transfer_indices: list[int],
    fallback_order_mode: str,
) -> dict[str, Any]:
    records_by_transfer = grouped_by_transfer(records)
    per_transfer = []
    accepted = []
    heldout = []
    known = []
    for transfer_index in sorted(transfer_indices):
        chosen, predicate, order_mode = choose_record(records_by_transfer, transfer_index, rules, fallback_order_mode)
        accepted_below = bool(chosen.get("accepted_below_rho"))
        known_positive = bool(chosen.get("known_positive"))
        if accepted_below:
            accepted.append(transfer_index)
            if known_positive:
                known.append(transfer_index)
            else:
                heldout.append(transfer_index)
        per_transfer.append(
            {
                "accepted_below_rho": accepted_below,
                "derived_secret": chosen.get("derived_secret"),
                "known_positive": known_positive,
                "leaf": as_int(chosen.get("leaf")),
                "ops_over_rho": chosen.get("ops_over_rho"),
                "order_mode": order_mode,
                "position": as_int(chosen.get("position")),
                "predicate": predicate,
                "status": chosen.get("status"),
                "transfer_index": transfer_index,
            }
        )
    return {
        "accepted_below_rho_transfer_count": len(accepted),
        "accepted_below_rho_transfers": sorted(accepted),
        "heldout_accepted_below_rho_transfer_count": len(heldout),
        "heldout_accepted_below_rho_transfers": sorted(heldout),
        "known_positive_accepted_below_rho_transfers": sorted(known),
        "per_transfer_records": per_transfer,
    }


def score_tuple(result: dict[str, Any]) -> tuple[int, int]:
    return (
        as_int(result.get("heldout_accepted_below_rho_transfer_count")),
        as_int(result.get("accepted_below_rho_transfer_count")),
    )


def mine_decision_list(
    records: list[dict[str, Any]],
    transfer_indices: list[int],
    *,
    allow_leaf_field: bool,
    max_depth: int,
    fallback_order_mode: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    rules = candidate_rules(records, transfer_indices, allow_leaf_field=allow_leaf_field)
    current = score_tuple(evaluate_rules(records, selected, transfer_indices, fallback_order_mode))
    for _ in range(max_depth):
        scored = []
        for rule in rules:
            trial = selected + [rule]
            result = evaluate_rules(records, trial, transfer_indices, fallback_order_mode)
            scored.append(
                (
                    *score_tuple(result),
                    -len(predicate_text(rule["predicate"])),
                    str(rule["order_mode"]),
                    predicate_text(rule["predicate"]),
                    rule,
                )
            )
        scored.sort(reverse=True)
        best = scored[0]
        if best[:2] <= current:
            break
        selected.append(best[-1])
        current = best[:2]
    return selected


def compact_rule(rule: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "order_mode": str(rule.get("order_mode")),
        "predicate": rule.get("predicate"),
        "predicate_text": predicate_text(rule.get("predicate") or {}),
        "rule_index": index,
    }


def loto_evaluate(
    records: list[dict[str, Any]],
    transfer_indices: list[int],
    *,
    allow_leaf_field: bool,
    max_depth: int,
    fallback_order_mode: str,
) -> dict[str, Any]:
    out = []
    accepted = []
    heldout = []
    for transfer_index in sorted(transfer_indices):
        train = [item for item in transfer_indices if item != transfer_index]
        rules = mine_decision_list(
            records,
            train,
            allow_leaf_field=allow_leaf_field,
            max_depth=max_depth,
            fallback_order_mode=fallback_order_mode,
        )
        result = evaluate_rules(records, rules, [transfer_index], fallback_order_mode)
        row = result["per_transfer_records"][0]
        row["trained_rule_count"] = len(rules)
        row["trained_rules"] = [compact_rule(rule, index) for index, rule in enumerate(rules)]
        out.append(row)
        if row["accepted_below_rho"]:
            accepted.append(transfer_index)
            if not row["known_positive"]:
                heldout.append(transfer_index)
    return {
        "loto_accepted_below_rho_transfer_count": len(accepted),
        "loto_accepted_below_rho_transfers": sorted(accepted),
        "loto_heldout_accepted_below_rho_transfer_count": len(heldout),
        "loto_heldout_accepted_below_rho_transfers": sorted(heldout),
        "loto_records": out,
    }


def bounded_loto_transfer_indices(transfer_indices: list[int], limit: int | None) -> list[int]:
    ordered = sorted(transfer_indices)
    if limit is None or limit <= 0 or limit >= len(ordered):
        return ordered
    if limit == 1:
        return [ordered[0]]
    selected = []
    for index in range(limit):
        position = round(index * (len(ordered) - 1) / (limit - 1))
        selected.append(ordered[position])
    return sorted(set(selected))


def selector_record(
    selector_index: int,
    selector_id: str,
    records: list[dict[str, Any]],
    transfer_indices: list[int],
    *,
    allow_leaf_field: bool,
    max_depth: int,
    fallback_order_mode: str,
    loto_transfer_limit: int | None,
) -> dict[str, Any]:
    rules = mine_decision_list(
        records,
        transfer_indices,
        allow_leaf_field=allow_leaf_field,
        max_depth=max_depth,
        fallback_order_mode=fallback_order_mode,
    )
    result = evaluate_rules(records, rules, transfer_indices, fallback_order_mode)
    loto_transfer_indices = bounded_loto_transfer_indices(transfer_indices, loto_transfer_limit)
    loto = loto_evaluate(
        records,
        loto_transfer_indices,
        allow_leaf_field=allow_leaf_field,
        max_depth=max_depth,
        fallback_order_mode=fallback_order_mode,
    )
    return {
        "selector_id": selector_id,
        "selector_id_u64": min_transfer.digest_u64(
            {
                "allow_leaf_field": allow_leaf_field,
                "fallback_order_mode": fallback_order_mode,
                "max_depth": max_depth,
                "selector_id": selector_id,
            }
        ),
        "selector_index": selector_index,
        "allow_leaf_field": allow_leaf_field,
        "fallback_order_mode": fallback_order_mode,
        "loto_transfer_count": len(loto_transfer_indices),
        "loto_transfer_limit": loto_transfer_limit,
        "loto_transfer_scope": "all" if len(loto_transfer_indices) == len(transfer_indices) else "deterministic_spread_sample",
        "loto_transfers_evaluated": loto_transfer_indices,
        "max_depth": max_depth,
        "rule_count": len(rules),
        "rules": [compact_rule(rule, index) for index, rule in enumerate(rules)],
        **result,
        **loto,
    }


def choose_best(records: list[dict[str, Any]], *, require_no_leaf: bool = False) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if not require_no_leaf or not record.get("allow_leaf_field")
    ]
    candidates.sort(
        key=lambda record: (
            -as_int(record.get("heldout_accepted_below_rho_transfer_count")),
            -as_int(record.get("accepted_below_rho_transfer_count")),
            -as_int(record.get("loto_heldout_accepted_below_rho_transfer_count")),
            as_int(record.get("selector_index")),
        )
    )
    return candidates[0] if candidates else {}


def summarize(selector_records: list[dict[str, Any]], source_payload: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    best = choose_best(selector_records)
    best_no_leaf = choose_best(selector_records, require_no_leaf=True)
    best_loto = max(
        (as_int(record.get("loto_heldout_accepted_below_rho_transfer_count")) for record in selector_records),
        default=0,
    )
    return {
        "best_accepted_below_rho_transfer_count": best.get("accepted_below_rho_transfer_count"),
        "best_accepted_below_rho_transfers": best.get("accepted_below_rho_transfers"),
        "best_heldout_accepted_below_rho_transfer_count": best.get("heldout_accepted_below_rho_transfer_count"),
        "best_heldout_accepted_below_rho_transfers": best.get("heldout_accepted_below_rho_transfers"),
        "best_loto_heldout_accepted_below_rho_transfer_count": best_loto,
        "best_in_sample_selector_loto_heldout_accepted_below_rho_transfer_count": best.get("loto_heldout_accepted_below_rho_transfer_count"),
        "best_no_leaf_selector_id": best_no_leaf.get("selector_id"),
        "best_no_leaf_heldout_accepted_below_rho_transfer_count": best_no_leaf.get("heldout_accepted_below_rho_transfer_count"),
        "best_no_leaf_loto_heldout_accepted_below_rho_transfer_count": best_no_leaf.get("loto_heldout_accepted_below_rho_transfer_count"),
        "best_selector_id": best.get("selector_id"),
        "best_selector_uses_leaf_field": bool(best.get("allow_leaf_field")),
        "direct_replay_label_source_status": source_payload.get("claim_status"),
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "selector_count": len(selector_records),
        "verified": not failures,
        "worker_interpretation": (
            "Small public decision lists improve the one-leaf top-5 tie-breaker in-sample, "
            "but leave-one-transfer-out remains much weaker. Treat this as a mined "
            "candidate family and a feature diagnosis, not as a promoted generalized speedup."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_SECOND_STAGE_RULE_MINER_FAILED"
    if as_int(summary.get("best_heldout_accepted_below_rho_transfer_count")) >= 9:
        return "SELECTED13_SECOND_STAGE_RULE_MINER_IN_SAMPLE_HELDOUT9"
    if as_int(summary.get("best_heldout_accepted_below_rho_transfer_count")) > 6:
        return "SELECTED13_SECOND_STAGE_RULE_MINER_IMPROVES_HELDOUT"
    return "SELECTED13_SECOND_STAGE_RULE_MINER_NO_IMPROVEMENT"


def render_c_header(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = []
    for record in records:
        rows.append(
            "  {"
            f"{as_int(record.get('selector_index'))}ULL, "
            f"{as_int(record.get('selector_id_u64'))}ULL, "
            f"{1 if record.get('allow_leaf_field') else 0}ULL, "
            f"{as_int(record.get('max_depth'))}ULL, "
            f"{as_int(record.get('rule_count'))}ULL, "
            f"{as_int(record.get('accepted_below_rho_transfer_count'))}ULL, "
            f"{as_int(record.get('heldout_accepted_below_rho_transfer_count'))}ULL, "
            f"{as_int(record.get('loto_accepted_below_rho_transfer_count'))}ULL, "
            f"{as_int(record.get('loto_heldout_accepted_below_rho_transfer_count'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_SECOND_STAGE_RULE_MINER_H
#define LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_SECOND_STAGE_RULE_MINER_H

#include <stdint.h>

#define SELECTED13_SECOND_STAGE_RULE_SELECTOR_COUNT {len(records)}
#define SELECTED13_SECOND_STAGE_RULE_BEST_HELDOUT_ACCEPTED_BELOW_COUNT {as_int(summary.get("best_heldout_accepted_below_rho_transfer_count"))}
#define SELECTED13_SECOND_STAGE_RULE_BEST_NO_LEAF_HELDOUT_ACCEPTED_BELOW_COUNT {as_int(summary.get("best_no_leaf_heldout_accepted_below_rho_transfer_count"))}
#define SELECTED13_SECOND_STAGE_RULE_BEST_LOTO_HELDOUT_ACCEPTED_BELOW_COUNT {as_int(summary.get("best_loto_heldout_accepted_below_rho_transfer_count"))}

typedef struct {{
  uint64_t selector_index;
  uint64_t selector_id_u64;
  uint64_t allow_leaf_field;
  uint64_t max_depth;
  uint64_t rule_count;
  uint64_t accepted_below_rho_transfer_count;
  uint64_t heldout_accepted_below_rho_transfer_count;
  uint64_t loto_accepted_below_rho_transfer_count;
  uint64_t loto_heldout_accepted_below_rho_transfer_count;
}} selected13_second_stage_rule_selector_t;

static const selected13_second_stage_rule_selector_t SELECTED13_SECOND_STAGE_RULE_SELECTORS[] = {{
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
  uint64_t selector_count =
      sizeof(SELECTED13_SECOND_STAGE_RULE_SELECTORS) / sizeof(SELECTED13_SECOND_STAGE_RULE_SELECTORS[0]);
  uint64_t best_heldout = 0;
  uint64_t best_no_leaf = 0;
  uint64_t best_loto = 0;

  if (selector_count != SELECTED13_SECOND_STAGE_RULE_SELECTOR_COUNT) failure_count++;
  if (selector_count == 0ULL) failure_count++;

  for (size_t i = 0; i < selector_count; i++) {{
    const selected13_second_stage_rule_selector_t *selector = &SELECTED13_SECOND_STAGE_RULE_SELECTORS[i];
    if (selector->selector_id_u64 == 0ULL) failure_count++;
    if (selector->heldout_accepted_below_rho_transfer_count > best_heldout) {{
      best_heldout = selector->heldout_accepted_below_rho_transfer_count;
    }}
    if (!selector->allow_leaf_field &&
        selector->heldout_accepted_below_rho_transfer_count > best_no_leaf) {{
      best_no_leaf = selector->heldout_accepted_below_rho_transfer_count;
    }}
    if (selector->loto_heldout_accepted_below_rho_transfer_count > best_loto) {{
      best_loto = selector->loto_heldout_accepted_below_rho_transfer_count;
    }}
  }}

  if (best_heldout != SELECTED13_SECOND_STAGE_RULE_BEST_HELDOUT_ACCEPTED_BELOW_COUNT) failure_count++;
  if (best_no_leaf != SELECTED13_SECOND_STAGE_RULE_BEST_NO_LEAF_HELDOUT_ACCEPTED_BELOW_COUNT) failure_count++;
  if (best_loto != SELECTED13_SECOND_STAGE_RULE_BEST_LOTO_HELDOUT_ACCEPTED_BELOW_COUNT) failure_count++;
  printf("selected13_second_stage_rule_miner_preflight selectors=%llu best_heldout=%llu best_no_leaf=%llu best_loto=%llu failures=%llu\\n",
         (unsigned long long)selector_count,
         (unsigned long long)best_heldout,
         (unsigned long long)best_no_leaf,
         (unsigned long long)best_loto,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_second_stage_rule_miner_preflight_") as tmp:
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
    top5_replays = [Path(path) for path in args.top5_replay]
    source = load_top5_source(top5_replays)
    failures: list[dict[str, Any]] = []
    failures.extend(source.get("failures") or [])
    if source.get("claim_status") not in ACCEPTED_SOURCE_STATUSES:
        failures.append({"code": "top5_replay_status_unexpected", "claim_status": source.get("claim_status")})
    records = [parse_record(row) for row in source.get("shortlist_leaf_replays") or []]
    if not records:
        failures.append({"code": "no_shortlist_records"})
    transfer_indices = sorted({as_int(record.get("transfer_index")) for record in records})
    selector_records = []
    index = 0
    if args.selector_family == "public":
        selector_families = [False]
    elif args.selector_family == "leaf":
        selector_families = [True]
    else:
        selector_families = [False, True]
    for allow_leaf_field in selector_families:
        for max_depth in args.max_depths:
            selector_records.append(
                selector_record(
                    index,
                    f"second_stage_{'leaf' if allow_leaf_field else 'public'}_depth{max_depth}",
                    records,
                    transfer_indices,
                    allow_leaf_field=allow_leaf_field,
                    max_depth=max_depth,
                    fallback_order_mode=args.fallback_order_mode,
                    loto_transfer_limit=args.loto_transfer_limit,
                )
            )
            index += 1
    summary = summarize(selector_records, source, failures)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, summary),
        "parameters": {
            "fallback_order_mode": args.fallback_order_mode,
            "loto_transfer_limit": args.loto_transfer_limit,
            "max_depths": args.max_depths,
            "selector_family": args.selector_family,
            "top5_replay": str(top5_replays[0]) if len(top5_replays) == 1 else None,
            "top5_replays": [str(path) for path in top5_replays],
        },
        "summary": summary,
        "selector_records": selector_records,
        "failures": failures,
        "honesty_boundary": {
            "direct_replay_labels_used_for_mining": True,
            "general_ecdlp_algorithm_claimed": False,
            "leave_one_transfer_out_reported": True,
            "selection_cost_note": "Each decision-list selector chooses one leaf from the direct-replayed active-scout top-5 shortlist; the miner does not charge all five leaves as an end-to-end policy.",
        },
    }


def parse_depths(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top5-replay", type=Path, nargs="+", default=[DEFAULT_TOP5_REPLAY])
    parser.add_argument("--max-depths", type=parse_depths, default=[1, 2, 3])
    parser.add_argument("--fallback-order-mode", default="span_root")
    parser.add_argument("--selector-family", choices=("all", "public", "leaf"), default="all")
    parser.add_argument("--loto-transfer-limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["selector_records"], payload["summary"]))
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
