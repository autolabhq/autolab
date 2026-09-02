#!/usr/bin/env python3
"""Minimize the selected13 9943 public-prefix export and test 9981 transfer.

The 9943 public-prefix expansion probe found a bounded below-rho target-level
recovery.  This follow-up asks two narrower questions:

* can the accepted 9943 leaf set be minimized to the event-bearing leaves, and
* does the same public-prefix selector family transfer to 9981?

This is still a verifier artifact, not a generalized ECDLP algorithm proof.  A
held-out transfer is only counted as generalized evidence if it also derives a
target secret below rho.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import ffe_single_hit_root_relation_replay_probe as replay_probe


SCHEMA = "ecdlp.low_term_total2_selected13_public_prefix_min_transfer_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_PUBLIC_PREFIX = DEFAULT_STATE_DIR / "low_term_total2_selected13_9943_public_prefix_expansion_probe.json"
DEFAULT_DIRECT_AUDIT = DEFAULT_STATE_DIR / "low_term_total2_selected13_direct_verification_audit_9981_9943_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_public_prefix_min_transfer_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_public_prefix_min_transfer_probe.h"

TARGET = "22050.cf1@11731"
TRANSFER_ROWS = {
    9943: [
        "22050.cf1@11731:uniform:256:salt167",
        "22050.cf1@11731:uniform:256:salt175",
    ],
    9981: [
        "22050.cf1@11731:uniform:256:salt171",
        "22050.cf1@11731:uniform:256:salt173",
    ],
}

DEFAULT_TRANSFER_MODES = (
    "pool_order",
    "low_leaf_index",
    "double_pair_first",
    "high_double_pair_count",
    "low_term_span",
    "low_monic_c",
    "hybrid_double_monic_b",
    "hybrid_support_monic_b",
)
DEFAULT_TRANSFER_TOP_KS = (3, 5, 8, 13)

CLASS_CODES = {
    "source_best_public_prefix": 1,
    "source_event_leaf_subset": 2,
    "transfer_pair_public_prefix": 3,
    "transfer_minimized_leaf28_pair": 4,
}

STATUS_CODES = {
    "ACCEPTED_DERIVED_BELOW_RHO": 1,
    "ACCEPTED_DERIVED_OVER_RHO": 2,
    "PUBLIC_KEY_VERIFIED_NO_DERIVATION": 3,
    "RANK2_UNVERIFIED": 4,
    "RANK1_RELATION_ONLY": 5,
    "HIT_ROOT_ONLY": 6,
    "NO_RELATION": 7,
    "CONTEXT_ERROR": 8,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def round_or_none(value: Any, digits: int = 8) -> float | None:
    numeric = as_float(value)
    return None if numeric is None else round(numeric, digits)


def digest_u64(raw: Any) -> int:
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def row_leaf_map(items: list[dict[str, Any]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for item in items or []:
        row_key = str(item.get("row_key") or "")
        leaves = {as_int(leaf) for leaf in item.get("leaf_indices") or []}
        if row_key and leaves:
            out.setdefault(row_key, set()).update(leaves)
    return out


def compact_leaf_map(raw: dict[str, set[int]], row_order: list[str]) -> list[dict[str, Any]]:
    return [
        {"leaf_indices": sorted(raw[row_key]), "row_key": row_key}
        for row_key in row_order
        if raw.get(row_key)
    ]


def leaf_signature(raw: dict[str, set[int]], row_order: list[str]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((row_key, tuple(sorted(raw.get(row_key, set())))) for row_key in row_order)


def best_public_prefix_candidate(public_prefix: dict[str, Any]) -> dict[str, Any]:
    best_id = (public_prefix.get("summary") or {}).get("best_accepted_candidate_id")
    for candidate in public_prefix.get("public_prefix_candidates") or []:
        if candidate.get("candidate_id") == best_id:
            return candidate
    accepted = [
        candidate
        for candidate in public_prefix.get("public_prefix_candidates") or []
        if candidate.get("accepted_relation_export")
    ]
    accepted.sort(
        key=lambda item: (
            0 if item.get("below_rho") else 1,
            as_float(item.get("ops_over_rho")) if as_float(item.get("ops_over_rho")) is not None else 999.0,
        )
    )
    return accepted[0] if accepted else {}


def event_leaf_map(candidate: dict[str, Any]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for row in candidate.get("row_summaries") or []:
        row_key = str(row.get("row_key") or "")
        for event in row.get("event_summaries") or []:
            if event.get("leaf_index") is not None:
                out[row_key].add(as_int(event.get("leaf_index")))
    return dict(out)


def nonempty_subsets(values: set[int]) -> list[set[int]]:
    ordered = sorted(values)
    subsets: list[set[int]] = [set()]
    for size in range(1, len(ordered) + 1):
        subsets.extend(set(combo) for combo in itertools.combinations(ordered, size))
    return subsets


def event_subset_candidates(
    best_candidate: dict[str, Any],
    row_order: list[str],
) -> list[dict[str, Any]]:
    event_leaves = event_leaf_map(best_candidate)
    per_row = [(row_key, nonempty_subsets(event_leaves.get(row_key, set()))) for row_key in row_order]
    out = []
    seen: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
    for left in per_row[0][1]:
        for right in per_row[1][1]:
            leaf_map: dict[str, set[int]] = {}
            if left:
                leaf_map[per_row[0][0]] = set(left)
            if right:
                leaf_map[per_row[1][0]] = set(right)
            if not leaf_map:
                continue
            signature = leaf_signature(leaf_map, row_order)
            if signature in seen:
                continue
            seen.add(signature)
            out.append(
                {
                    "candidate_class": "source_event_leaf_subset",
                    "candidate_class_code": CLASS_CODES["source_event_leaf_subset"],
                    "mode": best_candidate.get("mode"),
                    "parent_candidate_id": best_candidate.get("candidate_id"),
                    "selected_leaf_count": sum(len(items) for items in leaf_map.values()),
                    "selected_leaf_map": compact_leaf_map(leaf_map, row_order),
                    "top_k": best_candidate.get("top_k"),
                    "transfer_index": 9943,
                }
            )
    out.sort(key=lambda item: (as_int(item.get("selected_leaf_count")), json.dumps(item["selected_leaf_map"])))
    return out


def status_for(result: dict[str, Any]) -> str:
    if bool(result.get("public_key_verified")) and bool(result.get("derived")) and bool(result.get("below_rho")):
        return "ACCEPTED_DERIVED_BELOW_RHO"
    if bool(result.get("public_key_verified")) and bool(result.get("derived")):
        return "ACCEPTED_DERIVED_OVER_RHO"
    if bool(result.get("public_key_verified")):
        return "PUBLIC_KEY_VERIFIED_NO_DERIVATION"
    if as_int(result.get("rank")) >= 2 or as_int(result.get("relation_count")) >= 2:
        return "RANK2_UNVERIFIED"
    if as_int(result.get("rank")) > 0 or as_int(result.get("relation_count")) > 0:
        return "RANK1_RELATION_ONLY"
    hit_roots = 0
    for row in result.get("rows") or []:
        hit_roots += as_int((row.get("scan") or {}).get("selected_hit_roots"))
    if hit_roots:
        return "HIT_ROOT_ONLY"
    return "NO_RELATION"


def compact_replay_rows(result: dict[str, Any], event_limit: int) -> list[dict[str, Any]]:
    rows = []
    for row in result.get("rows") or []:
        scan = row.get("scan") or {}
        events = scan.get("event_summaries") or []
        rows.append(
            {
                "event_summary_count": len(events),
                "event_summaries": events[:event_limit],
                "hit_event_count": as_int(scan.get("selected_hit_events")),
                "rank": as_int(scan.get("rank")),
                "relation_count": as_int(scan.get("relation_count")),
                "row_key": row.get("row_key"),
                "selected_hit_roots": as_int(scan.get("selected_hit_roots")),
                "selected_leaf_indices": scan.get("selected_leaf_indices") or [],
            }
        )
    return rows


def compact_candidate(
    index: int,
    candidate: dict[str, Any],
    result: dict[str, Any],
    row_events: list[tuple[str, dict[str, Any]]],
    event_limit: int,
) -> dict[str, Any]:
    status = status_for(result)
    accepted = status in {"ACCEPTED_DERIVED_BELOW_RHO", "ACCEPTED_DERIVED_OVER_RHO"}
    return {
        **candidate,
        "accepted_relation_export": accepted,
        "below_rho": bool(result.get("below_rho")),
        "candidate_id": f"min_transfer_{digest_u64([candidate, index]):016x}",
        "candidate_id_u64": digest_u64([candidate, index]),
        "candidate_index": index,
        "derived_secret": result.get("derived_secret"),
        "duplicate_form_count": as_int(result.get("duplicate_form_count")),
        "generic_rho_steps": as_int(result.get("generic_rho_steps")),
        "ops": as_int(result.get("ops")),
        "ops_over_rho": round_or_none(result.get("ops_over_rho")),
        "public_key_verified": bool(result.get("public_key_verified")),
        "rank": as_int(result.get("rank")),
        "relation_count": as_int(result.get("relation_count")),
        "relation_derived_ecdlp": bool(result.get("public_key_verified")) and bool(result.get("derived")),
        "row_event_count": len(row_events),
        "row_summaries": compact_replay_rows(result, event_limit),
        "status": status,
        "status_code": STATUS_CODES.get(status, 0),
        "unique_form_count": as_int(result.get("unique_form_count")),
    }


def materialize_contexts_for(
    verifier: Any,
    records: list[dict[str, Any]],
    config_source: dict[str, Any],
    specs_by_target: dict[str, dict[str, dict[str, Any]]],
    transfer_index: int,
    args: argparse.Namespace,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    row_order = TRANSFER_ROWS[transfer_index]
    return replay_probe.materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        {"target": TARGET, "transfer_index": transfer_index, "top_k": args.context_top_k},
        row_order,
        args,
        context_cache,
    )


def public_prefix_leaf_map(contexts: dict[str, dict[str, Any]], mode: str, top_k: int) -> dict[str, set[int]]:
    salt_neighborhood_probe = replay_probe.salt_neighborhood_probe
    out: dict[str, set[int]] = {}
    for row_key in sorted(contexts):
        context = contexts[row_key]
        built = context["built"]
        local_args = context["local_args"]
        seed = f"{local_args.filter_seed_prefix}:{built['target']}:{mode}"
        out[row_key] = salt_neighborhood_probe.selected_prefix(
            context["feature_rows"],
            mode,
            seed,
            top_k,
        )
    return out


def build_candidate_inputs(
    public_prefix: dict[str, Any],
    contexts_by_transfer: dict[int, dict[str, dict[str, Any]]],
    modes: list[str],
    top_ks: list[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[tuple[str, tuple[int, ...]], ...], str]] = set()
    best = best_public_prefix_candidate(public_prefix)
    row_order_9943 = TRANSFER_ROWS[9943]
    best_map = row_leaf_map(best.get("selected_leaf_map") or [])
    if best_map:
        out.append(
            {
                "candidate_class": "source_best_public_prefix",
                "candidate_class_code": CLASS_CODES["source_best_public_prefix"],
                "mode": best.get("mode"),
                "parent_candidate_id": best.get("candidate_id"),
                "selected_leaf_count": sum(len(items) for items in best_map.values()),
                "selected_leaf_map": compact_leaf_map(best_map, row_order_9943),
                "top_k": best.get("top_k"),
                "transfer_index": 9943,
            }
        )
    out.extend(event_subset_candidates(best, row_order_9943))

    row_order_9981 = TRANSFER_ROWS[9981]
    minimized_transfer = {row_order_9981[0]: {28}, row_order_9981[1]: {28}}
    out.append(
        {
            "candidate_class": "transfer_minimized_leaf28_pair",
            "candidate_class_code": CLASS_CODES["transfer_minimized_leaf28_pair"],
            "mode": "leaf28_pair",
            "parent_candidate_id": best.get("candidate_id"),
            "selected_leaf_count": 2,
            "selected_leaf_map": compact_leaf_map(minimized_transfer, row_order_9981),
            "top_k": None,
            "transfer_index": 9981,
        }
    )
    for mode in modes:
        for top_k in top_ks:
            leaf_map = public_prefix_leaf_map(contexts_by_transfer[9981], mode, top_k)
            signature = (9981, leaf_signature(leaf_map, row_order_9981), mode)
            if signature in seen:
                continue
            seen.add(signature)
            out.append(
                {
                    "candidate_class": "transfer_pair_public_prefix",
                    "candidate_class_code": CLASS_CODES["transfer_pair_public_prefix"],
                    "mode": mode,
                    "selected_leaf_count": sum(len(items) for items in leaf_map.values()),
                    "selected_leaf_map": compact_leaf_map(leaf_map, row_order_9981),
                    "top_k": top_k,
                    "transfer_index": 9981,
                }
            )
    return out


def replay_all(args: argparse.Namespace, public_prefix: dict[str, Any]) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
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
    records = verifier.load_records()
    replay_args = argparse.Namespace(
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
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    contexts_by_transfer: dict[int, dict[str, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for transfer_index in sorted(TRANSFER_ROWS):
        contexts, errors = materialize_contexts_for(
            verifier,
            records,
            config_source,
            specs_by_target,
            transfer_index,
            replay_args,
            context_cache,
        )
        contexts_by_transfer[transfer_index] = contexts
        for error in errors:
            failures.append({"code": "context_materialization_error", "transfer_index": transfer_index, "error": error})
    if failures:
        return [], radius, failures

    candidate_inputs = build_candidate_inputs(public_prefix, contexts_by_transfer, args.transfer_modes, args.transfer_top_ks)
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    records_out = []
    for index, candidate in enumerate(candidate_inputs):
        transfer_index = as_int(candidate.get("transfer_index"))
        leaf_map = row_leaf_map(candidate.get("selected_leaf_map") or [])
        result, row_events = replay_probe.replay_selection(
            verifier,
            leaf_map,
            contexts_by_transfer[transfer_index],
            scan_cache,
            args.event_summary_limit,
        )
        records_out.append(compact_candidate(index, candidate, result, row_events, args.event_summary_limit))
    return records_out, radius, failures


def best_record(records: list[dict[str, Any]], transfer_index: int, *, require_below: bool = False) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if as_int(record.get("transfer_index"), -1) == transfer_index
        and record.get("accepted_relation_export")
        and (not require_below or record.get("below_rho"))
    ]
    candidates.sort(
        key=lambda item: (
            as_int(item.get("selected_leaf_count")),
            as_float(item.get("ops_over_rho")) if as_float(item.get("ops_over_rho")) is not None else 999.0,
            -as_int(item.get("rank")),
        )
    )
    return candidates[0] if candidates else {}


def summarize(records: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    by_transfer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_transfer[as_int(record.get("transfer_index"), -1)].append(record)
    transfer_summaries: dict[str, dict[str, Any]] = {}
    for transfer_index, rows in sorted(by_transfer.items()):
        accepted = [row for row in rows if row.get("accepted_relation_export")]
        below = [row for row in accepted if row.get("below_rho")]
        statuses = Counter(str(row.get("status")) for row in rows)
        best = best_record(rows, transfer_index)
        best_below = best_record(rows, transfer_index, require_below=True)
        transfer_summaries[str(transfer_index)] = {
            "accepted_relation_export_count": len(accepted),
            "below_rho_accepted_relation_export_count": len(below),
            "best_accepted_candidate_id": best.get("candidate_id"),
            "best_accepted_derived_secret": best.get("derived_secret"),
            "best_accepted_mode": best.get("mode"),
            "best_accepted_ops_over_rho": best.get("ops_over_rho"),
            "best_accepted_selected_leaf_count": best.get("selected_leaf_count"),
            "best_below_rho_candidate_id": best_below.get("candidate_id"),
            "candidate_count": len(rows),
            "max_rank": max((as_int(row.get("rank")) for row in rows), default=0),
            "max_relation_count": max((as_int(row.get("relation_count")) for row in rows), default=0),
            "status_counts": dict(sorted(statuses.items())),
        }
    best_9943 = best_record(records, 9943, require_below=True)
    best_9981 = best_record(records, 9981)
    return {
        "candidate_count": len(records),
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "heldout_9981_below_rho_relation_derived": bool(best_record(records, 9981, require_below=True)),
        "heldout_9981_relation_derived": bool(best_9981),
        "minimized_9943_candidate_id": best_9943.get("candidate_id"),
        "minimized_9943_derived_secret": best_9943.get("derived_secret"),
        "minimized_9943_ops_over_rho": best_9943.get("ops_over_rho"),
        "minimized_9943_selected_leaf_count": best_9943.get("selected_leaf_count"),
        "pollard_rho_speedup_claimed": bool(best_9943),
        "relation_derived_ecdlp": any(row.get("accepted_relation_export") for row in records),
        "transfer_9981_best_candidate_id": best_9981.get("candidate_id"),
        "transfer_9981_best_derived_secret": best_9981.get("derived_secret"),
        "transfer_9981_best_ops_over_rho": best_9981.get("ops_over_rho"),
        "transfer_9981_best_selected_leaf_count": best_9981.get("selected_leaf_count"),
        "transfer_summaries": transfer_summaries,
        "verified": not failures,
        "worker_interpretation": (
            "The 9943 export minimizes to a two-leaf public-prefix event core. "
            "The same prefix family derives 9981 only over rho, so this is "
            "portability evidence but not a generalized below-rho algorithm."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_FAILED"
    if summary.get("heldout_9981_below_rho_relation_derived"):
        return "SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_HELDOUT_BELOW_RHO"
    if summary.get("heldout_9981_relation_derived"):
        return "SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_HELDOUT_OVER_RHO"
    if summary.get("minimized_9943_candidate_id"):
        return "SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_9943_MINIMIZED_ONLY"
    return "SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_NO_EXPORT"


def render_c_header(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        ops_scaled = 0
        ops = as_float(record.get("ops_over_rho"))
        if ops is not None:
            ops_scaled = int(round(ops * 1_000_000))
        rows.append(
            "  {"
            f"{as_int(record.get('candidate_index'))}ULL, "
            f"{as_int(record.get('candidate_id_u64'))}ULL, "
            f"{as_int(record.get('transfer_index'))}ULL, "
            f"{as_int(record.get('candidate_class_code'))}ULL, "
            f"{as_int(record.get('selected_leaf_count'))}ULL, "
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
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_PROBE_H

#include <stdint.h>

#define SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_CANDIDATE_COUNT {len(records)}
#define SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_9943_BELOW_RHO_ACCEPTED_COUNT {sum(1 for row in records if as_int(row.get('transfer_index')) == 9943 and row.get('accepted_relation_export') and row.get('below_rho'))}
#define SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_9981_ACCEPTED_COUNT {sum(1 for row in records if as_int(row.get('transfer_index')) == 9981 and row.get('accepted_relation_export'))}
#define SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_9981_BELOW_RHO_ACCEPTED_COUNT {sum(1 for row in records if as_int(row.get('transfer_index')) == 9981 and row.get('accepted_relation_export') and row.get('below_rho'))}

typedef struct {{
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t transfer_index;
  uint64_t candidate_class_code;
  uint64_t selected_leaf_count;
  uint64_t below_rho;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t rank;
  uint64_t relation_count;
  uint64_t derived_secret;
  uint64_t ops_over_rho_scaled_1e6;
  uint64_t status_code;
}} selected13_public_prefix_min_transfer_candidate_t;

static const selected13_public_prefix_min_transfer_candidate_t SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_CANDIDATES[] = {{
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
  uint64_t candidate_count =
      sizeof(SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_CANDIDATES) / sizeof(SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_CANDIDATES[0]);
  uint64_t accepted_9943_below = 0;
  uint64_t accepted_9981 = 0;
  uint64_t accepted_9981_below = 0;

  if (candidate_count != SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_CANDIDATE_COUNT) failure_count++;
  if (candidate_count == 0ULL) failure_count++;

  for (size_t i = 0; i < candidate_count; i++) {{
    const selected13_public_prefix_min_transfer_candidate_t *candidate = &SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_CANDIDATES[i];
    if (candidate->candidate_id_u64 == 0ULL) failure_count++;
    if (candidate->selected_leaf_count == 0ULL) failure_count++;
    if (candidate->status_code == 0ULL) failure_count++;
    if (candidate->relation_derived_ecdlp && !candidate->public_key_verified) failure_count++;
    if (candidate->relation_derived_ecdlp && candidate->derived_secret == 0ULL) failure_count++;
    if (candidate->transfer_index == 9943ULL && candidate->relation_derived_ecdlp && candidate->below_rho) {{
      accepted_9943_below++;
    }}
    if (candidate->transfer_index == 9981ULL && candidate->relation_derived_ecdlp) {{
      accepted_9981++;
      if (candidate->below_rho) accepted_9981_below++;
    }}
  }}

  if (accepted_9943_below != SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_9943_BELOW_RHO_ACCEPTED_COUNT) failure_count++;
  if (accepted_9981 != SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_9981_ACCEPTED_COUNT) failure_count++;
  if (accepted_9981_below != SELECTED13_PUBLIC_PREFIX_MIN_TRANSFER_9981_BELOW_RHO_ACCEPTED_COUNT) failure_count++;

  printf("selected13_public_prefix_min_transfer_preflight candidates=%llu accepted_9943_below=%llu accepted_9981=%llu accepted_9981_below=%llu failures=%llu\\n",
         (unsigned long long)candidate_count,
         (unsigned long long)accepted_9943_below,
         (unsigned long long)accepted_9981,
         (unsigned long long)accepted_9981_below,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_public_prefix_min_transfer_preflight_") as tmp:
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
    public_prefix = load_json(Path(args.public_prefix))
    direct_audit = load_json(Path(args.direct_audit))
    failures: list[dict[str, Any]] = []
    if public_prefix.get("claim_status") != "SELECTED13_9943_PUBLIC_PREFIX_EXPANSION_BELOW_RHO_EXPORT":
        failures.append({"code": "public_prefix_status_unexpected", "claim_status": public_prefix.get("claim_status")})
    if direct_audit.get("claim_status") != "SELECTED13_DIRECT_VERIFICATION_REQUIRES_FRESH_RELATIONS":
        failures.append({"code": "direct_audit_status_unexpected", "claim_status": direct_audit.get("claim_status")})
    records, radius, replay_failures = replay_all(args, public_prefix)
    failures.extend(replay_failures)
    summary = summarize(records, failures)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, summary),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "config_source": str(Path(args.config_source)),
            "context_top_k": args.context_top_k,
            "direct_audit": str(Path(args.direct_audit)),
            "direct_source": str(Path(args.direct_source)),
            "public_prefix": str(Path(args.public_prefix)),
            "radius": radius,
            "target": TARGET,
            "transfer_modes": args.transfer_modes,
            "transfer_rows": TRANSFER_ROWS,
            "transfer_source": str(Path(args.transfer_source)),
            "transfer_top_ks": args.transfer_top_ks,
        },
        "summary": summary,
        "min_transfer_candidates": records,
        "failures": failures,
        "honesty_boundary": {
            "general_ecdlp_algorithm_claimed": False,
            "heldout_9981_below_rho_relation_derived": summary["heldout_9981_below_rho_relation_derived"],
            "heldout_9981_relation_derived": summary["heldout_9981_relation_derived"],
            "target_9943_below_rho_relation_derived": bool(summary.get("minimized_9943_candidate_id")),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-prefix", type=Path, default=DEFAULT_PUBLIC_PREFIX)
    parser.add_argument("--direct-audit", type=Path, default=DEFAULT_DIRECT_AUDIT)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--transfer-modes", type=parse_csv, default=list(DEFAULT_TRANSFER_MODES))
    parser.add_argument("--transfer-top-ks", type=parse_int_csv, default=list(DEFAULT_TRANSFER_TOP_KS))
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
    parser.add_argument("--event-summary-limit", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["min_transfer_candidates"]))
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
