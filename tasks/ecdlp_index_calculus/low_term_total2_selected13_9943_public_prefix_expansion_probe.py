#!/usr/bin/env python3
"""Replay public-prefix leaf expansions for the selected13 9943 target.

The second-stage lattice exhausted the visible 9943 worklist leaves, and the
9842 salt-neighbor carryover died under the 176 -> 175 salt mutation.  This
probe changes the relation source: it materializes the actual 9943 verifier
contexts, selects leaves from public feature rankings, keeps only candidates
that add at least one leaf outside the exhausted visible universe, and replays
those candidates through the verifier path.

The output is a target-level verifier artifact.  A below-rho public-key-verified
derivation is evidence for this bounded 9943 target, not a generalized ECDLP
algorithm by itself.
"""

from __future__ import annotations

import argparse
import hashlib
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


SCHEMA = "ecdlp.low_term_total2_selected13_9943_public_prefix_expansion_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LATTICE = DEFAULT_STATE_DIR / "low_term_total2_selected13_9943_second_stage_lattice_probe.json"
DEFAULT_SALT_NEIGHBOR = DEFAULT_STATE_DIR / "low_term_total2_selected13_9943_salt_neighbor_carryover_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_9943_public_prefix_expansion_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_9943_public_prefix_expansion_probe.h"

TARGET = "22050.cf1@11731"
TRANSFER = 9943
ROW_KEY_ORDER = [
    "22050.cf1@11731:uniform:256:salt167",
    "22050.cf1@11731:uniform:256:salt175",
]
DEFAULT_VISIBLE_UNIVERSE = {
    "22050.cf1@11731:uniform:256:salt167": [8, 56, 65, 79, 90],
    "22050.cf1@11731:uniform:256:salt175": [8, 65, 79, 90],
}
DEFAULT_MODES = (
    "pool_order",
    "low_leaf_index",
    "double_pair_first",
    "no_double_pair_first",
    "high_double_pair_count",
    "low_double_pair_count",
    "low_term_support",
    "low_term_span",
    "low_monic_b",
    "low_monic_c",
    "hybrid_double_monic_b",
    "hybrid_support_monic_b",
    "zero_double_pair_low_monic_b",
    "high_shape_concentration",
    "hash_leaf",
)
DEFAULT_TOP_KS = (1, 2, 3, 5, 8, 13)

CLASS_CODES = {
    "pair_public_prefix": 1,
    "seed_salt167_public_prefix_salt175": 2,
    "public_prefix_salt167_seed_salt175": 3,
}

STATUS_CODES = {
    "ACCEPTED_RELATION_DERIVED_BELOW_RHO": 1,
    "ACCEPTED_RELATION_DERIVED_OVER_RHO": 2,
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


def visible_universe(lattice: dict[str, Any]) -> dict[str, set[int]]:
    raw = lattice.get("summary", {}).get("leaf_universe") or lattice.get("leaf_universe")
    if not isinstance(raw, dict):
        raw = DEFAULT_VISIBLE_UNIVERSE
    out: dict[str, set[int]] = {}
    for row_key, leaves in raw.items():
        out[str(row_key)] = {as_int(leaf) for leaf in leaves or []}
    return out


def compact_leaf_map(raw: dict[str, set[int]]) -> list[dict[str, Any]]:
    return [
        {"leaf_indices": sorted(raw[row_key]), "row_key": row_key}
        for row_key in ROW_KEY_ORDER
        if raw.get(row_key)
    ]


def leaf_signature(raw: dict[str, set[int]]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((row_key, tuple(sorted(raw.get(row_key, set())))) for row_key in ROW_KEY_ORDER)


def new_leaf_map(raw: dict[str, set[int]], visible: dict[str, set[int]]) -> dict[str, set[int]]:
    return {
        row_key: set(raw.get(row_key, set())) - set(visible.get(row_key, set()))
        for row_key in ROW_KEY_ORDER
        if set(raw.get(row_key, set())) - set(visible.get(row_key, set()))
    }


def prefix_seed(context: dict[str, Any], mode: str) -> str:
    built = context["built"]
    local_args = context["local_args"]
    return f"{local_args.filter_seed_prefix}:{built['target']}:{mode}"


def selected_prefixes(
    contexts: dict[str, dict[str, Any]],
    modes: list[str],
    top_ks: list[int],
) -> dict[str, dict[str, dict[int, set[int]]]]:
    salt_neighborhood_probe = replay_probe.salt_neighborhood_probe
    out: dict[str, dict[str, dict[int, set[int]]]] = {}
    for row_key, context in contexts.items():
        out[row_key] = {}
        for mode in modes:
            out[row_key][mode] = {}
            seed = prefix_seed(context, mode)
            for top_k in top_ks:
                out[row_key][mode][top_k] = salt_neighborhood_probe.selected_prefix(
                    context["feature_rows"],
                    mode,
                    seed,
                    top_k,
                )
    return out


def prefix_profiles(
    contexts: dict[str, dict[str, Any]],
    modes: list[str],
    top_ks: list[int],
    prefixes: dict[str, dict[str, dict[int, set[int]]]],
    visible: dict[str, set[int]],
) -> list[dict[str, Any]]:
    profiles = []
    salt_neighborhood_probe = replay_probe.salt_neighborhood_probe
    for row_key in ROW_KEY_ORDER:
        context = contexts.get(row_key)
        if not context:
            continue
        for mode in modes:
            for top_k in top_ks:
                leaves = prefixes[row_key][mode][top_k]
                profiles.append(
                    {
                        "mode": mode,
                        "new_leaf_indices": sorted(set(leaves) - visible.get(row_key, set())),
                        "public_profile": salt_neighborhood_probe.selected_leaf_public_profile(
                            context["feature_rows"],
                            leaves,
                        ),
                        "row_key": row_key,
                        "selected_leaf_indices": sorted(leaves),
                        "top_k": top_k,
                    }
                )
    return profiles


def candidate_leaf_sets(
    prefixes: dict[str, dict[str, dict[int, set[int]]]],
    modes: list[str],
    top_ks: list[int],
    visible: dict[str, set[int]],
) -> list[dict[str, Any]]:
    seed = {ROW_KEY_ORDER[0]: {8}, ROW_KEY_ORDER[1]: {8}}
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()

    def add(candidate_class: str, mode: str, top_k: int, leaves: dict[str, set[int]]) -> None:
        normalized = {row_key: set(leaves.get(row_key, set())) for row_key in ROW_KEY_ORDER if leaves.get(row_key)}
        signature = leaf_signature(normalized)
        if signature in seen:
            return
        seen.add(signature)
        new_leaves = new_leaf_map(normalized, visible)
        new_leaf_count = sum(len(items) for items in new_leaves.values())
        if new_leaf_count <= 0:
            return
        candidates.append(
            {
                "candidate_class": candidate_class,
                "candidate_class_code": CLASS_CODES.get(candidate_class, 0),
                "mode": mode,
                "new_leaf_count": new_leaf_count,
                "new_leaf_map": compact_leaf_map(new_leaves),
                "selected_leaf_count": sum(len(items) for items in normalized.values()),
                "selected_leaf_map": compact_leaf_map(normalized),
                "top_k": top_k,
            }
        )

    for mode in modes:
        for top_k in top_ks:
            left = set(prefixes[ROW_KEY_ORDER[0]][mode][top_k])
            right = set(prefixes[ROW_KEY_ORDER[1]][mode][top_k])
            add("pair_public_prefix", mode, top_k, {ROW_KEY_ORDER[0]: left, ROW_KEY_ORDER[1]: right})
            add(
                "seed_salt167_public_prefix_salt175",
                mode,
                top_k,
                {ROW_KEY_ORDER[0]: set(seed[ROW_KEY_ORDER[0]]), ROW_KEY_ORDER[1]: right | seed[ROW_KEY_ORDER[1]]},
            )
            add(
                "public_prefix_salt167_seed_salt175",
                mode,
                top_k,
                {ROW_KEY_ORDER[0]: left | seed[ROW_KEY_ORDER[0]], ROW_KEY_ORDER[1]: set(seed[ROW_KEY_ORDER[1]])},
            )
    return candidates


def status_for(result: dict[str, Any]) -> str:
    if bool(result.get("public_key_verified")) and bool(result.get("derived")) and bool(result.get("below_rho")):
        return "ACCEPTED_RELATION_DERIVED_BELOW_RHO"
    if bool(result.get("public_key_verified")) and bool(result.get("derived")):
        return "ACCEPTED_RELATION_DERIVED_OVER_RHO"
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
    accepted = status in {
        "ACCEPTED_RELATION_DERIVED_BELOW_RHO",
        "ACCEPTED_RELATION_DERIVED_OVER_RHO",
    }
    return {
        **candidate,
        "accepted_relation_export": accepted,
        "below_rho": bool(result.get("below_rho")),
        "candidate_id": f"public_prefix_{digest_u64(candidate):016x}",
        "candidate_id_u64": digest_u64(candidate),
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


def replay_candidates(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], list[dict[str, Any]]]:
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
    )
    contexts, context_errors = replay_probe.materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        {"target": TARGET, "transfer_index": TRANSFER, "top_k": args.context_top_k},
        ROW_KEY_ORDER,
        replay_args,
        {},
    )
    if context_errors:
        return [], radius, context_errors, []
    prefixes = selected_prefixes(contexts, args.modes, args.top_ks)
    visible = visible_universe(load_json(Path(args.lattice)))
    if not candidates:
        candidates = candidate_leaf_sets(prefixes, args.modes, args.top_ks, visible)
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    records_out = []
    for index, candidate in enumerate(candidates):
        row_leaves = {
            str(item.get("row_key")): {as_int(leaf) for leaf in item.get("leaf_indices") or []}
            for item in candidate.get("selected_leaf_map") or []
        }
        result, row_events = replay_probe.replay_selection(
            verifier,
            row_leaves,
            contexts,
            scan_cache,
            args.event_summary_limit,
        )
        records_out.append(compact_candidate(index, candidate, result, row_events, args.event_summary_limit))
    profiles = prefix_profiles(contexts, args.modes, args.top_ks, prefixes, visible)
    return records_out, radius, [], profiles


def claim_status(failures: list[dict[str, Any]], records: list[dict[str, Any]]) -> str:
    if failures:
        return "SELECTED13_9943_PUBLIC_PREFIX_EXPANSION_FAILED"
    if any(record.get("accepted_relation_export") and record.get("below_rho") for record in records):
        return "SELECTED13_9943_PUBLIC_PREFIX_EXPANSION_BELOW_RHO_EXPORT"
    if any(record.get("accepted_relation_export") for record in records):
        return "SELECTED13_9943_PUBLIC_PREFIX_EXPANSION_EXPORT_OVER_RHO"
    return "SELECTED13_9943_PUBLIC_PREFIX_EXPANSION_NO_EXPORT"


def summarize(records: list[dict[str, Any]], failures: list[dict[str, Any]], salt_neighbor: dict[str, Any]) -> dict[str, Any]:
    accepted = [record for record in records if record.get("accepted_relation_export")]
    below = [record for record in accepted if record.get("below_rho")]
    status_counts = Counter(str(record.get("status")) for record in records)
    class_counts = Counter(str(record.get("candidate_class")) for record in records)
    best = min(
        accepted,
        key=lambda row: (
            0 if row.get("below_rho") else 1,
            as_float(row.get("ops_over_rho")) if as_float(row.get("ops_over_rho")) is not None else 999.0,
            -as_int(row.get("rank")),
        ),
        default={},
    )
    salt_summary = salt_neighbor.get("summary") if isinstance(salt_neighbor, dict) else {}
    return {
        "accepted_relation_export_count": len(accepted),
        "best_accepted_candidate_id": best.get("candidate_id"),
        "best_accepted_derived_secret": best.get("derived_secret"),
        "best_accepted_mode": best.get("mode"),
        "best_accepted_ops_over_rho": best.get("ops_over_rho"),
        "best_accepted_top_k": best.get("top_k"),
        "below_rho_accepted_relation_export_count": len(below),
        "candidate_class_counts": dict(sorted(class_counts.items())),
        "candidate_count": len(records),
        "context_error_count": sum(1 for failure in failures if failure.get("code") == "context_materialization_error"),
        "failure_count": len(failures),
        "max_rank": max((as_int(record.get("rank")) for record in records), default=0),
        "max_relation_count": max((as_int(record.get("relation_count")) for record in records), default=0),
        "modes_tested": sorted({str(record.get("mode")) for record in records}),
        "new_leaf_candidate_count": sum(1 for record in records if as_int(record.get("new_leaf_count")) > 0),
        "pollard_rho_speedup_claimed": bool(below),
        "rank2_or_better_unverified_count": sum(1 for record in records if str(record.get("status")) == "RANK2_UNVERIFIED"),
        "relation_derived_ecdlp": bool(accepted),
        "salt_neighbor_prior_accepted_relation_export_count": as_int(
            (salt_summary or {}).get("accepted_relation_export_count")
        ),
        "salt_neighbor_prior_claim_status": salt_neighbor.get("claim_status"),
        "single_target_speedup_scope": (
            "bounded 9943 public-prefix replay under the existing preassociation cost model"
            if below
            else None
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "target": TARGET,
        "top_ks_tested": sorted({as_int(record.get("top_k")) for record in records}),
        "transfer_index": TRANSFER,
        "verified": not failures,
        "worker_interpretation": (
            "Public feature prefixes outside the exhausted 9943 leaf universe produce "
            "relation-derived target exports. The best accepted candidate is below rho "
            "for this bounded target, but this is not yet a generalized ECDLP algorithm."
            if below
            else "Public feature prefixes did not produce a below-rho accepted target export."
        ),
    }


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
            f"{as_int(record.get('candidate_class_code'))}ULL, "
            f"{as_int(record.get('top_k'))}ULL, "
            f"{as_int(record.get('selected_leaf_count'))}ULL, "
            f"{as_int(record.get('new_leaf_count'))}ULL, "
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
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_9943_PUBLIC_PREFIX_EXPANSION_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_9943_PUBLIC_PREFIX_EXPANSION_PROBE_H

#include <stdint.h>

#define SELECTED13_9943_PUBLIC_PREFIX_CANDIDATE_COUNT {len(records)}
#define SELECTED13_9943_PUBLIC_PREFIX_ACCEPTED_COUNT {sum(1 for record in records if record.get('accepted_relation_export'))}
#define SELECTED13_9943_PUBLIC_PREFIX_BELOW_RHO_ACCEPTED_COUNT {sum(1 for record in records if record.get('accepted_relation_export') and record.get('below_rho'))}
#define SELECTED13_9943_PUBLIC_PREFIX_DERIVED_COUNT {sum(1 for record in records if record.get('relation_derived_ecdlp'))}

#define SELECTED13_9943_PUBLIC_PREFIX_STATUS_ACCEPTED_BELOW_RHO 1ULL
#define SELECTED13_9943_PUBLIC_PREFIX_STATUS_ACCEPTED_OVER_RHO 2ULL
#define SELECTED13_9943_PUBLIC_PREFIX_STATUS_VERIFIED_NO_DERIVATION 3ULL
#define SELECTED13_9943_PUBLIC_PREFIX_STATUS_RANK2_UNVERIFIED 4ULL
#define SELECTED13_9943_PUBLIC_PREFIX_STATUS_RANK1_ONLY 5ULL
#define SELECTED13_9943_PUBLIC_PREFIX_STATUS_HIT_ROOT_ONLY 6ULL
#define SELECTED13_9943_PUBLIC_PREFIX_STATUS_NO_RELATION 7ULL
#define SELECTED13_9943_PUBLIC_PREFIX_STATUS_CONTEXT_ERROR 8ULL

typedef struct {{
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t candidate_class_code;
  uint64_t top_k;
  uint64_t selected_leaf_count;
  uint64_t new_leaf_count;
  uint64_t below_rho;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t rank;
  uint64_t relation_count;
  uint64_t derived_secret;
  uint64_t ops_over_rho_scaled_1e6;
  uint64_t status_code;
}} selected13_9943_public_prefix_candidate_t;

static const selected13_9943_public_prefix_candidate_t SELECTED13_9943_PUBLIC_PREFIX_CANDIDATES[] = {{
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
      sizeof(SELECTED13_9943_PUBLIC_PREFIX_CANDIDATES) / sizeof(SELECTED13_9943_PUBLIC_PREFIX_CANDIDATES[0]);
  uint64_t accepted_count = 0;
  uint64_t below_rho_accepted_count = 0;
  uint64_t derived_count = 0;

  if (candidate_count != SELECTED13_9943_PUBLIC_PREFIX_CANDIDATE_COUNT) failure_count++;
  if (candidate_count == 0ULL) failure_count++;

  for (size_t i = 0; i < candidate_count; i++) {{
    const selected13_9943_public_prefix_candidate_t *candidate = &SELECTED13_9943_PUBLIC_PREFIX_CANDIDATES[i];
    if (candidate->candidate_id_u64 == 0ULL) failure_count++;
    if (candidate->new_leaf_count == 0ULL) failure_count++;
    if (candidate->status_code == 0ULL) failure_count++;
    if (candidate->relation_derived_ecdlp && !candidate->public_key_verified) failure_count++;
    if (candidate->relation_derived_ecdlp) derived_count++;
    if (candidate->status_code == SELECTED13_9943_PUBLIC_PREFIX_STATUS_ACCEPTED_BELOW_RHO ||
        candidate->status_code == SELECTED13_9943_PUBLIC_PREFIX_STATUS_ACCEPTED_OVER_RHO) {{
      accepted_count++;
      if (!candidate->relation_derived_ecdlp) failure_count++;
      if (candidate->derived_secret == 0ULL) failure_count++;
      if (candidate->status_code == SELECTED13_9943_PUBLIC_PREFIX_STATUS_ACCEPTED_BELOW_RHO) {{
        below_rho_accepted_count++;
        if (!candidate->below_rho) failure_count++;
      }}
    }}
  }}

  if (accepted_count != SELECTED13_9943_PUBLIC_PREFIX_ACCEPTED_COUNT) failure_count++;
  if (below_rho_accepted_count != SELECTED13_9943_PUBLIC_PREFIX_BELOW_RHO_ACCEPTED_COUNT) failure_count++;
  if (derived_count != SELECTED13_9943_PUBLIC_PREFIX_DERIVED_COUNT) failure_count++;

  printf("selected13_9943_public_prefix_preflight candidates=%llu accepted=%llu below_rho_accepted=%llu derived=%llu failures=%llu\\n",
         (unsigned long long)candidate_count,
         (unsigned long long)accepted_count,
         (unsigned long long)below_rho_accepted_count,
         (unsigned long long)derived_count,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_9943_public_prefix_preflight_") as tmp:
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
    lattice = load_json(Path(args.lattice))
    salt_neighbor = load_json(Path(args.salt_neighbor))
    failures: list[dict[str, Any]] = []
    if lattice.get("claim_status") != "SELECTED13_9943_SECOND_STAGE_LATTICE_NO_EXPORT":
        failures.append({"code": "lattice_status_unexpected", "claim_status": lattice.get("claim_status")})
    if salt_neighbor.get("claim_status") != "SELECTED13_9943_SALT_NEIGHBOR_CARRYOVER_NO_EXPORT":
        failures.append({"code": "salt_neighbor_status_unexpected", "claim_status": salt_neighbor.get("claim_status")})
    visible = visible_universe(lattice)
    records, radius, context_errors, profiles = replay_candidates(args, [])
    for error in context_errors:
        failures.append({"code": "context_materialization_error", "error": error})
    summary = summarize(records, failures, salt_neighbor)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, records),
        "parameters": {
            "bank_source": str(Path(args.bank_source)),
            "config_source": str(Path(args.config_source)),
            "context_top_k": args.context_top_k,
            "direct_source": str(Path(args.direct_source)),
            "lattice": str(Path(args.lattice)),
            "modes": args.modes,
            "radius": radius,
            "row_keys": ROW_KEY_ORDER,
            "salt_neighbor": str(Path(args.salt_neighbor)),
            "seed_leaf_map": compact_leaf_map({ROW_KEY_ORDER[0]: {8}, ROW_KEY_ORDER[1]: {8}}),
            "target": TARGET,
            "top_ks": args.top_ks,
            "transfer_index": TRANSFER,
            "transfer_source": str(Path(args.transfer_source)),
            "visible_leaf_universe": compact_leaf_map(visible),
        },
        "summary": summary,
        "prefix_profiles": profiles,
        "public_prefix_candidates": records,
        "failures": failures,
        "honesty_boundary": {
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_scope": summary.get("single_target_speedup_scope"),
            "target_level_pollard_rho_speedup_claimed": summary["pollard_rho_speedup_claimed"],
            "target_relation_derived_ecdlp": summary["relation_derived_ecdlp"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", type=Path, default=DEFAULT_LATTICE)
    parser.add_argument("--salt-neighbor", type=Path, default=DEFAULT_SALT_NEIGHBOR)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--modes", type=parse_csv, default=list(DEFAULT_MODES))
    parser.add_argument("--top-ks", type=parse_int_csv, default=list(DEFAULT_TOP_KS))
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
    header_path.write_text(render_c_header(payload["public_prefix_candidates"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = claim_status(payload["failures"], payload["public_prefix_candidates"])
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
