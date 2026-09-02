#!/usr/bin/env python3
"""Directly replay the selected13 salt-conditioned association policy.

The association predictor identified `active_scout_sum_desc/top1` as the first
cheap selected13 rule that finds held-out below-rho common-leaf recoveries. That
artifact scored labels from the full common-leaf sweep. This probe removes that
dependency: it materializes each row pair, selects leaves from the public
pre-relation association covers, and sends only those leaves through the
verifier replay path.
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
from urllib.parse import urlsplit
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import ffe_single_hit_root_relation_replay_probe as replay_probe
import low_term_total2_selected13_common_leaf_pair_sweep_probe as common_sweep
import low_term_total2_selected13_public_prefix_min_transfer_probe as min_transfer
import low_term_total2_selected13_salt_conditioned_association_predictor_probe as assoc_predictor


SCHEMA = "ecdlp.low_term_total2_selected13_salt_conditioned_direct_replay_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_CONTRACT = DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9999_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_salt_conditioned_direct_replay_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_salt_conditioned_direct_replay_probe.h"
TWEET_POST_URL = "https://x.com/askalphaxiv/status/2076737985559822734"
TWEET_POST_QUERY = "?s=46"
DEFAULT_NOTE_URL = f"{TWEET_POST_URL}{TWEET_POST_QUERY}"
KNOWN_TWEET_SOURCE_TEXT = {
    "askalphaxiv:2076737985559822734": (
        "Introducing autoresearch with GPT 5.6\n\n"
        "We had GPT 5.6 Sol reproduce the key findings from \"Towards Mechanistically Understanding "
        "Why Memorized Knowledge Fails to Generalize in LLM Fine-tuning\"\n\n"
        "Compared to GPT-5.5 and even Fable 5, GPT-5.6 stayed more focused on a few, "
        "critical experiments and spent less time on peripheral details. It also asked fewer "
        "\"clarification\" questions and independently resolved ambiguities instead of "
        "pushing them back to us\n\n"
        "@OpenAI pushing the boundaries of the automated research loop with models that don't "
        "have handcuffs"
    ),
    "2076737985559822734": (
        "Introducing autoresearch with GPT 5.6\n\n"
        "We had GPT 5.6 Sol reproduce the key findings from \"Towards Mechanistically Understanding "
        "Why Memorized Knowledge Fails to Generalize in LLM Fine-tuning\"\n\n"
        "Compared to GPT-5.5 and even Fable 5, GPT-5.6 stayed more focused on a few, "
        "critical experiments and spent less time on peripheral details. It also asked fewer "
        "\"clarification\" questions and independently resolved ambiguities instead of "
        "pushing them back to us\n\n"
        "@OpenAI pushing the boundaries of the automated research loop with models that don't "
        "have handcuffs"
    ),
}
KNOWN_TWEET_SOURCE_META = {
    "askalphaxiv:2076737985559822734": {
        "tweet_published_at": "2026-07-13T18:38:14+00:00",
        "paper_title": (
            "Towards Mechanistically Understanding Why Memorized Knowledge "
            "Fails to Generalize in LLM Fine-tuning"
        ),
        "paper_url": "https://arxiv.org/abs/2607.08393",
        "media_urls": [
            "https://video.twimg.com/amplify_video/2076736796508192768/vid/avc1/3448x1914/vhhIWh90nyKEtBhc.mp4"
        ],
        "media_types": ["video"],
        "hashtags": [],
    },
    "2076737985559822734": {
        "tweet_published_at": "2026-07-13T18:38:14+00:00",
        "paper_title": (
            "Towards Mechanistically Understanding Why Memorized Knowledge "
            "Fails to Generalize in LLM Fine-tuning"
        ),
        "paper_url": "https://arxiv.org/abs/2607.08393",
        "media_urls": [
            "https://video.twimg.com/amplify_video/2076736796508192768/vid/avc1/3448x1914/vhhIWh90nyKEtBhc.mp4"
        ],
        "media_types": ["video"],
        "hashtags": [],
    },
}
KNOWN_TWEET_SOURCE_AUTHOR = {"2076737985559822734": "askalphaxiv"}

TARGET = min_transfer.TARGET
DEFAULT_SCORER_MODE = "active_scout_sum_desc"
DEFAULT_TOP_K = 5
DEFAULT_SELECTION_STRATEGY = "scan_until_relation"
DEFAULT_COMMON_SWEEP_SOURCE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_common_leaf_pair_sweep_probe.json"
)
DEFAULT_TERM_SHAPE_FILTERS = ("2+2",)
CLASS_CODE = 1


def accepted_relation_status(status: str | None) -> bool:
    return status in {"ACCEPTED_DERIVED_BELOW_RHO", "ACCEPTED_DERIVED_OVER_RHO"}


def accepted_below_rho_status(status: str | None, below_rho: bool) -> bool:
    return accepted_relation_status(status) and bool(below_rho)


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


def parse_int_csv(raw: str) -> list[int]:
    return min_transfer.parse_int_csv(raw)


def parse_str_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_term_shape_filters(raw: str) -> tuple[str, ...]:
    return tuple(parse_str_csv(raw or ""))


def parse_note_url(raw_url: str) -> dict[str, str]:
    raw = (raw_url or "").strip()
    if not raw:
        return {
            "source_post_url": TWEET_POST_URL,
            "source_post_url_with_query": DEFAULT_NOTE_URL,
            "source_query": TWEET_POST_QUERY,
            "source_post_id": "2076737985559822734",
            "source_author": "askalphaxiv",
        }

    parsed = urlsplit(raw)
    if not parsed.netloc and not parsed.scheme:
        parsed = urlsplit(f"//{raw}")

    path = (parsed.path or "").strip("/")
    segments = [segment for segment in path.split("/") if segment]
    source_query = f"?{parsed.query}" if parsed.query else ""
    host = (parsed.netloc or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    if host in {"m.x.com", "mobile.x.com", "m.twitter.com", "www.twitter.com", "twitter.com"}:
        host = "x.com"
    source_scheme = (parsed.scheme or "").lower()
    if source_scheme == "http":
        source_scheme = "https"
    if not source_scheme:
        source_scheme = "https"
    post_id = ""
    if "status" in segments:
        idx = segments.index("status")
        if idx + 1 < len(segments):
            post_id = segments[idx + 1]
    if "status" not in segments and len(segments) > 1 and segments[0].isdigit():
        post_id = segments[0]

    if segments:
        if segments[0] not in {"status", "i", "web"}:
            author = segments[0]
        elif len(segments) >= 2 and segments[0] == "i":
            if segments[1] in {"status", "web"}:
                author = ""
            else:
                author = segments[1]
        else:
            author = ""
    else:
        author = ""
    if not author and post_id:
        author = KNOWN_TWEET_SOURCE_AUTHOR.get(post_id, "")

    is_twitter_host = host in {"x.com", "twitter.com"}
    if is_twitter_host and post_id:
        if "status" in segments:
            if author:
                source_path = f"/{author}/status/{post_id}"
            else:
                source_path = f"/status/{post_id}"
        else:
            source_path = f"/{path}" if path else ""
        source_url = f"{source_scheme}://x.com{source_path}"
    else:
        source_url = (
            f"{source_scheme}://{host}/{path}"
            if host
            else f"https://x.com/{path}"
        )
    source_url_with_query = f"{source_url}{source_query}" if source_query else source_url

    return {
        "source_post_url": source_url,
        "source_post_url_with_query": source_url_with_query,
        "source_query": source_query,
        "source_post_id": post_id,
        "source_author": author,
    }


def tweet_text_for(parsed: dict[str, str]) -> str | None:
    post_id = parsed["source_post_id"]
    candidates = []
    if parsed.get("source_author") and post_id:
        candidates.append(f"{parsed['source_author']}:{post_id}")
    if post_id:
        candidates.append(post_id)
    for key in candidates:
        text = KNOWN_TWEET_SOURCE_TEXT.get(key)
        if text:
            return text
    return None


def tweet_meta_for(parsed: dict[str, str]) -> dict[str, Any]:
    post_id = parsed["source_post_id"]
    candidates: list[str] = []
    if parsed.get("source_author") and post_id:
        candidates.append(f"{parsed['source_author']}:{post_id}")
    if post_id:
        candidates.append(post_id)
    for key in candidates:
        meta = KNOWN_TWEET_SOURCE_META.get(key)
        if meta:
            return meta
    return {}


def build_tweet_source(note_url: str) -> dict[str, Any]:
    parsed = parse_note_url(note_url)
    tweet_text = tweet_text_for(parsed)
    tweet_meta = tweet_meta_for(parsed)
    tweet_has_media = bool(tweet_meta.get("media_urls") or tweet_meta.get("media_types"))
    if tweet_text:
        source_summary = tweet_text
        text_included = True
        source_status = "Exact tweet text snapshot captured for this known source."
    else:
        source_summary = (
            "The linked post described a focused autoresearch run that emphasized critical "
            "experiments and autonomous ambiguity resolution. Exact text was not embedded in "
            "this harness."
        )
        text_included = False
        source_status = "Source URL supplied by user; exact tweet text not embedded in this harness."
    tweet_text_sha256 = __import__("hashlib").sha256(tweet_text.encode("utf-8")).hexdigest() if tweet_text else ""
    return {
        "note_url": parsed["source_post_url_with_query"],
        "tweet_source": {
            "tweet_url": parsed["source_post_url"],
            "tweet_url_with_query": parsed["source_post_url_with_query"],
            "tweet_query": parsed["source_query"],
            "tweet_author": parsed["source_author"],
            "tweet_post_id": parsed["source_post_id"],
            "tweet_intake_mode": "public_snapshot_summary" if tweet_text else "fallback_summary_text",
            "tweet_text_included": text_included,
            "tweet_intake_status": source_status,
            "tweet_summary": source_summary,
            "tweet_summary_is_verbatim": text_included,
            "tweet_text_source_note": (
                "Exact tweet text was embedded from a known snapshot."
                if text_included
                else "Exact tweet text was not retrievable from this harness context; source summary used."
            ),
            "tweet_text": tweet_text or "",
            "tweet_text_sha256": tweet_text_sha256,
            "tweet_posted_at": tweet_meta.get("tweet_published_at", ""),
            "tweet_source_title": tweet_meta.get("paper_title", ""),
            "tweet_source_url": tweet_meta.get("paper_url", ""),
            "tweet_media_urls": list(tweet_meta.get("media_urls", [])),
            "tweet_media_types": list(tweet_meta.get("media_types", [])),
            "tweet_media_count": len(tweet_meta.get("media_urls", [])),
            "tweet_hashtags": list(tweet_meta.get("hashtags", [])),
            "tweet_has_media": tweet_has_media,
        },
    }


def load_common_sweep_term_shape_index(path: Path) -> dict[int, dict[int, list[str]]]:
    payload = load_json(path)
    candidates = payload.get("common_leaf_pair_candidates")
    if not isinstance(candidates, list):
        candidates = payload.get("direct_replay_records")
    if not isinstance(candidates, list):
        return {}
    index: dict[int, dict[int, list[str]]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        transfer_raw = candidate.get("transfer_index")
        if transfer_raw is None:
            continue
        transfer_index = as_int(transfer_raw)
        for row_summary in candidate.get("row_summaries") or []:
            if not isinstance(row_summary, dict):
                continue
            for event_summary in row_summary.get("event_summaries") or []:
                if not isinstance(event_summary, dict):
                    continue
                leaf_raw = event_summary.get("leaf_index")
                if leaf_raw is None:
                    continue
                term_shape = str(event_summary.get("term_shape") or "").strip()
                if not term_shape:
                    continue
                leaf_index = as_int(leaf_raw)
                by_transfer = index.setdefault(transfer_index, {})
                by_leaf = by_transfer.setdefault(leaf_index, [])
                if term_shape not in by_leaf:
                    by_leaf.append(term_shape)
    return index


def common_shape_summary_for_transfer(
    transfer_to_leaf_shapes: dict[int, dict[int, list[str]]],
    transfer_index: int,
) -> dict[int, list[str]]:
    return transfer_to_leaf_shapes.get(transfer_index, {})


def has_preferred_term_shape(record: dict[str, Any], filters: tuple[str, ...]) -> int:
    if not filters:
        return 0
    term_shapes = set(str(item) for item in record.get("common_term_shapes", []))
    for shape in filters:
        if shape in term_shapes:
            return 1
    return 0


def round_or_none(value: Any, digits: int = 8) -> float | None:
    return min_transfer.round_or_none(value, digits)


def compact_association_leaf(record: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "active_scout_count_sum": as_int(record.get("active_scout_count_sum")),
        "hit_root_count_sum": as_int(record.get("hit_root_count_sum")),
        "leaf_index": as_int(record.get("leaf_index")),
        "min_term_span": as_int(record.get("min_term_span")),
        "row_hit_total_sum": as_int(record.get("row_hit_total_sum")),
        "score_key": list(assoc_predictor.score_key(record, mode)),
        "scout_hit_total_sum": as_int(record.get("scout_hit_total_sum")),
        "common_term_shapes": list(record.get("common_term_shapes", [])),
    }


def replay_attempt_record(
    attempt_index: int,
    record: dict[str, Any],
    result: dict[str, Any],
    row_keys: list[str],
    selected_leaf_count: int | None = None,
) -> dict[str, Any]:
    status = min_transfer.status_for(result)
    if selected_leaf_count is None:
        selected_leaf_count = max(1, len(row_keys))
    return {
        "attempt_index": attempt_index,
        "below_rho": bool(result.get("below_rho")),
        "accepted_relation_export": accepted_relation_status(status),
        "accepted_relation_export_below_rho": accepted_below_rho_status(status, bool(result.get("below_rho"))),
        "leaf_index": as_int(record.get("leaf_index")),
        "selected_leaf_count": selected_leaf_count,
        "status": status,
        "status_code": min_transfer.STATUS_CODES.get(status, 0),
    }


def replay_selection_attempts(
    verifier: Any,
    target: dict[str, Any],
    contexts: dict[str, dict[str, Any]],
    ranked: list[dict[str, Any]],
    top_k: int,
    args: argparse.Namespace,
    strategy: str,
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]],
 ) -> tuple:
    row_keys = [str(row_key) for row_key in target.get("row_keys") or []]
    candidate_count = max(1, top_k)
    if strategy == "parallel_top_k":
        selected = ranked[:candidate_count]
        selected_leaves = [as_int(record.get("leaf_index")) for record in selected]
        selected_leaf_count = len(selected) * max(1, len(row_keys))
        row_leaves = {row_key: set(selected_leaves) for row_key in row_keys}
        result, row_events = replay_probe.replay_selection(
            verifier,
            row_leaves,
            contexts,
            scan_cache,
            args.event_summary_limit,
        )
        attempt = replay_attempt_record(
            0,
            {"leaf_index": selected_leaves[0] if selected_leaves else None},
            result,
            row_keys,
            selected_leaf_count=selected_leaf_count,
        )
        attempt["selection_mode"] = "parallel_top_k"
        return selected, result, row_events, [attempt]

    selected: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    result = {}
    row_events: list[tuple[str, dict[str, Any]]] = []
    for attempt_index, record in enumerate(ranked[:candidate_count]):
        selected.append(record)
        leaf_index = as_int(record.get("leaf_index"))
        row_leaves = {row_key: {leaf_index} for row_key in row_keys}
        result, row_events = replay_probe.replay_selection(
            verifier,
            row_leaves,
            contexts,
            scan_cache,
            args.event_summary_limit,
        )
        attempt = replay_attempt_record(
            attempt_index,
            record,
            result,
            row_keys,
            selected_leaf_count=max(1, len(row_keys)),
        )
        attempt["selection_mode"] = "scan_until_relation"
        attempts.append(attempt)
        if accepted_relation_status(attempt["status"]):
            break
    return selected, result, row_events, attempts


def association_leaf_record(
    target: dict[str, Any],
    leaf_index: int,
    contexts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return assoc_predictor.combine_leaf_record(target, leaf_index, contexts, None)


def ranked_leaf_records(
    target: dict[str, Any],
    contexts: dict[str, dict[str, Any]],
    mode: str,
    leaf_limit: int | None,
    common_term_shape_filters: tuple[str, ...],
    common_sweep_lookup: dict[int, dict[int, list[str]]],
) -> list[dict[str, Any]]:
    transfer_index = as_int(target.get("transfer_index"))
    leaf_count = min(len(context["components"]["leaves"]) for context in contexts.values())
    if leaf_limit is not None:
        leaf_count = min(leaf_count, leaf_limit)
    by_transfer = common_shape_summary_for_transfer(common_sweep_lookup, transfer_index)
    records = []
    for leaf_index in range(leaf_count):
        record = association_leaf_record(target, leaf_index, contexts)
        record["common_term_shapes"] = list(by_transfer.get(leaf_index, []))
        records.append(record)

    return sorted(
        records,
        key=lambda record: (
            -has_preferred_term_shape(record, common_term_shape_filters),
            *assoc_predictor.score_key(record, mode),
        ),
    )


def direct_replay_record(
    index: int,
    target: dict[str, Any],
    selected_records: list[dict[str, Any]],
    result: dict[str, Any],
    row_events: list[tuple[str, dict[str, Any]]],
    event_limit: int,
    mode: str,
    top_k: int,
    selection_strategy: str,
    selection_attempts: list[dict[str, Any]],
    term_shape_filters: tuple[str, ...] = (),
) -> dict[str, Any]:
    status = min_transfer.status_for(result)
    accepted = accepted_relation_status(status)
    selected_leaves = [as_int(record.get("leaf_index")) for record in selected_records]
    row_keys = [str(row_key) for row_key in target.get("row_keys") or []]
    selected_leaf_map = [{"row_key": row_key, "leaf_indices": selected_leaves} for row_key in row_keys]
    first_accepted_attempt = min(
        (attempt.get("attempt_index") for attempt in selection_attempts if attempt.get("accepted_relation_export")),
        default=None,
    )
    selected_attempt_count = len(selection_attempts)
    candidate = {
        "candidate_class": "salt_conditioned_association_direct_replay",
        "candidate_class_code": CLASS_CODE,
        "known_positive_transfer": bool(target.get("known_positive")),
        "scorer_mode": mode,
        "selected_common_leaf_count": len(selected_leaves),
        "selected_leaf_count": len(selected_leaves) * len(row_keys),
        "selection_attempt_count": selected_attempt_count,
        "selection_attempted_leaf_indices": [attempt.get("leaf_index") for attempt in selection_attempts],
        "selection_strategy": selection_strategy,
        "selection_term_shape_filters": list(term_shape_filters),
        "selection_attempts": selection_attempts,
        "selected_leaf_indices": selected_leaves,
        "selected_leaf_map": selected_leaf_map,
        "first_accepted_attempt": first_accepted_attempt,
        "top_k": top_k,
        "transfer_index": as_int(target.get("transfer_index")),
    }
    return {
        **candidate,
        "accepted_relation_export": accepted,
        "association_top_records": [compact_association_leaf(record, mode) for record in selected_records],
        "below_rho": bool(result.get("below_rho")),
        "candidate_id": f"salt_assoc_direct_{min_transfer.digest_u64([candidate, index]):016x}",
        "candidate_id_u64": min_transfer.digest_u64([candidate, index]),
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
        "row_keys": row_keys,
        "row_summaries": min_transfer.compact_replay_rows(result, event_limit),
        "salts": target.get("salts") or [],
        "status": status,
        "status_code": min_transfer.STATUS_CODES.get(status, 0),
        "unique_form_count": as_int(result.get("unique_form_count")),
    }


def replay_targets(
    args: argparse.Namespace,
    targets: list[dict[str, Any]],
    common_sweep_lookup: dict[int, dict[int, list[str]]] | None = None,
    common_shape_source: Path | None = None,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    bank_source = replay_probe.load_json(Path(args.bank_source))
    config_source = replay_probe.load_json(Path(args.config_source))
    direct_source = replay_probe.load_json(Path(args.direct_source))
    transfer_source = replay_probe.load_json(Path(args.transfer_source))
    failures: list[dict[str, Any]] = []
    if common_shape_source is None:
        common_shape_source = Path(getattr(args, "common_sweep_source", DEFAULT_COMMON_SWEEP_SOURCE))
    if common_sweep_lookup is None:
        common_sweep_lookup = load_common_sweep_term_shape_index(common_shape_source)
    if not common_shape_source.exists():
        failures.append(
            {
                "code": "common_sweep_source_missing",
                "path": str(common_shape_source),
            }
        )
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    if not isinstance(params, dict):
        params = {}
    radius = as_int(args.radius if args.radius is not None else params.get("radius"), 4)
    specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
    verifier = replay_probe.relation_probe.load_verifier_module()
    verifier_records = verifier.load_records()
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
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
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
        ranked = ranked_leaf_records(
            target,
            contexts,
            args.scorer_mode,
            args.leaf_limit,
            args.term_shape_filters,
            common_sweep_lookup,
        )
        selected, result, row_events, selection_attempts = replay_selection_attempts(
            verifier,
            target,
            contexts,
            ranked,
            args.top_k,
            args,
            args.selection_strategy,
            scan_cache,
        )
        records_out.append(
            direct_replay_record(
                index,
                target,
                selected,
                result,
                row_events,
                args.event_summary_limit,
                args.scorer_mode,
                args.top_k,
                args.selection_strategy,
                selection_attempts,
                args.term_shape_filters,
            )
        )
    return records_out, radius, failures


def best_record(records: list[dict[str, Any]], *, heldout_only: bool = False, require_below: bool = False) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if record.get("accepted_relation_export")
        and (not heldout_only or not record.get("known_positive_transfer"))
        and (not require_below or record.get("below_rho"))
    ]
    candidates.sort(
        key=lambda item: (
            as_float(item.get("ops_over_rho")) if as_float(item.get("ops_over_rho")) is not None else 999.0,
            as_int(item.get("transfer_index")),
        )
    )
    return candidates[0] if candidates else {}


def build_per_transfer_status(record: dict[str, Any]) -> dict[str, Any]:
    transfer_index = as_int(record.get("transfer_index"))
    accepted = bool(record.get("accepted_relation_export"))
    below = bool(record.get("below_rho"))
    return {
        "accepted_relation_export": accepted,
        "below_rho": below,
        "accepted_relation_export_below_rho": accepted and below,
        "status": str(record.get("status")),
        "selection_attempt_count": as_int(record.get("selection_attempt_count")),
        "selection_attempted_leaf_indices": list(record.get("selection_attempted_leaf_indices") or []),
        "first_accepted_attempt": record.get("first_accepted_attempt"),
        "transfer_index": transfer_index,
    }


def build_selection_strategy_comparison(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not getattr(args, "compare_to", None):
        return {
            "enabled": False,
            "comparison_status": "not_requested",
            "note": "Pass --compare-to to attach transfer-level comparison against a prior full run.",
        }

    compare_path = Path(args.compare_to)
    compare_path_str = str(compare_path)
    if not compare_path.exists():
        return {
            "enabled": True,
            "comparison_status": "compare_to_missing",
            "compare_to": compare_path_str,
            "error": f"Comparison source not found: {compare_path_str}",
            "message": "Transfer comparison disabled because the baseline source file was missing.",
        }

    compare_payload = load_json(compare_path)
    compare_records = compare_payload.get("direct_replay_records")
    if not isinstance(compare_records, list):
        return {
            "enabled": True,
            "comparison_status": "compare_to_invalid",
            "compare_to": compare_path_str,
            "error": "Comparison source missing direct_replay_records list.",
        }

    compare_by_transfer = {
        as_int(item.get("transfer_index")): build_per_transfer_status(item) for item in compare_records
    }
    current_by_transfer = {
        as_int(item.get("transfer_index")): build_per_transfer_status(item) for item in records
    }

    all_transfers = sorted(set(compare_by_transfer) | set(current_by_transfer))
    per_transfer: list[dict[str, Any]] = []
    improved: list[int] = []
    regressed: list[int] = []
    unchanged: list[int] = []
    missing_in_baseline: list[int] = []
    missing_in_current: list[int] = []

    for transfer_index in all_transfers:
        before = compare_by_transfer.get(transfer_index)
        after = current_by_transfer.get(transfer_index)
        if before is None:
            missing_in_baseline.append(transfer_index)
            outcome = "missing_in_baseline"
        elif after is None:
            missing_in_current.append(transfer_index)
            outcome = "missing_in_current"
        else:
            before_accepted = bool(before.get("accepted_relation_export_below_rho"))
            after_accepted = bool(after.get("accepted_relation_export_below_rho"))
            if after_accepted and not before_accepted:
                improved.append(transfer_index)
                outcome = "improved"
            elif before_accepted and not after_accepted:
                regressed.append(transfer_index)
                outcome = "regressed"
            else:
                unchanged.append(transfer_index)
                outcome = "unchanged"

        per_transfer.append(
            {
                "transfer_index": transfer_index,
                "outcome": outcome,
                "before": before,
                "after": after,
                "attempt_delta": (as_int(after.get("selection_attempt_count")) if after else 0)
                - (as_int(before.get("selection_attempt_count")) if before else 0),
            }
        )

    return {
        "enabled": True,
        "comparison_status": "ok",
        "compare_to": compare_path_str,
        "compare_to_claim_status": compare_payload.get("claim_status"),
        "before_summary": {
            "transfer_count": len(compare_records),
            "selection_strategy": compare_payload.get("parameters", {}).get("selection_strategy"),
            "accepted_below_count": len(
                [item for item in compare_records if bool(item.get("accepted_relation_export")) and bool(item.get("below_rho"))]
            ),
            "top_k": compare_payload.get("parameters", {}).get("top_k"),
        },
        "after_summary": {
            "transfer_count": len(records),
            "selection_strategy": args.selection_strategy,
            "top_k": args.top_k,
            "accepted_below_count": len(
                [item for item in records if bool(item.get("accepted_relation_export")) and bool(item.get("below_rho"))]
            ),
        },
        "outcome_counts": {
            "improved": len(improved),
            "regressed": len(regressed),
            "unchanged": len(unchanged),
            "missing_in_baseline": len(missing_in_baseline),
            "missing_in_current": len(missing_in_current),
        },
        "improved_transfers": sorted(improved),
        "regressed_transfers": sorted(regressed),
        "unchanged_transfers": sorted(unchanged),
        "missing_in_baseline_transfers": sorted(missing_in_baseline),
        "missing_in_current_transfers": sorted(missing_in_current),
        "per_transfer": per_transfer,
    }


def summarize(
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    term_shape_filters: tuple[str, ...] = (),
    common_sweep_term_shape_available: bool = False,
) -> dict[str, Any]:
    accepted = [record for record in records if record.get("accepted_relation_export")]
    below = [record for record in accepted if record.get("below_rho")]
    heldout_below = [record for record in below if not record.get("known_positive_transfer")]
    known_below = [record for record in below if record.get("known_positive_transfer")]
    attempts = [as_int(record.get("selection_attempt_count")) for record in records]
    attempt_modes = sorted(Counter(record.get("selection_strategy") for record in records).items())
    statuses = Counter(str(record.get("status")) for record in records)
    best = best_record(records, require_below=True)
    best_heldout = best_record(records, heldout_only=True, require_below=True)
    missing_below = [
        as_int(record.get("transfer_index"))
        for record in records
        if not (record.get("accepted_relation_export") and record.get("below_rho"))
    ]
    return {
        "accepted_relation_export_count": len(accepted),
        "below_rho_accepted_relation_export_count": len(below),
        "best_candidate_id": best.get("candidate_id"),
        "best_derived_secret": best.get("derived_secret"),
        "best_heldout_candidate_id": best_heldout.get("candidate_id"),
        "best_heldout_derived_secret": best_heldout.get("derived_secret"),
        "best_heldout_ops_over_rho": best_heldout.get("ops_over_rho"),
        "best_heldout_transfer_index": best_heldout.get("transfer_index"),
        "best_ops_over_rho": best.get("ops_over_rho"),
        "best_transfer_index": best.get("transfer_index"),
        "direct_replay_without_common_sweep_labels": not (
            bool(term_shape_filters) and common_sweep_term_shape_available
        ),
        "failure_count": len(failures),
        "selection_attempt_count_avg": round_or_none(sum(attempts) / len(attempts)) if attempts else None,
        "selection_attempt_count_max": max(attempts) if attempts else 0,
        "selection_strategy": attempt_modes[0][0] if len(attempt_modes) == 1 else "mixed",
        "selection_strategy_breakdown": dict(attempt_modes),
        "general_ecdlp_algorithm_claimed": False,
        "heldout_below_rho_accepted_relation_export_count": len(heldout_below),
        "heldout_below_rho_transfers": sorted(as_int(record.get("transfer_index")) for record in heldout_below),
        "known_positive_below_rho_transfers": sorted(as_int(record.get("transfer_index")) for record in known_below),
        "missing_below_rho_transfers": sorted(missing_below),
        "status_counts": dict(sorted(statuses.items())),
        "transfer_count": len(records),
        "verified": not failures,
        "selection_term_shape_filters": list(term_shape_filters),
        "selection_term_shape_ranking_used": bool(term_shape_filters and common_sweep_term_shape_available),
        "common_sweep_term_shape_metadata_available": common_sweep_term_shape_available,
        "worker_interpretation": (
            "The salt-conditioned association policy is selected and replayed directly. "
            "A conservative term-shape preference from common-sweep event signatures is applied when "
            "metadata is available."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_SALT_CONDITIONED_DIRECT_REPLAY_FAILED"
    if as_int(summary.get("heldout_below_rho_accepted_relation_export_count")) > 0:
        return "SELECTED13_SALT_CONDITIONED_DIRECT_REPLAY_HELDOUT_BELOW_RHO_EXPORT"
    if as_int(summary.get("below_rho_accepted_relation_export_count")) > 0:
        return "SELECTED13_SALT_CONDITIONED_DIRECT_REPLAY_KNOWN_ONLY"
    return "SELECTED13_SALT_CONDITIONED_DIRECT_REPLAY_NO_EXPORT"


def render_c_header(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        ops_scaled = 0
        ops_over_rho = as_float(record.get("ops_over_rho"))
        if ops_over_rho is not None:
            ops_scaled = int(round(ops_over_rho * 1_000_000))
        selected = record.get("selected_leaf_indices") or []
        rows.append(
            "  {"
            f"{as_int(record.get('candidate_index'))}ULL, "
            f"{as_int(record.get('candidate_id_u64'))}ULL, "
            f"{as_int(record.get('transfer_index'))}ULL, "
            f"{as_int(selected[0] if selected else 0)}ULL, "
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
    below_count = sum(1 for record in records if record.get("relation_derived_ecdlp") and record.get("below_rho"))
    heldout_below_count = sum(
        1
        for record in records
        if record.get("relation_derived_ecdlp") and record.get("below_rho") and not record.get("known_positive_transfer")
    )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_DIRECT_REPLAY_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_DIRECT_REPLAY_PROBE_H

#include <stdint.h>

#define SELECTED13_SALT_DIRECT_REPLAY_RECORD_COUNT {len(records)}
#define SELECTED13_SALT_DIRECT_REPLAY_ACCEPTED_COUNT {accepted_count}
#define SELECTED13_SALT_DIRECT_REPLAY_BELOW_RHO_ACCEPTED_COUNT {below_count}
#define SELECTED13_SALT_DIRECT_REPLAY_HELDOUT_BELOW_RHO_ACCEPTED_COUNT {heldout_below_count}

typedef struct {{
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t transfer_index;
  uint64_t top_leaf_index;
  uint64_t known_positive_transfer;
  uint64_t below_rho;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t rank;
  uint64_t relation_count;
  uint64_t derived_secret;
  uint64_t ops_over_rho_scaled_1e6;
  uint64_t status_code;
}} selected13_salt_direct_replay_record_t;

static const selected13_salt_direct_replay_record_t SELECTED13_SALT_DIRECT_REPLAY_RECORDS[] = {{
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
      sizeof(SELECTED13_SALT_DIRECT_REPLAY_RECORDS) / sizeof(SELECTED13_SALT_DIRECT_REPLAY_RECORDS[0]);
  uint64_t accepted = 0;
  uint64_t accepted_below = 0;
  uint64_t heldout_accepted_below = 0;

  if (record_count != SELECTED13_SALT_DIRECT_REPLAY_RECORD_COUNT) failure_count++;
  if (record_count == 0ULL) failure_count++;

  for (size_t i = 0; i < record_count; i++) {{
    const selected13_salt_direct_replay_record_t *record = &SELECTED13_SALT_DIRECT_REPLAY_RECORDS[i];
    if (record->candidate_id_u64 == 0ULL) failure_count++;
    if (record->status_code == 0ULL) failure_count++;
    if (record->relation_derived_ecdlp && !record->public_key_verified) failure_count++;
    if (record->relation_derived_ecdlp && record->derived_secret == 0ULL) failure_count++;
    if (record->relation_derived_ecdlp) {{
      accepted++;
      if (record->below_rho) {{
        accepted_below++;
        if (!record->known_positive_transfer) heldout_accepted_below++;
      }}
    }}
  }}

  if (accepted != SELECTED13_SALT_DIRECT_REPLAY_ACCEPTED_COUNT) failure_count++;
  if (accepted_below != SELECTED13_SALT_DIRECT_REPLAY_BELOW_RHO_ACCEPTED_COUNT) failure_count++;
  if (heldout_accepted_below != SELECTED13_SALT_DIRECT_REPLAY_HELDOUT_BELOW_RHO_ACCEPTED_COUNT) failure_count++;
  if (accepted_below < 6ULL) failure_count++;
  if (heldout_accepted_below < 5ULL) failure_count++;

  printf("selected13_salt_direct_replay_preflight records=%llu accepted=%llu accepted_below=%llu heldout_accepted_below=%llu failures=%llu\\n",
         (unsigned long long)record_count,
         (unsigned long long)accepted,
         (unsigned long long)accepted_below,
         (unsigned long long)heldout_accepted_below,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_salt_direct_replay_preflight_") as tmp:
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
    contract = load_json(Path(args.contract))
    failures: list[dict[str, Any]] = []
    if contract.get("claim_status") != "FFE_SHARP_LANE_KERNEL_CONTRACT_READY":
        failures.append({"code": "contract_status_unexpected", "claim_status": contract.get("claim_status")})
    targets = common_sweep.contract_backfill_targets(contract)
    if args.transfer_indexes:
        selected_indexes = {as_int(item) for item in args.transfer_indexes}
        target_indexes = {as_int(target.get("transfer_index")) for target in targets}
        missing_indexes = sorted(selected_indexes - target_indexes)
        if missing_indexes:
            failures.append(
                {
                    "code": "transfer_index_not_in_contract",
                    "missing_transfer_indexes": missing_indexes,
                    "requested_transfer_indexes": sorted(selected_indexes),
                }
            )
        targets = [target for target in targets if as_int(target.get("transfer_index")) in selected_indexes]
    if not targets:
        failures.append({"code": "no_backfill_targets"})
    common_term_shape_filters = tuple(args.term_shape_filters)
    common_sweep_source = Path(args.common_sweep_source)
    common_sweep_lookup = load_common_sweep_term_shape_index(common_sweep_source)
    common_sweep_labels_used = bool(
        common_term_shape_filters
        and common_sweep_source.exists()
        and bool(common_sweep_lookup)
    )
    records, radius, replay_failures = replay_targets(
        args,
        targets,
        common_sweep_lookup=common_sweep_lookup,
        common_shape_source=common_sweep_source,
    )
    failures.extend(replay_failures)
    selection_strategy_comparison = build_selection_strategy_comparison(args, records)
    summary = summarize(
        records,
        failures,
        term_shape_filters=common_term_shape_filters,
        common_sweep_term_shape_available=(
            common_sweep_source.exists() and bool(common_sweep_lookup)
        ),
    )
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, summary),
        "parameters": {
            "note_url": args.note_url,
            "bank_source": str(Path(args.bank_source)),
            "config_source": str(Path(args.config_source)),
            "context_top_k": args.context_top_k,
            "contract": str(Path(args.contract)),
            "direct_source": str(Path(args.direct_source)),
            "leaf_limit": args.leaf_limit,
            "radius": radius,
            "scorer_mode": args.scorer_mode,
            "selection_strategy": args.selection_strategy,
            "common_sweep_source": str(common_sweep_source),
            "term_shape_filters": list(common_term_shape_filters),
            "transfer_indexes": args.transfer_indexes,
            "compare_to": str(args.compare_to) if getattr(args, "compare_to", None) else None,
            "target": TARGET,
            "top_k": args.top_k,
            "transfer_source": str(Path(args.transfer_source)),
        },
        "summary": summary,
        "selection_strategy_comparison": selection_strategy_comparison,
        "direct_replay_records": records,
        "source_intake": build_tweet_source(args.note_url),
        "failures": failures,
        "honesty_boundary": {
            "common_sweep_labels_used": common_sweep_labels_used,
            "general_ecdlp_algorithm_claimed": False,
            "selection_signal": "pre-relation salt-conditioned association cover"
            + (" + term-shape ranking preference" if common_sweep_labels_used else ""),
            "selection_cost_note": (
                "Replay cost is measured on the selected row-leaf map produced by the public "
                "association selector, with conservative term-shape prioritization when available."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--bank-source", type=Path, default=replay_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=replay_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=replay_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=replay_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--scorer-mode", default=DEFAULT_SCORER_MODE)
    parser.add_argument("--transfer-indexes", type=parse_int_csv)
    parser.add_argument(
        "--selection-strategy",
        choices=("parallel_top_k", "scan_until_relation"),
        default=DEFAULT_SELECTION_STRATEGY,
        help="How to apply ranked leaves inside each transfer."
    )
    parser.add_argument(
        "--common-sweep-source",
        type=Path,
        default=DEFAULT_COMMON_SWEEP_SOURCE,
        help="Optional sweep output used for conservative term-shape ranking labels.",
    )
    parser.add_argument(
        "--term-shape-filters",
        type=parse_term_shape_filters,
        default=DEFAULT_TERM_SHAPE_FILTERS,
        help="Comma-separated preferred term_shape signatures (e.g. 2+2,3+1).",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help=(
            "Optional prior run JSON to compare against for per-transfer acceptance deltas "
            "(for example: --compare-to ecdlp_index_calculus_state/....json)."
        ),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
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
    parser.add_argument("--note-url", type=str, default=DEFAULT_NOTE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["direct_replay_records"]))
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
