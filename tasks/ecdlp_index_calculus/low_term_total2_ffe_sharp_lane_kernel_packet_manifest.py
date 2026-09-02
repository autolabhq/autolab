#!/usr/bin/env python3
"""Emit typed ABI packets for the sharp-lane FFE kernel contract.

The kernel contract says which public salt pairs and row checks a lower-level
FFE/summation-polynomial worker must reproduce.  This script lowers that JSON
contract into compact packets: verifier target context, first-pass public
groups, second-pass row slots, support/family bitmasks, exact certificate
hashes, and an optional C header.

This remains a pre-execution gate.  It does not evaluate summation polynomials,
export new direct/rank rows, or claim an ECDLP speedup.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_kernel_packet_manifest.v1"
DEFAULT_CONTRACT = Path(
    "ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9959_probe.json"
)
DEFAULT_OUT = Path(
    "ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9959_probe.json"
)
DEFAULT_HEADER_OUT = Path(
    "ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9959_probe.h"
)
DEFAULT_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
)
DEFAULT_FRONTIER_TARGETS = Path(
    os.environ.get("ECDLP_FRONTIER_TARGETS", "/Volumes/Volume/autolab/ecdlp_index_calculus_state/frontier_targets.json")
)
ROW_KEY_RE = re.compile(
    r"^(?P<target>[^:]+):(?P<stream>[^:]+):(?P<width>\d+):salt(?P<salt>\d+)$"
)
CLASS_CODES = {
    "direct_rank_backfill_row": 1,
    "inherited_promotion_row": 2,
    "exact_positive_row": 3,
    "neutral_exported_row": 4,
}
REPLAY_CLASS_CODES = {
    "direct_rank_backfill": 1,
    "inherited_positive_replay": 2,
    "exact_positive_replay": 3,
    "neutral_exported_replay": 4,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def sorted_ints(raw: Any) -> list[int]:
    return sorted(int_value(item) for item in (raw or []))


def stable_hash(material: Any, prefix: str) -> str:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    import hashlib

    return f"{prefix}_{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]}"


def support_mask(raw: Any) -> int:
    mask = 0
    for item in sorted_ints(raw):
        if item < 0 or item >= 64:
            raise ValueError(f"support index outside u64 mask range: {item}")
        mask |= 1 << item
    return mask


def support_masks(raw: Any) -> list[int]:
    return [support_mask(item) for item in (raw or [])]


def parse_target(raw: str) -> tuple[str, int]:
    label, sep, prime = str(raw).partition("@")
    if not sep or not label:
        raise ValueError(f"target must be label@prime, got {raw!r}")
    return label, int(prime)


def parse_row_key(raw: str) -> dict[str, Any]:
    match = ROW_KEY_RE.fullmatch(str(raw))
    if match is None:
        return {"row_key": raw, "parse_status": "failed"}
    return {
        "parse_status": "ok",
        "row_key": raw,
        "salt": int(match.group("salt")),
        "stream": match.group("stream"),
        "target": match.group("target"),
        "width": int(match.group("width")),
    }


def load_verifier_module(task_dir: Path) -> Any:
    env_main = task_dir / "environment" / "main.py"
    data_path = task_dir / "environment" / "lmfdb_curves.json"
    if not env_main.is_file():
        raise FileNotFoundError(f"verifier environment missing: {env_main}")
    if not data_path.is_file():
        raise FileNotFoundError(f"LMFDB data missing: {data_path}")
    old_app_dir = os.environ.get("APP_DIR")
    old_lmfdb_data = os.environ.get("LMFDB_DATA")
    os.environ["APP_DIR"] = str(env_main.parent)
    os.environ["LMFDB_DATA"] = str(data_path)
    try:
        spec = importlib.util.spec_from_file_location("ecdlp_verifier_main_for_packet_manifest", env_main)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import verifier helpers from {env_main}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old_app_dir is None:
            os.environ.pop("APP_DIR", None)
        else:
            os.environ["APP_DIR"] = old_app_dir
        if old_lmfdb_data is None:
            os.environ.pop("LMFDB_DATA", None)
        else:
            os.environ["LMFDB_DATA"] = old_lmfdb_data


def frontier_candidates(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = load_json(path)
    return {
        str(item.get("target")): item
        for item in data.get("candidates") or []
        if isinstance(item, dict) and item.get("target") is not None
    }


def target_contexts(contract: dict[str, Any], task_dir: Path, frontier_targets: Path) -> dict[str, dict[str, Any]]:
    targets = sorted(
        {
            str(item.get("public_first_pass", {}).get("target"))
            for item in (contract.get("contract") or {}).get("first_pass_contract") or []
            if item.get("public_first_pass", {}).get("target") is not None
        }
    )
    verifier = load_verifier_module(task_dir)
    records = verifier.load_records()
    by_label = {str(record["label"]): record for record in records}
    frontier_by_target = frontier_candidates(frontier_targets)
    contexts: dict[str, dict[str, Any]] = {}
    for index, target in enumerate(targets):
        label, p = parse_target(target)
        record = by_label.get(label)
        if record is None:
            contexts[target] = {
                "context_status": "missing_verifier_record",
                "label": label,
                "p": p,
                "target": target,
                "target_index": index,
            }
            continue
        inv = verifier.reduction_invariants(record, p)
        base_order = int(inv["base_order"])
        frontier = frontier_by_target.get(target) or {}
        frontier_base_order = int_value(frontier.get("base_order"), -1) if frontier else None
        generic_rho_steps = math.ceil(math.sqrt(math.pi * base_order / 2.0))
        contexts[target] = {
            "ainvs": [int(value) for value in record["ainvs"]],
            "ainvs_mod_p": [int(value) % p for value in record["ainvs"]],
            "base": verifier.point_to_json(inv["base"]),
            "base_order": base_order,
            "context_status": "verified",
            "frontier_base_order": frontier_base_order,
            "frontier_generic_rho_steps": frontier.get("generic_rho_steps") if frontier else None,
            "frontier_priority_score": frontier.get("priority_score") if frontier else None,
            "frontier_target_available": bool(frontier),
            "generic_rho_steps": generic_rho_steps,
            "group_order": int(inv["order"]),
            "label": label,
            "p": p,
            "precomputed_target": bool(inv.get("precomputed_target")),
            "target": target,
            "target_index": index,
        }
    return contexts


def row_lookup(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in (contract.get("contract") or {}).get("second_pass_contract") or []:
        if isinstance(row, dict) and row.get("row_id") is not None:
            rows[str(row["row_id"])] = row
    return rows


def exact_hash_by_row(contract: dict[str, Any]) -> dict[str, str]:
    out = {}
    for row in (contract.get("contract") or {}).get("exact_certificate_checks") or []:
        if isinstance(row, dict) and row.get("row_id") is not None:
            out[str(row["row_id"])] = str(row.get("certificate_hash") or "")
    return out


def inherited_rows(contract: dict[str, Any]) -> set[str]:
    return {
        str(row.get("row_id"))
        for row in (contract.get("contract") or {}).get("inherited_promotion_checks") or []
        if isinstance(row, dict) and row.get("row_id") is not None
    }


def full_family_backfill_rows(contract: dict[str, Any]) -> set[str]:
    rows: set[str] = set()
    for item in (contract.get("contract") or {}).get("direct_rank_backfill_manifest") or []:
        rows.update(str(row_id) for row_id in item.get("full_family_row_ids") or [])
    return rows


def packet_rows(
    group: dict[str, Any],
    row_by_id: dict[str, dict[str, Any]],
    exact_hashes: dict[str, str],
    inherited_ids: set[str],
    full_family_backfill_ids: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for offset, row_id in enumerate(group.get("second_pass_row_ids") or []):
        row = row_by_id.get(str(row_id), {})
        check_class = str(row.get("check_class") or "")
        family_masks = support_masks(row.get("matched_families") or [])
        rows.append(
            {
                "accepted_exact_certificate_hash": exact_hashes.get(str(row_id)),
                "check_class": check_class,
                "class_code": CLASS_CODES.get(check_class, 0),
                "direct_status": row.get("direct_status"),
                "evidence_positive": bool(row.get("evidence_positive")),
                "family_masks": family_masks,
                "first_pass_id": group.get("first_pass_id"),
                "group_id": group.get("group_id"),
                "is_full_family_backfill": str(row_id) in full_family_backfill_ids,
                "packet_row_offset": offset,
                "requires_direct_rank_export": check_class == "direct_rank_backfill_row",
                "requires_exact_certificate_before_promotion": str(row_id) in inherited_ids,
                "row_check_hash": row.get("row_check_hash"),
                "row_id": str(row_id),
                "score_certificate_source": row.get("score_certificate_source"),
                "selected_support_mask": support_mask(row.get("selected_term_support") or []),
                "selector": row.get("selector"),
                "top_k": int_value(row.get("top_k")),
                "transfer_index": int_value(row.get("transfer_index")),
            }
        )
    return rows


def build_packets(contract: dict[str, Any], contexts: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_by_id = row_lookup(contract)
    exact_hashes = exact_hash_by_row(contract)
    inherited_ids = inherited_rows(contract)
    full_family_backfill_ids = full_family_backfill_rows(contract)
    packets = []
    flat_rows = []
    row_offset = 0
    for packet_index, group in enumerate((contract.get("contract") or {}).get("first_pass_contract") or []):
        public = group.get("public_first_pass") or {}
        target = str(public.get("target"))
        context = contexts.get(target) or {}
        salts = sorted_ints(public.get("salts"))
        row_slots = [parse_row_key(row_key) for row_key in public.get("row_keys") or []]
        rows = packet_rows(group, row_by_id, exact_hashes, inherited_ids, full_family_backfill_ids)
        for row in rows:
            row["packet_index"] = packet_index
            row["global_row_offset"] = row_offset
            flat_rows.append(row)
            row_offset += 1
        material = {
            "first_pass_id": group.get("first_pass_id"),
            "group_id": group.get("group_id"),
            "row_ids": [row["row_id"] for row in rows],
            "salts": salts,
            "target": target,
            "transfer_index": int_value(public.get("transfer_index")),
        }
        packets.append(
            {
                "class_code": REPLAY_CLASS_CODES.get(str(group.get("replay_class")), 0),
                "direct_statuses": group.get("direct_statuses") or [],
                "first_pass_id": group.get("first_pass_id"),
                "global_row_offset": row_offset - len(rows),
                "group_id": group.get("group_id"),
                "packet_hash": stable_hash(material, "packet"),
                "packet_index": packet_index,
                "public_first_pass": {
                    "row_keys": public.get("row_keys") or [],
                    "salt_gap": int_value(public.get("salt_gap")),
                    "salt_min_mod4": int_value(public.get("salt_min_mod4")),
                    "salts": salts,
                    "target": target,
                    "transfer_index": int_value(public.get("transfer_index")),
                },
                "replay_class": group.get("replay_class"),
                "row_count": len(rows),
                "row_slots": row_slots,
                "target_index": int_value(context.get("target_index"), -1),
                "target_p": int_value(context.get("p"), -1),
                "u64_words": [
                    int_value(context.get("target_index"), -1),
                    int_value(public.get("transfer_index")),
                    salts[0] if len(salts) > 0 else 0,
                    salts[1] if len(salts) > 1 else 0,
                    int_value(public.get("salt_gap")),
                    int_value(public.get("salt_min_mod4")),
                    len(rows),
                    sum(1 for row in rows if row["check_class"] == "exact_positive_row"),
                    sum(1 for row in rows if row["check_class"] == "inherited_promotion_row"),
                    sum(1 for row in rows if row["check_class"] == "direct_rank_backfill_row"),
                ],
            }
        )
    return packets, flat_rows


def validate_manifest(
    contract: dict[str, Any],
    contexts: dict[str, dict[str, Any]],
    packets: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    def fail(code: str, **details: Any) -> None:
        failures.append({"code": code, **details})

    if contract.get("claim_status") != "FFE_SHARP_LANE_KERNEL_CONTRACT_READY":
        fail("source_contract_not_ready", claim_status=contract.get("claim_status"))
    if contract.get("failures"):
        fail("source_contract_has_failures", failures=contract.get("failures"))

    for target, context in contexts.items():
        if context.get("context_status") != "verified":
            fail("target_context_not_verified", target=target, context_status=context.get("context_status"))
            continue
        if context.get("frontier_base_order") is not None and int_value(context.get("frontier_base_order")) != int_value(context.get("base_order")):
            fail(
                "frontier_base_order_mismatch",
                target=target,
                frontier_base_order=context.get("frontier_base_order"),
                verifier_base_order=context.get("base_order"),
            )
        if context.get("frontier_generic_rho_steps") is not None and int_value(context.get("frontier_generic_rho_steps")) != int_value(context.get("generic_rho_steps")):
            fail(
                "frontier_rho_steps_mismatch",
                target=target,
                frontier_generic_rho_steps=context.get("frontier_generic_rho_steps"),
                verifier_generic_rho_steps=context.get("generic_rho_steps"),
            )

    first_ids = [str(packet.get("first_pass_id")) for packet in packets]
    if len(first_ids) != len(set(first_ids)):
        fail("duplicate_first_pass_ids", duplicates=sorted(key for key, count in Counter(first_ids).items() if count > 1))

    row_ids = [str(row.get("row_id")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        fail("duplicate_row_ids", duplicates=sorted(key for key, count in Counter(row_ids).items() if count > 1))

    rows_by_first: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_first[str(row.get("first_pass_id"))].append(row)
        if not (int_value(row.get("selected_support_mask")) & (1 << 13)):
            fail("row_missing_selected13", row_id=row.get("row_id"))
        if row.get("class_code") == 0:
            fail("unknown_row_check_class", row_id=row.get("row_id"), check_class=row.get("check_class"))
        if row.get("check_class") == "exact_positive_row" and not row.get("accepted_exact_certificate_hash"):
            fail("exact_row_missing_certificate_hash", row_id=row.get("row_id"))
        if row.get("check_class") == "inherited_promotion_row" and not row.get("requires_exact_certificate_before_promotion"):
            fail("inherited_row_not_gated", row_id=row.get("row_id"))
        if row.get("check_class") == "direct_rank_backfill_row" and not row.get("requires_direct_rank_export"):
            fail("backfill_row_not_marked_for_export", row_id=row.get("row_id"))

    for packet in packets:
        public = packet.get("public_first_pass") or {}
        target = str(public.get("target"))
        salts = sorted_ints(public.get("salts"))
        if target not in contexts:
            fail("packet_missing_target_context", packet_index=packet.get("packet_index"), target=target)
        if len(salts) != 2:
            fail("packet_bad_salt_count", packet_index=packet.get("packet_index"), salts=salts)
        else:
            if int_value(public.get("salt_gap")) != max(salts) - min(salts):
                fail("packet_salt_gap_mismatch", packet_index=packet.get("packet_index"), salts=salts)
            if int_value(public.get("salt_min_mod4")) != min(salts) % 4:
                fail("packet_salt_mod_mismatch", packet_index=packet.get("packet_index"), salts=salts)
        if len(packet.get("row_slots") or []) != 2:
            fail("packet_row_key_slot_count_mismatch", packet_index=packet.get("packet_index"), row_slots=packet.get("row_slots"))
        parsed_salts = sorted(int_value(slot.get("salt"), -1) for slot in packet.get("row_slots") or [])
        if parsed_salts != salts:
            fail("packet_row_key_salt_mismatch", packet_index=packet.get("packet_index"), parsed_salts=parsed_salts, salts=salts)
        parsed_targets = {str(slot.get("target")) for slot in packet.get("row_slots") or [] if slot.get("parse_status") == "ok"}
        if parsed_targets != {target}:
            fail("packet_row_key_target_mismatch", packet_index=packet.get("packet_index"), parsed_targets=sorted(parsed_targets), target=target)
        if packet.get("row_count") != len(rows_by_first[str(packet.get("first_pass_id"))]):
            fail("packet_row_count_mismatch", packet_index=packet.get("packet_index"), packet_row_count=packet.get("row_count"), observed=len(rows_by_first[str(packet.get("first_pass_id"))]))
        if packet.get("class_code") == 0:
            fail("unknown_replay_class", packet_index=packet.get("packet_index"), replay_class=packet.get("replay_class"))

    source_summary = contract.get("summary") or {}
    if int_value(source_summary.get("first_pass_group_count")) != len(packets):
        fail("source_first_pass_count_mismatch", source=source_summary.get("first_pass_group_count"), observed=len(packets))
    if int_value(source_summary.get("row_check_count")) != len(rows):
        fail("source_row_count_mismatch", source=source_summary.get("row_check_count"), observed=len(rows))
    if int_value(source_summary.get("exact_certificate_check_count")) != sum(1 for row in rows if row.get("check_class") == "exact_positive_row"):
        fail("source_exact_count_mismatch")
    if int_value(source_summary.get("inherited_promotion_check_count")) != sum(1 for row in rows if row.get("check_class") == "inherited_promotion_row"):
        fail("source_inherited_count_mismatch")
    if int_value(source_summary.get("backfill_group_count")) != sum(1 for packet in packets if packet.get("replay_class") == "direct_rank_backfill"):
        fail("source_backfill_group_count_mismatch")

    return failures


def summarize(contexts: dict[str, dict[str, Any]], packets: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(str(row.get("check_class")) for row in rows)
    replay_counts = Counter(str(packet.get("replay_class")) for packet in packets)
    return {
        "packet_count": len(packets),
        "row_count": len(rows),
        "row_class_counts": dict(sorted(class_counts.items())),
        "replay_class_counts": dict(sorted(replay_counts.items())),
        "targets": sorted(contexts),
        "target_context_status_counts": dict(sorted(Counter(str(ctx.get("context_status")) for ctx in contexts.values()).items())),
        "unique_packet_hash_count": len({str(packet.get("packet_hash")) for packet in packets}),
    }


def c_u64_array(values: list[int]) -> str:
    return "{" + ", ".join(f"{int(value)}ULL" for value in values) + "}"


def render_c_header(payload: dict[str, Any]) -> str:
    packets = payload["packets"]
    rows = payload["rows"]
    packet_lines = []
    for packet in packets:
        packet_lines.append(
            "  {"
            f"{int_value(packet.get('target_index'))}ULL, "
            f"{int_value((packet.get('public_first_pass') or {}).get('transfer_index'))}ULL, "
            f"{c_u64_array(sorted_ints((packet.get('public_first_pass') or {}).get('salts')))}, "
            f"{int_value((packet.get('public_first_pass') or {}).get('salt_gap'))}ULL, "
            f"{int_value((packet.get('public_first_pass') or {}).get('salt_min_mod4'))}ULL, "
            f"{int_value(packet.get('global_row_offset'))}ULL, "
            f"{int_value(packet.get('row_count'))}ULL, "
            f"{int_value(packet.get('class_code'))}ULL"
            "},"
        )
    row_lines = []
    for row in rows:
        family = list(row.get("family_masks") or [])[:3]
        while len(family) < 3:
            family.append(0)
        row_lines.append(
            "  {"
            f"{int_value(row.get('packet_index'))}ULL, "
            f"{int_value(row.get('class_code'))}ULL, "
            f"{int_value(row.get('selected_support_mask'))}ULL, "
            f"{c_u64_array(family)}, "
            f"{1 if row.get('requires_direct_rank_export') else 0}ULL, "
            f"{1 if row.get('requires_exact_certificate_before_promotion') else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_PACKET_MANIFEST_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_PACKET_MANIFEST_H

#include <stdint.h>

#define SELECTED13_SHARP_PACKET_COUNT {len(packets)}
#define SELECTED13_SHARP_ROW_COUNT {len(rows)}

typedef struct {{
  uint64_t target_index;
  uint64_t transfer_index;
  uint64_t salts[2];
  uint64_t salt_gap;
  uint64_t salt_min_mod4;
  uint64_t row_offset;
  uint64_t row_count;
  uint64_t replay_class_code;
}} selected13_sharp_first_pass_packet_t;

typedef struct {{
  uint64_t packet_index;
  uint64_t class_code;
  uint64_t selected_support_mask;
  uint64_t family_masks[3];
  uint64_t requires_direct_rank_export;
  uint64_t requires_exact_certificate_before_promotion;
}} selected13_sharp_second_pass_row_t;

static const selected13_sharp_first_pass_packet_t SELECTED13_SHARP_FIRST_PASS_PACKETS[] = {{
{chr(10).join(packet_lines)}
}};

static const selected13_sharp_second_pass_row_t SELECTED13_SHARP_SECOND_PASS_ROWS[] = {{
{chr(10).join(row_lines)}
}};

#endif
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, type=Path)
    parser.add_argument("--task-dir", default=DEFAULT_TASK_DIR, type=Path)
    parser.add_argument("--frontier-targets", default=DEFAULT_FRONTIER_TARGETS, type=Path)
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--c-header-out", default=DEFAULT_HEADER_OUT, type=Path)
    parser.add_argument("--no-c-header", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_json(args.contract)
    contexts = target_contexts(contract, args.task_dir, args.frontier_targets)
    packets, rows = build_packets(contract, contexts)
    failures = validate_manifest(contract, contexts, packets, rows)
    payload = {
        "artifacts": {
            "contract": str(args.contract),
            "frontier_targets": str(args.frontier_targets),
            "task_dir": str(args.task_dir),
        },
        "claim_status": "FFE_SHARP_LANE_KERNEL_PACKET_MANIFEST_READY" if not failures else "FFE_SHARP_LANE_KERNEL_PACKET_MANIFEST_FAILED_CHECK",
        "created_at": now_iso(),
        "failures": failures,
        "honesty_boundary": [
            "This is a typed packet manifest for the selected13 sharp-lane contract.",
            "It does not evaluate summation polynomials, run a native kernel, export direct/rank rows, or recover an ECDLP scalar.",
            "The optional C header is an ABI surface; matching it is necessary but not sufficient for a rho-beating algorithm claim.",
        ],
        "packets": packets,
        "rows": rows,
        "schema": SCHEMA,
        "source_summary": contract.get("summary"),
        "summary": summarize(contexts, packets, rows),
        "target_contexts": contexts,
    }
    write_json(args.out, payload)
    if not args.no_c_header:
        args.c_header_out.parent.mkdir(parents=True, exist_ok=True)
        args.c_header_out.write_text(render_c_header(payload))
        payload["artifacts"]["c_header"] = str(args.c_header_out)
        write_json(args.out, payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "failures": failures,
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
