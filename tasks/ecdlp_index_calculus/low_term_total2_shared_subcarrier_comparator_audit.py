#!/usr/bin/env python3
"""Audit exported and missing rows in a shared-subcarrier work-order queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_shared_subcarrier_comparator_audit.v1"
DEFAULT_TOKENS = "selected_has=13,salt_min_mod4=2"


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


def parse_tokens(raw: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not tokens:
        raise ValueError("At least one token is required")
    return tokens


def family_key(raw: Any) -> str:
    return ",".join(str(int_value(item)) for item in (raw or []))


def support_key(raw: Any) -> str:
    return ",".join(str(int_value(item)) for item in (raw or []))


def rank_gain(row: dict[str, Any]) -> int:
    cert = row.get("best_direct_certificate") or {}
    return int_value(cert.get("rank_gain"))


def unique_gain(row: dict[str, Any]) -> int:
    cert = row.get("best_direct_certificate") or {}
    return int_value(cert.get("unique_factor_relation_gain"))


def classification(row: dict[str, Any]) -> str:
    cert = row.get("best_direct_certificate") or {}
    return str(cert.get("classification") or "missing")


def form_supports(row: dict[str, Any]) -> list[str]:
    cert = row.get("best_direct_certificate") or {}
    return [support_key(item) for item in cert.get("form_supports") or []]


def has_required_tokens(row: dict[str, Any], required: tuple[str, ...]) -> bool:
    return set(required) <= set(row.get("matched_public_tokens") or [])


def row_brief(row: dict[str, Any]) -> dict[str, Any]:
    brief: dict[str, Any] = {
        "classification": classification(row),
        "direct_status": row.get("direct_status"),
        "form_supports": form_supports(row),
        "matched_families": row.get("matched_families") or [],
        "rank_gain": rank_gain(row),
        "range": row.get("range"),
        "row_keys": row.get("row_keys") or [],
        "selected_term_support": row.get("selected_term_support") or [],
        "support_size": len(row.get("selected_term_support") or []),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": int_value(row.get("transfer_index")),
        "unique_factor_relation_gain": unique_gain(row),
    }
    return brief


def sorted_unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for row in sorted(
        rows,
        key=lambda item: (
            int_value(item.get("transfer_index")),
            str(item.get("direct_status")),
            str(item.get("selected_term_support")),
            str(item.get("row_keys")),
        ),
    ):
        key = (
            int_value(row.get("transfer_index")),
            str(row.get("direct_status")),
            tuple(row.get("selected_term_support") or []),
            tuple(row.get("row_keys") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_status_counts = Counter(str(row.get("direct_status")) for row in rows)
    classification_counts = Counter(classification(row) for row in rows)
    form_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for row in rows:
        for form in form_supports(row):
            form_counts[form] += 1
        for family in row.get("matched_families") or []:
            family_counts[family_key(family)] += 1
    exported = [row for row in rows if row.get("direct_status") == "direct_certificate_exported"]
    missing = [row for row in rows if row.get("direct_status") == "direct_certificate_missing"]
    return {
        "classification_counts": dict(sorted(classification_counts.items())),
        "direct_status_counts": dict(sorted(direct_status_counts.items())),
        "exported_rank_gain_count": sum(1 for row in exported if rank_gain(row) > 0),
        "exported_unique_gain_total": sum(unique_gain(row) for row in exported),
        "family_match_counts": dict(sorted(family_counts.items())),
        "form_support_counts": dict(sorted(form_counts.items())),
        "missing_transfers": sorted({int_value(row.get("transfer_index")) for row in missing}),
        "row_count": len(rows),
        "unique_transfer_count": len({int_value(row.get("transfer_index")) for row in rows}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workorder", required=True, type=Path)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workorder = load_json(args.workorder)
    required_tokens = parse_tokens(args.tokens)
    shared_rows = [row for row in workorder.get("candidate_rows") or [] if has_required_tokens(row, required_tokens)]
    shared_rows = sorted_unique(shared_rows)
    full_family_rows = [row for row in shared_rows if len(row.get("matched_families") or []) >= 3]
    partial_rows = [row for row in shared_rows if len(row.get("matched_families") or []) < 3]
    exported_rows = [row for row in shared_rows if row.get("direct_status") == "direct_certificate_exported"]
    missing_rows = [row for row in shared_rows if row.get("direct_status") == "direct_certificate_missing"]
    exported_rank_gain = [row for row in exported_rows if rank_gain(row) > 0]

    claim_status = (
        "SHARED_SUBCARRIER_HAS_MISSING_FULL_FAMILY_TARGETS_AND_EXPORTED_RANK_GAIN_COMPARATORS"
        if any(len(row.get("matched_families") or []) >= 3 for row in missing_rows) and exported_rank_gain
        else "SHARED_SUBCARRIER_HAS_MISSING_FULL_FAMILY_TARGETS"
        if any(len(row.get("matched_families") or []) >= 3 for row in missing_rows)
        else "SHARED_SUBCARRIER_HAS_EXPORTED_RANK_GAIN_COMPARATORS"
        if exported_rank_gain
        else "SHARED_SUBCARRIER_COMPARATOR_HAS_NO_RANK_GAIN_EXPORTS"
    )
    payload = {
        "artifacts": {
            "workorder": str(args.workorder),
        },
        "claim_status": claim_status,
        "created_at": now_iso(),
        "exported_rank_gain_rows": [row_brief(row) for row in exported_rank_gain],
        "full_family_shared_rows": [row_brief(row) for row in full_family_rows],
        "honesty_boundary": [
            "This audit compares work-order rows; it does not create direct/rank evidence.",
            "Rows marked direct_certificate_missing still require direct/rank export.",
            "Exported rows are posthoc comparators for the public subcarrier, not an ECDLP speedup proof.",
        ],
        "missing_shared_rows": [row_brief(row) for row in missing_rows],
        "parameters": {
            "required_tokens": list(required_tokens),
        },
        "partial_shared_rows": [row_brief(row) for row in partial_rows],
        "schema": SCHEMA,
        "summary": {
            "exported_rank_gain_transfers": sorted({int_value(row.get("transfer_index")) for row in exported_rank_gain}),
            "missing_full_family_transfers": sorted(
                {
                    int_value(row.get("transfer_index"))
                    for row in missing_rows
                    if len(row.get("matched_families") or []) >= 3
                }
            ),
            "shared": summarize(shared_rows),
            "shared_exported": summarize(exported_rows),
            "shared_full_family": summarize(full_family_rows),
            "shared_missing": summarize(missing_rows),
            "shared_partial": summarize(partial_rows),
        },
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": claim_status, "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
