#!/usr/bin/env python3
"""Audit frozen public-selector candidates against direct rank certificates.

The accepted-form public feature miner promotes support-scout rows using only
source-side tokens.  This audit joins those promoted rows to a later direct
missing-column bridge audit, so we can measure whether the public selector
actually captured accepted-form rank gain on held-out rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_promoted_candidate_rank_audit.v1"


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


def float_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def support_key(raw: Any) -> str:
    return ",".join(str(item) for item in sorted(int_value(item) for item in (raw or [])))


def row_key_key(raw: Any) -> str:
    return "|".join(sorted(str(item) for item in (raw or [])))


def identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("transfer_index")),
        str(row.get("selector")),
        int_value(row.get("top_k")),
        row_key_key(row.get("row_keys")),
        support_key(row.get("selected_term_support")),
    )


def family_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("selector")),
            str(int_value(row.get("top_k"))),
            support_key(row.get("selected_term_support")),
        ]
    )


def compact_rules(row: dict[str, Any]) -> list[str]:
    return [str(rule.get("token")) for rule in (row.get("matched_rules") or [])]


def in_range(row: dict[str, Any], start: int | None, end: int | None) -> bool:
    transfer = int_value(row.get("transfer_index"))
    return (start is None or transfer >= start) and (end is None or transfer <= end)


def summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accepted_missing_rank_gain_count": sum(
            1 for row in rows if row.get("direct_audit_classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN"
        ),
        "candidate_count": len(rows),
        "direct_audit_match_count": sum(1 for row in rows if row.get("direct_audit_matched")),
        "direct_below_rho_count": sum(1 for row in rows if row.get("direct_audit_below_rho")),
        "direct_classifications": dict(Counter(str(row.get("direct_audit_classification")) for row in rows)),
        "direct_rank_gain_count": sum(1 for row in rows if int_value(row.get("direct_audit_rank_gain")) > 0),
        "posthoc_direct_verified_count": sum(1 for row in rows if row.get("direct_public_key_verified_posthoc")),
        "posthoc_shared_verified_count": sum(1 for row in rows if row.get("shared_product_public_key_verified_posthoc")),
        "transfers": sorted({int_value(row.get("transfer_index")) for row in rows}),
        "unmatched_count": sum(1 for row in rows if not row.get("direct_audit_matched")),
    }


def summarize_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[family_key(row)].append(row)
    summaries = []
    for key, items in buckets.items():
        first = items[0]
        summaries.append(
            {
                "accepted_missing_rank_gain_count": sum(
                    1
                    for item in items
                    if item.get("direct_audit_classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN"
                ),
                "candidate_count": len(items),
                "direct_audit_match_count": sum(1 for item in items if item.get("direct_audit_matched")),
                "direct_rank_gain_count": sum(1 for item in items if int_value(item.get("direct_audit_rank_gain")) > 0),
                "family_key": key,
                "posthoc_direct_verified_count": sum(
                    1 for item in items if item.get("direct_public_key_verified_posthoc")
                ),
                "selected_term_support": sorted(int_value(item) for item in (first.get("selected_term_support") or [])),
                "selector": first.get("selector"),
                "top_k": int_value(first.get("top_k")),
                "transfers": sorted({int_value(item.get("transfer_index")) for item in items}),
                "unmatched_count": sum(1 for item in items if not item.get("direct_audit_matched")),
            }
        )
    summaries.sort(
        key=lambda item: (
            -int_value(item.get("accepted_missing_rank_gain_count")),
            -int_value(item.get("direct_rank_gain_count")),
            -int_value(item.get("posthoc_direct_verified_count")),
            -int_value(item.get("candidate_count")),
            str(item.get("family_key")),
        )
    )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miner", required=True, type=Path)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--candidate-field", default="promoted_holdout_candidates")
    parser.add_argument("--audit-start", type=int)
    parser.add_argument("--audit-end", type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    miner = load_json(args.miner)
    bridge = load_json(args.bridge_audit)
    candidates = [
        row
        for row in (miner.get(args.candidate_field) or [])
        if in_range(row, args.audit_start, args.audit_end)
    ]
    certificates = [
        row
        for row in (bridge.get("certificates") or [])
        if in_range(row, args.audit_start, args.audit_end)
    ]
    certs_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cert in certificates:
        certs_by_identity[identity(cert)].append(cert)

    audited_candidates = []
    promoted_identities = {identity(candidate) for candidate in candidates}
    for candidate in candidates:
        matches = certs_by_identity.get(identity(candidate), [])
        cert = matches[0] if matches else None
        rank_score = cert.get("rank_score") if cert else {}
        audited_candidates.append(
            {
                "accepted_missing_columns": cert.get("accepted_missing_columns") if cert else [],
                "accepted_missing_not_selected_columns": cert.get("accepted_missing_not_selected_columns")
                if cert
                else [],
                "direct_audit_below_rho": bool(cert and (float_value(cert.get("direct_ops_over_rho")) or 999.0) < 1.0),
                "direct_audit_classification": cert.get("classification") if cert else "NO_DIRECT_AUDIT_MATCH",
                "direct_audit_duplicate_match_count": len(matches),
                "direct_audit_matched": cert is not None,
                "direct_audit_rank_gain": rank_score.get("rank_gain") if rank_score else None,
                "direct_audit_unique_factor_relation_gain": rank_score.get("unique_factor_relation_gain")
                if rank_score
                else None,
                "direct_ops_over_rho": float_value(candidate.get("direct_ops_over_rho")),
                "direct_public_key_verified_posthoc": bool(candidate.get("direct_public_key_verified_posthoc")),
                "form_supports": cert.get("form_supports") if cert else [],
                "matched_rule_count": int_value(candidate.get("matched_rule_count")),
                "matched_rule_tokens": compact_rules(candidate),
                "promotion_tier": candidate.get("promotion_tier"),
                "range": candidate.get("range"),
                "row_keys": candidate.get("row_keys") or [],
                "score": int_value(candidate.get("score")),
                "selected_term_support": sorted(
                    int_value(item) for item in (candidate.get("selected_term_support") or [])
                ),
                "selector": candidate.get("selector"),
                "shared_product_public_key_verified_posthoc": bool(
                    candidate.get("shared_product_public_key_verified_posthoc")
                ),
                "top_k": int_value(candidate.get("top_k")),
                "transfer_index": int_value(candidate.get("transfer_index")),
            }
        )

    non_promoted_certificates = [
        cert
        for cert in certificates
        if identity(cert) not in promoted_identities
    ]
    captured_rank_gain = [
        row
        for row in audited_candidates
        if row.get("direct_audit_classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN"
    ]
    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
            "miner": str(args.miner),
        },
        "audited_candidates": audited_candidates,
        "claim_status": (
            "PROMOTED_PUBLIC_SELECTOR_HAS_HOLDOUT_ACCEPTED_MISSING_RANK_GAIN"
            if captured_rank_gain
            else "PROMOTED_PUBLIC_SELECTOR_HAS_NO_HOLDOUT_ACCEPTED_MISSING_RANK_GAIN"
            if audited_candidates
            else "NO_PROMOTED_PUBLIC_SELECTOR_CANDIDATES_IN_AUDIT_RANGE"
        ),
        "created_at": now_iso(),
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "The promoted selector came from the miner; direct audit fields are used only for holdout evaluation.",
            "Direct accepted-form rank gain is not target descent or a deployed-curve speedup.",
            "Unmatched candidates may be outside the current direct-certificate horizon.",
        ],
        "parameters": {
            "audit_end": args.audit_end,
            "audit_start": args.audit_start,
            "candidate_field": args.candidate_field,
        },
        "schema": SCHEMA,
        "summary": {
            "candidate_summary": summarize_candidates(audited_candidates),
            "direct_audit_certificate_count": len(certificates),
            "direct_audit_classifications": dict(Counter(str(cert.get("classification")) for cert in certificates)),
            "direct_audit_rank_gain_count": sum(
                1 for cert in certificates if int_value((cert.get("rank_score") or {}).get("rank_gain")) > 0
            ),
            "family_summaries": summarize_families(audited_candidates),
            "non_promoted_direct_audit_certificate_count": len(non_promoted_certificates),
            "non_promoted_direct_audit_rank_gain_count": sum(
                1
                for cert in non_promoted_certificates
                if int_value((cert.get("rank_score") or {}).get("rank_gain")) > 0
            ),
        },
    }
    write_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps({"claim_status": payload["claim_status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
