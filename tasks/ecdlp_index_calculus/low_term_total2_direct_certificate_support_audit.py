#!/usr/bin/env python3
"""Audit factor-support motifs in exported direct relation certificates.

Selected-leaf scouts can show that a source row touches desirable factor
columns, but the accepted public relation forms may collapse to a different
support.  This audit reads direct/source relation-equation certificate files
and reports the factor-column supports actually present in exported forms.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_KNOWN_COLUMNS = "7,8,9,10,11,12,13,14"
DEFAULT_PRIORITY_COLUMNS = "15"
DEFAULT_TARGET_MOTIFS = "8:11,9:11,10:13,10:14,11:13"
DEFAULT_AVOID_MOTIFS = "2:4,3:5"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


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


def parse_paths(raw: str) -> list[Path]:
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_set(raw: str) -> set[int]:
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def parse_motifs(raw: str) -> set[tuple[int, ...]]:
    motifs: set[tuple[int, ...]] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        motifs.add(tuple(sorted(int(item.strip()) for item in chunk.split(":") if item.strip())))
    return motifs


def form_support(form: dict[str, Any], order: int) -> tuple[int, ...]:
    coeffs = form.get("coeffs") or []
    return tuple(
        index
        for index, coeff in enumerate(coeffs[1:])
        if int_value(coeff) % order != 0
    )


def motif_hits(support: tuple[int, ...], motifs: set[tuple[int, ...]]) -> list[list[int]]:
    support_set = set(support)
    return [list(motif) for motif in sorted(motifs) if set(motif).issubset(support_set)]


def cert_key(cert: dict[str, Any]) -> str:
    selected = cert.get("selected") or {}
    return (
        f"{selected.get('target')}|{selected.get('transfer_index')}|"
        f"{selected.get('selector')}|{selected.get('top_k')}"
    )


def analyze_certificate(
    source: Path,
    cert: dict[str, Any],
    known_columns: set[int],
    priority_columns: set[int],
    target_motifs: set[tuple[int, ...]],
    avoid_motifs: set[tuple[int, ...]],
) -> dict[str, Any]:
    order = int_value(cert.get("order"))
    supports = [form_support(form, order) for form in cert.get("forms") or []]
    known_only = [support for support in supports if support and set(support) <= known_columns]
    motif_supports = [support for support in supports if motif_hits(support, target_motifs)]
    priority_supports = [support for support in supports if set(support) & priority_columns]
    avoid_supports = [support for support in supports if motif_hits(support, avoid_motifs)]
    selected = cert.get("selected") or {}
    return {
        "avoid_supports": [list(support) for support in sorted(set(avoid_supports))],
        "certificate_status": cert.get("certificate_status"),
        "case": cert_key(cert),
        "derived_secret": cert.get("derived_secret"),
        "direct_ops_over_rho": selected.get("direct_ops_over_rho"),
        "forms_count": len(supports),
        "known_only_supports": [list(support) for support in sorted(set(known_only))],
        "motif_supports": [list(support) for support in sorted(set(motif_supports))],
        "priority_supports": [list(support) for support in sorted(set(priority_supports))],
        "rank": cert.get("rank"),
        "selected_term_support": cert.get("selected_term_support") or [],
        "source": str(source),
        "unique_form_supports": [list(support) for support in sorted(set(supports))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", required=True)
    parser.add_argument("--known-columns", default=DEFAULT_KNOWN_COLUMNS)
    parser.add_argument("--priority-columns", default=DEFAULT_PRIORITY_COLUMNS)
    parser.add_argument("--target-motifs", default=DEFAULT_TARGET_MOTIFS)
    parser.add_argument("--avoid-motifs", default=DEFAULT_AVOID_MOTIFS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    known_columns = parse_int_set(args.known_columns)
    priority_columns = parse_int_set(args.priority_columns)
    target_motifs = parse_motifs(args.target_motifs)
    avoid_motifs = parse_motifs(args.avoid_motifs)
    rows = []
    support_counter: Counter[tuple[int, ...]] = Counter()
    for path in parse_paths(args.certificates):
        payload = load_json(path)
        for cert in payload.get("certificates") or []:
            row = analyze_certificate(
                path,
                cert,
                known_columns,
                priority_columns,
                target_motifs,
                avoid_motifs,
            )
            rows.append(row)
            support_counter.update(tuple(support) for support in row["unique_form_supports"])

    direct_costs = [
        value
        for value in (float_value(row.get("direct_ops_over_rho")) for row in rows)
        if value is not None
    ]
    motif_form_count = sum(1 for row in rows if row["motif_supports"])
    priority_form_count = sum(1 for row in rows if row["priority_supports"])
    known_only_form_count = sum(1 for row in rows if row["known_only_supports"])
    summary = {
        "all_certificates_pass": all(
            row["certificate_status"] == "PUBLIC_DIRECT_RELATION_EQUATIONS_VERIFY_PUBLIC_KEY"
            for row in rows
        )
        if rows
        else False,
        "avoid_support_certificate_count": sum(1 for row in rows if row["avoid_supports"]),
        "certificate_count": len(rows),
        "direct_below_rho_count": sum(
            1 for value in direct_costs if value < 1.0
        ),
        "known_only_support_certificate_count": known_only_form_count,
        "motif_support_certificate_count": motif_form_count,
        "priority_support_certificate_count": priority_form_count,
        "unique_form_support_count": len(support_counter),
    }
    if motif_form_count:
        claim_status = "RELATION_FORMS_INCLUDE_TARGET_MOTIFS"
    elif priority_form_count:
        claim_status = "RELATION_FORMS_KEEP_PRIORITY_COLUMNS_BUT_NOT_TARGET_MOTIFS"
    else:
        claim_status = "SELECTED_SUPPORT_MOTIFS_DID_NOT_SURVIVE_RELATION_FORMS"
    payload = {
        "created_at": now_iso(),
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "MODEL-BOUND: this inspects exported public relation forms, not a complete target-descent proof.",
            "A selected-support motif is not counted here unless the accepted relation form itself contains that support.",
        ],
        "parameters": {
            "avoid_motifs": [list(motif) for motif in sorted(avoid_motifs)],
            "known_columns": sorted(known_columns),
            "priority_columns": sorted(priority_columns),
            "target_motifs": [list(motif) for motif in sorted(target_motifs)],
        },
        "schema": "ecdlp.low_term_total2_direct_certificate_support_audit.v1",
        "summary": {**summary, "claim_status": claim_status},
        "support_counts": [
            {"count": count, "support": list(support)}
            for support, count in support_counter.most_common()
        ],
        "certificates": rows,
    }
    write_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
