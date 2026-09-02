#!/usr/bin/env python3
"""Build a work-order queue for the shared low-term total2 public subcarrier.

The family split miners showed that several clean accepted-form rank-gain
families share the same public/source-side tokens:

    salt_min_mod4=2 AND selected_has=13

This script applies that public rule to support-scout rows and marks whether
the row already has a direct certificate export. It emits work orders only; it
does not claim rank gain for rows without direct/rank labels.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_shared_subcarrier_workorder.v1"
DEFAULT_FAMILIES = "11,15;10,14;0,5"
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


def float_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_families(raw: str) -> list[tuple[int, ...]]:
    families: list[tuple[int, ...]] = []
    for chunk in raw.split(";"):
        values = tuple(sorted(int(part.strip()) for part in chunk.split(",") if part.strip()))
        if values:
            if len(values) < 2:
                raise ValueError("Each family must contain at least two columns")
            families.append(values)
    if not families:
        raise ValueError("At least one family is required")
    return families


def parse_tokens(raw: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not tokens:
        raise ValueError("At least one token is required")
    return tokens


def support_tuple(raw: Any) -> tuple[int, ...]:
    return tuple(sorted(int_value(item) for item in (raw or [])))


def salts_from_row_keys(row_keys: Any) -> list[int]:
    salts = []
    for row_key in row_keys or []:
        match = re.search(r"salt(\d+)", str(row_key))
        if match:
            salts.append(int(match.group(1)))
    return sorted(salts)


def range_from_path(path: Path) -> str | None:
    match = re.search(r"_(\d+)_(\d+)_probe\.json$", path.name)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def glob_paths(pattern: str) -> list[Path]:
    return sorted(Path(path) for path in glob.glob(pattern))


def in_range(row: dict[str, Any], start: int | None, end: int | None) -> bool:
    transfer = int_value(row.get("transfer_index"))
    return (start is None or transfer >= start) and (end is None or transfer <= end)


def public_tokens(row: dict[str, Any]) -> set[str]:
    selected = support_tuple(row.get("selected_term_support"))
    selected_set = set(selected)
    top_k = int_value(row.get("top_k"))
    tokens = {
        f"support_size={len(selected)}",
        f"top_k={top_k}",
    }
    for column in selected:
        tokens.add(f"selected_has={column}")

    salts = salts_from_row_keys(row.get("row_keys") or [])
    if salts:
        gap = salts[-1] - salts[0] if len(salts) >= 2 else 0
        tokens.update(
            {
                f"salt_adjacent={gap == 1}",
                f"salt_count={len(salts)}",
                f"salt_gap={gap}",
                f"salt_max_mod4={salts[-1] % 4}",
                f"salt_min_mod4={salts[0] % 4}",
                f"salt_pair_mod4={','.join(str(salt % 4) for salt in salts)}",
                f"salt_sum_mod4={sum(salts) % 4}",
            }
        )
    if 13 in selected_set and top_k:
        tokens.add(f"selected_has=13|top_k={top_k}")
    return tokens


def family_key(family: tuple[int, ...]) -> str:
    return ",".join(str(item) for item in family)


def compatible_families(row: dict[str, Any], families: list[tuple[int, ...]]) -> list[list[int]]:
    selected = set(support_tuple(row.get("selected_term_support")))
    return [list(family) for family in families if set(family) <= selected]


def support_report_key(row: dict[str, Any]) -> tuple[int, int, tuple[int, ...]]:
    return (
        int_value(row.get("transfer_index")),
        int_value(row.get("top_k")),
        support_tuple(row.get("selected_term_support")),
    )


def bridge_support_index(bridge: dict[str, Any]) -> dict[tuple[int, int, tuple[int, ...]], dict[str, Any]]:
    index: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]] = {}
    for row in bridge.get("support_reports") or []:
        key = support_report_key(row)
        index[key] = row
    return index


def certificate_index(bridge: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in bridge.get("certificates") or []:
        index[int_value(row.get("transfer_index"))].append(row)
    return index


def rank_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("rank_gain"))


def unique_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("unique_factor_relation_gain"))


def direct_status(row: dict[str, Any], support_index: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]]) -> str:
    support_row = support_index.get(support_report_key(row))
    if support_row is None:
        return "support_report_missing"
    if support_row.get("direct_certificate_exported"):
        return "direct_certificate_exported"
    return "direct_certificate_missing"


def score_row(
    row: dict[str, Any],
    required_tokens: tuple[str, ...],
    families: list[tuple[int, ...]],
    support_index: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]],
) -> tuple[int, list[str], list[list[int]], str]:
    tokens = public_tokens(row)
    matched_tokens = [token for token in required_tokens if token in tokens]
    family_matches = compatible_families(row, families)
    status = direct_status(row, support_index)
    score = 0
    score += 100 * int(len(matched_tokens) == len(required_tokens))
    score += 15 * len(matched_tokens)
    score += 12 * len(family_matches)
    score += 5 * int(row.get("public_product_gate_selected") is True)
    score += 3 * int(status == "direct_certificate_missing")
    score -= 20 * int(status == "direct_certificate_exported")
    return score, matched_tokens, family_matches, status


def load_scout_rows(paths: list[Path], start: int | None, end: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        artifact_range = range_from_path(path)
        for row in payload.get("case_reports") or []:
            if not in_range(row, start, end):
                continue
            item = dict(row)
            item["artifact"] = str(path)
            item["range"] = artifact_range
            rows.append(item)
    return rows


def candidate_summary(
    row: dict[str, Any],
    score: int,
    matched_tokens: list[str],
    family_matches: list[list[int]],
    status: str,
    certs_by_transfer: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    transfer = int_value(row.get("transfer_index"))
    certificates = certs_by_transfer.get(transfer, [])
    best_certificate = max(certificates, key=rank_gain, default=None)
    summary: dict[str, Any] = {
        "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
        "direct_status": status,
        "matched_families": family_matches,
        "matched_public_tokens": matched_tokens,
        "priority_hits": sorted(int_value(item) for item in row.get("priority_hits") or []),
        "public_product_gate_selected": bool(row.get("public_product_gate_selected")),
        "range": row.get("range"),
        "row_keys": row.get("row_keys") or [],
        "salts": salts_from_row_keys(row.get("row_keys") or []),
        "score": score,
        "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
        "selector": row.get("selector"),
        "shared_product_ops_over_rho": float_value(row.get("shared_product_ops_over_rho")),
        "target": row.get("target"),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": transfer,
    }
    if best_certificate:
        summary["best_direct_certificate"] = {
            "accepted_missing_columns": sorted(int_value(item) for item in best_certificate.get("accepted_missing_columns") or []),
            "classification": best_certificate.get("classification"),
            "form_supports": [list(support_tuple(item)) for item in best_certificate.get("form_supports") or []],
            "rank_gain": rank_gain(best_certificate),
            "unique_factor_relation_gain": unique_gain(best_certificate),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-scouts", required=True)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = parse_families(args.families)
    required_tokens = parse_tokens(args.tokens)
    paths = glob_paths(args.support_scouts)
    bridge = load_json(args.bridge_audit)
    support_index = bridge_support_index(bridge)
    certs_by_transfer = certificate_index(bridge)
    scout_rows = load_scout_rows(paths, args.start, args.end)

    candidates = []
    status_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    token_match_counts: Counter[str] = Counter()
    for row in scout_rows:
        score, matched_tokens, family_matches, status = score_row(row, required_tokens, families, support_index)
        status_counts[status] += 1
        for family in family_matches:
            family_counts[",".join(str(item) for item in family)] += 1
        token_match_counts[str(len(matched_tokens))] += 1
        if matched_tokens or family_matches:
            candidates.append(candidate_summary(row, score, matched_tokens, family_matches, status, certs_by_transfer))

    candidates.sort(
        key=lambda item: (
            -int_value(item.get("score")),
            item.get("direct_status") != "direct_certificate_missing",
            -len(item.get("matched_families") or []),
            int_value(item.get("transfer_index")),
            int_value(item.get("top_k")),
            str(item.get("selected_term_support")),
        )
    )
    limited = candidates[: max(1, args.candidate_limit)]
    shared = [
        item
        for item in candidates
        if len(item.get("matched_public_tokens") or []) == len(required_tokens)
    ]
    shared_missing = [item for item in shared if item.get("direct_status") == "direct_certificate_missing"]
    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
            "support_scouts": args.support_scouts,
        },
        "candidate_rows": limited,
        "claim_status": (
            "SHARED_SUBCARRIER_HAS_DIRECT_MISSING_WORK_ORDERS"
            if shared_missing
            else "SHARED_SUBCARRIER_ONLY_ALREADY_EXPORTED_OR_ABSENT"
            if shared
            else "SHARED_SUBCARRIER_NOT_PRESENT_IN_SCOUT_WINDOW"
        ),
        "created_at": now_iso(),
        "honesty_boundary": [
            "This is a work-order queue, not a rank-gain certificate.",
            "Scores use public support, selector, and salt tokens from support-scout rows.",
            "Direct/rank labels are used only to mark already exported rows and summarize posthoc known certificates.",
            "Rows with direct_certificate_missing still need direct/rank export before any ECDLP progress claim.",
        ],
        "parameters": {
            "candidate_limit": args.candidate_limit,
            "end": args.end,
            "families": [list(family) for family in families],
            "required_tokens": list(required_tokens),
            "start": args.start,
        },
        "schema": SCHEMA,
        "summary": {
            "candidate_count": len(candidates),
            "direct_status_counts": dict(sorted(status_counts.items())),
            "family_match_counts": dict(sorted(family_counts.items())),
            "scout_row_count": len(scout_rows),
            "shared_direct_missing_count": len(shared_missing),
            "shared_subcarrier_count": len(shared),
            "shared_subcarrier_direct_status_counts": dict(sorted(Counter(item.get("direct_status") for item in shared).items())),
            "token_match_counts": dict(sorted(token_match_counts.items())),
            "top_missing_transfers": sorted({int_value(item.get("transfer_index")) for item in shared_missing})[:40],
        },
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
