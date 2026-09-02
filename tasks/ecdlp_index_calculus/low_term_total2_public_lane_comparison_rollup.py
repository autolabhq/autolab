#!/usr/bin/env python3
"""Roll up public-carrier lane comparator audits into a single comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_public_lane_comparison_rollup.v1"


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


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def parse_lane(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--lane entries must be name=path")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("Lane name must be non-empty")
    return name, Path(path.strip())


def lane_summary(name: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    shared = summary.get("shared") or {}
    exported = summary.get("shared_exported") or {}
    full = summary.get("shared_full_family") or {}
    missing = summary.get("shared_missing") or {}
    exported_rows = payload.get("exported_rank_gain_rows") or []
    row_count = int_value(shared.get("row_count"))
    exported_count = int_value(exported.get("row_count"))
    exported_rank_gain_count = int_value(exported.get("exported_rank_gain_count"))
    full_exported_count = int_value((full.get("direct_status_counts") or {}).get("direct_certificate_exported"))
    full_rank_gain_count = int_value(full.get("exported_rank_gain_count"))
    return {
        "artifact": str(path),
        "claim_status": payload.get("claim_status"),
        "exported_rank_gain_count": exported_rank_gain_count,
        "exported_rank_gain_rate": ratio(exported_rank_gain_count, exported_count),
        "exported_rank_gain_transfers": summary.get("exported_rank_gain_transfers") or [],
        "exported_unique_gain_total": int_value(exported.get("exported_unique_gain_total")),
        "full_family_exported_rank_gain_count": full_rank_gain_count,
        "full_family_exported_rank_gain_rate": ratio(full_rank_gain_count, full_exported_count),
        "full_family_exported_row_count": full_exported_count,
        "missing_full_family_transfers": summary.get("missing_full_family_transfers") or [],
        "missing_row_count": int_value(missing.get("row_count")),
        "name": name,
        "row_count": row_count,
        "top_exported_rank_gain_rows": exported_rows[:8],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", action="append", required=True, help="Lane in name=path form")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lanes = []
    for raw in args.lane:
        name, path = parse_lane(raw)
        lanes.append(lane_summary(name, path, load_json(path)))
    lanes.sort(
        key=lambda item: (
            -int_value(item.get("full_family_exported_rank_gain_count")),
            -float(item.get("full_family_exported_rank_gain_rate") or 0),
            -int_value(item.get("exported_rank_gain_count")),
            item.get("name") or "",
        )
    )
    best = lanes[0] if lanes else {}
    payload = {
        "claim_status": (
            "PUBLIC_LANES_HAVE_EXPORTED_RANK_GAIN_AND_MISSING_TARGETS"
            if any(item.get("exported_rank_gain_count") for item in lanes)
            and any(item.get("missing_full_family_transfers") for item in lanes)
            else "PUBLIC_LANES_NEED_MORE_DIRECT_EXPORT"
        ),
        "created_at": now_iso(),
        "honesty_boundary": [
            "This rollup compares comparator-audit artifacts only.",
            "It does not create new direct/rank evidence.",
            "Missing targets remain work orders until direct/rank export lands.",
        ],
        "lanes": lanes,
        "schema": SCHEMA,
        "summary": {
            "best_lane_by_full_family_rank_gain": best.get("name"),
            "lane_count": len(lanes),
            "total_exported_rank_gain_count": sum(int_value(item.get("exported_rank_gain_count")) for item in lanes),
            "union_missing_full_family_transfers": sorted(
                {
                    int_value(transfer)
                    for item in lanes
                    for transfer in item.get("missing_full_family_transfers") or []
                }
            ),
        },
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
