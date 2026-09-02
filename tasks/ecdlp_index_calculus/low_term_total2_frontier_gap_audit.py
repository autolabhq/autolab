#!/usr/bin/env python3
"""Report direct/rank/scout range coverage for low-term total2 artifacts.

The mounted AutoLab stream can expose newer direct/rank/scout batches while an
intermediate batch is still missing.  This audit records the contiguous frontier
and any noncontiguous tail ranges before downstream bridge/selector artifacts
consume the files.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_frontier_gap_audit.v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_paths(raw: str) -> list[Path]:
    paths: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        matches = sorted(Path(path) for path in glob.glob(item))
        paths.extend(matches or [Path(item)])
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def range_from_name(path: Path) -> tuple[int, int] | None:
    pairs = re.findall(r"(?<!\d)(\d{4})_(\d{4})(?!\d)", path.name)
    if not pairs:
        return None
    start, end = pairs[-1]
    return int(start), int(end)


def range_label(item: tuple[int, int]) -> str:
    return f"{item[0]}_{item[1]}"


def collect_ranges(paths: list[Path], start_min: int, end_max: int | None) -> list[tuple[int, int]]:
    ranges = []
    for path in paths:
        item = range_from_name(path)
        if item is None:
            continue
        if item[1] < start_min:
            continue
        if end_max is not None and item[0] > end_max:
            continue
        ranges.append(item)
    return sorted(set(ranges))


def expected_ranges(start_min: int, end_max: int, width: int) -> list[tuple[int, int]]:
    ranges = []
    start = start_min
    while start <= end_max:
        end = min(start + width - 1, end_max)
        ranges.append((start, end))
        start += width
    return ranges


def contiguous_end(expected: list[tuple[int, int]], available: set[tuple[int, int]]) -> int | None:
    last = None
    for item in expected:
        if item not in available:
            break
        last = item[1]
    return last


def summarize_group(name: str, ranges: list[tuple[int, int]], expected: list[tuple[int, int]]) -> dict[str, Any]:
    available = set(ranges)
    missing = [item for item in expected if item not in available]
    return {
        "available_count": len(ranges),
        "available_ranges": [range_label(item) for item in ranges],
        "contiguous_end": contiguous_end(expected, available),
        "latest_end": ranges[-1][1] if ranges else None,
        "latest_range": range_label(ranges[-1]) if ranges else None,
        "missing_count": len(missing),
        "missing_ranges": [range_label(item) for item in missing],
        "name": name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", required=True)
    parser.add_argument("--rank-scorers", required=True)
    parser.add_argument("--support-scouts", required=True)
    parser.add_argument("--start-min", required=True, type=int)
    parser.add_argument("--end-max", required=True, type=int)
    parser.add_argument("--range-width", type=int, default=8)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    expected = expected_ranges(args.start_min, args.end_max, args.range_width)
    groups = {
        "certificates": collect_ranges(parse_paths(args.certificates), args.start_min, args.end_max),
        "rank_scorers": collect_ranges(parse_paths(args.rank_scorers), args.start_min, args.end_max),
        "support_scouts": collect_ranges(parse_paths(args.support_scouts), args.start_min, args.end_max),
    }
    common = set(expected)
    for ranges in groups.values():
        common &= set(ranges)
    union = set().union(*(set(ranges) for ranges in groups.values()))
    missing_any = [item for item in expected if item not in common]
    missing_all = [item for item in expected if item not in union]
    contiguous_common_end = contiguous_end(expected, common)
    tail_after_gap = [
        item
        for item in sorted(common)
        if contiguous_common_end is not None and item[0] > contiguous_common_end
    ]
    payload = {
        "claim_status": (
            "FRONTIER_HAS_NONCONTIGUOUS_COMPLETE_TAIL"
            if tail_after_gap and missing_any
            else "FRONTIER_HAS_GAPS_NO_COMPLETE_TAIL"
            if missing_any
            else "FRONTIER_CONTIGUOUS"
        ),
        "created_at": now_iso(),
        "expected_ranges": [range_label(item) for item in expected],
        "group_summaries": {
            name: summarize_group(name, ranges, expected)
            for name, ranges in groups.items()
        },
        "parameters": {
            "end_max": args.end_max,
            "range_width": args.range_width,
            "start_min": args.start_min,
        },
        "schema": SCHEMA,
        "summary": {
            "common_complete_ranges": [range_label(item) for item in sorted(common)],
            "contiguous_common_end": contiguous_common_end,
            "latest_common_complete_end": max((item[1] for item in common), default=None),
            "missing_any_count": len(missing_any),
            "missing_any_ranges": [range_label(item) for item in missing_any],
            "missing_all_count": len(missing_all),
            "missing_all_ranges": [range_label(item) for item in missing_all],
            "noncontiguous_complete_tail_ranges": [range_label(item) for item in tail_after_gap],
        },
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
