#!/usr/bin/env python3
"""Scout support-diverse source cases in known-column miss windows.

The known-column pressure screen found a relation-support motif gap: some
fresh windows have no structural known-support forms under the current direct
pressure family.  This audit stays one step earlier.  It reads selected-leaf
support scout artifacts for those miss windows and asks whether adjacent public
selectors/top-k choices expose different factor-column support before relation
verification.

This is a manifest/scout, not a replay proof.  A selected support containing a
target motif only means the source case is worth replaying with the direct
certificate or known-factor screen; accepted relation forms and target descent
remain separate gates.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
DEFAULT_OUT = Path(
    "ecdlp_index_calculus_state/"
    "low_term_total2_source_support_diversity_scout_22050_col15_no_structural_3064_3215_probe.json"
)
DEFAULT_WINDOWS = "3064_3071,3080_3087,3136_3143,3168_3175,3200_3207,3208_3215"
DEFAULT_DEGENERATE_SUPPORT = "0,4,5,6,7,10,11,14,15"
DEFAULT_KNOWN_COLUMNS = "7,8,9,10,11,12,13,14"
DEFAULT_PRIORITY_COLUMNS = "15"
DEFAULT_MOTIFS = "8:11,9:11,10:13,10:14,11:13"
DEFAULT_AVOID_MOTIFS = "2:4,3:5"

WINDOW_RE = re.compile(r"_(\d{4})_(\d{4})(?:_|_col15_|_probe)")


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


def parse_int_set(raw: str) -> set[int]:
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def parse_support(raw: str) -> tuple[int, ...]:
    return tuple(sorted(parse_int_set(raw)))


def parse_motifs(raw: str) -> set[tuple[int, ...]]:
    motifs: set[tuple[int, ...]] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        motifs.add(tuple(sorted(int(item.strip()) for item in chunk.split(":") if item.strip())))
    return motifs


def parse_windows(raw: str) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        start, end = chunk.replace("-", "_").split("_", 1)
        windows.append((int(start), int(end)))
    return windows


def window_label(window: tuple[int, int]) -> str:
    return f"{window[0]}_{window[1]}"


def window_from_path(path: Path) -> tuple[int, int] | None:
    match = WINDOW_RE.search(path.name)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def split_paths(raw: str) -> list[Path]:
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def default_support_scout_path(state_dir: Path, window: tuple[int, int]) -> Path:
    label = window_label(window)
    return (
        state_dir
        / f"low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_{label}_probe.json"
    )


def default_pressure_path(state_dir: Path, window: tuple[int, int]) -> Path:
    label = window_label(window)
    return state_dir / f"low_term_total2_known_column_pressure_direct_screen_22050_col15_{label}_probe.json"


def source_path_from_support_artifact(path: Path, payload: dict[str, Any]) -> Path | None:
    raw = ((payload.get("artifacts") or {}).get("source"))
    if not raw:
        return None
    source = Path(str(raw))
    if source.is_absolute():
        return source
    root = path.parent.parent
    return root / source


def motif_hits(support: set[int], motifs: set[tuple[int, ...]]) -> list[list[int]]:
    return [list(motif) for motif in sorted(motifs) if set(motif).issubset(support)]


def support_signature(values: Any) -> tuple[int, ...]:
    return tuple(sorted(int_value(value) for value in values or []))


def candidate_score(candidate: dict[str, Any]) -> tuple[int, int, int, float, int, int, int, str]:
    direct_ops = float_value(candidate.get("direct_ops_over_rho"))
    shared_ops = float_value(candidate.get("shared_product_ops_over_rho"))
    best_ops = min([value for value in [direct_ops, shared_ops] if value is not None] or [999.0])
    return (
        int(candidate.get("rank_score") or 0),
        int(candidate.get("direct_public_key_verified")),
        int(candidate.get("shared_product_public_key_verified")),
        -best_ops,
        -int(candidate.get("selected_support_size") or 0),
        -int(candidate.get("avoid_motif_count") or 0),
        int(candidate.get("transfer_index") or 0),
        str(candidate.get("selector")),
    )


def compact_pressure(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    summary = payload.get("summary") or {}
    return {
        "artifact": str(path),
        "claim_status": payload.get("claim_status"),
        "public_known_factor_verified_count": int_value(summary.get("public_known_factor_verified_count")),
        "scanned_case_count": int_value(summary.get("scanned_case_count")),
        "selected_report_count": int_value(summary.get("selected_report_count")),
        "structural_known_support_case_count": int_value(summary.get("structural_known_support_case_count")),
        "support_report_count": int_value(summary.get("support_report_count")),
    }


def candidate_from_report(
    report: dict[str, Any],
    path: Path,
    source_path: Path | None,
    degenerate_support: tuple[int, ...],
    known_columns: set[int],
    priority_columns: set[int],
    target_motifs: set[tuple[int, ...]],
    avoid_motifs: set[tuple[int, ...]],
) -> dict[str, Any]:
    support_tuple = support_signature(report.get("selected_term_support"))
    support = set(support_tuple)
    hits = motif_hits(support, target_motifs)
    avoid_hits = motif_hits(support, avoid_motifs)
    priority_hits = sorted(support & priority_columns)
    known_hits = sorted(support & known_columns)
    unknown_hits = sorted(support - known_columns - priority_columns)
    direct_ops = float_value(report.get("direct_ops_over_rho"))
    shared_ops = float_value(report.get("shared_product_ops_over_rho"))
    direct_verified = bool(report.get("direct_public_key_verified"))
    shared_verified = bool(report.get("shared_product_public_key_verified"))
    public_gate_selected = bool(report.get("public_product_gate_selected"))
    below_rho = any(value is not None and value < 1.0 for value in [direct_ops, shared_ops])
    outside_degenerate = support_tuple != degenerate_support
    rank_score = (
        6 * int(outside_degenerate)
        + 4 * len(hits)
        + 3 * int(bool(priority_hits))
        + 2 * int(direct_verified)
        + 2 * int(shared_verified)
        + 2 * int(public_gate_selected)
        + 1 * int(below_rho)
        + len(known_hits)
        - 3 * len(avoid_hits)
        - max(0, len(unknown_hits) - 3)
    )
    return {
        "artifact": str(path),
        "avoid_motif_count": len(avoid_hits),
        "avoid_motif_hits": avoid_hits,
        "candidate_case_arg": (
            f"{report.get('target')}|{int_value(report.get('transfer_index'))}|"
            f"{report.get('selector')}|{int_value(report.get('top_k'))}"
        ),
        "direct_ops_over_rho": direct_ops,
        "direct_public_key_verified": direct_verified,
        "known_column_hits": known_hits,
        "motif_hit_count": len(hits),
        "motif_hits": hits,
        "outside_degenerate_support": outside_degenerate,
        "priority_hits": priority_hits,
        "public_product_gate_selected": public_gate_selected,
        "rank_score": rank_score,
        "row_keys": [str(row) for row in report.get("row_keys") or []],
        "selected_support_size": len(support_tuple),
        "selected_term_support": list(support_tuple),
        "selector": report.get("selector"),
        "shared_product_ops_over_rho": shared_ops,
        "shared_product_public_key_verified": shared_verified,
        "source_path": str(source_path) if source_path else None,
        "target": report.get("target"),
        "top_k": int_value(report.get("top_k")),
        "transfer_index": int_value(report.get("transfer_index")),
        "unknown_column_hits": unknown_hits,
    }


def scan_support_artifact(
    path: Path,
    degenerate_support: tuple[int, ...],
    known_columns: set[int],
    priority_columns: set[int],
    target_motifs: set[tuple[int, ...]],
    avoid_motifs: set[tuple[int, ...]],
) -> dict[str, Any]:
    payload = load_json(path)
    source_path = source_path_from_support_artifact(path, payload)
    candidates = [
        candidate_from_report(
            report,
            path,
            source_path,
            degenerate_support,
            known_columns,
            priority_columns,
            target_motifs,
            avoid_motifs,
        )
        for report in payload.get("case_reports") or []
        if isinstance(report, dict)
    ]
    candidates.sort(key=candidate_score, reverse=True)
    support_counter = Counter(
        ",".join(str(value) for value in candidate["selected_term_support"])
        for candidate in candidates
    )
    window = window_from_path(path)
    return {
        "artifact": str(path),
        "candidates": candidates,
        "source_path": str(source_path) if source_path else None,
        "support_summary": {
            "candidate_count": len(candidates),
            "direct_verified_motif_candidate_count": sum(
                1 for row in candidates if row["motif_hits"] and row["direct_public_key_verified"]
            ),
            "motif_candidate_count": sum(1 for row in candidates if row["motif_hits"]),
            "outside_degenerate_count": sum(1 for row in candidates if row["outside_degenerate_support"]),
            "priority_candidate_count": sum(1 for row in candidates if row["priority_hits"]),
            "shared_verified_motif_candidate_count": sum(
                1 for row in candidates if row["motif_hits"] and row["shared_product_public_key_verified"]
            ),
            "unique_support_count": len(support_counter),
            "top_supports": [
                {"count": count, "support": support}
                for support, count in support_counter.most_common(10)
            ],
        },
        "top_candidates": candidates[:8],
        "window": window_label(window) if window else None,
    }


def aggregate_manifest(per_window: list[dict[str, Any]], max_cases_per_source: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in per_window:
        for candidate in window.get("top_candidates") or []:
            source_path = candidate.get("source_path")
            if source_path and (
                candidate.get("motif_hits")
                or candidate.get("priority_hits")
                or candidate.get("outside_degenerate_support")
            ):
                grouped[str(source_path)].append(candidate)
    manifest = []
    for source_path, candidates in sorted(grouped.items()):
        candidates.sort(key=candidate_score, reverse=True)
        selected = candidates[:max_cases_per_source]
        manifest.append(
            {
                "case_args": ";".join(str(row["candidate_case_arg"]) for row in selected),
                "cases": selected,
                "direct_certificate_command": (
                    "python3 tasks/ecdlp_index_calculus/"
                    "low_term_total2_direct_source_relation_equation_certificate.py "
                    f"--source {source_path} --cases '"
                    + ";".join(str(row["candidate_case_arg"]) for row in selected)
                    + "' --out <local_or_live_output.json>"
                ),
                "source": source_path,
            }
        )
    return manifest


def stat(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "max": None, "mean": None, "min": None, "sum": 0.0}
    return {
        "count": len(values),
        "max": round(max(values), 8),
        "mean": round(mean(values), 8),
        "min": round(min(values), 8),
        "sum": round(sum(values), 8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--support-scouts", default="")
    parser.add_argument("--pressure-screens", default="")
    parser.add_argument("--degenerate-support", default=DEFAULT_DEGENERATE_SUPPORT)
    parser.add_argument("--known-columns", default=DEFAULT_KNOWN_COLUMNS)
    parser.add_argument("--priority-columns", default=DEFAULT_PRIORITY_COLUMNS)
    parser.add_argument("--target-motifs", default=DEFAULT_MOTIFS)
    parser.add_argument("--avoid-motifs", default=DEFAULT_AVOID_MOTIFS)
    parser.add_argument("--max-cases-per-source", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    windows = parse_windows(args.windows)
    support_paths = (
        split_paths(args.support_scouts)
        if args.support_scouts
        else [default_support_scout_path(args.state_dir, window) for window in windows]
    )
    pressure_paths = (
        split_paths(args.pressure_screens)
        if args.pressure_screens
        else [default_pressure_path(args.state_dir, window) for window in windows]
    )
    degenerate_support = parse_support(args.degenerate_support)
    known_columns = parse_int_set(args.known_columns)
    priority_columns = parse_int_set(args.priority_columns)
    target_motifs = parse_motifs(args.target_motifs)
    avoid_motifs = parse_motifs(args.avoid_motifs)

    missing_support = [str(path) for path in support_paths if not path.exists()]
    if missing_support:
        raise SystemExit(f"missing support scout artifacts: {missing_support}")

    pressure = [compact_pressure(path) for path in pressure_paths if path.exists()]
    per_window = [
        scan_support_artifact(
            path,
            degenerate_support,
            known_columns,
            priority_columns,
            target_motifs,
            avoid_motifs,
        )
        for path in support_paths
    ]
    all_candidates = [candidate for window in per_window for candidate in window.get("candidates") or []]
    top_all = sorted(all_candidates, key=candidate_score, reverse=True)[:20]
    direct_ops = [
        value
        for value in (float_value(row.get("direct_ops_over_rho")) for row in all_candidates)
        if value is not None
    ]
    shared_ops = [
        value
        for value in (float_value(row.get("shared_product_ops_over_rho")) for row in all_candidates)
        if value is not None
    ]
    manifest = aggregate_manifest(per_window, args.max_cases_per_source)
    summary = {
        "avoid_motif_candidate_count": sum(1 for row in all_candidates if row["avoid_motif_hits"]),
        "candidate_manifest_source_count": len(manifest),
        "direct_ops_over_rho": stat(direct_ops),
        "direct_verified_motif_candidate_count": sum(
            1 for row in all_candidates if row["motif_hits"] and row["direct_public_key_verified"]
        ),
        "motif_candidate_count": sum(1 for row in all_candidates if row["motif_hits"]),
        "outside_degenerate_case_count": sum(
            1 for row in all_candidates if row["outside_degenerate_support"]
        ),
        "pressure_no_structural_window_count": sum(
            1
            for row in pressure
            if row["selected_report_count"] == 0
            or row["structural_known_support_case_count"] == 0
        ),
        "priority_candidate_count": sum(1 for row in all_candidates if row["priority_hits"]),
        "shared_ops_over_rho": stat(shared_ops),
        "shared_verified_motif_candidate_count": sum(
            1 for row in all_candidates if row["motif_hits"] and row["shared_product_public_key_verified"]
        ),
        "support_case_count": len(all_candidates),
        "support_scout_window_count": len(per_window),
        "window_count": len(windows),
    }
    summary["claim_status"] = (
        "SOURCE_SUPPORT_DIVERSITY_REPLAY_CANDIDATES_FOUND"
        if summary["motif_candidate_count"] or summary["outside_degenerate_case_count"]
        else "NO_SOURCE_SUPPORT_DIVERSITY_FOUND"
    )
    payload = {
        "created_at": now_iso(),
        "direct_certificate_manifest": manifest,
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "MODEL-BOUND: selected-leaf support is pre-relation evidence and not a target-descent proof.",
            "A motif hit here means selected source leaves expose the requested columns; accepted relation-form support must still be replayed.",
            "This artifact reads mounted AutoLab state and writes a local scout; it does not mutate the live campaign state.",
        ],
        "parameters": {
            "avoid_motifs": [list(motif) for motif in sorted(avoid_motifs)],
            "degenerate_support": list(degenerate_support),
            "known_columns": sorted(known_columns),
            "priority_columns": sorted(priority_columns),
            "state_dir": str(args.state_dir),
            "target_motifs": [list(motif) for motif in sorted(target_motifs)],
            "windows": [window_label(window) for window in windows],
        },
        "per_window": per_window,
        "pressure_screens": pressure,
        "schema": "ecdlp.low_term_total2_source_support_diversity_scout.v1",
        "summary": summary,
        "top_candidates": top_all,
    }
    write_json(args.out, payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
