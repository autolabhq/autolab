#!/usr/bin/env python3
"""Summarize frozen target-67 branch activation windows.

The branch-aware repeated-coordinate package now has two distinct gates:
public activation of a target-67 repeated-coordinate candidate, then the
two-row relation replay.  This probe keeps those stages separate so dry windows
are recorded as activation abstentions instead of relation failures.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_branch_selector_miner as branch_miner


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_branch_activation_scan.json"
DEFAULT_CANDIDATE_CLAUSE = "b_mod4=3&salt_mod2_pattern=0,1,0"

MANIFEST_RE = re.compile(
    r"ffe_public_repeated_coordinate_frozen_pipeline_target67_"
    r"(?P<start>\d+)_(?P<end>\d+)_(?P<label>fresh|control)_manifest\.json$"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    return json.loads(resolved.read_text()) if resolved.exists() else {}


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def feature_record(values: dict[str, Any]) -> dict[str, list[str]]:
    features: dict[str, list[str]] = {}
    for key, value in values.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float, str)):
            features.setdefault(key, []).append(str(value))
    return features


def candidate_matches(candidate: dict[str, Any], clauses: list[tuple[str, ...]]) -> bool:
    if not clauses:
        return True
    values = branch_miner.public_feature_values(candidate, "coordinate")
    record = {"features": feature_record(values)}
    return any(branch_miner.clause_matches(clause, record) for clause in clauses)


def parse_window(raw: str) -> tuple[int, int]:
    cleaned = raw.strip().replace("_", "-")
    parts = cleaned.split("-", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("window must be START-END")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window bounds must be integers") from exc
    if end < start:
        raise argparse.ArgumentTypeError("window end must be >= start")
    return start, end


def parse_window_range(raw: str) -> list[tuple[int, int]]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("window range must be START:END:WIDTH")
    try:
        start = int(parts[0])
        end = int(parts[1])
        width = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window range values must be integers") from exc
    if width <= 0 or end < start:
        raise argparse.ArgumentTypeError("range requires END >= START and WIDTH > 0")
    windows = []
    current = start
    while current <= end:
        windows.append((current, min(current + width - 1, end)))
        current += width
    return windows


def discover_windows(state_dir: Path) -> list[tuple[int, int]]:
    windows: set[tuple[int, int]] = set()
    for path in resolve_path(state_dir).glob(
        "ffe_public_repeated_coordinate_frozen_pipeline_target67_*_fresh_manifest.json"
    ):
        match = MANIFEST_RE.search(path.name)
        if match:
            windows.add((int(match.group("start")), int(match.group("end"))))
    return sorted(windows)


def manifest_path(state_dir: Path, start: int, end: int) -> Path:
    return (
        state_dir
        / f"ffe_public_repeated_coordinate_frozen_pipeline_target67_{start}_{end}_fresh_manifest.json"
    )


def branch_replay_path(state_dir: Path, start: int, end: int, suffix: str) -> Path:
    return (
        state_dir
        / f"ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_{suffix}_{start}_{end}.json"
    )


def load_coordinate_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = (((manifest.get("artifacts") or {}).get("coordinate_gate")) or "")
    if not path:
        return []
    data = load_json(Path(path))
    return [candidate for candidate in data.get("candidates") or [] if isinstance(candidate, dict)]


def summarize_branch_replay(path: Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return {"path": str(path), "exists": False}
    data = load_json(path)
    summary = data.get("summary") or {}
    candidate_filter = data.get("candidate_filter_summary") or {}
    cases = [case for case in data.get("cases") or [] if isinstance(case, dict)]
    verified_ops = [
        float((case.get("guarded_replay") or {}).get("ops_over_rho"))
        for case in cases
        if (case.get("guarded_replay") or {}).get("guarded_public_key_verified")
        and (case.get("guarded_replay") or {}).get("ops_over_rho") is not None
    ]
    return {
        "path": str(path),
        "exists": True,
        "input_candidate_count": int(candidate_filter.get("input_candidate_count") or 0),
        "matched_candidate_count": int(candidate_filter.get("matched_candidate_count") or 0),
        "input_pair_case_count": int(summary.get("input_pair_case_count") or 0),
        "relation_pair_case_count": int(summary.get("relation_pair_case_count") or 0),
        "guarded_verified_pair_case_count": int(summary.get("guarded_verified_pair_case_count") or 0),
        "guarded_below_rho_pair_case_count": int(summary.get("guarded_below_rho_pair_case_count") or 0),
        "min_guard_passed_ops_over_rho": summary.get("min_guard_passed_ops_over_rho"),
        "mean_guard_passed_ops_over_rho": summary.get("mean_guard_passed_ops_over_rho"),
        "mean_verified_ops_over_rho": mean_or_none(verified_ops),
    }


def summarize_window(
    state_dir: Path,
    start: int,
    end: int,
    clauses: list[tuple[str, ...]],
    branch_suffix: str,
) -> dict[str, Any]:
    mpath = manifest_path(state_dir, start, end)
    manifest = load_json(mpath)
    exists = bool(manifest)
    selector = manifest.get("selector_summary") or {}
    coordinate = manifest.get("coordinate_gate_summary") or {}
    guard = manifest.get("guard_replay_summary") or {}
    candidates = load_coordinate_candidates(manifest) if exists else []
    matched_candidates = [candidate for candidate in candidates if candidate_matches(candidate, clauses)]
    branch = summarize_branch_replay(branch_replay_path(state_dir, start, end, branch_suffix))
    return {
        "window": f"{start}-{end}",
        "window_name": f"w{start}_{end}_fresh",
        "manifest_path": str(mpath),
        "manifest_exists": exists,
        "stress_source": (manifest.get("parameters") or {}).get("stress_source"),
        "selected_public_case_count": int(selector.get("selected_public_case_count") or 0),
        "all_public_bounded_case_count": int(selector.get("all_public_bounded_case_count") or 0),
        "target_count": int(selector.get("target_count") or 0),
        "coordinate_candidate_count": int(coordinate.get("candidate_count") or 0),
        "coordinate_verified_count": int(coordinate.get("replayed_public_key_verified_count") or 0),
        "coordinate_verified_below_rho_count": int(coordinate.get("replayed_verified_below_rho_count") or 0),
        "coordinate_min_ops_over_rho": coordinate.get("min_replayed_ops_over_rho"),
        "branch_clause_match_count": len(matched_candidates),
        "guard_selected_case_count": int(guard.get("selected_case_count") or 0),
        "guard_relation_case_count": int(guard.get("relation_case_count") or 0),
        "guard_verified_count": int(guard.get("guarded_verified_case_count") or 0),
        "guard_below_rho_count": int(guard.get("guarded_below_rho_count") or 0),
        "branch_replay": branch,
    }


def summarize_scan(windows: list[dict[str, Any]]) -> dict[str, Any]:
    branch = [window.get("branch_replay") or {} for window in windows]
    verified_ops = [
        float(item["min_guard_passed_ops_over_rho"])
        for item in branch
        if item.get("min_guard_passed_ops_over_rho") is not None
    ]
    activation_windows = [
        window["window"]
        for window in windows
        if int(window.get("coordinate_candidate_count") or 0) > 0
        or int(window.get("branch_clause_match_count") or 0) > 0
    ]
    branch_relation_windows = [
        window["window"]
        for window in windows
        if int((window.get("branch_replay") or {}).get("relation_pair_case_count") or 0) > 0
    ]
    branch_below_windows = [
        window["window"]
        for window in windows
        if int((window.get("branch_replay") or {}).get("guarded_below_rho_pair_case_count") or 0) > 0
    ]
    branch_replay_missing_windows = [
        window["window"]
        for window in windows
        if not (window.get("branch_replay") or {}).get("exists")
    ]
    branch_replay_existing_windows = [
        window["window"]
        for window in windows
        if (window.get("branch_replay") or {}).get("exists")
    ]
    return {
        "window_count": len(windows),
        "manifest_count": sum(1 for window in windows if window.get("manifest_exists")),
        "branch_replay_window_count": len(branch_replay_existing_windows),
        "selected_public_window_count": sum(
            1 for window in windows if int(window.get("selected_public_case_count") or 0) > 0
        ),
        "coordinate_candidate_window_count": sum(
            1 for window in windows if int(window.get("coordinate_candidate_count") or 0) > 0
        ),
        "branch_clause_window_count": sum(
            1 for window in windows if int(window.get("branch_clause_match_count") or 0) > 0
        ),
        "branch_relation_window_count": len(branch_relation_windows),
        "branch_below_rho_window_count": len(branch_below_windows),
        "total_selected_public_cases": sum(int(window.get("selected_public_case_count") or 0) for window in windows),
        "total_coordinate_candidates": sum(int(window.get("coordinate_candidate_count") or 0) for window in windows),
        "total_branch_clause_matches": sum(int(window.get("branch_clause_match_count") or 0) for window in windows),
        "total_branch_pair_cases": sum(int(item.get("input_pair_case_count") or 0) for item in branch),
        "total_branch_relation_pair_cases": sum(int(item.get("relation_pair_case_count") or 0) for item in branch),
        "total_branch_verified_pair_cases": sum(int(item.get("guarded_verified_pair_case_count") or 0) for item in branch),
        "total_branch_below_rho_pair_cases": sum(int(item.get("guarded_below_rho_pair_case_count") or 0) for item in branch),
        "min_branch_ops_over_rho": min(verified_ops) if verified_ops else None,
        "activation_windows": activation_windows,
        "branch_relation_windows": branch_relation_windows,
        "branch_below_rho_windows": branch_below_windows,
        "branch_replay_existing_windows": branch_replay_existing_windows,
        "branch_replay_missing_windows": branch_replay_missing_windows,
        "abstention_windows": [
            window["window"]
            for window in windows
            if int(window.get("selected_public_case_count") or 0) == 0
            and int(window.get("coordinate_candidate_count") or 0) == 0
        ],
        "branch_replay_status_counts": dict(
            sorted(Counter("exists" if item.get("exists") else "missing" for item in branch).items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--window", action="append", type=parse_window, default=[])
    parser.add_argument("--window-range", action="append", type=parse_window_range, default=[])
    parser.add_argument("--candidate-clause", action="append", default=[DEFAULT_CANDIDATE_CLAUSE])
    parser.add_argument("--branch-suffix", default="subset2")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    windows: list[tuple[int, int]] = []
    windows.extend(args.window)
    for group in args.window_range:
        windows.extend(group)
    if not windows:
        windows = discover_windows(args.state_dir)
    windows = sorted(set(windows))

    clauses = [branch_miner.parse_clause(raw) for raw in args.candidate_clause]
    records = [
        summarize_window(args.state_dir, start, end, clauses, str(args.branch_suffix))
        for start, end in windows
    ]
    output = {
        "schema": "ecdlp_public_repeated_coordinate_branch_activation_scan_v1",
        "parameters": {
            "state_dir": str(args.state_dir),
            "windows": [f"{start}-{end}" for start, end in windows],
            "candidate_clauses": [branch_miner.clause_text(clause) for clause in clauses],
            "branch_suffix": str(args.branch_suffix),
        },
        "summary": summarize_scan(records),
        "windows": records,
    }
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
