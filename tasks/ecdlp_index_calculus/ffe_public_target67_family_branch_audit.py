#!/usr/bin/env python3
"""Audit the frozen target-67 family branch across validation windows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_target67_family_branch_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def leaf_tuple(case: dict[str, Any]) -> tuple[int, ...]:
    leaves = case.get("unique_leaf_indices")
    if leaves is None:
        leaves = case.get("leaf_indices")
    return tuple(sorted({int(leaf) for leaf in leaves or []}))


def is_target67_targetcap1(case: dict[str, Any]) -> bool:
    return (
        str(case.get("target")) == "67.a1@9803"
        and str(case.get("policy")) == "fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0"
    )


def public_case_summary(source: Path) -> dict[str, Any]:
    data = load_json(source)
    cases = [case for case in data.get("positive_cases") or [] if isinstance(case, dict)]
    target_cases = [case for case in cases if is_target67_targetcap1(case)]
    rank_positive = [case for case in target_cases if int(case.get("rank") or 0) > 0]
    relation_positive = [case for case in target_cases if int(case.get("relation_count") or 0) > 0]
    verified = [case for case in target_cases if bool(case.get("public_key_verified"))]
    verified_below = [
        case
        for case in target_cases
        if bool(case.get("public_key_verified")) and bool(case.get("below_rho"))
    ]
    below = [case for case in target_cases if bool(case.get("below_rho"))]
    best_target_case = None
    if target_cases:
        best_target_case = min(
            target_cases,
            key=lambda case: (
                float(case.get("ops_over_rho") or 10**9),
                int(case.get("transfer_index") or 0),
                int(case.get("top_k") or 0),
                leaf_tuple(case),
            ),
        )
    return {
        "source": str(source),
        "case_count": len(cases),
        "target67_targetcap1_candidate_count": len(target_cases),
        "target67_targetcap1_below_rho_count": len(below),
        "target67_targetcap1_rank_positive_count": len(rank_positive),
        "target67_targetcap1_relation_positive_count": len(relation_positive),
        "target67_targetcap1_verified_label_count": len(verified),
        "target67_targetcap1_verified_below_rho_count": len(verified_below),
        "target67_targetcap1_leaf_selector_counts": dict(
            sorted(Counter(str(case.get("leaf_selector") or "") for case in target_cases).items())
        ),
        "target67_targetcap1_transfer_counts": dict(
            sorted(
                Counter(str(case.get("transfer_index") or 0) for case in target_cases).items(),
                key=lambda item: int(item[0]),
            )
        ),
        "best_target67_targetcap1_case": compact_public_case(best_target_case) if best_target_case else None,
    }


def compact_public_case(case: dict[str, Any] | None) -> dict[str, Any] | None:
    if not case:
        return None
    return {
        "target": case.get("target"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "leaf_selector": case.get("leaf_selector"),
        "ops_over_rho": case.get("ops_over_rho"),
        "below_rho": bool(case.get("below_rho")),
        "public_key_verified": bool(case.get("public_key_verified")),
        "rank": int(case.get("rank") or 0),
        "relation_count": int(case.get("relation_count") or 0),
        "selected_row_count": int(case.get("selected_row_count") or 0),
        "selected_leaf_count": int(case.get("selected_leaf_count") or 0),
        "row_salts": [int(salt) for salt in case.get("row_salts") or []],
        "unique_leaf_indices": list(leaf_tuple(case)),
    }


def validation_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    params = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
    cases = [case for case in data.get("cases") or [] if isinstance(case, dict)]
    return {
        "source": str(path),
        "schema": data.get("schema"),
        "family_rule": params.get("family_rule"),
        "select_best_per_transfer": bool(params.get("select_best_per_transfer")),
        "selected_case_count": int(summary.get("selected_case_count") or 0),
        "selected_verified_label_count": int(summary.get("selected_verified_label_count") or 0),
        "selected_verified_below_rho_count": int(summary.get("selected_verified_below_rho_count") or 0),
        "selected_false_positive_count": int(summary.get("selected_false_positive_count") or 0),
        "selected_transfer_counts": summary.get("selected_transfer_counts") or {},
        "selected_leaf_selector_counts": summary.get("selected_leaf_selector_counts") or {},
        "best_verified_below_rho_case": summary.get("best_verified_below_rho_case"),
        "case_keys": [case.get("public_system_key") for case in cases],
    }


def classify_window(public_summary: dict[str, Any], validation: dict[str, Any]) -> str:
    if validation["selected_verified_below_rho_count"] > 0:
        return "selected_verified_below_rho"
    if validation["selected_case_count"] > 0:
        return "selected_false_positive_or_unverified"
    if public_summary["target67_targetcap1_candidate_count"] == 0:
        return "abstained_no_target67_targetcap1_candidates"
    if public_summary["target67_targetcap1_verified_below_rho_count"] == 0:
        return "abstained_target67_targetcap1_candidates_unverified_or_not_below_rho"
    return "abstained_family_rule_missed_verified_target67_targetcap1"


def parse_pair(raw: str) -> tuple[str, Path, Path]:
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("window must be name|public_source|validation_source")
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", action="append", type=parse_pair, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    windows: list[dict[str, Any]] = []
    for name, public_source, validation_source in args.window:
        public_summary = public_case_summary(public_source)
        validation = validation_summary(validation_source)
        windows.append(
            {
                "name": name,
                "public_source": str(public_source),
                "validation_source": str(validation_source),
                "classification": classify_window(public_summary, validation),
                "public_summary": public_summary,
                "validation_summary": validation,
            }
        )

    classification_counts = Counter(window["classification"] for window in windows)
    validation_totals = {
        "selected_case_count": sum(
            int(window["validation_summary"]["selected_case_count"]) for window in windows
        ),
        "selected_verified_below_rho_count": sum(
            int(window["validation_summary"]["selected_verified_below_rho_count"]) for window in windows
        ),
        "selected_false_positive_count": sum(
            int(window["validation_summary"]["selected_false_positive_count"]) for window in windows
        ),
    }
    result = {
        "schema": "ecdlp_public_target67_family_branch_audit_v1",
        "method": "public_window_audit_of_rank_first_target67_family_branch",
        "summary": {
            "window_count": len(windows),
            "classification_counts": dict(sorted(classification_counts.items())),
            **validation_totals,
            "precision_when_selected": (
                validation_totals["selected_verified_below_rho_count"]
                / validation_totals["selected_case_count"]
                if validation_totals["selected_case_count"]
                else None
            ),
        },
        "windows": windows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
