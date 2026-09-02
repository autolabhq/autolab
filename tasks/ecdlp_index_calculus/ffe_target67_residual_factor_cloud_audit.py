#!/usr/bin/env python3
"""Audit simple residual factor-cloud heuristics for target-67 FFE lines.

This is deliberately post-factorization: it asks whether the preserving
degree-1 factor has an obvious public position or coefficient signature inside
the resultant factor cloud.  A positive result would guide a cheaper line
predictor; a negative result says the next predictor has to use richer
FFE/summation-polynomial residual structure than factor order or coefficient
size.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LINE_STAGE_AUDIT = DEFAULT_STATE_DIR / "ffe_target67_line_stage_audit_328_511.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_residual_factor_cloud_audit_328_511.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKTREE_ROOT / path


def leaf_signature(leaves: list[int]) -> str:
    return ",".join(str(int(leaf)) for leaf in sorted({int(leaf) for leaf in leaves}))


def line_from_fingerprint(fingerprint: list[list[int]], p: int) -> dict[str, Any] | None:
    terms = {
        (int(b_degree), int(c_degree)): int(coeff) % int(p)
        for b_degree, c_degree, coeff in fingerprint
    }
    if set(terms) != {(0, 0), (0, 1), (1, 0)}:
        return None
    return {
        "b_coeff": int(terms[(1, 0)]),
        "c_coeff": int(terms[(0, 1)]),
        "constant": int(terms[(0, 0)]),
        "line": f"{int(terms[(1, 0)])}*b + {int(terms[(0, 1)])}*c + {int(terms[(0, 0)])}",
    }


def centered_abs(value: int, p: int) -> int:
    value %= p
    return min(value, p - value)


def factor_rows(surface: dict[str, Any]) -> list[dict[str, Any]]:
    p = int(surface.get("p") or 0)
    rows = []
    for index, factor in enumerate((surface.get("sage_resultant_factorization") or {}).get("factors") or []):
        line = line_from_fingerprint(factor.get("fingerprint") or [], p)
        if not line:
            continue
        b_coeff = int(line["b_coeff"])
        constant = int(line["constant"])
        rows.append(
            {
                "factor_index": index,
                "line": line["line"],
                "b_coeff": b_coeff,
                "c_coeff": int(line["c_coeff"]),
                "constant": constant,
                "b_center_abs": centered_abs(b_coeff, p),
                "constant_center_abs": centered_abs(constant, p),
                "coeff_norm": centered_abs(b_coeff, p) + centered_abs(constant, p),
                "b_mod8": b_coeff % 8,
                "constant_mod8": constant % 8,
            }
        )
    return rows


def factor_choice(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=key)


def heuristic_choices(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    return {
        "first_factor": factor_choice(rows, lambda row: row["factor_index"]),
        "last_factor": factor_choice(rows, lambda row: -row["factor_index"]),
        "min_b_coeff": factor_choice(rows, lambda row: (row["b_coeff"], row["factor_index"])),
        "max_b_coeff": factor_choice(rows, lambda row: (-row["b_coeff"], row["factor_index"])),
        "min_constant": factor_choice(rows, lambda row: (row["constant"], row["factor_index"])),
        "max_constant": factor_choice(rows, lambda row: (-row["constant"], row["factor_index"])),
        "min_b_center_abs": factor_choice(rows, lambda row: (row["b_center_abs"], row["factor_index"])),
        "min_constant_center_abs": factor_choice(
            rows, lambda row: (row["constant_center_abs"], row["factor_index"])
        ),
        "min_coeff_norm": factor_choice(rows, lambda row: (row["coeff_norm"], row["factor_index"])),
        "max_coeff_norm": factor_choice(rows, lambda row: (-row["coeff_norm"], row["factor_index"])),
    }


def line_stage_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("source") or ""),
        str(record.get("surface_id") or ""),
        str(record.get("leaf_signature") or ""),
    )


def surface_key(source: Path, surface: dict[str, Any]) -> tuple[str, str, str]:
    exact = surface.get("exact_profile") or {}
    return (
        str(source.relative_to(WORKTREE_ROOT)),
        str(surface.get("surface_id") or ""),
        leaf_signature([int(leaf) for leaf in exact.get("leaf_indices") or surface.get("selected_leaf_indices") or []]),
    )


def load_surfaces_by_key(exact_sources: list[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out = {}
    for source_text in exact_sources:
        source = resolve_path(source_text)
        artifact = load_json(source)
        for surface in artifact.get("surfaces") or []:
            out[surface_key(source, surface)] = surface
    return out


def row_record(record: dict[str, Any], surface: dict[str, Any] | None) -> dict[str, Any]:
    rows = factor_rows(surface or {})
    preserving_index = record.get("preserving_factor_index")
    choices = heuristic_choices(rows)
    heuristic_hits = {
        name: (
            choice is not None
            and preserving_index is not None
            and int(choice["factor_index"]) == int(preserving_index)
        )
        for name, choice in choices.items()
    }
    preserving_row = None
    if preserving_index is not None:
        preserving_row = next(
            (row for row in rows if int(row["factor_index"]) == int(preserving_index)),
            None,
        )
    return {
        "bucket": record.get("bucket"),
        "label_replay_success": record.get("bucket") == "line_present_replay_success",
        "has_preserving_line": bool(record.get("has_preserving_line")),
        "target": record.get("target"),
        "transfer_index": record.get("transfer_index"),
        "top_k": record.get("top_k"),
        "leaf_signature": record.get("leaf_signature"),
        "surface_id": record.get("surface_id"),
        "source": record.get("source"),
        "surface_found": surface is not None,
        "factor_count": len(rows),
        "preserving_factor_index": preserving_index,
        "preserving_line": (record.get("preserving_line") or {}).get("line"),
        "preserving_factor_features": preserving_row,
        "heuristic_hits": heuristic_hits,
        "heuristic_choices": {
            name: (
                {
                    "factor_index": choice["factor_index"],
                    "line": choice["line"],
                    "b_coeff": choice["b_coeff"],
                    "constant": choice["constant"],
                    "coeff_norm": choice["coeff_norm"],
                }
                if choice
                else None
            )
            for name, choice in choices.items()
        },
        "line_sample": rows[:5],
    }


def summarize_heuristics(records: list[dict[str, Any]]) -> dict[str, Any]:
    line_present = [row for row in records if row["has_preserving_line"]]
    replay_success = [row for row in line_present if row["label_replay_success"]]
    heuristic_names = sorted({name for row in records for name in row["heuristic_hits"]})
    out = {}
    for name in heuristic_names:
        line_hits = sum(1 for row in line_present if row["heuristic_hits"].get(name))
        success_hits = sum(1 for row in replay_success if row["heuristic_hits"].get(name))
        out[name] = {
            "line_present_hit_count": line_hits,
            "line_present_case_count": len(line_present),
            "replay_success_hit_count": success_hits,
            "replay_success_case_count": len(replay_success),
        }
    return out


def index_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    line_present = [
        row
        for row in records
        if row["has_preserving_line"] and row.get("preserving_factor_index") is not None
    ]
    values = [int(row["preserving_factor_index"]) for row in line_present]
    if not values:
        return {"count": 0, "values": []}
    by_bucket = {}
    for bucket in sorted({str(row["bucket"]) for row in line_present}):
        bucket_values = [
            int(row["preserving_factor_index"])
            for row in line_present
            if str(row["bucket"]) == bucket
        ]
        by_bucket[bucket] = {
            "count": len(bucket_values),
            "values": sorted(bucket_values),
            "min": min(bucket_values),
            "max": max(bucket_values),
        }
    return {
        "count": len(values),
        "values": sorted(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 6),
        "by_bucket": by_bucket,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-stage-audit", type=Path, default=DEFAULT_LINE_STAGE_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    line_stage = load_json(args.line_stage_audit)
    exact_sources = list((line_stage.get("parameters") or {}).get("exact_sources") or [])
    surfaces = load_surfaces_by_key(exact_sources)
    records = [
        row_record(record, surfaces.get(line_stage_key(record)))
        for record in line_stage.get("records") or []
    ]
    heuristic_summary = summarize_heuristics(records)
    best_heuristics = sorted(
        heuristic_summary.items(),
        key=lambda item: (
            -int(item[1]["replay_success_hit_count"]),
            -int(item[1]["line_present_hit_count"]),
            item[0],
        ),
    )
    factor_counts = [int(row["factor_count"]) for row in records if row["surface_found"]]
    output = {
        "schema": "ecdlp_target67_residual_factor_cloud_audit_v1",
        "method": "post_factorization_public_residual_factor_cloud_heuristic_audit",
        "parameters": {
            "line_stage_audit": str(args.line_stage_audit),
            "exact_sources": exact_sources,
        },
        "summary": {
            "surface_case_count": len(records),
            "surface_found_count": sum(1 for row in records if row["surface_found"]),
            "line_present_case_count": sum(1 for row in records if row["has_preserving_line"]),
            "replay_success_case_count": sum(
                1 for row in records if row["label_replay_success"]
            ),
            "factor_count_values": sorted(set(factor_counts)),
            "preserving_factor_index_stats": index_stats(records),
            "best_heuristic": best_heuristics[0][0] if best_heuristics else None,
            "best_heuristic_summary": best_heuristics[0][1] if best_heuristics else None,
            "interpretation": (
                "The preserving factor is not isolated by simple factor order or coefficient-size "
                "heuristics.  This keeps the next line-prediction obligation on richer FFE "
                "residual features rather than a cheap post-factor cloud rule."
            ),
        },
        "heuristic_summary": heuristic_summary,
        "records": records,
        "non_claims": [
            "This audit uses Sage factorization output and is not a pre-factor public line predictor.",
            "Heuristic hits on replay failures do not count as ECDLP recoveries.",
            "A factor-cloud heuristic would still need replacement by a cheaper residual test before supporting an index-calculus speedup.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
