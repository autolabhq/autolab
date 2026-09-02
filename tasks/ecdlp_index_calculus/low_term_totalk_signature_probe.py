#!/usr/bin/env python3
"""Aggregate verifier-backed low-term-support total-k stress certificates.

The original signature aggregator was deliberately narrow and only accepted
total-2 leaf selectors.  This variant keeps the same downstream positive-case
shape but makes the accepted leaf selector family configurable, so total-3 and
total-4 rank-rescue experiments can be promoted without editing old probes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_totalk_signature_probe.json"
DEFAULT_SELECTOR_RE = re.compile(r"^mode_(?:cost_)?low_term_support_total[0-9]+$")
DEFAULT_SEED = "ecdlp-frontier-signed-dual-sieve-v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_sources(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_allowed_selectors(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def row_salt(row_key: str) -> int | None:
    try:
        return int(str(row_key).rsplit("salt", 1)[1])
    except (IndexError, ValueError):
        return None


def selector_allowed(selector: str, allowed: set[str]) -> bool:
    return selector in allowed if allowed else bool(DEFAULT_SELECTOR_RE.fullmatch(selector))


def case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    leaf_signature = tuple(
        (
            str(item.get("row_key")),
            tuple(int(leaf) for leaf in item.get("leaf_indices") or []),
        )
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict)
    )
    return (
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        str(case.get("leaf_selector")),
        leaf_signature,
    )


def inferred_surface_id(target: str, transfer_index: int, row_key: str, seed: str) -> str:
    challenge_seed = f"{seed}:shared-transfer:{transfer_index}:{target}"
    return f"{target}|{row_key}|{challenge_seed}"


def compact_case(
    path: Path,
    policy: str,
    row_selector: str | None,
    row: dict[str, Any],
    seed: str,
) -> dict[str, Any]:
    row_leaf_keys = [item for item in row.get("row_leaf_keys") or [] if isinstance(item, dict)]
    row_salts = []
    leaf_indices = []
    per_row = []
    target = str(row.get("target"))
    transfer_index = int(row.get("transfer_index") or 0)
    surface_ids = []
    for item in row_leaf_keys:
        row_key = str(item.get("row_key"))
        salt = row_salt(str(item.get("row_key")))
        leaves = [int(leaf) for leaf in item.get("leaf_indices") or []]
        surface_id = inferred_surface_id(target, transfer_index, row_key, seed)
        surface_ids.append(surface_id)
        if salt is not None:
            row_salts.append(salt)
        leaf_indices.extend(leaves)
        per_row.append(
            {
                "row_key": row_key,
                "surface_id": surface_id,
                "salt": salt,
                "leaf_indices": leaves,
            }
        )
    return {
        "source": str(path),
        "policy": str(policy),
        "row_selector": row_selector,
        "leaf_selector": row.get("selector"),
        "target": target,
        "transfer_index": transfer_index,
        "top_k": int(row.get("top_k") or 0),
        "ops_over_rho": row.get("ops_over_rho"),
        "below_rho": bool(row.get("below_rho")),
        "public_key_verified": bool(row.get("public_key_verified")),
        "selected_leaf_count": int(row.get("selected_leaf_count") or 0),
        "selected_row_count": int(row.get("selected_row_count") or 0),
        "relation_count": int(row.get("relation_count") or 0),
        "rank": int(row.get("rank") or 0),
        "row_salts": row_salts,
        "leaf_indices": leaf_indices,
        "unique_leaf_indices": sorted(set(leaf_indices)),
        "surface_ids": sorted(set(surface_ids)),
        "row_leaf_keys": per_row,
    }


def collect_cases(
    state_dir: Path,
    sources: list[str],
    allowed_selectors: set[str],
    min_rank: int,
    min_relations: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    positives: list[dict[str, Any]] = []
    verified_over_rho: list[dict[str, Any]] = []
    below_rho_unverified: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source in sources:
        path = state_dir / source
        data = load_json(path)
        for policy, result in (data.get("policies") or {}).items():
            row_selector = (result.get("row_selector") or {}).get("selector")
            for row in result.get("stress_leaf_results") or []:
                if not isinstance(row, dict):
                    continue
                selector = str(row.get("selector"))
                if not selector_allowed(selector, allowed_selectors):
                    continue
                case = compact_case(path, str(policy), row_selector, row, seed)
                key = case_key(case)
                if key in seen:
                    continue
                seen.add(key)
                verified = bool(case["public_key_verified"])
                below_rho = bool(case["below_rho"])
                enough_rank = int(case["rank"]) >= min_rank
                enough_relations = int(case["relation_count"]) >= min_relations
                if verified and below_rho and enough_rank and enough_relations:
                    positives.append(case)
                elif verified:
                    verified_over_rho.append(case)
                elif below_rho:
                    below_rho_unverified.append(case)
    sort_key = lambda row: (
        float(row.get("ops_over_rho") or 10**9),
        str(row.get("target")),
        int(row.get("transfer_index") or 0),
        int(row.get("top_k") or 0),
        str(row.get("policy")),
        str(row.get("leaf_selector")),
    )
    return (
        sorted(positives, key=sort_key),
        sorted(verified_over_rho, key=sort_key),
        sorted(below_rho_unverified, key=sort_key),
    )


def selected_output_cases(
    positive_source: str,
    strict_positive_cases: list[dict[str, Any]],
    verified_over_rho: list[dict[str, Any]],
    below_rho_unverified: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if positive_source == "strict":
        return strict_positive_cases
    if positive_source == "below_rho_all":
        return sorted(
            strict_positive_cases + below_rho_unverified,
            key=lambda row: (
                float(row.get("ops_over_rho") or 10**9),
                str(row.get("target")),
                int(row.get("transfer_index") or 0),
                int(row.get("top_k") or 0),
                str(row.get("policy")),
                str(row.get("leaf_selector")),
            ),
        )
    if positive_source == "verified_over_rho":
        return verified_over_rho
    if positive_source == "below_rho_unverified":
        return below_rho_unverified
    raise ValueError(f"unsupported positive source: {positive_source}")


def summarize_cases(
    positive_cases: list[dict[str, Any]],
    verified_over_rho: list[dict[str, Any]],
    below_rho_unverified: list[dict[str, Any]],
) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_policy: Counter[str] = Counter()
    by_row_selector: Counter[str] = Counter()
    by_leaf_selector: Counter[str] = Counter()
    leaf_by_target: dict[str, Counter[int]] = defaultdict(Counter)
    salt_by_target: dict[str, Counter[int]] = defaultdict(Counter)
    signature_by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for case in positive_cases:
        target = str(case.get("target"))
        by_target[target].append(case)
        by_policy[str(case.get("policy"))] += 1
        by_row_selector[str(case.get("row_selector"))] += 1
        by_leaf_selector[str(case.get("leaf_selector"))] += 1
        for leaf in case.get("leaf_indices") or []:
            leaf_by_target[target][int(leaf)] += 1
        for salt in case.get("row_salts") or []:
            salt_by_target[target][int(salt)] += 1
        signature = ",".join(str(leaf) for leaf in case.get("unique_leaf_indices") or [])
        if signature:
            signature_by_target[target][signature] += 1

    target_summaries = []
    ffe_targets = []
    for target, rows in sorted(by_target.items()):
        best = min(rows, key=lambda row: float(row.get("ops_over_rho") or 10**9))
        target_summaries.append(
            {
                "target": target,
                "case_count": len(rows),
                "transfer_indices": sorted({int(row.get("transfer_index") or 0) for row in rows}),
                "best_ops_over_rho": best.get("ops_over_rho"),
                "best_case": best,
                "leaf_index_counts": dict(leaf_by_target[target].most_common()),
                "row_salt_counts": dict(salt_by_target[target].most_common()),
                "leaf_signature_counts": dict(signature_by_target[target].most_common()),
            }
        )
        ffe_targets.append(
            {
                "target": target,
                "priority": f"total{int(best.get('selected_leaf_count') or 0)}_rank{int(best.get('rank') or 0)}",
                "best_ops_over_rho": best.get("ops_over_rho"),
                "leaf_indices": best.get("unique_leaf_indices") or [],
                "row_salts": best.get("row_salts") or [],
                "transfer_index": best.get("transfer_index"),
                "top_k": best.get("top_k"),
                "reason": (
                    "Promote this verifier-backed below-rho low-term-support "
                    "certificate into the FFE/summation-polynomial scorer."
                ),
            }
        )

    best_verified_over_rho = verified_over_rho[0] if verified_over_rho else None
    best_below_rho_unverified = below_rho_unverified[0] if below_rho_unverified else None
    return {
        "below_rho_case_count": len(positive_cases),
        "target_count": len(by_target),
        "policy_counts": dict(by_policy.most_common()),
        "row_selector_counts": dict(by_row_selector.most_common()),
        "leaf_selector_counts": dict(by_leaf_selector.most_common()),
        "target_summaries": target_summaries,
        "ffe_relation_generation_targets": sorted(
            ffe_targets,
            key=lambda row: (
                float(row.get("best_ops_over_rho") or 10**9),
                str(row.get("target")),
            ),
        ),
        "verified_over_rho_case_count": len(verified_over_rho),
        "best_verified_over_rho_case": best_verified_over_rho,
        "below_rho_unverified_case_count": len(below_rho_unverified),
        "best_below_rho_unverified_case": best_below_rho_unverified,
        "interpretation": (
            "Positive cases are strict: below rho, public-key verified, and "
            "meeting the rank/relation thresholds. Verified over-rho and "
            "below-rho unverified cases are kept only as next-probe diagnostics."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--allowed-leaf-selectors")
    parser.add_argument("--min-rank", type=int, default=2)
    parser.add_argument("--min-relations", type=int, default=2)
    parser.add_argument(
        "--positive-source",
        choices=("strict", "below_rho_all", "verified_over_rho", "below_rho_unverified"),
        default="strict",
        help=(
            "Which case family to write to positive_cases. Non-strict values "
            "are diagnostic FFE inputs, not breakthrough certificates."
        ),
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sources = parse_sources(args.sources)
    allowed_selectors = parse_allowed_selectors(args.allowed_leaf_selectors)
    positives, verified_over_rho, below_rho_unverified = collect_cases(
        args.state_dir,
        sources,
        allowed_selectors,
        int(args.min_rank),
        int(args.min_relations),
        str(args.seed),
    )
    output_positive_cases = selected_output_cases(
        str(args.positive_source),
        positives,
        verified_over_rho,
        below_rho_unverified,
    )
    output = {
        "schema": "ecdlp_low_term_totalk_signature_probe_v1",
        "method": "aggregate_low_term_total_k_below_rho_certificates",
        "parameters": {
            "state_dir": str(args.state_dir),
            "sources": sources,
            "allowed_leaf_selectors": sorted(allowed_selectors),
            "default_selector_regex": DEFAULT_SELECTOR_RE.pattern if not allowed_selectors else None,
            "min_rank": args.min_rank,
            "min_relations": args.min_relations,
            "positive_source": args.positive_source,
            "seed": args.seed,
        },
        "positive_cases": output_positive_cases,
        "diagnostics": {
            "verified_over_rho_cases": verified_over_rho[:16],
            "below_rho_unverified_cases": below_rho_unverified[:16],
        },
        "summary": {
            **summarize_cases(positives, verified_over_rho, below_rho_unverified),
            "positive_source": args.positive_source,
            "positive_source_case_count": len(output_positive_cases),
            "positive_source_surface_ids": sorted(
                {
                    str(surface_id)
                    for case in output_positive_cases
                    for surface_id in case.get("surface_ids") or []
                }
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
