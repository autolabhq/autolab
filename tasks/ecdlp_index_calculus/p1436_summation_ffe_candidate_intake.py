#!/usr/bin/env python3
"""Audit an existing FFE candidate against the P1436/R68 new-row admission gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.p1436_summation_ffe_candidate_intake.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SELECTOR = STATE_DIR / "ffe_public_independent_row_expander_selector_184_191.json"
DEFAULT_BASELINE = STATE_DIR / "ffe_public_companion_assembly_replay_184_191.json"
DEFAULT_CANDIDATE = STATE_DIR / "ffe_public_independent_row_expander_replay_184_191.json"
DEFAULT_R68 = WORKTREE_ROOT / "p1553_ffe_fixed_sum_information_conservation_report_r68.json"
DEFAULT_OUTPUT = STATE_DIR / "p1436_summation_ffe_candidate_intake_expander_184_191.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    selector = read_json(args.selector)
    baseline = read_json(args.baseline)
    candidate = read_json(args.candidate)
    r68 = read_json(args.r68)
    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]

    relation_delta = int(candidate_summary["challenge_group_relation_count_sum"]) - int(
        baseline_summary["challenge_group_relation_count_sum"]
    )
    rank_delta = int(candidate_summary["challenge_group_max_rank"]) - int(
        baseline_summary["challenge_group_max_rank"]
    )
    source_case_delta = int(candidate_summary["source_case_count"]) - int(
        baseline_summary["source_case_count"]
    )
    signature_keys_remain = "signature-provided" in str(
        candidate_summary.get("next_obligation") or ""
    )
    replay_hash = sha256_file(args.candidate)
    contract = {
        "source_enumerator_id": "ffe-public-independent-row-expander-184-191",
        "scalar_blind": not signature_keys_remain,
        "new_factor_row_count": max(0, relation_delta),
        "independent_new_factor_row_count": max(0, rank_delta),
        "measured_source_operations": None,
        "direct_pair_complement_operations": None,
        "replay_artifact_sha256": replay_hash,
    }
    failures = []
    if signature_keys_remain:
        failures.append("source_enumerator_not_fully_public_scalar_blind")
    if relation_delta <= 0:
        failures.append("no_new_factor_rows")
    if rank_delta <= 0:
        failures.append("no_independent_new_factor_rows")
    failures.append("absolute_source_operations_not_reported")
    failures.append("direct_pair_complement_operations_not_reported")

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "ffe_public_independent_row_expander_184_191",
        "classification": "EXISTING_CANDIDATE_R68_ADMISSION_REJECTED",
        "comparison": {
            "baseline_source_case_count": baseline_summary["source_case_count"],
            "candidate_source_case_count": candidate_summary["source_case_count"],
            "added_source_case_count": source_case_delta,
            "baseline_challenge_group_relation_count_sum": baseline_summary[
                "challenge_group_relation_count_sum"
            ],
            "candidate_challenge_group_relation_count_sum": candidate_summary[
                "challenge_group_relation_count_sum"
            ],
            "new_factor_row_count_proxy": max(0, relation_delta),
            "baseline_challenge_group_max_rank": baseline_summary[
                "challenge_group_max_rank"
            ],
            "candidate_challenge_group_max_rank": candidate_summary[
                "challenge_group_max_rank"
            ],
            "independent_new_factor_row_count_proxy": max(0, rank_delta),
            "candidate_public_key_verified_group_count": candidate_summary[
                "challenge_group_public_key_verified_count"
            ],
            "candidate_group_count": candidate_summary["challenge_group_count"],
        },
        "new_factor_row_discovery_contract": contract,
        "admission": {
            "lane_admitted": False,
            "status": "rejected_no_new_independent_rows_and_incomplete_cost_contract",
            "failures": failures,
            "product_quotient_information_credit": 0,
        },
        "r68_binding": {
            "path": str(args.r68),
            "sha256": sha256_file(args.r68),
            "information_conservation_pass": r68["pass"],
            "ffe_product_relations_add_information_beyond_fixed_sum_rows": r68[
                "result"
            ]["ffe_product_relations_add_information_beyond_fixed_sum_rows"],
            "pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows": r68[
                "result"
            ]["pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows"],
        },
        "provenance": {
            "selector_path": str(args.selector),
            "selector_sha256": sha256_file(args.selector),
            "selector_schema": selector.get("schema"),
            "baseline_replay_path": str(args.baseline),
            "baseline_replay_sha256": sha256_file(args.baseline),
            "candidate_replay_path": str(args.candidate),
            "candidate_replay_sha256": replay_hash,
        },
        "conclusion": (
            "Adding 18 public-diversity source cases produced no new challenge-group "
            "relation and no rank increase. This existing FFE candidate fails the R68 "
            "new-factor-row gate and cannot be credited toward a generic-prime ECDLP "
            "speedup."
        ),
        "algorithm_breakthrough": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--r68", type=Path, default=DEFAULT_R68)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison = payload["comparison"]
    print(
        f"output={args.output} added_cases={comparison['added_source_case_count']} "
        f"new_rows={comparison['new_factor_row_count_proxy']} "
        f"new_rank={comparison['independent_new_factor_row_count_proxy']} "
        f"admitted={payload['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
