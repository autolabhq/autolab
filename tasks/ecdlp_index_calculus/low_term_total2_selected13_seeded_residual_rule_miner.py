#!/usr/bin/env python3
"""Mine public residual branches from an already frozen selected13 policy.

The second-stage rule miner searches decision lists from scratch. This probe is
more surgical: it starts from a verified frozen-policy replay artifact, checks
that the seed policy reproduces the artifact on the merged top-5 diagnostics,
then ranks public candidate insertions against only the remaining top-5-covered
misses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import low_term_total2_selected13_salt_conditioned_second_stage_rule_miner as second_stage


SCHEMA = "ecdlp.low_term_total2_selected13_seeded_residual_rule_miner.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SEED_VALIDATION = (
    DEFAULT_STATE_DIR
    / "low_term_total2_selected13_second_stage_public_position_guard106_then_span_le6_position_validation_10056_10607_full147_probe.json"
)
DEFAULT_TOP5_REPLAYS = [
    DEFAULT_STATE_DIR / f"low_term_total2_selected13_second_stage_public_depth1_all_top5_diagnostic_10056_10607_{suffix}_probe.json"
    for suffix in ("top32", "q32_63", "q64_95", "q96_127", "q128_146")
]
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_seeded_residual_rule_miner_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_seeded_residual_rule_miner_probe.h"

PUBLIC_FIELDS = ("position", "active", "root", "rowhit", "scout", "span")
ORDER_MODES = ("position", "position_desc", "span", "span_root", "root", "active", "span_desc")
TRIPLE_FIELD_SETS = (
    ("position", "span"),
    ("position", "root", "span"),
    ("position", "rowhit", "span"),
    ("position", "scout", "span"),
    ("root", "rowhit", "span"),
    ("root", "scout", "span"),
    ("active", "root", "span"),
    ("active", "scout", "span"),
)
ORDER_CODES = {name: index for index, name in enumerate(ORDER_MODES, start=1)}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: Any, default: int = 0) -> int:
    return second_stage.as_int(value, default)


def predicate_text(predicate: dict[str, Any]) -> str:
    if predicate.get("kind") == "fallback":
        return "fallback"
    if predicate.get("kind") == "all_of" or "all" in predicate:
        return "&".join(predicate_text(item) for item in predicate.get("all") or [])
    field = str(predicate.get("field"))
    return f"{field}{predicate.get('op')}{predicate.get('value')}"


def predicate_match(record: dict[str, Any], predicate: dict[str, Any]) -> bool:
    if predicate.get("kind") == "fallback":
        return True
    if predicate.get("kind") == "all_of" or "all" in predicate:
        return all(predicate_match(record, item) for item in predicate.get("all") or [])
    value = as_int(record.get(str(predicate.get("field"))))
    target = as_int(predicate.get("value"))
    op = str(predicate.get("op"))
    if op == "==":
        return value == target
    if op == "<=":
        return value <= target
    if op == ">=":
        return value >= target
    return False


def compact_rule(rule: dict[str, Any], index: int) -> dict[str, Any]:
    order_mode = str(rule.get("order_mode") or rule.get("fallback_order_mode") or "")
    return {
        "order_mode": order_mode,
        "predicate": rule.get("predicate"),
        "predicate_text": predicate_text(rule.get("predicate") or {}),
        "rule_index": index,
    }


def seed_rules(seed_validation: dict[str, Any]) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    frozen = seed_validation.get("frozen_rule") or {}
    raw_rules = list(frozen.get("rules") or [])
    rules = []
    fallback_order_mode = "span_root"
    for raw in raw_rules:
        if raw.get("fallback_order_mode"):
            fallback_order_mode = str(raw.get("fallback_order_mode"))
            continue
        rules.append(
            {
                "order_mode": str(raw.get("order_mode")),
                "predicate": raw.get("predicate"),
                "predicate_text": predicate_text(raw.get("predicate") or {}),
            }
        )
    return rules, fallback_order_mode, raw_rules


def grouped_by_transfer(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        out.setdefault(as_int(record.get("transfer_index")), []).append(record)
    for rows in out.values():
        rows.sort(key=lambda row: second_stage.order_key(row, "position"))
    return out


def choose_record(
    records_by_transfer: dict[int, list[dict[str, Any]]],
    transfer_index: int,
    rules: list[dict[str, Any]],
    fallback_order_mode: str,
) -> tuple[dict[str, Any], str, str]:
    candidates = records_by_transfer.get(transfer_index, [])
    for rule in rules:
        matching = [record for record in candidates if predicate_match(record, rule["predicate"])]
        if matching:
            order_mode = str(rule["order_mode"])
            return sorted(matching, key=lambda record: second_stage.order_key(record, order_mode))[0], rule["predicate_text"], order_mode
    return (
        sorted(candidates, key=lambda record: second_stage.order_key(record, fallback_order_mode))[0],
        "fallback",
        fallback_order_mode,
    )


def accepted_transfers(
    records_by_transfer: dict[int, list[dict[str, Any]]],
    transfer_indices: list[int],
    rules: list[dict[str, Any]],
    fallback_order_mode: str,
) -> set[int]:
    out = set()
    for transfer_index in transfer_indices:
        chosen, _, _ = choose_record(records_by_transfer, transfer_index, rules, fallback_order_mode)
        if chosen.get("accepted_below_rho"):
            out.add(transfer_index)
    return out


def condition_from_terms(terms: tuple[tuple[str, str, int], ...]) -> dict[str, Any]:
    if len(terms) == 1:
        field, op, value = terms[0]
        return {"field": field, "kind": "comparison", "op": op, "value": value}
    return {
        "all": [
            {"field": field, "kind": "comparison", "op": op, "value": value}
            for field, op, value in terms
        ],
        "kind": "all_of",
    }


def candidate_predicates(
    records: list[dict[str, Any]],
    residual_accepted: list[dict[str, Any]],
    *,
    max_predicate_arity: int,
    max_candidate_predicates: int,
) -> list[dict[str, Any]]:
    atoms: list[tuple[str, str, int]] = []
    for field in PUBLIC_FIELDS:
        values = sorted({as_int(record.get(field)) for record in residual_accepted})
        for value in values:
            atoms.append((field, "==", value))
            atoms.append((field, "<=", value))
            atoms.append((field, ">=", value))
    atoms = [
        atom
        for atom in atoms
        if 0 < sum(predicate_match(record, condition_from_terms((atom,))) for record in records) < len(records)
    ]
    predicates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(terms: tuple[tuple[str, str, int], ...]) -> None:
        terms = tuple(sorted(terms))
        predicate = condition_from_terms(terms)
        text = predicate_text(predicate)
        if text in seen:
            return
        if any(predicate_match(record, predicate) for record in residual_accepted):
            predicates.append(predicate)
            seen.add(text)

    for atom in atoms:
        add((atom,))
    if max_predicate_arity <= 1:
        return predicates
    for left_index, left in enumerate(atoms):
        for right in atoms[left_index + 1 :]:
            if left[0] != right[0]:
                add((left, right))
    if max_predicate_arity <= 2:
        return predicates
    by_field = {field: [atom for atom in atoms if atom[0] == field] for field in PUBLIC_FIELDS}
    for fields in TRIPLE_FIELD_SETS:
        pools = [by_field[field] for field in fields]
        if any(not pool for pool in pools):
            continue
        for first in pools[0]:
            for second in pools[1]:
                if len(pools) == 2:
                    add((first, second))
                    continue
                for third in pools[2]:
                    add((first, second, third))
    predicates.sort(
        key=lambda predicate: (
            -sum(predicate_match(record, predicate) for record in residual_accepted),
            sum(predicate_match(record, predicate) for record in records),
            len(predicate_text(predicate)),
            predicate_text(predicate),
        )
    )
    return predicates[:max_candidate_predicates]


def candidate_insert_indices(seed: list[dict[str, Any]], mode: str) -> list[int]:
    if mode == "all":
        return list(range(len(seed) + 1))
    span_fallback = next(
        (
            index
            for index, rule in enumerate(seed)
            if rule.get("predicate_text") == "span<=6"
            or (
                (rule.get("predicate") or {}).get("field") == "span"
                and (rule.get("predicate") or {}).get("op") == "<="
            )
        ),
        len(seed),
    )
    return [span_fallback]


def score_candidate(
    candidate_index: int,
    seed: list[dict[str, Any]],
    fallback_order_mode: str,
    records_by_transfer: dict[int, list[dict[str, Any]]],
    transfer_indices: list[int],
    seed_accepted: set[int],
    predicate: dict[str, Any],
    order_mode: str,
    insert_index: int,
) -> dict[str, Any]:
    rule = {
        "order_mode": order_mode,
        "predicate": predicate,
        "predicate_text": predicate_text(predicate),
    }
    trial = seed[:insert_index] + [rule] + seed[insert_index:]
    accepted = accepted_transfers(records_by_transfer, transfer_indices, trial, fallback_order_mode)
    gains = sorted(accepted - seed_accepted)
    losses = sorted(seed_accepted - accepted)
    return {
        "accepted_below_rho_transfer_count": len(accepted),
        "candidate_index": candidate_index,
        "gain_count": len(gains),
        "gain_transfers": gains,
        "insert_index": insert_index,
        "loss_count": len(losses),
        "loss_transfers": losses,
        "net_gain": len(gains) - len(losses),
        "order_mode": order_mode,
        "predicate": predicate,
        "predicate_text": predicate_text(predicate),
        "rule_id_u64": second_stage.min_transfer.digest_u64(
            {
                "insert_index": insert_index,
                "order_mode": order_mode,
                "predicate": predicate,
            }
        ),
    }


def residual_hit_count(
    records_by_transfer: dict[int, list[dict[str, Any]]],
    residual_actionable: list[int],
    predicate: dict[str, Any],
    order_mode: str,
) -> int:
    hits = 0
    rule = {
        "order_mode": order_mode,
        "predicate": predicate,
        "predicate_text": predicate_text(predicate),
    }
    for transfer_index in residual_actionable:
        candidates = [record for record in records_by_transfer.get(transfer_index, []) if predicate_match(record, predicate)]
        if not candidates:
            continue
        chosen = sorted(candidates, key=lambda record: second_stage.order_key(record, order_mode))[0]
        if chosen.get("accepted_below_rho"):
            hits += 1
    return hits


def choose_top_candidates(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    records.sort(
        key=lambda row: (
            -as_int(row.get("accepted_below_rho_transfer_count")),
            -as_int(row.get("net_gain")),
            -as_int(row.get("gain_count")),
            as_int(row.get("loss_count")),
            len(str(row.get("predicate_text"))),
            str(row.get("order_mode")),
            as_int(row.get("insert_index")),
        )
    )
    return records[:limit]


def render_c_header(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = []
    for record in records:
        rows.append(
            "  {"
            f"{as_int(record.get('candidate_index'))}ULL, "
            f"{as_int(record.get('rule_id_u64'))}ULL, "
            f"{as_int(record.get('insert_index'))}ULL, "
            f"{ORDER_CODES.get(str(record.get('order_mode')), 0)}ULL, "
            f"{as_int(record.get('accepted_below_rho_transfer_count'))}ULL, "
            f"{as_int(record.get('gain_count'))}ULL, "
            f"{as_int(record.get('loss_count'))}ULL, "
            f"{as_int(record.get('net_gain'))}LL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_SEEDED_RESIDUAL_RULE_MINER_H
#define LOW_TERM_TOTAL2_SELECTED13_SEEDED_RESIDUAL_RULE_MINER_H

#include <stdint.h>

#define SELECTED13_SEEDED_RESIDUAL_CANDIDATE_COUNT {len(records)}
#define SELECTED13_SEEDED_RESIDUAL_SEED_ACCEPTED_BELOW_COUNT {as_int(summary.get("seed_accepted_below_rho_transfer_count"))}
#define SELECTED13_SEEDED_RESIDUAL_BEST_ACCEPTED_BELOW_COUNT {as_int(summary.get("best_accepted_below_rho_transfer_count"))}
#define SELECTED13_SEEDED_RESIDUAL_BEST_NO_LOSS_ACCEPTED_BELOW_COUNT {as_int(summary.get("best_no_loss_accepted_below_rho_transfer_count"))}

typedef struct {{
  uint64_t candidate_index;
  uint64_t rule_id_u64;
  uint64_t insert_index;
  uint64_t order_mode_code;
  uint64_t accepted_below_rho_transfer_count;
  uint64_t gain_count;
  uint64_t loss_count;
  int64_t net_gain;
}} selected13_seeded_residual_candidate_t;

static const selected13_seeded_residual_candidate_t SELECTED13_SEEDED_RESIDUAL_CANDIDATES[] = {{
{chr(10).join(rows)}
}};

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  uint64_t failure_count = 0;
  uint64_t candidate_count =
      sizeof(SELECTED13_SEEDED_RESIDUAL_CANDIDATES) / sizeof(SELECTED13_SEEDED_RESIDUAL_CANDIDATES[0]);
  uint64_t best_accepted = SELECTED13_SEEDED_RESIDUAL_SEED_ACCEPTED_BELOW_COUNT;
  uint64_t best_no_loss = SELECTED13_SEEDED_RESIDUAL_SEED_ACCEPTED_BELOW_COUNT;

  if (candidate_count != SELECTED13_SEEDED_RESIDUAL_CANDIDATE_COUNT) failure_count++;
  for (size_t i = 0; i < candidate_count; i++) {{
    const selected13_seeded_residual_candidate_t *candidate =
        &SELECTED13_SEEDED_RESIDUAL_CANDIDATES[i];
    if (candidate->rule_id_u64 == 0ULL) failure_count++;
    if (candidate->order_mode_code == 0ULL) failure_count++;
    if (candidate->accepted_below_rho_transfer_count > best_accepted) {{
      best_accepted = candidate->accepted_below_rho_transfer_count;
    }}
    if (candidate->loss_count == 0ULL &&
        candidate->accepted_below_rho_transfer_count > best_no_loss) {{
      best_no_loss = candidate->accepted_below_rho_transfer_count;
    }}
  }}
  if (best_accepted != SELECTED13_SEEDED_RESIDUAL_BEST_ACCEPTED_BELOW_COUNT) failure_count++;
  if (best_no_loss != SELECTED13_SEEDED_RESIDUAL_BEST_NO_LOSS_ACCEPTED_BELOW_COUNT) failure_count++;
  printf("selected13_seeded_residual_rule_miner_preflight candidates=%llu seed_accepted=%llu best=%llu best_no_loss=%llu failures=%llu\\n",
         (unsigned long long)candidate_count,
         (unsigned long long)SELECTED13_SEEDED_RESIDUAL_SEED_ACCEPTED_BELOW_COUNT,
         (unsigned long long)best_accepted,
         (unsigned long long)best_no_loss,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_seeded_residual_rule_miner_preflight_") as tmp:
        tmp_path = Path(tmp)
        c_path = tmp_path / "preflight.c"
        exe_path = tmp_path / "preflight"
        local_header = tmp_path / header_path.name
        c_path.write_text(source)
        local_header.write_text(header_path.read_text())
        compile_cmd = ["cc", "-std=c99", "-Wall", "-Wextra", "-O2", str(c_path), "-o", str(exe_path)]
        compile_run = subprocess.run(compile_cmd, text=True, capture_output=True, check=False)
        if compile_run.returncode != 0:
            return {
                "compile_command": compile_cmd,
                "compile_returncode": compile_run.returncode,
                "compile_stderr": compile_run.stderr,
                "verified": False,
            }
        preflight_run = subprocess.run([str(exe_path)], text=True, capture_output=True, check=False)
        return {
            "compile_command": compile_cmd,
            "compile_returncode": compile_run.returncode,
            "preflight_returncode": preflight_run.returncode,
            "preflight_stdout": preflight_run.stdout.strip(),
            "preflight_stderr": preflight_run.stderr.strip(),
            "verified": preflight_run.returncode == 0,
        }


def summarize(
    seed_accepted: set[int],
    seed_artifact_accepted: set[int],
    top_records: list[dict[str, Any]],
    residual_actionable: list[int],
    residual_materialization: list[int],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    best = top_records[0] if top_records else {}
    no_loss = [record for record in top_records if as_int(record.get("loss_count")) == 0]
    best_no_loss = no_loss[0] if no_loss else {}
    if seed_accepted != seed_artifact_accepted:
        failures.append(
            {
                "code": "seed_policy_top5_replay_mismatch",
                "extra_from_seed": sorted(seed_accepted - seed_artifact_accepted),
                "missing_from_seed": sorted(seed_artifact_accepted - seed_accepted),
            }
        )
    return {
        "best_accepted_below_rho_transfer_count": best.get("accepted_below_rho_transfer_count", len(seed_accepted)),
        "best_gain_count": best.get("gain_count", 0),
        "best_gain_transfers": best.get("gain_transfers", []),
        "best_loss_count": best.get("loss_count", 0),
        "best_loss_transfers": best.get("loss_transfers", []),
        "best_no_loss_accepted_below_rho_transfer_count": best_no_loss.get(
            "accepted_below_rho_transfer_count", len(seed_accepted)
        ),
        "best_no_loss_gain_count": best_no_loss.get("gain_count", 0),
        "best_no_loss_gain_transfers": best_no_loss.get("gain_transfers", []),
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "residual_rule_selection_miss_count": len(residual_actionable),
        "residual_rule_selection_misses": residual_actionable,
        "residual_top5_materialization_miss_count": len(residual_materialization),
        "residual_top5_materialization_misses": residual_materialization,
        "seed_accepted_below_rho_transfer_count": len(seed_accepted),
        "seed_artifact_accepted_below_rho_transfer_count": len(seed_artifact_accepted),
        "verified": not failures,
        "worker_interpretation": (
            "This seeded miner ranks public candidate branches against an already "
            "verified frozen policy. Candidate labels come from direct top-5 replay, "
            "so any promoted branch still needs a fresh one-leaf verifier replay."
        ),
    }


def claim_status(failures: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if failures:
        return "SELECTED13_SEEDED_RESIDUAL_RULE_MINER_FAILED"
    if as_int(summary.get("best_no_loss_gain_count")) > 0:
        return "SELECTED13_SEEDED_RESIDUAL_RULE_MINER_FOUND_NO_LOSS_GAIN"
    if as_int(summary.get("best_gain_count")) > as_int(summary.get("best_loss_count")):
        return "SELECTED13_SEEDED_RESIDUAL_RULE_MINER_FOUND_NET_GAIN"
    return "SELECTED13_SEEDED_RESIDUAL_RULE_MINER_NO_GAIN"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    seed_validation = load_json(Path(args.seed_validation))
    source = second_stage.load_top5_source([Path(path) for path in args.top5_replay])
    failures = list(source.get("failures") or [])
    records = [second_stage.parse_record(row) for row in source.get("shortlist_leaf_replays") or []]
    records_by_transfer = grouped_by_transfer(records)
    seed, fallback_order_mode, raw_seed_rules = seed_rules(seed_validation)
    transfer_indices = sorted(as_int(item) for item in (seed_validation.get("summary") or {}).get("selected_validation_transfers") or [])
    seed_artifact_accepted = set(
        as_int(item)
        for item in (seed_validation.get("summary") or {}).get("below_rho_accepted_relation_export_transfers") or []
    )
    seed_accepted = accepted_transfers(records_by_transfer, transfer_indices, seed, fallback_order_mode)
    top5_covered = {
        transfer_index
        for transfer_index in transfer_indices
        if any(row.get("accepted_below_rho") for row in records_by_transfer.get(transfer_index, []))
    }
    residual_actionable = sorted(top5_covered - seed_accepted)
    residual_materialization = sorted(set(transfer_indices) - top5_covered)
    residual_accepted_records = [
        row
        for transfer_index in residual_actionable
        for row in records_by_transfer.get(transfer_index, [])
        if row.get("accepted_below_rho")
    ]
    predicates = candidate_predicates(
        records,
        residual_accepted_records,
        max_predicate_arity=args.max_predicate_arity,
        max_candidate_predicates=args.max_candidate_predicates,
    )
    inserts = candidate_insert_indices(seed, args.insert_mode)
    scored_specs = []
    spec_index = 0
    for predicate in predicates:
        text = predicate_text(predicate)
        for order_mode in ORDER_MODES:
            residual_hits = residual_hit_count(records_by_transfer, residual_actionable, predicate, order_mode)
            for insert_index in inserts:
                scored_specs.append(
                    (
                        residual_hits,
                        -len(text),
                        text,
                        order_mode,
                        -insert_index,
                        spec_index,
                        predicate,
                        order_mode,
                        insert_index,
                    )
                )
                spec_index += 1
    scored_specs.sort(reverse=True)
    candidates = []
    for index, spec in enumerate(scored_specs[: args.full_score_candidate_limit]):
        predicate = spec[6]
        order_mode = spec[7]
        insert_index = spec[8]
        candidates.append(
            score_candidate(
                index,
                seed,
                fallback_order_mode,
                records_by_transfer,
                transfer_indices,
                seed_accepted,
                predicate,
                order_mode,
                insert_index,
            )
        )
    top_records = choose_top_candidates(candidates, args.top_candidate_limit)
    summary = summarize(
        seed_accepted,
        seed_artifact_accepted,
        top_records,
        residual_actionable,
        residual_materialization,
        failures,
    )
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, summary),
        "parameters": {
            "insert_mode": args.insert_mode,
            "max_candidate_predicates": args.max_candidate_predicates,
            "max_predicate_arity": args.max_predicate_arity,
            "full_score_candidate_limit": args.full_score_candidate_limit,
            "seed_validation": str(Path(args.seed_validation)),
            "top5_replays": [str(Path(path)) for path in args.top5_replay],
            "top_candidate_limit": args.top_candidate_limit,
        },
        "seed_policy": {
            "fallback_order_mode": fallback_order_mode,
            "raw_frozen_rules": raw_seed_rules,
            "rules": [compact_rule(rule, index) for index, rule in enumerate(seed)],
            "selector_id": (seed_validation.get("frozen_rule") or {}).get("selector_id"),
        },
        "summary": summary,
        "top_candidate_records": top_records,
        "failures": failures,
        "honesty_boundary": {
            "direct_top5_replay_labels_used_for_mining": True,
            "fresh_one_leaf_replay_required_before_promotion": True,
            "general_ecdlp_algorithm_claimed": False,
            "seed_policy_verified_elsewhere": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-validation", type=Path, default=DEFAULT_SEED_VALIDATION)
    parser.add_argument("--top5-replay", type=Path, nargs="+", default=DEFAULT_TOP5_REPLAYS)
    parser.add_argument("--insert-mode", choices=("before_span_fallback", "all"), default="before_span_fallback")
    parser.add_argument("--max-candidate-predicates", type=int, default=512)
    parser.add_argument("--max-predicate-arity", type=int, default=2)
    parser.add_argument("--full-score-candidate-limit", type=int, default=512)
    parser.add_argument("--top-candidate-limit", type=int, default=64)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["top_candidate_records"], payload["summary"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = claim_status(payload["failures"], payload["summary"])
    write_json(Path(args.out), payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "out": str(args.out),
                "summary": payload["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
