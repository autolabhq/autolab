#!/usr/bin/env python3
"""Mine a frozen public x-match orientation rule from compact audit artifacts.

This script may use ``valid_relation`` labels from training windows to rank
candidate rules.  The emitted rule string itself is a public ``where:`` filter
over x-match metadata and is intended to be replayed on unseen windows by
``ffe_public_linear_factor_xmatch_orientation_audit.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def support_key(match: dict[str, Any]) -> str:
    support = sorted({int(index) for index in match.get("unsigned_indices") or []})
    return "+".join(str(index) for index in support)


def row_salt(row_key: str) -> int | None:
    if ":salt" not in row_key:
        return None
    try:
        return int(row_key.rsplit(":salt", 1)[1])
    except ValueError:
        return None


def best_case_keys(artifact: dict[str, Any]) -> set[str]:
    best = ((artifact.get("summary") or {}).get("best_verified_rules") or [])
    if best and best[0].get("case_key"):
        return {str(best[0]["case_key"])}
    out = set()
    for case in artifact.get("cases") or []:
        if any(match.get("valid_relation") for row in case.get("rows") or [] for match in row.get("xmatches") or []):
            out.add(str(case.get("case_key")))
    return out


def iter_training_cases(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for artifact in artifacts:
        wanted = best_case_keys(artifact)
        for case in artifact.get("cases") or []:
            if str(case.get("case_key")) in wanted:
                selected.append(case)
    return selected


def atoms_for_cases(cases: list[dict[str, Any]]) -> list[str]:
    atoms: set[str] = {
        "candidate_eq_scheduled",
        "candidate_lt_scheduled",
        "candidate_gt_scheduled",
    }
    row_counts: set[int] = set()
    for case in cases:
        for row in case.get("rows") or []:
            xmatches = row.get("xmatches") or []
            row_counts.add(len(xmatches))
            salt = row_salt(str(row.get("row_key") or ""))
            for match in xmatches:
                atoms.add(f"candidate_pos={int(match.get('candidate_pos') or 0)}")
                atoms.add(f"scheduled_trial={int(match.get('scheduled_trial') or 0)}")
                atoms.add(f"support={support_key(match)}")
                atoms.add(f"term_shape={match.get('term_shape')}")
                if salt is not None:
                    for modulus in (2, 3, 4):
                        atoms.add(f"salt_mod{modulus}={salt % modulus}")
    for count in sorted(row_counts):
        atoms.add(f"row_xmatch_count={count}")
        atoms.add(f"row_xmatch_count<={count}")
        atoms.add(f"row_xmatch_count>={count}")
    return sorted(atom for atom in atoms if atom and not atom.endswith("=None"))


def atom_matches(atom: str, match: dict[str, Any], row: dict[str, Any]) -> bool:
    if atom == "candidate_eq_scheduled":
        return int(match.get("candidate_pos") or 0) == int(match.get("scheduled_trial") or 0)
    if atom == "candidate_lt_scheduled":
        return int(match.get("candidate_pos") or 0) < int(match.get("scheduled_trial") or 0)
    if atom == "candidate_gt_scheduled":
        return int(match.get("candidate_pos") or 0) > int(match.get("scheduled_trial") or 0)
    if atom.startswith("candidate_pos="):
        return int(match.get("candidate_pos") or 0) == int(atom.split("=", 1)[1])
    if atom.startswith("scheduled_trial="):
        return int(match.get("scheduled_trial") or 0) == int(atom.split("=", 1)[1])
    if atom.startswith("support="):
        return support_key(match) == atom.split("=", 1)[1]
    if atom.startswith("term_shape="):
        return str(match.get("term_shape")) == atom.split("=", 1)[1]
    if atom.startswith("salt_mod"):
        lhs, rhs = atom.split("=", 1)
        salt = row_salt(str(row.get("row_key") or ""))
        return salt is not None and salt % int(lhs.removeprefix("salt_mod")) == int(rhs)
    if atom.startswith("row_xmatch_count<="):
        return len(row.get("xmatches") or []) <= int(atom.split("<=", 1)[1])
    if atom.startswith("row_xmatch_count>="):
        return len(row.get("xmatches") or []) >= int(atom.split(">=", 1)[1])
    if atom.startswith("row_xmatch_count="):
        return len(row.get("xmatches") or []) == int(atom.split("=", 1)[1])
    raise ValueError(f"unknown atom: {atom}")


def clause_matches(clause: tuple[str, ...], match: dict[str, Any], row: dict[str, Any]) -> bool:
    return all(atom_matches(atom, match, row) for atom in clause)


def rule_matches(rule: tuple[tuple[str, ...], ...], match: dict[str, Any], row: dict[str, Any]) -> bool:
    return any(clause_matches(clause, match, row) for clause in rule)


def estimate_case_ops(case: dict[str, Any], rule: tuple[tuple[str, ...], ...]) -> dict[str, Any]:
    selected = valid = invalid = 0
    measured_ops = 0
    rho = None
    rows_used = 0
    for row in case.get("rows") or []:
        xmatches = row.get("xmatches") or []
        selected_for_row = [match for match in xmatches if rule_matches(rule, match, row)]
        core_ops = int(((row.get("baseline_costs") or {}).get("projected_core_ops_before_filtered_association")) or 0)
        measured_ops += core_ops + len(row.get("selected_leaf_indices") or []) + 2 * len(selected_for_row)
        if selected_for_row or xmatches:
            rows_used += 1
        for match in selected_for_row:
            selected += 1
            if match.get("valid_relation"):
                valid += 1
            else:
                invalid += 1
    for result in case.get("rule_results") or []:
        charged = result.get("charged_models") or {}
        if charged.get("rho") is not None:
            rho = int(charged["rho"])
            break
    return {
        "case_key": case.get("case_key"),
        "selected_xmatch_count": selected,
        "selected_valid_relation_count": valid,
        "selected_invalid_relation_count": invalid,
        "estimated_measured_ops": measured_ops,
        "estimated_measured_ops_over_rho": round(measured_ops / max(1, int(rho or 1)), 8) if rho else None,
        "rho": rho,
        "rows_used": rows_used,
    }


def rule_text(rule: tuple[tuple[str, ...], ...]) -> str:
    return "where:" + "|".join("&".join(clause) for clause in rule)


def build_candidate_rules(atoms: list[str], max_clause_size: int, max_dnf_clauses: int, top_clause_count: int, cases: list[dict[str, Any]]) -> list[tuple[tuple[str, ...], ...]]:
    clauses: list[tuple[str, ...]] = []
    for size in range(1, max_clause_size + 1):
        clauses.extend(tuple(combo) for combo in itertools.combinations(atoms, size))
    scored_clauses = []
    for clause in clauses:
        as_rule = (clause,)
        summaries = [estimate_case_ops(case, as_rule) for case in cases]
        scored_clauses.append((score_summaries(summaries, len(clause)), clause))
    top_clauses = [clause for _score, clause in sorted(scored_clauses)[:top_clause_count]]
    rules: set[tuple[tuple[str, ...], ...]] = {(clause,) for clause in top_clauses}
    for count in range(2, max_dnf_clauses + 1):
        for combo in itertools.combinations(top_clauses, count):
            rules.add(tuple(sorted(combo)))
    return sorted(rules, key=lambda rule: (len(rule), sum(len(clause) for clause in rule), rule_text(rule)))


def score_summaries(summaries: list[dict[str, Any]], complexity: int) -> tuple[Any, ...]:
    covered_cases = sum(1 for summary in summaries if summary["selected_valid_relation_count"] > 0)
    total_valid = sum(int(summary["selected_valid_relation_count"]) for summary in summaries)
    total_invalid = sum(int(summary["selected_invalid_relation_count"]) for summary in summaries)
    total_selected = sum(int(summary["selected_xmatch_count"]) for summary in summaries)
    worst_ratio = max(
        [float(summary["estimated_measured_ops_over_rho"]) for summary in summaries if summary.get("estimated_measured_ops_over_rho") is not None],
        default=999.0,
    )
    return (
        -covered_cases,
        total_invalid,
        -total_valid,
        total_selected,
        round(worst_ratio, 8),
        complexity,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-artifact", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-clause-size", type=int, default=3)
    parser.add_argument("--max-dnf-clauses", type=int, default=2)
    parser.add_argument("--top-clause-count", type=int, default=64)
    parser.add_argument("--top-rules", type=int, default=16)
    args = parser.parse_args()

    artifacts = [load_json(path) for path in args.training_artifact]
    cases = iter_training_cases(artifacts)
    atoms = atoms_for_cases(cases)
    rules = build_candidate_rules(
        atoms,
        max_clause_size=int(args.max_clause_size),
        max_dnf_clauses=int(args.max_dnf_clauses),
        top_clause_count=int(args.top_clause_count),
        cases=cases,
    )
    scored = []
    for rule in rules:
        summaries = [estimate_case_ops(case, rule) for case in cases]
        scored.append(
            {
                "rule": rule_text(rule),
                "score": score_summaries(summaries, sum(len(clause) for clause in rule)),
                "training_case_summaries": summaries,
            }
        )
    scored.sort(key=lambda item: tuple(item["score"]))
    output = {
        "schema": "ecdlp_public_xmatch_orientation_rule_miner_v1",
        "method": "label_ranked_public_where_rule_search",
        "training_artifacts": [str(path) for path in args.training_artifact],
        "training_case_count": len(cases),
        "candidate_atom_count": len(atoms),
        "candidate_atoms": atoms,
        "frozen_rule": scored[0]["rule"] if scored else None,
        "top_rules": scored[: int(args.top_rules)],
        "interpretation": (
            "The miner ranks rules using training valid_relation labels, but "
            "each emitted where: rule is a public filter over x-match metadata. "
            "The frozen_rule must be replayed on a later unseen window before "
            "it can support a speedup claim."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"frozen_rule": output["frozen_rule"], "top_rules": output["top_rules"][:3]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
