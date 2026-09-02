#!/usr/bin/env python3
"""Audit public fourth-leaf marginal lift triggers.

The root-diversity companion selector can now retain exact total4-over-total3
marginal row/leaf profiles.  This probe audits which public fourth-leaf lift
features explain the exact replay recoveries, then optionally emits a narrowed
bridge JSON for replay under a declared trigger.

Verifier labels are used only for retrospective audit and scoring.  The emitted
bridge is selected only from public row/leaf fields and the explicit trigger
arguments.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SELECTOR_SOURCE = (
    DEFAULT_STATE_DIR
    / "ffe_public_root_diversity_companion_selector_total3_total4_public_bounded_full_policy_pair_cap4_marginal_200_207.json"
)
DEFAULT_REPLAY_SOURCE = (
    DEFAULT_STATE_DIR
    / "ffe_public_root_diversity_companion_replay_total3_total4_public_bounded_full_policy_pair_cap4_marginal_exact_200_207.json"
)
DEFAULT_POLICY_SOURCE = (
    DEFAULT_STATE_DIR
    / "ffe_prefactor_unique_leaf_single_hit_root_policy_total3_total4_public_bounded_full_200_207.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "ffe_public_fourth_leaf_trigger_audit_total3_total4_public_bounded_full_policy_pair_cap4_marginal_exact_200_207.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def int_list(values: Any) -> list[int]:
    out = []
    for value in values or []:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def ratio(values: tuple[int, int]) -> float | None:
    numerator, denominator = values
    return round(numerator / denominator, 8) if denominator else None


def leaf_key(values: Any) -> str:
    return ",".join(str(value) for value in int_list(values))


def policy_root(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    chosen = row.get("chosen_candidate") or {}
    try:
        return int(chosen.get("root"))
    except (TypeError, ValueError):
        roots = int_list(row.get("pre_factor_selected_hit_roots"))
        return roots[0] if roots else None


def policy_compact(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "policy_surface_known": False,
            "policy_public_zero_recovered": None,
            "policy_chosen_root": None,
            "policy_pre_factor_selected_hit_roots": [],
            "policy_total_ops_over_rho": None,
            "policy_direct_total_ops_over_rho": None,
            "policy_chosen_false_positive": None,
        }
    return {
        "policy_surface_known": True,
        "policy_public_zero_recovered": bool(row.get("public_zero_recovered")),
        "policy_chosen_root": policy_root(row),
        "policy_pre_factor_selected_hit_roots": int_list(row.get("pre_factor_selected_hit_roots")),
        "policy_total_ops_over_rho": round_or_none(row.get("total_ops_over_rho")),
        "policy_direct_total_ops_over_rho": round_or_none(row.get("direct_total_ops_over_rho")),
        "policy_chosen_false_positive": bool(row.get("chosen_false_positive")),
    }


def replay_case_index(replay: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("case_key")): case
        for case in replay.get("cases") or []
        if isinstance(case, dict) and case.get("case_key") is not None
    }


def group_index(replay: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_case_key: dict[str, dict[str, Any]] = {}
    groups = []
    for group in replay.get("challenge_groups") or []:
        if not isinstance(group, dict):
            continue
        groups.append(group)
        for case_key in group.get("case_keys") or []:
            by_case_key[str(case_key)] = group
    return by_case_key, groups


def retained_scan_for_row(
    replay_case: dict[str, Any] | None,
    row_key: str,
    leaf_indices: list[int],
) -> dict[str, Any] | None:
    if not replay_case:
        return None
    replay = replay_case.get("retained_only_replay") or {}
    leaves = sorted(int(leaf) for leaf in leaf_indices)
    for row in replay.get("rows") or []:
        if str(row.get("row_key")) != row_key:
            continue
        scan = row.get("scan") or {}
        if sorted(int(leaf) for leaf in scan.get("selected_leaf_indices") or []) == leaves:
            return scan
    return None


def row_record(
    case: dict[str, Any],
    row: dict[str, Any],
    replay_case: dict[str, Any] | None,
    group: dict[str, Any] | None,
    policy_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row_key = str(row.get("row_key") or "")
    surface_id = str(row.get("surface_id") or "")
    leaf_indices = int_list(row.get("leaf_indices"))
    base_leaf_indices = int_list(row.get("base_leaf_indices"))
    added_leaf_indices = int_list(row.get("added_leaf_indices"))
    scan = retained_scan_for_row(replay_case, row_key, leaf_indices)
    retained = (replay_case or {}).get("retained_only_replay") or {}
    salt = row.get("salt")
    try:
        salt_int = int(salt)
    except (TypeError, ValueError):
        salt_int = None
    record = {
        "record_key": "|".join([str(case.get("case_key")), row_key, leaf_key(leaf_indices)]),
        "case_key": str(case.get("case_key")),
        "target": str(case.get("target")),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "leaf_selector": case.get("leaf_selector"),
        "row_key": row_key,
        "row_schedule_key": row.get("row_schedule_key"),
        "surface_id": surface_id,
        "salt": salt_int,
        "salt_mod_2": None if salt_int is None else salt_int % 2,
        "salt_mod_4": None if salt_int is None else salt_int % 4,
        "salt_mod_8": None if salt_int is None else salt_int % 8,
        "salt_mod_16": None if salt_int is None else salt_int % 16,
        "leaf_indices": leaf_indices,
        "base_leaf_indices": base_leaf_indices,
        "added_leaf_indices": added_leaf_indices,
        "base_leaf_key": leaf_key(base_leaf_indices),
        "added_leaf_key": leaf_key(added_leaf_indices),
        "base_added_key": f"{leaf_key(base_leaf_indices)}->{leaf_key(added_leaf_indices)}",
        "is_anchor_surface": surface_id in set(case.get("anchor_surface_ids") or []),
        "is_companion_surface": surface_id in set(case.get("companion_surface_ids") or []),
        "signature_public_key_verified": bool(case.get("signature_public_key_verified")),
        "signature_rank": int(case.get("signature_rank") or 0),
        "signature_relation_count": int(case.get("signature_relation_count") or 0),
        "source_ops_over_rho": round_or_none(case.get("ops_over_rho")),
        "retained_case_public_key_verified": bool(retained.get("public_key_verified")),
        "retained_case_derived": bool(retained.get("derived")),
        "retained_case_rank": int(retained.get("rank") or 0),
        "retained_case_relation_count": int(retained.get("relation_count") or 0),
        "retained_case_ops_over_rho": round_or_none(retained.get("ops_over_rho")),
        "retained_case_derived_secret": retained.get("derived_secret"),
        "retained_row_relation_count": int((scan or {}).get("relation_count") or 0),
        "retained_row_rank": int((scan or {}).get("rank") or 0),
        "retained_row_public_key_verified": bool((scan or {}).get("row_public_key_verified")),
        "retained_row_ops": int((scan or {}).get("preassociation_filter_ops") or 0),
        "retained_row_ops_over_rho": round_or_none((scan or {}).get("preassociation_filter_ops_over_rho")),
        "retained_row_selected_hit_events": int((scan or {}).get("selected_hit_events") or 0),
        "retained_row_selected_hit_roots": int((scan or {}).get("selected_hit_roots") or 0),
        "challenge_group_key": None,
        "challenge_group_case_count": None,
        "challenge_group_public_key_verified": False,
        "challenge_group_rank": None,
        "challenge_group_relation_count": None,
        "challenge_group_derived_secret": None,
    }
    if group:
        record.update(
            {
                "challenge_group_key": f"{group.get('target')}|{group.get('transfer_index')}",
                "challenge_group_case_count": int(group.get("case_count") or 0),
                "challenge_group_public_key_verified": bool(group.get("public_key_verified")),
                "challenge_group_rank": int(group.get("rank") or 0),
                "challenge_group_relation_count": int(group.get("relation_count") or 0),
                "challenge_group_derived_secret": group.get("derived_secret"),
            }
        )
    record.update(policy_compact(policy_rows.get(surface_id)))
    return record


def collect_row_records(
    selector: dict[str, Any],
    replay: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    replay_by_case = replay_case_index(replay)
    groups_by_case, _groups = group_index(replay)
    policy_rows = {
        str(row.get("surface_id")): row
        for row in policy.get("rows") or []
        if isinstance(row, dict) and row.get("surface_id") is not None
    }
    records = []
    for case in selector.get("retained_source_cases") or []:
        if not isinstance(case, dict):
            continue
        case_key = str(case.get("case_key"))
        replay_case = replay_by_case.get(case_key)
        group = groups_by_case.get(case_key)
        for row in case.get("row_leaf_keys") or []:
            if isinstance(row, dict):
                records.append(row_record(case, row, replay_case, group, policy_rows))
    return sorted(
        records,
        key=lambda row: (
            str(row.get("target")),
            int(row.get("transfer_index") or 0),
            str(row.get("base_added_key")),
            str(row.get("case_key")),
            str(row.get("row_key")),
        ),
    )


RulePredicate = Callable[[dict[str, Any]], bool]


def make_rule(name: str, description: str, complexity: int, predicate: RulePredicate) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "complexity": complexity,
        "predicate": predicate,
    }


def candidate_rules(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for value in sorted({str(row.get("added_leaf_key")) for row in records}):
        rules.append(
            make_rule(
                f"added_leaf={value}",
                f"added leaf set is {value}",
                1,
                lambda row, value=value: str(row.get("added_leaf_key")) == value,
            )
        )
    for value in sorted({str(row.get("base_leaf_key")) for row in records}):
        rules.append(
            make_rule(
                f"base_leaf={value}",
                f"base leaf set is {value}",
                1,
                lambda row, value=value: str(row.get("base_leaf_key")) == value,
            )
        )
    for value in sorted({str(row.get("base_added_key")) for row in records}):
        rules.append(
            make_rule(
                f"base_added={value}",
                f"base leaf set to added leaf set is {value}",
                2,
                lambda row, value=value: str(row.get("base_added_key")) == value,
            )
        )
    for value in sorted({str(row.get("target")) for row in records}):
        rules.append(
            make_rule(
                f"target={value}",
                f"target is {value}",
                1,
                lambda row, value=value: str(row.get("target")) == value,
            )
        )
    for base_added in sorted({str(row.get("base_added_key")) for row in records}):
        for target in sorted({str(row.get("target")) for row in records}):
            rules.append(
                make_rule(
                    f"target={target};base_added={base_added}",
                    f"target is {target} and base/added leaf set is {base_added}",
                    3,
                    lambda row, target=target, base_added=base_added: str(row.get("target")) == target
                    and str(row.get("base_added_key")) == base_added,
                )
            )
    for base_added in sorted({str(row.get("base_added_key")) for row in records}):
        for top_k in sorted({int(row.get("top_k") or 0) for row in records}):
            rules.append(
                make_rule(
                    f"top_k={top_k};base_added={base_added}",
                    f"top_k is {top_k} and base/added leaf set is {base_added}",
                    3,
                    lambda row, top_k=top_k, base_added=base_added: int(row.get("top_k") or 0) == top_k
                    and str(row.get("base_added_key")) == base_added,
                )
            )
    for base_added in sorted({str(row.get("base_added_key")) for row in records}):
        for policy in sorted({str(row.get("policy")) for row in records}):
            rules.append(
                make_rule(
                    f"policy={policy};base_added={base_added}",
                    f"policy is {policy} and base/added leaf set is {base_added}",
                    3,
                    lambda row, policy=policy, base_added=base_added: str(row.get("policy")) == policy
                    and str(row.get("base_added_key")) == base_added,
                )
            )
    return rules


def evaluate_rule(
    rule: dict[str, Any],
    records: list[dict[str, Any]],
    replay_groups: list[dict[str, Any]],
) -> dict[str, Any] | None:
    selected = [row for row in records if rule["predicate"](row)]
    if not selected:
        return None
    selected_case_keys = sorted({str(row.get("case_key")) for row in selected})
    selected_case_key_set = set(selected_case_keys)
    covered_groups = [
        group
        for group in replay_groups
        if set(str(case_key) for case_key in group.get("case_keys") or []).issubset(selected_case_key_set)
    ]
    verified_groups = [group for group in covered_groups if group.get("public_key_verified")]
    selected_transfers = sorted(
        {
            f"{row.get('target')}|{row.get('transfer_index')}"
            for row in selected
        }
    )
    verified_transfers = sorted(
        {
            f"{group.get('target')}|{group.get('transfer_index')}"
            for group in verified_groups
        }
    )
    relation_rows = [row for row in selected if int(row.get("retained_row_relation_count") or 0) > 0]
    no_relation_rows = [row for row in selected if int(row.get("retained_row_relation_count") or 0) == 0]
    row_ratios = [
        float(row["retained_row_ops_over_rho"])
        for row in selected
        if row.get("retained_row_ops_over_rho") is not None
    ]
    return {
        "name": rule["name"],
        "description": rule["description"],
        "complexity": int(rule["complexity"]),
        "selected_row_count": len(selected),
        "selected_case_count": len(selected_case_keys),
        "selected_transfer_count": len(selected_transfers),
        "selected_transfers": selected_transfers,
        "selected_case_keys": selected_case_keys,
        "selected_row_relation_count_nonzero": len(relation_rows),
        "selected_row_no_relation_count": len(no_relation_rows),
        "selected_row_relation_precision": ratio((len(relation_rows), len(selected))),
        "selected_row_mean_ops_over_rho": mean_or_none(row_ratios),
        "covered_group_count": len(covered_groups),
        "covered_verified_group_count": len(verified_groups),
        "covered_verified_group_precision": ratio((len(verified_groups), len(covered_groups))),
        "verified_transfers": verified_transfers,
        "verified_derived_secrets": sorted(
            {
                int(group.get("derived_secret"))
                for group in verified_groups
                if group.get("derived_secret") is not None
            }
        ),
        "covered_groups": [
            {
                "target": group.get("target"),
                "transfer_index": int(group.get("transfer_index") or 0),
                "case_count": int(group.get("case_count") or 0),
                "relation_count": int(group.get("relation_count") or 0),
                "rank": int(group.get("rank") or 0),
                "public_key_verified": bool(group.get("public_key_verified")),
                "derived_secret": group.get("derived_secret"),
                "case_keys": group.get("case_keys") or [],
            }
            for group in covered_groups
        ],
    }


def rule_sort_key(result: dict[str, Any], total_verified_groups: int) -> tuple[Any, ...]:
    covers_all = int(int(result.get("covered_verified_group_count") or 0) == total_verified_groups)
    return (
        -covers_all,
        -int(result.get("covered_verified_group_count") or 0),
        int(result.get("selected_row_count") or 0),
        int(result.get("complexity") or 0),
        -int(result.get("selected_row_relation_count_nonzero") or 0),
        str(result.get("name")),
    )


def evaluate_rules(records: list[dict[str, Any]], replay_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_verified_groups = sum(1 for group in replay_groups if group.get("public_key_verified"))
    results = [
        result
        for rule in candidate_rules(records)
        for result in [evaluate_rule(rule, records, replay_groups)]
        if result is not None
    ]
    return sorted(results, key=lambda result: rule_sort_key(result, total_verified_groups))


def aggregate_counts(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get(field))].append(row)
    out = []
    for key, rows in sorted(grouped.items()):
        relation_rows = [row for row in rows if int(row.get("retained_row_relation_count") or 0) > 0]
        verified_groups = {
            row.get("challenge_group_key")
            for row in rows
            if row.get("challenge_group_public_key_verified")
        }
        out.append(
            {
                field: key,
                "row_count": len(rows),
                "case_count": len({str(row.get("case_key")) for row in rows}),
                "transfer_count": len({f"{row.get('target')}|{row.get('transfer_index')}" for row in rows}),
                "relation_row_count": len(relation_rows),
                "relation_row_precision": ratio((len(relation_rows), len(rows))),
                "verified_group_count": len(verified_groups),
            }
        )
    return out


def parse_trigger_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target": args.trigger_target,
        "base_leaf_indices": args.trigger_base_leaf,
        "added_leaf_indices": args.trigger_added_leaf,
        "top_k": args.trigger_top_k,
        "policy": args.trigger_policy,
        "leaf_selector": args.trigger_leaf_selector,
        "anchor_role": args.trigger_anchor_role,
    }


def matches_trigger(row: dict[str, Any], trigger: dict[str, Any]) -> bool:
    if trigger.get("target") and str(row.get("target")) != str(trigger["target"]):
        return False
    if trigger.get("top_k") is not None and int(row.get("top_k") or 0) != int(trigger["top_k"]):
        return False
    if trigger.get("policy") and str(row.get("policy")) != str(trigger["policy"]):
        return False
    if trigger.get("leaf_selector") and str(row.get("leaf_selector")) != str(trigger["leaf_selector"]):
        return False
    if trigger.get("base_leaf_indices"):
        wanted = sorted(int(leaf) for leaf in trigger["base_leaf_indices"])
        if sorted(int(leaf) for leaf in row.get("base_leaf_indices") or []) != wanted:
            return False
    if trigger.get("added_leaf_indices"):
        wanted = sorted(int(leaf) for leaf in trigger["added_leaf_indices"])
        if sorted(int(leaf) for leaf in row.get("added_leaf_indices") or []) != wanted:
            return False
    role = trigger.get("anchor_role")
    if role == "anchor" and not row.get("is_anchor_surface"):
        return False
    if role == "companion" and not row.get("is_companion_surface"):
        return False
    return True


def filtered_case_copy(case: dict[str, Any], selected_surfaces: set[str]) -> dict[str, Any] | None:
    rows = [
        row
        for row in case.get("row_leaf_keys") or []
        if isinstance(row, dict) and str(row.get("surface_id")) in selected_surfaces
    ]
    if not rows:
        return None
    out = dict(case)
    out["row_leaf_keys"] = rows
    out["marginal_lift_row_leaf_keys"] = rows
    out["surface_ids"] = sorted({str(row.get("surface_id")) for row in rows})
    out["marginal_lift_surface_ids"] = out["surface_ids"]
    out["anchor_surface_ids"] = sorted(set(out.get("anchor_surface_ids") or []).intersection(selected_surfaces))
    out["companion_surface_ids"] = sorted(set(out.get("companion_surface_ids") or []).intersection(selected_surfaces))
    out["trigger_retention_mode"] = "public_fourth_leaf_trigger_exact_profiles"
    return out


def emit_trigger_bridge(
    selector: dict[str, Any],
    records: list[dict[str, Any]],
    trigger: dict[str, Any],
) -> dict[str, Any]:
    selected_records = [row for row in records if matches_trigger(row, trigger)]
    selected_surfaces = {str(row.get("surface_id")) for row in selected_records}
    selected_case_keys = {str(row.get("case_key")) for row in selected_records}
    retained_source_cases = []
    for case in selector.get("retained_source_cases") or []:
        if not isinstance(case, dict) or str(case.get("case_key")) not in selected_case_keys:
            continue
        filtered = filtered_case_copy(case, selected_surfaces)
        if filtered is not None:
            retained_source_cases.append(filtered)
    retained_surfaces = [
        row
        for row in selector.get("retained_surfaces") or []
        if isinstance(row, dict) and str(row.get("surface_id")) in selected_surfaces
    ]
    known_surfaces = {str(row.get("surface_id")) for row in retained_surfaces}
    for surface_id in sorted(selected_surfaces - known_surfaces):
        sample = next(row for row in selected_records if str(row.get("surface_id")) == surface_id)
        retained_surfaces.append(
            {
                "surface_id": surface_id,
                "target": sample.get("target"),
                "transfer_index": sample.get("transfer_index"),
                "row_key": sample.get("row_key"),
                "salt": sample.get("salt"),
                "source": "trigger_selected_public_companion_surface",
            }
        )
    return {
        "schema": "ffe_public_fourth_leaf_trigger_bridge/v1",
        "method": "public_fourth_leaf_trigger_exact_profile_bridge",
        "parameters": {
            **(selector.get("parameters") or {}),
            "trigger": trigger,
        },
        "summary": {
            "bridge_status": "public_fourth_leaf_trigger_bridge_ready",
            "selected_row_count": len(selected_records),
            "selected_case_count": len(retained_source_cases),
            "selected_surface_count": len(selected_surfaces),
            "selected_challenge_count": len(
                {
                    f"{row.get('target')}|{row.get('transfer_index')}"
                    for row in selected_records
                }
            ),
            "next_obligation": "Replay this trigger bridge, then freeze the same public trigger before testing a fresh total3/total4 window.",
        },
        "retained_surfaces": sorted(retained_surfaces, key=lambda row: str(row.get("surface_id"))),
        "retained_source_cases": sorted(
            retained_source_cases,
            key=lambda row: (str(row.get("target")), int(row.get("transfer_index") or 0), str(row.get("case_key"))),
        ),
        "non_claims": [
            "This bridge is a public-field trigger applied to an already mined selector output.",
            "A fresh-window result is still required before claiming a general ECDLP index-calculus speedup.",
        ],
    }


def build_audit(
    selector: dict[str, Any],
    replay: dict[str, Any],
    policy: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = collect_row_records(selector, replay, policy)
    _groups_by_case, replay_groups = group_index(replay)
    rule_results = evaluate_rules(records, replay_groups)
    verified_groups = [group for group in replay_groups if group.get("public_key_verified")]
    trigger = parse_trigger_args(args)
    trigger_selected = [row for row in records if matches_trigger(row, trigger)] if any(trigger.values()) else []
    trigger_rule = None
    if trigger_selected:
        trigger_rule = evaluate_rule(
            {
                "name": "declared_trigger",
                "description": "explicit trigger CLI arguments",
                "complexity": sum(1 for value in trigger.values() if value),
                "predicate": lambda row: matches_trigger(row, trigger),
            },
            records,
            replay_groups,
        )
    return {
        "schema": "ffe_public_fourth_leaf_trigger_audit/v1",
        "method": "retrospective_public_feature_audit_for_exact_total4_marginal_lifts",
        "parameters": {
            "selector_source": str(args.selector_source),
            "replay_source": str(args.replay_source),
            "policy_source": str(args.policy_source),
            "trigger": trigger,
        },
        "summary": {
            "row_record_count": len(records),
            "case_count": len({str(row.get("case_key")) for row in records}),
            "challenge_count": len({f"{row.get('target')}|{row.get('transfer_index')}" for row in records}),
            "retained_relation_row_count": sum(
                1 for row in records if int(row.get("retained_row_relation_count") or 0) > 0
            ),
            "verified_challenge_group_count": len(verified_groups),
            "verified_challenge_groups": [
                {
                    "target": group.get("target"),
                    "transfer_index": int(group.get("transfer_index") or 0),
                    "case_count": int(group.get("case_count") or 0),
                    "rank": int(group.get("rank") or 0),
                    "relation_count": int(group.get("relation_count") or 0),
                    "derived_secret": group.get("derived_secret"),
                }
                for group in verified_groups
            ],
            "recommended_rule_name": rule_results[0]["name"] if rule_results else None,
            "recommended_rule_description": rule_results[0]["description"] if rule_results else None,
            "declared_trigger_selected_row_count": len(trigger_selected),
            "declared_trigger_rule": trigger_rule,
            "next_obligation": "Freeze the recommended public trigger and replay it on a fresh total3/total4 window beyond 200-207.",
        },
        "feature_summaries": {
            "by_base_added_key": aggregate_counts(records, "base_added_key"),
            "by_added_leaf_key": aggregate_counts(records, "added_leaf_key"),
            "by_base_leaf_key": aggregate_counts(records, "base_leaf_key"),
        },
        "rule_results": rule_results[:25],
        "row_records": records,
        "non_claims": [
            "The audit uses replay labels retrospectively to score public features.",
            "The recommended rule is not a fresh-window validation.",
            "The emitted trigger bridge, if requested, remains below-rho evidence only after replay verification.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-source", type=Path, default=DEFAULT_SELECTOR_SOURCE)
    parser.add_argument("--replay-source", type=Path, default=DEFAULT_REPLAY_SOURCE)
    parser.add_argument("--policy-source", type=Path, default=DEFAULT_POLICY_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trigger-bridge-out", type=Path)
    parser.add_argument("--trigger-target")
    parser.add_argument("--trigger-base-leaf", type=int, action="append", default=[])
    parser.add_argument("--trigger-added-leaf", type=int, action="append", default=[])
    parser.add_argument("--trigger-top-k", type=int)
    parser.add_argument("--trigger-policy")
    parser.add_argument("--trigger-leaf-selector")
    parser.add_argument(
        "--trigger-anchor-role",
        choices=["anchor", "companion"],
        help="Optionally require the marginal surface to be a root-policy anchor or a companion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selector = load_json(args.selector_source)
    replay = load_json(args.replay_source)
    policy = load_json(args.policy_source)
    audit = build_audit(selector, replay, policy, args)
    write_json(args.out, audit)
    print(json.dumps(audit["summary"], indent=2, sort_keys=True))
    if args.trigger_bridge_out:
        trigger = parse_trigger_args(args)
        bridge = emit_trigger_bridge(selector, audit["row_records"], trigger)
        write_json(args.trigger_bridge_out, bridge)
        print(json.dumps(bridge["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
