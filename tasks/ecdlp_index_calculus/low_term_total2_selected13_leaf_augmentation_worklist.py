#!/usr/bin/env python3
"""Build selected13 leaf-augmentation work items for fresh direct emission.

The direct-verification audit shows that the materialized selected13 target
rows still need fresh direct relations.  This script turns that audit into a
small replay worklist: exact selector variants, positive-control leaf unions,
and single-leaf augmentations seeded by nearby verified certificates.

The output is a worklist only.  It does not replay the verifier path, export a
relation, derive an ECDLP scalar, or claim a Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_leaf_augmentation_worklist.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_AUDIT = DEFAULT_STATE_DIR / "low_term_total2_selected13_direct_verification_audit_9981_9943_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_leaf_augmentation_worklist_9981_9943_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_leaf_augmentation_worklist_9981_9943_probe.h"
PRIMARY_TRANSFERS = [9981, 9943]

CLASS_CODES = {
    "exact_selector_replay": 1,
    "positive_control_leaf_union": 2,
    "single_leaf_positive_control_augmentation": 3,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def digest_u64(raw: Any) -> int:
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def all_objects(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(value, dict):
        out.append(value)
        for child in value.values():
            out.extend(all_objects(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(all_objects(child))
    return out


def row_leaf_map(raw: Any) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        if not row_key:
            continue
        out.setdefault(row_key, set()).update(as_int(leaf) for leaf in item.get("leaf_indices") or [])
    return out


def compact_row_leaf_map(raw: dict[str, set[int]]) -> list[dict[str, Any]]:
    return [
        {"leaf_indices": sorted(leaves), "row_key": row_key}
        for row_key, leaves in sorted(raw.items())
        if leaves
    ]


def leaf_count(raw: dict[str, set[int]]) -> int:
    return sum(len(leaves) for leaves in raw.values())


def row_key_tuple(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in (raw or [])))


def target_rows(source: dict[str, Any], target: str, transfer: int) -> list[dict[str, Any]]:
    rows = []
    for item in all_objects(source):
        if str(item.get("target") or "") != target:
            continue
        if as_int(item.get("transfer_index"), -1) != transfer:
            continue
        if not item.get("row_leaf_keys"):
            continue
        rows.append(item)
    rows.sort(
        key=lambda row: (
            as_float(row.get("ops_over_rho")) if as_float(row.get("ops_over_rho")) is not None else 999.0,
            -as_int(row.get("rank")),
            -as_int(row.get("relation_count")),
            str(row.get("selector") or row.get("row_selector") or ""),
            as_int(row.get("top_k")),
        )
    )
    return rows


def find_source_row(
    source: dict[str, Any],
    target: str,
    transfer: int,
    selector: str,
    top_k: int,
) -> dict[str, Any] | None:
    for row in target_rows(source, target, transfer):
        if str(row.get("selector") or row.get("row_selector") or "") != selector:
            continue
        if as_int(row.get("top_k"), -1) != top_k:
            continue
        return row
    return None


def union_leaf_maps(base: dict[str, set[int]], addition: dict[str, set[int]]) -> dict[str, set[int]]:
    out = {row_key: set(leaves) for row_key, leaves in base.items()}
    for row_key, leaves in addition.items():
        out.setdefault(row_key, set()).update(leaves)
    return out


def positional_control_union(
    base: dict[str, set[int]],
    control: dict[str, set[int]],
) -> tuple[dict[str, set[int]], dict[str, list[int]]]:
    base_keys = sorted(base)
    control_keys = sorted(control)
    out = {row_key: set(leaves) for row_key, leaves in base.items()}
    additions: dict[str, list[int]] = {}
    if not base_keys or not control_keys:
        return out, additions
    for index, row_key in enumerate(base_keys):
        control_key = control_keys[min(index, len(control_keys) - 1)]
        before = set(out[row_key])
        out[row_key].update(control.get(control_key, set()))
        added = sorted(out[row_key] - before)
        if added:
            additions[row_key] = added
    return out, additions


def diff_leaf_maps(candidate: dict[str, set[int]], base: dict[str, set[int]]) -> dict[str, list[int]]:
    additions = {}
    for row_key, leaves in candidate.items():
        added = sorted(set(leaves) - set(base.get(row_key, set())))
        if added:
            additions[row_key] = added
    return additions


def removed_leaf_maps(candidate: dict[str, set[int]], base: dict[str, set[int]]) -> dict[str, list[int]]:
    removed = {}
    for row_key, leaves in base.items():
        missing = sorted(set(leaves) - set(candidate.get(row_key, set())))
        if missing:
            removed[row_key] = missing
    return removed


def compact_source_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "below_rho": bool(row.get("below_rho")),
        "ops_over_rho": as_float(row.get("ops_over_rho")),
        "public_key_verified": bool(row.get("public_key_verified")),
        "rank": as_int(row.get("rank")),
        "relation_count": as_int(row.get("relation_count")),
        "row_keys": [str(item) for item in row.get("row_keys") or []],
        "selector": row.get("selector") or row.get("row_selector"),
        "top_k": as_int(row.get("top_k"), -1),
    }


def stable_work_item_id(material: dict[str, Any]) -> str:
    return f"leaf_aug_{digest_u64(material):016x}"


def build_work_item(
    target_audit: dict[str, Any],
    candidate_class: str,
    candidate_leaf_map: dict[str, set[int]],
    base_leaf_map: dict[str, set[int]],
    source_row: dict[str, Any] | None,
    reason: str,
    positive_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    additions = diff_leaf_maps(candidate_leaf_map, base_leaf_map)
    removals = removed_leaf_maps(candidate_leaf_map, base_leaf_map)
    transfer = as_int(target_audit.get("transfer_index"), -1)
    item = {
        "accepted_relation_export": False,
        "added_leaf_count": sum(len(leaves) for leaves in additions.values()),
        "added_leaf_indices_by_row": additions,
        "backfill_row_check_hash": target_audit.get("backfill_row_check_hash"),
        "backfill_row_id": target_audit.get("backfill_row_id"),
        "base_classification": target_audit.get("classification"),
        "base_relation_count": as_int((target_audit.get("exact_product_gate_rows") or [{}])[0].get("relation_count")),
        "base_rank": as_int((target_audit.get("exact_product_gate_rows") or [{}])[0].get("rank")),
        "candidate_class": candidate_class,
        "candidate_class_code": CLASS_CODES.get(candidate_class, 0),
        "candidate_row_leaf_keys": compact_row_leaf_map(candidate_leaf_map),
        "candidate_selected_leaf_count": leaf_count(candidate_leaf_map),
        "candidate_source_row": compact_source_row(source_row or {}),
        "fresh_direct_verification_required": True,
        "positive_control": positive_control,
        "reason": reason,
        "relation_derived_ecdlp": False,
        "removed_leaf_count": sum(len(leaves) for leaves in removals.values()),
        "removed_leaf_indices_by_row": removals,
        "target": target_audit.get("target"),
        "transfer_index": transfer,
        "worker_acceptance_gate": "fresh_direct_relation_replay_must_verify_public_key",
    }
    item["work_item_id"] = stable_work_item_id(
        {
            "candidate_class": candidate_class,
            "leaf_map": item["candidate_row_leaf_keys"],
            "positive_control": positive_control,
            "transfer": transfer,
        }
    )
    return item


def work_item_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    class_rank = {
        "exact_selector_replay": 0,
        "positive_control_leaf_union": 1,
        "single_leaf_positive_control_augmentation": 2,
    }.get(str(item.get("candidate_class")), 9)
    source = item.get("candidate_source_row") or {}
    transfer = as_int(item.get("transfer_index"), 999999)
    try:
        transfer_rank = PRIMARY_TRANSFERS.index(transfer)
    except ValueError:
        transfer_rank = len(PRIMARY_TRANSFERS)
    return (
        transfer_rank,
        transfer,
        class_rank,
        -as_int(source.get("rank")),
        -as_int(source.get("relation_count")),
        as_int(item.get("added_leaf_count")),
        as_int(item.get("candidate_selected_leaf_count")),
        as_float(source.get("ops_over_rho")) if as_float(source.get("ops_over_rho")) is not None else 999.0,
        str(item.get("work_item_id")),
    )


def dedupe_work_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for item in sorted(items, key=work_item_sort_key):
        leaf_signature = json.dumps(item.get("candidate_row_leaf_keys"), sort_keys=True)
        key = f"{item.get('transfer_index')}|{leaf_signature}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def build_target_items(
    target_audit: dict[str, Any],
    source: dict[str, Any],
    max_items: int,
) -> list[dict[str, Any]]:
    target = str(target_audit.get("target") or "")
    transfer = as_int(target_audit.get("transfer_index"), -1)
    exact_row = (target_audit.get("exact_source_policy_rows") or [{}])[0]
    baseline_selector = str(exact_row.get("selector") or "")
    baseline_top_k = as_int(exact_row.get("top_k"), -1)
    baseline_source_row = find_source_row(source, target, transfer, baseline_selector, baseline_top_k)
    if baseline_source_row is None:
        return []
    baseline_map = row_leaf_map(baseline_source_row.get("row_leaf_keys"))
    rows = target_rows(source, target, transfer)

    items: list[dict[str, Any]] = []
    for row in rows:
        if as_int(row.get("relation_count")) <= 0 and as_int(row.get("rank")) <= 0:
            continue
        candidate_map = row_leaf_map(row.get("row_leaf_keys"))
        items.append(
            build_work_item(
                target_audit,
                "exact_selector_replay",
                candidate_map,
                baseline_map,
                row,
                "Existing exact-transfer selector has verifier relation evidence; replay it against the materialized row identity and test independence.",
            )
        )

    for control in target_audit.get("positive_control_certificates") or []:
        control_transfer = as_int(control.get("transfer_index"), -1)
        control_selector = str(control.get("selector") or "")
        control_top_k = as_int(control.get("top_k"), -1)
        control_row = find_source_row(source, target, control_transfer, control_selector, control_top_k)
        if control_row is None:
            continue
        control_map = row_leaf_map(control_row.get("row_leaf_keys"))
        union_map, additions = positional_control_union(baseline_map, control_map)
        control_summary = {
            "derived_secret": control.get("derived_secret"),
            "public_key_verified": bool(control.get("public_key_verified")),
            "rank": as_int(control.get("rank")),
            "relation_count": as_int(control_row.get("relation_count")),
            "selector": control_selector,
            "top_k": control_top_k,
            "transfer_index": control_transfer,
        }
        if additions:
            items.append(
                build_work_item(
                    target_audit,
                    "positive_control_leaf_union",
                    union_map,
                    baseline_map,
                    control_row,
                    "Union the materialized row leaves with row-position-matched leaves from a same-target public-key-verified control.",
                    control_summary,
                )
            )
        for row_key, leaves in sorted(additions.items()):
            for leaf in leaves:
                single_map = {key: set(values) for key, values in baseline_map.items()}
                single_map.setdefault(row_key, set()).add(leaf)
                items.append(
                    build_work_item(
                        target_audit,
                        "single_leaf_positive_control_augmentation",
                        single_map,
                        baseline_map,
                        control_row,
                        f"Test positive-control leaf {leaf} on {row_key} as the smallest fresh-emission augmentation.",
                        control_summary,
                    )
                )
    return dedupe_work_items(items, max_items)


def validate_sources(audit: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if audit.get("claim_status") != "SELECTED13_DIRECT_VERIFICATION_REQUIRES_FRESH_RELATIONS":
        failures.append({"code": "direct_verification_audit_status_unexpected", "claim_status": audit.get("claim_status")})
    if audit.get("failures"):
        failures.append({"code": "direct_verification_audit_has_failures", "failures": audit.get("failures")})
    return failures


def render_c_header(work_items: list[dict[str, Any]]) -> str:
    rows = []
    for index, item in enumerate(work_items):
        rows.append(
            "  {"
            f"{as_int(item.get('transfer_index'))}ULL, "
            f"{index}ULL, "
            f"{digest_u64(item.get('work_item_id'))}ULL, "
            f"{as_int(item.get('candidate_class_code'))}ULL, "
            f"{as_int(item.get('base_rank'))}ULL, "
            f"{as_int(item.get('base_relation_count'))}ULL, "
            f"{as_int((item.get('candidate_source_row') or {}).get('rank'))}ULL, "
            f"{as_int((item.get('candidate_source_row') or {}).get('relation_count'))}ULL, "
            f"{as_int(item.get('candidate_selected_leaf_count'))}ULL, "
            f"{as_int(item.get('added_leaf_count'))}ULL, "
            f"{as_int(item.get('removed_leaf_count'))}ULL, "
            f"{1 if item.get('fresh_direct_verification_required') else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_LEAF_AUGMENTATION_WORKLIST_H
#define LOW_TERM_TOTAL2_SELECTED13_LEAF_AUGMENTATION_WORKLIST_H

#include <stdint.h>

#define SELECTED13_LEAF_AUGMENTATION_WORK_ITEM_COUNT {len(work_items)}
#define SELECTED13_LEAF_AUGMENTATION_RELATION_EXPORT_COUNT 0
#define SELECTED13_LEAF_AUGMENTATION_RELATION_DERIVED_ECDLP 0

#define SELECTED13_LEAF_AUG_CLASS_EXACT_SELECTOR_REPLAY 1ULL
#define SELECTED13_LEAF_AUG_CLASS_POSITIVE_CONTROL_UNION 2ULL
#define SELECTED13_LEAF_AUG_CLASS_SINGLE_LEAF 3ULL

typedef struct {{
  uint64_t transfer_index;
  uint64_t work_item_index;
  uint64_t work_item_id_u64;
  uint64_t candidate_class_code;
  uint64_t base_rank;
  uint64_t base_relation_count;
  uint64_t observed_rank;
  uint64_t observed_relation_count;
  uint64_t selected_leaf_count;
  uint64_t added_leaf_count;
  uint64_t removed_leaf_count;
  uint64_t fresh_direct_verification_required;
}} selected13_leaf_augmentation_work_item_t;

static const selected13_leaf_augmentation_work_item_t SELECTED13_LEAF_AUGMENTATION_WORK_ITEMS[] = {{
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
  uint64_t item_count =
      sizeof(SELECTED13_LEAF_AUGMENTATION_WORK_ITEMS) / sizeof(SELECTED13_LEAF_AUGMENTATION_WORK_ITEMS[0]);
  uint64_t fresh_required_count = 0;
  uint64_t exact_selector_count = 0;
  uint64_t positive_union_count = 0;
  uint64_t single_leaf_count = 0;

  if (item_count != SELECTED13_LEAF_AUGMENTATION_WORK_ITEM_COUNT) failure_count++;
  if (SELECTED13_LEAF_AUGMENTATION_RELATION_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_LEAF_AUGMENTATION_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (item_count == 0ULL) failure_count++;

  for (size_t i = 0; i < item_count; i++) {{
    const selected13_leaf_augmentation_work_item_t *item = &SELECTED13_LEAF_AUGMENTATION_WORK_ITEMS[i];
    fresh_required_count += item->fresh_direct_verification_required;
    if (item->candidate_class_code == SELECTED13_LEAF_AUG_CLASS_EXACT_SELECTOR_REPLAY) exact_selector_count++;
    if (item->candidate_class_code == SELECTED13_LEAF_AUG_CLASS_POSITIVE_CONTROL_UNION) positive_union_count++;
    if (item->candidate_class_code == SELECTED13_LEAF_AUG_CLASS_SINGLE_LEAF) single_leaf_count++;
    if (item->work_item_id_u64 == 0ULL) failure_count++;
    if (item->candidate_class_code == 0ULL) failure_count++;
    if (item->selected_leaf_count == 0ULL) failure_count++;
  }}

  printf("selected13_leaf_augmentation_preflight items=%llu fresh_required=%llu exact_selector=%llu positive_union=%llu single_leaf=%llu failures=%llu\\n",
         (unsigned long long)item_count,
         (unsigned long long)fresh_required_count,
         (unsigned long long)exact_selector_count,
         (unsigned long long)positive_union_count,
         (unsigned long long)single_leaf_count,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_leaf_aug_preflight_") as tmp:
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


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = Path(args.audit)
    audit = load_json(audit_path)
    failures = validate_sources(audit)
    work_items: list[dict[str, Any]] = []
    loaded_sources: dict[str, dict[str, Any]] = {}
    for target_audit in audit.get("target_audits") or []:
        if not isinstance(target_audit, dict):
            continue
        source_path = str((target_audit.get("source_artifacts") or {}).get("frontier_public_leaf_policy") or "")
        if not source_path:
            failures.append({"code": "source_policy_path_missing", "transfer_index": target_audit.get("transfer_index")})
            continue
        if source_path not in loaded_sources:
            loaded_sources[source_path] = load_json(Path(source_path)) if Path(source_path).exists() else {}
        if not loaded_sources[source_path]:
            failures.append({"code": "source_policy_missing", "path": source_path, "transfer_index": target_audit.get("transfer_index")})
            continue
        work_items.extend(build_target_items(target_audit, loaded_sources[source_path], args.max_items_per_target))

    work_items.sort(key=work_item_sort_key)
    class_counts = Counter(str(item.get("candidate_class")) for item in work_items)
    transfers = Counter(str(item.get("transfer_index")) for item in work_items)
    summary = {
        "accepted_relation_export_count": 0,
        "candidate_class_counts": dict(sorted(class_counts.items())),
        "failure_count": len(failures),
        "fresh_direct_verification_required_count": sum(1 for item in work_items if item.get("fresh_direct_verification_required")),
        "max_items_per_target": args.max_items_per_target,
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "target_transfer_counts": dict(sorted(transfers.items(), key=lambda item: int(item[0]))),
        "verified": not failures,
        "work_item_count": len(work_items),
        "worker_interpretation": (
            "Work items are leaf-set replay candidates for fresh FFE/direct relation emission. "
            "Exact selector variants reuse current target evidence; positive-control union and "
            "single-leaf items seed fresh augmentations from nearby verified certificates."
        ),
    }
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": "SELECTED13_LEAF_AUGMENTATION_WORKLIST_READY" if not failures and work_items else "SELECTED13_LEAF_AUGMENTATION_WORKLIST_FAILED",
        "parameters": {
            "audit": str(audit_path),
            "max_items_per_target": args.max_items_per_target,
        },
        "summary": summary,
        "work_items": work_items,
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "worklist_only": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--max-items-per-target", type=int, default=8)
    parser.add_argument("--out", default=str(DEFAULT_OUT), type=Path)
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["work_items"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = "SELECTED13_LEAF_AUGMENTATION_WORKLIST_FAILED"
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
