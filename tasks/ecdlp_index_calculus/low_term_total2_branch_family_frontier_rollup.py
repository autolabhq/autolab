#!/usr/bin/env python3
"""Roll up the branch-family duplicate-hit frontier.

The live AutoLab state advances in small JSON shards: branch streaming sweeps,
duplicate-hit certificate exports, rank-novelty gates, and occasional direct
source bridge certificates.  This script reads those certified artifacts and
produces a durable local frontier summary.  It does not recompute the algebra;
it keeps the mounted verifier outputs separate from the local handoff.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
SCHEMA = "ecdlp.low_term_total2_branch_family_frontier_rollup.v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def artifact_label(path: Path, state_dir: Path) -> str:
    return str(Path(state_dir.name) / path.name)


def json_paths(state_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in state_dir.glob(pattern)
        if path.is_file() and path.suffix == ".json" and not path.name.startswith("._")
    )


def parse_window(raw: Any) -> tuple[int, int] | None:
    if not isinstance(raw, str):
        return None
    match = re.fullmatch(r"(\d+)_(\d+)", raw)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def range_from_windows(windows: list[Any]) -> tuple[int, int] | None:
    parsed = [item for item in (parse_window(window) for window in windows) if item]
    if not parsed:
        return None
    return min(start for start, _end in parsed), max(end for _start, end in parsed)


def range_from_name(path: Path) -> tuple[int, int] | None:
    name = path.name
    for pattern in (
        r"_min(\d{4})_.*?_through(\d{4})",
        r"_val(\d{4})_.*?_through(\d{4})",
    ):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1)), int(match.group(2))
    pairs = re.findall(r"(?<!\d)(\d{4})_(\d{4})(?!\d)", name)
    if pairs:
        start, end = pairs[-1]
        return int(start), int(end)
    return None


def artifact_range(path: Path, payload: dict[str, Any] | None = None) -> tuple[int, int] | None:
    by_name = range_from_name(path)
    if by_name:
        return by_name
    if payload:
        groups = payload.get("groups") or []
        by_group = range_from_windows([group.get("window") for group in groups if isinstance(group, dict)])
        if by_group:
            return by_group
        by_sources = range_from_windows(payload.get("source_windows") or [])
        if by_sources:
            return by_sources
    return None


def range_label(item: tuple[int, int]) -> str:
    return f"{item[0]}_{item[1]}"


def latest_by_range(paths: list[Path], state_dir: Path) -> tuple[Path, dict[str, Any], tuple[int, int]] | None:
    ranked: list[tuple[tuple[int, int], Path, dict[str, Any]]] = []
    for path in paths:
        payload = load_json(path)
        item_range = artifact_range(path, payload)
        if item_range:
            ranked.append((item_range, path, payload))
    if not ranked:
        return None
    item_range, path, payload = max(ranked, key=lambda item: (item[0][1], item[0][0], item[1].name))
    return path, payload, item_range


def collect_by_range(state_dir: Path, pattern: str, start_min: int) -> dict[tuple[int, int], tuple[Path, dict[str, Any]]]:
    by_range: dict[tuple[int, int], tuple[Path, dict[str, Any]]] = {}
    for path in json_paths(state_dir, pattern):
        payload = load_json(path)
        item_range = artifact_range(path, payload)
        if not item_range or item_range[1] < start_min:
            continue
        current = by_range.get(item_range)
        if current is None or path.name > current[0].name:
            by_range[item_range] = (path, payload)
    return by_range


def model_metric(section: dict[str, Any], model: str, key: str) -> Any:
    return ((section.get("model_summaries") or {}).get(model) or {}).get(key)


def validation_metrics(payload: dict[str, Any], cost_key: str) -> dict[str, Any]:
    validation = (payload.get("summary") or {}).get("validation") or payload.get("validation") or {}
    return {
        "attempt_count": int_value(validation.get("attempt_count") or validation.get("attempts_scanned")),
        "context_count": int_value(validation.get("context_count")),
        "full_attempt_ops_over_rho": float_value(
            model_metric(validation, "family_full_attempt_setup_integer", "ops_over_rho")
        ),
        "full_attempt_stop_cost_over_rho": float_value(
            model_metric(validation, "family_full_attempt_setup_integer", cost_key)
        ),
        "full_context_ops_over_rho": float_value(
            model_metric(validation, "family_full_context_setup_integer", "ops_over_rho")
        ),
        "full_context_stop_cost_over_rho": float_value(
            model_metric(validation, "family_full_context_setup_integer", cost_key)
        ),
        "group_count": int_value(validation.get("group_count")),
        "hit_context_count": int_value(validation.get("hit_context_count")),
        "rho": int_value(validation.get("rho")),
        "selected_count": int_value(validation.get("selected_count")),
        "true_stop_count": int_value(validation.get("true_stop_count")),
        "verified_stop_count": int_value(validation.get("verified_stop_count") or validation.get("true_stop_count")),
    }


def summarize_streaming(path: Path, payload: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    return {
        "artifact": artifact_label(path, state_dir),
        "claim_status": payload.get("claim_status"),
        "validation": validation_metrics(payload, "cost_per_verified_stop_over_rho"),
    }


def summarize_rank_quality(path: Path, payload: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    return {
        "artifact": artifact_label(path, state_dir),
        "claim_status": payload.get("claim_status"),
        "validation": validation_metrics(payload, "cost_per_true_stop_over_rho"),
    }


def summarize_cert_export(path: Path, payload: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    return {
        "artifact": artifact_label(path, state_dir),
        "branch_counts": summary.get("branch_counts") or {},
        "certificate_count": int_value(summary.get("certificate_count")),
        "claim_status": payload.get("claim_status"),
        "passing_certificate_count": int_value(summary.get("passing_certificate_count")),
        "rejected_attempt_count": int_value(summary.get("rejected_attempt_count")),
        "windows": summary.get("windows") or [],
    }


def summarize_rank_novelty(path: Path, payload: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    return {
        "artifact": artifact_label(path, state_dir),
        "candidate_count": int_value(summary.get("candidate_count")),
        "claim_status": payload.get("claim_status"),
        "dependent_unique_gain_candidate_count": int_value(
            summary.get("dependent_unique_gain_candidate_count")
        ),
        "max_rank_gain": int_value(summary.get("max_rank_gain")),
        "rank_gain_candidate_count": int_value(summary.get("rank_gain_candidate_count")),
        "selected_count": int_value(summary.get("selected_count")),
        "selected_verified_rank2_count": int_value(summary.get("selected_verified_rank2_count")),
        "unique_gain_candidate_count": int_value(summary.get("unique_gain_candidate_count")),
        "verified_rank2_count": int_value(summary.get("verified_rank2_count")),
    }


def support_from_coeffs(coeffs: list[Any], order: int) -> tuple[int, ...]:
    return tuple(
        index
        for index, coeff in enumerate(coeffs[1:])
        if order and int_value(coeff) % order != 0
    )


def collect_certificate_rows(
    certificate_paths: list[Path], state_dir: Path, start_min: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in certificate_paths:
        payload = load_json(path)
        item_range = artifact_range(path, payload)
        if not item_range or item_range[1] < start_min:
            continue
        for offset, cert in enumerate(payload.get("certificates") or []):
            selected = cert.get("selected") or {}
            order = int_value(cert.get("order"))
            form_supports = sorted(
                {
                    support_from_coeffs(form.get("coeffs") or [], order)
                    for form in cert.get("forms") or []
                    if isinstance(form, dict)
                }
            )
            rows.append(
                {
                    "artifact": artifact_label(path, state_dir),
                    "artifact_offset": offset,
                    "branch_key": selected.get("branch_key") or cert.get("branch_key"),
                    "direct_ops_over_rho": float_value(selected.get("direct_ops_over_rho")),
                    "forms_count": int_value(cert.get("forms_count")),
                    "form_supports": [list(support) for support in form_supports],
                    "order": order,
                    "public_key_verified": bool(
                        cert.get("public_key_verified") or cert.get("upstream_public_key_verified")
                    ),
                    "rank": int_value(cert.get("rank")),
                    "range": range_label(item_range),
                    "target": selected.get("target"),
                    "transfer_index": int_value(selected.get("transfer_index")),
                    "window": cert.get("window") or selected.get("window"),
                }
            )
    return rows


def collect_rank_novelty_scores(
    rank_novelty: dict[tuple[int, int], tuple[Path, dict[str, Any]]],
    state_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_range, (path, payload) in sorted(rank_novelty.items()):
        for score in payload.get("scores") or []:
            relations = score.get("target_eliminated_relations") or []
            supports = sorted(
                {
                    tuple(relation.get("factor_support") or [])
                    for relation in relations
                    if relation.get("factor_support")
                }
            )
            rows.append(
                {
                    "artifact": artifact_label(path, state_dir),
                    "branch_key": score.get("branch_key"),
                    "claim_status": score.get("claim_status"),
                    "form_supports": [list(support) for support in supports],
                    "public_key_verified_rank2": bool(score.get("public_key_verified_rank2")),
                    "range": range_label(item_range),
                    "rank_gain": int_value(score.get("rank_gain")),
                    "transfer_index": int_value(score.get("transfer_index")),
                    "unique_factor_relation_gain": int_value(score.get("unique_factor_relation_gain")),
                    "window": score.get("window"),
                }
            )
    return rows


def score_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("window"),
        row.get("transfer_index"),
        row.get("branch_key"),
    )


def annotate_certificate_rows(
    cert_rows: list[dict[str, Any]], novelty_scores: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    score_by_key = {score_key(score): score for score in novelty_scores}
    annotated = []
    for row in cert_rows:
        score = score_by_key.get(score_key(row))
        annotated.append(
            {
                **row,
                "rank_frontier_claim": score.get("claim_status") if score else None,
                "rank_gain": int_value(score.get("rank_gain")) if score else None,
                "rank_novelty_scored": score is not None,
                "target_eliminated_supports": score.get("form_supports") if score else [],
                "unique_factor_relation_gain": int_value(score.get("unique_factor_relation_gain"))
                if score
                else None,
            }
        )
    return annotated


def classify_slice(
    streaming: dict[str, Any] | None,
    cert_export: dict[str, Any] | None,
    rank_novelty: dict[str, Any] | None,
) -> str:
    verified = int_value(((streaming or {}).get("validation") or {}).get("verified_stop_count"))
    certs = int_value((cert_export or {}).get("certificate_count"))
    novelty_candidates = int_value((rank_novelty or {}).get("candidate_count"))
    novelty_selected = int_value((rank_novelty or {}).get("selected_verified_rank2_count"))
    novelty_rank_gain = int_value((rank_novelty or {}).get("rank_gain_candidate_count"))
    if certs == 0 and verified == 0 and novelty_candidates == 0:
        return "NO_DUPLICATE_HIT_SUPPLY"
    if novelty_selected > 0 or novelty_rank_gain > 0:
        return "RANK_FRONTIER_GAIN"
    if certs > 0 and rank_novelty and novelty_rank_gain == 0:
        return "LOCAL_RECOVERY_RANK_DEPENDENT"
    if certs > 0:
        return "LOCAL_RECOVERY_RANK_FRONTIER_UNGRADED"
    return "BRANCH_HITS_WITHOUT_CERTIFICATES"


def collect_slices(
    state_dir: Path,
    start_min: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[tuple[int, int], tuple[Path, dict[str, Any]]]]]:
    streaming = collect_by_range(
        state_dir,
        "low_term_total2_branch_family_streaming_sweep_min*_pos16_24_forward*.json",
        start_min,
    )
    rank_quality = collect_by_range(
        state_dir,
        "low_term_total2_branch_family_rank_quality_gate_pos16_24_cal5135_val*_fixed4_forms2_through*_probe.json",
        start_min,
    )
    cert_exports = collect_by_range(
        state_dir,
        "low_term_total2_branch_family_duplicate_hit_relation_certificates_*_probe.json",
        start_min,
    )
    rank_novelty = collect_by_range(
        state_dir,
        "low_term_total2_branch_family_rank_novelty_gate_*_probe.json",
        start_min,
    )
    all_ranges = sorted(set(streaming) | set(rank_quality) | set(cert_exports) | set(rank_novelty))
    slices = []
    for item_range in all_ranges:
        stream_summary = (
            summarize_streaming(*streaming[item_range], state_dir) if item_range in streaming else None
        )
        rank_quality_summary = (
            summarize_rank_quality(*rank_quality[item_range], state_dir)
            if item_range in rank_quality
            else None
        )
        cert_summary = (
            summarize_cert_export(*cert_exports[item_range], state_dir)
            if item_range in cert_exports
            else None
        )
        novelty_summary = (
            summarize_rank_novelty(*rank_novelty[item_range], state_dir)
            if item_range in rank_novelty
            else None
        )
        slices.append(
            {
                "classification": classify_slice(stream_summary, cert_summary, novelty_summary),
                "certificate_export": cert_summary,
                "range": range_label(item_range),
                "rank_novelty": novelty_summary,
                "rank_quality": rank_quality_summary,
                "streaming": stream_summary,
            }
        )
    return slices, {
        "certificate_exports": cert_exports,
        "rank_novelty": rank_novelty,
        "rank_quality": rank_quality,
        "streaming": streaming,
    }


def unique_relation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    order_audits = payload.get("order_audits") or {}
    if not order_audits:
        return {}
    order, audit = sorted(order_audits.items())[0]
    samples = audit.get("unique_relation_samples") or []
    covered_columns = sorted(
        {
            int_value(column)
            for sample in samples
            for column in (sample.get("factor_support") or [])
        }
    )
    support_rows = [
        {
            "factor_support": sample.get("factor_support") or [],
            "multiplicity": int_value(sample.get("multiplicity")),
            "window": ((sample.get("example") or {}).get("window")),
            "transfer_index": int_value((sample.get("example") or {}).get("transfer_index")),
        }
        for sample in samples
    ]
    factor_count = int_value(audit.get("factor_variable_count"))
    uncovered_columns = [
        column for column in range(factor_count) if column not in set(covered_columns)
    ] if factor_count else []
    return {
        "order": order,
        "audit": {
            "certificate_count": int_value(audit.get("certificate_count")),
            "deficiency": int_value(audit.get("deficiency")),
            "factor_rank": int_value(audit.get("rank") or audit.get("augmented_rank")),
            "factor_variable_count": factor_count,
            "factor_variables_derivable_count": int_value(audit.get("factor_variables_derivable_count")),
            "full_rank": bool(audit.get("full_rank")),
            "matrix_consistent": bool(audit.get("matrix_consistent")),
            "unique_factor_relation_count": int_value(audit.get("unique_factor_relation_count")),
        },
        "covered_factor_columns": covered_columns,
        "support_rows": support_rows,
        "uncovered_factor_columns": uncovered_columns,
    }


def collect_latest_combined(state_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    specs = {
        "rank_candidate_scorer": "low_term_total2_branch_family_duplicate_hit_rank_candidate_scorer_5480_*_probe.json",
        "target_eliminated_rank": "low_term_total2_branch_family_duplicate_hit_target_eliminated_rank_5480_*_combined_probe.json",
        "linear_descent_audit": "low_term_total2_branch_family_duplicate_hit_linear_descent_audit_5480_*_combined_probe.json",
    }
    for key, pattern in specs.items():
        latest = latest_by_range(json_paths(state_dir, pattern), state_dir)
        if not latest:
            continue
        path, payload, item_range = latest
        entry = {
            "artifact": artifact_label(path, state_dir),
            "claim_status": payload.get("claim_status"),
            "range": range_label(item_range),
            "summary": payload.get("summary") or {},
        }
        if key == "target_eliminated_rank":
            entry["unique_relation_frontier"] = unique_relation_summary(payload)
        if key == "rank_candidate_scorer":
            entry["greedy_selected"] = payload.get("greedy_selected") or []
        artifacts[key] = entry
    return artifacts


def collect_direct_bridge(state_dir: Path) -> dict[str, Any] | None:
    latest = latest_by_range(
        json_paths(
            state_dir,
            "low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json",
        ),
        state_dir,
    )
    if not latest:
        return None
    cert_path, cert_payload, item_range = latest
    start, end = item_range
    scorer_latest = latest_by_range(
        json_paths(state_dir, f"*factor_rank_candidate_scorer*{start}_{end}*_probe.json"),
        state_dir,
    )
    target_latest = latest_by_range(
        json_paths(state_dir, f"*factor_column_target_scout*{start}_{end}*_probe.json"),
        state_dir,
    )
    cert_rows = collect_certificate_rows([cert_path], state_dir, start)
    scorer_entry: dict[str, Any] | None = None
    if scorer_latest:
        scorer_path, scorer_payload, _score_range = scorer_latest
        direct_scores = [
            score
            for score in scorer_payload.get("scores") or []
            if (score.get("candidate") or {}).get("artifact", "").endswith(cert_path.name)
        ]
        scorer_entry = {
            "artifact": artifact_label(scorer_path, state_dir),
            "claim_status": scorer_payload.get("claim_status"),
            "direct_certificate_scores": direct_scores,
            "greedy_selected": scorer_payload.get("greedy_selected") or [],
            "summary": scorer_payload.get("summary") or {},
        }
    target_entry: dict[str, Any] | None = None
    if target_latest:
        target_path, target_payload, _target_range = target_latest
        target_entry = {
            "artifact": artifact_label(target_path, state_dir),
            "claim_status": target_payload.get("claim_status"),
            "summary": target_payload.get("summary") or {},
            "work_orders": ((target_payload.get("summary") or {}).get("work_orders") or {}),
        }
    return {
        "certificate_artifact": artifact_label(cert_path, state_dir),
        "certificate_rows": cert_rows,
        "claim_status": cert_payload.get("claim_status"),
        "range": range_label(item_range),
        "rank_candidate_scorer": scorer_entry,
        "summary": cert_payload.get("summary") or {},
        "target_column_scout": target_entry,
    }


def summarize_totals(
    slices: list[dict[str, Any]],
    annotated_certificates: list[dict[str, Any]],
    combined: dict[str, Any],
    direct_bridge: dict[str, Any] | None,
) -> dict[str, Any]:
    classification_counts = Counter(slice_item["classification"] for slice_item in slices)
    direct_scores = (
        ((direct_bridge or {}).get("rank_candidate_scorer") or {}).get("direct_certificate_scores") or []
    )
    direct_rank_gain = sum(int_value(score.get("rank_gain")) for score in direct_scores)
    latest_branch = slices[-1] if slices else {}
    target_frontier = (
        (combined.get("target_eliminated_rank") or {}).get("unique_relation_frontier") or {}
    )
    target_audit = target_frontier.get("audit") or {}
    return {
        "branch_bank_certificate_count": int_value(target_audit.get("certificate_count")),
        "branch_bank_deficiency": int_value(target_audit.get("deficiency")),
        "branch_bank_factor_rank": int_value(target_audit.get("factor_rank")),
        "branch_bank_unique_factor_relation_count": int_value(
            target_audit.get("unique_factor_relation_count")
        ),
        "branch_bank_uncovered_factor_columns": target_frontier.get("uncovered_factor_columns") or [],
        "classification_counts": dict(sorted(classification_counts.items())),
        "direct_bridge_latest_rank_gain": direct_rank_gain,
        "direct_bridge_latest_range": (direct_bridge or {}).get("range"),
        "latest_branch_range": latest_branch.get("range"),
        "latest_branch_classification": latest_branch.get("classification"),
        "local_certificate_count": len(annotated_certificates),
        "rank_novel_certificate_count": sum(
            1 for row in annotated_certificates if int_value(row.get("rank_gain")) > 0
        ),
        "target_descent_implemented": False,
    }


def build_next_work_orders(
    combined: dict[str, Any],
    direct_bridge: dict[str, Any] | None,
    latest_range: str | None,
) -> list[dict[str, Any]]:
    target_frontier = (
        (combined.get("target_eliminated_rank") or {}).get("unique_relation_frontier") or {}
    )
    uncovered = target_frontier.get("uncovered_factor_columns") or []
    direct_orders = (
        ((direct_bridge or {}).get("target_column_scout") or {}).get("work_orders") or {}
    )
    work_orders: list[dict[str, Any]] = []
    if latest_range:
        work_orders.append(
            {
                "action": "VALIDATE_NEXT_SOURCE_SLICE",
                "detail": f"Continue branch-family validation after {latest_range} as source artifacts arrive.",
            }
        )
    if uncovered:
        work_orders.append(
            {
                "action": "PRE_SETUP_RANK_NOVELTY_TARGETING",
                "detail": "Prefer public source selectors that touch currently uncovered factor columns before full branch setup.",
                "priority_columns": uncovered,
            }
        )
    for order, payload in sorted(direct_orders.items()):
        work_orders.append(
            {
                "action": payload.get("action"),
                "detail": payload.get("rationale"),
                "order": order,
                "priority_columns": payload.get("priority_columns") or [],
                "source": "direct_bridge_target_column_scout",
            }
        )
    work_orders.append(
        {
            "action": "FRESH_TARGET_MAPPING_TEST",
            "detail": "Test whether a fresh public target can be mapped into the existing order-specific factor bank; current local recoveries are not target descent.",
        }
    )
    return work_orders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR, type=Path)
    parser.add_argument("--start-min", default=5480, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_dir = args.state_dir
    slices, artifact_maps = collect_slices(state_dir, args.start_min)
    cert_paths = [path for path, _payload in artifact_maps["certificate_exports"].values()]
    cert_rows = collect_certificate_rows(cert_paths, state_dir, args.start_min)
    novelty_scores = collect_rank_novelty_scores(artifact_maps["rank_novelty"], state_dir)
    annotated_certificates = annotate_certificate_rows(cert_rows, novelty_scores)
    combined = collect_latest_combined(state_dir)
    direct_bridge = collect_direct_bridge(state_dir)
    summary = summarize_totals(slices, annotated_certificates, combined, direct_bridge)
    latest_range = summary.get("latest_branch_range")
    payload = {
        "branch_certificate_rows": annotated_certificates,
        "claim_status": (
            "FRONTIER_HAS_LOCAL_RELATIONS_TARGET_DESCENT_OPEN"
            if summary["branch_bank_certificate_count"]
            else "FRONTIER_HAS_NO_BRANCH_RELATION_BANK"
        ),
        "combined_branch_bank": combined,
        "created_at": now_iso(),
        "direct_bridge": direct_bridge,
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "MODEL-BOUND: this rollup trusts mounted verifier artifacts and does not recompute relation algebra.",
            "The branch-family bank contains local relation certificates, not a complete target-descent algorithm.",
            "Any below-rho local stop remains a precursor until source-density, factor-rank, and fresh-target mapping costs beat Pollard rho end to end.",
        ],
        "next_work_orders": build_next_work_orders(combined, direct_bridge, latest_range),
        "parameters": {
            "start_min": args.start_min,
            "state_dir": str(state_dir),
        },
        "schema": SCHEMA,
        "slices": slices,
        "summary": summary,
    }
    write_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps({"claim_status": payload["claim_status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
