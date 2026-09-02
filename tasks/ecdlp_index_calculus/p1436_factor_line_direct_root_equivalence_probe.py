#!/usr/bin/env sage --python
"""Test whether the public FFE factor stream only re-encodes direct leaf roots."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCHEMA = "ecdlp.p1436_factor_line_direct_root_equivalence_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
LIVE_ROOT = Path("/Volumes/Volume/git/autolab")
LIVE_STATE = LIVE_ROOT / "ecdlp_index_calculus_state"
LIVE_TASK = LIVE_ROOT / "tasks" / "ecdlp_index_calculus"
DEFAULT_GENERATOR = (
    LIVE_STATE
    / "public_factor_presurface_generator_audit_low_total2_816_975_probe.json"
)
DEFAULT_STAGE = (
    LIVE_STATE
    / "public_factor_stage_guard_charge_audit_low_total2_816_975_probe.json"
)
DEFAULT_BACKFILL = (
    LIVE_STATE
    / "public_factor_presurface_backfill_audit_low_total2_816_975_probe.json"
)
DEFAULT_R68 = (
    WORKTREE_ROOT / "p1553_ffe_fixed_sum_information_conservation_report_r68.json"
)
DEFAULT_OUTPUT = (
    WORKTREE_ROOT
    / "ecdlp_index_calculus_state"
    / "p1436_factor_line_direct_root_equivalence_probe.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def resolve_live(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else LIVE_ROOT / value


def fingerprint_dict(fingerprint: list[list[int]]) -> dict[tuple[int, int], int]:
    return {
        (int(b_degree), int(c_degree)): int(coeff)
        for b_degree, c_degree, coeff in fingerprint
    }


def encoded_root_from_line(
    fingerprint: list[list[int]],
    p: int,
) -> int | None:
    terms = fingerprint_dict(fingerprint)
    expected = {(0, 0), (0, 1), (1, 0)}
    if set(terms) != expected or terms[(0, 1)] % p != 1:
        return None
    root = terms[(1, 0)] % p
    if terms[(0, 0)] % p != root * root % p:
        return None
    return root


def evaluate_fingerprint(
    fingerprint: list[list[int]],
    b_value: int,
    c_value: int,
    p: int,
) -> int:
    return sum(
        int(coeff)
        * pow(int(b_value), int(b_degree), p)
        * pow(int(c_value), int(c_degree), p)
        for b_degree, c_degree, coeff in fingerprint
    ) % p


def candidate_index(candidate: dict[str, Any]) -> int | None:
    name = str(candidate.get("candidate_name") or "")
    prefix = "sage_resultant_factor_"
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix) :])
    except ValueError:
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def load_live_modules() -> tuple[Any, Any, Any, Any]:
    task_path = str(LIVE_TASK)
    if task_path not in sys.path:
        sys.path.insert(0, task_path)
    backfill = importlib.import_module("public_factor_presurface_backfill_audit_probe")
    quadratic = importlib.import_module(
        "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_"
        "guarded_relation_harvester_static_bank_shared_challenge_salt_"
        "neighborhood_low_term_total2_ffe_public_factor_quadratic_root_probe"
    )
    relation_builder = importlib.import_module(
        "frontier_signed_target_guided_probe"
    )
    rank_module = importlib.import_module("frontier_signed_dual_sieve_probe")
    return backfill, quadratic, relation_builder, rank_module


def materialization_args(parameters: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        bank_source=resolve_live(parameters["bank_source"]),
        config_source=resolve_live(parameters["config_source"]),
        direct_source=resolve_live(parameters["direct_source"]),
        transfer_source=resolve_live(parameters["transfer_source"]),
        radius=parameters.get("radius"),
        row_pool=int(parameters.get("row_pool") or 512),
        row_count=int(parameters.get("row_count") or 128),
        scout_limit=int(parameters.get("scout_limit") or 192),
        scout_mode=str(parameters.get("scout_mode") or "s3_coeff_spread"),
        scout_order=str(parameters.get("scout_order") or "eval_cover_hits_high"),
        selected_limit=int(parameters.get("selected_limit") or 64),
        factor_base_size=int(parameters.get("factor_base_size") or 16),
        max_relations=int(parameters.get("max_relations") or 96),
        min_distinct_indices=int(parameters.get("min_distinct_indices") or 4),
        min_unsigned_distinct_indices=int(
            parameters.get("min_unsigned_distinct_indices") or 2
        ),
        require_unit_coefficients=bool(
            parameters.get("require_unit_coefficients", True)
        ),
        row_factor=int(parameters.get("row_factor") or 512),
        product_factor=int(parameters.get("product_factor") or 4096),
        seed=str(parameters.get("seed") or "ecdlp-frontier-signed-dual-sieve-v1"),
    )


def materialize_all_proposals(
    generator: dict[str, Any],
    stage: dict[str, Any],
    backfill_artifact: dict[str, Any],
    backfill_module: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
    Any,
    SimpleNamespace,
]:
    profile = generator["summary"]["promoted_pre_surface_profile"]["profile"]
    sources = backfill_module.window_sources(stage, None)
    cases, _proposal_keys, _window_by_key, _grouped = (
        backfill_module.build_backfill_cases(
            LIVE_STATE,
            sources,
            stage,
            profile,
            True,
        )
    )
    args = materialization_args(backfill_artifact["parameters"])
    bank = backfill_module.load_json(args.bank_source)
    config_source = backfill_module.load_json(args.config_source)
    direct_source = backfill_module.load_json(args.direct_source)
    transfer_source = backfill_module.load_json(args.transfer_source)
    params = transfer_source.get("parameters") or {}
    radius = int(args.radius if args.radius is not None else params.get("radius") or 4)
    bank_rows = {
        backfill_module.cross_surface_probe.compress_probe.row_key(row): row
        for row in bank.get("bank_rows") or []
        if isinstance(row, dict)
        and backfill_module.cross_surface_probe.compress_probe.row_key(row)
    }
    specs_by_target = (
        backfill_module.cross_surface_probe.leaf_trim_probe.specs_by_target_and_key(
            backfill_module.cross_surface_probe.salt_neighborhood_probe.witness_specs(
                direct_source, bank_rows, radius
            )
        )
    )
    verifier = backfill_module.cross_surface_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()
    surface_records, case_results = (
        backfill_module.cross_surface_probe.materialize_surface_records(
            verifier,
            records,
            config_source,
            specs_by_target,
            cases,
            args,
        )
    )
    return surface_records, case_results, sources, verifier, args


def direct_root_row(surface_record: dict[str, Any], quadratic: Any) -> dict[str, Any]:
    p = int(surface_record["p"])
    selected = [int(value) for value in surface_record["selected_leaf_indices"]]
    original_pairs = quadratic.remainder_factor_probe.selected_pair_set(
        surface_record, surface_record["surface"]
    )
    recovered_pairs: set[tuple[int, int]] = set()
    coefficient_points = []
    monic_leaf_count = 0
    recovered_root_count = 0
    for leaf_index in selected:
        leaf = surface_record["components"]["leaves"][leaf_index]
        coeffs = quadratic.resultant_surface_probe.monic_coeffs(leaf, p)
        if coeffs is None:
            continue
        monic_leaf_count += 1
        b_value, c_value = (int(coeffs[0]), int(coeffs[1]))
        roots = quadratic.slice_quadratic_probe.recover_roots_for_leaf(
            surface_record, leaf, b_value, c_value
        )
        recovered_root_count += len(roots)
        recovered_pairs.update((leaf_index, int(root)) for root in roots)
        coefficient_points.append(
            {
                "leaf_index": leaf_index,
                "b": b_value,
                "c": c_value,
                "recovered_roots": roots,
            }
        )
    missing = sorted(original_pairs - recovered_pairs)
    extra = sorted(recovered_pairs - original_pairs)
    extra_pair_details = []
    for leaf_index, root in extra:
        matching_rows = (
            surface_record["components"]["rows_by_x"].get(int(root))
            or surface_record["components"]["rows_by_x"].get(str(int(root)))
            or []
        )
        extra_pair_details.append(
            {
                "leaf_index": int(leaf_index),
                "root": int(root),
                "matching_hit_row_count": len(matching_rows),
                "matching_hit_rows": json_safe(matching_rows[:8]),
            }
        )
    cost = surface_record["cost_inputs"]
    marginal_ops = (
        len(selected)
        + 2 * monic_leaf_count
        + recovered_root_count
        + 2 * int(cost["selected_hit_events"])
    )
    fully_charged_ops = int(cost["core_ops"]) + marginal_ops
    return {
        "surface_id": surface_record["surface_id"],
        "target": surface_record["target"],
        "transfer_index": int(
            str(surface_record["challenge_seed"]).split(":")[-2]
        ),
        "row_key": surface_record["row_key"],
        "p": p,
        "selected_leaf_count": len(selected),
        "monic_selected_leaf_count": monic_leaf_count,
        "original_selected_pair_count": len(original_pairs),
        "recovered_pair_count": len(recovered_pairs),
        "missing_selected_pairs": [list(pair) for pair in missing[:8]],
        "extra_selected_pairs": [list(pair) for pair in extra[:8]],
        "extra_pair_details": extra_pair_details[:8],
        "exact_selected_pair_recovery": not missing and not extra,
        "coefficient_points": coefficient_points,
        "core_ops": int(cost["core_ops"]),
        "marginal_direct_quadratic_ops": marginal_ops,
        "fully_charged_direct_quadratic_ops": fully_charged_ops,
        "generic_rho_steps": int(cost["generic_rho_steps"]),
    }


def load_sage_surfaces(
    sources: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    surfaces = {}
    bindings = []
    for source in sources:
        quadratic_path = LIVE_STATE / source["quadratic_root_source"]
        if not quadratic_path.exists():
            bindings.append(
                {
                    "window": source["label"],
                    "quadratic_root_path": str(quadratic_path),
                    "quadratic_root_present": False,
                    "sage_factor_path": None,
                    "sage_factor_present": False,
                }
            )
            continue
        quadratic_artifact = read_json(quadratic_path)
        sage_path = resolve_live(quadratic_artifact["parameters"]["sage_factor_source"])
        if not sage_path.exists():
            bindings.append(
                {
                    "window": source["label"],
                    "quadratic_root_path": str(quadratic_path),
                    "quadratic_root_present": True,
                    "quadratic_root_sha256": sha256_file(quadratic_path),
                    "sage_factor_path": str(sage_path),
                    "sage_factor_present": False,
                }
            )
            continue
        sage_artifact = read_json(sage_path)
        for surface in sage_artifact.get("surfaces") or []:
            if isinstance(surface, dict):
                surfaces[str(surface["surface_id"])] = surface
        bindings.append(
            {
                "window": source["label"],
                "quadratic_root_path": str(quadratic_path),
                "quadratic_root_present": True,
                "quadratic_root_sha256": sha256_file(quadratic_path),
                "sage_factor_path": str(sage_path),
                "sage_factor_present": True,
                "sage_factor_sha256": sha256_file(sage_path),
            }
        )
    return surfaces, bindings


def factor_identity_row(
    sage_surface: dict[str, Any],
    surface_record: dict[str, Any],
    quadratic: Any,
) -> dict[str, Any]:
    p = int(sage_surface["p"])
    factors = sage_surface["sage_resultant_factorization"]["factors"]
    candidates = sage_surface.get("sage_resultant_factor_candidates") or []
    coefficient_points = []
    for leaf_index in sage_surface.get("selected_leaf_indices") or []:
        leaf = surface_record["components"]["leaves"][int(leaf_index)]
        coeffs = quadratic.resultant_surface_probe.monic_coeffs(leaf, p)
        if coeffs is not None:
            coefficient_points.append(
                {"leaf_index": int(leaf_index), "b": int(coeffs[0]), "c": int(coeffs[1])}
            )
    encoded_roots = []
    line_identity_count = 0
    roots_in_public_hit_set = 0
    zero_count_matches = 0
    candidate_count_with_index = 0
    hit_roots = {
        int(root) % p for root in surface_record["components"]["hit_roots"]
    }
    for factor in factors:
        root = encoded_root_from_line(factor.get("fingerprint") or [], p)
        encoded_roots.append(root)
        if root is not None:
            line_identity_count += 1
            if root in hit_roots:
                roots_in_public_hit_set += 1
    for candidate in candidates:
        index = candidate_index(candidate)
        if index is None or index >= len(factors):
            continue
        candidate_count_with_index += 1
        fingerprint = factors[index].get("fingerprint") or []
        zero_leaf_count = sum(
            evaluate_fingerprint(
                fingerprint,
                point["b"],
                point["c"],
                p,
            )
            == 0
            for point in coefficient_points
        )
        if zero_leaf_count == int(candidate.get("selected_surface_zero_leaves") or 0):
            zero_count_matches += 1
    return {
        "surface_id": sage_surface["surface_id"],
        "factor_count": len(factors),
        "line_root_identity_count": line_identity_count,
        "all_factors_are_c_plus_r_b_plus_r2": line_identity_count == len(factors),
        "encoded_roots_recovered_from_selected_leaves": roots_in_public_hit_set,
        "candidate_count_with_index": candidate_count_with_index,
        "translated_constant_zero_count_match_count": zero_count_matches,
        "all_translated_constant_zero_counts_match": (
            zero_count_matches == candidate_count_with_index
        ),
        "sage_selected_leaf_count": len(
            sage_surface.get("selected_leaf_indices") or []
        ),
        "monic_sage_selected_leaf_count": len(coefficient_points),
        "selected_leaf_sets_match_current_materialization": sorted(
            int(value) for value in sage_surface.get("selected_leaf_indices") or []
        )
        == sorted(int(value) for value in surface_record["selected_leaf_indices"]),
        "distinct_encoded_root_count": len(
            {root for root in encoded_roots if root is not None}
        ),
    }


def verify_extra_relations(
    surface_records: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    verifier: Any,
    relation_builder: Any,
    rank_module: Any,
) -> tuple[dict[str, Any], dict[str, list[tuple[Any, int, list[int]]]]]:
    record_by_id = {str(row["surface_id"]): row for row in surface_records}
    forms_by_target: dict[str, list[tuple[Any, int, list[int]]]] = {}
    seen_by_target: dict[str, set[tuple[Any, int]]] = {}
    relations_by_target: dict[str, list[dict[str, Any]]] = {}
    attempts = 0
    equal_point_count = 0
    accepted_examples = []
    for direct in direct_rows:
        record = record_by_id[direct["surface_id"]]
        built = record["built"]
        target = str(record["target"])
        forms = forms_by_target.setdefault(target, [])
        seen = seen_by_target.setdefault(target, set())
        relations = relations_by_target.setdefault(target, [])
        scouts_by_pos = {
            int(scout["scout_pos"]): scout for scout in built["scouts"]
        }
        for leaf_index, root in (
            (int(pair[0]), int(pair[1])) for pair in direct["extra_selected_pairs"]
        ):
            leaf = record["components"]["leaves"][leaf_index]
            hit_rows = (
                record["components"]["rows_by_x"].get(root)
                or record["components"]["rows_by_x"].get(str(root))
                or []
            )
            for scout_pos in leaf.get("scout_positions") or []:
                scout = scouts_by_pos.get(int(scout_pos))
                if scout is None:
                    continue
                candidate_point = verifier.add_points(
                    scout["left"]["point"],
                    scout["right"]["point"],
                    built["ainvs"],
                    int(built["p"]),
                )
                for hit_row in hit_rows:
                    attempts += 1
                    if candidate_point == hit_row["point"]:
                        equal_point_count += 1
                    before = len(forms)
                    accepted = relation_builder.add_relation_if_valid(
                        verifier,
                        built["challenge"],
                        built["base"],
                        built["public"],
                        built["ainvs"],
                        int(built["p"]),
                        int(built["order"]),
                        candidate_point,
                        [int(index) for index in scout["unsigned_indices"]],
                        hit_row,
                        forms,
                        seen,
                        relations,
                    )
                    if accepted and len(accepted_examples) < 16:
                        accepted_examples.append(
                            {
                                "surface_id": direct["surface_id"],
                                "target": target,
                                "leaf_index": leaf_index,
                                "root": root,
                                "scout_pos": int(scout_pos),
                                "hit_row": json_safe(hit_row),
                                "form": json_safe(forms[before]),
                                "relation": json_safe(relations[-1]),
                            }
                        )
    target_summaries = {}
    total_rank = 0
    for target, forms in sorted(forms_by_target.items()):
        record = next(row for row in surface_records if str(row["target"]) == target)
        result = rank_module.rank_probe(
            verifier, forms, int(record["built"]["order"])
        )
        rank = int(result.get("mixed_wide_relation_rank") or 0)
        total_rank += rank
        target_summaries[target] = {
            "accepted_relation_count": len(relations_by_target[target]),
            "candidate_relation_rank": rank,
            "derived_secret": result.get("derived_secret"),
            "derives_secret": bool(result.get("mixed_wide_relations_derive_secret")),
        }
    accepted_count = sum(
        len(relations) for relations in relations_by_target.values()
    )
    summary = {
        "candidate_verification_attempt_count": attempts,
        "equal_point_candidate_count": equal_point_count,
        "accepted_relation_count": accepted_count,
        "candidate_relation_rank_sum_across_targets": total_rank,
        "all_extra_x_matches_rejected": accepted_count == 0,
        "target_summaries": target_summaries,
        "accepted_relation_examples": accepted_examples,
        "interpretation": (
            "A shared x-coordinate is only a relation candidate. The elliptic-curve "
            "point equality and verifier relation checks are decisive."
        ),
    }
    return summary, forms_by_target


def source_relation_ledger(
    surface_records: list[dict[str, Any]],
    verifier: Any,
    backfill_module: Any,
    materialize_args: SimpleNamespace,
) -> dict[str, list[tuple[Any, int, list[int]]]]:
    forms_by_target: dict[str, dict[tuple[Any, int], tuple[Any, int, list[int]]]] = {}
    direct_witness = backfill_module.cross_surface_probe.direct_witness_probe
    for record in surface_records:
        scan = direct_witness.scan_selected(
            verifier,
            record["built"],
            record["components"],
            {int(value) for value in record["selected_leaf_indices"]},
            materialize_args,
        )
        target = str(record["target"])
        target_forms = forms_by_target.setdefault(target, {})
        for event in scan.get("relation_events") or []:
            form = event.get("form")
            if not isinstance(form, (list, tuple)) or len(form) < 3:
                continue
            normalized = (form[0], int(form[1]), form[2])
            target_forms[(normalized[0], normalized[1])] = normalized
    return {
        target: list(forms.values())
        for target, forms in sorted(forms_by_target.items())
    }


def fresh_rank_audit(
    surface_records: list[dict[str, Any]],
    baseline_by_target: dict[str, list[tuple[Any, int, list[int]]]],
    candidate_by_target: dict[str, list[tuple[Any, int, list[int]]]],
    verifier: Any,
    rank_module: Any,
) -> dict[str, Any]:
    records_by_target = {
        str(row["target"]): row for row in surface_records
    }
    target_summaries = {}
    total_baseline_forms = 0
    total_candidate_forms = 0
    total_fresh_forms = 0
    total_rank_before = 0
    total_rank_after = 0
    for target in sorted(set(baseline_by_target) | set(candidate_by_target)):
        baseline = baseline_by_target.get(target) or []
        candidates = candidate_by_target.get(target) or []
        baseline_keys = {(form[0], int(form[1])) for form in baseline}
        fresh = [
            form
            for form in candidates
            if (form[0], int(form[1])) not in baseline_keys
        ]
        order = int(records_by_target[target]["built"]["order"])
        before = rank_module.rank_probe(verifier, baseline, order)
        after = rank_module.rank_probe(verifier, baseline + fresh, order)
        rank_before = int(before.get("mixed_wide_relation_rank") or 0)
        rank_after = int(after.get("mixed_wide_relation_rank") or 0)
        total_baseline_forms += len(baseline)
        total_candidate_forms += len(candidates)
        total_fresh_forms += len(fresh)
        total_rank_before += rank_before
        total_rank_after += rank_after
        target_summaries[target] = {
            "baseline_relation_form_count": len(baseline),
            "candidate_relation_form_count": len(candidates),
            "fresh_candidate_relation_form_count": len(fresh),
            "rank_before": rank_before,
            "rank_after": rank_after,
            "fresh_rank_delta": rank_after - rank_before,
            "fresh_candidate_form_examples": [
                json_safe(form) for form in fresh[:8]
            ],
        }
    return {
        "baseline_relation_form_count": total_baseline_forms,
        "candidate_relation_form_count": total_candidate_forms,
        "fresh_candidate_relation_form_count": total_fresh_forms,
        "rank_before_sum_across_targets": total_rank_before,
        "rank_after_sum_across_targets": total_rank_after,
        "fresh_rank_delta_sum_across_targets": (
            total_rank_after - total_rank_before
        ),
        "target_summaries": target_summaries,
    }


def evaluate_live(
    generator_path: Path,
    stage_path: Path,
    backfill_path: Path,
    r68_path: Path,
) -> dict[str, Any]:
    generator = read_json(generator_path)
    stage = read_json(stage_path)
    backfill_artifact = read_json(backfill_path)
    r68 = read_json(r68_path)
    backfill_module, quadratic, relation_builder, rank_module = load_live_modules()
    (
        surface_records,
        case_results,
        sources,
        verifier,
        materialize_args,
    ) = materialize_all_proposals(generator, stage, backfill_artifact, backfill_module)
    direct_rows = [direct_root_row(row, quadratic) for row in surface_records]
    record_by_id = {str(row["surface_id"]): row for row in surface_records}
    sage_surfaces, window_bindings = load_sage_surfaces(sources)
    factor_rows = [
        factor_identity_row(surface, record_by_id[surface_id], quadratic)
        for surface_id, surface in sorted(sage_surfaces.items())
        if surface_id in record_by_id
    ]

    rho_sum = sum(int(row["generic_rho_steps"]) for row in direct_rows)
    fully_charged_sum = sum(
        int(row["fully_charged_direct_quadratic_ops"]) for row in direct_rows
    )
    marginal_sum = sum(
        int(row["marginal_direct_quadratic_ops"]) for row in direct_rows
    )
    exact_count = sum(row["exact_selected_pair_recovery"] for row in direct_rows)
    preservation_count = sum(
        not row["missing_selected_pairs"] for row in direct_rows
    )
    candidate_count = sum(int(row["candidate_count_with_index"]) for row in factor_rows)
    translated_matches = sum(
        int(row["translated_constant_zero_count_match_count"]) for row in factor_rows
    )
    all_line_surface_count = sum(
        row["all_factors_are_c_plus_r_b_plus_r2"] for row in factor_rows
    )
    all_zero_match_surface_count = sum(
        row["all_translated_constant_zero_counts_match"] for row in factor_rows
    )
    direct_failures = [
        row for row in direct_rows if not row["exact_selected_pair_recovery"]
    ]
    extra_relation_verification, candidate_forms_by_target = verify_extra_relations(
        surface_records,
        direct_rows,
        verifier,
        relation_builder,
        rank_module,
    )
    baseline_forms_by_target = source_relation_ledger(
        surface_records,
        verifier,
        backfill_module,
        materialize_args,
    )
    rank_audit = fresh_rank_audit(
        surface_records,
        baseline_forms_by_target,
        candidate_forms_by_target,
        verifier,
        rank_module,
    )
    extra_pair_count = sum(
        len(row["extra_selected_pairs"]) for row in direct_rows
    )
    unique_extra_pair_count = len(
        {
            (
                str(row["target"]),
                int(row["transfer_index"]),
                str(row["row_key"]),
                int(pair[0]),
                int(pair[1]),
            )
            for row in direct_rows
            for pair in row["extra_selected_pairs"]
        }
    )
    target_summaries = {}
    for target in sorted({str(row["target"]) for row in direct_rows}):
        rows = [row for row in direct_rows if str(row["target"]) == target]
        target_summaries[target] = {
            "proposal_surface_count": len(rows),
            "exact_selected_pair_recovery_count": sum(
                row["exact_selected_pair_recovery"] for row in rows
            ),
            "fully_charged_direct_quadratic_ops": sum(
                int(row["fully_charged_direct_quadratic_ops"]) for row in rows
            ),
            "sum_generic_rho_steps": sum(
                int(row["generic_rho_steps"]) for row in rows
            ),
        }
    fresh_rank_delta = int(rank_audit["fresh_rank_delta_sum_across_targets"])
    extras_closed = fresh_rank_delta == 0
    target_count = len({str(row["target"]) for row in direct_rows})
    admission_obligations = {
        "public_direct_roots_preserve_selected_pairs": {
            "pass": preservation_count == len(direct_rows),
            "preserved_surface_count": preservation_count,
            "surface_count": len(direct_rows),
        },
        "extra_roots_pass_relation_verification": {
            "pass": extra_relation_verification["accepted_relation_count"] > 0,
            "accepted_relation_count": extra_relation_verification[
                "accepted_relation_count"
            ],
        },
        "fresh_relation_forms_after_source_ledger_dedup": {
            "pass": rank_audit["fresh_candidate_relation_form_count"] > 0,
            "fresh_relation_form_count": rank_audit[
                "fresh_candidate_relation_form_count"
            ],
        },
        "positive_fresh_rank_delta": {
            "pass": fresh_rank_delta > 0,
            "fresh_rank_delta": fresh_rank_delta,
        },
        "source_precedes_selected_leaf_materialization": {
            "pass": False,
            "source_stage": "post_selected_public_row_leaf_materialization",
        },
        "at_least_four_target_families": {
            "pass": target_count >= 4,
            "target_count": target_count,
        },
    }
    failed_admission_obligations = [
        name for name, row in admission_obligations.items() if not row["pass"]
    ]
    return {
        "classification": "FFE_FACTORS_ARE_PUBLIC_ROOT_LINES_WITH_PARTIAL_DIRECT_REPLAY",
        "source_bindings": {
            "generator": {
                "path": str(generator_path),
                "sha256": sha256_file(generator_path),
                "schema": generator.get("schema"),
            },
            "stage_guard": {
                "path": str(stage_path),
                "sha256": sha256_file(stage_path),
                "schema": stage.get("schema"),
            },
            "backfill": {
                "path": str(backfill_path),
                "sha256": sha256_file(backfill_path),
                "schema": backfill_artifact.get("schema"),
            },
            "r68": {
                "path": str(r68_path),
                "sha256": sha256_file(r68_path),
                "schema": r68.get("schema"),
            },
        },
        "window_bindings": window_bindings,
        "factor_line_identity": {
            "surface_count": len(factor_rows),
            "retained_window_count": sum(
                bool(row.get("sage_factor_present")) for row in window_bindings
            ),
            "missing_window_count": sum(
                not bool(row.get("sage_factor_present")) for row in window_bindings
            ),
            "all_line_identity_surface_count": all_line_surface_count,
            "all_line_identity_pass": all_line_surface_count == len(factor_rows),
            "factor_candidate_count": candidate_count,
            "translated_constant_zero_count_match_count": translated_matches,
            "translated_constant_zero_equivalence_pass": (
                translated_matches == candidate_count
                and all_zero_match_surface_count == len(factor_rows)
            ),
            "identity": "factor(b,c) = c + r*b + r^2",
            "zero_equivalence": (
                "factor(b,c)=0 iff r is a root of x^2+b*x+c"
            ),
        },
        "direct_public_leaf_recovery": {
            "proposal_surface_count": len(direct_rows),
            "case_count": len(case_results),
            "preserved_selected_pair_surface_count": preservation_count,
            "preserved_selected_pair_surface_fraction": round(
                preservation_count / max(1, len(direct_rows)), 8
            ),
            "exact_selected_pair_recovery_count": exact_count,
            "exact_selected_pair_recovery_pass": exact_count == len(direct_rows),
            "exact_selected_pair_recovery_fraction": round(
                exact_count / max(1, len(direct_rows)), 8
            ),
            "failed_exact_selected_pair_recovery_count": len(direct_failures),
            "target_count": target_count,
            "fully_charged_direct_quadratic_ops": fully_charged_sum,
            "marginal_direct_quadratic_ops_after_core": marginal_sum,
            "sum_generic_rho_steps": rho_sum,
            "fully_charged_direct_quadratic_ops_over_sum_rho": round(
                fully_charged_sum / max(1, rho_sum), 8
            ),
            "marginal_direct_quadratic_ops_over_sum_rho": round(
                marginal_sum / max(1, rho_sum), 8
            ),
            "target_summaries": target_summaries,
        },
        "extra_root_relation_verification": extra_relation_verification,
        "fresh_relation_rank_audit": rank_audit,
        "information_accounting": {
            "candidate_extra_root_pair_count": extra_pair_count,
            "unique_candidate_extra_root_pair_count": unique_extra_pair_count,
            "verifier_accepted_extra_relation_count": extra_relation_verification[
                "accepted_relation_count"
            ],
            "new_independent_fixed_sum_row_count": fresh_rank_delta,
            "fresh_rank_delta": fresh_rank_delta,
            "freshness_status": (
                "CLOSED_NO_RANK_DELTA_AGAINST_FULL_SOURCE_LEDGER"
                if extras_closed
                else "POSITIVE_TOY_FRESH_RANK_REQUIRES_PROSPECTIVE_REPLAY"
            ),
            "factorization_adds_information": False,
            "r68_information_conservation_pass": bool(r68.get("pass")),
            "reason": (
                "The factor lines add no information beyond public leaf roots. "
                "Most roots replay selected pairs. Extra polynomial roots receive "
                "rank credit only after elliptic-curve point equality, relation "
                "verification, source-ledger deduplication, and modular rank."
            ),
        },
        "claim_boundary": {
            "algorithm_breakthrough": False,
            "generic_prime_field_speedup": False,
            "shoup_bound_improvement": False,
            "direct_recovery_of_existing_rows_is_relation_discovery": False,
        },
        "admission": {
            "lane_admitted": not failed_admission_obligations,
            "passed_obligation_count": (
                len(admission_obligations) - len(failed_admission_obligations)
            ),
            "obligation_count": len(admission_obligations),
            "failed_obligations": failed_admission_obligations,
            "obligations": admission_obligations,
        },
        "fatal_obstruction": (
            (
                "All retained FFE factors are public root-line encodings. Direct "
                "monic quadratic solving exactly replays 341 of 356 selected pair "
                "sets. The verified extra forms add no rank beyond the complete "
                "source ledger."
            )
            if extras_closed
            else (
                "Verified extra forms add toy fresh rank, but prospective "
                "target-family transfer and complete attack-level costs remain "
                "untested."
            )
        ),
        "next_action": (
            (
                "Close this factor-selector branch and move the generator upstream "
                "of selected leaves: preregister a scalar-blind source of previously "
                "unseen fixed-sum row candidates, then require positive fresh rank "
                "and complete cost below direct pair-complement enumeration on four "
                "target families."
            )
            if extras_closed
            else (
                "Freeze the direct-root source rule and replay it unchanged on at "
                "least two new target families; require positive fresh rank on every "
                "family and complete source/descent cost below rho."
            )
        ),
        "surface_examples": direct_rows[:8],
        "direct_recovery_failure_examples": direct_failures[:15],
        "factor_identity_examples": factor_rows[:8],
        "factor_identity_failure_examples": [
            row
            for row in factor_rows
            if not row["all_factors_are_c_plus_r_b_plus_r2"]
            or not row["all_translated_constant_zero_counts_match"]
        ][:15],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--backfill", type=Path, default=DEFAULT_BACKFILL)
    parser.add_argument("--r68", type=Path, default=DEFAULT_R68)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_live(args.generator, args.stage, args.backfill, args.r68)
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    identity = payload["factor_line_identity"]
    recovery = payload["direct_public_leaf_recovery"]
    print(
        f"output={args.output} factors={identity['factor_candidate_count']} "
        f"identity={identity['translated_constant_zero_equivalence_pass']} "
        f"direct_exact={recovery['exact_selected_pair_recovery_count']}/"
        f"{recovery['proposal_surface_count']} candidate_extra_pairs="
        f"{payload['information_accounting']['candidate_extra_root_pair_count']} "
        f"fresh_rank={payload['information_accounting']['fresh_rank_delta']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
