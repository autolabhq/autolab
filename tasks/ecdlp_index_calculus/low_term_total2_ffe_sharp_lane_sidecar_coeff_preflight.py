#!/usr/bin/env python3
"""Extract sidecar coefficient replay material for selected13 priorities.

The source-gap preflight identifies which priority backfill masks have exact
source witnesses.  This script resolves the mounted direct-relation sidecar
JSON files for those witnesses, extracts the actual coefficient forms, and
emits a native-checkable header for a lower-level FFE/summation-polynomial
worker.

This is coefficient replay material only.  It does not transplant coefficients
into a new transfer, regenerate FFE sidecars, export direct/rank rows, solve
ECDLP, or claim a Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_sidecar_coeff_preflight.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_EXACT_REPLAY = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_exact_certificate_replay_selected13_9696_9999_probe.json"
)
DEFAULT_SOURCE_GAP = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_source_gap_preflight_selected13_9696_9999_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_sidecar_coeff_preflight_selected13_9696_9999_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_sidecar_coeff_preflight_selected13_9696_9999_probe.h"
)
DEFAULT_SIDECAR_ROOTS = [
    DEFAULT_STATE_DIR,
    Path("/Volumes/Volume/git/autolab/ecdlp_index_calculus_state"),
    Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state"),
]

COEFF_COUNT = 17
FAMILY_MASK_COUNT = 3
PRIMARY_BACKFILL_TRANSFERS = [9981, 9943]
TIER_CODES = {
    "same_row_key_exact_support": 1,
    "one_salt_neighbor_exact_support": 2,
    "support_span_only": 3,
}
EXPECTED_PRIMARY_MASK_COUNT = 6
EXPECTED_COEFF_FORM_COUNT = 12
EXPECTED_UNIQUE_COEFF_FORM_COUNT = 8
EXPECTED_SAME_ROW_KEY_FORM_COUNT = 4
EXPECTED_ONE_SALT_NEIGHBOR_FORM_COUNT = 4
EXPECTED_SUPPORT_SPAN_ONLY_FORM_COUNT = 4
EXPECTED_SOURCE_TRANSFERS = [9742, 9754, 9842]


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


def canonical_json(raw: Any) -> str:
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def digest_u64(raw: Any) -> int:
    text = str(raw or "")
    suffix = text.rsplit("_", 1)[-1]
    if suffix and all(ch in "0123456789abcdefABCDEF" for ch in suffix):
        return int(suffix[-16:], 16)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def form_hash_u64(form: dict[str, Any], order: int) -> int:
    payload = {
        "coeffs": [as_int(value) % order for value in form.get("coeffs") or []],
        "order": order,
        "rhs": as_int(form.get("rhs")) % order,
        "terms": [as_int(value) for value in form.get("terms") or []],
    }
    return int(hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16], 16)


def support_mask_from_coeffs(coeffs: list[Any], order: int) -> int:
    mask = 0
    for index, coeff in enumerate(coeffs[1:]):
        if order and as_int(coeff) % order != 0:
            mask |= 1 << index
    return mask


def normalize_row_keys(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in raw or []))


def parse_sidecar_roots(raw: str | None) -> list[Path]:
    if not raw:
        return DEFAULT_SIDECAR_ROOTS
    roots = [Path(item.strip()) for item in raw.split(",") if item.strip()]
    return roots or DEFAULT_SIDECAR_ROOTS


def resolve_sidecar(path_hint: str, roots: list[Path]) -> Path | None:
    hinted = Path(path_hint)
    if hinted.is_file():
        return hinted
    basename = hinted.name
    for root in roots:
        candidate = root / basename
        if candidate.is_file():
            return candidate
    return None


def exact_records_by_transfer(exact_replay: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        as_int(record.get("transfer_index"), -1): record
        for record in exact_replay.get("records") or []
        if isinstance(record, dict)
    }


def sidecar_artifact_for_transfer(records: dict[int, dict[str, Any]], transfer: int) -> str | None:
    record = records.get(transfer) or {}
    material = record.get("recomputed_certificate_material") or {}
    return material.get("artifact")


def source_gap_primary_items(source_gap: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        item
        for item in source_gap.get("source_gap_items") or []
        if isinstance(item, dict) and as_int(item.get("transfer_index"), -1) in PRIMARY_BACKFILL_TRANSFERS
    ]
    return sorted(items, key=lambda item: PRIMARY_BACKFILL_TRANSFERS.index(as_int(item.get("transfer_index"), -1)))


def find_sidecar_certificate(sidecar: dict[str, Any], witness: dict[str, Any], exact_record: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
    transfer = as_int(witness.get("transfer_index"), -1)
    exact_material = exact_record.get("recomputed_certificate_material") or {}
    expected_keys = normalize_row_keys(witness.get("row_keys") or exact_material.get("row_keys"))
    selector = exact_record.get("selector") or exact_material.get("selector")
    target = exact_record.get("target") or exact_material.get("target")
    top_k = as_int(exact_record.get("top_k") or exact_material.get("top_k"))
    selected_mask = as_int(exact_record.get("selected_support_mask"))
    for offset, cert in enumerate(sidecar.get("certificates") or []):
        selected = cert.get("selected") or {}
        cert_support_mask = support_mask_from_support(cert.get("selected_term_support") or [])
        if as_int(selected.get("transfer_index"), -1) != transfer:
            continue
        if selected.get("target") != target:
            continue
        if selected.get("selector") != selector:
            continue
        if as_int(selected.get("top_k")) != top_k:
            continue
        if normalize_row_keys(selected.get("row_keys")) != expected_keys:
            continue
        if selected_mask and cert_support_mask != selected_mask:
            continue
        return offset, cert
    return -1, None


def support_mask_from_support(raw: Any) -> int:
    mask = 0
    for item in raw or []:
        value = as_int(item)
        if value >= 0:
            mask |= 1 << value
    return mask


def coeff_rows_for_mask(
    primary_item: dict[str, Any],
    mask_witness: dict[str, Any],
    witness: dict[str, Any],
    exact_record: dict[str, Any],
    sidecar_path: Path,
    sidecar: dict[str, Any],
    cert_offset: int,
    cert: dict[str, Any],
) -> list[dict[str, Any]]:
    order = as_int(cert.get("order"))
    rows = []
    for form_index, form in enumerate(cert.get("forms") or []):
        coeffs = [as_int(value) % order for value in form.get("coeffs") or []]
        while len(coeffs) < COEFF_COUNT:
            coeffs.append(0)
        coeffs = coeffs[:COEFF_COUNT]
        support_mask = support_mask_from_coeffs(coeffs, order)
        if support_mask != as_int(mask_witness.get("family_mask")):
            continue
        rows.append(
            {
                "backfill_materialized_row_keys": primary_item.get("materialized_row_keys") or [],
                "backfill_row_check_hash": primary_item.get("row_check_hash"),
                "backfill_row_check_hash_u64": as_int(primary_item.get("row_check_hash_u64")) or digest_u64(
                    primary_item.get("row_check_hash")
                ),
                "backfill_salts": primary_item.get("salts") or [],
                "backfill_transfer_index": as_int(primary_item.get("transfer_index"), -1),
                "coeff_count": len(coeffs),
                "coeffs": coeffs,
                "form_hash_u64": form_hash_u64(form, order),
                "form_index": form_index,
                "rhs": as_int(form.get("rhs")) % order,
                "sidecar_certificate_offset": cert_offset,
                "sidecar_path": str(sidecar_path),
                "source_certificate_hash": witness.get("certificate_hash"),
                "source_certificate_hash_u64": as_int(witness.get("certificate_hash_u64"))
                or digest_u64(witness.get("certificate_hash")),
                "source_certificate_status": cert.get("certificate_status"),
                "source_derived_secret": cert.get("derived_secret"),
                "source_form_support_mask": support_mask,
                "source_public_key_verified": bool(cert.get("public_key_verified")),
                "source_row_keys": (cert.get("selected") or {}).get("row_keys") or [],
                "source_salts": witness.get("salts") or [],
                "source_tier": mask_witness.get("source_tier"),
                "source_tier_code": as_int(mask_witness.get("source_tier_code")),
                "source_transfer_index": as_int(witness.get("transfer_index"), -1),
                "terms": [as_int(value) for value in form.get("terms") or []],
            }
        )
    return rows


def build_coeff_material(
    exact_replay: dict[str, Any],
    source_gap: dict[str, Any],
    roots: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = exact_records_by_transfer(exact_replay)
    source_items = []
    coeff_rows = []
    failures = []
    for primary_item in source_gap_primary_items(source_gap):
        mask_items = []
        for mask_witness in primary_item.get("mask_witnesses") or []:
            witness_list = mask_witness.get("witnesses") or []
            if not witness_list:
                failures.append(
                    {
                        "code": "mask_witness_missing_source",
                        "backfill_transfer_index": primary_item.get("transfer_index"),
                        "family_mask": mask_witness.get("family_mask"),
                    }
                )
                continue
            witness = witness_list[0]
            source_transfer = as_int(witness.get("transfer_index"), -1)
            exact_record = records.get(source_transfer) or {}
            artifact_hint = sidecar_artifact_for_transfer(records, source_transfer)
            if not artifact_hint:
                failures.append({"code": "source_sidecar_artifact_hint_missing", "source_transfer_index": source_transfer})
                continue
            sidecar_path = resolve_sidecar(artifact_hint, roots)
            if sidecar_path is None:
                failures.append(
                    {
                        "artifact_hint": artifact_hint,
                        "code": "source_sidecar_not_found",
                        "source_transfer_index": source_transfer,
                    }
                )
                continue
            sidecar = load_json(sidecar_path)
            cert_offset, cert = find_sidecar_certificate(sidecar, witness, exact_record)
            if cert is None:
                failures.append(
                    {
                        "code": "source_sidecar_certificate_not_found",
                        "sidecar_path": str(sidecar_path),
                        "source_transfer_index": source_transfer,
                    }
                )
                continue
            rows = coeff_rows_for_mask(primary_item, mask_witness, witness, exact_record, sidecar_path, sidecar, cert_offset, cert)
            if not rows:
                failures.append(
                    {
                        "code": "source_sidecar_mask_coeff_rows_missing",
                        "family_mask": mask_witness.get("family_mask"),
                        "sidecar_path": str(sidecar_path),
                        "source_transfer_index": source_transfer,
                    }
                )
            coeff_rows.extend(rows)
            mask_items.append(
                {
                    "backfill_transfer_index": primary_item.get("transfer_index"),
                    "coefficient_form_count": len(rows),
                    "family_mask": as_int(mask_witness.get("family_mask")),
                    "sidecar_certificate_offset": cert_offset,
                    "sidecar_path": str(sidecar_path),
                    "source_tier": mask_witness.get("source_tier"),
                    "source_transfer_index": source_transfer,
                }
            )
        source_items.append(
            {
                "backfill_transfer_index": as_int(primary_item.get("transfer_index"), -1),
                "mask_sources": mask_items,
                "source_gap_class": primary_item.get("source_gap_class"),
                "source_tier_counts": primary_item.get("source_tier_counts"),
            }
        )
    return source_items, coeff_rows, failures


def validate_sources(exact_replay: dict[str, Any], source_gap: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if exact_replay.get("claim_status") != "FFE_SHARP_LANE_EXACT_CERTIFICATE_REPLAY_READY":
        failures.append({"code": "exact_replay_not_ready", "claim_status": exact_replay.get("claim_status")})
    if exact_replay.get("failures"):
        failures.append({"code": "exact_replay_has_failures", "failures": exact_replay.get("failures")})
    if source_gap.get("claim_status") != "FFE_SHARP_LANE_SOURCE_GAP_PREFLIGHT_READY":
        failures.append({"code": "source_gap_not_ready", "claim_status": source_gap.get("claim_status")})
    if source_gap.get("failures"):
        failures.append({"code": "source_gap_has_failures", "failures": source_gap.get("failures")})
    if (source_gap.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "source_gap_summary_not_verified", "summary": source_gap.get("summary")})
    return failures


def tier_form_count(rows: list[dict[str, Any]], tier: str) -> int:
    return sum(1 for row in rows if row.get("source_tier") == tier)


def validate_coeff_rows(source_items: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(source_items) != len(PRIMARY_BACKFILL_TRANSFERS):
        failures.append({"code": "primary_source_item_count_mismatch", "observed": len(source_items)})
    if [as_int(item.get("backfill_transfer_index"), -1) for item in source_items] != PRIMARY_BACKFILL_TRANSFERS:
        failures.append({"code": "primary_source_order_mismatch"})
    if sum(len(item.get("mask_sources") or []) for item in source_items) != EXPECTED_PRIMARY_MASK_COUNT:
        failures.append({"code": "primary_mask_source_count_mismatch"})
    if len(rows) != EXPECTED_COEFF_FORM_COUNT:
        failures.append({"code": "coefficient_form_count_mismatch", "observed": len(rows)})
    if len({as_int(row.get("form_hash_u64")) for row in rows}) != EXPECTED_UNIQUE_COEFF_FORM_COUNT:
        failures.append({"code": "unique_coefficient_form_count_mismatch"})
    if tier_form_count(rows, "same_row_key_exact_support") != EXPECTED_SAME_ROW_KEY_FORM_COUNT:
        failures.append({"code": "same_row_key_form_count_mismatch"})
    if tier_form_count(rows, "one_salt_neighbor_exact_support") != EXPECTED_ONE_SALT_NEIGHBOR_FORM_COUNT:
        failures.append({"code": "one_salt_neighbor_form_count_mismatch"})
    if tier_form_count(rows, "support_span_only") != EXPECTED_SUPPORT_SPAN_ONLY_FORM_COUNT:
        failures.append({"code": "support_span_only_form_count_mismatch"})
    source_transfers = sorted({as_int(row.get("source_transfer_index"), -1) for row in rows})
    if source_transfers != EXPECTED_SOURCE_TRANSFERS:
        failures.append({"code": "source_transfer_set_mismatch", "observed": source_transfers})
    for row in rows:
        order = 11779
        coeffs = [as_int(value) for value in row.get("coeffs") or []]
        if len(coeffs) != COEFF_COUNT or as_int(row.get("coeff_count")) != COEFF_COUNT:
            failures.append({"code": "coefficient_width_mismatch", "form_hash_u64": row.get("form_hash_u64")})
        if support_mask_from_coeffs(coeffs, order) != as_int(row.get("source_form_support_mask")):
            failures.append({"code": "coefficient_support_mask_mismatch", "form_hash_u64": row.get("form_hash_u64")})
        if as_int(row.get("source_tier_code")) != TIER_CODES.get(str(row.get("source_tier")), 0):
            failures.append({"code": "source_tier_code_mismatch", "form_hash_u64": row.get("form_hash_u64")})
        if not bool(row.get("source_public_key_verified")):
            failures.append({"code": "source_public_key_not_verified", "form_hash_u64": row.get("form_hash_u64")})
        if row.get("source_certificate_status") != "PUBLIC_DIRECT_RELATION_EQUATIONS_VERIFY_PUBLIC_KEY":
            failures.append({"code": "source_certificate_status_not_verified", "form_hash_u64": row.get("form_hash_u64")})
        if not Path(str(row.get("sidecar_path"))).is_file():
            failures.append({"code": "sidecar_path_not_readable", "sidecar_path": row.get("sidecar_path")})
    return failures


def c_u64_array(values: list[int]) -> str:
    return "{" + ", ".join(f"{as_int(value)}ULL" for value in values) + "}"


def render_c_header(rows: list[dict[str, Any]]) -> str:
    row_lines = []
    for index, row in enumerate(rows):
        coeffs = [as_int(value) for value in row.get("coeffs") or []]
        while len(coeffs) < COEFF_COUNT:
            coeffs.append(0)
        row_lines.append(
            "  {"
            f"{index}ULL, "
            f"{as_int(row.get('backfill_transfer_index'))}ULL, "
            f"{as_int(row.get('source_transfer_index'))}ULL, "
            f"{as_int(row.get('source_tier_code'))}ULL, "
            f"{as_int(row.get('source_form_support_mask'))}ULL, "
            f"{as_int(row.get('form_hash_u64'))}ULL, "
            f"{as_int(row.get('rhs'))}ULL, "
            f"{as_int(row.get('source_certificate_hash_u64'))}ULL, "
            f"{as_int(row.get('backfill_row_check_hash_u64'))}ULL, "
            f"{c_u64_array(coeffs[:COEFF_COUNT])}"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_SIDECAR_COEFF_PREFLIGHT_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_SIDECAR_COEFF_PREFLIGHT_H

#include <stdint.h>

#define SELECTED13_SIDECAR_COEFF_FORM_COUNT {len(rows)}
#define SELECTED13_SIDECAR_COEFF_WIDTH {COEFF_COUNT}
#define SELECTED13_SIDECAR_UNIQUE_COEFF_FORM_COUNT {len({as_int(row.get("form_hash_u64")) for row in rows})}
#define SELECTED13_SIDECAR_SAME_ROW_KEY_FORM_COUNT {tier_form_count(rows, "same_row_key_exact_support")}
#define SELECTED13_SIDECAR_ONE_SALT_NEIGHBOR_FORM_COUNT {tier_form_count(rows, "one_salt_neighbor_exact_support")}
#define SELECTED13_SIDECAR_SUPPORT_SPAN_ONLY_FORM_COUNT {tier_form_count(rows, "support_span_only")}

#define SELECTED13_SOURCE_TIER_SAME_ROW_KEY 1ULL
#define SELECTED13_SOURCE_TIER_ONE_SALT_NEIGHBOR 2ULL
#define SELECTED13_SOURCE_TIER_SUPPORT_SPAN_ONLY 3ULL

typedef struct {{
  uint64_t coeff_form_index;
  uint64_t backfill_transfer_index;
  uint64_t source_transfer_index;
  uint64_t source_tier_code;
  uint64_t support_mask;
  uint64_t form_hash_u64;
  uint64_t rhs;
  uint64_t source_certificate_hash_u64;
  uint64_t backfill_row_check_hash_u64;
  uint64_t coeffs[SELECTED13_SIDECAR_COEFF_WIDTH];
}} selected13_sidecar_coeff_form_t;

static const selected13_sidecar_coeff_form_t SELECTED13_SIDECAR_COEFF_FORMS[] = {{
{chr(10).join(row_lines)}
}};

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

static uint64_t support_mask_from_coeffs(const uint64_t coeffs[SELECTED13_SIDECAR_COEFF_WIDTH]) {{
  uint64_t mask = 0;
  for (uint64_t i = 1; i < SELECTED13_SIDECAR_COEFF_WIDTH; i++) {{
    if (coeffs[i] % 11779ULL != 0) {{
      mask |= (1ULL << (i - 1));
    }}
  }}
  return mask;
}}

int main(void) {{
  uint64_t failure_count = 0;
  uint64_t same_row_key_form_count = 0;
  uint64_t one_salt_neighbor_form_count = 0;
  uint64_t support_span_only_form_count = 0;
  uint64_t transfer9981_form_count = 0;
  uint64_t transfer9943_form_count = 0;
  uint64_t nonzero_hash_count = 0;
  const size_t form_count =
      sizeof(SELECTED13_SIDECAR_COEFF_FORMS) / sizeof(SELECTED13_SIDECAR_COEFF_FORMS[0]);
  if (form_count != SELECTED13_SIDECAR_COEFF_FORM_COUNT) failure_count++;

  for (size_t i = 0; i < form_count; i++) {{
    const selected13_sidecar_coeff_form_t *form = &SELECTED13_SIDECAR_COEFF_FORMS[i];
    if (form->coeff_form_index != i) failure_count++;
    if (form->form_hash_u64 == 0 || form->source_certificate_hash_u64 == 0 ||
        form->backfill_row_check_hash_u64 == 0) {{
      failure_count++;
    }} else {{
      nonzero_hash_count++;
    }}
    if (support_mask_from_coeffs(form->coeffs) != form->support_mask) failure_count++;
    if (form->rhs >= 11779ULL) failure_count++;
    if (form->source_tier_code == SELECTED13_SOURCE_TIER_SAME_ROW_KEY) {{
      same_row_key_form_count++;
    }} else if (form->source_tier_code == SELECTED13_SOURCE_TIER_ONE_SALT_NEIGHBOR) {{
      one_salt_neighbor_form_count++;
    }} else if (form->source_tier_code == SELECTED13_SOURCE_TIER_SUPPORT_SPAN_ONLY) {{
      support_span_only_form_count++;
    }} else {{
      failure_count++;
    }}
    if (form->backfill_transfer_index == 9981ULL) {{
      transfer9981_form_count++;
    }} else if (form->backfill_transfer_index == 9943ULL) {{
      transfer9943_form_count++;
    }} else {{
      failure_count++;
    }}
  }}

  if (same_row_key_form_count != SELECTED13_SIDECAR_SAME_ROW_KEY_FORM_COUNT) failure_count++;
  if (one_salt_neighbor_form_count != SELECTED13_SIDECAR_ONE_SALT_NEIGHBOR_FORM_COUNT) failure_count++;
  if (support_span_only_form_count != SELECTED13_SIDECAR_SUPPORT_SPAN_ONLY_FORM_COUNT) failure_count++;
  if (same_row_key_form_count != {EXPECTED_SAME_ROW_KEY_FORM_COUNT}ULL) failure_count++;
  if (one_salt_neighbor_form_count != {EXPECTED_ONE_SALT_NEIGHBOR_FORM_COUNT}ULL) failure_count++;
  if (support_span_only_form_count != {EXPECTED_SUPPORT_SPAN_ONLY_FORM_COUNT}ULL) failure_count++;
  if (transfer9981_form_count != 6ULL || transfer9943_form_count != 6ULL) failure_count++;
  if (nonzero_hash_count != form_count) failure_count++;

  printf("{{");
  printf("\\\"form_count\\\":%llu,", (unsigned long long)form_count);
  printf("\\\"same_row_key_form_count\\\":%llu,", (unsigned long long)same_row_key_form_count);
  printf("\\\"one_salt_neighbor_form_count\\\":%llu,", (unsigned long long)one_salt_neighbor_form_count);
  printf("\\\"support_span_only_form_count\\\":%llu,", (unsigned long long)support_span_only_form_count);
  printf("\\\"transfer9981_form_count\\\":%llu,", (unsigned long long)transfer9981_form_count);
  printf("\\\"transfer9943_form_count\\\":%llu,", (unsigned long long)transfer9943_form_count);
  printf("\\\"nonzero_hash_count\\\":%llu,", (unsigned long long)nonzero_hash_count);
  printf("\\\"failure_count\\\":%llu", (unsigned long long)failure_count);
  printf("}}\\n");
  return failure_count == 0 ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path, compiler: str) -> dict[str, Any]:
    c_source = render_preflight_c(header_path.name)
    source_hash = hashlib.sha256(c_source.encode("utf-8")).hexdigest()
    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root)
    with tempfile.TemporaryDirectory(prefix="ecdlp_selected13_sidecar_coeff_", dir=str(temp_root)) as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "selected13_sidecar_coeff_preflight.c"
        exe_path = temp_path / "selected13_sidecar_coeff_preflight"
        c_path.write_text(c_source)
        command = [
            compiler,
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(header_path.parent),
            str(c_path),
            "-o",
            str(exe_path),
        ]
        compile_run = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        if compile_run.returncode != 0:
            return {
                "compiled": False,
                "compile_command": command,
                "compile_returncode": compile_run.returncode,
                "compile_stdout": compile_run.stdout,
                "compile_stderr": compile_run.stderr,
                "executed": False,
                "c_source_sha256": source_hash,
            }
        native_run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False, env=env)
        try:
            native_summary = json.loads(native_run.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            native_summary = None
        return {
            "compiled": True,
            "compile_command": command,
            "compile_returncode": compile_run.returncode,
            "compile_stdout": compile_run.stdout,
            "compile_stderr": compile_run.stderr,
            "executed": True,
            "run_returncode": native_run.returncode,
            "run_stdout": native_run.stdout,
            "run_stderr": native_run.stderr,
            "native_summary": native_summary,
            "c_source_sha256": source_hash,
        }


def compare_native(rows: list[dict[str, Any]], native: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not native.get("compiled"):
        failures.append({"code": "native_preflight_compile_failed", "compile_stderr": native.get("compile_stderr")})
        return failures
    if as_int(native.get("run_returncode"), -1) != 0:
        failures.append({"code": "native_preflight_run_failed", "run_stdout": native.get("run_stdout")})
    summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    if not summary:
        failures.append({"code": "native_preflight_stdout_not_json", "run_stdout": native.get("run_stdout")})
        return failures
    if as_int(summary.get("form_count"), -1) != len(rows):
        failures.append({"code": "native_form_count_mismatch"})
    if as_int(summary.get("failure_count"), -1) != 0:
        failures.append({"code": "native_preflight_reported_failures", "native_summary": summary})
    return failures


def summarize(source_items: list[dict[str, Any]], rows: list[dict[str, Any]], native_summary: dict[str, Any]) -> dict[str, Any]:
    sidecar_paths = sorted({str(row.get("sidecar_path")) for row in rows})
    return {
        "coefficient_form_count": len(rows),
        "mounted_sidecar_count": len(sidecar_paths),
        "mounted_sidecar_writable": all(os.access(path, os.W_OK) for path in sidecar_paths),
        "native_preflight_failure_count": as_int(native_summary.get("failure_count"), -1),
        "native_preflight_verified": as_int(native_summary.get("failure_count"), -1) == 0,
        "one_salt_neighbor_form_count": tier_form_count(rows, "one_salt_neighbor_exact_support"),
        "primary_backfill_transfers": [as_int(item.get("backfill_transfer_index"), -1) for item in source_items],
        "same_row_key_form_count": tier_form_count(rows, "same_row_key_exact_support"),
        "sidecar_coeff_interpretation": (
            "Mounted sidecars provide coefficient forms for the priority support witnesses; "
            "the next worker still must regenerate/export direct rows for 9981 and 9943."
        ),
        "source_sidecar_paths": sidecar_paths,
        "source_transfers": sorted({as_int(row.get("source_transfer_index"), -1) for row in rows}),
        "support_span_only_form_count": tier_form_count(rows, "support_span_only"),
        "unique_coefficient_form_count": len({as_int(row.get("form_hash_u64")) for row in rows}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-replay", type=Path, default=DEFAULT_EXACT_REPLAY)
    parser.add_argument("--source-gap", type=Path, default=DEFAULT_SOURCE_GAP)
    parser.add_argument("--sidecar-roots", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--no-c-header", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exact_replay = load_json(args.exact_replay)
    source_gap = load_json(args.source_gap)
    roots = parse_sidecar_roots(args.sidecar_roots)
    source_items, coeff_rows, failures = build_coeff_material(exact_replay, source_gap, roots)
    failures = validate_sources(exact_replay, source_gap) + failures
    failures.extend(validate_coeff_rows(source_items, coeff_rows))

    native: dict[str, Any] = {"compiled": False, "executed": False, "native_summary": {}}
    if not args.no_c_header:
        args.c_header_out.parent.mkdir(parents=True, exist_ok=True)
        args.c_header_out.write_text(render_c_header(coeff_rows))
        native = run_native_preflight(args.c_header_out, args.cc)
        failures.extend(compare_native(coeff_rows, native))

    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    verified = not failures
    payload = {
        "artifacts": {
            "c_header": None if args.no_c_header else str(args.c_header_out),
            "exact_replay": str(args.exact_replay),
            "source_gap": str(args.source_gap),
        },
        "claim_status": (
            "FFE_SHARP_LANE_SIDECAR_COEFF_PREFLIGHT_READY"
            if verified
            else "FFE_SHARP_LANE_SIDECAR_COEFF_PREFLIGHT_FAILED_CHECK"
        ),
        "coefficient_forms": coeff_rows,
        "created_at": now_iso(),
        "failures": failures,
        "honesty_boundary": [
            "This extracts mounted source-sidecar coefficient forms for replay planning only.",
            "It does not transplant coefficients into a backfill transfer or export new direct/rank rows.",
            "It does not evaluate summation polynomials, solve finite-field equations, solve ECDLP, or claim a Pollard-rho speedup.",
        ],
        "native_preflight": native,
        "parameters": {
            "coefficient_width": COEFF_COUNT,
            "expected_coefficient_form_count": EXPECTED_COEFF_FORM_COUNT,
            "expected_source_transfers": EXPECTED_SOURCE_TRANSFERS,
            "primary_backfill_transfers": PRIMARY_BACKFILL_TRANSFERS,
            "sidecar_roots": [str(root) for root in roots],
            "source_tier_codes": TIER_CODES,
        },
        "schema": SCHEMA,
        "source_items": source_items,
        "summary": summarize(source_items, coeff_rows, native_summary),
    }
    payload["summary"]["verified"] = verified
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "failures": failures,
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
