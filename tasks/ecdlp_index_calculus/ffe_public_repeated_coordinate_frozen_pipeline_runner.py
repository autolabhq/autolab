#!/usr/bin/env python3
"""Run the frozen repeated-coordinate row/guard pipeline on a stress artifact.

This is the promotion harness for the current target-67 repeated-coordinate
candidate mechanism:

1. select bounded low-term total3/total4 public cases from a stress artifact;
2. mine repeated monic-coordinate gates from that public selector;
3. replay frozen row rules with the frozen ``candidate_pos_span=0`` form guard.

The harness writes all intermediate artifacts and a summary manifest.  It is
intended for fresh windows: do not treat a control run on calibration data as
fresh validation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = WORKTREE_ROOT / "tasks" / "ecdlp_index_calculus"
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_TARGET = "67.a1@9803"
DEFAULT_ACTIVATION_RULE = (
    "activate:b_minus_c_mod16=0&source_ops_millirhos>=1312|"
    "c_mod16=10&source_ops_millirhos>=1368"
)
DEFAULT_ROW_RULES = [
    (
        "activated",
        "row_activate:b_minus_c_mod16=0&salt_mod4=3|"
        "b_mod16=10&salt_delta_to_max=1|"
        "salt_delta_from_min<=1&salt_mod2=0",
    ),
    (
        "all_root",
        "row_activate:b_mod5=4&salt=203|leaf_min_mod8=4&salt_delta_from_min=1|"
        "salt_mod4=3&transfer_mod16=8",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_command(cmd: list[str], cwd: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/codex_pycache")
    if "ECDLP_TASK_DIR" not in env and Path("/Volumes/Volume/autolab/tasks/ecdlp_index_calculus").exists():
        env["ECDLP_TASK_DIR"] = "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus"
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def require_success(result: dict[str, Any]) -> None:
    if int(result["returncode"]) != 0:
        raise RuntimeError(
            "command failed with return code "
            f"{result['returncode']}: {' '.join(result['cmd'])}\n"
            f"stdout:\n{result['stdout']}\n"
            f"stderr:\n{result['stderr']}\n"
        )


def artifact_paths(out_prefix: Path) -> dict[str, Path]:
    return {
        "public_selector": out_prefix.with_name(out_prefix.name + "_public_selector.json"),
        "coordinate_gate": out_prefix.with_name(out_prefix.name + "_coordinate_gate.json"),
        "guard_replay": out_prefix.with_name(out_prefix.name + "_guard_replay.json"),
        "manifest": out_prefix.with_name(out_prefix.name + "_manifest.json"),
    }


def summarize_selector(path: Path) -> dict[str, Any]:
    data = load_json(path)
    return data.get("summary") or {}


def summarize_gate(path: Path) -> dict[str, Any]:
    data = load_json(path)
    return data.get("summary") or {}


def summarize_guard(path: Path) -> dict[str, Any]:
    data = load_json(path)
    return data.get("summary") or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-source", type=Path, required=True)
    parser.add_argument("--window-name", required=True)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--max-ops-over-rho", type=float, default=1.5)
    parser.add_argument("--max-cases-per-challenge", type=int, default=0)
    parser.add_argument("--gate-max-candidates", type=int, default=256)
    parser.add_argument("--gate-replay-top", type=int, default=256)
    parser.add_argument("--activation-rule", default=DEFAULT_ACTIVATION_RULE)
    parser.add_argument("--row-rule", action="append", default=[])
    parser.add_argument("--form-guard", default="candidate_pos_span=0")
    parser.add_argument("--event-summary-limit", type=int, default=16)
    parser.add_argument("--allow-over-rho", action="store_true", default=True)
    parser.add_argument("--no-allow-over-rho", dest="allow_over_rho", action="store_false")
    args = parser.parse_args()

    paths = artifact_paths(args.out_prefix)
    row_rules = list(args.row_rule or [])
    if not row_rules:
        row_rules = [f"{name}|{rule}" for name, rule in DEFAULT_ROW_RULES]

    commands: list[dict[str, Any]] = []

    selector_cmd = [
        sys.executable,
        str(TASK_DIR / "low_term_totalk_public_stress_selector_probe.py"),
        "--stress-source",
        str(args.stress_source),
        "--max-ops-over-rho",
        str(args.max_ops_over_rho),
        "--max-cases-per-challenge",
        str(args.max_cases_per_challenge),
        "--out",
        str(paths["public_selector"]),
    ]
    if args.allow_over_rho:
        selector_cmd.append("--allow-over-rho")
    result = run_command(selector_cmd, WORKTREE_ROOT)
    commands.append(result)
    require_success(result)

    gate_cmd = [
        sys.executable,
        str(TASK_DIR / "ffe_public_repeated_coordinate_gate_miner.py"),
        "--signature-source",
        str(paths["public_selector"]),
        "--target",
        str(args.target),
        "--min-row-count",
        "2",
        "--min-profile-count",
        "2",
        "--max-candidates",
        str(args.gate_max_candidates),
        "--replay-top",
        str(args.gate_replay_top),
        "--include-axis-replay",
        "--event-summary-limit",
        str(args.event_summary_limit),
        "--out",
        str(paths["coordinate_gate"]),
    ]
    result = run_command(gate_cmd, WORKTREE_ROOT)
    commands.append(result)
    require_success(result)

    guard_cmd = [
        sys.executable,
        str(TASK_DIR / "ffe_public_repeated_coordinate_form_guard_replay_probe.py"),
        "--window",
        f"{args.window_name}|{paths['coordinate_gate']}",
        "--activation-rule",
        str(args.activation_rule),
        "--form-guard",
        str(args.form_guard),
        "--event-summary-limit",
        str(args.event_summary_limit),
        "--out",
        str(paths["guard_replay"]),
    ]
    for row_rule in row_rules:
        guard_cmd.extend(["--row-rule", row_rule])
    result = run_command(guard_cmd, WORKTREE_ROOT)
    commands.append(result)
    require_success(result)

    manifest = {
        "schema": "ecdlp_public_repeated_coordinate_frozen_pipeline_runner_v1",
        "method": "public_selector_to_coordinate_gate_to_frozen_form_guard_replay",
        "parameters": {
            "stress_source": str(args.stress_source),
            "window_name": str(args.window_name),
            "target": str(args.target),
            "max_ops_over_rho": float(args.max_ops_over_rho),
            "max_cases_per_challenge": int(args.max_cases_per_challenge),
            "allow_over_rho": bool(args.allow_over_rho),
            "gate_max_candidates": int(args.gate_max_candidates),
            "gate_replay_top": int(args.gate_replay_top),
            "activation_rule": str(args.activation_rule),
            "row_rules": row_rules,
            "form_guard": str(args.form_guard),
            "event_summary_limit": int(args.event_summary_limit),
        },
        "artifacts": {name: str(path) for name, path in paths.items() if name != "manifest"},
        "selector_summary": summarize_selector(paths["public_selector"]),
        "coordinate_gate_summary": summarize_gate(paths["coordinate_gate"]),
        "guard_replay_summary": summarize_guard(paths["guard_replay"]),
        "commands": commands,
        "non_claims": [
            "Fresh validation requires a stress source not used in guard discovery.",
            "The form guard is event-stage and does not avoid row-scan cost.",
            "This harness preserves exact command provenance for promotion tests.",
        ],
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps({
        "artifacts": manifest["artifacts"],
        "selector_summary": manifest["selector_summary"],
        "coordinate_gate_summary": manifest["coordinate_gate_summary"],
        "guard_replay_summary": manifest["guard_replay_summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
