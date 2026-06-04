#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass


TOP_K = 10
QUALITY_TOOLS = 1100
QUALITY_QUERIES = 400
BENCH_TOOLS = 3600
BENCH_QUERIES = 1200
BENCH_RUNS = 3
MRR_GATE = 0.82
RECALL_GATE = 0.94
BASELINE = 3.85
REFERENCE = 0.40

SERVICES = [
    ("github", "code_hosting", "repo", "pull request system", "repository"),
    ("gitlab", "code_hosting", "project", "merge request system", "project_id"),
    ("bitbucket", "code_hosting", "workspace repo", "source branch", "workspace"),
    ("linear", "project", "issue tracker", "roadmap", "team_id"),
    ("jira", "project", "ticket board", "sprint", "project_key"),
    ("asana", "project", "task board", "portfolio", "workspace_id"),
    ("slack", "chat", "channel", "workspace chat", "channel_id"),
    ("teams", "chat", "team channel", "meeting chat", "team_id"),
    ("discord", "chat", "server channel", "community chat", "channel_id"),
    ("gmail", "email", "mailbox", "inbox", "email_address"),
    ("outlook", "email", "mail calendar", "exchange", "account_id"),
    ("calendar", "calendar", "schedule", "meeting time", "calendar_id"),
    ("drive", "files", "cloud file", "document storage", "file_id"),
    ("dropbox", "files", "shared file", "folder", "path"),
    ("box", "files", "enterprise file", "shared folder", "file_id"),
    ("notion", "docs", "workspace page", "database page", "page_id"),
    ("confluence", "docs", "space page", "wiki", "space_key"),
    ("postgres", "database", "sql database", "table schema", "connection"),
    ("mysql", "database", "sql instance", "table", "connection"),
    ("redis", "database", "cache key", "kv store", "key"),
    ("snowflake", "warehouse", "data warehouse", "analytics table", "warehouse"),
    ("bigquery", "warehouse", "dataset table", "analytics sql", "dataset"),
    ("s3", "storage", "bucket object", "blob storage", "bucket"),
    ("kubernetes", "infra", "cluster pod", "namespace", "namespace"),
    ("aws", "infra", "cloud resource", "iam", "account_id"),
    ("sentry", "observability", "error event", "issue alert", "project"),
    ("datadog", "observability", "metric monitor", "dashboard", "monitor_id"),
    ("browser", "web", "web page", "dom element", "url"),
    ("search", "web", "web search", "answer engine", "query"),
    ("filesystem", "local_files", "local path", "workspace file", "path"),
    ("shell", "local_exec", "terminal", "command runner", "command"),
    ("python", "code_exec", "python runtime", "notebook", "code"),
]

ACTIONS = [
    ("create", "create", "open", "make"),
    ("update", "update", "edit", "change"),
    ("delete", "delete", "remove", "archive"),
    ("search", "search", "find", "look up"),
    ("list", "list", "show", "enumerate"),
    ("get", "get", "fetch", "read"),
    ("send", "send", "deliver", "email"),
    ("post", "post", "publish", "announce"),
    ("upload", "upload", "attach", "add file"),
    ("download", "download", "export", "save"),
    ("run", "run", "execute", "launch"),
    ("query", "query", "select", "ask"),
    ("summarize", "summarize", "digest", "brief"),
    ("schedule", "schedule", "book", "plan"),
    ("comment", "comment", "reply", "respond"),
    ("approve", "approve", "accept", "confirm"),
]

OBJECTS = [
    ("pull_request", "pull request", "pr", "code review", "pull_number"),
    ("issue", "issue", "ticket", "bug report", "issue_id"),
    ("message", "message", "chat note", "announcement", "text"),
    ("email", "email", "mail", "inbox item", "subject"),
    ("event", "event", "meeting", "calendar slot", "start_time"),
    ("file", "file", "document", "attachment", "path"),
    ("folder", "folder", "directory", "collection", "folder_id"),
    ("page", "page", "doc page", "wiki article", "page_id"),
    ("database", "database", "data source", "workspace db", "database_id"),
    ("table", "table", "relation", "schema table", "table_name"),
    ("query", "query", "search request", "lookup", "query_text"),
    ("record", "record", "row", "object", "record_id"),
    ("branch", "branch", "git ref", "source ref", "branch_name"),
    ("repository", "repository", "repo", "project", "repo"),
    ("metric", "metric", "timeseries", "measurement", "metric_name"),
    ("alert", "alert", "incident", "warning", "alert_id"),
    ("dashboard", "dashboard", "report", "chart board", "dashboard_id"),
    ("command", "command", "terminal command", "shell step", "command"),
]

VARIANTS = [
    ("default", "standard", "normal", "dry_run"),
    ("draft", "draft", "unpublished", "draft_id"),
    ("bulk", "bulk", "batch", "batch_size"),
    ("metadata", "metadata", "summary", "include_metadata"),
    ("permission", "permission", "access", "permission_scope"),
    ("thread", "thread", "reply", "thread_id"),
]


@dataclass
class Tool:
    id: int
    name: str
    domain: str
    description: str
    parameters: str
    combo: tuple[int, int, int, int]


@dataclass
class Query:
    id: int
    text: str
    gold_id: int


def clean(s: str, limit: int) -> str:
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return s[: limit - 1]


def seed_value() -> str:
    return os.environ.get("TOOL_ROUTER_SECRET_SEED", "autolab-agent-tool-routing-local-v1")


def make_rng(split: str) -> random.Random:
    return random.Random(f"{seed_value()}::{split}")


def seed_tag() -> str:
    return hashlib.sha256(seed_value().encode("utf-8")).hexdigest()[:12]


def all_base_triples():
    triples = []
    for si in range(len(SERVICES)):
        for ai in range(len(ACTIONS)):
            for oi in range(len(OBJECTS)):
                triples.append((si, ai, oi))
    return triples


def make_tool(tool_id: int, combo: tuple[int, int, int, int]) -> Tool:
    si, ai, oi, vi = combo
    svc = SERVICES[si]
    act = ACTIONS[ai]
    obj = OBJECTS[oi]
    var = VARIANTS[vi]
    suffix = "" if var[0] == "default" else f"_{var[0]}"
    name = f"{svc[0]}.{act[0]}_{obj[0]}{suffix}"
    desc = (
        f"{act[1]} {obj[1]} in {svc[0]}. Use for {svc[2]}, {svc[3]}, and {svc[1]} workflows. "
        f"Action aliases include {act[2]} and {act[3]}. Object aliases include {obj[2]} and {obj[3]}. "
        f"Variant terms: {var[1]}, {var[2]}."
    )
    params = (
        f"{svc[4]} string; {obj[4]} string; action_{act[0]} string; object_{obj[0]} string; "
        f"mode_{var[0]} string; {var[3]} string; {act[2]}; {act[3]}; {obj[2]}; {obj[3]}; {svc[2]}"
    )
    return Tool(
        id=tool_id,
        name=clean(name, 80),
        domain=clean(svc[1], 32),
        description=clean(desc, 320),
        parameters=clean(params, 220),
        combo=combo,
    )


def generate_tools(n_tools: int, split: str) -> list[Tool]:
    rng = make_rng(f"{split}:tools")
    triples = all_base_triples()
    rng.shuffle(triples)
    combos = []
    for triple in triples:
        variants = list(range(len(VARIANTS)))
        rng.shuffle(variants)
        for vi in variants:
            combos.append((*triple, vi))
            if len(combos) == n_tools:
                break
        if len(combos) == n_tools:
            break
    public_ids = list(range(n_tools))
    rng.shuffle(public_ids)
    return [make_tool(public_ids[i], combos[i]) for i in range(n_tools)]


def query_for_tool(qid: int, tool: Tool, rng: random.Random) -> Query:
    si, ai, oi, vi = tool.combo
    svc = SERVICES[si]
    act = ACTIONS[ai]
    obj = OBJECTS[oi]
    var = VARIANTS[vi]
    verb = rng.choice([act[1], act[0], act[2], act[3]])
    thing = rng.choice([obj[1], obj[0].replace("_", " "), obj[2], obj[3]])
    place = rng.choice([svc[0], f"{svc[0]} {svc[2]}", f"{svc[0]} {svc[3]}"])
    mode = rng.choice([var[0], var[1], var[2], var[3].replace("_", " ")])
    param = rng.choice([svc[4].replace("_", " "), obj[4].replace("_", " "), var[3].replace("_", " ")])
    distractor = rng.choice(["", "", "audit note", "planner step", "from prior run"])
    styles = [
        f"need to {verb} the {thing} in {place} with {mode} handling {param} {distractor}",
        f"{place}: {verb} {thing}; include {param} and use {mode}",
        f"route this agent step to {verb} a {thing} for {place}, {mode} path, {param}",
        f"tool for {verb} {thing} using {place}; parameters mention {param} and {mode}",
        f"please {verb} {thing} from {place} workspace, prefer {mode} variant and {param}",
        f"{mode} {place} {thing} operation; user wants to {verb}; parameter {param}",
    ]
    return Query(id=qid, text=clean(rng.choice(styles), 512), gold_id=tool.id)


def generate_queries(tools: list[Tool], n_queries: int, split: str) -> list[Query]:
    rng = make_rng(f"{split}:queries")
    queries = [query_for_tool(i, rng.choice(tools), rng) for i in range(n_queries)]
    rng.shuffle(queries)
    for i, query in enumerate(queries):
        query.id = i
    return queries


def payload_for(tools: list[Tool], queries: list[Query]) -> str:
    lines = [f"TSRPY1 {len(tools)} {len(queries)} {TOP_K}"]
    for tool in tools:
        lines.append(json.dumps({
            "id": tool.id,
            "name": tool.name,
            "domain": tool.domain,
            "description": tool.description,
            "parameters": tool.parameters,
        }, separators=(",", ":")))
    for query in queries:
        lines.append(json.dumps({"id": query.id, "text": query.text}, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def mix64(x: int) -> int:
    x &= (1 << 64) - 1
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
    x ^= x >> 31
    return x & ((1 << 64) - 1)


def update_checksum(h: int, qid: int, rank: int, tid: int) -> int:
    x = (tid + 1009) ^ ((qid + 17) << 21) ^ ((rank + 3) << 47)
    h ^= mix64(x)
    h = (h * 1099511628211) & ((1 << 64) - 1)
    return h


def run_ranker(tools: list[Tool], queries: list[Query]) -> dict:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
    }
    proc = subprocess.run(
        [sys.executable, "/app/main.py", "--rank-stdin"],
        input=payload_for(tools, queries),
        text=True,
        capture_output=True,
        timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ranker failed rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")

    gold = {q.id: q.gold_id for q in queries}
    seen_queries = set()
    hits = 0
    rr_sum = 0.0
    checksum = 1469598103934665603

    for line in proc.stdout.splitlines():
        if not line.startswith("Q "):
            continue
        parts = line.split()
        if len(parts) < 2 + TOP_K:
            raise RuntimeError(f"bad result line: {line}")
        qid = int(parts[1])
        ids = [int(x) for x in parts[2 : 2 + TOP_K]]
        if qid not in gold:
            raise RuntimeError(f"unknown query id: {qid}")
        if qid in seen_queries:
            raise RuntimeError(f"duplicate query id: {qid}")
        seen_queries.add(qid)
        found = False
        for rank, tid in enumerate(ids):
            checksum = update_checksum(checksum, qid, rank, tid)
            if tid == gold[qid] and not found:
                hits += 1
                rr_sum += 1.0 / (rank + 1)
                found = True

    if len(seen_queries) != len(queries):
        raise RuntimeError(f"missing results: got {len(seen_queries)} expected {len(queries)}")

    m = re.search(r"time=([0-9.]+)", proc.stderr)
    if not m:
        raise RuntimeError(f"ranker did not report time on stderr: {proc.stderr}")
    return {
        "metric": float(m.group(1)),
        "mrr": rr_sum / len(queries),
        "recall": hits / len(queries),
        "checksum": checksum,
    }


def score_reward(metric: float) -> tuple[float, float]:
    speedup = BASELINE / metric
    ref_speedup = BASELINE / REFERENCE
    reward = 0.0 if speedup <= 1.0 else min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
    return round(max(0.0, reward), 4), round(speedup, 2)


def write_reward(path: str, txt_path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(payload.get("reward", 0.0)))


def evaluate_split(name: str, n_tools: int, n_queries: int) -> dict:
    tools = generate_tools(n_tools, name)
    queries = generate_queries(tools, n_queries, name)
    return run_ranker(tools, queries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-file", default="/logs/verifier/reward.json")
    parser.add_argument("--reward-txt", default="/logs/verifier/reward.txt")
    args = parser.parse_args()

    print(f"seed_tag={seed_tag()}")
    quality = evaluate_split("quality", QUALITY_TOOLS, QUALITY_QUERIES)
    print(
        "quality "
        f"time={quality['metric']:.6f} mrr={quality['mrr']:.6f} "
        f"recall={quality['recall']:.6f} checksum={quality['checksum']}"
    )
    if quality["mrr"] < MRR_GATE or quality["recall"] < RECALL_GATE:
        write_reward(
            args.reward_file,
            args.reward_txt,
            {
                "reward": 0.0,
                "correctness": False,
                "reason": "quality_gate_failed",
                "metric": None,
                "speedup": None,
                "mrr": round(quality["mrr"], 6),
                "recall": round(quality["recall"], 6),
            },
        )
        return 0

    bench_tools = generate_tools(BENCH_TOOLS, "benchmark")
    bench_queries = generate_queries(bench_tools, BENCH_QUERIES, "benchmark")
    runs = []
    for i in range(BENCH_RUNS):
        result = run_ranker(bench_tools, bench_queries)
        runs.append(result)
        print(
            f"run {i + 1}: time={result['metric']:.6f} "
            f"mrr={result['mrr']:.6f} recall={result['recall']:.6f} "
            f"checksum={result['checksum']}"
        )
        if result["mrr"] < MRR_GATE or result["recall"] < RECALL_GATE:
            write_reward(
                args.reward_file,
                args.reward_txt,
                {
                    "reward": 0.0,
                    "correctness": False,
                    "reason": "benchmark_quality_gate_failed",
                    "metric": result["metric"],
                    "speedup": None,
                    "mrr": round(result["mrr"], 6),
                    "recall": round(result["recall"], 6),
                },
            )
            return 0

    metrics = sorted(r["metric"] for r in runs)
    median = metrics[len(metrics) // 2]
    chosen = min(runs, key=lambda r: abs(r["metric"] - median))
    reward, speedup = score_reward(median)
    payload = {
        "reward": reward,
        "correctness": True,
        "metric": round(median, 6),
        "speedup": speedup,
        "mrr": round(chosen["mrr"], 6),
        "recall": round(chosen["recall"], 6),
        "checksum": chosen["checksum"],
    }
    print(
        f"median={median:.6f} speedup={speedup} reward={reward} "
        f"mrr={payload['mrr']} recall={payload['recall']}"
    )
    write_reward(args.reward_file, args.reward_txt, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
