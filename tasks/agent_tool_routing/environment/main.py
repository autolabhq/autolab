#!/usr/bin/env python3
import json
import random
import sys
import time

import retriever


TOP_K = 10
PUBLIC_TOOLS = 650
PUBLIC_QUERIES = 220

SERVICES = [
    ("github", "code_hosting", "repo", "pull request system", "repository"),
    ("gitlab", "code_hosting", "project", "merge request system", "project_id"),
    ("linear", "project", "issue tracker", "roadmap", "team_id"),
    ("jira", "project", "ticket board", "sprint", "project_key"),
    ("slack", "chat", "channel", "workspace chat", "channel_id"),
    ("gmail", "email", "mailbox", "inbox", "email_address"),
    ("calendar", "calendar", "schedule", "meeting time", "calendar_id"),
    ("drive", "files", "cloud file", "document storage", "file_id"),
    ("notion", "docs", "workspace page", "database page", "page_id"),
    ("postgres", "database", "sql database", "table schema", "connection"),
    ("s3", "storage", "bucket object", "blob storage", "bucket"),
    ("sentry", "observability", "error event", "issue alert", "project"),
    ("browser", "web", "web page", "dom element", "url"),
    ("search", "web", "web search", "answer engine", "query"),
    ("filesystem", "local_files", "local path", "workspace file", "path"),
    ("shell", "local_exec", "terminal", "command runner", "command"),
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
]

OBJECTS = [
    ("pull_request", "pull request", "pr", "code review", "pull_number"),
    ("issue", "issue", "ticket", "bug report", "issue_id"),
    ("message", "message", "chat note", "announcement", "text"),
    ("email", "email", "mail", "inbox item", "subject"),
    ("event", "event", "meeting", "calendar slot", "start_time"),
    ("file", "file", "document", "attachment", "path"),
    ("page", "page", "doc page", "wiki article", "page_id"),
    ("database", "database", "data source", "workspace db", "database_id"),
    ("table", "table", "relation", "schema table", "table_name"),
    ("query", "query", "search request", "lookup", "query_text"),
    ("branch", "branch", "git ref", "source ref", "branch_name"),
    ("alert", "alert", "incident", "warning", "alert_id"),
]

VARIANTS = [
    ("default", "standard", "normal", "dry_run"),
    ("draft", "draft", "unpublished", "draft_id"),
    ("bulk", "bulk", "batch", "batch_size"),
    ("metadata", "metadata", "summary", "include_metadata"),
    ("permission", "permission", "access", "permission_scope"),
]


def clean(s, limit):
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")[: limit - 1]


def all_combos():
    combos = []
    for si in range(len(SERVICES)):
        for ai in range(len(ACTIONS)):
            for oi in range(len(OBJECTS)):
                combos.append((si, ai, oi))
    return combos


def make_tool(tool_id, combo):
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
    return {
        "id": tool_id,
        "name": clean(name, 80),
        "domain": clean(svc[1], 32),
        "description": clean(desc, 320),
        "parameters": clean(params, 220),
        "_combo": combo,
    }


def generate_tools(n_tools, seed):
    rng = random.Random(seed)
    base = all_combos()
    rng.shuffle(base)
    combos = []
    for triple in base:
        order = list(range(len(VARIANTS)))
        rng.shuffle(order)
        for vi in order:
            combos.append((*triple, vi))
            if len(combos) == n_tools:
                break
        if len(combos) == n_tools:
            break

    ids = list(range(n_tools))
    rng.shuffle(ids)
    return [make_tool(ids[i], combos[i]) for i in range(n_tools)]


def query_for_tool(qid, tool, rng):
    si, ai, oi, vi = tool["_combo"]
    svc = SERVICES[si]
    act = ACTIONS[ai]
    obj = OBJECTS[oi]
    var = VARIANTS[vi]
    verb = rng.choice([act[1], act[0], act[2], act[3]])
    thing = rng.choice([obj[1], obj[0].replace("_", " "), obj[2], obj[3]])
    place = rng.choice([svc[0], f"{svc[0]} {svc[2]}", f"{svc[0]} {svc[3]}"])
    mode = rng.choice([var[0], var[1], var[2], var[3].replace("_", " ")])
    param = rng.choice([svc[4].replace("_", " "), obj[4].replace("_", " "), var[3].replace("_", " ")])
    styles = [
        f"need to {verb} the {thing} in {place} with {mode} handling {param}",
        f"{place}: {verb} {thing}; include {param} and use {mode}",
        f"route this agent step to {verb} a {thing} for {place}, {mode} path, {param}",
        f"tool for {verb} {thing} using {place}; parameters mention {param} and {mode}",
        f"please {verb} {thing} from {place} workspace, prefer {mode} variant and {param}",
        f"{mode} {place} {thing} operation; user wants to {verb}; parameter {param}",
    ]
    return {"id": qid, "text": clean(rng.choice(styles), 512), "gold_id": tool["id"]}


def generate_queries(tools, n_queries, seed):
    rng = random.Random(seed)
    queries = [query_for_tool(i, rng.choice(tools), rng) for i in range(n_queries)]
    rng.shuffle(queries)
    for i, query in enumerate(queries):
        query["id"] = i
    return queries


def mix64(x):
    x &= (1 << 64) - 1
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
    x ^= x >> 31
    return x & ((1 << 64) - 1)


def update_checksum(h, qid, rank, tid):
    x = (tid + 1009) ^ ((qid + 17) << 21) ^ ((rank + 3) << 47)
    h ^= mix64(x)
    h = (h * 1099511628211) & ((1 << 64) - 1)
    return h


def normalize_result(ids, k):
    if ids is None:
        ids = []
    ids = list(ids)[:k]
    out = []
    for value in ids:
        try:
            out.append(int(value))
        except Exception:
            out.append(-1)
    if len(out) < k:
        out.extend([-1] * (k - len(out)))
    return out


def evaluate(tools, queries, k):
    public_tools = [{key: value for key, value in t.items() if not key.startswith("_")} for t in tools]
    gold = {q["id"]: q["gold_id"] for q in queries}
    t0 = time.perf_counter()
    index = retriever.build_index(public_tools)
    hits = 0
    rr_sum = 0.0
    checksum = 1469598103934665603
    for query in queries:
        ids = normalize_result(retriever.retrieve_tools(index, query["text"], k), k)
        found = False
        for rank, tid in enumerate(ids):
            checksum = update_checksum(checksum, query["id"], rank, tid)
            if tid == gold[query["id"]] and not found:
                hits += 1
                rr_sum += 1.0 / (rank + 1)
                found = True
    free_fn = getattr(retriever, "free_index", None)
    if callable(free_fn):
        free_fn(index)
    elapsed = time.perf_counter() - t0
    return {
        "metric": elapsed,
        "mrr": rr_sum / len(queries),
        "recall": hits / len(queries),
        "checksum": checksum,
    }


def run_verify():
    tools = generate_tools(PUBLIC_TOOLS, "public-tools-v1")
    queries = generate_queries(tools, PUBLIC_QUERIES, "public-queries-v1")
    result = evaluate(tools, queries, TOP_K)
    print(
        f"verify time={result['metric']:.6f} mrr={result['mrr']:.6f} "
        f"recall={result['recall']:.6f} checksum={result['checksum']}"
    )
    return 0


def run_rank_stdin():
    header = sys.stdin.readline()
    if not header:
        return 2
    parts = header.strip().split()
    if len(parts) != 4 or parts[0] != "TSRPY1":
        return 2
    n_tools, n_queries, k = map(int, parts[1:])
    if n_tools <= 0 or n_queries < 0 or k <= 0 or k > 50:
        return 2

    tools = [json.loads(sys.stdin.readline()) for _ in range(n_tools)]
    queries = [json.loads(sys.stdin.readline()) for _ in range(n_queries)]

    t0 = time.perf_counter()
    index = retriever.build_index(tools)
    print("BEGIN_RESULTS")
    for query in queries:
        ids = normalize_result(retriever.retrieve_tools(index, query["text"], k), k)
        print("Q", query["id"], *ids)
    free_fn = getattr(retriever, "free_index", None)
    if callable(free_fn):
        free_fn(index)
    elapsed = time.perf_counter() - t0
    print(f"time={elapsed:.6f} queries={n_queries} tools={n_tools}", file=sys.stderr)
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--rank-stdin":
        return run_rank_stdin()
    return run_verify()


if __name__ == "__main__":
    raise SystemExit(main())
