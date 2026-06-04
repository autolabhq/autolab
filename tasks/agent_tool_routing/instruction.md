# Agent Tool Routing

Optimize `/app/retriever.py` to route natural-language agent requests to the right tool schemas.

## Problem

Modern agents often have hundreds or thousands of available tools: GitHub, Slack, calendars, databases, filesystems, browsers, search APIs, monitoring systems, and internal services. Before the agent can call a tool, it needs a fast lexical router that returns the most relevant tool schemas for the current request.

The public driver includes a development workload for iteration. The verifier uses a separate hidden workload: it sends only tool schemas and query text to `/app/main.py --rank-stdin` and keeps gold tool IDs outside `/app`. For each query, your retriever must return the top 10 candidate tool IDs. The verifier computes MRR@10 and Recall@10 against hidden gold IDs.

## Files

| Path | Permission |
|------|------------|
| `/app/retriever.py` | **Edit** |
| `/app/main.py` | Read-only |

## API

Implement these functions in `retriever.py`:

```python
def build_index(tools):
    ...

def retrieve_tools(index, query, k):
    ...
```

`tools` is a list of dictionaries. Each dictionary has:

```python
{
    "id": int,
    "name": str,
    "domain": str,
    "description": str,
    "parameters": str,
}
```

`build_index()` is called once before a batch of queries. It may return any Python object. `retrieve_tools()` must return a list of up to `k` integer tool IDs in ranked order. Use `-1` only if you cannot fill all `k` slots.

## Goal

Minimize hidden benchmark runtime while passing the quality gate:

```text
MRR@10    >= 0.82
Recall@10 >= 0.94
```

The starter solution is intentionally simple but slow.

## Rules

- Edit only `/app/retriever.py`.
- Do not modify the driver.
- Python standard library only. No external packages are required or allowed.
- Single process, single threaded.
- No external network access.
- The verifier workload is hidden and query order is shuffled. Implement a real retrieval method rather than relying on query counters, public IDs, or public data generation details.
- Time budget: 2 hours
