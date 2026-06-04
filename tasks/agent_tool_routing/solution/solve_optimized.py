import heapq
import math
import re
from collections import defaultdict


TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "please", "prefer",
    "route", "step", "that", "the", "this", "to", "tool", "use",
    "using", "user", "wants", "with", "workspace", "agent", "need",
    "needs", "include", "includes", "parameter", "parameters", "string",
    "bool", "handling", "handle", "path", "operation", "request",
}

FIELD_WEIGHTS = {
    "name": 9,
    "parameters": 5,
    "domain": 3,
    "description": 2,
}


def tokenize(text):
    out = []
    for match in TOKEN_RE.finditer(text.lower()):
        tok = match.group(0)
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        if len(tok) >= 2 and tok not in STOPWORDS:
            out.append(tok)
    return out


def build_index(tools):
    token_weights = defaultdict(dict)
    max_id = -1

    for tool in tools:
        tool_id = int(tool["id"])
        max_id = max(max_id, tool_id)
        for field, weight in FIELD_WEIGHTS.items():
            for tok in set(tokenize(tool.get(field, ""))):
                token_weights[tok][tool_id] = token_weights[tok].get(tool_id, 0) + weight

    n_tools = max(1, len(tools))
    postings = {}
    for tok, weights in token_weights.items():
        # Downweight broad tokens that appear in many tools while keeping scores
        # as integers for faster accumulation.
        idf = 1.0 + math.log((1.0 + n_tools) / (1.0 + len(weights)))
        postings[tok] = [(tool_id, int(score * idf * 256.0)) for tool_id, score in weights.items()]

    return {
        "postings": postings,
        "scores": [0] * (max_id + 1),
        "seen": [0] * (max_id + 1),
        "stamp": 0,
    }


def retrieve_tools(index, query, k):
    tokens = list(dict.fromkeys(tokenize(query)))
    if not tokens:
        return [-1] * k

    index["stamp"] += 1
    stamp = index["stamp"]
    scores = index["scores"]
    seen = index["seen"]
    touched = []

    for tok in tokens:
        for tool_id, weight in index["postings"].get(tok, ()):
            if seen[tool_id] != stamp:
                seen[tool_id] = stamp
                scores[tool_id] = 0
                touched.append(tool_id)
            scores[tool_id] += weight

    if not touched:
        return [-1] * k

    best = heapq.nlargest(k, touched, key=lambda tool_id: (scores[tool_id], -tool_id))
    if len(best) < k:
        best.extend([-1] * (k - len(best)))
    return best
