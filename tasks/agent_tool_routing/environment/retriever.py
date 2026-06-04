import re


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
    tokens = []
    for match in TOKEN_RE.finditer(text.lower()):
        tok = match.group(0)
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        if len(tok) >= 2 and tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


def build_index(tools):
    prepared = []
    for tool in tools:
        prepared.append({
            "id": int(tool["id"]),
            "fields": {
                field: set(tokenize(tool.get(field, "")))
                for field in FIELD_WEIGHTS
            },
        })
    return {"tools": prepared}


def _score_tool(entry, query_tokens):
    score = 0
    for field, weight in FIELD_WEIGHTS.items():
        field_tokens = entry["fields"][field]
        if not field_tokens:
            continue
        for tok in query_tokens:
            if tok in field_tokens:
                score += weight
    return score


def retrieve_tools(index, query, k):
    query_tokens = list(dict.fromkeys(tokenize(query)))
    if not query_tokens:
        return [-1] * k

    scored = []
    for entry in index["tools"]:
        score = _score_tool(entry, query_tokens)
        if score > 0:
            scored.append((score, -entry["id"], entry["id"]))

    scored.sort(reverse=True)
    out = [tool_id for _, _, tool_id in scored[:k]]
    if len(out) < k:
        out.extend([-1] * (k - len(out)))
    return out
