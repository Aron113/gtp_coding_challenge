import base64
import binascii
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AsyncExitStack, asynccontextmanager
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel
import tiktoken

from ghost_chains import router as ghost_chains_router
from showdown.router import router as showdown_router
from tool_box import (
    NAME,
    answer_question,
    enforce_token_limit,
    solve_arithmetic,
    solve_shape,
    solve_shape_count,
)


ENCODING = tiktoken.get_encoding("o200k_base")
RETRIEVAL_TOKEN_BUDGET = 900
STUDY_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"
STUDY_URLS = [
    f"{STUDY_BASE_URL}/study-materials",
    f"{STUDY_BASE_URL}/study-materials/1",
    f"{STUDY_BASE_URL}/study-materials/2",
    f"{STUDY_BASE_URL}/study-materials/3",
    f"{STUDY_BASE_URL}/study-materials/4",
    f"{STUDY_BASE_URL}/study-materials/5",
]
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
}


def _token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def _normalize_terms(text: str) -> set[str]:
    terms = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    return {t for t in terms if len(t) > 1 and t not in STOP_WORDS}


def _fetch_text(url: str, timeout: float = 3.0) -> str:
    req = UrlRequest(url, headers={"User-Agent": "tool-box-nursery/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _chunk_text(text: str) -> list[str]:
    clean = text.replace("\r\n", "\n").strip()
    if not clean:
        return []

    blocks = [b.strip() for b in re.split(r"\n\s*\n+", clean) if b.strip()]
    chunks: list[str] = []
    bucket: list[str] = []
    bucket_len = 0

    for block in blocks:
        if len(block) > 1100:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
            for sent in sentences:
                if bucket_len and bucket_len + len(sent) + 1 > 900:
                    chunks.append(" ".join(bucket))
                    bucket = []
                    bucket_len = 0
                bucket.append(sent)
                bucket_len += len(sent) + 1
            continue

        if bucket_len and bucket_len + len(block) + 2 > 900:
            chunks.append("\n\n".join(bucket))
            bucket = []
            bucket_len = 0

        bucket.append(block)
        bucket_len += len(block) + 2

    if bucket:
        chunks.append("\n\n".join(bucket))

    return [c for c in chunks if c]


@lru_cache(maxsize=1)
def _study_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fetched: list[str] = []
    with ThreadPoolExecutor(max_workers=min(6, len(STUDY_URLS))) as pool:
        future_map = {
            pool.submit(_fetch_text, url, 2.0): url for url in STUDY_URLS
        }
        for fut in as_completed(future_map):
            try:
                fetched.append(fut.result())
            except Exception:
                continue

    for txt in fetched:
        for chunk in _chunk_text(txt):
            rows.append(
                {
                    "text": chunk,
                    "tokens": _token_count(chunk),
                    "terms": _normalize_terms(chunk),
                }
            )
    return rows


def _retrieve_passages(question: str, max_tokens: int = RETRIEVAL_TOKEN_BUDGET) -> list[str]:
    q_terms = _normalize_terms(question)
    rows = _study_index()
    if not rows:
        return []

    scored: list[tuple[float, int]] = []
    for i, row in enumerate(rows):
        terms = row["terms"]
        overlap = q_terms.intersection(terms)
        score = float(len(overlap))
        if question.lower() in row["text"].lower():
            score += 5.0
        if len(overlap) >= 2:
            score += 1.5
        if score > 0:
            scored.append((score, i))

    if not scored:
        # Fall back to short opening chunks if overlap misses due to naming mismatch.
        scored = [(0.01, i) for i in range(min(10, len(rows)))]

    scored.sort(key=lambda p: p[0], reverse=True)

    chosen: list[str] = []
    total = 0
    seen: set[str] = set()
    for _, idx in scored:
        row = rows[idx]
        txt = row["text"]
        cost = int(row["tokens"])
        if txt in seen or cost <= 0:
            continue
        if total + cost > max_tokens:
            continue
        chosen.append(txt)
        seen.add(txt)
        total += cost
        if total >= max_tokens - 20:
            break

    return chosen


def _extract_map_id(question: str) -> str | None:
    m = re.search(r"map_id\s*[:=]\s*([A-Za-z0-9\-_.]+)", question)
    if m:
        return m.group(1)
    m = re.search(r"\b([0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{12})\b", question)
    if m:
        return m.group(1)
    return None


def _extract_route_nodes(question: str) -> tuple[str | None, str | None]:
    patterns = [
        r"from\s+([A-Za-z0-9_\-]+)\s+to\s+([A-Za-z0-9_\-]+)",
        r"at\s+([A-Za-z0-9_\-]+)\s+.*\bto\s+([A-Za-z0-9_\-]+)",
        r"current(?:ly)?\s*(?:at|node)?\s*[:=]?\s*([A-Za-z0-9_\-]+).*?(?:dest(?:ination)?|target)\s*[:=]?\s*([A-Za-z0-9_\-]+)",
    ]
    lower = question.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            return m.group(1).upper(), m.group(2).upper()
    return None, None


def _extract_hops_left(question: str) -> int | None:
    patterns = [
        r"hops?\s+left\s*[:=]?\s*(\d+)",
        r"(\d+)\s+hops?\s+left",
        r"remaining\s+hops?\s*[:=]?\s*(\d+)",
        r"allowance\s*[:=]?\s*(\d+)",
        r"limit\s*[:=]?\s*(\d+)",
    ]
    lower = question.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            return int(m.group(1))
    return None


def _extract_visited(question: str) -> set[str]:
    visited: set[str] = set()
    m = re.search(r"visited\s*[:=]\s*([^\n]+)", question, flags=re.IGNORECASE)
    if not m:
        return visited
    candidates = re.findall(r"[A-Za-z0-9_\-]+", m.group(1))
    return {c.upper() for c in candidates}


def _graph_candidates(map_id: str) -> list[str]:
    encoded = quote_plus(map_id)
    return [
        f"{STUDY_BASE_URL}/graph?map_id={encoded}",
    ]


@lru_cache(maxsize=256)
def _fetch_graph(map_id: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for url in _graph_candidates(map_id):
        try:
            payload = _fetch_text(url)
            data = json.loads(payload)
            if isinstance(data, dict) and "adjacency" in data and "tolls" in data:
                return data
        except (HTTPError, URLError, ValueError) as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise RuntimeError("Unable to fetch graph")


def _normalize_graph(data: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    adjacency_raw = data.get("adjacency", {})
    tolls_raw = data.get("tolls", {})

    adjacency: dict[str, dict[str, float]] = {}
    for src, edges in adjacency_raw.items():
        src_u = str(src).upper()
        adjacency[src_u] = {}
        if isinstance(edges, dict):
            for dst, w in edges.items():
                try:
                    adjacency[src_u][str(dst).upper()] = float(w)
                except Exception:
                    continue

    tolls: dict[str, float] = {}
    for node, t in tolls_raw.items():
        try:
            tolls[str(node).upper()] = float(t)
        except Exception:
            tolls[str(node).upper()] = 0.0

    return adjacency, tolls


def _best_next_hop(
    adjacency: dict[str, dict[str, float]],
    tolls: dict[str, float],
    current: str,
    destination: str,
    hops_left: int | None,
    visited: set[str],
) -> str:
    current_u = current.upper()
    destination_u = destination.upper()
    if current_u == destination_u:
        return current_u

    neighbors = adjacency.get(current_u, {})
    legal_neighbors = [n for n in neighbors.keys() if n not in visited or n == destination_u]
    if not legal_neighbors:
        return current_u

    if hops_left is not None:
        if hops_left <= 0:
            return current_u

        @lru_cache(maxsize=None)
        def cost_with_limit(node: str, remaining: int) -> float:
            if node == destination_u:
                return 0.0
            if remaining == 0:
                return float("inf")

            best = float("inf")
            for nxt, edge_w in adjacency.get(node, {}).items():
                if nxt in visited and nxt != destination_u:
                    continue
                c = float(edge_w) + float(tolls.get(nxt, 0.0)) + cost_with_limit(nxt, remaining - 1)
                if c < best:
                    best = c
            return best

        best_hop = None
        best_cost = float("inf")
        for nxt in legal_neighbors:
            total = float(neighbors[nxt]) + float(tolls.get(nxt, 0.0)) + cost_with_limit(nxt, hops_left - 1)
            if total < best_cost:
                best_cost = total
                best_hop = nxt

        return best_hop if best_hop is not None else legal_neighbors[0]

    # No hop limit: Dijkstra over edge + entry toll cost.
    import heapq

    dist: dict[str, float] = {current_u: 0.0}
    first_hop: dict[str, str] = {}
    pq: list[tuple[float, str]] = [(0.0, current_u)]
    seen: set[str] = set()

    while pq:
        cost, node = heapq.heappop(pq)
        if node in seen:
            continue
        seen.add(node)

        if node == destination_u:
            break

        for nxt, edge_w in adjacency.get(node, {}).items():
            if nxt in visited and nxt != destination_u:
                continue
            step_cost = float(edge_w) + float(tolls.get(nxt, 0.0))
            new_cost = cost + step_cost
            if new_cost < dist.get(nxt, float("inf")):
                dist[nxt] = new_cost
                first_hop[nxt] = nxt if node == current_u else first_hop[node]
                heapq.heappush(pq, (new_cost, nxt))

    if destination_u in first_hop:
        return first_hop[destination_u]

    # If destination is unreachable, choose cheapest legal adjacent move.
    return min(legal_neighbors, key=lambda n: float(neighbors[n]) + float(tolls.get(n, 0.0)))


def _next_hop_from_question(
    question: str,
    current: str | None = None,
    destination: str | None = None,
    map_id: str | None = None,
    hops_left: int | None = None,
    visited: set[str] | None = None,
) -> str:
    src, dst = _extract_route_nodes(question)
    current = (current or src or "").upper()
    destination = (destination or dst or "").upper()
    map_id = map_id or _extract_map_id(question)
    if hops_left is None:
        hops_left = _extract_hops_left(question)
    merged_visited = set(visited or set())
    merged_visited.update(_extract_visited(question))

    if not current or not destination or not map_id:
        return current or destination or ""

    data = _fetch_graph(map_id)
    adjacency, tolls = _normalize_graph(data)

    # Ensure the current node is considered visited for future loop avoidance.
    merged_visited.add(current)
    return _best_next_hop(adjacency, tolls, current, destination, hops_left, merged_visited)

# Real MCP server (JSON-RPC over the Streamable HTTP transport), not a bespoke
# REST shim - the evaluator connects as an actual MCP client and does the
# initialize/tools-list/tools-call handshake. stateless_http=True so answers
# don't depend on in-memory session state surviving across requests/workers.
#
# transport_security is passed explicitly because FastMCP auto-enables DNS
# rebinding protection whenever `host` looks like localhost - and `host`
# defaults to "127.0.0.1", which we never set. That silently restricts the
# allowed Host header to 127.0.0.1/localhost, so every request behind a real
# domain (e.g. *.onrender.com) is rejected with "421 Misdirected Request".
# It passes locally and fails in production, which is exactly what happened.
mcp_server = FastMCP(
    name="Tool Box Nursery",
    instructions=(
        "A nursery-stage assistant. Use `get_name` for its name, `calculate` "
        "for arithmetic, `identify_shape` for what shape a base64 PNG shows, "
        "and `count_shapes` for how many shapes a base64 PNG contains. `ask` "
        "answers any of the above from the raw question text."
    ),
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)


# Every tool returns a plain string with structured_output=False. With a return
# annotation FastMCP also emits an outputSchema, and non-object results get
# wrapped as structuredContent {"result": ...}; a client that prefers structured
# content would then answer with that wrapper object instead of the bare value.
@mcp_server.tool(structured_output=False)
def get_name() -> str:
    """The assistant's name. Answers "What is your name?"."""
    return enforce_token_limit(NAME)


@mcp_server.tool(structured_output=False)
def calculate(expression: str) -> str:
    """Evaluate arithmetic and return the number, e.g. "2 + 2" -> "4".
    Supports +, -, *, / and parentheses. Accepts either a bare expression
    ("2 + 2") or the whole question ("What is 2 + 2?")."""
    return enforce_token_limit(str(solve_arithmetic(expression)))


@mcp_server.tool(structured_output=False)
def identify_shape(image_base64: str) -> str:
    """Identify the shape in a base64-encoded PNG. Returns exactly one of
    "rectangle", "triangle", or "circle"."""
    return enforce_token_limit(solve_shape(image_base64))


@mcp_server.tool(structured_output=False)
def count_shapes(image_base64: str) -> str:
    """Count how many shapes a base64-encoded PNG contains. Returns the
    count as a number, e.g. "3"."""
    return enforce_token_limit(str(solve_shape_count(image_base64)))


@mcp_server.tool(structured_output=False)
def ask(question: str, image: str | None = None) -> str:
    """Answer any nursery question from its raw text: the assistant's name,
    arithmetic (+, -, *, /), or the shape / shape count of a base64-encoded
    PNG. Pass the PNG via `image`, or embedded in `question`."""
    lower = question.lower()
    if any(k in lower for k in ["map_id", "get from", "from ", " to "]):
        return enforce_token_limit(_next_hop_from_question(question))
    if any(k in lower for k in ["study", "material", "revise", "exam", "when was", "where was", "who was"]):
        return enforce_token_limit(json.dumps(_retrieve_passages(question), ensure_ascii=True))

    payload: dict[str, Any] = {"question": question}
    if image:
        payload["image"] = image
    return enforce_token_limit(str(answer_question(payload)))


@mcp_server.tool(structured_output=False)
def retrieve_passages(question: str) -> str:
    """Return relevant study passages for a revision question as a JSON array
    of strings, with a strict total budget <= 900 tokens using o200k_base."""
    passages = _retrieve_passages(question, RETRIEVAL_TOKEN_BUDGET)
    return enforce_token_limit(json.dumps(passages, ensure_ascii=True))


@mcp_server.tool(structured_output=False)
def next_hop(
    question: str,
    current: str | None = None,
    destination: str | None = None,
    map_id: str | None = None,
    hops_left: int | None = None,
    visited: list[str] | None = None,
) -> str:
    """Return the next adjacent node toward the destination for the map in
    question/map_id. Cost is edge weights plus destination-node entry tolls.
    If hops_left is provided, returns the best constrained next step."""
    visited_set = {v.upper() for v in (visited or [])}
    hop = _next_hop_from_question(
        question=question,
        current=current,
        destination=destination,
        map_id=map_id,
        hops_left=hops_left,
        visited=visited_set,
    )
    return enforce_token_limit(hop)


mcp_app = mcp_server.streamable_http_app()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(mcp_server.session_manager.run())
        yield


app = FastAPI(title="Tool Box Nursery", version="1.0.0", lifespan=lifespan)
app.include_router(ghost_chains_router)
app.include_router(showdown_router)


class SquareRequest(BaseModel):
    number: int | float


class SquareResponse(BaseModel):
    answer: int | float


class SolveRequest(BaseModel):
    payload: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/event")
async def event(request: Request) -> dict[str, bool]:
    """Telemetry receiver for run progress events."""
    try:
        data = await request.json()
        problem = data.get("problem", "unknown")
        attempt = data.get("attempt", 1)
        print(f"[Telemetry] Problem: {problem} | Attempt: {attempt}")
    except Exception:
        pass
    return {"ok": True}


@app.post("/callback")
async def callback(request: Request) -> dict[str, Any]:
    """Evaluation result JSON receiver."""
    try:
        data = await request.json()
        print(f"[Evaluation Result] {data}")
    except Exception:
        pass
    return {"status": "ok"}


@app.post("/square", response_model=SquareResponse)
def square(payload: SquareRequest) -> SquareResponse:
    return SquareResponse(answer=payload.number * payload.number)


PRIORITY_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
DEFAULT_PRIORITY = 2


def build_adapt_output(adapt_input: dict) -> dict:
    user = adapt_input.get("user", {})
    metadata = adapt_input.get("metadata", {})

    priority = PRIORITY_MAP.get(metadata.get("priority"), DEFAULT_PRIORITY)
    action = str(adapt_input.get("action", "")).lower()

    return {
        "id": user.get("id"),
        "name": user.get("fullName"),
        "action": action,
        "priority": priority,
    }


def build_slo_output(heartbeats: list, slo_query: dict) -> dict:
    service = slo_query.get("service")
    since = slo_query.get("since")

    deduped = {}
    for hb in heartbeats:
        if hb.get("service") != service:
            continue
        timestamp = hb.get("timestamp")
        if since is not None and timestamp < since:
            continue
        deduped.setdefault((hb.get("service"), timestamp), hb)

    rows = list(deduped.values())
    if not rows:
        return {"availability": 0.0, "p95LatencyMs": 0}

    ok_count = sum(1 for row in rows if row.get("status") == "OK")
    availability = ok_count / len(rows)

    latencies = sorted(row.get("latencyMs") for row in rows)
    rank = math.ceil(0.95 * len(latencies))
    p95_latency = latencies[rank - 1]

    return {"availability": availability, "p95LatencyMs": p95_latency}


@app.post("/solve")
def solve(request: SolveRequest) -> dict:
    try:
        decoded = base64.b64decode(request.payload)
        data = json.loads(decoded)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        raise HTTPException(
            status_code=400, detail="payload must be base64-encoded JSON"
        )

    adapt_output = build_adapt_output(data.get("adaptInput", {}))
    slo_output = build_slo_output(
        data.get("heartbeats", []), data.get("sloQuery", {})
    )

    return {
        "adaptOutput": adapt_output,
        "sloOutput": slo_output,
    }


# Mounted last, and at "/" rather than "/mcp": FastMCP's streamable_http_app()
# only defines a route at its internal streamable_http_path (default "/mcp").
# Mounting *that* app at "/mcp" would require Starlette to strip the "/mcp"
# prefix and match the remainder against "/", which only matches paths with a
# trailing slash ("/mcp/") - a bare POST /mcp 307-redirects to "/mcp/" first,
# and most HTTP/MCP clients refuse to auto-follow a 307 on a POST, so the
# request just hangs. Mounting at "/" instead means "/mcp" matches the sub-app's
# own "/mcp" route directly, with no redirect. Must come after every other
# route above, since a "/" mount would otherwise shadow all of them.
app.mount("/", mcp_app)