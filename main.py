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
from kan_chiong_delivery import solve as kan_chiong_solve
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


def _fetch_text(url: str, timeout: float = 3.5) -> str:
    req = UrlRequest(url, headers={"User-Agent": "tool-box-nursery/1.0", "Accept": "*/*"})
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

    merged_visited.add(current)
    return _best_next_hop(adjacency, tolls, current, destination, hops_left, merged_visited)


# ---------------------------------------------------------
# Stage 3: Calendar & Scheduling Helpers
# ---------------------------------------------------------

def _hhmm_to_minutes(value: str) -> int:
    parts = value.strip().split(":", 1)
    return int(parts[0]) * 60 + int(parts[1])


def _minutes_to_hhmm(value: int) -> str:
    hh = value // 60
    mm = value % 60
    return f"{hh:02d}:{mm:02d}"


def _extract_meeting_request(question: str) -> dict[str, Any]:
    day_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question)
    day = day_match.group(1) if day_match else ""

    duration_match = re.search(r"(\d+)\s*[- ]?minute", question, flags=re.IGNORECASE)
    duration = int(duration_match.group(1)) if duration_match else 60

    between_match = re.search(
        r"between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})",
        question,
        flags=re.IGNORECASE,
    )
    start_bound = between_match.group(1) if between_match else "08:00"
    end_bound = between_match.group(2) if between_match else "18:00"

    if len(start_bound.split(":")[0]) == 1:
        start_bound = f"0{start_bound}"
    if len(end_bound.split(":")[0]) == 1:
        end_bound = f"0{end_bound}"

    people: list[str] = []
    people_match = re.search(
        r"(?:when\s+)?you\s+and\s+([^,]+(?:,[^,]+)*?)(?:\s+are\s+all\s+free|\s+can\s+meet|\s+for\b|[.?])",
        question,
        flags=re.IGNORECASE,
    )
    if people_match:
        raw_names = people_match.group(1)
        raw_names = re.sub(r"\band\b", ",", raw_names, flags=re.IGNORECASE)
        tokens = [p.strip().strip(".,;:").lower() for p in raw_names.split(",") if p.strip()]
        for t in tokens:
            cleaned = re.sub(r"[^a-z0-9_\-]", "", t)
            if cleaned and cleaned != "you" and cleaned not in people:
                people.append(cleaned)

    return {
        "day": day,
        "duration": duration,
        "start": start_bound,
        "end": end_bound,
        "people": people,
    }


def _fetch_schedule(person: str, day: str) -> list[tuple[int, int]]:
    person_clean = quote_plus(person.strip().lower())
    day_clean = quote_plus(day.strip())
    urls = [
        f"{STUDY_BASE_URL}/schedule/{person_clean}/{day_clean}",
        f"{STUDY_BASE_URL}/schedules/{person_clean}/{day_clean}",
    ]
    for url in urls:
        try:
            payload = _fetch_text(url, timeout=3.0)
            data = json.loads(payload)
            busy = data.get("busy", []) if isinstance(data, dict) else []
            intervals: list[tuple[int, int]] = []
            for item in busy:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                try:
                    start = _hhmm_to_minutes(str(item[0]))
                    end = _hhmm_to_minutes(str(item[1]))
                    if end > start:
                        intervals.append((start, end))
                except Exception:
                    continue
            return intervals
        except Exception:
            continue
    return []


def _extract_self_busy(day: str) -> list[tuple[int, int]]:
    inbox_urls = [
        f"{STUDY_BASE_URL}/inbox",
        f"{STUDY_BASE_URL}/emails",
        f"{STUDY_BASE_URL}/messages",
    ]
    text = ""
    for url in inbox_urls:
        try:
            text = _fetch_text(url, timeout=3.0)
            if text and len(text.strip()) > 0:
                break
        except Exception:
            continue

    if not text:
        return []

    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    intervals: list[tuple[int, int]] = []

    for block in blocks:
        if not re.search(r"Response:\s*ACCEPTED\b", block, flags=re.IGNORECASE):
            continue

        when_match = re.search(
            r"When:\s*(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})",
            block,
            flags=re.IGNORECASE,
        )
        if not when_match:
            continue

        msg_day, start_str, end_str = when_match.group(1), when_match.group(2), when_match.group(3)
        if msg_day != day:
            continue

        try:
            start = _hhmm_to_minutes(start_str)
            end = _hhmm_to_minutes(end_str)
            if end > start:
                intervals.append((start, end))
        except Exception:
            continue

    return intervals


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _is_slot_free(start: int, end: int, merged_busy: list[tuple[int, int]]) -> bool:
    for busy_start, busy_end in merged_busy:
        if start < busy_end and end > busy_start:
            return False
    return True


def _find_earliest_window(question: str) -> dict[str, str]:
    req = _extract_meeting_request(question)
    day = req["day"]
    start_bound = _hhmm_to_minutes(req["start"])
    end_bound = _hhmm_to_minutes(req["end"])
    duration = int(req["duration"])

    all_busy: list[tuple[int, int]] = []

    with ThreadPoolExecutor(max_workers=max(1, len(req["people"]) + 1)) as pool:
        future_inbox = pool.submit(_extract_self_busy, day)
        future_people = {
            pool.submit(_fetch_schedule, person, day): person
            for person in req["people"]
        }

        try:
            all_busy.extend(future_inbox.result())
        except Exception:
            pass

        for fut in as_completed(future_people):
            try:
                all_busy.extend(fut.result())
            except Exception:
                pass

    merged_busy = _merge_intervals(all_busy)

    # Meetings strictly begin on the hour or the half hour
    current_start = start_bound
    if current_start % 30 != 0:
        current_start += 30 - (current_start % 30)

    while current_start + duration <= end_bound:
        candidate_end = current_start + duration
        if _is_slot_free(current_start, candidate_end, merged_busy):
            return {
                "start": _minutes_to_hhmm(current_start),
                "end": _minutes_to_hhmm(candidate_end),
            }
        current_start += 30

    return {
        "start": _minutes_to_hhmm(start_bound),
        "end": _minutes_to_hhmm(start_bound + duration),
    }


# ---------------------------------------------------------
# FastMCP Server Setup
# ---------------------------------------------------------

mcp_server = FastMCP(
    name="Tool Box Nursery",
    instructions=(
        "A multi-stage assistant. Use `get_name` for its name, `calculate` "
        "for arithmetic, `identify_shape` for what shape a base64 PNG shows, "
        "`count_shapes` for how many shapes a base64 PNG contains, "
        "`retrieve_passages` for study revision chunks, `next_hop` for graph "
        "routing, `get_schedule` to fetch friends' busy intervals, `get_inbox` "
        "to check own accepted meetings, and `find_meeting_window` to compute earliest "
        "overlapping free slots. `ask` auto-routes from raw question text."
    ),
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)


@mcp_server.tool(structured_output=False)
def get_name() -> str:
    """The assistant's name. Answers 'What is your name?'."""
    return enforce_token_limit(NAME)


@mcp_server.tool(structured_output=False)
def calculate(expression: str) -> str:
    """Evaluate arithmetic and return the number, e.g. '2 + 2' -> '4'."""
    return enforce_token_limit(str(solve_arithmetic(expression)))


@mcp_server.tool(structured_output=False)
def identify_shape(image_base64: str) -> str:
    """Identify the shape in a base64-encoded PNG."""
    return enforce_token_limit(solve_shape(image_base64))


@mcp_server.tool(structured_output=False)
def count_shapes(image_base64: str) -> str:
    """Count how many shapes a base64-encoded PNG contains."""
    return enforce_token_limit(str(solve_shape_count(image_base64)))


@mcp_server.tool(structured_output=False)
def ask(question: str, image: str | None = None) -> str:
    """Answer any nursery question from its raw text."""
    lower = question.lower()
    if any(k in lower for k in ["earliest", "window", "all free", "between", "lunch", "24-hour", "free"]):
        return enforce_token_limit(json.dumps(_find_earliest_window(question), ensure_ascii=True))
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
    """Return relevant study passages for a revision question as a JSON array of strings."""
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
    """Return the next adjacent node toward the destination for the map."""
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


@mcp_server.tool(structured_output=False)
def get_schedule(person: str, day: str) -> str:
    """Fetch the busy intervals for a person on a specific date (YYYY-MM-DD)."""
    intervals = _fetch_schedule(person, day)
    busy_list = [[_minutes_to_hhmm(s), _minutes_to_hhmm(e)] for s, e in intervals]
    return enforce_token_limit(json.dumps({"person": person, "day": day, "busy": busy_list}, ensure_ascii=True))


@mcp_server.tool(structured_output=False)
def get_inbox() -> str:
    """Fetch own email inbox containing invitation replies."""
    inbox_urls = [
        f"{STUDY_BASE_URL}/inbox",
        f"{STUDY_BASE_URL}/emails",
        f"{STUDY_BASE_URL}/messages",
    ]
    for url in inbox_urls:
        try:
            txt = _fetch_text(url, timeout=3.0)
            if txt:
                return enforce_token_limit(txt)
        except Exception:
            continue
    return enforce_token_limit("")


@mcp_server.tool(structured_output=False)
def find_meeting_window(question: str) -> str:
    """Find the earliest window that satisfies the scheduling request.
    Returns JSON with zero-padded HH:MM strings: {"start":"..","end":".."}."""
    window = _find_earliest_window(question)
    return enforce_token_limit(json.dumps(window, ensure_ascii=True))


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


@app.post("/kan-cheong-delivery-driver")
async def kan_cheong_delivery_driver(request: Request) -> Any:
    try:
        body = (await request.body()).decode("utf-8")
        data = json.loads(body)
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict) and "start_coordinate" not in data:
            for key in ("data", "payload", "input"):
                if key in data:
                    inner = data[key]
                    data = json.loads(inner) if isinstance(inner, str) else inner
                    break
        return json.loads(kan_chiong_solve(json.dumps(data)))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid input: {exc}")


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


app.mount("/", mcp_app)