import base64
import binascii
import json
import math
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from showdown.router import router as showdown_router
from tool_box import answer_question

# Real MCP server (JSON-RPC over the Streamable HTTP transport), not a bespoke
# REST shim - the evaluator connects as an actual MCP client and does the
# initialize/tools-list/tools-call handshake. stateless_http=True so answers
# don't depend on in-memory session state surviving across requests/workers.
mcp_server = FastMCP(
    name="Tool Box Nursery",
    instructions=(
        "A nursery-stage assistant. Call `ask` with the question text "
        "(e.g. 'What is your name?', 'What is 2 + 2?', 'What shape is this?', "
        "'How many shapes are in this image?'). For shape questions, either "
        "embed the base64-encoded PNG directly in `question` or pass it via "
        "`image`."
    ),
    stateless_http=True,
    streamable_http_path="/",
)


@mcp_server.tool()
def ask(question: str, image: str | None = None) -> str | int | float:
    """Answer a nursery question: the bot's name, arithmetic (+, -, *, /),
    or the shape / shape count in a base64-encoded PNG. Returns the bot's
    name as a string, an arithmetic result as a number, a shape as one of
    "rectangle", "triangle", "circle", or a shape count as an integer."""
    payload: dict[str, Any] = {"question": question}
    if image:
        payload["image"] = image
    return answer_question(payload)


mcp_app = mcp_server.streamable_http_app()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(mcp_server.session_manager.run())
        yield


app = FastAPI(title="Tool Box Nursery", version="1.0.0", lifespan=lifespan)
app.include_router(showdown_router)
app.mount("/mcp", mcp_app)


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