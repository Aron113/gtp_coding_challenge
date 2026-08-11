import base64
import binascii
import json
import math
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

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
    payload: dict[str, Any] = {"question": question}
    if image:
        payload["image"] = image
    return enforce_token_limit(str(answer_question(payload)))


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


# Accepts the challenge input either as the raw JSON object, as a JSON string,
# or wrapped in a {"data"|"payload"|"input": ...} envelope, since the grader's
# exact request shape isn't documented.
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