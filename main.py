from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from tool_box import answer_question


app = FastAPI(title="Tool Box Nursery", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/mcp")
def mcp_get(question: str | None = None) -> JSONResponse:
    answer = answer_question(question or "")
    return JSONResponse({"answer": answer})


@app.post("/mcp", response_model=None)
async def mcp_post(request: Request) -> Response:
    payload = None
    try:
        payload = await request.json()
    except Exception:
        payload = None

    wants_plain_text = "text/plain" in request.headers.get("accept", "").lower()
    answer = answer_question(payload)

    if wants_plain_text:
        return PlainTextResponse(str(answer))

    return JSONResponse({"answer": answer})


@app.post("/event")
async def event(request: Request) -> dict[str, bool]:
    await request.json()
    return {"ok": True}


@app.post("/square")
def square(number: int | float) -> dict[str, int | float]:
    return {"answer": number * number}
