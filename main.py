import base64
import binascii
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()


class SquareRequest(BaseModel):
    number: int | float


class SquareResponse(BaseModel):
    answer: int | float


@app.post("/square", response_model=SquareResponse)
def square(payload: SquareRequest) -> SquareResponse:
    return SquareResponse(answer=payload.number * payload.number)


class SolveRequest(BaseModel):
    payload: str


PRIORITY_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
DEFAULT_PRIORITY = 2


@app.post("/solve")
def solve(request: SolveRequest) -> dict:
    try:
        decoded = base64.b64decode(request.payload)
        data = json.loads(decoded)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="payload must be base64-encoded JSON")

    adapt_input = data.get("adaptInput", {})
    user = adapt_input.get("user", {})
    metadata = adapt_input.get("metadata", {})

    priority = PRIORITY_MAP.get(metadata.get("priority"), DEFAULT_PRIORITY)
    action = str(adapt_input.get("action", "")).lower()

    return {
        "adaptOutput": {
            "id": user.get("id"),
            "name": user.get("fullName"),
            "action": action,
            "priority": priority,
        }
    }
