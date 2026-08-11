from fastapi import FastAPI
from pydantic import BaseModel

from showdown.router import router as showdown_router


app = FastAPI()
app.include_router(showdown_router)


class SquareRequest(BaseModel):
    number: int | float


class SquareResponse(BaseModel):
    answer: int | float


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/square", response_model=SquareResponse)
def square(payload: SquareRequest) -> SquareResponse:
    return SquareResponse(answer=payload.number * payload.number)
