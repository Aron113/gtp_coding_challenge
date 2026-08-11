from fastapi import FastAPI
<<<<<<< HEAD
=======
from pydantic import BaseModel
>>>>>>> dafd7896a7da514e765a6f366d72421f3b148374


app = FastAPI()


<<<<<<< HEAD
@app.get("/")
def hello_world():
	return {"message": "Hello, world!"}
=======
class SquareRequest(BaseModel):
    number: int | float


class SquareResponse(BaseModel):
    answer: int | float


@app.post("/square", response_model=SquareResponse)
def square(payload: SquareRequest) -> SquareResponse:
    return SquareResponse(answer=payload.number * payload.number)
>>>>>>> dafd7896a7da514e765a6f366d72421f3b148374
