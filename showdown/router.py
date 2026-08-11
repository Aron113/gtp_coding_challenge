import logging

from fastapi import APIRouter

from showdown.models import MoveRequest, MoveResponse
from showdown.strategy import choose_move


logger = logging.getLogger("showdown")
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/move", response_model=MoveResponse)
def move(payload: MoveRequest) -> MoveResponse:
    response = choose_move(payload)
    logger.info(
        "match=%s hand=%s round=%s number=%s community=%s action=%s amount=%s",
        payload.match_id,
        payload.hand_number,
        payload.round,
        payload.your_number,
        payload.community_number,
        response.action,
        response.amount,
    )
    return response
