import logging
from typing import Any

from fastapi import APIRouter, Request

from showdown.models import MoveRequest, MoveResponse
from showdown.strategy import choose_move


logger = logging.getLogger("showdown")
router = APIRouter()


def emergency_move(raw: Any) -> MoveResponse:
    """The cheapest legal action we can name from whatever arrived.

    Reached only when the payload would not parse or the strategy raised. It
    never bets: checking is free and folding risks nothing, and in a one-shot
    tournament an answer that costs a blind beats a mispay.
    """
    legal: list[str] = []
    to_call = 0
    if isinstance(raw, dict):
        candidate = raw.get("legal_actions")
        if isinstance(candidate, list):
            legal = [a for a in candidate if isinstance(a, str)]
        if isinstance(raw.get("to_call"), int):
            to_call = raw["to_call"]

    if "check" in legal:
        return MoveResponse(action="check")
    if to_call <= 0 and "call" in legal:
        return MoveResponse(action="call")
    if "fold" in legal:
        return MoveResponse(action="fold")
    if "call" in legal:
        return MoveResponse(action="call")
    return MoveResponse(action="check")


@router.post("/move", response_model=MoveResponse)
async def move(request: Request) -> MoveResponse:
    """Answer a request to act.

    This endpoint must not fail. Five mispays in a row folds the bot out of a
    phase 4 game it gets no retry at, and a 500 counts as a mispay just as a
    timeout does - so parsing, strategy and legality are each guarded
    separately, and every path ends at a legal action.
    """
    try:
        raw = await request.json()
    except Exception:
        raw = {}

    try:
        payload = MoveRequest.model_validate(raw)
    except Exception:
        logger.exception("showdown: payload did not validate; answering safely")
        return emergency_move(raw)

    try:
        response = choose_move(payload)
        # Last line of defence: an illegal action is scored as a mispay, so it
        # is worth re-checking even though the strategy already validates.
        if payload.legal_actions and response.action not in payload.legal_actions:
            logger.warning("showdown: strategy returned illegal %s", response.action)
            return emergency_move(raw)
    except Exception:
        logger.exception("showdown: strategy raised; answering safely")
        return emergency_move(raw)

    logger.info(
        "phase=%s rule=%s leg=%s match=%s hand=%s round=%s number=%s community=%s action=%s amount=%s",
        payload.phase,
        payload.table_rule,
        payload.leg_number,
        payload.match_id,
        payload.hand_number,
        payload.round,
        payload.your_number,
        payload.community_number,
        response.action,
        response.amount,
    )
    return response
