from showdown.models import MoveRequest, MoveResponse
from showdown.strategy_phase1 import choose_move as choose_phase1_move
from showdown.strategy_phase2 import choose_move as choose_phase2_move
from showdown.strategy_phase3 import choose_move as choose_phase3_move
from showdown.strategy_phase4 import choose_move as choose_phase4_move


def choose_move(payload: MoveRequest) -> MoveResponse:
    if payload.phase >= 4:
        return choose_phase4_move(payload)
    if payload.phase >= 3:
        return choose_phase3_move(payload)
    if payload.phase >= 2:
        return choose_phase2_move(payload)
    return choose_phase1_move(payload)
