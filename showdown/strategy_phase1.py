from __future__ import annotations

from showdown.models import MoveRequest, MoveResponse
from showdown.strategy_common import (
    build_opponent_stats,
    finalize_move,
    find_opponent,
    raise_like,
)


def choose_move(payload: MoveRequest) -> MoveResponse:
    legal = set(payload.legal_actions)
    if len(legal) == 1:
        return finalize_move(payload, MoveResponse(action=payload.legal_actions[0]))

    opponent = find_opponent(payload)
    stats = build_opponent_stats(payload, opponent.seat)

    if payload.round == "post_reveal" and payload.community_number is not None:
        move = _decide_post_reveal(payload, stats)
    else:
        move = _decide_pre_reveal(payload, stats)

    return finalize_move(payload, move)


def _decide_pre_reveal(payload: MoveRequest, stats) -> MoveResponse:
    number = payload.your_number
    to_call = payload.to_call

    if to_call == 0:
        if "bet" in payload.legal_actions and (
            number >= 11 or (number >= 9 and stats.fold_rate >= 0.30)
        ):
            return raise_like(payload, "bet", pressure=0.12 if number < 12 else 0.2)
        return MoveResponse(action="check")

    if number >= 12 and "raise" in payload.legal_actions and stats.fold_rate >= 0.2:
        return raise_like(payload, "raise", pressure=0.18)

    if number >= 10:
        return MoveResponse(action="call")

    price_ratio = to_call / max(payload.pot + to_call, 1)
    if number >= 8 and price_ratio <= 0.22:
        return MoveResponse(action="call")
    if number >= 6 and stats.pre_raise_rate >= 0.45 and price_ratio <= 0.12:
        return MoveResponse(action="call")
    return MoveResponse(action="fold")


def _decide_post_reveal(payload: MoveRequest, stats) -> MoveResponse:
    has_pair = payload.community_number == payload.your_number
    if has_pair:
        if payload.to_call == 0 and "bet" in payload.legal_actions:
            return raise_like(payload, "bet", pressure=0.4)
        if "raise" in payload.legal_actions and payload.max_raise_to not in (None, 0):
            return raise_like(payload, "raise", pressure=0.45)
        return MoveResponse(action="call" if payload.to_call > 0 else "check")

    strength = _post_reveal_win_rate(payload.your_number, payload.community_number or 0)
    adjusted_strength = min(
        0.98, strength + stats.bluff_rate * 0.18 + stats.post_bet_rate * 0.05
    )

    if payload.to_call == 0:
        if "bet" in payload.legal_actions and (
            strength >= 0.68 or (strength >= 0.5 and stats.fold_rate >= 0.35)
        ):
            pressure = 0.16 if strength < 0.75 else 0.28
            return raise_like(payload, "bet", pressure=pressure)
        return MoveResponse(action="check")

    pot_odds = payload.to_call / max(payload.pot + payload.to_call, 1)
    if adjusted_strength >= pot_odds + 0.08:
        if (
            "raise" in payload.legal_actions
            and strength >= 0.78
            and stats.post_bet_rate >= 0.45
        ):
            return raise_like(payload, "raise", pressure=0.2)
        return MoveResponse(action="call")

    if stats.bluff_rate >= 0.3 and adjusted_strength >= pot_odds - 0.02:
        return MoveResponse(action="call")

    return MoveResponse(action="fold")


def _post_reveal_win_rate(your_number: int, community_number: int) -> float:
    wins = 0
    ties = 0
    for opponent_number in range(1, 14):
        if opponent_number == community_number:
            continue
        if opponent_number < your_number:
            wins += 1
        elif opponent_number == your_number:
            ties += 1
    return (wins + 0.5 * ties) / 13.0
