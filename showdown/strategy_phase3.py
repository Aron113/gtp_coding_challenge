from __future__ import annotations

from showdown.models import MoveRequest, MoveResponse
from showdown.strategy_common import (
    build_player_trends,
    finalize_move,
    leader_seats,
    live_opponents,
    players_yet_to_act,
    raise_like,
    table_leader_delta,
    your_rank,
)


def choose_move(payload: MoveRequest) -> MoveResponse:
    if len(set(payload.legal_actions)) == 1:
        return finalize_move(payload, MoveResponse(action=payload.legal_actions[0]))

    opponents = live_opponents(payload)
    opponent_count = len(opponents)
    trends = build_player_trends(payload)
    behind_you = players_yet_to_act(payload)
    leader_gap = table_leader_delta(payload) - _your_delta(payload)
    rank = your_rank(payload)
    leader_live = any(player.seat in leader_seats(payload) for player in opponents)
    heads_up = opponent_count <= 1

    if payload.round == "post_reveal" and payload.community_number is not None:
        move = _decide_post_reveal(
            payload, opponent_count, behind_you, leader_gap, rank, leader_live, heads_up
        )
    else:
        move = _decide_pre_reveal(
            payload,
            opponent_count,
            trends,
            behind_you,
            leader_gap,
            rank,
            leader_live,
            heads_up,
        )
    return finalize_move(payload, move)


def _decide_pre_reveal(
    payload: MoveRequest,
    opponent_count: int,
    trends: dict[int, object],
    behind_you: int,
    leader_gap: int,
    rank: int,
    leader_live: bool,
    heads_up: bool,
) -> MoveResponse:
    strength = _pre_reveal_strength(payload.your_number, payload.table_rule)
    multiway_penalty = 0.12 * max(0, opponent_count - 1)
    position_bonus = 0.05 if behind_you == 0 else 0.0
    urgency_bonus = 0.05 if _need_to_chase(payload, leader_gap, rank) else 0.0
    duel_bonus = 0.08 if heads_up else 0.0
    leader_bonus = 0.04 if heads_up and leader_live else 0.0
    adjusted = strength - multiway_penalty + position_bonus + urgency_bonus + duel_bonus + leader_bonus

    to_call = payload.to_call
    risk_ratio = to_call / max(payload.your_stack, 1)
    pot_ratio = to_call / max(payload.pot + to_call, 1)
    foldy_table = any(
        trend.sample_size >= 6 and trend.fold_rate >= 0.35 for trend in trends.values()
    )

    if to_call == 0:
        if "bet" in payload.legal_actions and adjusted >= (0.84 if opponent_count >= 2 else 0.72) and behind_you <= 1:
            pressure = 0.08 if opponent_count >= 2 else 0.16
            if _is_special_premium(payload):
                pressure += 0.05
            if heads_up and leader_live:
                pressure += 0.03
            return raise_like(payload, "bet", pressure=pressure)
        if "bet" in payload.legal_actions and adjusted >= 0.74 and foldy_table and behind_you == 0:
            return raise_like(payload, "bet", pressure=0.08)
        return MoveResponse(action="check")

    if "raise" in payload.legal_actions and adjusted >= (0.94 if opponent_count >= 2 else 0.8) and risk_ratio <= (0.14 if opponent_count >= 2 else 0.22) and behind_you == 0:
        return raise_like(payload, "raise", pressure=0.12 if opponent_count >= 2 else 0.16)

    call_margin = 0.12 + 0.05 * max(0, opponent_count - 1)
    if adjusted >= (0.72 if opponent_count >= 2 else 0.62) and pot_ratio <= adjusted - call_margin and risk_ratio <= (0.18 if opponent_count >= 2 else 0.28):
        return MoveResponse(action="call")
    if adjusted >= (0.82 if opponent_count >= 2 else 0.7) and risk_ratio <= (0.28 if opponent_count >= 2 else 0.34) and behind_you == 0:
        return MoveResponse(action="call")
    return MoveResponse(action="fold")


def _decide_post_reveal(
    payload: MoveRequest,
    opponent_count: int,
    behind_you: int,
    leader_gap: int,
    rank: int,
    leader_live: bool,
    heads_up: bool,
) -> MoveResponse:
    equity = _multiway_equity(payload, opponent_count)
    pair = _is_pair(payload)
    stack_risk = payload.to_call / max(payload.your_stack, 1)
    pot_odds = payload.to_call / max(payload.pot + payload.to_call, 1)

    if pair:
        equity = min(0.99, equity + (0.08 if payload.table_rule == "pair_bounty" else 0.03))
    if _need_to_chase(payload, leader_gap, rank):
        equity += 0.04
    if heads_up:
        equity += 0.08
    if heads_up and leader_live:
        equity += 0.04

    if payload.to_call == 0:
        if "bet" in payload.legal_actions and pair and behind_you <= 1:
            return raise_like(payload, "bet", pressure=0.22 if heads_up else 0.12)
        if "bet" in payload.legal_actions and equity >= (0.82 if opponent_count >= 2 else 0.66) and behind_you == 0:
            return raise_like(payload, "bet", pressure=0.14 if heads_up else 0.09)
        return MoveResponse(action="check")

    if pair and "raise" in payload.legal_actions and equity >= (0.9 if opponent_count >= 2 else 0.78) and stack_risk <= (0.22 if opponent_count >= 2 else 0.35) and behind_you == 0:
        return raise_like(payload, "raise", pressure=0.2 if heads_up else 0.16)

    threshold = pot_odds + 0.08 + 0.06 * max(0, opponent_count - 1)
    if pair and payload.table_rule != "low_ball":
        threshold -= 0.06
    if heads_up:
        threshold -= 0.08
    if heads_up and leader_live:
        threshold -= 0.04
    if equity >= threshold and stack_risk <= (0.26 if opponent_count >= 2 else 0.42):
        return MoveResponse(action="call")
    if equity >= pot_odds and stack_risk <= (0.08 if opponent_count >= 2 else 0.14) and behind_you == 0:
        return MoveResponse(action="call")
    return MoveResponse(action="fold")


def _pre_reveal_strength(your_number: int, table_rule: str) -> float:
    if table_rule == "low_ball":
        return (14 - your_number) / 13.0
    if table_rule == "wild_seven":
        if your_number == 7:
            return 0.97
        return your_number / 13.0
    if table_rule == "pair_bounty":
        return min(0.98, your_number / 13.0 + 0.04)
    return your_number / 13.0


def _multiway_equity(payload: MoveRequest, opponent_count: int) -> float:
    heads_up_equity = _heads_up_equity(payload)
    if opponent_count <= 1:
        return heads_up_equity
    return heads_up_equity ** opponent_count


def _heads_up_equity(payload: MoveRequest) -> float:
    wins = 0.0
    for opponent_number in range(1, 14):
        result = _compare_hands(
            your_number=payload.your_number,
            opponent_number=opponent_number,
            community_number=payload.community_number or 0,
            table_rule=payload.table_rule,
        )
        if result > 0:
            wins += 1.0
        elif result == 0:
            wins += 0.5
    return wins / 13.0


def _compare_hands(
    your_number: int, opponent_number: int, community_number: int, table_rule: str
) -> int:
    if table_rule == "low_ball":
        your_pair = your_number == community_number
        opponent_pair = opponent_number == community_number
        if your_pair != opponent_pair:
            return -1 if your_pair else 1
        if your_number != opponent_number:
            return 1 if your_number < opponent_number else -1
        return 0

    if table_rule == "wild_seven":
        your_pair_value = _wild_pair_value(your_number, community_number)
        opponent_pair_value = _wild_pair_value(opponent_number, community_number)
        if your_pair_value != opponent_pair_value:
            return 1 if your_pair_value > opponent_pair_value else -1
        if your_pair_value > 0:
            return 0
        if your_number != opponent_number:
            return 1 if your_number > opponent_number else -1
        return 0

    your_pair = your_number == community_number
    opponent_pair = opponent_number == community_number
    if your_pair != opponent_pair:
        return 1 if your_pair else -1
    if your_number != opponent_number:
        return 1 if your_number > opponent_number else -1
    return 0


def _wild_pair_value(number: int, community_number: int) -> int:
    if number == community_number:
        return community_number
    if number == 7:
        return 7
    return 0


def _is_pair(payload: MoveRequest) -> bool:
    if payload.table_rule == "wild_seven":
        return payload.your_number == 7 or payload.your_number == payload.community_number
    return payload.your_number == payload.community_number


def _is_special_premium(payload: MoveRequest) -> bool:
    if payload.table_rule == "wild_seven":
        return payload.your_number == 7
    if payload.table_rule == "low_ball":
        return payload.your_number <= 2
    return payload.your_number >= 12


def _your_delta(payload: MoveRequest) -> int:
    return next(player.chip_delta for player in payload.players if player.seat == payload.your_seat)


def _need_to_chase(payload: MoveRequest, leader_gap: int, rank: int) -> bool:
    late = payload.hand_number >= int(payload.total_hands * 0.65)
    return late and (leader_gap > 12 or rank > 1)
