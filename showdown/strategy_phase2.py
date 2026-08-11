from __future__ import annotations

from showdown.models import MoveRequest, MoveResponse
from showdown.strategy_common import (
    OpponentStats,
    build_opponent_stats,
    finalize_move,
    find_opponent,
    raise_like,
    safe_fallback,
)


def choose_move(payload: MoveRequest) -> MoveResponse:
    if len(set(payload.legal_actions)) == 1:
        return finalize_move(payload, MoveResponse(action=payload.legal_actions[0]))

    opponent = find_opponent(payload)
    stats = build_opponent_stats(payload, opponent.seat)
    opponent_name = opponent.name.lower()

    if payload.table_rule == "low_ball":
        if payload.round == "post_reveal" and payload.community_number is not None:
            move = _decide_low_ball_post_reveal(payload, stats, opponent.name)
        else:
            move = _decide_low_ball_pre_reveal(payload, stats, opponent.name)
    elif opponent_name == "remy" and payload.table_rule == "pair_bounty":
        if payload.round == "post_reveal" and payload.community_number is not None:
            move = _decide_remy_pair_bounty_post_reveal(payload, stats)
        else:
            move = _decide_remy_pair_bounty_pre_reveal(payload, stats)
    elif opponent_name == "remy" and payload.table_rule == "wild_seven":
        if payload.round == "post_reveal" and payload.community_number is not None:
            move = _decide_remy_wild_seven_post_reveal(payload, stats)
        else:
            move = _decide_remy_wild_seven_pre_reveal(payload, stats)
    elif payload.round == "post_reveal" and payload.community_number is not None:
        move = _decide_post_reveal(payload, stats)
    else:
        move = _decide_pre_reveal(payload, stats)
    return finalize_move(payload, move)


def _decide_low_ball_pre_reveal(
    payload: MoveRequest, stats: OpponentStats, opponent_name: str
) -> MoveResponse:
    number = payload.your_number
    to_call = payload.to_call
    risk_ratio = to_call / max(payload.your_stack, 1)
    pot_ratio = to_call / max(payload.pot + to_call, 1)
    is_nadia = opponent_name.lower() == "nadia"

    if to_call == 0:
        if "bet" in payload.legal_actions and number <= 3:
            pressure = 0.08 if is_nadia else 0.12
            return raise_like(payload, "bet", pressure=pressure)
        return MoveResponse(action="check")

    if number <= 2:
        if "raise" in payload.legal_actions and risk_ratio <= 0.14 and not is_nadia:
            return raise_like(payload, "raise", pressure=0.1)
        return MoveResponse(action="call")

    if number == 3:
        if risk_ratio <= 0.12 or pot_ratio <= 0.18:
            return MoveResponse(action="call")
        return MoveResponse(action="fold")

    if number == 4 and not is_nadia and risk_ratio <= 0.08 and pot_ratio <= 0.12:
        return MoveResponse(action="call")

    if (
        stats.sample_size >= 8
        and stats.pre_raise_rate >= 0.5
        and number <= 4
        and risk_ratio <= 0.06
        and pot_ratio <= 0.08
    ):
        return MoveResponse(action="call")

    return MoveResponse(action="fold")


def _decide_low_ball_post_reveal(
    payload: MoveRequest, stats: OpponentStats, opponent_name: str
) -> MoveResponse:
    number = payload.your_number
    to_call = payload.to_call
    equity = _post_reveal_equity(payload)
    stack_risk = to_call / max(payload.your_stack, 1)
    pot_odds = to_call / max(payload.pot + to_call, 1)
    is_pair = payload.community_number == number
    is_nadia = opponent_name.lower() == "nadia"

    if is_pair:
        if to_call == 0:
            return MoveResponse(action="check")
        if stack_risk <= 0.03 and pot_odds <= 0.05 and "call" in payload.legal_actions:
            return MoveResponse(action="call")
        return MoveResponse(action="fold")

    if to_call == 0:
        if "bet" in payload.legal_actions and number <= 3 and equity >= 0.78:
            pressure = 0.07 if is_nadia else 0.1
            return raise_like(payload, "bet", pressure=pressure)
        return MoveResponse(action="check")

    bluff_bonus = 0.0
    if not is_nadia and stats.sample_size >= 8 and stats.bluff_rate >= 0.3:
        bluff_bonus = 0.05
    effective_equity = equity + bluff_bonus

    if number <= 2:
        if "raise" in payload.legal_actions and effective_equity >= 0.9 and stack_risk <= 0.12 and not is_nadia:
            return raise_like(payload, "raise", pressure=0.1)
        if effective_equity >= pot_odds + 0.03 and stack_risk <= 0.3:
            return MoveResponse(action="call")
        return MoveResponse(action="fold")

    if number == 3:
        if effective_equity >= pot_odds + 0.08 and stack_risk <= 0.16:
            return MoveResponse(action="call")
        return MoveResponse(action="fold")

    if number == 4 and not is_nadia:
        if effective_equity >= pot_odds + 0.12 and stack_risk <= 0.08:
            return MoveResponse(action="call")
        return MoveResponse(action="fold")

    return MoveResponse(action="fold")


def _decide_remy_pair_bounty_pre_reveal(
    payload: MoveRequest, stats: OpponentStats
) -> MoveResponse:
    number = payload.your_number
    to_call = payload.to_call
    risk_ratio = to_call / max(payload.your_stack, 1)
    pot_ratio = to_call / max(payload.pot + to_call, 1)

    if to_call == 0:
        if "bet" in payload.legal_actions and (
            number >= 10 or (number >= 8 and stats.fold_rate >= 0.28)
        ):
            pressure = 0.12 if number < 12 else 0.18
            return raise_like(payload, "bet", pressure=pressure)
        return MoveResponse(action="check")

    if "raise" in payload.legal_actions and number >= 12 and risk_ratio <= 0.24:
        return raise_like(payload, "raise", pressure=0.16)

    if number >= 9 and risk_ratio <= 0.34:
        return MoveResponse(action="call")
    if number >= 7 and pot_ratio <= 0.18 and risk_ratio <= 0.18:
        return MoveResponse(action="call")
    if (
        stats.sample_size >= 6
        and stats.pre_raise_rate >= 0.42
        and number >= 6
        and pot_ratio <= 0.1
        and risk_ratio <= 0.1
    ):
        return MoveResponse(action="call")
    return MoveResponse(action="fold")


def _decide_remy_pair_bounty_post_reveal(
    payload: MoveRequest, stats: OpponentStats
) -> MoveResponse:
    has_pair = _is_pair(payload)
    equity = _post_reveal_equity(payload)
    effective_equity = min(0.99, equity + (0.14 if has_pair else 0.0))

    if payload.to_call == 0:
        if "bet" in payload.legal_actions and (has_pair or effective_equity >= 0.7):
            pressure = 0.22 if has_pair else 0.14
            return raise_like(payload, "bet", pressure=pressure)
        return MoveResponse(action="check")

    pot_odds = payload.to_call / max(payload.pot + payload.to_call, 1)
    stack_risk = payload.to_call / max(payload.your_stack, 1)
    bluff_bonus = 0.04 if stats.sample_size >= 6 and stats.bluff_rate >= 0.24 else 0.0
    effective_equity = min(0.99, effective_equity + bluff_bonus)

    if has_pair and "raise" in payload.legal_actions and stack_risk <= 0.32:
        return raise_like(payload, "raise", pressure=0.24)
    if effective_equity >= pot_odds + 0.04 and stack_risk <= 0.42:
        return MoveResponse(action="call")
    if effective_equity >= pot_odds - 0.01 and stack_risk <= 0.1:
        return MoveResponse(action="call")
    return MoveResponse(action="fold")


def _decide_remy_wild_seven_pre_reveal(
    payload: MoveRequest, stats: OpponentStats
) -> MoveResponse:
    number = payload.your_number
    to_call = payload.to_call
    risk_ratio = to_call / max(payload.your_stack, 1)
    pot_ratio = to_call / max(payload.pot + to_call, 1)

    if number == 7:
        if to_call == 0 and "bet" in payload.legal_actions:
            return raise_like(payload, "bet", pressure=0.18)
        if "raise" in payload.legal_actions and risk_ratio <= 0.3:
            return raise_like(payload, "raise", pressure=0.22)
        return MoveResponse(action="call")

    if to_call == 0:
        if "bet" in payload.legal_actions and (
            number >= 11 or (number >= 9 and stats.fold_rate >= 0.3)
        ):
            return raise_like(payload, "bet", pressure=0.12 if number < 12 else 0.18)
        return MoveResponse(action="check")

    if "raise" in payload.legal_actions and number >= 12 and risk_ratio <= 0.2:
        return raise_like(payload, "raise", pressure=0.15)
    if number >= 10 and risk_ratio <= 0.3:
        return MoveResponse(action="call")
    if number >= 8 and pot_ratio <= 0.16 and risk_ratio <= 0.14:
        return MoveResponse(action="call")
    return MoveResponse(action="fold")


def _decide_remy_wild_seven_post_reveal(
    payload: MoveRequest, stats: OpponentStats
) -> MoveResponse:
    has_pair = _is_pair(payload)
    equity = _post_reveal_equity(payload)
    effective_equity = equity + (0.05 if stats.sample_size >= 6 and stats.bluff_rate >= 0.24 else 0.0)

    if payload.to_call == 0:
        if "bet" in payload.legal_actions and (has_pair or effective_equity >= 0.74):
            pressure = 0.24 if has_pair else 0.14
            return raise_like(payload, "bet", pressure=pressure)
        return MoveResponse(action="check")

    pot_odds = payload.to_call / max(payload.pot + payload.to_call, 1)
    stack_risk = payload.to_call / max(payload.your_stack, 1)

    if has_pair:
        if "raise" in payload.legal_actions and stack_risk <= 0.34:
            return raise_like(payload, "raise", pressure=0.25)
        return MoveResponse(action="call")
    if effective_equity >= pot_odds + 0.05 and stack_risk <= 0.35:
        return MoveResponse(action="call")
    if effective_equity >= pot_odds and stack_risk <= 0.1:
        return MoveResponse(action="call")
    return MoveResponse(action="fold")


def _decide_pre_reveal(payload: MoveRequest, stats: OpponentStats) -> MoveResponse:
    strength = _pre_reveal_strength(payload.your_number, payload.table_rule)
    aggression_bonus = 0.06 if stats.sample_size >= 6 and stats.fold_rate >= 0.3 else 0.0
    strength += aggression_bonus

    if payload.to_call == 0:
        if "bet" in payload.legal_actions and strength >= 0.72:
            return raise_like(payload, "bet", pressure=_pressure_for_strength(strength, capped=True))
        return MoveResponse(action="check")

    risk_ratio = payload.to_call / max(payload.your_stack, 1)
    pot_ratio = payload.to_call / max(payload.pot + payload.to_call, 1)

    if "raise" in payload.legal_actions and strength >= 0.9 and risk_ratio <= 0.22:
        return raise_like(payload, "raise", pressure=_pressure_for_strength(strength, capped=True))

    if strength >= 0.62 and pot_ratio <= strength - 0.18 and risk_ratio <= 0.35:
        return MoveResponse(action="call")

    if (
        stats.sample_size >= 6
        and stats.pre_raise_rate >= 0.45
        and strength >= 0.52
        and pot_ratio <= strength - 0.05
        and risk_ratio <= 0.2
    ):
        return MoveResponse(action="call")

    return safe_fallback(payload) if payload.to_call == 0 else MoveResponse(action="fold")


def _decide_post_reveal(payload: MoveRequest, stats: OpponentStats) -> MoveResponse:
    equity = _post_reveal_equity(payload)
    bounty_bonus = 0.08 if payload.table_rule == "pair_bounty" and _is_pair(payload) else 0.0
    pressure_equity = min(0.99, equity + bounty_bonus)

    if payload.to_call == 0:
        if "bet" in payload.legal_actions and pressure_equity >= 0.66:
            pressure = _pressure_for_strength(pressure_equity, capped=True)
            return raise_like(payload, "bet", pressure=pressure)
        return MoveResponse(action="check")

    pot_odds = payload.to_call / max(payload.pot + payload.to_call, 1)
    stack_risk = payload.to_call / max(payload.your_stack, 1)
    bluff_bonus = 0.0
    if stats.sample_size >= 6 and stats.bluff_rate >= 0.28:
        bluff_bonus = 0.06

    call_threshold = pot_odds + 0.04
    effective_equity = pressure_equity + bluff_bonus

    if "raise" in payload.legal_actions and effective_equity >= 0.86 and stack_risk <= 0.3:
        return raise_like(payload, "raise", pressure=_pressure_for_strength(effective_equity, capped=True))

    if effective_equity >= call_threshold and stack_risk <= 0.45:
        return MoveResponse(action="call")

    if effective_equity >= pot_odds - 0.01 and stack_risk <= 0.12:
        return MoveResponse(action="call")

    return MoveResponse(action="fold")


def _pre_reveal_strength(your_number: int, table_rule: str) -> float:
    if table_rule == "low_ball":
        return (14 - your_number) / 13.0
    if table_rule == "wild_seven":
        if your_number == 7:
            return 0.96
        return your_number / 13.0
    if table_rule == "pair_bounty":
        return min(0.97, your_number / 13.0 + 0.04)
    return your_number / 13.0


def _post_reveal_equity(payload: MoveRequest) -> float:
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
    if payload.table_rule == "low_ball":
        return payload.your_number == payload.community_number
    return payload.your_number == payload.community_number


def _pressure_for_strength(strength: float, capped: bool) -> float:
    base = 0.08 + max(0.0, strength - 0.65) * 0.45
    if capped:
        return min(0.24, base)
    return min(0.4, base)
