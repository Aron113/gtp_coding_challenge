from __future__ import annotations

from dataclasses import dataclass

from showdown.models import MoveRequest, MoveResponse, RecentHand


@dataclass(slots=True)
class OpponentStats:
    pre_raise_rate: float = 0.0
    post_bet_rate: float = 0.0
    fold_rate: float = 0.0
    bluff_rate: float = 0.0
    sample_size: int = 0


def choose_move(payload: MoveRequest) -> MoveResponse:
    legal = set(payload.legal_actions)
    if len(legal) == 1:
        return _finalize_move(payload, MoveResponse(action=payload.legal_actions[0]))

    my_player = _find_player(payload, payload.your_seat)
    opponent = _find_opponent(payload)
    stats = _build_opponent_stats(payload, opponent.seat)

    if payload.round == "post_reveal" and payload.community_number is not None:
        move = _decide_post_reveal(payload, my_player.bet_this_round, stats)
    else:
        move = _decide_pre_reveal(payload, my_player.bet_this_round, stats)

    return _finalize_move(payload, move)


def _decide_pre_reveal(
    payload: MoveRequest, my_bet_this_round: int, stats: OpponentStats
) -> MoveResponse:
    number = payload.your_number
    to_call = payload.to_call

    if to_call == 0:
        if "bet" in payload.legal_actions and (
            number >= 11 or (number >= 9 and stats.fold_rate >= 0.30)
        ):
            return _raise_like(payload, "bet", pressure=0.12 if number < 12 else 0.2)
        return MoveResponse(action="check")

    if number >= 12 and "raise" in payload.legal_actions and stats.fold_rate >= 0.2:
        return _raise_like(payload, "raise", pressure=0.18)

    if number >= 10:
        return MoveResponse(action="call")

    price_ratio = to_call / max(payload.pot + to_call, 1)
    if number >= 8 and price_ratio <= 0.22:
        return MoveResponse(action="call")
    if number >= 6 and stats.pre_raise_rate >= 0.45 and price_ratio <= 0.12:
        return MoveResponse(action="call")
    return MoveResponse(action="fold")


def _decide_post_reveal(
    payload: MoveRequest, my_bet_this_round: int, stats: OpponentStats
) -> MoveResponse:
    has_pair = payload.community_number == payload.your_number
    if has_pair:
        if payload.to_call == 0 and "bet" in payload.legal_actions:
            return _raise_like(payload, "bet", pressure=0.4)
        if "raise" in payload.legal_actions and payload.max_raise_to not in (None, 0):
            return _raise_like(payload, "raise", pressure=0.45)
        return MoveResponse(action="call" if payload.to_call > 0 else "check")

    strength = _post_reveal_win_rate(payload.your_number, payload.community_number or 0)
    adjusted_strength = min(0.98, strength + stats.bluff_rate * 0.18 + stats.post_bet_rate * 0.05)

    if payload.to_call == 0:
        if "bet" in payload.legal_actions and (
            strength >= 0.68 or (strength >= 0.5 and stats.fold_rate >= 0.35)
        ):
            pressure = 0.16 if strength < 0.75 else 0.28
            return _raise_like(payload, "bet", pressure=pressure)
        return MoveResponse(action="check")

    pot_odds = payload.to_call / max(payload.pot + payload.to_call, 1)
    if adjusted_strength >= pot_odds + 0.08:
        if (
            "raise" in payload.legal_actions
            and strength >= 0.78
            and stats.post_bet_rate >= 0.45
        ):
            return _raise_like(payload, "raise", pressure=0.2)
        return MoveResponse(action="call")

    if stats.bluff_rate >= 0.3 and adjusted_strength >= pot_odds - 0.02:
        return MoveResponse(action="call")

    return MoveResponse(action="fold")


def _raise_like(payload: MoveRequest, action: str, pressure: float) -> MoveResponse:
    min_raise_to = payload.min_raise_to
    max_raise_to = payload.max_raise_to
    if action not in payload.legal_actions or min_raise_to is None or max_raise_to is None:
        return MoveResponse(action="call" if payload.to_call > 0 and "call" in payload.legal_actions else "check")

    if min_raise_to >= max_raise_to:
        return MoveResponse(action=action, amount=min_raise_to)

    span = max_raise_to - min_raise_to
    amount = min_raise_to + int(round(span * pressure))
    amount = max(min_raise_to, min(max_raise_to, amount))
    return MoveResponse(action=action, amount=amount)


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


def _build_opponent_stats(payload: MoveRequest, opponent_seat: int) -> OpponentStats:
    recent_hands = payload.recent_hands
    if not recent_hands:
        return OpponentStats()

    pre_raises = 0
    post_bets = 0
    opponent_folds = 0
    bluff_events = 0
    bluff_samples = 0

    for hand in recent_hands:
        opponent_actions = [a for a in hand.actions if a.seat == opponent_seat]
        if any(a.round == "pre_reveal" and a.action == "raise" for a in opponent_actions):
            pre_raises += 1
        if any(
            a.round == "post_reveal" and a.action in {"bet", "raise"} for a in opponent_actions
        ):
            post_bets += 1
        if opponent_actions and opponent_actions[-1].action == "fold":
            opponent_folds += 1

        if _is_bluff_showdown(hand, opponent_seat):
            bluff_events += 1
            bluff_samples += 1
        elif _is_seen_post_reveal_aggression(hand, opponent_seat):
            bluff_samples += 1

    sample_size = len(recent_hands)
    return OpponentStats(
        pre_raise_rate=pre_raises / sample_size,
        post_bet_rate=post_bets / sample_size,
        fold_rate=opponent_folds / sample_size,
        bluff_rate=(bluff_events / bluff_samples) if bluff_samples else 0.0,
        sample_size=sample_size,
    )


def _is_bluff_showdown(hand: RecentHand, opponent_seat: int) -> bool:
    seat_key = str(opponent_seat)
    shown_number = hand.shown_numbers.get(seat_key)
    if shown_number is None or hand.community_number is None:
        return False
    if not _is_seen_post_reveal_aggression(hand, opponent_seat):
        return False
    return shown_number != hand.community_number and shown_number <= 6


def _is_seen_post_reveal_aggression(hand: RecentHand, opponent_seat: int) -> bool:
    return any(
        action.seat == opponent_seat
        and action.round == "post_reveal"
        and action.action in {"bet", "raise"}
        for action in hand.actions
    )


def _find_player(payload: MoveRequest, seat: int):
    for player in payload.players:
        if player.seat == seat:
            return player
    raise ValueError(f"Seat {seat} not found in players")


def _find_opponent(payload: MoveRequest):
    for player in payload.players:
        if player.seat != payload.your_seat:
            return player
    raise ValueError("Opponent not found")


def _finalize_move(payload: MoveRequest, move: MoveResponse) -> MoveResponse:
    legal = payload.legal_actions
    if move.action not in legal:
        return _safe_fallback(payload)
    if move.action in {"bet", "raise"}:
        if move.amount is None:
            return _safe_fallback(payload)
        if payload.min_raise_to is None or payload.max_raise_to is None:
            return _safe_fallback(payload)
        if not (payload.min_raise_to <= move.amount <= payload.max_raise_to):
            return _safe_fallback(payload)
        return move
    return MoveResponse(action=move.action)


def _safe_fallback(payload: MoveRequest) -> MoveResponse:
    if "check" in payload.legal_actions:
        return MoveResponse(action="check")
    if "call" in payload.legal_actions:
        return MoveResponse(action="call")
    return MoveResponse(action="fold")
