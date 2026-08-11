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


def find_player(payload: MoveRequest, seat: int):
    for player in payload.players:
        if player.seat == seat:
            return player
    raise ValueError(f"Seat {seat} not found in players")


def find_opponent(payload: MoveRequest):
    for player in payload.players:
        if player.seat != payload.your_seat:
            return player
    raise ValueError("Opponent not found")


def safe_fallback(payload: MoveRequest) -> MoveResponse:
    if "check" in payload.legal_actions:
        return MoveResponse(action="check")
    if "call" in payload.legal_actions:
        return MoveResponse(action="call")
    return MoveResponse(action="fold")


def finalize_move(payload: MoveRequest, move: MoveResponse) -> MoveResponse:
    if move.action not in payload.legal_actions:
        return safe_fallback(payload)
    if move.action in {"bet", "raise"}:
        if move.amount is None:
            return safe_fallback(payload)
        if payload.min_raise_to is None or payload.max_raise_to is None:
            return safe_fallback(payload)
        if not (payload.min_raise_to <= move.amount <= payload.max_raise_to):
            return safe_fallback(payload)
        return move
    return MoveResponse(action=move.action)


def raise_like(payload: MoveRequest, action: str, pressure: float) -> MoveResponse:
    min_raise_to = payload.min_raise_to
    max_raise_to = payload.max_raise_to
    if action not in payload.legal_actions or min_raise_to is None or max_raise_to is None:
        if payload.to_call > 0 and "call" in payload.legal_actions:
            return MoveResponse(action="call")
        return MoveResponse(action="check")

    if min_raise_to >= max_raise_to:
        return MoveResponse(action=action, amount=min_raise_to)

    span = max_raise_to - min_raise_to
    amount = min_raise_to + int(round(span * pressure))
    amount = max(min_raise_to, min(max_raise_to, amount))
    return MoveResponse(action=action, amount=amount)


def build_opponent_stats(payload: MoveRequest, opponent_seat: int) -> OpponentStats:
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

        if is_bluff_showdown(hand, opponent_seat):
            bluff_events += 1
            bluff_samples += 1
        elif is_seen_post_reveal_aggression(hand, opponent_seat):
            bluff_samples += 1

    sample_size = len(recent_hands)
    return OpponentStats(
        pre_raise_rate=pre_raises / sample_size,
        post_bet_rate=post_bets / sample_size,
        fold_rate=opponent_folds / sample_size,
        bluff_rate=(bluff_events / bluff_samples) if bluff_samples else 0.0,
        sample_size=sample_size,
    )


def is_bluff_showdown(hand: RecentHand, opponent_seat: int) -> bool:
    seat_key = str(opponent_seat)
    shown_number = hand.shown_numbers.get(seat_key)
    if shown_number is None or hand.community_number is None:
        return False
    if not is_seen_post_reveal_aggression(hand, opponent_seat):
        return False
    return shown_number != hand.community_number and shown_number <= 6


def is_seen_post_reveal_aggression(hand: RecentHand, opponent_seat: int) -> bool:
    return any(
        action.seat == opponent_seat
        and action.round == "post_reveal"
        and action.action in {"bet", "raise"}
        for action in hand.actions
    )
