"""Phase 3 - a crowded table.

Two things drive this policy.

Equity is computed, not guessed. Before the reveal we know only our own
number, but the rule set is small enough to enumerate: every community number
against every opponent number, 169 outcomes, exactly. That replaces a linear
"number/13" guess that badly misprices the rules where high is not good -
under low_ball a 2 is a monster, and under wild_seven a 7 outruns everything
except a paired community.

Multiway is the whole phase. Beating one random hand is not beating three, so
equity is raised to the number of live opponents, and opponents who put money
in are not random - the continuation squeeze discounts equity further when
they show real aggression. The scoring compounds this: clearing a leg needs
delta >= +10 *and* strictly the highest delta at the table, so the endgame
plays for position on the leader rather than for chips in the abstract.
"""

from __future__ import annotations

from functools import lru_cache

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

RULES_WITH_LOW_WINS = {"low_ball"}

# Tunables. Calling needs to clear pot odds by a margin, not merely match
# them: the equity model assumes opponents hold random numbers, but anyone
# still putting money in multiway does not, and that optimism compounds once
# per live opponent. A thin margin is how a stack leaks away one defensible
# call at a time.
CALL_MARGIN_BASE = 0.14
CALL_MARGIN_PER_OPPONENT = 0.09
PRE_REVEAL_EXTRA_MARGIN = 0.06  # the community card is still unknown
IN_POSITION_DISCOUNT = 0.03

VALUE_BAR_BASE = 0.58
VALUE_BAR_PER_OPPONENT = 0.10
RAISE_BAR_BASE = 0.76
RAISE_BAR_PER_OPPONENT = 0.06


# --------------------------------------------------------------------------
# Rules and equity
# --------------------------------------------------------------------------


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
        your_value = _wild_pair_value(your_number, community_number)
        opponent_value = _wild_pair_value(opponent_number, community_number)
        if your_value != opponent_value:
            return 1 if your_value > opponent_value else -1
        if your_value > 0:
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


@lru_cache(maxsize=None)
def pre_reveal_equity(your_number: int, table_rule: str) -> float:
    """Exact chance of beating one random opponent, over every community card.

    Enumerated rather than approximated - 169 combinations is nothing, and the
    shape it produces is very different from a linear ramp under three of the
    four rules.
    """
    wins = 0.0
    for community in range(1, 14):
        for opponent in range(1, 14):
            result = _compare_hands(your_number, opponent, community, table_rule)
            wins += 1.0 if result > 0 else (0.5 if result == 0 else 0.0)
    return wins / 169.0


@lru_cache(maxsize=None)
def post_reveal_equity(your_number: int, community_number: int, table_rule: str) -> float:
    """Exact chance of beating one random opponent with the community known."""
    wins = 0.0
    for opponent in range(1, 14):
        result = _compare_hands(your_number, opponent, community_number, table_rule)
        wins += 1.0 if result > 0 else (0.5 if result == 0 else 0.0)
    return wins / 13.0


def _raw_equity(payload: MoveRequest) -> float:
    if payload.round == "post_reveal" and payload.community_number is not None:
        return post_reveal_equity(payload.your_number, payload.community_number, payload.table_rule)
    return pre_reveal_equity(payload.your_number, payload.table_rule)


def _aggression_seen(payload: MoveRequest) -> int:
    """Bets and raises by opponents in this betting round."""
    return sum(
        1
        for action in payload.current_hand_actions
        if action.round == payload.round
        and action.seat != payload.your_seat
        and action.action in {"bet", "raise"}
    )


def multiway_equity(payload: MoveRequest, opponent_count: int) -> float:
    """Equity against everyone still live, squeezed by shown aggression.

    Beating N opponents is roughly beating one N times over. Opponents who
    have already bet or raised are not drawing from the whole range, so each
    piece of aggression discounts the estimate; without that, a hand that beats
    75% of random numbers looks safe against three players who are all betting.
    """
    equity = _raw_equity(payload) ** max(1, opponent_count)
    return equity * (0.88 ** _aggression_seen(payload))


# --------------------------------------------------------------------------
# Leg context - phase 3 pays for topping the table, not for chips
# --------------------------------------------------------------------------


def _your_delta(payload: MoveRequest) -> int:
    return next(
        player.chip_delta for player in payload.players if player.seat == payload.your_seat
    )


def _lead_over_field(payload: MoveRequest) -> int:
    """Our delta minus the best opponent's. Positive means we are topping it."""
    others = [p.chip_delta for p in payload.players if p.seat != payload.your_seat]
    return _your_delta(payload) - (max(others) if others else 0)


def _leg_progress(payload: MoveRequest) -> float:
    if not payload.total_hands:
        return 0.0
    return min(1.0, payload.hand_number / payload.total_hands)


def _stance(payload: MoveRequest) -> float:
    """How much to widen (positive) or tighten (negative) this decision.

    Clearing needs delta >= +10 *and* strictly the highest delta. Sitting on a
    comfortable lead late is worth protecting, because extra chips buy nothing
    once we are top; being behind late is worth gambling for, because second
    place scores exactly as much as last.
    """
    progress = _leg_progress(payload)
    lead = _lead_over_field(payload)
    delta = _your_delta(payload)

    if progress < 0.5:
        return 0.0
    late = (progress - 0.5) * 2.0  # 0 at halfway, 1 at the end

    if lead > 0 and delta >= 10:
        # Ahead and already clearing: shed marginal spots.
        return -0.06 * late
    if lead < 0:
        # Behind: the gap has to be closed before the leg ends.
        urgency = min(1.0, abs(lead) / 25.0)
        return 0.10 * late * urgency
    if delta < 10:
        # Level but short of the +10 floor.
        return 0.05 * late
    return 0.0


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def choose_move(payload: MoveRequest) -> MoveResponse:
    if len(set(payload.legal_actions)) == 1:
        return finalize_move(payload, MoveResponse(action=payload.legal_actions[0]))

    opponents = live_opponents(payload)
    opponent_count = max(1, len(opponents))
    behind_you = players_yet_to_act(payload)
    in_position = behind_you == 0
    equity = multiway_equity(payload, opponent_count)
    stance = _stance(payload)

    trends = build_player_trends(payload)
    live_seats = {p.seat for p in opponents}
    foldy = [
        t for seat, t in trends.items()
        if seat in live_seats and t.sample_size >= 6 and t.fold_rate >= 0.35
    ]
    table_folds_often = len(foldy) >= max(1, opponent_count - 1)

    if payload.to_call == 0:
        return finalize_move(
            payload,
            _no_bet_facing(payload, equity, opponent_count, in_position, stance, table_folds_often),
        )
    return finalize_move(
        payload, _facing_bet(payload, equity, opponent_count, in_position, stance)
    )


def _no_bet_facing(
    payload: MoveRequest,
    equity: float,
    opponent_count: int,
    in_position: bool,
    stance: float,
    table_folds_often: bool,
) -> MoveResponse:
    can_bet = "bet" in payload.legal_actions

    # Value betting needs to beat everyone who calls, so the bar rises with
    # the number of players still live.
    value_bar = VALUE_BAR_BASE + VALUE_BAR_PER_OPPONENT * (opponent_count - 1) - stance
    if in_position:
        value_bar -= 0.04

    if can_bet and equity >= value_bar:
        # Size on strength: thin value stays small, dominant hands charge.
        pressure = 0.10 + 0.55 * max(0.0, equity - value_bar) / max(0.05, 1.0 - value_bar)
        return raise_like(payload, "bet", pressure=min(0.75, pressure))

    # A cheap stab only when position and history both say it can work.
    if (
        can_bet
        and in_position
        and table_folds_often
        and opponent_count == 1
        and equity >= 0.40
    ):
        return raise_like(payload, "bet", pressure=0.12)

    return MoveResponse(action="check")


def _facing_bet(
    payload: MoveRequest,
    equity: float,
    opponent_count: int,
    in_position: bool,
    stance: float,
) -> MoveResponse:
    to_call = payload.to_call
    pot_odds = to_call / max(payload.pot + to_call, 1)
    stack_risk = to_call / max(payload.your_stack, 1)

    # Margin over raw pot odds: money already in the pot is gone, but calling
    # multiway with a hand that is merely break-even against one player is how
    # a stack leaks away.
    margin = CALL_MARGIN_BASE + CALL_MARGIN_PER_OPPONENT * (opponent_count - 1) - stance
    if payload.round == "pre_reveal":
        margin += PRE_REVEAL_EXTRA_MARGIN
    if in_position:
        margin -= IN_POSITION_DISCOUNT

    raise_bar = RAISE_BAR_BASE + RAISE_BAR_PER_OPPONENT * (opponent_count - 1) - stance
    if (
        "raise" in payload.legal_actions
        and equity >= raise_bar
        and stack_risk <= 0.45
    ):
        pressure = 0.15 + 0.5 * max(0.0, equity - raise_bar) / max(0.05, 1.0 - raise_bar)
        return raise_like(payload, "raise", pressure=min(0.7, pressure))

    if equity >= pot_odds + margin:
        return MoveResponse(action="call")

    # Priced in cheaply enough that folding is worse than paying to see it.
    if stack_risk <= 0.04 and equity >= pot_odds:
        return MoveResponse(action="call")

    return MoveResponse(action="fold")


# Kept for callers/tests that reach for these names.
def _is_pair(payload: MoveRequest) -> bool:
    if payload.table_rule == "wild_seven":
        return payload.your_number == 7 or payload.your_number == payload.community_number
    return payload.your_number == payload.community_number


def _need_to_chase(payload: MoveRequest, leader_gap: int, rank: int) -> bool:
    late = payload.hand_number >= int(payload.total_hands * 0.65)
    return late and (leader_gap > 12 or rank > 1)


__all__ = [
    "choose_move",
    "pre_reveal_equity",
    "post_reveal_equity",
    "multiway_equity",
    "leader_seats",
    "table_leader_delta",
    "your_rank",
]
