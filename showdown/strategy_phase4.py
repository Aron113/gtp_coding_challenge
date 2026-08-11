"""Phase 4 - the final table.

Phase 3 pays for topping the table; phase 4 pays for *surviving* it. Each
round cuts the bottom third and reshuffles the survivors, and every finishing
rank pays something, so the objective is rank, not chips. Those are different
games:

  - Busting is the only unrecoverable outcome. A stack that is merely small
    still plays the next round; a stack of zero does not.
  - Chips above the cut line buy nothing this round. Once comfortably clear,
    variance is a cost, not an opportunity.
  - Chips below the cut line must be found before the hands run out, so a
    short stack late has to take on risk it would otherwise decline.

Tables run up to seven-handed, which is the other half of the problem. Equity
against six opponents is a much smaller number than against one, and the
thresholds have to move with the field size rather than sit where a
four-handed table left them.

The hand evaluation itself is phase 3's - exact enumerated equity, squeezed
for multiway and for shown aggression - so only the objective changes here.
"""

from __future__ import annotations

import math

from showdown.models import MoveRequest, MoveResponse
from showdown.strategy_common import (
    finalize_move,
    live_opponents,
    players_yet_to_act,
    raise_like,
)
from showdown.strategy_phase3 import multiway_equity

# Seven-handed pots need a wider margin than four-handed ones, and the cut
# means a marginal spot declined costs less than a marginal spot lost.
CALL_MARGIN_BASE = 0.22
CALL_MARGIN_PER_OPPONENT = 0.06
PRE_REVEAL_EXTRA_MARGIN = 0.06
IN_POSITION_DISCOUNT = 0.03

VALUE_BAR_BASE = 0.68
VALUE_BAR_PER_OPPONENT = 0.09
RAISE_BAR_BASE = 0.78
RAISE_BAR_PER_OPPONENT = 0.05

# Fraction of the field cut each round.
CUT_FRACTION = 1.0 / 3.0


def _seat_delta(payload: MoveRequest, seat: int) -> int:
    for player in payload.players:
        if player.seat == seat:
            return player.chip_delta
    return 0


def _survival_margin(payload: MoveRequest) -> float:
    """Where we sit against the cut line, as a fraction of a starting stack.

    Positive means clear of the cut, negative means inside it. Expressed in
    stacks so it means the same thing whatever the blinds are.
    """
    others = sorted(
        (p.chip_delta for p in payload.players if p.seat != payload.your_seat and not p.busted),
        reverse=True,
    )
    if not others:
        return 1.0

    field = len(others) + 1
    survivors = max(1, field - max(1, int(math.floor(field * CUT_FRACTION))))
    # The delta of the last player who survives the cut.
    cut_index = min(len(others), survivors) - 1
    cut_line = others[cut_index] if 0 <= cut_index < len(others) else others[-1]

    scale = max(payload.starting_stack, payload.big_blind * 10, 1)
    return (_seat_delta(payload, payload.your_seat) - cut_line) / scale


def _stance(payload: MoveRequest) -> float:
    """Widen (positive) or tighten (negative) according to survival pressure.

    Early, position against the cut is noise - there are too many hands left
    for it to mean anything, so this stays neutral and lets the equity model
    do the work. It only bites once the runway is short enough that the
    standing is unlikely to change on its own.
    """
    if not payload.total_hands:
        return 0.0
    progress = min(1.0, payload.hand_number / payload.total_hands)
    if progress < 0.55:
        return 0.0
    late = (progress - 0.55) / 0.45  # 0 -> 1 across the closing stretch

    margin = _survival_margin(payload)
    if margin >= 0.25:
        # Clear of the cut with little time left: stop buying variance.
        return -0.10 * late
    if margin < 0:
        # Inside the cut: the gap has to close before the hands run out.
        return 0.14 * late * min(1.0, 0.3 + abs(margin))
    # Just barely clear - hold station.
    return -0.03 * late


def _bust_risk_guard(payload: MoveRequest, stance: float) -> float:
    """Extra caution when a call could end the tournament.

    Busting forfeits every remaining round, so a hand that risks most of the
    stack has to clear a higher bar than pot odds alone suggest - unless we
    are already inside the cut, where standing still loses anyway.
    """
    if payload.your_stack <= 0:
        return 0.0
    exposure = payload.to_call / max(payload.your_stack, 1)
    if exposure < 0.5:
        return 0.0
    if stance > 0:  # already desperate; survival needs chips
        return 0.0
    return 0.10 * min(1.0, exposure)


def choose_move(payload: MoveRequest) -> MoveResponse:
    if not payload.legal_actions:
        return MoveResponse(action="check")
    if len(set(payload.legal_actions)) == 1:
        return finalize_move(payload, MoveResponse(action=payload.legal_actions[0]))

    opponents = live_opponents(payload)
    opponent_count = max(1, len(opponents))
    in_position = players_yet_to_act(payload) == 0
    equity = multiway_equity(payload, opponent_count)
    stance = _stance(payload)

    if payload.to_call == 0:
        value_bar = (
            VALUE_BAR_BASE + VALUE_BAR_PER_OPPONENT * (opponent_count - 1) - stance
        )
        if in_position:
            value_bar -= 0.04
        if "bet" in payload.legal_actions and equity >= value_bar:
            pressure = 0.10 + 0.5 * max(0.0, equity - value_bar) / max(0.05, 1.0 - value_bar)
            return finalize_move(
                payload, raise_like(payload, "bet", pressure=min(0.7, pressure))
            )
        return finalize_move(payload, MoveResponse(action="check"))

    pot_odds = payload.to_call / max(payload.pot + payload.to_call, 1)
    margin = (
        CALL_MARGIN_BASE
        + CALL_MARGIN_PER_OPPONENT * (opponent_count - 1)
        - stance
        + _bust_risk_guard(payload, stance)
    )
    if payload.round == "pre_reveal":
        margin += PRE_REVEAL_EXTRA_MARGIN
    if in_position:
        margin -= IN_POSITION_DISCOUNT

    raise_bar = RAISE_BAR_BASE + RAISE_BAR_PER_OPPONENT * (opponent_count - 1) - stance
    stack_risk = payload.to_call / max(payload.your_stack, 1)
    if (
        "raise" in payload.legal_actions
        and equity >= raise_bar
        and stack_risk <= 0.45
    ):
        pressure = 0.15 + 0.45 * max(0.0, equity - raise_bar) / max(0.05, 1.0 - raise_bar)
        return finalize_move(
            payload, raise_like(payload, "raise", pressure=min(0.65, pressure))
        )

    if equity >= pot_odds + margin:
        return finalize_move(payload, MoveResponse(action="call"))
    if stack_risk <= 0.04 and equity >= pot_odds:
        return finalize_move(payload, MoveResponse(action="call"))
    return finalize_move(payload, MoveResponse(action="fold"))
