"""SHOWDOWN bot — one endpoint that decides a poker-style action per request.

Each hand deals every player an independent, uniformly-random number from
1-13 plus one shared community number, also 1-13. Because the space is this
small and the draws are independent (not without replacement), the exact
probability that a given number beats N live opponents can be computed in
closed form instead of estimated — that calculation (see `equity`) is the
core of every decision here.

The endpoint is intentionally stateless: everything it needs (table rule,
seats, history) arrives on each request, and nothing is cached between
calls, per the challenge's "keep it fast and side-effect-free" guidance.
"""

import hashlib
import math
from functools import lru_cache

from fastapi import FastAPI

app = FastAPI()


# ---------------------------------------------------------------- hand ranking


def _pair_rank(number: int, community: int | None, table_rule: str) -> int | None:
    """The value of the pair `number` makes against `community`, or None."""
    if table_rule == "wild_seven" and number == 7:
        return 7
    if community is not None and number == community:
        return community
    return None


def _hand_key(number: int, community: int, table_rule: str) -> tuple[int, int]:
    """Sortable key where a HIGHER key always wins, regardless of table_rule."""
    pair = _pair_rank(number, community, table_rule)
    low_ball = table_rule == "low_ball"
    if pair is not None:
        return (0, -pair) if low_ball else (1, pair)
    return (1, -number) if low_ball else (0, number)


@lru_cache(maxsize=None)
def _rank_position(number: int, community: int, table_rule: str) -> int:
    """1 (worst) .. 13 (best) position of `number` among all 13 possible numbers."""
    order = sorted(range(1, 14), key=lambda n: _hand_key(n, community, table_rule))
    return order.index(number) + 1


@lru_cache(maxsize=None)
def _equity_known_community(rank: int, num_opponents: int) -> float:
    """Expected pot share for a rank-`rank` hand against `num_opponents`
    independent, uniformly-random opponents (each of the 13 numbers equally
    likely, ties split the pot evenly)."""
    if num_opponents == 0:
        return 1.0
    p_tie = 1 / 13
    p_worse = (rank - 1) / 13
    total = 0.0
    for t in range(num_opponents + 1):
        p = math.comb(num_opponents, t) * (p_tie**t) * (p_worse ** (num_opponents - t))
        total += p * (1 / (1 + t))
    return total


def equity(your_number: int, community: int | None, table_rule: str, num_opponents: int) -> float:
    """Probability-weighted share of the pot at a random showdown."""
    if num_opponents <= 0:
        return 1.0
    if community is not None:
        rank = _rank_position(your_number, community, table_rule)
        return _equity_known_community(rank, num_opponents)
    # Pre-reveal: average over all 13 equally-likely community numbers.
    return sum(
        _equity_known_community(_rank_position(your_number, c, table_rule), num_opponents)
        for c in range(1, 14)
    ) / 13


# ------------------------------------------------------------- opponent reads


def _recent_rate(recent_hands: list, opp_seats: set, predicate) -> float | None:
    total, hits = 0, 0
    for hand in recent_hands:
        for act in hand.get("actions", []):
            if act.get("seat") in opp_seats:
                total += 1
                if predicate(act):
                    hits += 1
    return hits / total if total else None


def _deterministic_unit(*parts) -> float:
    """A stable pseudo-random float in [0, 1) derived from request identifiers,
    used only to cap how often we bluff — not a source of real randomness."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


# --------------------------------------------------------------- bet sizing


def _size_to(pot: int, to_call: int, my_bet_this_round: int, frac: float, lo: int, hi: int) -> int:
    pot_after_call = pot + to_call
    target = my_bet_this_round + to_call + pot_after_call * frac
    target = max(lo, min(hi, target))
    return int(round(target))


# ------------------------------------------------------------------ decision


def decide(req: dict) -> dict:
    your_seat = req["your_seat"]
    players = req.get("players", [])
    me = next((p for p in players if p.get("seat") == your_seat), {})
    live_opponents = [
        p for p in players
        if p.get("seat") != your_seat and not p.get("folded") and not p.get("busted")
    ]
    opp_seats = {p.get("seat") for p in live_opponents}
    k = len(live_opponents)

    table_rule = req.get("table_rule", "standard")
    community = req.get("community_number")
    your_number = req["your_number"]
    pot = req.get("pot", 0)
    to_call = req.get("to_call", 0)
    legal = set(req.get("legal_actions", []))
    my_bet_this_round = me.get("bet_this_round", 0)

    eq = equity(your_number, community, table_rule, k)

    # pair_bounty pays a flat +5 for winning a showdown while holding a pair —
    # a small, bounded nudge toward value-betting an already-made pair, not a
    # full EV re-solve.
    if table_rule == "pair_bounty" and community is not None:
        if _pair_rank(your_number, community, table_rule) is not None:
            eq = min(1.0, eq + min(0.08, 5 / (pot + to_call + 5)))

    recent = req.get("recent_hands", [])
    fold_rate = _recent_rate(recent, opp_seats, lambda a: a.get("action") == "fold")
    aggro_rate = _recent_rate(recent, opp_seats, lambda a: a.get("action") in ("bet", "raise"))
    fold_bias = fold_rate if fold_rate is not None else 0.30
    aggro_bias = aggro_rate if aggro_rate is not None else 0.35

    pre_reveal = req.get("round") == "pre_reveal"
    loosen = 0.05 if fold_bias > 0.55 else 0.0
    tighten = 0.05 if aggro_bias > 0.55 and to_call > 0 else 0.0
    raise_bar = max(0.55, 0.72 - loosen + tighten)
    value_bar = max(0.45, 0.60 - loosen + tighten)
    call_margin = 1.15 if pre_reveal else 1.0

    bluff_roll = _deterministic_unit(req.get("match_id"), req.get("hand_number"), req.get("round"), your_seat)

    action: str | None = None
    amount: int | None = None

    if to_call == 0:
        if eq >= value_bar and "bet" in legal:
            frac = 0.4 + 0.9 * max(0.0, eq - 0.5)
            action, amount = "bet", _size_to(pot, to_call, my_bet_this_round, frac, req.get("min_raise_to", 0), req.get("max_raise_to", 0))
        elif "check" in legal:
            action = "check"
            if k == 1 and eq < 0.30 and "bet" in legal and bluff_roll < 0.12:
                action, amount = "bet", _size_to(pot, to_call, my_bet_this_round, 0.6, req.get("min_raise_to", 0), req.get("max_raise_to", 0))
        else:
            for fallback in ("check", "call", "fold"):
                if fallback in legal:
                    action = fallback
                    break
            else:
                action = sorted(legal)[0] if legal else "check"
    else:
        call_threshold = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
        if eq >= raise_bar and "raise" in legal:
            frac = 0.4 + 0.9 * max(0.0, eq - 0.5)
            action, amount = "raise", _size_to(pot, to_call, my_bet_this_round, frac, req.get("min_raise_to", 0), req.get("max_raise_to", 0))
        elif eq >= call_threshold * call_margin and "call" in legal:
            action = "call"
        elif "fold" in legal:
            cheap = to_call <= 0.25 * max(1, me.get("stack", 0))
            if fold_bias > 0.6 and cheap and "raise" in legal and bluff_roll < 0.10:
                action, amount = "raise", _size_to(pot, to_call, my_bet_this_round, 0.65, req.get("min_raise_to", 0), req.get("max_raise_to", 0))
            else:
                action = "fold"
        elif "call" in legal:
            action = "call"
        else:
            action = "check"

    result = {"action": action}
    if amount is not None:
        result["amount"] = amount
    return result


# --------------------------------------------------------------------- app


def _safe_fallback(payload: dict) -> dict:
    legal = payload.get("legal_actions") or []
    for fallback in ("check", "fold", "call"):
        if fallback in legal:
            return {"action": fallback}
    return {"action": legal[0]} if legal else {"action": "check"}


@app.post("/move")
async def move(payload: dict) -> dict:
    try:
        return decide(payload)
    except Exception:
        return _safe_fallback(payload)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
