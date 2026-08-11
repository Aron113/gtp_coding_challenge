"""A four-handed SHOWDOWN table, for measuring strategy changes.

Phase 3 is scored on topping the table, not on winning chips in the abstract,
so tuning by intuition is guesswork: a change that wins more chips against one
opponent type can still lose the leg. This plays whole 60-hand legs against a
mix of opponent archetypes and reports what the scoring actually rewards -
delta >= +10 *and* strictly the highest delta at the table.

The betting engine and the hand comparison mirror the rules the strategy
modules already encode. It is an approximation of the real table (the real
opponents are Dana, Miles and Theo, whose styles are not published), so treat
the absolute numbers as relative signal, not as a predicted score.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

from showdown.models import HandAction, MoveRequest, PlayerState, RecentHand
from showdown.strategy import choose_move

TABLE_RULES = ("standard", "low_ball", "pair_bounty", "wild_seven")


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


def compare(a: int, b: int, community: int, rule: str) -> int:
    """1 if a beats b, -1 if b beats a, 0 on a tie."""
    if rule == "low_ball":
        a_pair, b_pair = a == community, b == community
        if a_pair != b_pair:
            return -1 if a_pair else 1
        if a != b:
            return 1 if a < b else -1
        return 0
    if rule == "wild_seven":
        av = community if a == community else (7 if a == 7 else 0)
        bv = community if b == community else (7 if b == 7 else 0)
        if av != bv:
            return 1 if av > bv else -1
        if av > 0:
            return 0
        if a != b:
            return 1 if a > b else -1
        return 0
    a_pair, b_pair = a == community, b == community
    if a_pair != b_pair:
        return 1 if a_pair else -1
    if a != b:
        return 1 if a > b else -1
    return 0


# --------------------------------------------------------------------------
# Opponents
# --------------------------------------------------------------------------


def _strength(number: int, community: int | None, rule: str) -> float:
    """Rough 0..1 hand strength, shared by the scripted opponents."""
    if community is None:
        if rule == "low_ball":
            return (14 - number) / 13.0
        if rule == "wild_seven" and number == 7:
            return 0.95
        return number / 13.0
    wins = 0.0
    for other in range(1, 14):
        result = compare(number, other, community, rule)
        wins += 1.0 if result > 0 else (0.5 if result == 0 else 0.0)
    return wins / 13.0


def scripted_move(style: str, req: MoveRequest, rng: random.Random) -> tuple[str, int | None]:
    strength = _strength(req.your_number, req.community_number, req.table_rule)
    live = sum(1 for p in req.players if not p.folded and not p.busted)
    strength = strength ** max(1, live - 1)
    pot_odds = req.to_call / max(req.pot + req.to_call, 1)

    if style == "station":
        call_floor, aggression, bluff = 0.05, 0.05, 0.0
    elif style == "rock":
        call_floor, aggression, bluff = 0.30, 0.10, 0.0
    elif style == "maniac":
        call_floor, aggression, bluff = 0.10, 0.45, 0.30
    else:  # "solid"
        call_floor, aggression, bluff = 0.18, 0.25, 0.08

    can = set(req.legal_actions)
    if req.to_call == 0:
        if "bet" in can and (strength > 1 - aggression or rng.random() < bluff):
            lo, hi = req.min_raise_to or 0, req.max_raise_to or 0
            if hi >= lo > 0:
                size = lo + int((hi - lo) * min(0.35, 0.1 + strength * 0.3))
                return "bet", max(lo, min(hi, size))
        return ("check" if "check" in can else "call"), None

    if "raise" in can and strength > 1 - aggression * 0.5 and rng.random() < 0.6:
        lo, hi = req.min_raise_to or 0, req.max_raise_to or 0
        if hi >= lo > 0:
            return "raise", max(lo, min(hi, lo + int((hi - lo) * 0.25)))
    if strength >= pot_odds + call_floor and "call" in can:
        return "call", None
    return ("fold" if "fold" in can else "check"), None


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------


@dataclass
class Seat:
    seat: int
    name: str
    style: str
    stack: int
    delta: int = 0
    folded: bool = False
    busted: bool = False
    all_in: bool = False
    bet: int = 0
    number: int = 0
    committed: int = 0


@dataclass
class Table:
    rule: str
    hero_seat: int
    styles: list[str]
    stack: int = 100
    sb: int = 1
    bb: int = 2
    rng: random.Random = field(default_factory=random.Random)
    seats: list[Seat] = field(default_factory=list)
    history: list[RecentHand] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = ["Dana", "Miles", "Theo"]
        idx = 0
        for s in range(4):
            if s == self.hero_seat:
                self.seats.append(Seat(s, "You", "hero", self.stack))
            else:
                self.seats.append(Seat(s, names[idx], self.styles[idx], self.stack))
                idx += 1

    def live(self) -> list[Seat]:
        return [s for s in self.seats if not s.folded and not s.busted]

    def active(self) -> list[Seat]:
        return [s for s in self.seats if not s.busted]

    def _states(self) -> list[PlayerState]:
        return [
            PlayerState(
                seat=s.seat, name=s.name, folded=s.folded, chip_delta=s.delta,
                bet_this_round=s.bet, stack=s.stack, all_in=s.all_in, busted=s.busted,
            )
            for s in self.seats
        ]

    def _next_active(self, seat: int) -> int:
        order = sorted(s.seat for s in self.active())
        for candidate in order:
            if candidate > seat:
                return candidate
        return order[0]

    def play_leg(self, hands: int = 60) -> dict[str, int]:
        button = 0
        for hand_no in range(1, hands + 1):
            if len(self.active()) < 2:
                break
            while self.seats[button].busted:
                button = self._next_active(button)
            self._play_hand(hand_no, hands, button)
            button = self._next_active(button)
        return {s.name: s.delta for s in self.seats}

    def _play_hand(self, hand_no: int, total: int, button: int) -> None:
        for s in self.seats:
            s.folded = s.busted
            s.bet = 0
            s.committed = 0
            s.all_in = False
            s.number = self.rng.randint(1, 13)
        community = self.rng.randint(1, 13)
        pot = 0
        actions: list[HandAction] = []

        sb_seat = self._next_active(button)
        bb_seat = self._next_active(sb_seat)
        for seat, amount in ((sb_seat, self.sb), (bb_seat, self.bb)):
            player = self.seats[seat]
            pay = min(amount, player.stack)
            player.stack -= pay
            player.bet = pay
            player.committed = pay
            pot += pay
            if player.stack == 0:
                player.all_in = True

        pot = self._betting(
            "pre_reveal", self._next_active(bb_seat), pot, community, hand_no, total,
            button, actions, opening_bet=self.bb,
        )
        if len(self.live()) > 1:
            for s in self.seats:
                s.bet = 0
            pot = self._betting(
                "post_reveal", self._next_active(button), pot, community, hand_no, total,
                button, actions, opening_bet=0, community_known=True,
            )

        contenders = [s for s in self.live()]
        if contenders:
            best = contenders[0]
            winners = [best]
            for other in contenders[1:]:
                result = compare(other.number, best.number, community, self.rule)
                if result > 0:
                    best, winners = other, [other]
                elif result == 0:
                    winners.append(other)
            share = pot // len(winners)
            for w in winners:
                w.stack += share
            self.seats[winners[0].seat].stack += pot - share * len(winners)

        for s in self.seats:
            s.delta = s.stack - self.stack
            if s.stack <= 0 and not s.busted:
                s.busted = True

        self.history.append(
            RecentHand(
                hand_number=hand_no, community_number=community,
                winners=[w.seat for w in (winners if contenders else [])], pot=pot,
                shown_numbers={str(s.seat): s.number for s in contenders},
                actions=list(actions),
            )
        )
        self.history = self.history[-12:]

    def _betting(
        self, rnd: str, first: int, pot: int, community: int, hand_no: int, total: int,
        button: int, actions: list[HandAction], opening_bet: int,
        community_known: bool = False,
    ) -> int:
        current = opening_bet
        last_raise = max(self.bb, 1)
        order = sorted(s.seat for s in self.active())
        if first in order:
            i = order.index(first)
            order = order[i:] + order[:i]

        need_action = {s.seat for s in self.live() if not s.all_in}
        acted: set[int] = set()
        guard = 0

        while guard < 40:
            guard += 1
            progressed = False
            for seat in order:
                player = self.seats[seat]
                if player.folded or player.busted or player.all_in:
                    continue
                if len(self.live()) < 2:
                    return pot
                if seat in acted and player.bet == current:
                    continue
                progressed = True

                to_call = min(current - player.bet, player.stack)
                legal = ["fold"]
                if to_call == 0:
                    legal = ["check", "fold"]
                    if player.stack > 0:
                        legal.append("bet")
                else:
                    legal.append("call")
                    if player.stack > to_call:
                        legal.append("raise")

                min_raise_to = current + last_raise
                max_raise_to = player.bet + player.stack
                if min_raise_to > max_raise_to:
                    min_raise_to = max_raise_to
                    if "raise" in legal and max_raise_to <= current:
                        legal.remove("raise")
                    if "bet" in legal and max_raise_to <= 0:
                        legal.remove("bet")

                req = MoveRequest(
                    protocol_version=1, match_id="sim", phase=3, table_rule=self.rule,
                    small_blind=self.sb, big_blind=self.bb, starting_stack=self.stack,
                    your_stack=player.stack, hand_number=hand_no, total_hands=total,
                    round=rnd, your_number=player.number,
                    community_number=community if community_known else None,
                    your_seat=seat, button_seat=button, pot=pot, to_call=to_call,
                    min_raise_to=min_raise_to if ("bet" in legal or "raise" in legal) else None,
                    max_raise_to=max_raise_to if ("bet" in legal or "raise" in legal) else None,
                    legal_actions=legal, players=self._states(),
                    current_hand_actions=list(actions), recent_hands=list(self.history),
                    leg_number=1, total_legs=4,
                )

                if player.style == "hero":
                    move = choose_move(req)
                    action, amount = move.action, move.amount
                else:
                    action, amount = scripted_move(player.style, req, self.rng)
                if action not in legal:
                    action, amount = ("check" if "check" in legal else "fold"), None

                acted.add(seat)
                if action == "fold":
                    player.folded = True
                elif action == "check":
                    pass
                elif action == "call":
                    pay = min(to_call, player.stack)
                    player.stack -= pay
                    player.bet += pay
                    player.committed += pay
                    pot += pay
                    if player.stack == 0:
                        player.all_in = True
                else:  # bet / raise
                    target = max(min_raise_to, min(max_raise_to, amount or min_raise_to))
                    pay = min(target - player.bet, player.stack)
                    player.stack -= pay
                    player.bet += pay
                    player.committed += pay
                    pot += pay
                    if player.bet > current:
                        last_raise = max(player.bet - current, self.bb)
                        current = player.bet
                        acted = {seat}
                    if player.stack == 0:
                        player.all_in = True

                actions.append(HandAction(round=rnd, seat=seat, action=action, amount=amount))

            live_unacted = [
                s for s in self.live()
                if not s.all_in and (s.seat not in acted or s.bet != current)
            ]
            if not progressed or not live_unacted:
                break
        return pot


def run(legs: int, seed: int, hands: int = 60) -> None:
    style_pool = ["solid", "station", "maniac", "rock"]
    totals: dict[str, list[int]] = {r: [] for r in TABLE_RULES}
    cleared: dict[str, int] = {r: 0 for r in TABLE_RULES}

    for rule in TABLE_RULES:
        # Reset per rule so every rule sees the same opponents, seats and
        # deals. Without this the rules differ by luck as much as by rule -
        # "standard" and "pair_bounty" are scored identically here, so any gap
        # between them is pure noise and a useful check on the harness itself.
        rng = random.Random(seed)
        for leg in range(legs):
            styles = rng.sample(style_pool, 3)
            table = Table(
                rule=rule, hero_seat=leg % 4, styles=styles,
                rng=random.Random(rng.randrange(1 << 30)),
            )
            result = table.play_leg(hands)
            hero = result["You"]
            others = [v for k, v in result.items() if k != "You"]
            totals[rule].append(hero)
            if hero >= 10 and hero > max(others):
                cleared[rule] += 1

    print(f"{'rule':<14}{'avg delta':>11}{'cleared':>10}{'  (delta>=+10 and strictly top)':>0}")
    grand = 0
    for rule in TABLE_RULES:
        deltas = totals[rule]
        avg = sum(deltas) / len(deltas)
        grand += cleared[rule]
        print(f"{rule:<14}{avg:>11.1f}{cleared[rule]:>6}/{legs}")
    print(f"{'TOTAL':<14}{'':>11}{grand:>6}/{legs*len(TABLE_RULES)}"
          f"   -> {25*grand/legs:.1f} pts per 4-leg run (max 100)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--legs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hands", type=int, default=60)
    args = parser.parse_args()
    run(args.legs, args.seed, args.hands)
