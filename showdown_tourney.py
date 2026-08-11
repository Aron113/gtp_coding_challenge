"""Phase 4 knockout harness.

Phase 4 pays for rank, not chips, so measuring average chip delta would be
measuring the wrong thing. This runs the tournament shape described in the
brief - up to seven bots a table, 200 hands, bottom third cut each round,
survivors reshuffled until one table remains - and reports survival and
finishing rank.

The field is scripted archetypes, because the real field is other teams' bots
and cannot be practised against. Read the numbers as "does the survival
objective beat playing for chips", not as a predicted placing.
"""

from __future__ import annotations

import argparse
import math
import random

import showdown_sim as sim

STYLE_POOL = ["solid", "station", "maniac", "rock"]
FINAL_TABLE_PAY = {1: 100, 2: 90, 3: 80, 4: 70, 5: 60, 6: 50, 7: 40}


def play_table(rule: str, hero_index: int, styles: list[str], hands: int, rng: random.Random):
    """Return {name: delta} for one table; hero sits at `hero_index`."""
    table = sim.Table(
        rule=rule, hero_seat=hero_index, styles=styles, rng=rng,
    )
    return table.play_leg(hands)


def run_tournament(
    entrants: int, hero_phase: int, rng: random.Random, hands: int = 200
) -> tuple[int, int]:
    """Play one knockout. Returns (final rank, rounds survived).

    Rank 1 is the winner of the final table. A bot cut earlier gets the rank
    it held when cut, counted from the bottom of the surviving field.
    """
    # Field: hero plus scripted bots, each carrying a persistent style.
    field: list[dict] = [{"hero": True, "style": "hero", "alive": True}]
    for _ in range(entrants - 1):
        field.append({"hero": False, "style": rng.choice(STYLE_POOL), "alive": True})

    rounds = 0
    while True:
        alive = [p for p in field if p["alive"]]
        if len(alive) <= 1:
            break
        rounds += 1
        rng.shuffle(alive)

        # Seat the field in tables of at most seven.
        table_count = max(1, math.ceil(len(alive) / 7))
        tables: list[list[dict]] = [[] for _ in range(table_count)]
        for i, player in enumerate(alive):
            tables[i % table_count].append(player)

        rule = rng.choice(sim.TABLE_RULES)
        for group in tables:
            if len(group) < 2:
                group[0]["delta"] = 0
                continue
            hero_index = next(
                (i for i, p in enumerate(group) if p["hero"]), None
            )
            styles = [p["style"] for p in group if not p["hero"]]
            if hero_index is None:
                hero_index = 0
                styles = [p["style"] for p in group][1:]
            result = play_table(rule, hero_index, styles, hands,
                                random.Random(rng.randrange(1 << 30)))
            names = list(result.keys())
            # Map results back onto the seated players by seat order.
            for seat, player in enumerate(group):
                key = "You" if seat == hero_index else f"Player {seat + 1}"
                player["delta"] = result.get(key, 0)

        if len(alive) <= 7:
            # Final table: rank by delta, best first.
            alive.sort(key=lambda p: p["delta"], reverse=True)
            for rank, player in enumerate(alive, start=1):
                if player["hero"]:
                    return rank, rounds
            return len(alive), rounds

        # Cut the bottom third of the whole surviving field.
        alive.sort(key=lambda p: p["delta"], reverse=True)
        cut = max(1, int(math.floor(len(alive) * (1.0 / 3.0))))
        survivors = alive[: len(alive) - cut]
        for player in alive[len(alive) - cut :]:
            player["alive"] = False
            if player["hero"]:
                # Cut: rank is where we finished among those still standing.
                return len(survivors) + 1, rounds
        if not any(p["hero"] for p in survivors):
            return len(survivors) + 1, rounds

    return 1, rounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--entrants", type=int, default=40)
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--phase", type=int, default=4, help="hero phase to exercise")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    made_final = 0
    ranks: list[int] = []
    rounds_survived: list[int] = []

    original = sim.choose_move
    from showdown.models import MoveRequest
    from showdown.strategy import choose_move as dispatch

    def as_phase(req: MoveRequest):
        return dispatch(req.model_copy(update={"phase": args.phase}))

    sim.choose_move = as_phase
    try:
        for _ in range(args.runs):
            rank, rounds = run_tournament(
                args.entrants, args.phase, random.Random(rng.randrange(1 << 30)), args.hands
            )
            ranks.append(rank)
            rounds_survived.append(rounds)
            if rank <= 7:
                made_final += 1
    finally:
        sim.choose_move = original

    avg_rank = sum(ranks) / len(ranks)
    avg_rounds = sum(rounds_survived) / len(rounds_survived)
    finals_pay = sum(FINAL_TABLE_PAY.get(r, 0) for r in ranks if r <= 7)
    print(f"phase {args.phase} hero, {args.runs} tournaments of {args.entrants} entrants")
    print(f"  reached final table : {made_final}/{args.runs} ({100*made_final/args.runs:.1f}%)")
    print(f"  average finish rank : {avg_rank:.1f}")
    print(f"  average rounds      : {avg_rounds:.2f}")
    print(f"  avg pts from finals : {finals_pay/args.runs:.1f} (final-table pay only)")


if __name__ == "__main__":
    main()
