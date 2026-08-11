"""Recall harness for Stage 2 revision retrieval.

Questions are written the way the brief writes them: paraphrased away from the
source wording, so "sensor grid ... brought back into alignment" has to reach a
sentence that says "Kesterline array was recalibrated". A case passes only if
the required fact is inside the passages we would actually return, within
budget.
"""

from __future__ import annotations

import re
import sys

from nursery_revision import (
    TOKEN_BUDGET,
    budget_used,
    classify_question,
    find_passages,
)

HARD_LIMIT = 900

# (question, fact that must appear, note)
CASES: list[tuple[str, str, str]] = [
    # The brief's own worked example - deliberately shares no vocabulary with
    # the answer, and "alignment" appears only in the *wrong* document.
    ("When was the sensor grid last brought back into alignment?", "14 March", "brief example"),
    ("When did the air purifier break down at the deep sea base?", "2 November", "scrubber paraphrase"),
    ("How deep below the surface does the main undersea habitat sit?", "6,214", "depth"),
    ("Who is in charge of the undersea research outpost?", "Ansel Kovrith", "director"),
    ("What is the radio identifier used by the deep sea station?", "Umbral Seven", "call sign"),
    ("How often does the supply ship arrive at the trench base?", "19 days", "resupply"),
    ("What is the longest a diver may stay outside per excursion?", "47 minutes", "dive limit"),
    ("How tightly must the underwater microphone housing seal be fastened?", "12 newton-meters", "torque"),
    ("How many researchers live aboard the trench outpost?", "forty-one", "headcount"),
    ("What is the name of the main deep-sea craft used for long trips?", "Halcyon Drift", "submersible"),

    ("Who runs the city transport network?", "Dorian Fenwick", "transit head"),
    ("What is the most a rider pays in a single day?", "four pounds ninety", "fare cap"),
    ("How long is the busiest rail corridor end to end?", "thirty-four point two", "line length"),
    ("How many qualified drivers does the transit body employ?", "sixty-eight", "drivers"),
    ("What radio identifier does the transit control room use?", "Fantail Nine", "call sign"),
    ("When did the signalling failure snarl up the rail network?", "fifth of January", "incident"),

    ("What dose do participants settle on for the long run?", "240 milligram", "maintenance dose"),
    ("When did dosing start under the revised plan?", "3 June", "dosing start"),
    ("At what liver enzyme reading is a participant pulled from dosing?", "260 units per liter", "threshold"),
    ("Who leads the Velmara study?", "Reva Sandoval", "investigator"),
    ("How long must patients be watched after each injection?", "90 minutes", "observation"),
    ("What internal code tracks the trial paperwork?", "VLM-204-B", "tracking code"),

    ("Which release first carried the reworked deferred lighting?", "Release 14", "engine release"),
    ("Who is the lead architect of the game engine?", "Perrin Ashwicke", "architect"),
    ("How much texture memory may a frame use on the base console?", "512 megabytes", "memory cap"),
    ("How often does the automated regression suite run during milestones?", "every six hours", "cadence"),
    ("What build tag marks the current stable engine stream?", "Driftglass Nine", "build tag"),
    ("How long before a texture streaming pass counts as stalled?", "9 seconds", "stall"),

    ("Who chairs the growers cooperative board?", "Cordelia Vance", "chair"),
    ("How many member households hold active shares?", "fifty-four", "members"),
    ("Above what moisture level is grain marked down a grade?", "eighteen percent", "grading"),
    ("When did the chiller fail at the eastern store?", "6 April", "incident"),
    ("How long may a storage bay sit idle before it is taken back?", "ninety days", "forfeit"),
]


def normalise(text: str) -> str:
    return re.sub(r"[\s,]+", " ", text.lower())


def main() -> int:
    passed = failed = 0
    over_budget = 0
    failures: list[str] = []

    for question, fact, note in CASES:
        passages = find_passages(question)
        used = budget_used(passages)
        blob = normalise(" ".join(passages))
        hit = normalise(fact) in blob

        if used > HARD_LIMIT:
            over_budget += 1
        if hit and used <= HARD_LIMIT:
            passed += 1
        else:
            failed += 1
            failures.append(
                f"  MISS [{classify_question(question):7}] {question}\n"
                f"        want {fact!r} ({note}) | {len(passages)} passages, {used} tokens"
            )

    total = len(CASES)
    print(f"recall: {passed}/{total} ({100*passed/total:.0f}%)")
    print(f"over budget (>{HARD_LIMIT}): {over_budget}")
    if failures:
        print("\n" + "\n".join(failures))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
