from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RoundName = Literal["pre_reveal", "post_reveal"]
ActionName = Literal["check", "call", "bet", "raise", "fold"]


class PlayerState(BaseModel):
    seat: int
    name: str
    folded: bool
    chip_delta: int
    bet_this_round: int
    stack: int
    all_in: bool
    busted: bool


class HandAction(BaseModel):
    round: RoundName
    seat: int
    action: ActionName
    amount: int | None = None


class RecentHand(BaseModel):
    hand_number: int
    community_number: int | None = None
    winners: list[int] = Field(default_factory=list)
    pot: int
    shown_numbers: dict[str, int] = Field(default_factory=dict)
    actions: list[HandAction] = Field(default_factory=list)


class MoveRequest(BaseModel):
    """A request to act.

    Everything carries a default. Phase 4 is a one-shot knockout with no
    retries, and a request we reject as malformed is a mispay just as surely
    as a crash is - so a field we did not expect to be missing must never be
    the reason we fail to answer. The strategy layer treats these defaults as
    "unknown" and falls back to a safe legal action.
    """

    protocol_version: int = 1
    match_id: str = ""
    phase: int = 3
    table_rule: str = "standard"
    small_blind: int = 1
    big_blind: int = 2
    starting_stack: int = 0
    your_stack: int = 0
    hand_number: int = 1
    total_hands: int = 1
    round: RoundName = "pre_reveal"
    your_number: int = 0
    community_number: int | None = None
    your_seat: int = 0
    button_seat: int = 0
    pot: int = 0
    to_call: int = 0
    min_raise_to: int | None = None
    max_raise_to: int | None = None
    legal_actions: list[ActionName] = Field(default_factory=list)
    players: list[PlayerState] = Field(default_factory=list)
    current_hand_actions: list[HandAction] = Field(default_factory=list)
    recent_hands: list[RecentHand] = Field(default_factory=list)
    leg_number: int | None = None
    total_legs: int | None = None


class MoveResponse(BaseModel):
    action: ActionName
    amount: int | None = None
