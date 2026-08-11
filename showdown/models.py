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
    protocol_version: int
    match_id: str
    phase: int
    table_rule: str
    small_blind: int
    big_blind: int
    starting_stack: int
    your_stack: int
    hand_number: int
    total_hands: int
    round: RoundName
    your_number: int
    community_number: int | None = None
    your_seat: int
    button_seat: int
    pot: int
    to_call: int
    min_raise_to: int | None = None
    max_raise_to: int | None = None
    legal_actions: list[ActionName]
    players: list[PlayerState]
    current_hand_actions: list[HandAction] = Field(default_factory=list)
    recent_hands: list[RecentHand] = Field(default_factory=list)
    leg_number: int | None = None
    total_legs: int | None = None


class MoveResponse(BaseModel):
    action: ActionName
    amount: int | None = None
