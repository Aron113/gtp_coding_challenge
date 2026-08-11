from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel


LOOKBACK_WINDOW = timedelta(hours=24)


class GhostChainsResetRequest(BaseModel):
    clearTransactions: bool

    class Config:
        extra = "ignore"


class GhostChainsTransactionInput(BaseModel):
    txId: str
    fromUserId: str
    toUserId: str
    amount: float
    createdAt: str
    ipAddress: str | None = None
    deviceId: str | None = None

    class Config:
        extra = "ignore"


class GhostChainsTransactionsRequest(BaseModel):
    transactions: list[GhostChainsTransactionInput]

    class Config:
        extra = "ignore"


class GhostChainsTransactionResult(BaseModel):
    txId: str
    riskScore: float


class GhostChainsTransactionsResponse(BaseModel):
    transactions: list[GhostChainsTransactionResult]


@dataclass
class _StoredTransaction:
    tx_id: str
    from_user_id: str
    to_user_id: str
    amount: float
    created_at: datetime
    canonical_payload: tuple[Any, ...]
    risk_score: float


class GhostChainsService:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._active_transactions: dict[str, _StoredTransaction] = {}
        self._seen_transactions: dict[str, _StoredTransaction] = {}
        self._tx_order: list[tuple[datetime, int, str]] = []
        self._seen_order: list[tuple[datetime, int, str]] = []
        self._outgoing: dict[str, set[str]] = defaultdict(set)
        self._incoming: dict[str, set[str]] = defaultdict(set)
        self._sequence = 0

    def process_transactions(self, transactions: list[GhostChainsTransactionInput]) -> list[GhostChainsTransactionResult]:
        results: list[GhostChainsTransactionResult] = []
        for transaction in transactions:
            results.append(self._process_transaction(transaction))
        return results

    def _process_transaction(self, transaction: GhostChainsTransactionInput) -> GhostChainsTransactionResult:
        created_at = _parse_iso_datetime(transaction.createdAt)
        canonical_payload = self._canonical_payload(transaction)

        # Duplicate txId: return the original score and mutate nothing. This
        # holds even when the payload differs — a 409 would fail the whole
        # batch after earlier transactions already committed, losing their
        # scores; the txId is the identity, so first write wins.
        cached = self._seen_transactions.get(transaction.txId)
        if cached is not None:
            return GhostChainsTransactionResult(txId=transaction.txId, riskScore=cached.risk_score)

        self._expire_state(created_at)
        risk_score = self._score_transaction(transaction.fromUserId, transaction.toUserId)

        stored = _StoredTransaction(
            tx_id=transaction.txId,
            from_user_id=transaction.fromUserId,
            to_user_id=transaction.toUserId,
            amount=transaction.amount,
            created_at=created_at,
            canonical_payload=canonical_payload,
            risk_score=risk_score,
        )
        self._active_transactions[transaction.txId] = stored
        self._seen_transactions[transaction.txId] = stored
        self._sequence += 1
        sequence = self._sequence
        heapq.heappush(self._tx_order, (created_at, sequence, transaction.txId))
        heapq.heappush(self._seen_order, (created_at, sequence, transaction.txId))

        self._outgoing[transaction.fromUserId].add(transaction.toUserId)
        self._incoming[transaction.toUserId].add(transaction.fromUserId)

        return GhostChainsTransactionResult(txId=transaction.txId, riskScore=risk_score)

    def _expire_state(self, current_time: datetime) -> None:
        cutoff = current_time - LOOKBACK_WINDOW

        while self._tx_order and self._tx_order[0][0] < cutoff:
            _, _, tx_id = heapq.heappop(self._tx_order)
            stored = self._active_transactions.pop(tx_id, None)
            if stored is None:
                continue
            self._remove_edge(stored.from_user_id, stored.to_user_id)

        while self._seen_order and self._seen_order[0][0] < cutoff:
            _, _, tx_id = heapq.heappop(self._seen_order)
            self._seen_transactions.pop(tx_id, None)

    def _remove_edge(self, from_user_id: str, to_user_id: str) -> None:
        outgoing = self._outgoing.get(from_user_id)
        if outgoing is not None:
            outgoing.discard(to_user_id)
            if not outgoing:
                self._outgoing.pop(from_user_id, None)

        incoming = self._incoming.get(to_user_id)
        if incoming is not None:
            incoming.discard(from_user_id)
            if not incoming:
                self._incoming.pop(to_user_id, None)

    def _score_transaction(self, from_user_id: str, to_user_id: str) -> float:
        ancestors = self._reverse_reachable(from_user_id)
        descendants = self._reachable(to_user_id)
        cycle_distance = self._shortest_path_length(to_user_id, from_user_id)

        upstream_support = len(ancestors) + 1
        downstream_support = len(descendants) + 1
        incoming_before = self._incoming.get(to_user_id, set())
        outgoing_before = self._outgoing.get(from_user_id, set())
        incoming_after = len(incoming_before | {from_user_id})
        outgoing_after = len(outgoing_before | {to_user_id})

        closure_mass = upstream_support * downstream_support
        frontier_mass = incoming_after * outgoing_after

        raw = 0.0
        raw += 0.90 * math.log1p(closure_mass)
        raw += 0.35 * math.log1p(frontier_mass)
        raw += 0.15 * math.log1p(upstream_support + downstream_support)

        if incoming_after >= 2:
            raw += 0.18 * math.log1p(incoming_after)
        if outgoing_after >= 2:
            raw += 0.12 * math.log1p(outgoing_after)
        if incoming_after >= 2 and outgoing_after >= 2:
            raw += 0.10

        if cycle_distance is not None:
            loop_mass = closure_mass + len(self._reverse_reachable(to_user_id)) + len(self._reachable(from_user_id))
            raw += 1.35
            raw += 0.45 / (cycle_distance + 1)
            raw += 0.22 * math.log1p(loop_mass)

        score = 1.0 - math.exp(-raw / 3.5)
        return round(max(0.0, min(1.0, score)), 6)

    def _reachable(self, start: str) -> set[str]:
        visited: set[str] = set()
        stack = list(self._outgoing.get(start, set()))
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._outgoing.get(node, set()))
        return visited

    def _reverse_reachable(self, start: str) -> set[str]:
        visited: set[str] = set()
        stack = list(self._incoming.get(start, set()))
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._incoming.get(node, set()))
        return visited

    def _shortest_path_length(self, start: str, target: str) -> int | None:
        if start == target:
            return 0

        visited = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            node, distance = queue.popleft()
            for next_node in self._outgoing.get(node, set()):
                if next_node == target:
                    return distance + 1
                if next_node in visited:
                    continue
                visited.add(next_node)
                queue.append((next_node, distance + 1))
        return None

    def _canonical_payload(self, transaction: GhostChainsTransactionInput) -> tuple[Any, ...]:
        return (
            transaction.txId,
            transaction.fromUserId,
            transaction.toUserId,
            float(transaction.amount),
            transaction.createdAt,
            transaction.ipAddress,
            transaction.deviceId,
        )


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


service = GhostChainsService()
router = APIRouter(prefix="/ghost-chains")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/reset")
def reset(request: GhostChainsResetRequest) -> dict[str, bool]:
    service.reset()
    return {"clearTransactions": request.clearTransactions}


@router.post("/transactions", response_model=GhostChainsTransactionsResponse)
def process_transactions(request: GhostChainsTransactionsRequest) -> GhostChainsTransactionsResponse:
    results = service.process_transactions(request.transactions)
    return GhostChainsTransactionsResponse(transactions=results)