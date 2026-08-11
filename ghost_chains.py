from __future__ import annotations

import heapq
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Iterable

from fastapi import APIRouter
from pydantic import BaseModel


LOOKBACK_WINDOW = timedelta(hours=24)

# Phase 2 identity weights. Devices are personal, so reuse across accounts is
# a stronger control signal than a shared IP (office Wi-Fi / NAT aggregates
# unrelated users behind one address).
#   aligned:   the other user sits on the directed flow line through this tx —
#              identity lining up with structural flow, the strongest combo.
#   component: same weakly-connected component but off the flow line
#              (branch/sibling reuse).
#   cross:     reuse from a structurally disconnected component — a
#              coordination hint, not proof on its own.
_IDENTITY_WEIGHTS = {
    "device": {"aligned": 0.60, "component": 0.25, "cross": 0.30},
    "ip": {"aligned": 0.45, "component": 0.18, "cross": 0.22},
}
# A single cross-component reuse can be coincidence; halve it until corroborated.
_SINGLE_CROSS_DAMP = 0.5


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
    ip_address: str | None = None
    device_id: str | None = None


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
        # Identity index (Phase 2): identity value -> Counter of initiating
        # users with active transactions carrying that value.
        self._identity_users: dict[str, dict[str, Counter[str]]] = {
            "ip": defaultdict(Counter),
            "device": defaultdict(Counter),
        }
        # Active-transaction count per directed (from, to) pair, so an edge
        # only leaves the graph when its *last* active transaction expires.
        self._edge_counts: Counter[tuple[str, str]] = Counter()
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
        risk_score = self._score_transaction(
            transaction.fromUserId,
            transaction.toUserId,
            transaction.ipAddress,
            transaction.deviceId,
        )

        stored = _StoredTransaction(
            tx_id=transaction.txId,
            from_user_id=transaction.fromUserId,
            to_user_id=transaction.toUserId,
            amount=transaction.amount,
            created_at=created_at,
            canonical_payload=canonical_payload,
            risk_score=risk_score,
            ip_address=transaction.ipAddress,
            device_id=transaction.deviceId,
        )
        self._active_transactions[transaction.txId] = stored
        self._seen_transactions[transaction.txId] = stored
        self._sequence += 1
        sequence = self._sequence
        heapq.heappush(self._tx_order, (created_at, sequence, transaction.txId))
        heapq.heappush(self._seen_order, (created_at, sequence, transaction.txId))

        self._outgoing[transaction.fromUserId].add(transaction.toUserId)
        self._incoming[transaction.toUserId].add(transaction.fromUserId)
        self._edge_counts[(transaction.fromUserId, transaction.toUserId)] += 1
        if transaction.ipAddress is not None:
            self._identity_users["ip"][transaction.ipAddress][transaction.fromUserId] += 1
        if transaction.deviceId is not None:
            self._identity_users["device"][transaction.deviceId][transaction.fromUserId] += 1

        return GhostChainsTransactionResult(txId=transaction.txId, riskScore=risk_score)

    def _expire_state(self, current_time: datetime) -> None:
        cutoff = current_time - LOOKBACK_WINDOW

        while self._tx_order and self._tx_order[0][0] < cutoff:
            _, _, tx_id = heapq.heappop(self._tx_order)
            stored = self._active_transactions.pop(tx_id, None)
            if stored is None:
                continue
            self._remove_edge(stored.from_user_id, stored.to_user_id)
            if stored.ip_address is not None:
                self._release_identity("ip", stored.ip_address, stored.from_user_id)
            if stored.device_id is not None:
                self._release_identity("device", stored.device_id, stored.from_user_id)

        while self._seen_order and self._seen_order[0][0] < cutoff:
            _, _, tx_id = heapq.heappop(self._seen_order)
            self._seen_transactions.pop(tx_id, None)

    def _release_identity(self, dimension: str, value: str, user_id: str) -> None:
        index = self._identity_users[dimension]
        counter = index.get(value)
        if counter is None:
            return
        counter[user_id] -= 1
        if counter[user_id] <= 0:
            del counter[user_id]
        if not counter:
            index.pop(value, None)

    def _remove_edge(self, from_user_id: str, to_user_id: str) -> None:
        remaining = self._edge_counts[(from_user_id, to_user_id)] - 1
        if remaining > 0:
            self._edge_counts[(from_user_id, to_user_id)] = remaining
            return
        del self._edge_counts[(from_user_id, to_user_id)]

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

    def _score_transaction(
        self,
        from_user_id: str,
        to_user_id: str,
        ip_address: str | None = None,
        device_id: str | None = None,
    ) -> float:
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

        raw += self._identity_contribution(
            from_user_id, to_user_id, ip_address, device_id, ancestors, descendants
        )

        score = 1.0 - math.exp(-raw / 3.5)
        return round(max(0.0, min(1.0, score)), 6)

    def _identity_contribution(
        self,
        from_user_id: str,
        to_user_id: str,
        ip_address: str | None,
        device_id: str | None,
        ancestors: set[str],
        descendants: set[str],
    ) -> float:
        """Phase 2 identity signal, relative to the transaction's graph position.

        For each identity dimension present, other users who initiated active
        transactions with the same value are split by structural relation to
        this transaction: on its directed flow line (identity lining up with
        structural flow - strongest), elsewhere in the same weakly-connected
        component (branch reuse), or in a disconnected component (coordination
        hint, dampened when it is a single reuse). Dimensions are independent
        and additive; absent fields contribute nothing, so Phase 1 scoring is
        unchanged for identity-free traffic.
        """
        if ip_address is None and device_id is None:
            return 0.0

        flow_line = ancestors | descendants | {from_user_id, to_user_id}
        component: set[str] | None = None  # computed lazily, only when needed

        contribution = 0.0
        for dimension, value in (("ip", ip_address), ("device", device_id)):
            if value is None:
                continue
            counter = self._identity_users[dimension].get(value)
            if not counter:
                continue
            others = set(counter) - {from_user_id}
            if not others:
                # A user reusing their own device/address is normal behaviour.
                continue

            aligned = others & flow_line
            remainder = others - flow_line
            if remainder and component is None:
                component = self._weakly_connected((from_user_id, to_user_id))
            component_only = remainder & component if remainder else set()
            cross = remainder - component_only

            weights = _IDENTITY_WEIGHTS[dimension]
            contribution += weights["aligned"] * math.log1p(len(aligned))
            contribution += weights["component"] * math.log1p(len(component_only))
            cross_term = weights["cross"] * math.log1p(len(cross))
            if len(cross) == 1:
                cross_term *= _SINGLE_CROSS_DAMP
            contribution += cross_term

        return contribution

    def _weakly_connected(self, seeds: Iterable[str]) -> set[str]:
        visited = set(seeds)
        stack = list(visited)
        while stack:
            node = stack.pop()
            for neighbor in self._outgoing.get(node, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
            for neighbor in self._incoming.get(node, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited

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