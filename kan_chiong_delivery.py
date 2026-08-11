"""Kan Chiong Delivery Driver - time-dependent fastest route.

Fastest route on a road network where directional obstructions change edge
speed over time. No waiting at nodes is allowed, so the search runs Dijkstra
over (node, arrival_time) states rather than plain nodes: arriving *later* at
a node can be strictly better (an entry-blocked edge may reopen), which breaks
the FIFO property plain Dijkstra needs. Cycling edges to burn time is legal
and sometimes optimal (see challenge example 3).
"""

from __future__ import annotations

import heapq
import json
from datetime import datetime, timezone

EPS = 1e-9
# Safety valve for pathological inputs; well above anything the tests need.
MAX_POPS = 500_000


def _parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _format_time(ts: float) -> str:
    # Snap to a whole second when we're within float noise of one.
    if abs(ts - round(ts)) < 1e-6:
        ts = round(ts)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    if dt.microsecond == 0:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f").rstrip("0") + "Z"


def _factor_at(obstructions: list[tuple[float, float, float]], t: float) -> float:
    """Combined speed factor on a directed arc at time t.

    Intervals are half-open [start, end): an obstruction ending exactly at t
    no longer applies, matching example 3 where the edge is usable the instant
    a block lifts. Overlapping obstructions multiply.
    """
    factor = 1.0
    for start, end, speed in obstructions:
        if start <= t < end:
            factor *= speed
    return factor


def _traverse(
    obstructions: list[tuple[float, float, float]], depart: float, base_duration: float
) -> float | None:
    """Arrival time for entering the arc at `depart`, or None if entry is blocked.

    Entry with factor 0 is forbidden (no waiting at nodes). If a factor-0
    obstruction activates mid-traversal, only the untraveled remainder stalls
    until the obstruction lifts - the driver is already on the road.
    """
    if _factor_at(obstructions, depart) == 0.0:
        return None
    t = depart
    remaining = float(base_duration)
    if remaining <= 0:
        return t
    boundaries = sorted({b for s, e, _ in obstructions for b in (s, e)})
    while True:
        factor = _factor_at(obstructions, t)
        next_boundary = next((b for b in boundaries if b > t + EPS), None)
        if factor == 0.0:
            # Mid-traversal stall; obstructions are finite so a boundary exists.
            t = next_boundary
            continue
        if next_boundary is None:
            return t + remaining / factor
        progress = (next_boundary - t) * factor
        if progress >= remaining - EPS:
            return t + remaining / factor
        remaining -= progress
        t = next_boundary


def solve(data: str) -> str:
    payload = json.loads(data)

    start = tuple(payload["start_coordinate"])
    end = tuple(payload["end_coordinate"])
    start_time = _parse_time(payload["start_time"])

    no_route = json.dumps(
        {"total_duration_sec": None, "arrival_time": None, "path": []}
    )

    if start == end:
        return json.dumps(
            {
                "total_duration_sec": 0,
                "arrival_time": _format_time(start_time),
                "path": [],
            }
        )

    # Directional obstruction lookup: (edge_id, from, to) -> [(start, end, factor)]
    obstruction_map: dict[tuple, list[tuple[float, float, float]]] = {}
    last_obstruction_end = start_time
    for obs in payload.get("obstructions", []):
        key = (
            obs["edge_id"],
            tuple(obs["edge"]["from"]),
            tuple(obs["edge"]["to"]),
        )
        interval = (
            _parse_time(obs["start_time"]),
            _parse_time(obs["end_time"]),
            float(obs["speed_factor"]),
        )
        obstruction_map.setdefault(key, []).append(interval)
        last_obstruction_end = max(last_obstruction_end, interval[1])

    # Bidirectional edges -> two directed arcs, each with its own obstructions.
    adjacency: dict[tuple, list[tuple[tuple, str, float, list]]] = {}
    for edge in payload["edges"]:
        edge_id = edge["edge_id"]
        n1 = tuple(edge["node1"])
        n2 = tuple(edge["node2"])
        duration = float(edge["base_duration_sec"])
        for a, b in ((n1, n2), (n2, n1)):
            adjacency.setdefault(a, []).append(
                (b, edge_id, duration, obstruction_map.get((edge_id, a, b), []))
            )

    # Dijkstra over (node, time) states. States are deduped on (node, time);
    # once past every obstruction the network is static, so at most one state
    # per node is expanded beyond last_obstruction_end.
    states: list[tuple[tuple, float, int, str | None]] = [(start, start_time, -1, None)]
    heap: list[tuple[float, int]] = [(start_time, 0)]
    seen: set[tuple[tuple, float]] = {(start, round(start_time, 6))}
    settled_static: set[tuple] = set()

    pops = 0
    goal_state = -1
    while heap:
        pops += 1
        if pops > MAX_POPS:
            break
        t, state_id = heapq.heappop(heap)
        node = states[state_id][0]
        if node == end:
            goal_state = state_id
            break
        if t >= last_obstruction_end:
            if node in settled_static:
                continue
            settled_static.add(node)
        for neighbor, edge_id, duration, obstructions in adjacency.get(node, []):
            arrival = _traverse(obstructions, t, duration)
            if arrival is None:
                continue
            key = (neighbor, round(arrival, 6))
            if key in seen:
                continue
            seen.add(key)
            states.append((neighbor, arrival, state_id, edge_id))
            heapq.heappush(heap, (arrival, len(states) - 1))

    if goal_state < 0:
        return no_route

    path: list[str] = []
    cursor = goal_state
    while cursor >= 0:
        _, _, parent, edge_id = states[cursor]
        if edge_id is not None:
            path.append(edge_id)
        cursor = parent
    path.reverse()

    arrival_time = states[goal_state][1]
    total = arrival_time - start_time
    total_out = int(round(total)) if abs(total - round(total)) < 1e-6 else total
    return json.dumps(
        {
            "total_duration_sec": total_out,
            "arrival_time": _format_time(arrival_time),
            "path": path,
        }
    )
