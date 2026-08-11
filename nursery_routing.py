"""Stage 2, Part 2 - "Out after school".

The android moves one hop at a time and re-asks from wherever it now stands,
so every call answers the same question: given where I am, where do I step
next? The answer is always the first hop of the best whole route, recomputed
from the current node.

Cost model (from the brief):

    total cost = sum(edge weights) + sum(entry tolls)

Tolls are charged on *entry*, so the toll of the node you start from is never
paid and the destination's toll always is. Folding each node's toll into the
cost of entering it makes plain Dijkstra correct again - without that fold it
picks routes that win on edge weight and lose on total cost.
"""

from __future__ import annotations

import heapq
import math
import os
import re
import threading
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import httpx

# Where GET /graph?map_id=... lives. The maps are drawn on the challenge's
# side, so this points at their host, not ours; override without a redeploy.
GRAPH_API_BASE = os.getenv(
    "GRAPH_API_BASE", "https://tool-box-2591eaa24fa3.herokuapp.com"
).rstrip("/")
GRAPH_FETCH_TIMEOUT = float(os.getenv("GRAPH_FETCH_TIMEOUT", "4.0"))

_graph_cache: dict[str, tuple[dict[str, dict[str, float]], dict[str, float]]] = {}
_cache_lock = threading.Lock()


class GraphUnavailable(RuntimeError):
    """Raised when the map behind a map_id could not be read."""


# --------------------------------------------------------------------------
# Parsing the question
# --------------------------------------------------------------------------

# Node labels vary across the pool (A, N07, SITE_12, HUB-D), so the pattern
# stays permissive and the result is validated against the real adjacency.
_LABEL = r"[A-Za-z0-9_\-]+"

_FROM_TO_PATTERNS = (
    re.compile(rf"from\s+({_LABEL})\s+to\s+({_LABEL})", re.I),
    re.compile(rf"\b({_LABEL})\s*(?:->|→|=>)\s*({_LABEL})\b"),
    re.compile(rf"between\s+({_LABEL})\s+and\s+({_LABEL})", re.I),
)

_MAP_ID_PATTERN = re.compile(r"map[_\s-]?id\s*[:=]?\s*([A-Za-z0-9\-]+)", re.I)
_HOPS_PATTERN = re.compile(
    r"(\d+)\s*(?:hops?|steps?|moves?|edges?)\s*(?:left|remaining)?"
    r"|(?:hops?|steps?|moves?|edges?)\s*(?:left|remaining)\s*[:=]?\s*(\d+)",
    re.I,
)


def parse_question(text: str) -> dict[str, Any]:
    """Pull source, destination, map_id and any hop allowance out of the text."""
    parsed: dict[str, Any] = {}
    if not text:
        return parsed

    for pattern in _FROM_TO_PATTERNS:
        match = pattern.search(text)
        if match:
            parsed["source"], parsed["destination"] = match.group(1), match.group(2)
            break

    map_match = _MAP_ID_PATTERN.search(text)
    if map_match:
        parsed["map_id"] = map_match.group(1)

    hop_match = _HOPS_PATTERN.search(text)
    if hop_match:
        parsed["hops_left"] = int(hop_match.group(1) or hop_match.group(2))

    return parsed


# --------------------------------------------------------------------------
# Reading the map
# --------------------------------------------------------------------------


def normalise_graph(payload: Mapping[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Coerce a /graph envelope into (adjacency, tolls) with float weights.

    Every node named anywhere - as a source, as a neighbour, or in tolls - gets
    an adjacency entry, so a dead-end node is a node with no exits rather than
    a KeyError mid-search.
    """
    raw_adjacency = payload.get("adjacency") or payload.get("edges") or {}
    raw_tolls = payload.get("tolls") or {}

    adjacency: dict[str, dict[str, float]] = {}
    for node, neighbours in raw_adjacency.items():
        node = str(node)
        adjacency.setdefault(node, {})
        for neighbour, weight in (neighbours or {}).items():
            neighbour = str(neighbour)
            try:
                adjacency[node][neighbour] = float(weight)
            except (TypeError, ValueError):
                continue
            adjacency.setdefault(neighbour, {})

    tolls: dict[str, float] = {}
    for node, toll in raw_tolls.items():
        node = str(node)
        try:
            tolls[node] = float(toll)
        except (TypeError, ValueError):
            tolls[node] = 0.0
        adjacency.setdefault(node, {})

    return adjacency, tolls


def _candidate_graph_urls(map_id: str) -> list[str]:
    """Bases worth trying for GET /graph, most explicit first."""
    bases: list[str] = []
    if GRAPH_API_BASE:
        bases.append(GRAPH_API_BASE)
    seen: set[str] = set()
    urls: list[str] = []
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        urls.append(f"{base}/graph?map_id={map_id}")
    return urls


def fetch_graph(map_id: str, base_url: str | None = None) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Read a map by id, caching it for the rest of the process.

    The same map_id is asked about once per hop for a whole journey, so caching
    turns a dozen round trips into one and keeps every later hop well clear of
    the 10s ceiling.
    """
    if not map_id:
        raise GraphUnavailable("no map_id supplied")

    with _cache_lock:
        cached = _graph_cache.get(map_id)
    if cached is not None:
        return cached

    urls: list[str] = []
    if base_url:
        parsed = urlparse(base_url)
        trimmed = base_url.rstrip("/")
        if parsed.path.endswith("/graph"):
            urls.append(f"{trimmed}?map_id={map_id}")
        else:
            urls.append(f"{trimmed}/graph?map_id={map_id}")
    urls.extend(_candidate_graph_urls(map_id))

    if not urls:
        raise GraphUnavailable(
            "no graph endpoint configured; set GRAPH_API_BASE to the host serving /graph"
        )

    errors: list[str] = []
    for url in urls:
        try:
            response = httpx.get(url, timeout=GRAPH_FETCH_TIMEOUT)
            response.raise_for_status()
            adjacency, tolls = normalise_graph(response.json())
            if adjacency:
                with _cache_lock:
                    _graph_cache[map_id] = (adjacency, tolls)
                return adjacency, tolls
            errors.append(f"{url}: empty adjacency")
        except Exception as exc:  # noqa: BLE001 - any failure just tries the next base
            errors.append(f"{url}: {type(exc).__name__}")

    raise GraphUnavailable("; ".join(errors) or "graph fetch failed")


def cache_graph(map_id: str, adjacency: Mapping[str, Mapping[str, float]], tolls: Mapping[str, float]) -> None:
    """Seed the cache from a map handed to us inline."""
    if not map_id:
        return
    with _cache_lock:
        _graph_cache[map_id] = (
            {str(k): {str(n): float(w) for n, w in (v or {}).items()} for k, v in adjacency.items()},
            {str(k): float(v) for k, v in tolls.items()},
        )


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def _entry_cost(tolls: Mapping[str, float], node: str) -> float:
    return float(tolls.get(node, 0.0))


def resolve_label(adjacency: Mapping[str, Mapping[str, float]], label: str) -> str:
    """Match a label from the question to the one the map actually uses.

    The brief says not to assume anything about node labels, so nothing here
    normalises case - upper-casing "hub-a" would invent a node the map has
    never heard of and strand the journey. This only falls back to a
    case-insensitive match when the exact label is absent.
    """
    if not label or label in adjacency:
        return label
    folded = label.casefold()
    for node in adjacency:
        if node.casefold() == folded:
            return node
    # Labels sometimes travel with punctuation differences (HUB_D vs HUB-D).
    squashed = re.sub(r"[^a-z0-9]", "", folded)
    for node in adjacency:
        if re.sub(r"[^a-z0-9]", "", node.casefold()) == squashed:
            return node
    return label


def cheapest_route(
    adjacency: Mapping[str, Mapping[str, float]],
    tolls: Mapping[str, float],
    source: str,
    destination: str,
    blocked: Iterable[str] = (),
) -> tuple[list[str], float]:
    """Dijkstra where entering a node costs its edge weight plus its toll.

    Returns (path including both ends, total cost). The source's own toll is
    excluded - the android never enters the node it starts on.
    """
    blocked_set = {b for b in blocked if b != destination}
    if source == destination:
        return [source], 0.0

    best: dict[str, float] = {source: 0.0}
    parent: dict[str, str] = {}
    queue: list[tuple[float, str]] = [(0.0, source)]
    settled: set[str] = set()

    while queue:
        cost, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        if node == destination:
            break
        for neighbour, weight in (adjacency.get(node) or {}).items():
            if neighbour in settled or neighbour in blocked_set:
                continue
            candidate = cost + float(weight) + _entry_cost(tolls, neighbour)
            if candidate < best.get(neighbour, math.inf):
                best[neighbour] = candidate
                parent[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))

    if destination not in best:
        return [], math.inf

    path = [destination]
    while path[-1] != source:
        path.append(parent[path[-1]])
    path.reverse()
    return path, best[destination]


def cheapest_route_within_hops(
    adjacency: Mapping[str, Mapping[str, float]],
    tolls: Mapping[str, float],
    source: str,
    destination: str,
    max_hops: int,
    blocked: Iterable[str] = (),
) -> tuple[list[str], float]:
    """Cheapest route using at most `max_hops` edges.

    The allowance is set so the cheapest route does not fit, so this is a
    genuinely different question from Dijkstra: minimise cost *subject to* an
    edge count. Layered DP over (hops used, node) - dp[k][v] is the cheapest
    way to stand on v having spent k edges.
    """
    if source == destination:
        return [source], 0.0
    if max_hops <= 0:
        return [], math.inf

    blocked_set = {b for b in blocked if b != destination}
    if source not in adjacency:
        return [], math.inf

    dp: list[dict[str, float]] = [{} for _ in range(max_hops + 1)]
    parent: list[dict[str, str]] = [{} for _ in range(max_hops + 1)]
    dp[0][source] = 0.0

    for k in range(1, max_hops + 1):
        for node, cost in dp[k - 1].items():
            for neighbour, weight in (adjacency.get(node) or {}).items():
                if neighbour in blocked_set:
                    continue
                candidate = cost + float(weight) + _entry_cost(tolls, neighbour)
                if candidate < dp[k].get(neighbour, math.inf):
                    dp[k][neighbour] = candidate
                    parent[k][neighbour] = node

    best_k, best_cost = -1, math.inf
    for k in range(1, max_hops + 1):
        cost = dp[k].get(destination, math.inf)
        if cost < best_cost:
            best_k, best_cost = k, cost

    if best_k < 0:
        return [], math.inf

    path = [destination]
    node, k = destination, best_k
    while k > 0:
        node = parent[k][node]
        path.append(node)
        k -= 1
    path.reverse()
    return path, best_cost


def next_hop(
    adjacency: Mapping[str, Mapping[str, float]],
    tolls: Mapping[str, float],
    source: str,
    destination: str,
    hops_left: int | None = None,
    visited: Sequence[str] = (),
) -> str:
    """The single node label to step to next.

    Revisiting a node fails the journey outright, so anywhere already stood on
    is excluded from the search rather than merely discouraged. With positive
    costs an optimal route would not loop anyway; this makes that guarantee
    hold even when a hop allowance forces a route that is not the cheapest.
    """
    source = resolve_label(adjacency, source)
    destination = resolve_label(adjacency, destination)
    if source == destination:
        return source

    blocked = {resolve_label(adjacency, v) for v in visited}
    blocked.discard(source)

    if hops_left is not None and hops_left > 0:
        path, cost = cheapest_route_within_hops(
            adjacency, tolls, source, destination, hops_left, blocked
        )
        if len(path) >= 2 and cost < math.inf:
            return path[1]

    path, cost = cheapest_route(adjacency, tolls, source, destination, blocked)
    if len(path) >= 2 and cost < math.inf:
        return path[1]

    # Unreachable under the constraints. Any answer scores zero, but a real
    # neighbour keeps the failure to this journey instead of erroring out;
    # prefer the one that gets closest to the destination.
    neighbours = adjacency.get(source) or {}
    if not neighbours:
        return destination

    def remaining(candidate: str) -> tuple[float, float]:
        _, onward = cheapest_route(adjacency, tolls, candidate, destination, blocked)
        step = float(neighbours[candidate]) + _entry_cost(tolls, candidate)
        return (onward, step)

    unvisited = [n for n in neighbours if n not in blocked] or list(neighbours)
    return min(unvisited, key=remaining)


def route_cost(
    adjacency: Mapping[str, Mapping[str, float]],
    tolls: Mapping[str, float],
    path: Sequence[str],
) -> float:
    """Total cost of walking `path`, for checking our own arithmetic."""
    total = 0.0
    for previous, node in zip(path, path[1:]):
        weight = (adjacency.get(previous) or {}).get(node)
        if weight is None:
            return math.inf
        total += float(weight) + _entry_cost(tolls, node)
    return total
