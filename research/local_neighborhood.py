"""Bounded one- and two-class structural extraction neighborhoods.

This is a research control, with no mapped-cost or novelty claim. Candidates
may worsen both additive node cost and depth; callers choose how to rank them.
Only root-reachable DAG validity is checked, using the laboratory evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

from research.egraph_lab import EGraph, Metrics, evaluate


@dataclass(frozen=True)
class IncidenceIndex:
    """Immutable incidence snapshot, without materializing class pairs."""

    children: Mapping[str, tuple[str, ...]]
    parents: Mapping[str, tuple[str, ...]]
    decision_classes: tuple[str, ...]
    decision_set: frozenset[str]


@dataclass(frozen=True)
class Candidate:
    choices: Mapping[str, str]
    metrics: Metrics
    changed_classes: tuple[str, ...]


@dataclass(frozen=True)
class NeighborhoodStats:
    combinations: int
    attempted: int
    invalid: int
    duplicates: int
    returned: int
    exact: bool


@dataclass(frozen=True)
class NeighborhoodResult:
    region: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    stats: NeighborhoodStats


def build_incidence(graph: EGraph) -> IncidenceIndex:
    """Index children and parents from *all* alternatives, including unchosen ones.

    Children in EGraph are already class IDs. Storage is proportional to the
    distinct class-child incidences, even when many parents share one child.
    The graph must remain unchanged while this snapshot is reused.
    """
    children: dict[str, tuple[str, ...]] = {}
    parents: dict[str, set[str]] = {c: set() for c in graph.classes}
    for class_id, alternatives in graph.classes.items():
        potential = {child for n in alternatives for child in graph.nodes[n].children}
        children[class_id] = tuple(sorted(potential))
        for child in potential:
            parents[child].add(class_id)
    decisions = tuple(sorted(c for c, ns in graph.classes.items() if len(ns) >= 2))
    return IncidenceIndex(
        MappingProxyType(children),
        MappingProxyType({c: tuple(sorted(ps)) for c, ps in parents.items()}),
        decisions, frozenset(decisions),
    )


def _validate_full_selection(graph: EGraph, selection: Mapping[str, str]) -> Metrics:
    if selection.keys() != graph.classes.keys():
        raise ValueError("a full selection with exactly one choice per eclass is required")
    for class_id, node_id in selection.items():
        node = graph.nodes.get(node_id)
        if node is None or node.eclass != class_id:
            raise ValueError(f"invalid choice for eclass {class_id}")
    return evaluate(graph, selection)


def weighted_depth(graph: EGraph, selection: Mapping[str, str]) -> float:
    """Maximum selected root-to-leaf sum of node costs, including leaf costs.

    This structural proxy differs from unit-edge Metrics.depth and is not
    mapped delay. Validate the full selection, then evaluate the reachable
    DAG iteratively so shared nodes are computed once and deep paths work.
    """
    _validate_full_selection(graph, selection)
    depths: dict[str, float] = {}
    for root in graph.roots:
        stack = [(root, False)]
        while stack:
            class_id, exiting = stack.pop()
            if class_id in depths:
                continue
            node = graph.nodes[selection[class_id]]
            if exiting:
                depths[class_id] = node.cost + max((depths[c] for c in node.children), default=0.0)
            else:
                stack.append((class_id, True))
                stack.extend((c, False) for c in reversed(node.children))
    return max(depths[r] for r in graph.roots)


def propose_region(
    graph: EGraph,
    selection: Mapping[str, str],
    *,
    rng: random.Random,
    policy: str = "linked",
    index: IncidenceIndex | None = None,
) -> tuple[str, ...]:
    """Choose a reachable decision anchor and at most one decision partner.

    ``random`` samples a partner from all other decision classes. ``linked``
    samples from potential parents, children, and classes sharing a child,
    considering every alternative. Thus a currently unreachable partner can
    become active when the anchor switches. A missing partner leaves a
    singleton region; no reachable decision class gives an empty region.
    No quadratic collection of all linked pairs is constructed.
    """
    if policy not in {"random", "linked"}:
        raise ValueError("policy must be random or linked")
    metrics = _validate_full_selection(graph, selection)
    if index is None:
        index = build_incidence(graph)
    anchors = tuple(c for c in metrics.reachable if c in index.decision_set)
    if not anchors:
        return ()
    anchor = rng.choice(anchors)
    if policy == "random":
        partners = tuple(c for c in index.decision_classes if c != anchor)
    else:
        neighbors = set(index.children[anchor]) | set(index.parents[anchor])
        for child in index.children[anchor]:
            neighbors.update(index.parents[child])
        neighbors.discard(anchor)
        partners = tuple(sorted(neighbors & index.decision_set))
    return (anchor, rng.choice(partners)) if partners else (anchor,)


def _combination_ranks(total: int, limit: int, rng: random.Random) -> Iterator[int]:
    if total <= limit:
        yield from range(total)
        return
    # Floyd sampling works with arbitrarily large Python integers. It keeps
    # O(limit) ranks, never a Cartesian product or a full selection queue.
    selected: set[int] = set()
    for upper in range(total - limit, total):
        draw = rng.randrange(upper + 1)
        rank = upper if draw in selected else draw
        selected.add(rank)
        yield rank


def enumerate_region(
    graph: EGraph,
    selection: Mapping[str, str],
    region: Sequence[str],
    *,
    max_combinations: int = 256,
    rng: random.Random | None = None,
) -> NeighborhoodResult:
    """Try bounded combinations while freezing every choice outside the region.

    ``region`` contains one or two distinct classes. The product includes the
    original choices, which consume attempts like all other combinations.
    If the product exceeds the positive cap, sample unique combination ranks
    using the supplied RNG (or a deterministic seed of zero by default).

    Initial validation is excluded from ``attempted``. Every tried combination
    is either invalid, a duplicate reachable signature, or a returned candidate:
    attempted == invalid + duplicates + returned. The original reachable
    signature counts as a duplicate, as do repeated signatures caused by
    unreachable region choices. ``exact`` means every combination was tried;
    it says nothing about mapped optimality or a global extraction optimum.
    """
    if isinstance(max_combinations, bool) or not isinstance(max_combinations, int) or max_combinations < 1:
        raise ValueError("max_combinations must be a positive integer")
    region = tuple(region)
    if len(region) not in (1, 2) or len(set(region)) != len(region):
        raise ValueError("region must contain one or two distinct eclasses")
    if any(c not in graph.classes for c in region):
        raise ValueError("unknown eclass in region")
    original = _validate_full_selection(graph, selection)
    alternatives = tuple(graph.classes[c] for c in region)
    total = math.prod(len(ns) for ns in alternatives)
    seen = {original.signature}
    candidates = []
    attempted = invalid = duplicates = 0
    for rank in _combination_ranks(total, max_combinations, rng if rng is not None else random.Random(0)):
        choices = dict(selection)
        for class_id, members in reversed(tuple(zip(region, alternatives))):
            rank, member = divmod(rank, len(members))
            choices[class_id] = members[member]
        attempted += 1
        try:
            metrics = evaluate(graph, choices)
        except ValueError:
            invalid += 1
            continue
        if metrics.signature in seen:
            duplicates += 1
            continue
        seen.add(metrics.signature)
        candidates.append(Candidate(
            MappingProxyType(choices), metrics,
            tuple(c for c in region if choices[c] != selection[c]),
        ))
    return NeighborhoodResult(
        region, tuple(candidates),
        NeighborhoodStats(total, attempted, invalid, duplicates, len(candidates), total <= max_combinations),
    )
