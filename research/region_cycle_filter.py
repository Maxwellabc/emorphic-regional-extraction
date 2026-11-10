"""Exact root-reachable cycle checks after freezing a decision region's outside.

Compilation contracts frozen paths to region boundaries. Querying an assignment
does not construct a full selection, traverse frozen paths, or consume randomness.
The caller still uses evaluate for metrics and signatures of feasible candidates.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from research.egraph_lab import EGraph


@dataclass(frozen=True)
class BoundarySummary:
    """First region boundaries and whether a wholly frozen cycle is reachable."""

    targets: tuple[str, ...]
    reaches_frozen_cycle: bool


class FrozenRegionCycleFilter:
    """Immutable snapshot of root reachability and cycles for one fixed region.

    The complete input selection and each query's regional overrides must contain
    valid class members. Cyclic complete selections are supported: only cycles
    reachable from a root under the queried assignment cause rejection.
    """

    def __init__(self, graph: EGraph, selection: Mapping[str, str], region: Sequence[str]):
        self.region = tuple(region)
        self._region_set = frozenset(self.region)
        if len(self.region) != len(self._region_set) or not self._region_set <= graph.classes.keys():
            raise ValueError("region must contain distinct existing eclasses")
        if selection.keys() != graph.classes.keys():
            raise ValueError("a full selection with exactly one choice per eclass is required")
        frozen_children = {}
        for class_id, node_id in selection.items():
            node = graph.nodes.get(node_id)
            if node is None or node.eclass != class_id:
                raise ValueError(f"invalid choice for eclass {class_id}")
            if class_id not in self._region_set:
                frozen_children[class_id] = node.children

        # -1 means that a wholly frozen cycle is reachable. Other values are
        # bitsets of first region boundaries reached through frozen paths.
        summaries = {c: 1 << i for i, c in enumerate(self.region)}

        def summarize(start):
            if start in summaries:
                return summaries[start]
            active = {start}
            stack = [[start, iter(frozen_children[start]), 0]]
            while stack:
                frame = stack[-1]
                try:
                    child = next(frame[1])
                except StopIteration:
                    class_id, _, targets = stack.pop()
                    summaries[class_id] = targets
                    active.remove(class_id)
                    if stack:
                        stack[-1][2] |= targets
                    continue
                targets = summaries.get(child)
                if targets == -1 or child in active:
                    # Every active frame is an ancestor of this frozen cycle.
                    # Its remaining outgoing edges cannot restore feasibility.
                    for class_id, _, _ in stack:
                        summaries[class_id] = -1
                    return -1
                if targets is not None:
                    frame[2] |= targets
                else:
                    active.add(child)
                    stack.append([child, iter(frozen_children[child]), 0])
            return summaries[start]

        roots = 0
        for root in graph.roots:
            targets = summarize(root)
            if targets == -1:
                roots = -1
                break
            roots |= targets
        edges = []
        for class_id in self.region:
            alternatives = {}
            for node_id in graph.classes[class_id]:
                targets = 0
                for child in graph.nodes[node_id].children:
                    reached = summarize(child)
                    if reached == -1:
                        targets = -1
                        break
                    targets |= reached
                alternatives[node_id] = targets
            edges.append(alternatives)
        self._roots = roots
        self._edges = tuple(edges)
        self.frozen_classes_summarized = len(summaries) - len(self.region)
        self._frozen_children = frozen_children
        self._root_classes = graph.roots
        self._alternative_children = {n: graph.nodes[n].children for c in self.region for n in graph.classes[c]}
        self._node_edges = {n: targets for alternatives in edges for n, targets in alternatives.items()}
        self._descriptions: dict[str | None, BoundarySummary] = {}

    def _describe(self, key, targets, starts):
        if key in self._descriptions:
            return self._descriptions[key]
        if targets == -1:
            # Cycle rejection needs no boundary set, so compilation short-circuits
            # those paths. Recover their complete first-hit sets only on request.
            reached, visited = set(), set()
            stack = list(starts)
            while stack:
                class_id = stack.pop()
                if class_id in self._region_set:
                    reached.add(class_id)
                elif class_id not in visited:
                    visited.add(class_id)
                    stack.extend(self._frozen_children[class_id])
            boundary_classes = tuple(c for c in self.region if c in reached)
        else:
            boundary_classes = tuple(c for i, c in enumerate(self.region) if targets & (1 << i))
        result = BoundarySummary(boundary_classes, targets == -1)
        self._descriptions[key] = result
        return result

    @property
    def root_summary(self) -> BoundarySummary:
        """First region boundaries reached from any root, stopping at boundaries."""
        return self._describe(None, self._roots, self._root_classes)

    def alternative_summary(self, node_id: str) -> BoundarySummary:
        """First boundaries reached after taking one region alternative's edges.

        Targets remain complete even if a frozen cycle is also reachable. This
        diagnostic recovery is lazy and does not affect normal cycle queries.
        """
        if node_id not in self._node_edges:
            raise ValueError("alternative must belong to a class in the region")
        return self._describe(node_id, self._node_edges[node_id], self._alternative_children[node_id])

    def has_reachable_cycle(self, overrides: Mapping[str, str]) -> bool:
        """Exactly match evaluate's root-DAG cycle condition for this assignment.

        Every region class must be present, even if currently unreachable. This
        guards against silently checking a partial or mismatched region.
        """
        if overrides.keys() != self._region_set:
            raise ValueError("overrides must contain exactly the frozen filter's region")
        edges = []
        for class_id, alternatives in zip(self.region, self._edges):
            node_id = overrides[class_id]
            if node_id not in alternatives:
                raise ValueError(f"invalid regional choice for eclass {class_id}")
            edges.append(alternatives[node_id])
        if self._roots == -1:
            return True
        active = finished = 0
        roots = self._roots
        while roots:
            root = roots & -roots
            roots ^= root
            stack = [(root, False)]
            while stack:
                bit, exiting = stack.pop()
                if exiting:
                    active ^= bit
                    finished |= bit
                    continue
                if finished & bit:
                    continue
                if active & bit:
                    return True
                targets = edges[bit.bit_length() - 1]
                if targets == -1:
                    return True
                active |= bit
                stack.append((bit, True))
                while targets:
                    child = targets & -targets
                    targets ^= child
                    stack.append((child, False))
        return False
