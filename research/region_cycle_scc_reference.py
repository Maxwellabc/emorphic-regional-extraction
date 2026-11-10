"""Conventional full frozen-SCC boundary summaries for regional cycle queries.

Unlike the demand-driven first-hit compiler, this reference processes every
outside class. Iterative Tarjan SCCs give a sink-first condensation order; one
bitset DP then propagates region boundaries and reachable frozen-cycle flags.
The query method is exactly the existing frozen filter's implementation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from research.egraph_lab import EGraph
from research.region_cycle_filter import BoundarySummary, FrozenRegionCycleFilter


def _strong_components(adjacency):
    """Iterative Tarjan; component IDs are in sink-first condensation order."""
    count = len(adjacency)
    discovery, low = [-1] * count, [0] * count
    on_stack, component = [False] * count, [-1] * count
    active, components = [], []
    next_index = 0
    for start in range(count):
        if discovery[start] != -1:
            continue
        discovery[start] = low[start] = next_index
        next_index += 1
        active.append(start)
        on_stack[start] = True
        frames = [(start, iter(adjacency[start]))]
        while frames:
            vertex, children = frames[-1]
            child = next(children, None)
            if child is not None:
                if discovery[child] == -1:
                    discovery[child] = low[child] = next_index
                    next_index += 1
                    active.append(child)
                    on_stack[child] = True
                    frames.append((child, iter(adjacency[child])))
                elif on_stack[child] and discovery[child] < low[vertex]:
                    low[vertex] = discovery[child]
                continue
            frames.pop()
            if frames:
                parent = frames[-1][0]
                if low[vertex] < low[parent]:
                    low[parent] = low[vertex]
            if low[vertex] == discovery[vertex]:
                members = []
                while True:
                    member = active.pop()
                    on_stack[member] = False
                    component[member] = len(components)
                    members.append(member)
                    if member == vertex:
                        break
                components.append(members)
    return component, components


class FrozenRegionSCCCycleFilter:
    """Exact frozen-SCC reference with the original constructor/query contract.

    Selection validation and regional override validation are unchanged. A
    frozen cycle is rejected only when reached under the queried assignment;
    discovering an unreachable cyclic SCC during compilation is not rejection.
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

        boundary = {class_id: 1 << i for i, class_id in enumerate(self.region)}
        frozen_index = {class_id: i for i, class_id in enumerate(frozen_children)}
        adjacency, direct_targets = [], []
        for children in frozen_children.values():
            successors, targets = [], 0
            for child in children:
                if child in boundary:
                    targets |= boundary[child]
                else:
                    successors.append(frozen_index[child])
            adjacency.append(successors)
            direct_targets.append(targets)
        component, members = _strong_components(adjacency)
        condensation = [set() for _ in members]
        targets = [0] * len(members)
        cycles = [False] * len(members)
        for cid, vertices in enumerate(members):
            cycles[cid] = len(vertices) > 1 or vertices[0] in adjacency[vertices[0]]
            for vertex in vertices:
                targets[cid] |= direct_targets[vertex]
                for child in adjacency[vertex]:
                    destination = component[child]
                    if destination != cid:
                        condensation[cid].add(destination)
        self.frozen_classes_summarized = len(frozen_children)
        self.frozen_scc_count = len(members)
        self.frozen_cyclic_scc_count = sum(cycles)
        self.condensation_edge_count = sum(map(len, condensation))
        # Tarjan emits an SCC after every distinct successor SCC. Thus the
        # successor summaries already exist, with no separate topological sort.
        for cid, successors in enumerate(condensation):
            for successor in successors:
                targets[cid] |= targets[successor]
                cycles[cid] |= cycles[successor]

        def summarize(starts):
            reached, cyclic = 0, False
            for start in starts:
                if start in boundary:
                    reached |= boundary[start]
                else:
                    cid = component[frozen_index[start]]
                    reached |= targets[cid]
                    cyclic |= cycles[cid]
            return reached, cyclic

        self._complete_roots = summarize(graph.roots)
        self._roots = -1 if self._complete_roots[1] else self._complete_roots[0]
        self._complete_node_edges = {}
        edges = []
        for class_id in self.region:
            alternatives = {}
            for node_id in graph.classes[class_id]:
                reached, cyclic = summarize(graph.nodes[node_id].children)
                self._complete_node_edges[node_id] = reached, cyclic
                alternatives[node_id] = -1 if cyclic else reached
            edges.append(alternatives)
        self._edges = tuple(edges)
        self._descriptions: dict[str | None, BoundarySummary] = {}

    def _describe(self, key, summary):
        if key not in self._descriptions:
            targets, cyclic = summary
            self._descriptions[key] = BoundarySummary(
                tuple(c for i, c in enumerate(self.region) if targets & (1 << i)), cyclic)
        return self._descriptions[key]

    @property
    def root_summary(self) -> BoundarySummary:
        return self._describe(None, self._complete_roots)

    def alternative_summary(self, node_id: str) -> BoundarySummary:
        if node_id not in self._complete_node_edges:
            raise ValueError("alternative must belong to a class in the region")
        return self._describe(node_id, self._complete_node_edges[node_id])

    # Keep the query's exact validation and root-reachable compressed DFS common
    # to both compilers. No traversal of frozen SCCs occurs during a query.
    has_reachable_cycle = FrozenRegionCycleFilter.has_reachable_cycle
