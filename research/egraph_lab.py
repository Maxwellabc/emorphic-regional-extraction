"""Small, reproducible DAG-extraction laboratory.

The additive node cost and unit-edge depth here are structural proxies.  They
are not mapped ASAP7 area, ABC delay, logical equivalence checks, or a
reproduction of E-morphic's simulated annealing.  E-class equivalence is an
input assumption; only extraction structure is validated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import json
import math
from pathlib import Path
import time
from typing import Mapping


@dataclass(frozen=True)
class Node:
    id: str
    eclass: str
    op: str
    children: tuple[str, ...]  # E-class ids, resolved from serialized node ids.
    cost: float


@dataclass(frozen=True)
class EGraph:
    nodes: dict[str, Node]
    classes: dict[str, tuple[str, ...]]
    roots: tuple[str, ...]


@dataclass(frozen=True)
class Metrics:
    area: float
    depth: int
    reachable: tuple[str, ...]
    critical: tuple[str, ...]
    signature: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SearchResult:
    selection: dict[str, str]
    metrics: Metrics
    evaluations: int
    accepted: int
    elapsed_seconds: float
    exhausted: bool


@dataclass(frozen=True)
class MilpResult:
    selection: dict[str, str] | None
    metrics: Metrics | None
    status: int
    message: str
    lower_bound: float | None
    gap: float | None
    elapsed_seconds: float

    @property
    def proven_optimal(self) -> bool:
        """Optimality concerns additive area only, not the depth tie-break."""
        return self.status == 0 and self.metrics is not None


def graph_from_dict(data: Mapping) -> EGraph:
    """Read the egraph-serialize schema; reject dangling child node ids."""
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, dict) or not raw_nodes:
        raise ValueError("nodes must be a nonempty object")
    raw_roots = data.get("root_eclasses", data.get("roots"))
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ValueError("nonempty root_eclasses (or roots) is required")
    classes: dict[str, list[str]] = {}
    nodes = {}
    for node_id, raw in raw_nodes.items():
        if not isinstance(node_id, str) or not isinstance(raw, dict):
            raise ValueError("node ids must be strings and nodes must be objects")
        class_id = raw.get("eclass")
        if not isinstance(class_id, str):
            raise ValueError(f"invalid eclass for node {node_id}")
        cost = float(raw.get("cost", 1.0))
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"nonnegative finite cost required: {node_id}")
        children = raw.get("children", [])
        if not isinstance(children, list):
            raise ValueError(f"children must be a list: {node_id}")
        for child in children:
            if child not in raw_nodes:
                raise ValueError(f"dangling child node {child!r} in {node_id}")
        nodes[node_id] = Node(
            node_id, class_id, str(raw.get("op", "")),
            tuple(raw_nodes[c]["eclass"] for c in children), cost,
        )
        classes.setdefault(class_id, []).append(node_id)
    roots = tuple(dict.fromkeys(raw_roots))
    if any(not isinstance(c, str) or c not in classes for c in roots):
        raise ValueError("root eclass does not exist")
    return EGraph(nodes, {c: tuple(sorted(ns)) for c, ns in sorted(classes.items())}, roots)


def load_egraph(path: str | Path) -> EGraph:
    with Path(path).open(encoding="utf-8") as stream:
        return graph_from_dict(json.load(stream))


def tree_extract(graph: EGraph) -> dict[str, str]:
    """Minimum additive tree cost, with depth and node-id tie breaking.

    A depth tie-break prevents zero-cost cycles from replacing finite DAGs.
    Unproductive e-classes may remain absent; all roots must be productive.
    """
    best: dict[str, tuple[float, int, str]] = {}
    # A cheapest finite tree need not repeat an e-class along a path when
    # costs are nonnegative.  Hence |classes| Bellman-style sweeps suffice.
    for _ in range(len(graph.classes)):
        changed = False
        for class_id, node_ids in graph.classes.items():
            for node_id in node_ids:
                node = graph.nodes[node_id]
                if any(c not in best for c in node.children):
                    continue
                candidate = (
                    node.cost + sum(best[c][0] for c in node.children),
                    0 if not node.children else 1 + max(best[c][1] for c in node.children),
                    node_id,
                )
                if class_id not in best or candidate < best[class_id]:
                    best[class_id] = candidate
                    changed = True
        if not changed:
            break
    selection = {c: item[2] for c, item in best.items()}
    evaluate(graph, selection)
    return selection


def evaluate(graph: EGraph, selection: Mapping[str, str]) -> Metrics:
    """Validate root-reachable DAG and count each selected e-class once.

    Unreachable selections do not affect cost, depth, or signature.  DFS is
    iterative, so long paths are not constrained by Python recursion depth.
    """
    state: dict[str, int] = {}
    depths: dict[str, int] = {}
    critical_child: dict[str, str] = {}
    postorder: list[str] = []
    for root in graph.roots:
        if state.get(root) == 2:
            continue
        stack = [(root, False)]
        while stack:
            class_id, exiting = stack.pop()
            if exiting:
                node = graph.nodes[selection[class_id]]
                if node.children:
                    child = min(node.children, key=lambda c: (-depths[c], c))
                    critical_child[class_id] = child
                    depths[class_id] = 1 + depths[child]
                else:
                    depths[class_id] = 0
                state[class_id] = 2
                postorder.append(class_id)
                continue
            if state.get(class_id) == 2:
                continue
            if state.get(class_id) == 1:
                raise ValueError(f"cycle through eclass {class_id}")
            node_id = selection.get(class_id)
            if node_id not in graph.nodes:
                raise ValueError(f"missing/unknown choice for eclass {class_id}")
            node = graph.nodes[node_id]
            if node.eclass != class_id:
                raise ValueError(f"node {node_id} is not in eclass {class_id}")
            state[class_id] = 1
            stack.append((class_id, True))
            for child in reversed(node.children):
                stack.append((child, False))
    critical_root = min(graph.roots, key=lambda c: (-depths[c], c))
    critical = [critical_root]
    while critical[-1] in critical_child:
        critical.append(critical_child[critical[-1]])
    reachable = tuple(sorted(postorder))
    return Metrics(
        math.fsum(graph.nodes[selection[c]].cost for c in reachable),
        max(depths[r] for r in graph.roots), reachable, tuple(critical),
        tuple((c, selection[c]) for c in reachable),
    )


def _score(metrics: Metrics) -> tuple[float, int]:
    return metrics.area, metrics.depth


def _groups(graph: EGraph, metrics: Metrics) -> list[tuple[str, str]]:
    """Potential sharing and adjacent critical-path class pairs.

    Potential children include *unchosen* alternatives: otherwise the
    coordination barrier in the two-switch sharing example is invisible.
    """
    sharing: dict[str, list[str]] = {}
    active = set(metrics.reachable)
    for class_id in metrics.reachable:
        if len(graph.classes[class_id]) < 2:
            continue
        potential = {c for n in graph.classes[class_id] for c in graph.nodes[n].children}
        for child in sorted(potential):
            sharing.setdefault(child, []).append(class_id)
    pairs = {tuple(sorted(pair)) for owners in sharing.values() for pair in combinations(owners, 2)}
    pairs.update(
        tuple(sorted((a, b)))
        for a, b in zip(metrics.critical, metrics.critical[1:])
        if a in active and b in active
        and len(graph.classes[a]) > 1 and len(graph.classes[b]) > 1
    )
    return sorted(pairs)


def local_search(
    graph: EGraph,
    selection: Mapping[str, str],
    *,
    mode: str = "single",
    max_evaluations: int = 1000,
    max_group_size: int = 2,
) -> SearchResult:
    """Deterministic first-improvement descent under a candidate budget.

    Every candidate validation, including invalid candidates, consumes one
    evaluation. Initial validation is excluded. Group mode first tries a
    full single-point pass, then atomically changes two related classes.
    These are laboratory controls, not E-morphic's released SA baseline.
    """
    if mode not in {"single", "group"}:
        raise ValueError("mode must be single or group")
    if max_evaluations < 0 or max_group_size != 2:
        raise ValueError("budget must be nonnegative; only size-2 groups supported")
    started = time.perf_counter()
    current = dict(selection)
    metrics = evaluate(graph, current)
    evaluations = accepted = 0
    exhausted = False
    while evaluations < max_evaluations:
        improved = False
        candidates = (((c, n),) for c in metrics.reachable for n in graph.classes[c] if n != current[c])
        if mode == "group":
            def joint_candidates():
                yield from candidates
                for a, b in _groups(graph, metrics):
                    for na in graph.classes[a]:
                        if na == current[a]:
                            continue
                        for nb in graph.classes[b]:
                            if nb != current[b]:
                                yield ((a, na), (b, nb))
            proposals = joint_candidates()
        else:
            proposals = candidates
        for changes in proposals:
            if evaluations >= max_evaluations:
                break
            candidate = current | dict(changes)
            evaluations += 1
            try:
                result = evaluate(graph, candidate)
            except ValueError:
                continue
            if _score(result) < _score(metrics):
                current, metrics = candidate, result
                accepted += 1
                improved = True
                break
        if not improved:
            # A budget stop is not a certificate of local optimality.
            exhausted = evaluations < max_evaluations
            break
    return SearchResult(current, metrics, evaluations, accepted, time.perf_counter() - started, exhausted)


def exact_extract(graph: EGraph, *, max_combinations: int = 1_000_000) -> SearchResult:
    """Exhaustive small-instance DAG oracle with a preflight size limit."""
    count = math.prod(len(ns) for ns in graph.classes.values())
    if max_combinations < 1 or count > max_combinations:
        raise ValueError(f"exact oracle needs {count} combinations; limit is {max_combinations}")
    started = time.perf_counter()
    classes = tuple(graph.classes)
    best_selection = None
    best_metrics = None
    valid = 0
    for nodes in product(*(graph.classes[c] for c in classes)):
        selection = dict(zip(classes, nodes))
        try:
            metrics = evaluate(graph, selection)
        except ValueError:
            continue
        valid += 1
        if best_metrics is None or (_score(metrics), metrics.signature) < (_score(best_metrics), best_metrics.signature):
            best_selection, best_metrics = selection, metrics
    if best_selection is None or best_metrics is None:
        raise ValueError("no root-reachable acyclic extraction exists")
    return SearchResult(best_selection, best_metrics, count, valid, time.perf_counter() - started, True)


def milp_extract(graph: EGraph, *, time_limit: float = 30.0) -> MilpResult:
    """Optional HiGHS exact additive-DAG model, using installed SciPy.

    A binary variable selects an e-node; at most one node per e-class is
    selected, root classes have exactly one, and selected parents require
    one selected child implementation. Continuous class ranks, activated
    by each selected node, exclude cycles. Costs are charged once per
    selected e-node. All extra selected components can be removed because
    input costs are nonnegative. A time limit is never an optimality proof.

    SciPy API reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
    """
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("positive finite time_limit is required")
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    started = time.perf_counter()
    node_ids = tuple(sorted(graph.nodes))
    class_ids = tuple(graph.classes)
    node_index = {n: i for i, n in enumerate(node_ids)}
    rank_index = {c: len(node_ids) + i for i, c in enumerate(class_ids)}
    variable_count = len(node_ids) + len(class_ids)
    coefficients = np.zeros(variable_count)
    coefficients[:len(node_ids)] = [graph.nodes[n].cost for n in node_ids]
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:len(node_ids)] = 1
    upper = np.ones(variable_count)
    upper[len(node_ids):] = len(class_ids) - 1
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower_rows: list[float] = []
    upper_rows: list[float] = []

    def row(entries: list[tuple[int, float]], low: float, high: float):
        index = len(lower_rows)
        rows.extend([index] * len(entries))
        columns.extend(i for i, _ in entries)
        values.extend(v for _, v in entries)
        lower_rows.append(low)
        upper_rows.append(high)

    roots = set(graph.roots)
    for class_id, members in graph.classes.items():
        row([(node_index[n], 1.0) for n in members], 1.0 if class_id in roots else 0.0, 1.0)
    big_m = len(class_ids)
    for node_id, node in graph.nodes.items():
        for child in sorted(set(node.children)):
            row([(node_index[node_id], 1.0)] + [(node_index[n], -1.0) for n in graph.classes[child]], -np.inf, 0.0)
            # Duplicate columns (self edges) are intentionally summed by COO.
            row([(rank_index[node.eclass], 1.0), (rank_index[child], -1.0), (node_index[node_id], -big_m)], 1.0 - big_m, np.inf)
    matrix = coo_matrix((values, (rows, columns)), shape=(len(lower_rows), variable_count)).tocsc()
    result = milp(
        coefficients,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), upper),
        constraints=LinearConstraint(matrix, np.array(lower_rows), np.array(upper_rows)),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    selection = None
    metrics = None
    if result.x is not None:
        selection = {graph.nodes[n].eclass: n for n in node_ids if result.x[node_index[n]] > 0.5}
        metrics = evaluate(graph, selection)
        # Require an integral, structurally valid decoded incumbent; do not
        # report rounded fractional relaxations as actual extractions.
        if any(abs(result.x[i] - round(result.x[i])) > 1e-5 for i in range(len(node_ids))):
            raise ValueError("MILP returned a fractional incumbent")
        if result.status == 0 and not math.isclose(metrics.area, float(result.fun), rel_tol=1e-7, abs_tol=1e-6):
            raise ValueError("decoded DAG cost differs from optimal MILP objective")
    dual = getattr(result, "mip_dual_bound", None)
    gap = getattr(result, "mip_gap", None)
    return MilpResult(
        selection, metrics, int(result.status), str(result.message),
        None if dual is None else float(dual),
        None if gap is None else float(gap), time.perf_counter() - started,
    )
