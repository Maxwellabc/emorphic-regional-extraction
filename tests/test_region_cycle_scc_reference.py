"""Synthetic SCC/reference/evaluate agreement; no benchmark timing or tools."""
from itertools import combinations, product
import random
import unittest

from research.egraph_lab import EGraph, Node, evaluate
from research.region_cycle_filter import FrozenRegionCycleFilter
from research.region_cycle_scc_reference import FrozenRegionSCCCycleFilter, _strong_components


def graph_from_edges(edges, roots):
    nodes = {node_id: Node(node_id, cid, "synthetic", tuple(children), 1.0)
             for cid, alternatives in edges.items() for node_id, children in alternatives.items()}
    return EGraph(nodes, {cid: tuple(alternatives) for cid, alternatives in edges.items()}, tuple(roots))


def cyclic(graph, selection):
    try:
        evaluate(graph, selection)
    except ValueError as error:
        if "cycle" not in str(error):
            raise
        return True
    return False


class SCCReferenceTests(unittest.TestCase):
    def compare(self, graph, center, region):
        first = FrozenRegionCycleFilter(graph, center, region)
        scc = FrozenRegionSCCCycleFilter(graph, center, region)
        self.assertEqual(scc.root_summary, first.root_summary)
        for cid in region:
            for node_id in graph.classes[cid]:
                self.assertEqual(scc.alternative_summary(node_id), first.alternative_summary(node_id))
        count = 0
        for choices in product(*(graph.classes[cid] for cid in region)):
            overrides = dict(zip(region, choices))
            expected = cyclic(graph, center | overrides)
            self.assertEqual(scc.has_reachable_cycle(overrides), expected, (graph, center, region, overrides))
            self.assertEqual(first.has_reachable_cycle(overrides), expected)
            count += 1
        return count

    def test_exhaustive_three_class_graphs_centers_regions_and_assignments(self):
        classes = ("a", "b", "c")
        regions = [tuple(region) for size in range(4) for region in combinations(classes, size)]
        checked = 0
        for children in product((None, *classes), repeat=3):
            edges = {cid: {cid + "0": (), cid + "1": () if child is None else (child,)}
                     for cid, child in zip(classes, children)}
            for roots in (("a",), ("a", "c")):
                graph = graph_from_edges(edges, roots)
                for chosen in product(*(graph.classes[cid] for cid in classes)):
                    center = dict(zip(classes, chosen))
                    for region in regions:
                        checked += self.compare(graph, center, region)
        self.assertEqual(checked, 27648)

    def test_random_multifanout_graphs_all_regional_assignments(self):
        rng = random.Random(20261105)
        checked = 0
        for _ in range(180):
            classes = [f"c{i}" for i in range(rng.randrange(2, 13))]
            edges = {cid: {f"{cid}.{j}": tuple(rng.choice(classes) for _ in range(rng.randrange(5)))
                           for j in range(rng.randrange(1, 4))} for cid in classes}
            graph = graph_from_edges(edges, rng.sample(classes, rng.randrange(1, min(4, len(classes)) + 1)))
            center = {cid: rng.choice(graph.classes[cid]) for cid in classes}
            region = rng.sample(classes, rng.randrange(min(4, len(classes)) + 1))
            checked += self.compare(graph, center, region)
        self.assertGreater(checked, 700)

    def test_latent_frozen_scc_and_complete_boundaries(self):
        graph = graph_from_edges({
            "r": {"r0": (), "r1": ("x",)},
            "x": {"x0": ("y", "a")}, "y": {"y0": ("x", "tail")},
            "tail": {"t0": ("b",)}, "a": {"a0": ()}, "b": {"b0": ()}}, ("r",))
        center = {cid: members[0] for cid, members in graph.classes.items()}
        scc = FrozenRegionSCCCycleFilter(graph, center, ("b", "r", "a"))
        self.assertEqual(scc.frozen_cyclic_scc_count, 1)
        self.assertEqual(scc.alternative_summary("r1").targets, ("b", "a"))
        self.assertTrue(scc.alternative_summary("r1").reaches_frozen_cycle)
        self.assertFalse(scc.has_reachable_cycle({"b": "b0", "r": "r0", "a": "a0"}))
        self.assertTrue(scc.has_reachable_cycle({"b": "b0", "r": "r1", "a": "a0"}))
        self.compare(graph, center, ("b", "r", "a"))

    def test_joint_regional_cycle_through_shared_frozen_paths(self):
        graph = graph_from_edges({"r": {"r0": ("a",)},
            "a": {"a0": (), "a1": ("x", "x")}, "b": {"b0": (), "b1": ("y",)},
            "x": {"x0": ("b", "leaf")}, "y": {"y0": ("a", "leaf")}, "leaf": {"l0": ()}}, ("r",))
        center = {cid: members[0] for cid, members in graph.classes.items()}
        self.compare(graph, center, ("b", "a"))
        scc = FrozenRegionSCCCycleFilter(graph, center, ("a", "b"))
        self.assertTrue(scc.has_reachable_cycle({"a": "a1", "b": "b1"}))
        self.assertFalse(scc.has_reachable_cycle({"a": "a0", "b": "b1"}))

    def test_full_preprocessing_does_not_reject_unreachable_components(self):
        graph = graph_from_edges({"r": {"r0": (), "r1": ("used",)}, "used": {"u0": ()},
            "dead0": {"d0": ("dead1",)}, "dead1": {"d1": ("dead0",)}}, ("r",))
        center = {cid: members[0] for cid, members in graph.classes.items()}
        first = FrozenRegionCycleFilter(graph, center, ("r",))
        scc = FrozenRegionSCCCycleFilter(graph, center, ("r",))
        self.assertEqual(first.frozen_classes_summarized, 1)
        self.assertEqual(scc.frozen_classes_summarized, 3)
        self.assertEqual(scc.frozen_scc_count, 2)
        self.assertFalse(scc.has_reachable_cycle({"r": "r0"}))
        self.assertFalse(scc.has_reachable_cycle({"r": "r1"}))

    def test_long_chain_and_large_frozen_scc_are_iterative(self):
        count = 12000
        edges = {"r": {"r0": (), "r1": ("v0",)}}
        for i in range(count):
            edges[f"v{i}"] = {f"n{i}": (f"v{i+1}",) if i + 1 < count else ()}
        graph = graph_from_edges(edges, ("r",))
        center = {cid: members[0] for cid, members in graph.classes.items()}
        self.compare(graph, center, ("r",))
        self.assertEqual(FrozenRegionSCCCycleFilter(graph, center, ("r",)).frozen_scc_count, count)
        edges[f"v{count-1}"][f"n{count-1}"] = ("v0",)
        graph = graph_from_edges(edges, ("r",))
        self.compare(graph, center, ("r",))
        self.assertEqual(FrozenRegionSCCCycleFilter(graph, center, ("r",)).frozen_scc_count, 1)

    def test_scc_condensation_is_sink_first_and_deduplicates_edges(self):
        adjacency = [[1, 1, 3], [2], [1, 3], []]
        component, members = _strong_components(adjacency)
        self.assertEqual(component[1], component[2])
        self.assertEqual(len(members), 3)
        for vertex, children in enumerate(adjacency):
            for child in children:
                if component[vertex] != component[child]:
                    self.assertLess(component[child], component[vertex])

    def test_empty_region_multiple_roots_and_boundary_stopping(self):
        graph = graph_from_edges({"r": {"r0": ("a",), "r1": ()},
            "a": {"a0": ("a",), "a1": ()}, "s": {"s0": ()}}, ("r", "s"))
        center = {"r": "r0", "a": "a0", "s": "s0"}
        self.compare(graph, center, ())
        self.compare(graph, center, ("r", "a", "s"))
        scc = FrozenRegionSCCCycleFilter(graph, center, ("a",))
        self.assertFalse(scc.root_summary.reaches_frozen_cycle)
        self.assertEqual(scc.root_summary.targets, ("a",))
        self.assertFalse(scc.has_reachable_cycle({"a": "a1"}))

    def test_argument_validation_matches_original_even_with_unconditional_cycle(self):
        graph = graph_from_edges({"r": {"r0": ("r",)}, "a": {"a0": (), "a1": ("a",)}}, ("r",))
        center = {"r": "r0", "a": "a0"}
        for region, selection in ((("a", "a"), center), (("foreign",), center),
                (("a",), {"r": "r0"}), (("a",), center | {"extra": "a0"}),
                (("a",), center | {"a": "r0"}), (("a",), center | {"a": "missing"})):
            messages = []
            for cls in (FrozenRegionCycleFilter, FrozenRegionSCCCycleFilter):
                with self.assertRaises(ValueError) as error: cls(graph, selection, region)
                messages.append(str(error.exception))
            self.assertEqual(*messages)
        first, scc = (cls(graph, center, ("a",)) for cls in (FrozenRegionCycleFilter, FrozenRegionSCCCycleFilter))
        for override in ({}, {"a": "a0", "r": "r0"}, {"a": "r0"}, {"a": "missing"}):
            messages = []
            for check in (first, scc):
                with self.assertRaises(ValueError) as error: check.has_reachable_cycle(override)
                messages.append(str(error.exception))
            self.assertEqual(*messages)
        self.assertTrue(scc.has_reachable_cycle({"a": "a0"}))
        with self.assertRaisesRegex(ValueError, "alternative must belong"):
            scc.alternative_summary("r0")

    def test_snapshot_and_query_implementation_are_shared_without_mutation(self):
        graph = graph_from_edges({"r": {"r0": ("a",), "r1": ()}, "a": {"a0": (), "a1": ("a",)}}, ("r",))
        center, region = {"r": "r0", "a": "a0"}, ["a"]
        scc = FrozenRegionSCCCycleFilter(graph, center, region)
        center["r"] = "r1"
        region.clear()
        self.assertTrue(scc.has_reachable_cycle({"a": "a1"}))
        self.assertIs(FrozenRegionSCCCycleFilter.has_reachable_cycle, FrozenRegionCycleFilter.has_reachable_cycle)
        before = random.getstate()
        FrozenRegionSCCCycleFilter(graph, center, ("a",)).has_reachable_cycle({"a": "a0"})
        self.assertEqual(before, random.getstate())


if __name__ == "__main__":
    unittest.main()
