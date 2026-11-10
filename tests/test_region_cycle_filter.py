"""Compare frozen-boundary cycle compression with complete root-DAG checks."""
from itertools import product
import random
import unittest

from research.egraph_lab import EGraph, Node, evaluate
from research.region_cycle_filter import FrozenRegionCycleFilter


def make_graph(edges, roots=("r",)):
    nodes = {name: Node(name, class_id, "test", tuple(children), 1.0)
             for class_id, choices in edges.items() for name, children in choices.items()}
    return EGraph(nodes, {c: tuple(choices) for c, choices in edges.items()}, tuple(roots))


def cyclic(graph, choices):
    try:
        evaluate(graph, choices)
    except ValueError as error:
        if "cycle" not in str(error):
            raise
        return True
    return False


def reference_boundary_summary(graph, choices, region, starts):
    targets, visited, active = set(), set(), set()
    found_cycle = False

    def visit(class_id):
        nonlocal found_cycle
        if class_id in region:
            targets.add(class_id)
            return
        if class_id in active:
            found_cycle = True
            return
        if class_id in visited:
            return
        visited.add(class_id)
        active.add(class_id)
        for child in graph.nodes[choices[class_id]].children:
            visit(child)
        active.remove(class_id)

    for start in starts:
        visit(start)
    return targets, found_cycle


class RegionCycleFilterTests(unittest.TestCase):
    def test_root_can_drop_a_previously_reachable_cycle(self):
        graph = make_graph({"r": {"r0": ("a",), "r1": ()},
                            "a": {"a0": (), "a1": ("a",)}})
        choices = {"r": "r0", "a": "a0"}
        check = FrozenRegionCycleFilter(graph, choices, ("r", "a"))
        self.assertTrue(check.has_reachable_cycle({"r": "r0", "a": "a1"}))
        self.assertFalse(check.has_reachable_cycle({"r": "r1", "a": "a1"}))

    def test_unreachable_frozen_cycle_is_activated_only_by_selected_alternative(self):
        graph = make_graph({"r": {"r0": (), "r1": ("a",)},
                            "a": {"a0": ("b",)}, "b": {"b0": ("a",)}})
        check = FrozenRegionCycleFilter(graph, {"r": "r0", "a": "a0", "b": "b0"}, ("r",))
        self.assertFalse(check.has_reachable_cycle({"r": "r0"}))
        self.assertTrue(check.has_reachable_cycle({"r": "r1"}))

    def test_frozen_paths_form_joint_cycle_between_region_classes(self):
        graph = make_graph({"r": {"r0": ("a", "b")},
                            "a": {"a0": (), "a1": ("x",)},
                            "b": {"b0": (), "b1": ("y",)},
                            "x": {"x0": ("b",)}, "y": {"y0": ("a",)}})
        choices = {c: ns[0] for c, ns in graph.classes.items()}
        check = FrozenRegionCycleFilter(graph, choices, ("a", "b"))
        for a, b in product(graph.classes["a"], graph.classes["b"]):
            overrides = {"a": a, "b": b}
            self.assertEqual(check.has_reachable_cycle(overrides), cyclic(graph, choices | overrides))

    def test_multiple_roots_and_empty_region(self):
        graph = make_graph({"r": {"r0": ()}, "s": {"s0": ("a",)},
                            "a": {"a0": (), "a1": ("a",)}}, ("r", "s"))
        choices = {"r": "r0", "s": "s0", "a": "a0"}
        check = FrozenRegionCycleFilter(graph, choices, ("a",))
        self.assertTrue(check.has_reachable_cycle({"a": "a1"}))
        self.assertFalse(FrozenRegionCycleFilter(graph, choices, ()).has_reachable_cycle({}))

    def test_public_summaries_keep_all_boundaries_even_when_frozen_cycle_exists(self):
        graph = make_graph({"r": {"r0": ("a", "x")}, "a": {"a0": (), "a1": ("x",)},
                            "b": {"b0": ()}, "x": {"x0": ("x", "b")}})
        choices = {c: ns[0] for c, ns in graph.classes.items()}
        check = FrozenRegionCycleFilter(graph, choices, ("a", "b"))
        self.assertEqual(check.root_summary.targets, ("a", "b"))
        self.assertTrue(check.root_summary.reaches_frozen_cycle)
        self.assertEqual(check.alternative_summary("a1").targets, ("b",))
        self.assertTrue(check.alternative_summary("a1").reaches_frozen_cycle)
        self.assertFalse(check.alternative_summary("a0").reaches_frozen_cycle)
        self.assertEqual(check.alternative_summary("a0").targets, ())
        with self.assertRaises(ValueError):
            check.alternative_summary("x0")

    def test_shared_frozen_paths_and_long_chain_are_iterative(self):
        count = 3000
        edges = {"r": {"r0": ("n0", "n1")}}
        for i in range(count):
            edges[f"n{i}"] = {f"v{i}": (f"n{i + 1}",) if i + 1 < count else ("a",)}
        edges["a"] = {"a0": (), "a1": ("n0",)}
        graph = make_graph(edges)
        choices = {c: ns[0] for c, ns in graph.classes.items()}
        check = FrozenRegionCycleFilter(graph, choices, ("a",))
        self.assertFalse(check.has_reachable_cycle({"a": "a0"}))
        self.assertTrue(check.has_reachable_cycle({"a": "a1"}))

    def test_random_full_assignments_match_evaluate_including_unreachable_cycles(self):
        rng = random.Random(6354)
        comparisons = 0
        for _ in range(180):
            size = rng.randrange(2, 9)
            classes = [f"c{i}" for i in range(size)]
            edges = {c: {f"{c}.{j}": tuple(rng.choice(classes) for _ in range(rng.randrange(4)))
                         for j in range(3)} for c in classes}
            graph = make_graph(edges, rng.sample(classes, rng.randrange(1, min(size, 3) + 1)))
            choices = {c: rng.choice(graph.classes[c]) for c in classes}
            region = rng.sample(classes, rng.randrange(0, min(size, 4) + 1))
            check = FrozenRegionCycleFilter(graph, choices, region)
            summaries = [(check.root_summary, graph.roots)]
            summaries.extend((check.alternative_summary(n), graph.nodes[n].children)
                             for c in region for n in graph.classes[c])
            for summary, starts in summaries:
                targets, frozen_cycle = reference_boundary_summary(graph, choices, region, starts)
                self.assertEqual(set(summary.targets), targets)
                self.assertEqual(summary.reaches_frozen_cycle, frozen_cycle)
            for assigned in product(*(graph.classes[c] for c in region)):
                overrides = dict(zip(region, assigned))
                self.assertEqual(check.has_reachable_cycle(overrides), cyclic(graph, choices | overrides),
                                 (graph, choices, region, overrides))
                comparisons += 1
        self.assertGreater(comparisons, 2000)

    def test_invalid_input_rejected_and_snapshot_does_not_follow_selection_mutation(self):
        graph = make_graph({"r": {"r0": ("a",), "r1": ()}, "a": {"a0": (), "a1": ("a",)}})
        choices = {"r": "r0", "a": "a0"}
        check = FrozenRegionCycleFilter(graph, choices, ("a",))
        choices["r"] = "r1"
        self.assertTrue(check.has_reachable_cycle({"a": "a1"}))
        for region in (("a", "a"), ("foreign",)):
            with self.assertRaises(ValueError):
                FrozenRegionCycleFilter(graph, choices, region)
        for bad in ({"r": "r0"}, choices | {"extra": "a0"}, choices | {"a": "r0"}):
            with self.assertRaises(ValueError):
                FrozenRegionCycleFilter(graph, bad, ("a",))
        for bad in ({}, {"a": "a0", "r": "r0"}, {"a": "r0"}, {"a": "missing"}):
            with self.assertRaises(ValueError):
                check.has_reachable_cycle(bad)


if __name__ == "__main__":
    unittest.main()
