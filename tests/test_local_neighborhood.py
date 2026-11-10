"""Bounded structural neighborhoods; no mapped tools or timed workloads."""

from dataclasses import FrozenInstanceError
import random
import unittest
from unittest.mock import patch

from research.egraph_lab import EGraph, Node, evaluate, graph_from_dict, local_search, tree_extract
from research.local_neighborhood import build_incidence, enumerate_region, propose_region, weighted_depth


def n(eclass, children=(), cost=1):
    return {"eclass": eclass, "children": list(children), "cost": cost}


def make_graph(nodes, roots=("r",)):
    return graph_from_dict({"nodes": nodes, "root_eclasses": list(roots)})


def coordination_graph():
    # The same two-switch sharing barrier used by test_egraph_lab.
    return make_graph({
        "root": n("r", ["a_private", "b_private"], 0),
        "a_private": n("a", cost=2),
        "a_shared": n("a", ["s"], 0),
        "b_private": n("b", cost=2),
        "b_shared": n("b", ["s"], 0),
        "s": n("shared", cost=3),
    })


class LocalNeighborhoodTests(unittest.TestCase):
    def assert_accounted(self, result):
        stats = result.stats
        self.assertEqual(stats.attempted, stats.invalid + stats.duplicates + stats.returned)
        self.assertEqual(stats.returned, len(result.candidates))

    def test_weighted_depth_handles_shared_multioutput_dag_and_leaf_costs(self):
        graph = make_graph({
            "left": n("left", ["shared", "long0"], 1),
            "right": n("right", ["shared", "shared"], 4),
            "shared": n("shared", ["leaf"], 5),
            "long0": n("long0", ["long1"], 0),
            "long1": n("long1", ["long2"], 0),
            "long2": n("long2", ["leaf"], 0),
            "leaf": n("leaf", cost=2),
            "unreachable": n("unreachable", ["unreachable"], 999),
        }, ("left", "right", "leaf"))
        selection = {c: members[0] for c, members in graph.classes.items()}
        self.assertEqual(evaluate(graph, selection).critical[0], "left")
        self.assertEqual(weighted_depth(graph, selection), 11.0)
        leaf = make_graph({"r": n("r", cost=2.5)})
        self.assertEqual(weighted_depth(leaf, {"r": "r"}), 2.5)

    def test_weighted_depth_rejects_reachable_cycles(self):
        graph = make_graph({"r": n("r", ["a"], 2), "a": n("a", ["r"], 3)})
        with self.assertRaisesRegex(ValueError, "cycle"):
            weighted_depth(graph, {"r": "r", "a": "a"})

    def test_weighted_depth_uses_iterative_dynamic_programming(self):
        graph = make_graph({
            str(i): n(str(i), [str(i - 1)] if i else [], 0.25) for i in range(1200)
        }, ("1199",))
        self.assertEqual(weighted_depth(graph, {c: c for c in graph.classes}), 300.0)

    def test_incidence_includes_unchosen_alternatives_and_is_immutable(self):
        graph = coordination_graph()
        index = build_incidence(graph)
        self.assertEqual(index.children["a"], ("shared",))
        self.assertEqual(index.parents["shared"], ("a", "b"))
        self.assertEqual(index.children["r"], ("a", "b"))
        self.assertEqual(index.decision_classes, ("a", "b"))
        with self.assertRaises(TypeError):
            index.children["a"] = ()
        with self.assertRaises(TypeError):
            index.parents["shared"] = ()
        with self.assertRaises(FrozenInstanceError):
            index.decision_classes = ()

    def test_shared_child_link_escapes_single_point_barrier_without_pruning(self):
        graph = coordination_graph()
        initial = tree_extract(graph)
        single = local_search(graph, initial, mode="single")
        self.assertEqual(single.metrics.area, 4)
        region = propose_region(graph, initial, policy="linked", rng=random.Random(7))
        self.assertEqual(set(region), {"a", "b"})
        result = enumerate_region(graph, initial, region)
        self.assertEqual(result.stats.combinations, 4)
        self.assertEqual(result.stats.attempted, 4)
        self.assertEqual(result.stats.invalid, 0)
        self.assertEqual(result.stats.duplicates, 1)
        self.assertTrue(result.stats.exact)
        best = min(result.candidates, key=lambda c: c.metrics.area)
        self.assertEqual(best.metrics.area, 3)
        self.assertEqual(set(best.changed_classes), {"a", "b"})
        # Single switches worsen both proxies, but remain available to ranking.
        worse = [c for c in result.candidates if c.metrics.area > single.metrics.area]
        self.assertEqual(len(worse), 2)
        self.assertTrue(all(c.metrics.depth > single.metrics.depth for c in worse))
        self.assert_accounted(result)

    def test_single_class_region_and_all_outside_choices_are_frozen(self):
        graph = coordination_graph()
        initial = tree_extract(graph)
        result = enumerate_region(graph, initial, ("a",))
        self.assertEqual(result.stats.combinations, 2)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.changed_classes, ("a",))
        for c in initial:
            if c != "a":
                self.assertEqual(candidate.choices[c], initial[c])
        self.assertEqual(initial["a"], "a_private")
        with self.assertRaises(TypeError):
            candidate.choices["a"] = "a_private"
        self.assert_accounted(result)

    def test_cyclic_combinations_are_rejected_and_counted(self):
        graph = make_graph({
            "r": n("r", ["a0", "b0"], 0),
            "a0": n("a"), "a1": n("a", ["b0"]),
            "b0": n("b"), "b1": n("b", ["a0"]),
        })
        selection = {"r": "r", "a": "a0", "b": "b0"}
        result = enumerate_region(graph, selection, ("a", "b"))
        self.assertEqual(result.stats.attempted, 4)
        self.assertEqual(result.stats.invalid, 1)
        self.assertEqual(result.stats.duplicates, 1)
        self.assertEqual(result.stats.returned, 2)
        for candidate in result.candidates:
            self.assertEqual(evaluate(graph, candidate.choices), candidate.metrics)
        self.assert_accounted(result)

    def test_linked_partner_can_be_activated_through_unchosen_alternative(self):
        graph = make_graph({
            "r0": n("r", cost=2), "r1": n("r", ["z0"], 0),
            "z0": n("z", cost=9), "z1": n("z", cost=1),
        })
        selection = {"r": "r0", "z": "z0"}
        self.assertEqual(evaluate(graph, selection).reachable, ("r",))
        region = propose_region(graph, selection, policy="linked", rng=random.Random(3))
        self.assertEqual(region, ("r", "z"))
        result = enumerate_region(graph, selection, region)
        self.assertEqual(result.stats.duplicates, 2)
        self.assertEqual(result.stats.returned, 2)
        best = min(result.candidates, key=lambda c: c.metrics.area)
        self.assertEqual(best.metrics.area, 1)
        self.assertEqual(best.changed_classes, ("r", "z"))
        self.assertIn("z", best.metrics.reachable)
        self.assert_accounted(result)

    def test_distinct_unreachable_choices_repeat_one_reachable_signature(self):
        graph = make_graph({
            "r0": n("r"), "r1": n("r", cost=2),
            "z0": n("z"), "z1": n("z", ["z0"]),
        })
        result = enumerate_region(graph, {"r": "r0", "z": "z0"}, ("r", "z"))
        self.assertEqual(result.stats.attempted, 4)
        # The unreachable z1 self-cycle is immaterial to the root DAG.
        self.assertEqual(result.stats.invalid, 0)
        self.assertEqual(result.stats.duplicates, 3)
        self.assertEqual(result.stats.returned, 1)
        self.assert_accounted(result)

    def test_random_and_linked_policies_have_reachable_decision_anchors(self):
        graph = make_graph({
            "r": n("r", ["a0"], 0), "a0": n("a"), "a1": n("a", cost=2),
            "b0": n("b"), "b1": n("b", cost=2),
        })
        selection = tree_extract(graph)
        index = build_incidence(graph)
        for seed in range(8):
            self.assertEqual(propose_region(graph, selection, rng=random.Random(seed), index=index), ("a",))
            self.assertEqual(
                propose_region(graph, selection, policy="random", rng=random.Random(seed), index=index),
                ("a", "b"),
            )

    def test_parent_and_child_links_use_all_alternatives(self):
        graph = make_graph({
            "r0": n("r", ["a0"], 0), "r1": n("r", cost=2),
            "a0": n("a"), "a1": n("a", cost=2),
        })
        selection = {"r": "r0", "a": "a0"}
        proposals = {
            propose_region(graph, selection, rng=random.Random(seed)) for seed in range(12)
        }
        self.assertEqual(proposals, {("r", "a"), ("a", "r")})

    def test_no_reachable_decision_gives_empty_region(self):
        graph = make_graph({"r": n("r"), "z0": n("z"), "z1": n("z")})
        self.assertEqual(propose_region(graph, tree_extract(graph), rng=random.Random(0)), ())

    def test_cap_seed_and_exact_threshold(self):
        graph = make_graph({
            "r": n("r", ["a0", "b0"], 0),
            **{f"a{i}": n("a", cost=i) for i in range(10)},
            **{f"b{i}": n("b", cost=i) for i in range(10)},
        })
        selection = tree_extract(graph)
        first = enumerate_region(graph, selection, ("a", "b"), max_combinations=7, rng=random.Random(91))
        same = enumerate_region(graph, selection, ("a", "b"), max_combinations=7, rng=random.Random(91))
        other = enumerate_region(graph, selection, ("a", "b"), max_combinations=7, rng=random.Random(92))
        self.assertEqual(first, same)
        self.assertNotEqual(first.candidates, other.candidates)
        self.assertEqual(first.stats.combinations, 100)
        self.assertEqual(first.stats.attempted, 7)
        self.assertFalse(first.stats.exact)
        signatures = {c.metrics.signature for c in first.candidates}
        self.assertEqual(len(signatures), len(first.candidates))
        exact = enumerate_region(graph, selection, ("a", "b"), max_combinations=100)
        self.assertTrue(exact.stats.exact)
        self.assertEqual(exact.stats.attempted, 100)
        self.assertEqual(exact.stats.returned, 99)
        self.assert_accounted(first)
        self.assert_accounted(exact)

    def test_large_product_never_materializes_selection_queue(self):
        # Implicit alternatives make eager product/queue construction fail
        # immediately, without allocating a giant graph in the unit test.
        size = 10**12

        class ImplicitAlternatives:
            def __init__(self, class_id):
                self.class_id = class_id

            def __len__(self):
                return size

            def __getitem__(self, index):
                if not 0 <= index < size:
                    raise IndexError(index)
                return f"{self.class_id}:{index}"

            def __iter__(self):
                raise AssertionError("alternatives must be indexed, never materialized")

        class ImplicitNodes(dict):
            def __getitem__(self, node_id):
                if node_id == "r":
                    return Node("r", "r", "root", ("a", "b"), 0)
                class_id, number = node_id.split(":")
                return Node(node_id, class_id, "leaf", (), int(number))

            def get(self, node_id, default=None):
                return self[node_id]

            def __contains__(self, node_id):
                return node_id == "r" or node_id.split(":")[0] in {"a", "b"}

        graph = EGraph(ImplicitNodes(), {
            "r": ("r",), "a": ImplicitAlternatives("a"), "b": ImplicitAlternatives("b"),
        }, ("r",))
        selection = {"r": "r", "a": "a:0", "b": "b:0"}
        with patch("research.local_neighborhood.evaluate", wraps=evaluate) as evaluator:
            result = enumerate_region(graph, selection, ("a", "b"), max_combinations=5, rng=random.Random(4))
        self.assertEqual(result.stats.combinations, size * size)
        self.assertEqual(result.stats.attempted, 5)
        self.assertEqual(evaluator.call_count, 6)  # Original plus the bounded attempts.
        self.assertFalse(result.stats.exact)
        self.assertEqual(result.stats.returned, 5)
        self.assert_accounted(result)

    def test_index_does_not_materialize_quadratic_shared_parent_pairs(self):
        count = 300
        graph = make_graph({
            "r": n("r", ["p0.0"], 0), "s": n("s"),
            **{f"p{i}.0": n(f"p{i}") for i in range(count)},
            **{f"p{i}.1": n(f"p{i}", ["s"]) for i in range(count)},
        })
        index = build_incidence(graph)
        self.assertEqual(len(index.parents["s"]), count)
        self.assertEqual(sum(map(len, index.children.values())), count + 1)
        region = propose_region(graph, tree_extract(graph), rng=random.Random(1), index=index)
        self.assertEqual(region[0], "p0")
        self.assertIn(region[1], index.decision_set - {"p0"})

    def test_invalid_arguments_and_incomplete_selections_fail(self):
        graph = coordination_graph()
        selection = tree_extract(graph)
        for region in [(), ("a", "a"), ("a", "b", "r"), ("missing",)]:
            with self.subTest(region=region), self.assertRaises(ValueError):
                enumerate_region(graph, selection, region)
        for limit in [0, -1, True, 1.5, float("inf")]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                enumerate_region(graph, selection, ("a",), max_combinations=limit)
        with self.assertRaises(ValueError):
            propose_region(graph, selection, policy="unknown", rng=random.Random(0))
        incomplete = {c: n for c, n in selection.items() if c != "shared"}
        for bad in [incomplete, selection | {"shared": "a_private"}, selection | {"unknown": "s"}]:
            with self.assertRaises(ValueError):
                enumerate_region(graph, bad, ("a",))
            with self.assertRaises(ValueError):
                propose_region(graph, bad, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
