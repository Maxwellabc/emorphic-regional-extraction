"""Structural correctness checks, independent of mapped QoR claims."""

import importlib.util
from pathlib import Path
import random
import unittest
import uuid

from research.egraph_lab import evaluate, exact_extract, graph_from_dict, load_egraph, local_search, milp_extract, tree_extract


def make_graph(nodes, roots):
    return graph_from_dict({"nodes": nodes, "root_eclasses": roots})


def n(eclass, children=(), cost=1, op="test"):
    return {"eclass": eclass, "children": list(children), "cost": cost, "op": op}


def coordination_graph():
    return make_graph({
        "root": n("r", ["a_private", "b_private"], 0),
        "a_private": n("a", cost=2),
        "a_shared": n("a", ["s"], 0),
        "b_private": n("b", cost=2),
        "b_shared": n("b", ["s"], 0),
        "s": n("shared", cost=3),
    }, ["r"])


class EGraphLabTests(unittest.TestCase):
    def test_resolves_child_node_id_to_class(self):
        graph = make_graph({"out.0": n("out", ["in.1"]), "in.0": n("in", cost=1), "in.1": n("in", cost=10)}, ["out"])
        self.assertEqual(graph.nodes["out.0"].children, ("in",))
        metrics = evaluate(graph, tree_extract(graph))
        self.assertEqual(metrics.area, 2)

    def test_shared_children_are_counted_once(self):
        graph = make_graph({"r": n("r", ["a", "b"]), "a": n("a", ["s"]), "b": n("b", ["s"]), "s": n("s", cost=4)}, ["r"])
        result = evaluate(graph, tree_extract(graph))
        self.assertEqual(result.area, 7)
        self.assertEqual(result.depth, 2)
        self.assertEqual(len(result.reachable), 4)

    def test_repeated_child_does_not_create_false_cycle(self):
        graph = make_graph({"r": n("r", ["s", "s"]), "s": n("s")}, ["r"])
        self.assertEqual(evaluate(graph, tree_extract(graph)).area, 2)

    def test_cycle_missing_and_foreign_choices_rejected(self):
        graph = make_graph({"r": n("r", ["a"]), "a": n("a", ["r"]), "a_leaf": n("a")}, ["r"])
        for selection in ({"r": "r", "a": "a"}, {"r": "r"}, {"r": "a_leaf", "a": "a_leaf"}):
            with self.assertRaises(ValueError):
                evaluate(graph, selection)
        self.assertEqual(tree_extract(graph)["a"], "a_leaf")

    def test_unreachable_choices_do_not_affect_metrics(self):
        graph = make_graph({"r": n("r"), "z": n("z", ["z"], 99)}, ["r"])
        result = evaluate(graph, {"r": "r", "z": "z"})
        self.assertEqual(result.area, 1)
        self.assertEqual(result.signature, (("r", "r"),))

    def test_atomic_two_switches_escape_single_point_optimum(self):
        graph = coordination_graph()
        initial = tree_extract(graph)
        single = local_search(graph, initial, mode="single")
        grouped = local_search(graph, initial, mode="group")
        exact = exact_extract(graph)
        self.assertEqual(single.metrics.area, 4)
        self.assertEqual(grouped.metrics.area, 3)
        self.assertEqual(grouped.metrics.area, exact.metrics.area)

    def test_zero_cost_cycle_ties_remain_finite(self):
        graph = make_graph({"a_cycle": n("a", ["b_leaf"], 0), "a_leaf": n("a", cost=0), "b_leaf": n("b", ["a_leaf"], 0)}, ["a"])
        self.assertEqual(tree_extract(graph)["a"], "a_leaf")
        self.assertEqual(evaluate(graph, tree_extract(graph)).depth, 0)

    def test_multiple_roots_share_implementation(self):
        graph = make_graph({"a": n("a", ["s"]), "b": n("b", ["s"]), "s": n("s", cost=3)}, ["a", "b", "a"])
        self.assertEqual(evaluate(graph, tree_extract(graph)).area, 5)

    def test_invalid_graph_rejected(self):
        cases = [
            {"nodes": {"r": n("r", ["absent"])}, "root_eclasses": ["r"]},
            {"nodes": {"r": n("r", cost=-1)}, "root_eclasses": ["r"]},
            {"nodes": {"r": n("r", cost=float("nan"))}, "root_eclasses": ["r"]},
            {"nodes": {"r": n("r")}, "root_eclasses": []},
        ]
        for data in cases:
            with self.assertRaises(ValueError):
                graph_from_dict(data)

    def test_exact_preflight_limit_and_infeasibility(self):
        with self.assertRaises(ValueError):
            exact_extract(coordination_graph(), max_combinations=1)
        graph = make_graph({"r": n("r", ["r"])}, ["r"])
        with self.assertRaises(ValueError):
            exact_extract(graph)

    def test_budget_is_hard_limit(self):
        graph = coordination_graph()
        result = local_search(graph, tree_extract(graph), mode="group", max_evaluations=1)
        self.assertEqual(result.evaluations, 1)
        self.assertFalse(result.exhausted)

    def test_loader_roots_alias_and_default_cost(self):
        # Windows sandbox ACLs can make mkdtemp(mode=0700) inaccessible even
        # inside the workspace. Use a unique ordinary file in the existing
        # test directory, then remove exactly that file.
        path = Path(__file__).parent / f".loader-{uuid.uuid4().hex}.json"
        try:
            path.write_text('{"nodes":{"n":{"eclass":"c"}},"roots":["c"]}', encoding="utf-8")
            graph = load_egraph(path)
            self.assertEqual(evaluate(graph, tree_extract(graph)).area, 1)
        finally:
            path.unlink(missing_ok=True)

    def test_long_path_uses_iterative_validation(self):
        graph = make_graph({str(i): n(str(i), [str(i - 1)] if i else []) for i in range(1200)}, ["1199"])
        result = evaluate(graph, {str(i): str(i) for i in range(1200)})
        self.assertEqual(result.depth, 1199)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "optional SciPy is unavailable")
    def test_milp_matches_exhaustive_oracle_with_cycles(self):
        for seed in range(12):
            rng = random.Random(seed)
            nodes = {}
            for i in range(4):
                nodes[f"{i}.0"] = n(str(i), cost=rng.randrange(5))
                children = [f"{rng.randrange(4)}.0" for _ in range(rng.randrange(1, 3))]
                nodes[f"{i}.1"] = n(str(i), children, rng.randrange(3))
            graph = make_graph(nodes, ["2", "3"])
            expected = exact_extract(graph)
            result = milp_extract(graph, time_limit=5)
            self.assertTrue(result.proven_optimal, result.message)
            self.assertAlmostEqual(result.metrics.area, expected.metrics.area)
            self.assertAlmostEqual(result.lower_bound, expected.metrics.area)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "optional SciPy is unavailable")
    def test_milp_rejects_self_cycle_and_finds_joint_improvement(self):
        impossible = make_graph({"r": n("r", ["r"], 0)}, ["r"])
        result = milp_extract(impossible)
        self.assertEqual(result.status, 2)
        self.assertIsNone(result.metrics)
        result = milp_extract(coordination_graph())
        self.assertTrue(result.proven_optimal)
        self.assertEqual(result.metrics.area, 3)


if __name__ == "__main__":
    unittest.main()
