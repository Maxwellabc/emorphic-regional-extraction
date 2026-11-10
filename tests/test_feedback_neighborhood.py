"""Structural scope, exhaustion accounting, and pure feedback transitions."""
from dataclasses import FrozenInstanceError
import random
import unittest

from research.egraph_lab import evaluate, graph_from_dict
from research.feedback_neighborhood import (batch_scale, choose_batch_indices, next_adaptive_size,
                                           qor_score, sample_region_candidates)
from research.local_neighborhood import build_incidence


def node(class_id, children=(), cost=1):
    return {"eclass": class_id, "children": list(children), "cost": cost}


def graph(nodes, roots=("r",)):
    return graph_from_dict({"nodes": nodes, "root_eclasses": list(roots)})


def linked_graph():
    nodes = {"root": node("r", ["a0", "b0", "c0"], 0), "leaf": node("leaf"),
             "z0": node("z", cost=10), "z1": node("z", cost=20)}
    for c in "abc":
        nodes.update({f"{c}{i}": node(c, ["leaf"], i + 1) for i in range(4)})
    g = graph(nodes)
    return g, {c: members[0] for c, members in g.classes.items()}


def qor(delay, area=10):
    return {"delay_ps": delay, "area_library_units": area}


class FeedbackNeighborhoodTests(unittest.TestCase):
    def assert_accounted(self, result):
        stats = result.stats
        self.assertEqual(stats.attempted, stats.invalid + stats.duplicates + stats.returned)
        self.assertEqual(stats.returned, len(result.candidates))
        self.assertEqual(stats.candidate_shortfall, 3 - stats.returned)
        self.assertEqual(len(result.diagnostics), len(result.candidates))

    def test_region_outside_is_frozen_and_every_inside_choice_changes(self):
        g, selection = linked_graph()
        result = sample_region_candidates(g, selection, size=2, rng=random.Random(17))
        self.assertEqual(result.stats.actual_size, 2)
        self.assertEqual(result.stats.returned, 3)
        self.assertLessEqual(set(result.region), set("abc"))
        self.assertNotIn("z", result.region)
        for candidate, diagnostics in zip(result.candidates, result.diagnostics):
            self.assertEqual(candidate.changed_classes, result.region)
            for c in selection:
                if c in result.region:
                    self.assertNotEqual(candidate.choices[c], selection[c])
                else:
                    self.assertEqual(candidate.choices[c], selection[c])
            self.assertEqual(evaluate(g, candidate.choices), candidate.metrics)
            self.assertEqual(diagnostics.changed_reachable_classes, result.region)
            self.assertEqual(diagnostics.newly_reachable_classes, 0)
            self.assertEqual(diagnostics.dropped_reachable_classes, 0)
            with self.assertRaises(TypeError):
                candidate.choices["z"] = "z1"
        self.assertEqual(selection["z"], "z0")
        with self.assertRaises(FrozenInstanceError):
            result.stats.actual_size = 99
        self.assert_accounted(result)

    def test_connected_component_shortfall_never_fills_with_unrelated_decisions(self):
        g, selection = linked_graph()
        result = sample_region_candidates(g, selection, size=8, rng=random.Random(7))
        self.assertEqual(result.region, ("a", "b", "c"))
        self.assertEqual(result.stats.target_size, 8)
        self.assertEqual(result.stats.actual_size, 3)
        self.assertEqual(result.stats.region_shortfall, 5)
        self.assertEqual(result.stats.assignment_space, 27)
        self.assertEqual(result.stats.returned, 3)
        self.assert_accounted(result)

    def test_all_alternative_parent_child_links_allow_currently_unreachable_classes(self):
        g = graph({"r0": node("r"), "r1": node("r", ["z0"]),
                   "z0": node("z"), "z1": node("z", cost=2)})
        result = sample_region_candidates(g, {"r": "r0", "z": "z0"}, size=2, rng=random.Random(0))
        self.assertEqual(result.region, ("r", "z"))
        self.assertEqual(result.stats.returned, 1)
        self.assertEqual(result.stats.candidate_shortfall, 2)
        self.assertEqual(result.stats.stop_reason, "assignment_space_exhausted")
        self.assertEqual(result.diagnostics[0].newly_reachable_classes, 1)
        self.assertEqual(result.diagnostics[0].changed_reachable_classes, ("r", "z"))
        self.assert_accounted(result)

    def test_joint_reachable_cycles_are_rejected_and_bounded(self):
        g = graph({"r": node("r", ["a0", "b0"]), "a0": node("a"), "a1": node("a", ["b0"]),
                   "b0": node("b"), "b1": node("b", ["a0"])})
        result = sample_region_candidates(g, {"r": "r", "a": "a0", "b": "b0"},
                                          size=2, rng=random.Random(3), max_attempts=20)
        self.assertEqual(result.stats.assignment_space, 1)
        self.assertEqual(result.stats.attempted, 1)
        self.assertEqual(result.stats.invalid, 1)
        self.assertEqual(result.candidates, ())
        self.assertTrue(result.stats.assignment_space_exhausted)
        self.assertEqual(result.stats.stop_reason, "assignment_space_exhausted")
        self.assert_accounted(result)

    def test_reachable_dag_duplicates_include_unreachable_choice_differences(self):
        g = graph({"r0": node("r"), "r1": node("r", cost=2), "r2": node("r", ["z0"]),
                   "z0": node("z"), "z1": node("z", cost=2), "z2": node("z", cost=3)})
        result = sample_region_candidates(g, {"r": "r0", "z": "z0"}, size=2,
                                          rng=random.Random(2), max_attempts=50)
        self.assertEqual(result.stats.assignment_space, 4)
        self.assertEqual(result.stats.returned, 3)
        self.assertEqual(len({c.metrics.signature for c in result.candidates}), 3)
        history = [c.metrics.signature for c in result.candidates]
        exhausted = sample_region_candidates(g, {"r": "r0", "z": "z0"}, size=2,
                                             rng=random.Random(2), max_attempts=50, seen_signatures=history)
        self.assertEqual(exhausted.stats.attempted, 4)
        self.assertEqual(exhausted.stats.duplicates, 4)
        self.assertEqual(exhausted.stats.returned, 0)
        self.assertTrue(exhausted.stats.assignment_space_exhausted)
        self.assert_accounted(result)
        self.assert_accounted(exhausted)

    def test_seed_and_supplied_incidence_are_deterministic(self):
        g, selection = linked_graph()
        expected = sample_region_candidates(g, selection, size=2, rng=random.Random(91))
        indexed = sample_region_candidates(g, selection, size=2, rng=random.Random(91), index=build_incidence(g))
        self.assertEqual(expected, indexed)
        other = sample_region_candidates(g, selection, size=2, rng=random.Random(92))
        self.assertNotEqual(expected.candidates, other.candidates)
        reversed_graph = type(g)(dict(reversed(list(g.nodes.items()))), dict(reversed(list(g.classes.items()))), g.roots)
        same = sample_region_candidates(reversed_graph, selection, size=2, rng=random.Random(91))
        self.assertEqual(expected, same)

    def test_attempt_limit_and_no_decisions_report_explicit_exhaustion(self):
        g, selection = linked_graph()
        result = sample_region_candidates(g, selection, size=2, rng=random.Random(2), max_attempts=1)
        self.assertEqual(result.stats.attempted, 1)
        self.assertEqual(result.stats.returned, 1)
        self.assertFalse(result.stats.assignment_space_exhausted)
        self.assertEqual(result.stats.stop_reason, "max_attempts")
        self.assert_accounted(result)
        g = graph({"r": node("r"), "z0": node("z"), "z1": node("z")})
        empty = sample_region_candidates(g, {"r": "r", "z": "z0"}, size=2, rng=random.Random(2))
        self.assertEqual(empty.region, ())
        self.assertEqual(empty.stats.stop_reason, "no_reachable_decisions")
        self.assertEqual(empty.stats.region_shortfall, 2)
        self.assert_accounted(empty)

    def test_full_selection_and_limits_are_validated(self):
        g, selection = linked_graph()
        for bad in (selection | {"foreign": "z0"}, {c: n for c, n in selection.items() if c != "z"},
                    selection | {"a": "z0"}):
            with self.assertRaises(ValueError):
                sample_region_candidates(g, bad, size=2, rng=random.Random(0))
        for limits in ({"size": 3}, {"size": True}, {"size": 2, "count": 0},
                       {"size": 2, "max_attempts": 0}, {"size": 2, "count": True}):
            with self.assertRaises(ValueError):
                sample_region_candidates(g, selection, rng=random.Random(0), **limits)

    def test_six_batch_scales_and_adaptive_feedback(self):
        self.assertEqual([batch_scale("scheduled", i) for i in range(6)], [2, 8, 32, 2, 8, 32])
        self.assertEqual([batch_scale("ramp", i) for i in range(6)], [2, 8, 32, 32, 32, 32])
        for size in (2, 8, 32):
            self.assertEqual([batch_scale(f"fixed{size}", i) for i in range(6)], [size] * 6)
            self.assertEqual(batch_scale("adaptive", 1, adaptive_size=size), size)
        self.assertEqual(next_adaptive_size(2, qor(10), [qor(10)] * 3), 8)
        self.assertEqual(next_adaptive_size(8, qor(10), [qor(10)] * 3), 32)
        self.assertEqual(next_adaptive_size(32, qor(10), [qor(10)] * 3), 32)
        self.assertEqual(next_adaptive_size(32, qor(10), [qor(10), qor(9), qor(11)]), 2)
        self.assertEqual(next_adaptive_size(8, qor(10), [qor(10), qor(11), qor(10)]), 8)
        # Same delay with smaller area is a strict lexicographic improvement.
        self.assertEqual(next_adaptive_size(32, qor(10), [qor(10, 9), qor(10), qor(10)]), 2)

    def test_neutral_center_moves_without_overwriting_strict_global_best(self):
        signatures = [(('a', 'old'),), (('a', 'neutral'),), (('a', 'later'),)]
        result = choose_batch_indices(qor(10), signatures[0], qor(10), [qor(10)] * 3, signatures,
                                      neutral_moves=True)
        self.assertEqual(result.center_candidate_index, 1)
        self.assertIsNone(result.best_candidate_index)
        self.assertFalse(result.center_strict_improvement)
        self.assertTrue(result.neutral_move)
        strict = choose_batch_indices(qor(10), signatures[0], qor(10), [qor(10)] * 3, signatures)
        self.assertIsNone(strict.center_candidate_index)
        self.assertIsNone(strict.best_candidate_index)

    def test_strict_center_and_global_best_have_separate_thresholds(self):
        result = choose_batch_indices(qor(10), "center", qor(8), [qor(9), qor(9), qor(11)],
                                      ["first", "tied", "worse"], neutral_moves=True)
        self.assertEqual(result.center_candidate_index, 0)
        self.assertIsNone(result.best_candidate_index)
        self.assertTrue(result.center_strict_improvement)
        self.assertFalse(result.neutral_move)
        best = choose_batch_indices(qor(10), "center", qor(8), [qor(9), qor(7), qor(7)], ["x", "first", "second"])
        self.assertEqual(best.center_candidate_index, 1)
        self.assertEqual(best.best_candidate_index, 1)

    def test_feedback_rejects_incomplete_batches_and_nonfinite_qor(self):
        with self.assertRaises(ValueError):
            next_adaptive_size(2, qor(10), [qor(10)])
        with self.assertRaises(ValueError):
            choose_batch_indices(qor(10), "center", qor(10), [qor(10)] * 3, ["only-one"])
        with self.assertRaises(ValueError):
            choose_batch_indices(qor(8), "center", qor(10), [qor(10)] * 3, ["a", "b", "c"])
        with self.assertRaises(ValueError):
            qor_score(qor(float("nan")))
        with self.assertRaises(ValueError):
            batch_scale("adaptive", 6)
        with self.assertRaises(ValueError):
            batch_scale("unknown", 0)


if __name__ == "__main__":
    unittest.main()
