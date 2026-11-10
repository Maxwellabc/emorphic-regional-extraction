"""Require identical regions, metrics, counters, candidates, and RNG states."""
from pathlib import Path
import json
import random
import unittest

from research.egraph_lab import EGraph, Node, load_egraph
from research.feedback_neighborhood import sample_region_candidates as baseline
from research.feedback_neighborhood_filtered import sample_region_candidates as filtered
from research.local_neighborhood import build_incidence

ROOT = Path(__file__).resolve().parents[1]


class FilteredNeighborhoodTests(unittest.TestCase):
    def compare(self, graph, selection, *, seed, size, count=3, max_attempts=256, seen_signatures=()):
        old_rng, new_rng = random.Random(seed), random.Random(seed)
        index = build_incidence(graph)
        kwargs = dict(size=size, count=count, index=index, max_attempts=max_attempts,
                      seen_signatures=seen_signatures)
        expected = baseline(graph, selection, rng=old_rng, **kwargs)
        actual = filtered(graph, selection, rng=new_rng, **kwargs)
        self.assertEqual(actual, expected)
        self.assertEqual(new_rng.getstate(), old_rng.getstate())
        return expected

    def test_random_graphs_match_complete_sampler_and_rng_state(self):
        rng = random.Random(62370)
        for trial in range(90):
            size = rng.randrange(2, 14)
            classes = tuple(f"c{i}" for i in range(size))
            nodes, members = {}, {}
            for class_id in classes:
                alternatives = []
                for j in range(rng.randrange(2, 5)):
                    name = f"{class_id}.{j}"
                    children = () if j == 0 else tuple(rng.choice(classes) for _ in range(rng.randrange(4)))
                    nodes[name] = Node(name, class_id, "test", children, float(j + 1))
                    alternatives.append(name)
                members[class_id] = tuple(alternatives)
            graph = EGraph(nodes, members, tuple(rng.sample(classes, rng.randrange(1, 3))))
            selection = {c: ns[0] for c, ns in members.items()}
            for region_size in (2, 8, 32):
                previous = self.compare(graph, selection, seed=trial, size=region_size,
                                        count=2 + trial % 3, max_attempts=1 + trial % 40)
                self.compare(graph, selection, seed=trial, size=region_size,
                             seen_signatures=[c.metrics.signature for c in previous.candidates])

    def test_real_adder_initial_centers_include_256_attempt_rejected_region(self):
        run = ROOT / "results/feedback_screen/pilot_001/adder_fixed32_neutral0_s0"
        if not (run / "summary.json").is_file():
            self.skipTest("archived development pilot is unavailable")
        report = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        graph = load_egraph(run / "inputs/boolean_egraph.json")
        for center in (0, 1):
            choices = json.loads((run / f"eval_{center:03d}/choices.json").read_text(encoding="utf-8"))["choices"]
            for seed, size in ((0, 2), (0, 8), (0, 32), (7, 32)):
                result = self.compare(graph, choices, seed=seed, size=size)
                if center == 1 and seed == 0 and size == 32:
                    self.assertEqual(result.region, tuple(report["batches"][0]["proposals"][0]["region"]))
                    self.assertEqual(result.stats.invalid, 256)

    def test_invalid_arguments_match_baseline_without_random_draws(self):
        graph = EGraph({"a0": Node("a0", "a", "test", (), 1.0)}, {"a": ("a0",)}, ("a",))
        for kwargs in ({"size": True}, {"size": 3}, {"size": 2, "count": True},
                       {"size": 2, "max_attempts": 0}):
            for sampler in (baseline, filtered):
                rng = random.Random(0)
                initial = rng.getstate()
                with self.assertRaises(ValueError):
                    sampler(graph, {"a": "a0"}, rng=rng, **kwargs)
                self.assertEqual(rng.getstate(), initial)
        self.compare(graph, {"a": "a0"}, seed=0, size=2)


if __name__ == "__main__":
    unittest.main()
