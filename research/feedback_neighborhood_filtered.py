"""Identical regional sampling with exact frozen-boundary cycle prechecks.

The sampling loop is deliberately kept aligned with feedback_neighborhood.py
SHA256 516a7af55c0473b9fdb286fda1ecedd7c5f69a7a6795ed0bcf997eb75a8a80f9.
Only infeasible assignments avoid a complete selection copy and full evaluate.
Any future sampling change requires paired result and RNG-state regression.
"""
from __future__ import annotations

import math
import random
from types import MappingProxyType
from typing import Iterable, Mapping

from research.egraph_lab import EGraph, evaluate
from research.feedback_neighborhood import (
    CandidateDiagnostics, RegionSample, RegionSampleStats, SIZES, _neighbors,
)
from research.local_neighborhood import Candidate, IncidenceIndex, _validate_full_selection, build_incidence
from research.region_cycle_filter import FrozenRegionCycleFilter


def sample_region_candidates(
    graph: EGraph,
    selection: Mapping[str, str],
    *,
    size: int,
    count: int = 3,
    rng: random.Random,
    index: IncidenceIndex | None = None,
    max_attempts: int = 256,
    seen_signatures: Iterable[tuple[tuple[str, str], ...]] = (),
) -> RegionSample:
    """Preserve baseline draws, candidates and counters while checking cycles early."""
    if type(size) is not int or size not in SIZES:
        raise ValueError("size must be one of 2, 8, 32")
    if type(count) is not int or count < 1 or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("count and max_attempts must be positive integers")
    original = _validate_full_selection(graph, selection)
    index = build_incidence(graph) if index is None else index
    anchors = sorted(set(original.reachable) & index.decision_set)
    if not anchors:
        return RegionSample((), (), RegionSampleStats(size, 0, size, 0, 0, 0, 0, 0, True,
                                                     count, "no_reachable_decisions"), ())
    anchor = rng.choice(anchors)
    chosen = {anchor}
    frontier = _neighbors(anchor, index)
    while len(chosen) < size and frontier:
        class_id = rng.choice(sorted(frontier))
        chosen.add(class_id)
        frontier.update(_neighbors(class_id, index))
        frontier.difference_update(chosen)
    region = tuple(sorted(chosen))
    alternatives = [tuple(sorted(n for n in graph.classes[c] if n != selection[c])) for c in region]
    total = math.prod(len(members) for members in alternatives)
    cycle_filter = FrozenRegionCycleFilter(graph, selection, region)
    seen = {original.signature, *seen_signatures}
    candidates, diagnostics = [], []
    attempted = invalid = duplicates = 0
    slots: dict[int, int] = {}
    remaining = total
    before = set(original.reachable)
    while remaining and attempted < max_attempts and len(candidates) < count:
        draw = rng.randrange(remaining)
        rank = slots.get(draw, draw)
        remaining -= 1
        replacement = slots.pop(remaining, remaining)
        if draw != remaining:
            slots[draw] = replacement
        overrides = {}
        for class_id, members in reversed(tuple(zip(region, alternatives))):
            rank, offset = divmod(rank, len(members))
            overrides[class_id] = members[offset]
        attempted += 1
        if cycle_filter.has_reachable_cycle(overrides):
            invalid += 1
            continue
        choices = dict(selection)
        choices.update(overrides)
        # A precheck that admitted a cyclic assignment is an implementation error,
        # so do not suppress evaluate errors here and silently mask disagreement.
        metrics = evaluate(graph, choices)
        if metrics.signature in seen:
            duplicates += 1
            continue
        seen.add(metrics.signature)
        candidates.append(Candidate(MappingProxyType(choices), metrics, region))
        after = set(metrics.reachable)
        diagnostics.append(CandidateDiagnostics(tuple(c for c in region if c in before | after),
                                                len(after - before), len(before - after)))
    reason = ("requested_count" if len(candidates) == count else
              "assignment_space_exhausted" if not remaining else "max_attempts")
    stats = RegionSampleStats(size, len(region), size - len(region), attempted, invalid, duplicates,
                              len(candidates), total, not remaining, count - len(candidates), reason)
    return RegionSample(region, tuple(candidates), stats, tuple(diagnostics))
