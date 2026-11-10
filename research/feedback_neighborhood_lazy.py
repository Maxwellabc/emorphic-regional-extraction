"""Deterministic compile-on-first-cycle regional sampling.

Before a cycle is rejected this follows the frozen baseline's full-evaluate
path. Compilation is an existing lazy-initialization technique, not a novelty
claim. No physical labels, lookahead, threshold, or sampling parameter is added.
"""
from __future__ import annotations

import math
import random
from types import FunctionType, MappingProxyType
from typing import Iterable, Mapping

from research.egraph_lab import EGraph, evaluate
from research.feedback_neighborhood import (
    CandidateDiagnostics, RegionSample, RegionSampleStats, SIZES, _neighbors,
)
from research.local_neighborhood import Candidate, IncidenceIndex, _validate_full_selection, build_incidence
from research.region_cycle_filter import FrozenRegionCycleFilter
from research.region_cycle_scc_reference import FrozenRegionSCCCycleFilter


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
    """Preserve baseline results/RNG; build one filter only after a cycle.

    Supported graphs follow graph_from_dict's well-formed EGraph contract:
    valid class/member/operand/root references and finite nonnegative weights.
    The complete center must be valid and root-acyclic; unreachable cycles in
    its complete choices are permitted. A supplied immutable incidence index
    must belong to the same unchanged graph. Arguments/domain and result types
    are those of the frozen baseline.

    The frozen evaluator's ``cycle through eclass ...`` error is the only
    rejection that triggers compilation. Other ValueErrors before compilation
    propagate; an invalid selection admitted by a compiled checker is an
    implementation error. Neither is silently counted as an ordinary cycle.
    """
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
    cycle_filter = None
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
        # With no compiler this is the baseline's full copy and decode path.
        # After compilation the same rank decoding produces regional overrides.
        choices = dict(selection) if cycle_filter is None else {}
        for class_id, members in reversed(tuple(zip(region, alternatives))):
            rank, offset = divmod(rank, len(members))
            choices[class_id] = members[offset]
        attempted += 1
        if cycle_filter is None:
            try:
                metrics = evaluate(graph, choices)
            except ValueError as error:
                if not str(error).startswith("cycle through eclass "):
                    raise
                invalid += 1
                if remaining and attempted < max_attempts and len(candidates) < count:
                    cycle_filter = FrozenRegionCycleFilter(graph, selection, region)
                continue
        else:
            if cycle_filter.has_reachable_cycle(choices):
                invalid += 1
                continue
            complete = dict(selection)
            complete.update(choices)
            choices = complete
            try:
                metrics = evaluate(graph, choices)
            except ValueError as error:
                raise AssertionError("compiled cycle filter admitted an invalid root selection") from error
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


def with_cycle_filter(filter_class):
    """Clone only the constructor binding; never mutate frozen/module globals."""
    namespace = dict(sample_region_candidates.__globals__)
    namespace["FrozenRegionCycleFilter"] = filter_class
    result = FunctionType(sample_region_candidates.__code__, namespace,
                          sample_region_candidates.__name__, sample_region_candidates.__defaults__,
                          sample_region_candidates.__closure__)
    result.__kwdefaults__ = dict(sample_region_candidates.__kwdefaults__ or {})
    result.__annotations__ = dict(sample_region_candidates.__annotations__)
    result.__doc__ = sample_region_candidates.__doc__
    return result


sample_region_candidates_scc = with_cycle_filter(FrozenRegionSCCCycleFilter)
