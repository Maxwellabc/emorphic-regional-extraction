"""Bounded connected decision regions and pure mapped-feedback controls.

This module proposes structural candidates only. It does not map circuits or
claim logical equivalence, mapped quality, or algorithmic novelty.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
import random
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from research.egraph_lab import EGraph, evaluate
from research.local_neighborhood import Candidate, IncidenceIndex, _validate_full_selection, build_incidence

SIZES = (2, 8, 32)
POLICIES = ("adaptive", "fixed2", "fixed8", "fixed32", "scheduled", "ramp")


@dataclass(frozen=True)
class RegionSampleStats:
    target_size: int
    actual_size: int
    region_shortfall: int
    attempted: int
    invalid: int
    duplicates: int
    returned: int
    assignment_space: int
    assignment_space_exhausted: bool
    candidate_shortfall: int
    stop_reason: str


@dataclass(frozen=True)
class CandidateDiagnostics:
    changed_reachable_classes: tuple[str, ...]
    newly_reachable_classes: int
    dropped_reachable_classes: int


@dataclass(frozen=True)
class RegionSample:
    region: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    stats: RegionSampleStats
    diagnostics: tuple[CandidateDiagnostics, ...]


@dataclass(frozen=True)
class BatchDecision:
    center_candidate_index: int | None
    best_candidate_index: int | None
    center_strict_improvement: bool
    neutral_move: bool


def _neighbors(class_id: str, index: IncidenceIndex) -> set[str]:
    related = set(index.children[class_id]) | set(index.parents[class_id])
    for child in index.children[class_id]:
        related.update(index.parents[child])
    return (related & index.decision_set) - {class_id}


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
    """Sample noncenter choices inside one connected region; freeze its outside.

    The anchor is a current root-reachable decision class. Growth uses parent,
    child, and shared-child incidences over all alternatives. A disconnected
    component gives a smaller recorded region, without a global-random fill.
    Every region class switches to a noncenter alternative. Distinct complete
    assignments are drawn uniformly without replacement with O(attempts) state;
    root-reachable signatures are then deduplicated against the center, caller
    history, and earlier returned candidates. Unreachable cycles are immaterial
    to evaluate's root-DAG scope. A supplied index must belong to this graph.
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
    seen = {original.signature, *seen_signatures}
    candidates, diagnostics = [], []
    attempted = invalid = duplicates = 0
    # Sparse Fisher-Yates avoids materializing even a huge product of choices.
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
        choices = dict(selection)
        for class_id, members in reversed(tuple(zip(region, alternatives))):
            rank, offset = divmod(rank, len(members))
            choices[class_id] = members[offset]
        attempted += 1
        try:
            metrics = evaluate(graph, choices)
        except ValueError:
            invalid += 1
            continue
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


def qor_score(qor: Mapping[str, float]) -> tuple[Decimal, Decimal]:
    """Use the reported delay/area precision and strict lexicographic order."""
    score = tuple(Decimal(str(qor[key])) for key in ("delay_ps", "area_library_units"))
    if any(not value.is_finite() for value in score):
        raise ValueError("QoR must contain finite delay_ps and area_library_units")
    return score


def batch_scale(policy: str, batch_index: int, *, adaptive_size: int = 2) -> int:
    if policy not in POLICIES or type(batch_index) is not int or batch_index not in range(6):
        raise ValueError("unknown policy or batch index outside the six-batch protocol")
    if type(adaptive_size) is not int or adaptive_size not in SIZES:
        raise ValueError("adaptive_size must be one of 2, 8, 32")
    if policy == "adaptive":
        return adaptive_size
    if policy == "scheduled":
        return SIZES[batch_index % 3]
    if policy == "ramp":
        return SIZES[min(batch_index, len(SIZES) - 1)]
    return int(policy.removeprefix("fixed"))


def next_adaptive_size(current: int, center_qor: Mapping[str, float],
                       candidate_qors: Sequence[Mapping[str, float]]) -> int:
    if type(current) is not int or current not in SIZES or len(candidate_qors) != 3:
        raise ValueError("adaptive feedback requires a valid current size and three measured candidates")
    baseline = qor_score(center_qor)
    scores = [qor_score(qor) for qor in candidate_qors]
    if any(score < baseline for score in scores):
        return SIZES[0]
    if all(score == baseline for score in scores):
        return SIZES[min(SIZES.index(current) + 1, len(SIZES) - 1)]
    return current


def choose_batch_indices(
    center_qor: Mapping[str, float],
    center_signature,
    best_qor: Mapping[str, float],
    candidate_qors: Sequence[Mapping[str, float]],
    candidate_signatures: Sequence,
    *,
    neutral_moves: bool = False,
) -> BatchDecision:
    """Choose center and best separately after one complete shared-center batch.

    Ties use the first candidate in recorded order. Without strict improvement,
    the center can move to the first same-QoR different root-DAG candidate; the
    retained global best never moves on a tie. None means keep that old state.
    """
    if len(candidate_qors) != 3 or len(candidate_signatures) != 3:
        raise ValueError("batch state updates require three measured candidate QoRs and signatures")
    center_score, best_score = qor_score(center_qor), qor_score(best_qor)
    if best_score > center_score:
        raise ValueError("global best cannot be worse than the current search center")
    scores = [qor_score(qor) for qor in candidate_qors]
    winner = min(range(3), key=scores.__getitem__)
    best_index = winner if scores[winner] < best_score else None
    if scores[winner] < center_score:
        return BatchDecision(winner, best_index, True, False)
    if neutral_moves:
        neutral = next((i for i in range(3) if scores[i] == center_score
                        and candidate_signatures[i] != center_signature), None)
        if neutral is not None:
            return BatchDecision(neutral, best_index, False, True)
    return BatchDecision(None, best_index, False, False)
