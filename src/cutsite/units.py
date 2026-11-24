from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import Dataset
from .errors import PolicyError
from .policy import Policy, SimilarityConstraint
from .sim import DEFAULT_MAX_PAIRS, EXACTNESS_RANK, SimilarityResult, resolve


class UnionFind:
    def __init__(self, size: int):
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        parent = self._parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        if self._rank[a] < self._rank[b]:
            a, b = b, a
        self._parent[b] = a
        if self._rank[a] == self._rank[b]:
            self._rank[a] += 1
        return True


@dataclass
class Unit:
    key: int
    members: List[int]
    label_counts: Dict[str, int] = field(default_factory=dict)
    pinned: Optional[str] = None
    reasons: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class UnitGraph:
    units: List[Unit]
    similarity: Dict[str, SimilarityResult]
    merge_counts: Dict[str, int]
    dropped: List[int] = field(default_factory=list)
    spanning_units: int = 0

    @property
    def exactness(self) -> str:
        if not self.similarity:
            return "exact"
        return min(
            (result.exactness for result in self.similarity.values()),
            key=lambda value: EXACTNESS_RANK[value],
        )

    def unit_of(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for position, unit in enumerate(self.units):
            for member in unit.members:
                out[member] = position
        return out

    def to_dict(self) -> Dict[str, object]:
        sizes = sorted((unit.size for unit in self.units), reverse=True)
        total = sum(sizes) or 1
        return {
            "units": len(self.units),
            "entities": total,
            "largest_unit": sizes[0] if sizes else 0,
            "largest_unit_fraction": round(sizes[0] / total, 6) if sizes else 0.0,
            "singleton_units": sum(1 for size in sizes if size == 1),
            "size_percentiles": _percentiles(sizes),
            "merges": dict(sorted(self.merge_counts.items())),
            "pinned_units": sum(1 for unit in self.units if unit.pinned),
            "spanning_units": self.spanning_units,
            "dropped_entities": len(self.dropped),
            "exactness": self.exactness,
            "similarity": {key: result.to_dict() for key, result in sorted(self.similarity.items())},
        }


def _percentiles(sizes: Sequence[int]) -> Dict[str, int]:
    if not sizes:
        return {}
    ordered = sorted(sizes)

    def pick(fraction: float) -> int:
        position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return int(ordered[position])

    return {"p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": int(ordered[-1])}


def _constraint_id(constraint: SimilarityConstraint) -> str:
    return f"{constraint.key}:{constraint.metric}@{constraint.threshold:g}"


def build(
    policy: Policy,
    data: Dataset,
    *,
    exact_max_n: int = 20000,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    backend_overrides: Optional[Dict[str, str]] = None,
) -> UnitGraph:
    size = len(data)
    union = UnionFind(size)
    merge_counts: Dict[str, int] = {}
    similarity: Dict[str, SimilarityResult] = {}

    for constraint in policy.similarity_constraints:
        key = policy.dataset.similarity[constraint.key]
        values = data.column(key.column)
        requested = (backend_overrides or {}).get(constraint.key) or constraint.backend
        backend = resolve(
            constraint.metric,
            key.kind,
            size,
            requested=requested,
            exact_max_n=exact_max_n,
        )
        result = backend.pairs_above(
            values, key.kind, constraint.threshold, max_pairs=max_pairs
        )
        identifier = _constraint_id(constraint)
        similarity[identifier] = result
        merged = 0
        for left, right, _sim in result.pairs:
            if union.union(left, right):
                merged += 1
        merge_counts[identifier] = merged

    for constraint in policy.disjoint_constraints:
        values = data.column(constraint.key)
        buckets: Dict[str, int] = {}
        merged = 0
        for i, value in enumerate(values):
            if value is None:
                continue
            anchor = buckets.setdefault(value, i)
            if anchor != i and union.union(anchor, i):
                merged += 1
        merge_counts[f"disjoint:{constraint.key}"] = merged

    groups: Dict[int, List[int]] = {}
    for i in range(size):
        groups.setdefault(union.find(i), []).append(i)

    label_values = data.column(policy.dataset.label_column) if policy.dataset.label_column else None
    pins, spanning, dropped = _temporal_pins(policy, data, groups)

    units: List[Unit] = []
    for root in sorted(groups):
        members = groups[root]
        if all(member in dropped for member in members):
            continue
        kept = [member for member in members if member not in dropped]
        counts: Dict[str, int] = {}
        if label_values is not None:
            for member in kept:
                bucket = label_values[member] if label_values[member] is not None else "__missing__"
                counts[bucket] = counts.get(bucket, 0) + 1
        units.append(Unit(key=root, members=kept, label_counts=counts, pinned=pins.get(root)))

    return UnitGraph(
        units=units,
        similarity=similarity,
        merge_counts=merge_counts,
        dropped=sorted(dropped),
        spanning_units=spanning,
    )


def _temporal_pins(
    policy: Policy,
    data: Dataset,
    groups: Dict[int, List[int]],
) -> Tuple[Dict[int, str], int, set]:
    pins: Dict[int, str] = {}
    dropped: set = set()
    spanning = 0
    for constraint in policy.temporal_constraints:
        column = policy.dataset.dates[constraint.key]
        dates = data.dates(column)
        for root, members in groups.items():
            before = [m for m in members if dates[m] is not None and dates[m] < constraint.cutoff]
            after = [m for m in members if dates[m] is not None and dates[m] >= constraint.cutoff]
            if before and after:
                spanning += 1
                if constraint.on_span == "error":
                    raise PolicyError(
                        "a unit spans the temporal cutoff, so no assignment can satisfy the constraint",
                        remedy=(
                            "Set on_span to 'test' (conservative), 'train', or 'drop' on the temporal "
                            "constraint, or move the cutoff."
                        ),
                        details={
                            "cutoff": constraint.cutoff.isoformat(),
                            "unit_size": len(members),
                            "before": len(before),
                            "after": len(after),
                        },
                    )
                if constraint.on_span == "drop":
                    dropped.update(members)
                    continue
                target = constraint.after if constraint.on_span == "test" else constraint.before
                pins[root] = target
            elif after:
                pins[root] = constraint.after
            elif before:
                pins[root] = constraint.before
    return pins, spanning, dropped


def dates_available(policy: Policy, data: Dataset) -> Dict[str, List[Optional[_dt.date]]]:
    return {name: data.dates(column) for name, column in policy.dataset.dates.items()}
