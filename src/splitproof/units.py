from __future__ import annotations

import math

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import Dataset
from .digest import digest
from .errors import DataError, PolicyError, VacuousConstraint
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
    signature: str = ""
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
    stratify: str = "none"

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
        total = sum(sizes)
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
            "stratify": self.stratify,
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


MAX_STRATIFY_CLASSES = 50
NUMERIC_BUCKETS = 4
MISSING_CLASS = "__missing__"


def label_kind(policy: Policy, values: Sequence[Optional[str]]) -> str:
    """Whether a label should be treated as numeric or categorical.

    One rule, shared by stratification and by the baseline metric, so the two can
    never disagree about what a label is.
    """
    if policy.dataset.label_kind != "auto":
        return policy.dataset.label_kind
    present = [value for value in values if value is not None]
    if not present:
        return "categorical"
    try:
        numbers = [float(value) for value in present]
    except ValueError:
        return "categorical"
    numbers = [number for number in numbers if math.isfinite(number)]
    if not numbers:
        return "categorical"
    return "numeric" if len(set(numbers)) > NUMERIC_BUCKETS * 2 else "categorical"


def label_classes(policy: Policy, data: Dataset) -> Tuple[Optional[List[str]], str]:
    """Stratification classes per entity, and a word for how they were derived.

    A continuous label has as many values as rows, so stratifying on raw values is
    noise. Numeric labels become quantile buckets. A categorical label with more
    classes than MAX_STRATIFY_CLASSES is almost always an identifier, so
    stratification is skipped and the report says so.
    """
    if not policy.dataset.label_column:
        return None, "none"
    values = data.column(policy.dataset.label_column)
    present = [value for value in values if value is not None]
    kind = label_kind(policy, values)
    if kind == "numeric":
        try:
            parsed = [float(value) for value in present]
        except ValueError as exc:
            raise DataError(
                f"label column '{policy.dataset.label_column}' was declared numeric but holds "
                "a value that is not a number",
                remedy="Set dataset.label_kind to categorical, or clean the column.",
                details={"column": policy.dataset.label_column, "reason": str(exc)},
            ) from exc
        numeric = [number for number in parsed if math.isfinite(number)]
        if not numeric:
            return None, "none"
        import numpy as np

        quantiles = [index / NUMERIC_BUCKETS for index in range(1, NUMERIC_BUCKETS)]
        edges = np.quantile(numeric, quantiles)
        out: List[str] = []
        for value in values:
            if value is None:
                out.append(MISSING_CLASS)
                continue
            number = float(value)
            if not math.isfinite(number):
                out.append(MISSING_CLASS)
                continue
            out.append(f"q{int(np.searchsorted(edges, number, side='right'))}")
        return out, f"numeric, {NUMERIC_BUCKETS} quantile buckets"
    distinct = len({value for value in present})
    if distinct > MAX_STRATIFY_CLASSES:
        return None, f"skipped, {distinct} classes is more than {MAX_STRATIFY_CLASSES}"
    return [value if value is not None else MISSING_CLASS for value in values], "categorical"


def _signature(data: Dataset, members: Sequence[int]) -> str:
    """A content key for a unit, so a split does not depend on row order."""
    return digest(sorted(data.ids[index] for index in members))


def _constraint_id(constraint: SimilarityConstraint) -> str:
    """A key that distinguishes constraints that measure different things.

    Omitting params and backend let two constraints on the same key and threshold
    share a key, so a clean one silently overwrote a violated one and both reported
    satisfied. The threshold is written at full precision for the same reason: 0.5 and
    0.5000004 both render as "0.5" under %g.
    """
    suffix = ""
    if constraint.backend:
        suffix += f"/{constraint.backend}"
    if constraint.params:
        settings = ",".join(f"{name}={value}" for name, value in sorted(constraint.params))
        suffix += f"[{settings}]"
    return f"{constraint.key}:{constraint.metric}@{constraint.threshold!r}{suffix}"


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
            settings=constraint.settings,
        )
        result = backend.pairs_above(
            values, key.kind, constraint.threshold, max_pairs=max_pairs
        )
        identifier = _constraint_id(constraint)
        if size and len(result.missing) == size:
            raise VacuousConstraint(
                f"every entity lacks a usable value for '{constraint.key}', so the constraint "
                f"{identifier} could never bind and a clean result would prove nothing",
                remedy=(
                    "Check that the column holds what the key's 'kind' says it does, and that the "
                    "backend can parse it. A constraint that cannot bind is worse than no "
                    "constraint, because it reports as satisfied."
                ),
                details={
                    "constraint": identifier,
                    "key": constraint.key,
                    "column": key.column,
                    "kind": key.kind,
                    "backend": result.backend,
                    "entities": size,
                    "notes": list(result.notes),
                },
            )
        similarity[identifier] = result
        merged = 0
        if result.groups is not None:
            anchors: Dict[str, int] = {}
            for index, label in enumerate(result.groups):
                if label is None:
                    continue
                anchor = anchors.setdefault(label, index)
                if anchor != index and union.union(anchor, index):
                    merged += 1
        else:
            for left, right in result.pairs.edges():
                if union.union(left, right):
                    merged += 1
        merge_counts[identifier] = merged

    for disjoint in policy.disjoint_constraints:
        values = data.column(disjoint.key)
        if size and all(value is None for value in values):
            raise VacuousConstraint(
                f"every entity lacks a value for group '{disjoint.key}', so the disjoint "
                "constraint could never bind and a clean result would prove nothing",
                remedy="Check the column name and that it actually holds group labels.",
                details={"constraint": f"disjoint:{disjoint.key}", "key": disjoint.key, "entities": size},
            )
        buckets: Dict[str, int] = {}
        merged = 0
        for i, value in enumerate(values):
            if value is None:
                continue
            anchor = buckets.setdefault(value, i)
            if anchor != i and union.union(anchor, i):
                merged += 1
        merge_counts[f"disjoint:{disjoint.key}"] = merged

    groups: Dict[int, List[int]] = {}
    for i in range(size):
        groups.setdefault(union.find(i), []).append(i)

    label_values, stratify_note = label_classes(policy, data)
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
        units.append(
            Unit(
                key=root,
                members=kept,
                signature=_signature(data, kept),
                label_counts=counts,
                pinned=pins.get(root),
            )
        )

    return UnitGraph(
        units=units,
        similarity=similarity,
        merge_counts=merge_counts,
        dropped=sorted(dropped),
        spanning_units=spanning,
        stratify=stratify_note,
    )


def _temporal_pins(
    policy: Policy,
    data: Dataset,
    groups: Dict[int, List[int]],
) -> Tuple[Dict[int, str], int, set]:
    pins: Dict[int, str] = {}
    dropped: set = set()
    spanning = 0
    if len(policy.temporal_constraints) > 1:
        raise PolicyError(
            f"{len(policy.temporal_constraints)} temporal constraints were declared, and they "
            "would overwrite each other's placements",
            remedy=(
                "Declare one temporal cutoff. Two cutoffs on different date columns can demand "
                "opposite folds for the same unit, which no assignment satisfies."
            ),
            details={"count": len(policy.temporal_constraints)},
        )
    for constraint in policy.temporal_constraints:
        column = policy.dataset.dates[constraint.key]
        dates = data.dates(column)
        if dates and all(when is None for when in dates):
            raise VacuousConstraint(
                f"every entity lacks a date for '{constraint.key}', so the temporal constraint "
                "could never bind and a clean result would prove nothing",
                remedy="Check the column name and that it holds ISO dates.",
                details={"constraint": f"temporal:{constraint.key}", "column": column},
            )
        for root, members in groups.items():
            before = [m for m in members if (when := dates[m]) is not None and when < constraint.cutoff]
            after = [m for m in members if (when := dates[m]) is not None and when >= constraint.cutoff]
            if before and after:
                spanning += 1
                if constraint.on_span == "error":
                    raise PolicyError(
                        "a unit spans the temporal cutoff, so no assignment can satisfy the constraint",
                        remedy=(
                            "Set on_span to 'test' (the conservative choice) or 'drop' on the "
                            "temporal constraint, or move the cutoff."
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
                pins[root] = constraint.after
            elif after:
                pins[root] = constraint.after
            elif before:
                pins[root] = constraint.before
    return pins, spanning, dropped


def dates_available(policy: Policy, data: Dataset) -> Dict[str, List[Optional[_dt.date]]]:
    return {name: data.dates(column) for name, column in policy.dataset.dates.items()}
