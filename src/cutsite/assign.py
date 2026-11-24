from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import Dataset
from .errors import GiantUnit, InfeasibleSplit
from .policy import Policy
from .units import Unit, UnitGraph

DROPPED = "__dropped__"
MAX_FOLDS = 1000


@dataclass
class FoldStat:
    name: str
    target: int
    count: int
    units: int
    label_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "target": self.target,
            "count": self.count,
            "units": self.units,
            "drift": round((self.count - self.target) / self.target, 6) if self.target else None,
            "labels": dict(sorted(self.label_counts.items())),
        }


@dataclass
class Assignment:
    fold_of: Dict[int, str]
    folds: List[str]
    stats: List[FoldStat]
    dropped: List[int] = field(default_factory=list)

    def fold_members(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {name: [] for name in self.folds}
        for index, fold in self.fold_of.items():
            if fold in out:
                out[fold].append(index)
        for members in out.values():
            members.sort()
        return out

    def to_rows(self, data: Dataset) -> List[Tuple[str, str]]:
        return [(data.ids[i], self.fold_of.get(i, DROPPED)) for i in range(len(data))]

    def to_dict(self) -> Dict[str, object]:
        return {
            "folds": list(self.folds),
            "stats": [stat.to_dict() for stat in self.stats],
            "dropped": len(self.dropped),
        }


def _tiebreak(seed: int, unit: Unit) -> str:
    payload = f"{seed}:{unit.signature}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _targets(policy: Policy, assignable: int, fold_names: Sequence[str]) -> Dict[str, int]:
    if policy.scheme.kind == "kfold":
        base = assignable // len(fold_names)
        remainder = assignable - base * len(fold_names)
        return {
            name: base + (1 if position < remainder else 0)
            for position, name in enumerate(fold_names)
        }
    return {name: int(round(policy.scheme.fractions[name] * assignable)) for name in fold_names}


def assign(policy: Policy, graph: UnitGraph, *, tolerance: float = 0.1) -> Assignment:
    if policy.scheme.kind == "leave-unit-out":
        return _leave_unit_out(graph)

    fold_names = policy.scheme.fold_names
    assignable = sum(unit.size for unit in graph.units)
    targets = _targets(policy, assignable, fold_names)
    counts = {name: 0 for name in fold_names}
    unit_counts = {name: 0 for name in fold_names}
    label_counts: Dict[str, Dict[str, int]] = {name: {} for name in fold_names}
    fold_of: Dict[int, str] = {}

    pinned = [unit for unit in graph.units if unit.pinned]
    floating = [unit for unit in graph.units if not unit.pinned]

    for unit in sorted(pinned, key=lambda u: (-u.size, _tiebreak(policy.seed, u))):
        _place(unit, unit.pinned, fold_of, counts, unit_counts, label_counts)

    if floating:
        largest = max(unit.size for unit in floating)
        capacity = max(targets[name] * (1.0 + tolerance) for name in fold_names)
        if largest > capacity:
            offender = max(floating, key=lambda u: u.size)
            raise GiantUnit(
                f"the largest connected unit holds {largest} of {assignable} entities "
                f"({largest / max(assignable, 1):.1%}), which does not fit in any fold",
                remedy=(
                    "Your data is one homology blob at this threshold. Raise the similarity threshold, "
                    "use scheme.kind: leave-unit-out, or accept a larger tolerance with --tolerance. "
                    "A balanced split at this threshold does not exist."
                ),
                details={
                    "largest_unit": largest,
                    "largest_unit_fraction": round(largest / max(assignable, 1), 6),
                    "entities": assignable,
                    "fold_targets": dict(sorted(targets.items())),
                    "tolerance": tolerance,
                    "example_members": offender.members[:10],
                },
            )

    shares = _label_shares(graph.units)
    placement: Dict[int, str] = {}
    ordered = sorted(floating, key=lambda u: (-u.size, _tiebreak(policy.seed, u)))
    for position, unit in enumerate(ordered):
        choice = _best_fold(unit, fold_names, targets, counts, label_counts, shares)
        placement[position] = choice
        _place(unit, choice, fold_of, counts, unit_counts, label_counts)

    _fill_empty_folds(
        fold_names, ordered, placement, fold_of, counts, unit_counts, label_counts
    )

    empty = [name for name in fold_names if counts[name] == 0]
    if empty:
        raise InfeasibleSplit(
            f"fold(s) {empty} came out empty, and no unit could be moved to fill them",
            remedy=(
                f"There are {len(graph.units)} independent units for {len(fold_names)} folds, and "
                "the units that could move are pinned by a temporal constraint. Lower the "
                "similarity threshold, reduce the number of folds, or use "
                "scheme.kind: leave-unit-out."
            ),
            details={
                "empty_folds": empty,
                "units": len(graph.units),
                "movable_units": len(ordered),
                "folds": list(fold_names),
            },
        )

    stats = [
        FoldStat(
            name=name,
            target=targets[name],
            count=counts[name],
            units=unit_counts[name],
            label_counts=label_counts[name],
        )
        for name in fold_names
    ]
    return Assignment(fold_of=fold_of, folds=list(fold_names), stats=stats, dropped=list(graph.dropped))


def _fill_empty_folds(
    fold_names: Sequence[str],
    ordered: Sequence[Unit],
    placement: Dict[int, str],
    fold_of: Dict[int, str],
    counts: Dict[str, int],
    unit_counts: Dict[str, int],
    label_counts: Dict[str, Dict[str, int]],
) -> None:
    """Move units into folds the greedy pass left empty.

    Minimising total squared deviation can still put every unit in one fold when a
    fold's target is small and the units are large: an 80/20 split of two equal
    units scores better with both in train than with one in each. An empty fold is
    not a split, so the smallest movable unit is moved into it.
    """
    for name in fold_names:
        if counts[name] > 0:
            continue
        donors = [
            position
            for position, unit in enumerate(ordered)
            if unit_counts[placement[position]] > 1
        ]
        if not donors:
            return
        choice = min(donors, key=lambda position: (ordered[position].size, placement[position]))
        unit = ordered[choice]
        source = placement[choice]
        for member in unit.members:
            fold_of[member] = name
        counts[source] -= unit.size
        unit_counts[source] -= 1
        counts[name] += unit.size
        unit_counts[name] += 1
        for label, amount in unit.label_counts.items():
            label_counts[source][label] = label_counts[source].get(label, 0) - amount
            if label_counts[source][label] == 0:
                del label_counts[source][label]
            label_counts[name][label] = label_counts[name].get(label, 0) + amount
        placement[choice] = name


def _place(
    unit: Unit,
    fold: str,
    fold_of: Dict[int, str],
    counts: Dict[str, int],
    unit_counts: Dict[str, int],
    label_counts: Dict[str, Dict[str, int]],
) -> None:
    for member in unit.members:
        fold_of[member] = fold
    counts[fold] += unit.size
    unit_counts[fold] += 1
    for label, amount in unit.label_counts.items():
        label_counts[fold][label] = label_counts[fold].get(label, 0) + amount


def _label_shares(units: Sequence[Unit]) -> Dict[str, float]:
    totals: Dict[str, int] = {}
    for unit in units:
        for label, amount in unit.label_counts.items():
            totals[label] = totals.get(label, 0) + amount
    overall = sum(totals.values())
    if not overall:
        return {}
    return {label: amount / overall for label, amount in totals.items()}


def _best_fold(
    unit: Unit,
    fold_names: Sequence[str],
    targets: Dict[str, int],
    counts: Dict[str, int],
    label_counts: Dict[str, Dict[str, int]],
    shares: Dict[str, float],
) -> str:
    best_name = fold_names[0]
    best_score = None
    for candidate in fold_names:
        size_cost = 0.0
        for name in fold_names:
            target = max(targets[name], 1)
            placed = counts[name] + (unit.size if name == candidate else 0)
            size_cost += ((placed - targets[name]) / target) ** 2
        drift = 0.0
        if shares:
            target = max(targets[candidate], 1)
            for label, share in shares.items():
                placed = label_counts[candidate].get(label, 0) + unit.label_counts.get(label, 0)
                drift += abs(placed / target - share)
        score = (round(size_cost, 9), round(drift, 9), candidate)
        if best_score is None or score < best_score:
            best_score, best_name = score, candidate
    return best_name


def _leave_unit_out(graph: UnitGraph) -> Assignment:
    if len(graph.units) > MAX_FOLDS:
        raise InfeasibleSplit(
            f"leave-unit-out would make {len(graph.units)} folds, above the {MAX_FOLDS} cap",
            remedy="Use scheme.kind: kfold with a modest fold count instead.",
            details={"units": len(graph.units), "cap": MAX_FOLDS},
        )
    fold_of: Dict[int, str] = {}
    stats: List[FoldStat] = []
    names: List[str] = []
    for position, unit in enumerate(sorted(graph.units, key=lambda u: (-u.size, u.signature))):
        name = f"unit{position}"
        names.append(name)
        for member in unit.members:
            fold_of[member] = name
        stats.append(
            FoldStat(name=name, target=unit.size, count=unit.size, units=1, label_counts=dict(unit.label_counts))
        )
    return Assignment(fold_of=fold_of, folds=names, stats=stats, dropped=list(graph.dropped))


def describe(policy: Policy, data: Dataset, assignment: Assignment, graph: UnitGraph) -> List[FoldStat]:
    """Fold statistics derived from the finished assignment, not from how it was made.

    Everything here is recomputable from the policy, the data and the fold column
    alone, so a certificate's fold table can be re-derived by anyone verifying it.
    """
    from . import units as units_module

    members = assignment.fold_members()
    names = list(assignment.folds)
    assignable = sum(len(group) for group in members.values())
    scheme_names = policy.scheme.fold_names
    if scheme_names and set(names) == set(scheme_names):
        targets = _targets(policy, assignable, scheme_names)
    else:
        targets = {name: len(members[name]) for name in names}
    classes, _note = units_module.label_classes(policy, data)
    unit_of = graph.unit_of()
    stats: List[FoldStat] = []
    for name in names:
        indexes = members[name]
        labels: Dict[str, int] = {}
        if classes is not None:
            for index in indexes:
                bucket = classes[index]
                labels[bucket] = labels.get(bucket, 0) + 1
        stats.append(
            FoldStat(
                name=name,
                target=targets.get(name, len(indexes)),
                count=len(indexes),
                units=len({unit_of[index] for index in indexes if index in unit_of}),
                label_counts=labels,
            )
        )
    return stats


def from_column(data: Dataset, column: str) -> Assignment:
    values = data.column(column)
    fold_of: Dict[int, str] = {}
    counts: Dict[str, int] = {}
    for i, value in enumerate(values):
        fold = value if value is not None else DROPPED
        fold_of[i] = fold
        counts[fold] = counts.get(fold, 0) + 1
    folds = sorted(name for name in counts if name != DROPPED)
    stats = [
        FoldStat(name=name, target=counts[name], count=counts[name], units=0, label_counts={})
        for name in folds
    ]
    dropped = [i for i, fold in fold_of.items() if fold == DROPPED]
    return Assignment(fold_of=fold_of, folds=folds, stats=stats, dropped=dropped)
