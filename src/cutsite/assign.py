from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import Dataset
from .errors import GiantUnit, InfeasibleSplit, PolicyError
from .policy import Policy
from .units import Unit, UnitGraph

DROPPED = "__dropped__"
MAX_FOLDS = 1000
LABEL_WEIGHT = 0.25


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
    payload = f"{seed}:{unit.key}:{unit.size}".encode("utf-8")
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
    if policy.temporal_constraints and policy.scheme.kind != "holdout":
        raise PolicyError(
            "a temporal constraint only makes sense with a holdout scheme",
            remedy="Use scheme.kind: holdout with named train and test fractions.",
            details={"scheme": policy.scheme.kind},
        )
    for constraint in policy.temporal_constraints:
        for side in (constraint.before, constraint.after):
            if side not in fold_names:
                raise PolicyError(
                    f"temporal constraint names fold '{side}', which the scheme does not define",
                    remedy=f"Scheme folds are {sorted(fold_names)}. Rename, or add the fraction.",
                    details={"fold": side, "available": sorted(fold_names)},
                )

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

    for unit in sorted(floating, key=lambda u: (-u.size, _tiebreak(policy.seed, u))):
        choice = _best_fold(unit, fold_names, targets, counts, label_counts)
        _place(unit, choice, fold_of, counts, unit_counts, label_counts)

    empty = [name for name in fold_names if counts[name] == 0]
    if empty:
        raise InfeasibleSplit(
            f"fold(s) {empty} came out empty",
            remedy=(
                "There are fewer independent units than folds. Lower the similarity threshold, "
                "reduce the number of folds, or use scheme.kind: leave-unit-out."
            ),
            details={"empty_folds": empty, "units": len(graph.units), "folds": list(fold_names)},
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


def _best_fold(
    unit: Unit,
    fold_names: Sequence[str],
    targets: Dict[str, int],
    counts: Dict[str, int],
    label_counts: Dict[str, Dict[str, int]],
) -> str:
    best_name = fold_names[0]
    best_score = None
    for name in fold_names:
        target = max(targets[name], 1)
        occupancy = (counts[name] + unit.size) / target
        penalty = 0.0
        if unit.label_counts:
            for label, amount in unit.label_counts.items():
                current = label_counts[name].get(label, 0)
                penalty += (current + amount) / target
            penalty /= len(unit.label_counts)
        score = occupancy + LABEL_WEIGHT * penalty
        if best_score is None or score < best_score - 1e-12:
            best_score, best_name = score, name
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
    for position, unit in enumerate(sorted(graph.units, key=lambda u: (-u.size, u.key))):
        name = f"unit{position}"
        names.append(name)
        for member in unit.members:
            fold_of[member] = name
        stats.append(
            FoldStat(name=name, target=unit.size, count=unit.size, units=1, label_counts=dict(unit.label_counts))
        )
    return Assignment(fold_of=fold_of, folds=names, stats=stats, dropped=list(graph.dropped))


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
