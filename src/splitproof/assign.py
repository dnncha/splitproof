from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dataset import Dataset
from .errors import DataError
from .errors import GiantUnit, InfeasibleSplit
from .policy import DROPPED, MAX_FOLDS, UNASSIGNED, Policy
from .units import Unit, UnitGraph



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
        _place(unit, str(unit.pinned), fold_of, counts, unit_counts, label_counts)

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

    _fill_empty_folds(fold_names, ordered, placement, fold_of, counts, unit_counts, label_counts)
    _improve(fold_names, ordered, placement, fold_of, counts, unit_counts, label_counts, targets, shares)

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

    return Assignment(fold_of=fold_of, folds=list(fold_names), stats=[], dropped=list(graph.dropped))


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
        _relocate(choice, name, ordered, placement, fold_of, counts, unit_counts, label_counts)


def _relocate(
    position: int,
    destination: str,
    ordered: Sequence[Unit],
    placement: Dict[int, str],
    fold_of: Dict[int, str],
    counts: Dict[str, int],
    unit_counts: Dict[str, int],
    label_counts: Dict[str, Dict[str, int]],
) -> None:
    unit = ordered[position]
    source = placement[position]
    for member in unit.members:
        fold_of[member] = destination
    counts[source] -= unit.size
    unit_counts[source] -= 1
    counts[destination] += unit.size
    unit_counts[destination] += 1
    for label, amount in unit.label_counts.items():
        label_counts[source][label] -= amount
        label_counts[destination][label] = label_counts[destination].get(label, 0) + amount
    placement[position] = destination


def _size_cost(counts: Dict[str, int], targets: Dict[str, int]) -> float:
    return round(sum(((counts[name] - targets[name]) / max(targets[name], 1)) ** 2 for name in targets), 9)


def _label_cost(
    label_counts: Dict[str, Dict[str, int]], targets: Dict[str, int], shares: Dict[str, float]
) -> float:
    return round(
        sum(
            abs(label_counts[name].get(label, 0) / max(targets[name], 1) - share)
            for name in targets
            for label, share in shares.items()
        ),
        9,
    )


Bundle = Tuple[int, Tuple[int, ...]]


def _bundles(pool: Dict[int, List[int]]) -> List[Bundle]:
    """Every distinct way to take nothing, one unit or two units of given sizes out of a fold.

    Units of one size are interchangeable for fold sizes, so bundles are built over
    sizes rather than units, which bounds them by the number of distinct sizes.
    """
    sizes = sorted(size for size, members in pool.items() if members)
    out: List[Bundle] = [(0, ())]
    for index, size in enumerate(sizes):
        out.append((size, (size,)))
        if len(pool[size]) >= 2:
            out.append((2 * size, (size, size)))
        for other in sizes[index + 1 :]:
            out.append((size + other, (size, other)))
    return sorted(out)


def _improve(
    fold_names: Sequence[str],
    ordered: Sequence[Unit],
    placement: Dict[int, str],
    fold_of: Dict[int, str],
    counts: Dict[str, int],
    unit_counts: Dict[str, int],
    label_counts: Dict[str, Dict[str, int]],
    targets: Dict[str, int],
    shares: Dict[str, float],
) -> None:
    """Repair what the greedy pass misses: first fold sizes, then label balance.

    The greedy pass places the largest units first and never revisits them, so it
    can miss a packing that exists: sizes 13, 13, 13, 8, 3, 2, 2, 2, 1 at 7:2:7 put
    13 in a fold whose target is 7. It also balances labels only when two folds tie
    exactly on size, which almost never happens, so 20 singletons of two labels came
    out 7:3 in one fold. Sizes are repaired first and labels only by swapping units of
    equal size, so requested fractions are never traded for label balance.
    """
    _improve_sizes(fold_names, ordered, placement, fold_of, counts, unit_counts, label_counts, targets)
    if shares:
        _improve_labels(fold_names, ordered, placement, fold_of, counts, unit_counts, label_counts, targets, shares)


def _improve_sizes(
    fold_names: Sequence[str],
    ordered: Sequence[Unit],
    placement: Dict[int, str],
    fold_of: Dict[int, str],
    counts: Dict[str, int],
    unit_counts: Dict[str, int],
    label_counts: Dict[str, Dict[str, int]],
    targets: Dict[str, int],
) -> None:
    """Exchange bundles of up to two units between folds while fold-size error strictly falls.

    Either side of an exchange may be empty. For each bundle the ideal partner size
    has a closed form, so only the two nearest sizes are tried, and the chosen
    exchange is repeated while it keeps helping. Every accepted step lowers the
    cost, so the loop ends, and no step empties a fold.
    """
    pools: Dict[str, Dict[int, List[int]]] = {name: {} for name in fold_names}
    for position in sorted(range(len(ordered)), reverse=True):
        pools[placement[position]].setdefault(ordered[position].size, []).append(position)

    def cost_after(first: str, second: str, out_sizes: Tuple[int, ...], in_sizes: Tuple[int, ...]) -> Optional[float]:
        if unit_counts[first] - len(out_sizes) + len(in_sizes) < 1:
            return None
        if unit_counts[second] - len(in_sizes) + len(out_sizes) < 1:
            return None
        for sizes, fold in ((out_sizes, first), (in_sizes, second)):
            for size in set(sizes):
                if len(pools[fold].get(size, [])) < sizes.count(size):
                    return None
        trial = dict(counts)
        trial[first] += sum(in_sizes) - sum(out_sizes)
        trial[second] += sum(out_sizes) - sum(in_sizes)
        return _size_cost(trial, targets)

    def apply(first: str, second: str, out_sizes: Tuple[int, ...], in_sizes: Tuple[int, ...]) -> None:
        moves = [(pools[first][size].pop(), second) for size in out_sizes]
        moves += [(pools[second][size].pop(), first) for size in in_sizes]
        for position, destination in moves:
            _relocate(position, destination, ordered, placement, fold_of, counts, unit_counts, label_counts)
            pools[destination].setdefault(ordered[position].size, []).append(position)

    while True:
        bundles = {name: _bundles(pools[name]) for name in fold_names}
        sizes = {name: [size for size, _ in bundles[name]] for name in fold_names}
        best = _size_cost(counts, targets)
        choice: Optional[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]]] = None
        for first in fold_names:
            for second in fold_names:
                if first >= second:
                    continue
                weight_first = 1.0 / max(targets[first], 1) ** 2
                weight_second = 1.0 / max(targets[second], 1) ** 2
                ideal_shift = (
                    weight_first * (counts[first] - targets[first])
                    - weight_second * (counts[second] - targets[second])
                ) / (weight_first + weight_second)
                for out_size, out_sizes in bundles[first]:
                    spot = bisect.bisect_left(sizes[second], out_size - ideal_shift)
                    for nearest in (spot - 1, spot):
                        if not 0 <= nearest < len(bundles[second]):
                            continue
                        in_sizes = bundles[second][nearest][1]
                        if not out_sizes and not in_sizes:
                            continue
                        cost = cost_after(first, second, out_sizes, in_sizes)
                        if cost is not None and cost < best:
                            best, choice = cost, (first, second, out_sizes, in_sizes)
        if choice is None:
            return
        while True:
            apply(*choice)
            following = cost_after(*choice)
            if following is None or following >= best:
                break
            best = following


def _improve_labels(
    fold_names: Sequence[str],
    ordered: Sequence[Unit],
    placement: Dict[int, str],
    fold_of: Dict[int, str],
    counts: Dict[str, int],
    unit_counts: Dict[str, int],
    label_counts: Dict[str, Dict[str, int]],
    targets: Dict[str, int],
    shares: Dict[str, float],
) -> None:
    """Swap equal-size units between folds while label drift strictly falls.

    A swap of equal sizes leaves every fold size unchanged, so this cannot undo the
    size repair. For each pair of folds and each size, the unit whose labels lean
    furthest toward the first fold's surplus is swapped with the one leaning furthest
    the other way, found in one scan, and the swap is kept only if the exact drift
    falls. Every accepted swap lowers the drift, so the loop ends.
    """
    LabelShape = Tuple[Tuple[str, int], ...]
    pools: Dict[str, Dict[int, Dict[LabelShape, List[int]]]] = {name: {} for name in fold_names}

    def shape_of(position: int) -> LabelShape:
        return tuple(sorted(ordered[position].label_counts.items()))

    def store(position: int) -> None:
        unit = ordered[position]
        pools[placement[position]].setdefault(unit.size, {}).setdefault(shape_of(position), []).append(position)

    for position in sorted(range(len(ordered)), reverse=True):
        if ordered[position].label_counts:
            store(position)

    def surplus(name: str) -> Dict[str, float]:
        target = max(targets[name], 1)
        return {label: label_counts[name].get(label, 0) / target - share for label, share in shares.items()}

    def lean(shape: LabelShape, direction: Dict[str, float]) -> float:
        return sum(amount * direction.get(label, 0.0) for label, amount in shape)

    def swap(out_position: int, in_position: int) -> None:
        source, destination = placement[out_position], placement[in_position]
        for position in (out_position, in_position):
            members = pools[placement[position]][ordered[position].size][shape_of(position)]
            if members[-1] == position:
                members.pop()
            else:
                members.remove(position)
        _relocate(out_position, destination, ordered, placement, fold_of, counts, unit_counts, label_counts)
        _relocate(in_position, source, ordered, placement, fold_of, counts, unit_counts, label_counts)
        store(out_position)
        store(in_position)

    def candidate(first: str, second: str, size: int) -> Optional[Tuple[int, int]]:
        over_first, over_second = surplus(first), surplus(second)
        direction = {label: over_first[label] - over_second[label] for label in shares}
        leaving = [shape for shape, members in pools[first].get(size, {}).items() if members]
        arriving = [shape for shape, members in pools[second].get(size, {}).items() if members]
        if not leaving or not arriving:
            return None
        out_shape = max(leaving, key=lambda shape: (lean(shape, direction), shape))
        in_shape = min(arriving, key=lambda shape: (lean(shape, direction), shape))
        if out_shape == in_shape:
            return None
        return pools[first][size][out_shape][-1], pools[second][size][in_shape][-1]

    def drift_after(pair: Tuple[int, int]) -> float:
        out_position, in_position = pair
        swap(out_position, in_position)
        cost = _label_cost(label_counts, targets, shares)
        swap(out_position, in_position)
        return cost

    while True:
        best = _label_cost(label_counts, targets, shares)
        choice: Optional[Tuple[str, str, int, Tuple[int, int]]] = None
        for first in fold_names:
            for second in fold_names:
                if first >= second:
                    continue
                for size in sorted(set(pools[first]) & set(pools[second])):
                    pair = candidate(first, second, size)
                    if pair is None:
                        continue
                    cost = drift_after(pair)
                    if cost < best:
                        best, choice = cost, (first, second, size, pair)
        if choice is None:
            return
        first, second, size, pair = choice
        while pair is not None:
            swap(*pair)
            pair = candidate(first, second, size)
            if pair is None:
                break
            following = drift_after(pair)
            if following >= best:
                break
            best = following


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


def fold_order(name: str) -> Tuple[Any, ...]:
    """Natural ordering, so fold2 sorts before fold10.

    assign() creates folds in integer order while a fold column is read back
    lexicographically, which made a split fail verification against its own
    certificate once there were more than ten folds.
    """
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts) + (name,)




def check_audit_folds(
    data: Dataset, fold_of: Dict[int, str], source: str, *, allow_dropped: bool = False
) -> None:
    """An audited split must say where every entity went.

    Each of these silently excluded rows from every check and still certified the
    rest clean: a blank cell, the internal __dropped__ sentinel, and a split with
    only one fold (which trivially has no crossing pairs).
    """
    blank = sorted(data.ids[i] for i, fold in fold_of.items() if fold == UNASSIGNED)
    if blank:
        raise DataError(
            f"{len(blank)} entities have no fold in {source}",
            remedy=(
                "Every entity needs a fold, or the audit would grade only the rows that have "
                "one and report the rest as clean. Fill them in, or remove those rows."
            ),
            details={"missing": len(blank), "examples": blank[:10]},
        )
    reserved = sorted(data.ids[i] for i, fold in fold_of.items() if fold == DROPPED)
    if reserved and not allow_dropped:
        raise DataError(
            f"{source} uses the reserved fold name {DROPPED!r}",
            remedy=(
                "That name is internal to splitproof, which uses it for rows a policy drops. "
                "Rename the fold."
            ),
            details={"examples": reserved[:10]},
        )
    folds = {fold for fold in fold_of.values() if fold != DROPPED}
    if "" in folds:
        raise DataError(
            f"{source} contains an empty fold name",
            remedy="Fold names must be non-empty.",
            details={"folds": sorted(folds)},
        )
    if len(folds) < 2:
        raise DataError(
            f"{source} puts every entity in one fold ({sorted(folds)})",
            remedy=(
                "A single fold has no pairs that cross it, so every constraint would pass "
                "without measuring anything. An audit needs at least two folds."
            ),
            details={"folds": sorted(folds), "entities": len(data)},
        )


def from_column(data: Dataset, column: str, *, allow_dropped: bool = False) -> Assignment:
    values = data.column(column)
    fold_of: Dict[int, str] = {}
    counts: Dict[str, int] = {}
    for i, value in enumerate(values):
        fold = value if value is not None else UNASSIGNED
        fold_of[i] = fold
        counts[fold] = counts.get(fold, 0) + 1
    check_audit_folds(data, fold_of, f"column '{column}'", allow_dropped=allow_dropped)
    folds = sorted(
        (name for name in counts if name not in {DROPPED, UNASSIGNED}), key=fold_order
    )
    stats = [
        FoldStat(name=name, target=counts[name], count=counts[name], units=0, label_counts={})
        for name in folds
    ]
    dropped = sorted(i for i, fold in fold_of.items() if fold == DROPPED)
    return Assignment(fold_of=fold_of, folds=folds, stats=stats, dropped=dropped)
