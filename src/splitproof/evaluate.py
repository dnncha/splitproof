from __future__ import annotations

import math

import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .assign import DROPPED, Assignment
from .dataset import Dataset
from .errors import VacuousConstraint
from .policy import DisjointConstraint, Policy, SimilarityConstraint, TemporalConstraint
from .sim import EXACT, Neighbour, SimilarityResult, resolve
from .units import UnitGraph, _constraint_id
from .units import label_kind as units_label_kind

SATISFIED = "satisfied"
SATISFIED_UNPROVEN = "satisfied_unproven"
VIOLATED = "violated"

MAX_EXAMPLES = 20


@dataclass(frozen=True)
class Coverage:
    """What a constraint check actually saw, from which its status is derived.

    No check states its own status. A clean result is `satisfied` only when the search
    was exhaustive and every row in a fold carried a value the check could use; any
    unbound row makes it `satisfied_unproven`, and a check that bound no row at all is
    refused, because it proved nothing.
    """

    violations: int
    exhaustive: bool
    bound: int
    unbound: int

    @property
    def status(self) -> str:
        if self.violations:
            return VIOLATED
        if self.exhaustive and not self.unbound:
            return SATISFIED
        return SATISFIED_UNPROVEN


@dataclass
class ConstraintReport:
    id: str
    kind: str
    description: str
    coverage: Coverage
    measured: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.coverage.bound == 0 and not self.coverage.violations:
            raise VacuousConstraint(
                f"constraint '{self.id}' bound no row in any fold, so a clean result would prove nothing",
                remedy=(
                    "Every row that carries a value for this key was dropped or left unassigned. "
                    "Check the key's column, and whether on_span: drop removed the rows it guards."
                ),
                details={"constraint": self.id, "unbound": self.coverage.unbound},
            )

    @property
    def status(self) -> str:
        return self.coverage.status

    @property
    def ok(self) -> bool:
        return self.status != VIOLATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "status": self.status,
            "measured": dict(sorted(self.measured.items())),
            "examples": self.examples,
            "notes": list(self.notes),
        }


def check(policy: Policy, data: Dataset, assignment: Assignment, graph: UnitGraph) -> List[ConstraintReport]:
    reports: List[ConstraintReport] = []
    for constraint in policy.similarity_constraints:
        reports.append(_check_similarity(constraint, data, assignment, graph))
    for disjoint in policy.disjoint_constraints:
        reports.append(_check_disjoint(disjoint, data, assignment))
    for temporal in policy.temporal_constraints:
        reports.append(_check_temporal(temporal, policy, data, assignment))
    return reports


def _check_similarity(
    constraint: SimilarityConstraint,
    data: Dataset,
    assignment: Assignment,
    graph: UnitGraph,
) -> ConstraintReport:
    identifier = _constraint_id(constraint)
    result = graph.similarity.get(identifier)
    fold_of = assignment.fold_of
    if result is not None and result.groups is not None:
        total, count, worst, worst_examples, by_pair = _crossing_classes(result.groups, data, assignment)
    else:
        count, worst, worst_examples, by_pair = _crossing(result, data, assignment)
        total = len(result.pairs) if result else 0
    measured = {
        "pairs_at_or_above_threshold": total,
        "cross_fold_pairs_at_or_above_threshold": count,
        "cross_fold_pairs_by_fold_pair": by_pair,
        "max_cross_fold_similarity": round(worst, 6) if worst is not None else None,
        "values_absent": (len(result.missing) - len(result.unparsable)) if result else 0,
        "values_unparsable": len(result.unparsable) if result else 0,
        "exactness": result.exactness if result else "unknown",
        "backend": result.backend if result else "unknown",
    }
    unparsable = len(result.unparsable) if result else 0
    absent = (len(result.missing) - unparsable) if result else 0
    unusable = set(result.missing) if result else set(range(len(data)))
    in_folds = [index for index, fold in fold_of.items() if fold != DROPPED]
    coverage = Coverage(
        violations=count,
        exhaustive=result is not None and result.exactness == EXACT and not unparsable,
        bound=sum(1 for index in in_folds if index not in unusable),
        unbound=absent,
    )
    examples = [
        {
            "a": data.ids[left],
            "b": data.ids[right],
            "similarity": round(sim, 6),
            "a_fold": fold_of.get(left),
            "b_fold": fold_of.get(right),
        }
        for left, right, sim in worst_examples
    ]
    notes = list(result.notes) if result else ["similarity was not computed"]
    if unparsable and not count:
        notes.append(
            f"{unparsable} values could not be parsed by this backend, so they were compared "
            "to nothing; the search was not exhaustive over the declared key and the result is "
            "evidence rather than proof"
        )
    if absent and not count:
        notes.append(
            f"{absent} entities have no value for this key, so no pair involving them was "
            "compared; those rows sit in a fold unconstrained and the result is evidence "
            "rather than proof"
        )
    if coverage.status == SATISFIED_UNPROVEN and not unparsable and not absent:
        notes.append(
            "no crossing pair was found, but this backend does not guarantee it found every pair, "
            "so the result is evidence rather than proof"
        )
    return ConstraintReport(
        id=identifier,
        kind="similarity",
        description=constraint.describe(),
        coverage=coverage,
        measured=measured,
        examples=examples,
        notes=notes,
    )


def _crossing_classes(
    groups: Sequence[Optional[str]],
    data: Dataset,
    assignment: Assignment,
) -> Tuple[int, int, Optional[float], List[Tuple[int, int, float]], Dict[str, int]]:
    """Count crossings for an equivalence-class backend, arithmetically.

    Members of a class are related by definition, so the implied pair count is
    quadratic in class size and must never be materialised. Crossings come from
    per-class fold histograms instead.
    """
    histogram: Dict[str, Dict[str, int]] = {}
    witness: Dict[str, Dict[str, int]] = {}
    for index in sorted(range(len(groups)), key=lambda i: data.ids[i]):
        label = groups[index]
        if label is None:
            continue
        fold = assignment.fold_of.get(index, DROPPED)
        if fold == DROPPED:
            continue
        histogram.setdefault(label, {})[fold] = histogram.setdefault(label, {}).get(fold, 0) + 1
        witness.setdefault(label, {}).setdefault(fold, index)
    total = 0
    crossing = 0
    by_pair: Dict[str, int] = {}
    examples: List[Tuple[int, int, float]] = []
    for label in sorted(histogram):
        folds = histogram[label]
        size = sum(folds.values())
        total += size * (size - 1) // 2
        within = sum(count * (count - 1) // 2 for count in folds.values())
        crossing += size * (size - 1) // 2 - within
        names = sorted(folds)
        for first in range(len(names)):
            for second in range(first + 1, len(names)):
                left, right = names[first], names[second]
                amount = folds[left] * folds[right]
                key = "|".join(sorted((left, right)))
                by_pair[key] = by_pair.get(key, 0) + amount
                first_member, second_member = sorted(
                    (witness[label][left], witness[label][right]), key=lambda i: data.ids[i]
                )
                examples.append((first_member, second_member, 1.0))
    examples.sort(key=lambda example: (data.ids[example[0]], data.ids[example[1]]))
    worst = 1.0 if crossing else None
    return total, crossing, worst, examples[:MAX_EXAMPLES], dict(sorted(by_pair.items()))


def _crossing(
    result: Optional[SimilarityResult],
    data: Dataset,
    assignment: Assignment,
) -> Tuple[int, Optional[float], List[Tuple[int, int, float]], Dict[str, int]]:
    """Count cross-fold pairs without ever materialising them.

    The pair graph is quadratic in the worst case, so this works on the numpy
    arrays behind the PairSet and keeps only the worst MAX_EXAMPLES.
    """
    if result is None or len(result.pairs) == 0:
        return 0, None, [], {}
    fold_numbers = {name: number for number, name in enumerate(assignment.folds)}
    lookup = np.full(len(data), -1, dtype=np.int64)
    for index, fold in assignment.fold_of.items():
        if fold != DROPPED and fold in fold_numbers:
            lookup[index] = fold_numbers[fold]
    rows, cols, sims = result.pairs.rows, result.pairs.cols, result.pairs.sims
    left, right = lookup[rows], lookup[cols]
    crossing = (left >= 0) & (right >= 0) & (left != right)
    count = int(crossing.sum())
    if count == 0:
        return 0, None, [], {}
    positions = np.flatnonzero(crossing)
    crossing_sims = sims[positions]
    worst = float(crossing_sims.max())
    rank = np.empty(len(data), dtype=np.int64)
    rank[np.argsort(np.asarray(data.ids, dtype=object), kind="stable")] = np.arange(len(data), dtype=np.int64)
    first_rank, second_rank = rank[rows[positions]], rank[cols[positions]]
    low_rank = np.minimum(first_rank, second_rank)
    high_rank = np.maximum(first_rank, second_rank)
    if len(positions) > MAX_EXAMPLES:
        floor = np.partition(crossing_sims, len(crossing_sims) - MAX_EXAMPLES)[len(crossing_sims) - MAX_EXAMPLES]
        contenders = np.flatnonzero(crossing_sims >= floor)
    else:
        contenders = np.arange(len(positions))
    order = np.lexsort((high_rank[contenders], low_rank[contenders], -crossing_sims[contenders]))
    chosen = contenders[order[:MAX_EXAMPLES]]
    by_rank = np.argsort(rank)
    examples = [
        (int(by_rank[low_rank[c]]), int(by_rank[high_rank[c]]), float(crossing_sims[c])) for c in chosen
    ]
    names = list(assignment.folds)
    pair_left, pair_right = left[positions], right[positions]
    encoded = np.minimum(pair_left, pair_right) * len(names) + np.maximum(pair_left, pair_right)
    codes, amounts = np.unique(encoded, return_counts=True)
    by_pair: Dict[str, int] = {
        f"{names[int(code) // len(names)]}|{names[int(code) % len(names)]}": int(amount)
        for code, amount in zip(codes, amounts, strict=True)
    }
    return count, worst, examples, dict(sorted(by_pair.items()))


def _check_disjoint(constraint: DisjointConstraint, data: Dataset, assignment: Assignment) -> ConstraintReport:
    values = data.column(constraint.key)
    spread: Dict[str, set[str]] = {}
    bound = 0
    unbound = 0
    for i, value in enumerate(values):
        fold = assignment.fold_of.get(i, DROPPED)
        if fold == DROPPED:
            continue
        if value is None:
            unbound += 1
            continue
        bound += 1
        spread.setdefault(value, set()).add(fold)
    offenders = sorted((value for value, folds in spread.items() if len(folds) > 1))
    notes = (
        [
            f"{unbound} rows in a fold have no value for '{constraint.key}', so nothing "
            "constrains where they sit and the result is evidence rather than proof"
        ]
        if unbound and not offenders
        else []
    )
    return ConstraintReport(
        id=f"disjoint:{constraint.key}",
        kind="disjoint",
        description=constraint.describe(),
        coverage=Coverage(violations=len(offenders), exhaustive=True, bound=bound, unbound=unbound),
        notes=notes,
        measured={
            "distinct_values": len(spread),
            "values_in_more_than_one_fold": len(offenders),
            "missing_values": sum(1 for value in values if value is None),
        },
        examples=[{"value": value, "folds": sorted(spread[value])} for value in offenders[:MAX_EXAMPLES]],
    )


def _check_temporal(
    constraint: TemporalConstraint,
    policy: Policy,
    data: Dataset,
    assignment: Assignment,
) -> ConstraintReport:
    column = policy.dataset.dates[constraint.key]
    if constraint.before not in assignment.folds:
        raise VacuousConstraint(
            f"the temporal constraint guards fold '{constraint.before}', which this split does "
            f"not contain, so it could never bind",
            remedy=(
                f"The split's folds are {sorted(assignment.folds)}. Set the constraint's "
                "'before' to the fold that must stay clean of post-cutoff data."
            ),
            details={
                "constraint": f"temporal:{constraint.key}",
                "guards": constraint.before,
                "folds": sorted(assignment.folds),
            },
        )
    dates = data.dates(column)
    wrong_side: List[Dict[str, Any]] = []
    missing = 0
    checked = 0
    for i, when in enumerate(dates):
        fold = assignment.fold_of.get(i, DROPPED)
        if fold == DROPPED:
            continue
        if when is None:
            missing += 1
            continue
        checked += 1
        if when >= constraint.cutoff and fold == constraint.before:
            wrong_side.append(
                {
                    "id": data.ids[i],
                    "date": when.isoformat(),
                    "fold": fold,
                    "problem": f"dated on or after the cutoff but placed in '{constraint.before}'",
                }
            )
    return ConstraintReport(
        id=f"temporal:{constraint.key}@{constraint.cutoff.isoformat()}",
        kind="temporal",
        description=constraint.describe(),
        coverage=Coverage(violations=len(wrong_side), exhaustive=True, bound=checked, unbound=missing),
        measured={
            "cutoff": constraint.cutoff.isoformat(),
            "items_checked": checked,
            "items_after_the_cutoff_in_train": len(wrong_side),
            "items_without_a_date": missing,
            "on_span": constraint.on_span,
        },
        examples=sorted(wrong_side, key=lambda item: item["id"])[:MAX_EXAMPLES],
        notes=(
            [
                f"{missing} items have no date and were left where the scheme put them, so "
                "the result is evidence rather than proof"
            ]
            if missing and not wrong_side
            else []
        ),
    )


def reference_fold(assignment: Assignment) -> Optional[str]:
    """The fold held-out items are compared against, or None for cross-validation.

    Returning the largest fold when there is no 'train' made the cross-validation
    branch in diagnostics() unreachable, so a k-fold run compared every fold against
    one arbitrary fold instead of against the rest.
    """
    return "train" if "train" in assignment.folds else None





def diagnostics(
    policy: Policy,
    data: Dataset,
    assignment: Assignment,
    *,
    exact_max_n: int = 20000,
    skip_neighbours: bool = False,
    backend_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    members = assignment.fold_members()
    anchor = reference_fold(assignment)
    held_out_against_rest = anchor not in members
    out: Dict[str, Any] = {
        "reference_fold": "each fold against the rest" if held_out_against_rest else anchor,
        "keys": {},
    }
    if skip_neighbours or not policy.similarity_constraints or not members:
        return out
    if held_out_against_rest:
        probes = [
            (group, [i for name, other in members.items() if name != fold for i in other])
            for fold, group in sorted(members.items())
        ]
    else:
        targets = members.get(anchor, []) if anchor is not None else []
        queries = [index for fold, group in members.items() if fold != anchor for index in group]
        probes = [(queries, targets)]
    probes = [(q, t) for q, t in probes if q and t]
    if not probes:
        return out
    queries = [index for group, _ in probes for index in group]
    label_values = data.column(policy.dataset.label_column) if policy.dataset.label_column else None
    kind = units_label_kind(policy, label_values) if label_values is not None else None
    for constraint in policy.similarity_constraints:
        key = policy.dataset.similarity[constraint.key]
        backend = resolve(
            constraint.metric,
            key.kind,
            len(data),
            requested=(backend_overrides or {}).get(constraint.key) or constraint.backend,
            exact_max_n=exact_max_n,
            settings=constraint.settings,
        )
        neighbours: Dict[int, Optional[Neighbour]] = {}
        column = data.column(key.column)
        for probe_queries, probe_targets in probes:
            neighbours.update(
                backend.nearest(column, key.kind, probe_queries, probe_targets, data.ids)
            )
        sims = [n.similarity for n in neighbours.values() if n is not None]
        entry: Dict[str, Any] = {
            "metric": constraint.metric,
            "backend": backend.name,
            "threshold": constraint.threshold,
            "queries": len(queries),
            "with_a_neighbour": len(sims),
            "nearest_reference_similarity": _distribution(sims),
            "at_or_above_threshold": sum(1 for value in sims if value >= constraint.threshold),
        }
        if label_values is not None and kind is not None:
            entry["nearest_neighbour_baseline"] = _nn_baseline(
                neighbours, label_values, kind, queries
            )
        out["keys"][_constraint_id(constraint)] = entry
    return out


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return round(float(ordered[position]), 6)

    return {
        "min": round(float(ordered[0]), 6),
        "p50": pick(0.5),
        "p90": pick(0.9),
        "p99": pick(0.99),
        "max": round(float(ordered[-1]), 6),
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def _nn_baseline(
    neighbours: Dict[int, Optional[Neighbour]],
    label_values: Sequence[Optional[str]],
    kind: str,
    queries: Sequence[int],
) -> Dict[str, Any]:
    predicted: List[float] = []
    truth: List[float] = []
    correct = 0
    compared = 0
    for query in queries:
        neighbour = neighbours.get(query)
        actual = label_values[query]
        if neighbour is None or actual is None:
            continue
        guess = label_values[neighbour.index]
        if guess is None:
            continue
        if kind == "numeric":
            try:
                guess_value, actual_value = float(guess), float(actual)
            except ValueError:
                continue
            if not (math.isfinite(guess_value) and math.isfinite(actual_value)):
                continue
            predicted.append(guess_value)
            truth.append(actual_value)
            compared += 1
            continue
        compared += 1
        if guess == actual:
            correct += 1
    if compared == 0:
        return {"compared": 0, "note": "no held-out item had a labelled nearest neighbour"}
    if kind == "numeric":
        return {
            "compared": compared,
            "metric": "spearman",
            "value": _spearman(predicted, truth),
            "note": (
                "what a nearest-neighbour copy of the reference fold scores. Similarity reasoning "
                "is a legitimate model, so this is neither a floor nor a ceiling: a model that "
                "does not clearly beat it has not shown it learned anything beyond similarity, "
                "and beating it is not on its own evidence that the split is sound"
            ),
        }
    return {
        "compared": compared,
        "metric": "accuracy",
        "value": round(correct / compared, 6),
        "note": (
            "what a nearest-neighbour copy of the reference fold scores. Similarity reasoning "
            "is a legitimate model, so this is neither a floor nor a ceiling: a model that "
            "does not clearly beat it has not shown it learned anything beyond similarity, "
            "and beating it is not on its own evidence that the split is sound"
        ),
    }


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 3:
        return None
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    try:
        from scipy.stats import spearmanr
    except ModuleNotFoundError:
        return None
    value = spearmanr(list(left), list(right)).statistic
    if value != value:
        return None
    return round(float(value), 6)
