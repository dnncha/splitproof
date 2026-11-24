from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .assign import DROPPED, Assignment
from .dataset import Dataset
from .policy import DisjointConstraint, Policy, SimilarityConstraint, TemporalConstraint
from .sim import EXACT, resolve
from .units import UnitGraph, _constraint_id

SATISFIED = "satisfied"
SATISFIED_UNPROVEN = "satisfied_unproven"
VIOLATED = "violated"

MAX_EXAMPLES = 20


@dataclass
class ConstraintReport:
    id: str
    kind: str
    description: str
    status: str
    measured: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

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
    for constraint in policy.disjoint_constraints:
        reports.append(_check_disjoint(constraint, data, assignment))
    for constraint in policy.temporal_constraints:
        reports.append(_check_temporal(constraint, policy, data, assignment))
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
    crossing: List[Tuple[int, int, float]] = []
    if result is not None:
        for left, right, sim in result.pairs:
            left_fold, right_fold = fold_of.get(left, DROPPED), fold_of.get(right, DROPPED)
            if left_fold == DROPPED or right_fold == DROPPED:
                continue
            if left_fold != right_fold:
                crossing.append((left, right, sim))
    worst = max((sim for _l, _r, sim in crossing), default=None)
    measured = {
        "pairs_at_or_above_threshold": len(result.pairs) if result else 0,
        "cross_fold_pairs_at_or_above_threshold": len(crossing),
        "max_cross_fold_similarity": round(worst, 6) if worst is not None else None,
        "exactness": result.exactness if result else "unknown",
        "backend": result.backend if result else "unknown",
    }
    if crossing:
        status = VIOLATED
    elif result is not None and result.exactness == EXACT:
        status = SATISFIED
    else:
        status = SATISFIED_UNPROVEN
    examples = [
        {
            "a": data.ids[left],
            "b": data.ids[right],
            "similarity": round(sim, 6),
            "a_fold": fold_of.get(left),
            "b_fold": fold_of.get(right),
        }
        for left, right, sim in sorted(crossing, key=lambda item: -item[2])[:MAX_EXAMPLES]
    ]
    notes = list(result.notes) if result else ["similarity was not computed"]
    if status == SATISFIED_UNPROVEN:
        notes.append(
            "no crossing pair was found, but this backend does not guarantee it found every pair, "
            "so the result is evidence rather than proof"
        )
    return ConstraintReport(
        id=identifier,
        kind="similarity",
        description=constraint.describe(),
        status=status,
        measured=measured,
        examples=examples,
        notes=notes,
    )


def _check_disjoint(constraint: DisjointConstraint, data: Dataset, assignment: Assignment) -> ConstraintReport:
    values = data.column(constraint.key)
    spread: Dict[str, set] = {}
    for i, value in enumerate(values):
        if value is None:
            continue
        fold = assignment.fold_of.get(i, DROPPED)
        if fold == DROPPED:
            continue
        spread.setdefault(value, set()).add(fold)
    offenders = sorted((value for value, folds in spread.items() if len(folds) > 1))
    return ConstraintReport(
        id=f"disjoint:{constraint.key}",
        kind="disjoint",
        description=constraint.describe(),
        status=VIOLATED if offenders else SATISFIED,
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
    dates = data.dates(column)
    wrong_side: List[Dict[str, Any]] = []
    missing = 0
    for i, when in enumerate(dates):
        fold = assignment.fold_of.get(i, DROPPED)
        if fold == DROPPED:
            continue
        if when is None:
            missing += 1
            continue
        expected = constraint.before if when < constraint.cutoff else constraint.after
        if fold in (constraint.before, constraint.after) and fold != expected:
            wrong_side.append({"id": data.ids[i], "date": when.isoformat(), "fold": fold, "expected": expected})
    return ConstraintReport(
        id=f"temporal:{constraint.key}@{constraint.cutoff.isoformat()}",
        kind="temporal",
        description=constraint.describe(),
        status=VIOLATED if wrong_side else SATISFIED,
        measured={
            "cutoff": constraint.cutoff.isoformat(),
            "items_on_wrong_side": len(wrong_side),
            "items_without_a_date": missing,
            "on_span": constraint.on_span,
        },
        examples=wrong_side[:MAX_EXAMPLES],
        notes=(
            [f"{missing} items have no date and were left where the scheme put them"] if missing else []
        ),
    )


def reference_fold(assignment: Assignment) -> Optional[str]:
    if "train" in assignment.folds:
        return "train"
    if not assignment.stats:
        return None
    return max(assignment.stats, key=lambda stat: stat.count).name


def _label_kind(policy: Policy, values: Sequence[Optional[str]]) -> str:
    if policy.dataset.label_kind != "auto":
        return policy.dataset.label_kind
    present = [value for value in values if value is not None]
    if not present:
        return "categorical"
    try:
        numbers = [float(value) for value in present]
    except ValueError:
        return "categorical"
    return "numeric" if len(set(numbers)) > 10 else "categorical"


def diagnostics(
    policy: Policy,
    data: Dataset,
    assignment: Assignment,
    *,
    exact_max_n: int = 20000,
    skip_neighbours: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"reference_fold": reference_fold(assignment), "keys": {}}
    anchor = out["reference_fold"]
    if anchor is None or skip_neighbours or not policy.similarity_constraints:
        return out
    members = assignment.fold_members()
    targets = members.get(anchor, [])
    queries = [index for fold, group in members.items() if fold != anchor for index in group]
    if not targets or not queries:
        return out
    label_values = data.column(policy.dataset.label_column) if policy.dataset.label_column else None
    kind = _label_kind(policy, label_values) if label_values is not None else None
    for constraint in policy.similarity_constraints:
        key = policy.dataset.similarity[constraint.key]
        backend = resolve(
            constraint.metric,
            key.kind,
            len(data),
            requested=constraint.backend,
            exact_max_n=exact_max_n,
        )
        neighbours = backend.nearest(data.column(key.column), key.kind, queries, targets)
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
    neighbours: Dict[int, Any],
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
                "a nearest-neighbour copy of the reference fold scores this much; a model that does "
                "not clearly beat it has learned nothing beyond similarity"
            ),
        }
    return {
        "compared": compared,
        "metric": "accuracy",
        "value": round(correct / compared, 6),
        "note": (
            "a nearest-neighbour copy of the reference fold scores this much; a model that does "
            "not clearly beat it has learned nothing beyond similarity"
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
