from __future__ import annotations

import csv
import itertools

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from splitproof import api
from splitproof.policy import Policy


@st.composite
def _instances(draw):
    sizes = draw(st.lists(st.integers(1, 12), min_size=2, max_size=9))
    folds = draw(st.integers(2, 3))
    weights = draw(st.lists(st.integers(1, 9), min_size=folds, max_size=folds))
    seed = draw(st.integers(0, 10_000))
    return sizes, weights, seed


def _best_worst_deviation(sizes, targets):
    best = None
    for combo in itertools.product(range(len(targets)), repeat=len(sizes)):
        if len(set(combo)) < len(targets):
            continue
        counts = [0] * len(targets)
        for size, fold in zip(sizes, combo, strict=True):
            counts[fold] += size
        worst = max(abs(count - target) for count, target in zip(counts, targets, strict=True))
        best = worst if best is None else min(best, worst)
    return best


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(instance=_instances())
@example(instance=([3, 3, 3], [1, 1, 6], 0))
def test_every_fold_lands_within_one_unit_of_its_target_or_as_close_as_any_split_can(tmp_path_factory, instance):
    sizes, weights, seed = instance
    work = tmp_path_factory.mktemp("alloc")
    path = work / "d.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "grp"])
        for unit, size in enumerate(sizes):
            for member in range(size):
                writer.writerow([f"u{unit}m{member}", f"g{unit}"])
    names = ["train", "test", "valid"][: len(weights)]
    policy = Policy.from_dict(
        {
            "splitproof": 1,
            "dataset": {"path": str(path), "id": "id", "groups": ["grp"]},
            "scheme": {"kind": "holdout", "fractions": dict(zip(names, weights, strict=True))},
            "constraints": [{"kind": "disjoint", "key": "grp"}],
            "seed": seed,
        }
    )
    try:
        outcome = api.make_split(policy, skip_neighbours=True, tolerance=10.0)
    except Exception as error:
        assert getattr(error, "code", "") in {"SPLIT_INFEASIBLE", "GIANT_UNIT"}, error
        return
    largest = max(sizes)
    worst = max(abs(stat.count - stat.target) for stat in outcome.assignment.stats)
    if worst > largest:
        targets = [next(s.target for s in outcome.assignment.stats if s.name == name) for name in names]
        assert worst <= _best_worst_deviation(sizes, targets), (sizes, weights, outcome.assignment.stats)
    if len(sizes) >= len(names):
        assert all(stat.count > 0 for stat in outcome.assignment.stats), (sizes, weights)



def _cost(counts, targets):
    return round(sum(((counts[n] - targets[n]) / max(targets[n], 1)) ** 2 for n in targets), 9)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(instance=_instances())
def test_no_single_move_or_swap_improves_the_allocation(tmp_path_factory, instance):
    sizes, weights, seed = instance
    work = tmp_path_factory.mktemp("local")
    path = work / "d.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "grp"])
        for unit, size in enumerate(sizes):
            for member in range(size):
                writer.writerow([f"u{unit}m{member}", f"g{unit}"])
    names = ["train", "test", "valid"][: len(weights)]
    policy = Policy.from_dict(
        {
            "splitproof": 1,
            "dataset": {"path": str(path), "id": "id", "groups": ["grp"]},
            "scheme": {"kind": "holdout", "fractions": dict(zip(names, weights, strict=True))},
            "constraints": [{"kind": "disjoint", "key": "grp"}],
            "seed": seed,
        }
    )
    try:
        outcome = api.make_split(policy, skip_neighbours=True, tolerance=10.0)
    except Exception as error:
        assert getattr(error, "code", "") in {"SPLIT_INFEASIBLE", "GIANT_UNIT"}, error
        return
    targets = {stat.name: stat.target for stat in outcome.assignment.stats}
    counts = {stat.name: stat.count for stat in outcome.assignment.stats}
    members = {name: [] for name in names}
    fold_of = outcome.assignment.fold_of
    for unit, size in enumerate(sizes):
        first_row = sum(sizes[:unit])
        members[fold_of[first_row]].append(size)
    current = _cost(counts, targets)
    for source in names:
        for size in set(members[source]):
            for destination in names:
                if destination == source:
                    continue
                if len(members[source]) > 1:
                    moved = dict(counts)
                    moved[source] -= size
                    moved[destination] += size
                    assert _cost(moved, targets) >= current, ("move", size, source, destination, counts)
                for other in set(members[destination]):
                    swapped = dict(counts)
                    swapped[source] += other - size
                    swapped[destination] += size - other
                    assert _cost(swapped, targets) >= current, ("swap", size, other, counts)
