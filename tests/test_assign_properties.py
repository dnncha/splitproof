from __future__ import annotations

import csv
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from splitproof import api
from splitproof.policy import Policy


@st.composite
def _instances(draw):
    sizes = draw(st.lists(st.integers(1, 12), min_size=2, max_size=14))
    folds = draw(st.integers(2, 3))
    weights = draw(st.lists(st.integers(1, 9), min_size=folds, max_size=folds))
    seed = draw(st.integers(0, 10_000))
    return sizes, weights, seed


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(instance=_instances())
def test_every_fold_lands_within_one_unit_of_its_target(tmp_path_factory, instance):
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
    for stat in outcome.assignment.stats:
        assert abs(stat.count - stat.target) <= largest, (sizes, weights, stat)
    if len(sizes) >= len(names):
        assert all(stat.count > 0 for stat in outcome.assignment.stats), (sizes, weights)
