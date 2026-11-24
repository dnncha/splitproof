from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pytest

from splitproof import api
from splitproof.errors import VacuousConstraint
from splitproof.evaluate import SATISFIED, SATISFIED_UNPROVEN, VIOLATED, ConstraintReport, Coverage
from splitproof.policy import Policy


def _write(path: Path, rows: List[Dict[str, str]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _policy(path: Path, constraints: List[Dict]) -> Policy:
    return Policy.from_dict(
        {
            "splitproof": 1,
            "dataset": {"path": str(path), "id": "id", "groups": ["grp"], "dates": {"d": "date"}},
            "scheme": {"kind": "holdout", "fractions": {"train": 0.7, "test": 0.3}},
            "constraints": constraints,
            "seed": 1,
        }
    )


def _sparse(tmp_path: Path) -> Path:
    rows = [
        {"id": "a", "grp": "g1", "date": "2020-01-01", "fold": "train"},
        {"id": "b", "grp": "g2", "date": "2022-01-01", "fold": "test"},
    ]
    rows += [{"id": f"u{i}", "grp": "", "date": "", "fold": "train" if i % 2 else "test"} for i in range(8)]
    return _write(tmp_path / "sparse.csv", rows)


@pytest.mark.parametrize(
    "constraint",
    [{"kind": "disjoint", "key": "grp"}, {"kind": "temporal", "key": "d", "cutoff": "2021-01-01"}],
)
def test_a_constraint_with_rows_it_could_not_bind_is_never_proven(tmp_path, constraint):
    path = _sparse(tmp_path)
    for outcome in (
        api.make_split(_policy(path, [constraint]), skip_neighbours=True),
        api.audit(_policy(path, [constraint]), split_column="fold", skip_neighbours=True),
    ):
        assert [report.status for report in outcome.reports] == [SATISFIED_UNPROVEN]
        assert outcome.certificate["status"] == SATISFIED_UNPROVEN
        assert "evidence rather than proof" in " ".join(outcome.reports[0].notes)


def test_a_drop_that_removes_every_bindable_row_is_refused(tmp_path):
    rows = [
        {"id": "a", "grp": "g", "date": "2020-01-01"},
        {"id": "b", "grp": "g", "date": "2022-01-01"},
    ]
    rows += [{"id": f"u{i}", "grp": "", "date": ""} for i in range(8)]
    path = _write(tmp_path / "drop.csv", rows)
    policy = _policy(
        path,
        [
            {"kind": "disjoint", "key": "grp"},
            {"kind": "temporal", "key": "d", "cutoff": "2021-01-01", "on_span": "drop"},
        ],
    )
    with pytest.raises(VacuousConstraint):
        api.make_split(policy, skip_neighbours=True)


def test_status_is_derived_and_cannot_be_stated():
    assert Coverage(violations=1, exhaustive=False, bound=1, unbound=9).status == VIOLATED
    assert Coverage(violations=0, exhaustive=True, bound=10, unbound=0).status == SATISFIED
    assert Coverage(violations=0, exhaustive=True, bound=9, unbound=1).status == SATISFIED_UNPROVEN
    assert Coverage(violations=0, exhaustive=False, bound=10, unbound=0).status == SATISFIED_UNPROVEN
    with pytest.raises(TypeError):
        ConstraintReport(id="x", kind="disjoint", description="x", status=SATISFIED)  # type: ignore[call-arg]


def test_verify_refuses_a_certificate_that_drops_rows_the_policy_does_not(tmp_path, monkeypatch):
    from splitproof import assign, cert, dataset, evaluate, units
    from splitproof.errors import DataError

    monkeypatch.chdir(tmp_path)
    rows = [
        {"id": "r0", "grp": "g0", "date": "2020-01-01", "fold": "train"},
        {"id": "r1", "grp": "g0", "date": "2020-02-01", "fold": "train"},
        {"id": "r2", "grp": "g1", "date": "2022-01-01", "fold": "test"},
        {"id": "r3", "grp": "g1", "date": "2022-02-01", "fold": "test"},
        {"id": "r4", "grp": "g2", "date": "2020-03-01", "fold": "train"},
        {"id": "r5", "grp": "g2", "date": "2020-04-01", "fold": "__dropped__"},
    ]
    _write(tmp_path / "data.csv", rows)
    policy = Policy.from_dict(
        {
            "splitproof": 1,
            "dataset": {"path": "data.csv", "id": "id", "groups": ["grp"], "dates": {"d": "date"}},
            "scheme": {"kind": "holdout", "fractions": {"train": 0.7, "test": 0.3}},
            "constraints": [
                {"kind": "disjoint", "key": "grp"},
                {"kind": "temporal", "key": "d", "cutoff": "2021-01-01", "on_span": "drop"},
            ],
            "seed": 1,
        }
    )
    data = dataset.load_for(policy, None)
    forged = api.load_assignment(data, split_column="fold", allow_dropped=True)
    graph = units.build(policy, data)
    forged.stats = assign.describe(policy, data, forged, graph)
    reports = evaluate.check(policy, data, forged, graph)
    certificate = cert.build(
        policy, data, graph, forged, reports, evaluate.diagnostics(policy, data, forged, skip_neighbours=True)
    )
    cert.write(tmp_path / "forged.json", certificate)
    with pytest.raises(DataError) as error:
        api.verify(tmp_path / "forged.json", split_column="fold", skip_neighbours=True)
    assert error.value.details["dropped_without_cause"] == ["r5"]
