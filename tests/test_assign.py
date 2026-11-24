from __future__ import annotations

from pathlib import Path

import pytest

from cutsite import api, dataset as dataset_module, units as units_module
from cutsite.errors import VacuousConstraint
from cutsite.policy import Policy

from conftest import make_rows, write_csv


def test_two_equal_units_fill_both_folds(policy_dict, tmp_path: Path):
    rows = make_rows(families=2, members=3, seed=51)
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "pair.csv", rows))
    policy_dict["scheme"] = {"kind": "holdout", "fractions": {"train": 0.6, "test": 0.4}}
    outcome = api.make_split(Policy.from_dict(policy_dict))
    counts = {stat.name: stat.count for stat in outcome.assignment.stats}
    assert counts == {"train": 3, "test": 3}


def test_allocation_prefers_a_starved_fold_over_an_empty_one(policy_dict, tmp_path: Path):
    rows = make_rows(families=3, members=4, seed=52)
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "three.csv", rows))
    policy_dict["scheme"] = {"kind": "holdout", "fractions": {"train": 0.75, "test": 0.25}}
    outcome = api.make_split(Policy.from_dict(policy_dict))
    counts = {stat.name: stat.count for stat in outcome.assignment.stats}
    assert all(count > 0 for count in counts.values())
    assert counts["train"] + counts["test"] == 12


def test_many_units_land_near_the_requested_fractions(policy_dict, tmp_path: Path):
    rows = make_rows(families=40, members=3, seed=53)
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "many.csv", rows))
    policy_dict["dataset"]["groups"] = []
    policy_dict["constraints"] = [
        {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.3}
    ]
    policy_dict["scheme"] = {"kind": "holdout", "fractions": {"train": 0.8, "test": 0.2}}
    outcome = api.make_split(Policy.from_dict(policy_dict))
    stats = {stat.name: stat for stat in outcome.assignment.stats}
    assert abs(stats["test"].count - stats["test"].target) <= 3
    assert abs(stats["train"].count - stats["train"].target) <= 3


def test_a_continuous_label_is_bucketed_not_split_into_unique_classes(policy_dict, tmp_path: Path):
    rows = make_rows(families=20, members=3, seed=54)
    for position, row in enumerate(rows):
        row["activity"] = f"{position * 0.137:.5f}"
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "cont.csv", rows))
    policy = Policy.from_dict(policy_dict)
    data = dataset_module.load_for(policy)
    classes, note = units_module.label_classes(policy, data)
    assert "quantile" in note
    assert set(classes) <= {"q0", "q1", "q2", "q3"}


def test_a_small_discrete_label_stays_categorical(policy_dict, tmp_path: Path):
    rows = make_rows(families=12, members=4, seed=55)
    for row in rows:
        row["activity"] = "high" if row["family"] < "F06" else "low"
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "cat.csv", rows))
    policy = Policy.from_dict(policy_dict)
    data = dataset_module.load_for(policy)
    classes, note = units_module.label_classes(policy, data)
    assert note == "categorical"
    assert set(classes) == {"high", "low"}


def test_an_identifier_like_label_skips_stratification(policy_dict, tmp_path: Path):
    rows = make_rows(families=30, members=3, seed=56)
    for position, row in enumerate(rows):
        row["activity"] = f"sample-{position}"
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "ids.csv", rows))
    policy = Policy.from_dict(policy_dict)
    data = dataset_module.load_for(policy)
    classes, note = units_module.label_classes(policy, data)
    assert classes is None
    assert "more than" in note


def test_the_certificate_records_how_stratification_was_done(policy_dict):
    outcome = api.make_split(Policy.from_dict(policy_dict))
    assert outcome.certificate["units"]["stratify"] in {
        "categorical",
        "none",
        "numeric, 4 quantile buckets",
    }


def test_a_constraint_that_cannot_bind_is_refused(policy_dict, tmp_path: Path):
    rows = make_rows(families=6, members=3, seed=57)
    for row in rows:
        row["sequence"] = ""
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "blank.csv", rows))
    with pytest.raises(VacuousConstraint) as error:
        api.make_split(Policy.from_dict(policy_dict))
    assert error.value.code == "CONSTRAINT_VACUOUS"
    assert error.value.details["column"] == "sequence"
    assert "could never bind" in error.value.message


def test_a_partially_missing_key_is_still_allowed(policy_dict, tmp_path: Path):
    rows = make_rows(families=8, members=3, seed=58)
    rows[0]["sequence"] = ""
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "partial.csv", rows))
    outcome = api.make_split(Policy.from_dict(policy_dict))
    assert outcome.ok
    similarity = next(iter(outcome.certificate["units"]["similarity"].values()))
    assert similarity["missing_values"] == 1
