from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import pytest

from splitproof import api
from splitproof.errors import PolicyError
from splitproof.evaluate import SATISFIED, VIOLATED
from splitproof.policy import Policy

from conftest import make_rows, write_csv


def _policy(path: Path, **over) -> Dict:
    base = {
        "splitproof": 1,
        "dataset": {
            "path": str(path),
            "id": "accession",
            "label": "activity",
            "similarity": {"sequence": {"column": "sequence", "kind": "protein"}},
            "groups": [],
            "dates": {"deposited": "deposited"},
        },
        "scheme": {"kind": "holdout", "fractions": {"train": 0.7, "test": 0.3}},
        "constraints": [
            {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.3}
        ],
        "seed": 7,
    }
    base.update(over)
    return base


def _straddling(tmp_path: Path) -> Path:
    rows = make_rows(families=10, members=4, seed=21)
    for row in rows:
        family = int(row["accession"][1:3])
        member = int(row["accession"][-1])
        if family < 3:
            row["deposited"] = "2024-06-15" if member % 2 == 0 else "2025-06-15"
        elif family % 2 == 0:
            row["deposited"] = "2024-06-15"
        else:
            row["deposited"] = "2025-06-15"
    return write_csv(tmp_path / "straddle.csv", rows)


def test_a_split_splitproof_built_passes_its_own_temporal_check(tmp_path: Path):
    payload = _policy(_straddling(tmp_path))
    payload["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}
    )
    outcome = api.make_split(Policy.from_dict(payload))
    temporal = next(report for report in outcome.reports if report.kind == "temporal")
    assert temporal.status == SATISFIED
    assert temporal.measured["items_checked"] == len(outcome.data)
    assert temporal.measured["items_after_the_cutoff_in_train"] == 0
    assert outcome.certificate["status"] == SATISFIED


def test_a_late_item_in_train_is_the_only_temporal_violation(tmp_path: Path):
    rows = make_rows(families=8, members=2, seed=22)
    for position, row in enumerate(rows):
        row["deposited"] = "2025-06-15" if position % 2 else "2024-06-15"
        row["fold"] = "train" if position < len(rows) - 2 else "test"
    path = write_csv(tmp_path / "late-in-train.csv", rows)
    payload = _policy(path)
    payload["constraints"] = [{"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}]
    outcome = api.audit(Policy.from_dict(payload), split_column="fold", skip_neighbours=True)
    temporal = outcome.reports[0]
    assert temporal.status == VIOLATED
    assert temporal.measured["items_after_the_cutoff_in_train"] == 7
    assert all("on or after the cutoff" in example["problem"] for example in temporal.examples)


def test_an_early_item_in_test_is_not_a_temporal_violation(tmp_path: Path):
    rows = make_rows(families=8, members=2, seed=23)
    for position, row in enumerate(rows):
        row["deposited"] = "2024-06-15"
        row["fold"] = "train" if position % 2 else "test"
    path = write_csv(tmp_path / "all-early.csv", rows)
    payload = _policy(path)
    payload["constraints"] = [{"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}]
    outcome = api.audit(Policy.from_dict(payload), split_column="fold", skip_neighbours=True)
    assert outcome.reports[0].status == SATISFIED
    assert outcome.reports[0].measured["items_after_the_cutoff_in_train"] == 0


def test_a_temporal_audit_refuses_a_split_without_the_guarded_fold(tmp_path: Path):
    from splitproof.errors import VacuousConstraint

    rows = make_rows(families=6, members=2, seed=24)
    for position, row in enumerate(rows):
        row["deposited"] = "2025-06-15"
        row["fold"] = "Train" if position % 2 else "Test"
    path = write_csv(tmp_path / "capitalised.csv", rows)
    payload = _policy(path)
    payload["constraints"] = [{"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}]
    with pytest.raises(VacuousConstraint) as error:
        api.audit(Policy.from_dict(payload), split_column="fold", skip_neighbours=True)
    assert error.value.details["guards"] == "train"
    assert error.value.details["folds"] == ["Test", "Train"]


def test_on_span_train_is_refused_as_self_contradictory(tmp_path: Path):
    payload = _policy(_straddling(tmp_path))
    payload["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01", "on_span": "train"}
    )
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert "the very thing the constraint forbids" in error.value.message


def test_an_all_blank_group_column_is_refused(tmp_path: Path):
    from splitproof.errors import VacuousConstraint

    rows = make_rows(families=6, members=3, seed=25)
    for row in rows:
        row["donor_id"] = ""
    path = write_csv(tmp_path / "blank-group.csv", rows)
    payload = _policy(path)
    payload["dataset"]["groups"] = ["donor_id"]
    payload["constraints"] = [{"kind": "disjoint", "key": "donor_id"}]
    with pytest.raises(VacuousConstraint) as error:
        api.make_split(Policy.from_dict(payload))
    assert error.value.details["key"] == "donor_id"


def test_an_all_blank_date_column_is_refused(tmp_path: Path):
    from splitproof.errors import VacuousConstraint

    rows = make_rows(families=6, members=3, seed=26)
    for row in rows:
        row["deposited"] = ""
    path = write_csv(tmp_path / "blank-date.csv", rows)
    payload = _policy(path)
    payload["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}
    )
    with pytest.raises(VacuousConstraint):
        api.make_split(Policy.from_dict(payload))


@pytest.mark.parametrize(
    "scheme",
    [{"kind": "kfold", "folds": 4}, {"kind": "leave-unit-out"}],
)
def test_a_temporal_constraint_is_refused_where_it_could_not_bind(tmp_path: Path, scheme):
    payload = _policy(_straddling(tmp_path), scheme=scheme)
    payload["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}
    )
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert error.value.code == "POLICY_INVALID"
    assert "named folds" in error.value.message


def test_a_temporal_constraint_naming_an_unknown_fold_is_refused(tmp_path: Path):
    payload = _policy(_straddling(tmp_path))
    payload["constraints"].append(
        {
            "kind": "temporal",
            "key": "deposited",
            "cutoff": "2025-01-01",
            "before": "training",
        }
    )
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert "does not define" in error.value.message


def test_row_order_does_not_change_the_split(tmp_path: Path):
    rows = make_rows(families=20, members=4, seed=31)
    shuffled = list(rows)
    random.Random(5).shuffle(shuffled)

    def assignment_for(ordered, name):
        path = write_csv(tmp_path / name, ordered)
        outcome = api.make_split(Policy.from_dict(_policy(path)))
        return {outcome.data.ids[i]: fold for i, fold in outcome.assignment.fold_of.items()}

    assert assignment_for(rows, "plain.csv") == assignment_for(shuffled, "shuffled.csv")


def test_crossings_are_broken_down_by_fold_pair(tmp_path: Path):
    rows = make_rows(families=12, members=6, seed=32)
    for position, row in enumerate(rows):
        row["fold"] = ["train", "valid", "test"][position % 3]
    path = write_csv(tmp_path / "three.csv", rows)
    outcome = api.audit(Policy.from_dict(_policy(path)), split_column="fold", skip_neighbours=True)
    by_pair = outcome.reports[0].measured["cross_fold_pairs_by_fold_pair"]
    assert set(by_pair) == {"test|train", "test|valid", "train|valid"}
    assert sum(by_pair.values()) == outcome.reports[0].measured[
        "cross_fold_pairs_at_or_above_threshold"
    ]


def test_backend_parameters_are_part_of_the_policy_digest(tmp_path: Path):
    rows = make_rows(families=8, members=3, seed=33)
    path = write_csv(tmp_path / "k.csv", rows)
    plain = api.make_split(Policy.from_dict(_policy(path)))
    tuned = _policy(path)
    tuned["constraints"][0]["params"] = {"num_perm": 64}
    tuned["constraints"][0]["backend"] = "minhash"
    adjusted = api.make_split(Policy.from_dict(tuned))
    assert plain.certificate["policy_digest"] != adjusted.certificate["policy_digest"]
    assert adjusted.certificate["policy"]["constraints"][0]["params"] == {"num_perm": 64}


def test_an_unknown_backend_parameter_is_refused(tmp_path: Path):
    from splitproof.errors import BackendUnavailable

    rows = make_rows(families=6, members=3, seed=34)
    path = write_csv(tmp_path / "bad.csv", rows)
    payload = _policy(path)
    payload["constraints"][0]["params"] = {"nonsense": 3}
    with pytest.raises(BackendUnavailable) as error:
        api.make_split(Policy.from_dict(payload))
    assert "accepted" in error.value.details


def _certify(tmp_path: Path, payload, name: str):
    from splitproof import cert as cert_module

    outcome = api.make_split(Policy.from_dict(payload))
    split_path = tmp_path / f"{name}.split.csv"
    cert_path = tmp_path / f"{name}.cert.json"
    api.write_split(split_path, outcome.data, outcome.assignment, "accession")
    cert_module.write(cert_path, outcome.certificate)
    return outcome, split_path, cert_path


@pytest.mark.parametrize(
    "scheme", [{"kind": "leave-unit-out"}, {"kind": "kfold", "folds": 12}]
)
def test_a_split_with_more_than_ten_folds_verifies_against_itself(tmp_path: Path, scheme):
    """Folds were created in integer order and read back lexicographically.

    With eleven or more folds, unit10 sorted before unit2, so a freshly built split
    failed verification against its own certificate.
    """
    rows = make_rows(families=14, members=3, seed=71)
    payload = _policy(write_csv(tmp_path / "many.csv", rows), scheme=scheme)
    payload["constraints"] = [
        {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.3}
    ]
    _outcome, split_path, cert_path = _certify(tmp_path, payload, "many")
    result = api.verify(cert_path, split_path=split_path, data_path=Path(payload["dataset"]["path"]))
    assert result.ok, [check for check in result.checks if not check["ok"]]


def test_verify_rejects_a_boolean_swapped_for_a_number(tmp_path: Path):
    import json

    rows = make_rows(families=10, members=3, seed=72)
    payload = _policy(write_csv(tmp_path / "b.csv", rows))
    payload["constraints"] = [
        {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.3}
    ]
    _outcome, split_path, cert_path = _certify(tmp_path, payload, "b")
    certificate = json.loads(cert_path.read_text(encoding="utf-8"))
    certificate["units"]["singleton_units"] = True
    cert_path.write_text(json.dumps(certificate), encoding="utf-8")
    result = api.verify(cert_path, split_path=split_path, data_path=Path(payload["dataset"]["path"]))
    assert not result.ok
    assert result.to_dict()["constraints"], "verify must report the constraints it re-measured"


def test_row_order_does_not_change_the_reported_examples(tmp_path: Path):
    import random

    rows = make_rows(families=10, members=4, seed=73)
    for position, row in enumerate(rows):
        row["fold"] = "train" if position % 2 else "test"
    shuffled = list(rows)
    random.Random(3).shuffle(shuffled)

    def examples_for(ordered, name):
        payload = _policy(write_csv(tmp_path / name, ordered))
        outcome = api.audit(
            Policy.from_dict(payload), split_column="fold", skip_neighbours=True
        )
        return [
            (example["a"], example["b"], example["similarity"])
            for example in outcome.reports[0].examples
        ]

    assert examples_for(rows, "plain.csv") == examples_for(shuffled, "shuffled.csv")


def test_two_constraints_on_the_same_key_do_not_overwrite_each_other(tmp_path: Path):
    rows = make_rows(families=10, members=3, seed=74)
    payload = _policy(write_csv(tmp_path / "two.csv", rows))
    payload["constraints"] = [
        {
            "kind": "similarity",
            "key": "sequence",
            "metric": "kmer_jaccard",
            "threshold": 0.1,
            "params": {"k": 3},
        },
        {
            "kind": "similarity",
            "key": "sequence",
            "metric": "kmer_jaccard",
            "threshold": 0.1,
            "params": {"k": 12},
        },
    ]
    outcome = api.make_split(Policy.from_dict(payload))
    identifiers = [report.id for report in outcome.reports]
    assert len(set(identifiers)) == 2, identifiers
    assert len(outcome.certificate["units"]["similarity"]) == 2


def test_more_than_one_temporal_constraint_is_refused(tmp_path: Path):
    rows = make_rows(families=6, members=2, seed=75)
    for row in rows:
        row["deposited"] = "2024-06-15"
        row["released"] = "2025-06-15"
    path = write_csv(tmp_path / "two-dates.csv", rows)
    payload = _policy(path)
    payload["dataset"]["dates"] = {"deposited": "deposited", "released": "released"}
    payload["constraints"] += [
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"},
        {"kind": "temporal", "key": "released", "cutoff": "2025-01-01"},
    ]
    with pytest.raises(PolicyError) as error:
        api.make_split(Policy.from_dict(payload))
    assert "overwrite each other" in error.value.message


@pytest.mark.parametrize(
    "fold_values,fragment",
    [
        (["train", "", "test", "train"], "no fold"),
        (["train", "train", "train", "train"], "one fold"),
    ],
)
def test_an_audit_refuses_a_split_column_that_cannot_be_graded(tmp_path: Path, fold_values, fragment):
    from splitproof.errors import DataError

    rows = make_rows(families=2, members=2, seed=76)
    for row, value in zip(rows, fold_values, strict=True):
        row["fold"] = value
    path = write_csv(tmp_path / "bad-folds.csv", rows)
    with pytest.raises(DataError) as error:
        api.audit(Policy.from_dict(_policy(path)), split_column="fold", skip_neighbours=True)
    assert fragment in error.value.message


def test_a_split_file_assigning_one_id_twice_is_refused(tmp_path: Path):
    from splitproof.errors import DataError

    rows = make_rows(families=4, members=2, seed=77)
    path = write_csv(tmp_path / "dupe.csv", rows)
    data_ids = [row["accession"] for row in rows]
    split = tmp_path / "dupe.split.csv"
    lines = ["accession,fold"] + [f"{entity},train" for entity in data_ids]
    lines.append(f"{data_ids[0]},test")
    split.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DataError) as error:
        api.audit(Policy.from_dict(_policy(path)), split_path=split, skip_neighbours=True)
    assert "more than one fold" in error.value.message


def test_a_policy_cannot_name_an_executable_for_a_backend(tmp_path: Path):
    from splitproof.errors import BackendUnavailable

    rows = make_rows(families=4, members=2, seed=78)
    payload = _policy(write_csv(tmp_path / "rce.csv", rows))
    payload["constraints"] = [
        {
            "kind": "similarity",
            "key": "sequence",
            "metric": "identity",
            "threshold": 0.3,
            "params": {"binary": "/bin/sh"},
        }
    ]
    with pytest.raises(BackendUnavailable) as error:
        api.make_split(Policy.from_dict(payload))
    assert "binary" in error.value.details["given"]
    assert "binary" not in error.value.details["accepted"]


def test_a_dropped_split_verifies_against_its_own_certificate(tmp_path: Path):
    """Round 2 wired audit-input checks into verify and broke on_span: drop.

    write_split records policy-dropped rows with the internal sentinel, which the new
    check refused, so any split with a spanning unit could not be verified at all.
    """
    payload = _policy(_straddling(tmp_path))
    payload["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01", "on_span": "drop"}
    )
    outcome, split_path, cert_path = _certify(tmp_path, payload, "dropped")
    assert outcome.certificate["assignment"]["dropped"] > 0
    result = api.verify(
        cert_path, split_path=split_path, data_path=Path(payload["dataset"]["path"])
    )
    assert result.ok, [check for check in result.checks if not check["ok"]]


def test_a_holdout_with_numbered_fold_names_verifies(tmp_path: Path):
    rows = make_rows(families=10, members=3, seed=81)
    payload = _policy(
        write_csv(tmp_path / "numbered.csv", rows),
        scheme={"kind": "holdout", "fractions": {"s2": 0.5, "s10": 0.5}},
    )
    outcome, split_path, cert_path = _certify(tmp_path, payload, "numbered")
    assert outcome.assignment.folds == ["s2", "s10"]
    result = api.verify(
        cert_path, split_path=split_path, data_path=Path(payload["dataset"]["path"])
    )
    assert result.ok, [check for check in result.checks if not check["ok"]]


def test_thresholds_that_print_alike_are_still_distinct_constraints(tmp_path: Path):
    rows = make_rows(families=10, members=3, seed=82)
    payload = _policy(write_csv(tmp_path / "alike.csv", rows))
    payload["constraints"] = [
        {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.5},
        {
            "kind": "similarity",
            "key": "sequence",
            "metric": "kmer_jaccard",
            "threshold": 0.5000004,
        },
    ]
    outcome = api.make_split(Policy.from_dict(payload))
    assert len({report.id for report in outcome.reports}) == 2
    assert len(outcome.certificate["units"]["similarity"]) == 2


def test_an_exactly_repeated_constraint_is_refused(tmp_path: Path):
    rows = make_rows(families=6, members=2, seed=83)
    payload = _policy(write_csv(tmp_path / "dupe-constraint.csv", rows))
    payload["constraints"] = payload["constraints"] * 2
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert "repeats" in error.value.message


def test_verify_will_not_follow_a_data_path_the_certificate_chose(tmp_path: Path):
    import json

    from splitproof.errors import DataError

    rows = make_rows(families=8, members=3, seed=84)
    payload = _policy(write_csv(tmp_path / "source.csv", rows))
    _outcome, split_path, cert_path = _certify(tmp_path, payload, "source")
    certificate = json.loads(cert_path.read_text(encoding="utf-8"))
    certificate["source"]["path"] = "/etc/hosts"
    cert_path.write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises(DataError) as error:
        api.verify(cert_path, split_path=split_path)
    assert "outside this directory" in error.value.message


def test_a_key_populated_on_one_row_is_not_proven(tmp_path: Path):
    rows = make_rows(families=6, members=3, seed=85)
    for position, row in enumerate(rows):
        row["fold"] = "train" if position % 2 else "test"
        if position:
            row["sequence"] = ""
    path = write_csv(tmp_path / "sparse.csv", rows)
    outcome = api.audit(
        Policy.from_dict(_policy(path)), split_column="fold", skip_neighbours=True
    )
    assert outcome.certificate["status"] == "satisfied_unproven"
    assert outcome.reports[0].measured["values_absent"] == len(rows) - 1
    assert any("unconstrained" in note for note in outcome.reports[0].notes)


@pytest.mark.parametrize("name", ["__dropped__", "__unassigned__", "__anything"])
def test_a_reserved_fold_name_is_refused_at_parse_time(tmp_path: Path, name: str):
    path = write_csv(tmp_path / "d.csv", make_rows(families=2, members=2, seed=3))
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(_policy(path, scheme={"kind": "holdout", "fractions": {"train": 0.5, name: 0.5}}))
    assert "reserved" in error.value.message


def test_fold_names_that_differ_only_by_whitespace_are_refused(tmp_path: Path):
    path = write_csv(tmp_path / "d.csv", make_rows(families=2, members=2, seed=3))
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(_policy(path, scheme={"kind": "holdout", "fractions": {"train": 8, " train": 1, "test": 1}}))
    assert "twice" in error.value.message


def test_an_absurd_fold_count_is_refused_before_any_work(tmp_path: Path):
    path = write_csv(tmp_path / "d.csv", make_rows(families=2, members=2, seed=3))
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(_policy(path, scheme={"kind": "kfold", "folds": 10**9}))
    assert error.value.details["cap"] == 1000
