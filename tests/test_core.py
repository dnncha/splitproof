from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from cutsite import api, cert as cert_module, dataset as dataset_module, units as units_module
from cutsite.assign import DROPPED
from cutsite.errors import DataError, GiantUnit, PolicyError
from cutsite.evaluate import SATISFIED, VIOLATED
from cutsite.policy import Policy
from cutsite.sim.kmer import ExactJaccard, MinHashJaccard, tokenise
from cutsite.sim.mmseqs import parse_search_tsv
from cutsite.units import UnionFind

from conftest import make_rows, write_csv


def test_unionfind_merges_transitively():
    union = UnionFind(5)
    assert union.union(0, 1)
    assert union.union(1, 2)
    assert not union.union(0, 2)
    assert union.find(0) == union.find(2)
    assert union.find(3) != union.find(0)


def test_tokenise_handles_short_values():
    assert tokenise("ACDEF", 3) == ["ACD", "CDE", "DEF"]
    assert tokenise("AC", 3) == ["AC"]
    assert tokenise("", 3) == []


def test_exact_jaccard_is_symmetric_and_finds_identical_pairs():
    values = ["ACDEFGHIKLMNPQRST", "ACDEFGHIKLMNPQRST", "WWWWWWWWWWWWWWWWW"]
    result = ExactJaccard().pairs_above(values, "protein", 0.9)
    assert [(a, b) for a, b, _ in result.pairs] == [(0, 1)]
    assert result.pairs[0][2] == pytest.approx(1.0)
    assert result.exactness == "exact"


def test_minhash_recovers_most_exact_pairs():
    rows = make_rows(families=8, members=4, seed=3)
    values = [row["sequence"] for row in rows]
    exact = {(a, b) for a, b, _ in ExactJaccard().pairs_above(values, "protein", 0.3).pairs}
    approximate = {(a, b) for a, b, _ in MinHashJaccard().pairs_above(values, "protein", 0.3).pairs}
    assert approximate <= exact
    assert len(approximate) >= 0.8 * len(exact)


def test_minhash_declares_itself_a_lower_bound():
    result = MinHashJaccard().pairs_above(["ACDEFGHIKLM" * 3, "ACDEFGHIKLM" * 3], "protein", 0.5)
    assert result.exactness == "lower_bound"
    assert any("lower bound" in note for note in result.notes)


def test_mmseqs_tsv_parsing_dedupes_and_normalises_percentages():
    pairs = parse_search_tsv(
        "s0\ts1\t95.0\ns1\ts0\t0.9\ns2\ts2\t1.0\ns0\ts9\t0.5\n",
        {"s0": 0, "s1": 1, "s2": 2},
    )
    assert pairs == [(0, 1, 0.95)]


def test_policy_rejects_constraint_referencing_unknown_key(policy_dict):
    policy_dict["constraints"] = [
        {"kind": "similarity", "key": "structure", "metric": "kmer_jaccard", "threshold": 0.3}
    ]
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(policy_dict)
    assert "structure" in str(error.value)


def test_policy_rejects_empty_constraints(policy_dict):
    policy_dict["constraints"] = []
    with pytest.raises(PolicyError):
        Policy.from_dict(policy_dict)


def test_policy_rejects_identity_metric_on_molecule_key(policy_dict):
    policy_dict["dataset"]["similarity"]["ligand"] = {"column": "sequence", "kind": "molecule"}
    policy_dict["constraints"] = [
        {"kind": "similarity", "key": "ligand", "metric": "identity", "threshold": 0.3}
    ]
    with pytest.raises(PolicyError):
        Policy.from_dict(policy_dict)


def test_duplicate_ids_are_rejected(tmp_path: Path):
    rows = make_rows(families=2, members=2)
    rows[1]["accession"] = rows[0]["accession"]
    path = write_csv(tmp_path / "dupes.csv", rows)
    with pytest.raises(DataError) as error:
        dataset_module.load(path, "accession")
    assert "appears on rows" in str(error.value)


def test_split_keeps_homologous_families_together(policy_dict):
    policy = Policy.from_dict(policy_dict)
    outcome = api.make_split(policy)
    folds = outcome.assignment.fold_of
    by_family: Dict[str, set] = {}
    families = outcome.data.column("family")
    for index, fold in folds.items():
        by_family.setdefault(families[index], set()).add(fold)
    assert all(len(seen) == 1 for seen in by_family.values())
    assert outcome.certificate["status"] == SATISFIED
    assert outcome.ok


def test_split_satisfies_declared_disjoint_group(policy_dict):
    policy_dict["constraints"].append({"kind": "disjoint", "key": "donor_id"})
    policy = Policy.from_dict(policy_dict)
    outcome = api.make_split(policy)
    reports = {report.id: report for report in outcome.reports}
    assert reports["disjoint:donor_id"].status == SATISFIED
    assert reports["disjoint:donor_id"].measured["values_in_more_than_one_fold"] == 0


def test_split_is_deterministic_for_a_seed(policy_dict):
    first = api.make_split(Policy.from_dict(policy_dict))
    second = api.make_split(Policy.from_dict(policy_dict))
    assert first.certificate["assignment"]["digest"] == second.certificate["assignment"]["digest"]


def test_changing_the_seed_can_change_the_assignment(policy_dict):
    first = api.make_split(Policy.from_dict(policy_dict))
    policy_dict["seed"] = 99
    second = api.make_split(Policy.from_dict(policy_dict))
    assert first.certificate["policy_digest"] != second.certificate["policy_digest"]


def test_giant_unit_is_an_explicit_refusal(policy_dict, tmp_path: Path):
    rows = make_rows(families=1, members=40, mutation_rate=0.01, seed=5)
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "blob.csv", rows))
    policy_dict["constraints"] = [
        {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.1}
    ]
    policy = Policy.from_dict(policy_dict)
    with pytest.raises(GiantUnit) as error:
        api.make_split(policy)
    assert error.value.details["largest_unit_fraction"] > 0.5
    assert "does not fit" in error.value.message


def test_random_split_audit_reports_violations(policy_dict, tmp_path: Path):
    rows = make_rows()
    for position, row in enumerate(rows):
        row["fold"] = "train" if position % 2 == 0 else "test"
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "leaky.csv", rows))
    policy = Policy.from_dict(policy_dict)
    outcome = api.audit(policy, split_column="fold")
    report = outcome.reports[0]
    assert report.status == VIOLATED
    assert report.measured["cross_fold_pairs_at_or_above_threshold"] > 0
    assert report.examples
    assert outcome.certificate["status"] == VIOLATED


def test_audit_surfaces_the_nearest_neighbour_baseline(policy_dict, tmp_path: Path):
    rows = make_rows()
    for position, row in enumerate(rows):
        row["fold"] = "train" if position % 2 == 0 else "test"
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "leaky.csv", rows))
    outcome = api.audit(Policy.from_dict(policy_dict), split_column="fold")
    entry = next(iter(outcome.certificate["diagnostics"]["keys"].values()))
    baseline = entry["nearest_neighbour_baseline"]
    assert baseline["compared"] > 0
    assert baseline["value"] is not None
    assert entry["nearest_reference_similarity"]["max"] > 0.3


def test_temporal_constraint_puts_late_items_in_test(policy_dict):
    policy_dict["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}
    )
    policy = Policy.from_dict(policy_dict)
    outcome = api.make_split(policy)
    dates = outcome.data.dates("deposited")
    for index, fold in outcome.assignment.fold_of.items():
        if fold == DROPPED:
            continue
        if dates[index] is not None and dates[index].year >= 2025:
            assert fold == "test"
    reports = {report.kind: report for report in outcome.reports}
    assert reports["temporal"].status == SATISFIED


def _straddling_rows():
    rows = make_rows(families=8, members=4, seed=21)
    for row in rows:
        family = int(row["accession"][1:3])
        member = int(row["accession"][-1])
        if family < 2:
            row["deposited"] = "2024-06-15" if member % 2 == 0 else "2025-06-15"
        else:
            row["deposited"] = "2024-06-15" if family % 2 == 0 else "2025-06-15"
    return rows


def test_temporal_on_span_error_refuses_rather_than_guessing(policy_dict, tmp_path: Path):
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "straddle.csv", _straddling_rows()))
    policy_dict["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01", "on_span": "error"}
    )
    with pytest.raises(PolicyError) as error:
        api.make_split(Policy.from_dict(policy_dict))
    assert "spans the temporal cutoff" in error.value.message


def test_temporal_on_span_test_keeps_the_whole_unit_out_of_train(policy_dict, tmp_path: Path):
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "straddle.csv", _straddling_rows()))
    policy_dict["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01", "on_span": "test"}
    )
    outcome = api.make_split(Policy.from_dict(policy_dict))
    assert outcome.certificate["units"]["spanning_units"] > 0
    families = outcome.data.column("family")
    straddled = {
        families[index]
        for index, fold in outcome.assignment.fold_of.items()
        if fold == "test"
    }
    for index, fold in outcome.assignment.fold_of.items():
        if families[index] in straddled:
            assert fold == "test"


def test_temporal_on_span_drop_removes_the_unit(policy_dict, tmp_path: Path):
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "straddle.csv", _straddling_rows()))
    policy_dict["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01", "on_span": "drop"}
    )
    outcome = api.make_split(Policy.from_dict(policy_dict))
    assert outcome.certificate["units"]["dropped_entities"] > 0
    assert all(fold != DROPPED for index, fold in outcome.assignment.fold_of.items() if index not in outcome.assignment.dropped)


def test_kfold_fills_every_fold(policy_dict):
    policy_dict["scheme"] = {"kind": "kfold", "folds": 4}
    outcome = api.make_split(Policy.from_dict(policy_dict))
    assert len(outcome.assignment.folds) == 4
    assert all(stat.count > 0 for stat in outcome.assignment.stats)


def test_verify_accepts_an_untouched_split(policy_dict, tmp_path: Path):
    policy = Policy.from_dict(policy_dict)
    outcome = api.make_split(policy)
    split_path = tmp_path / "split.csv"
    cert_path = tmp_path / "cutsite.cert.json"
    api.write_split(split_path, outcome.data, outcome.assignment, "accession")
    cert_module.write(cert_path, outcome.certificate)
    result = api.verify(cert_path, split_path=split_path)
    assert result.ok
    assert all(check["ok"] for check in result.checks)


def test_verify_rejects_a_swapped_assignment(policy_dict, tmp_path: Path):
    policy = Policy.from_dict(policy_dict)
    outcome = api.make_split(policy)
    split_path = tmp_path / "split.csv"
    cert_path = tmp_path / "cutsite.cert.json"
    api.write_split(split_path, outcome.data, outcome.assignment, "accession")
    cert_module.write(cert_path, outcome.certificate)
    lines = split_path.read_text(encoding="utf-8").splitlines()
    header, body = lines[0], lines[1:]
    flipped = []
    for line in body:
        entity_id, fold = line.split(",")
        flipped.append(f"{entity_id},{'test' if fold == 'train' else 'train'}")
    split_path.write_text("\n".join([header] + flipped) + "\n", encoding="utf-8")
    result = api.verify(cert_path, split_path=split_path)
    assert not result.ok
    assert any(check["name"] == "assignment_digest" and not check["ok"] for check in result.checks)


def test_verify_rejects_edited_data(policy_dict, tmp_path: Path):
    policy = Policy.from_dict(policy_dict)
    outcome = api.make_split(policy)
    split_path = tmp_path / "split.csv"
    cert_path = tmp_path / "cutsite.cert.json"
    api.write_split(split_path, outcome.data, outcome.assignment, "accession")
    cert_module.write(cert_path, outcome.certificate)
    data_path = Path(policy_dict["dataset"]["path"])
    text = data_path.read_text(encoding="utf-8").replace("D0", "D5")
    data_path.write_text(text, encoding="utf-8")
    result = api.verify(cert_path, split_path=split_path)
    assert not result.ok
    assert any(check["name"] == "dataset_digest" and not check["ok"] for check in result.checks)


def test_certificate_records_backend_exactness(policy_dict):
    outcome = api.make_split(Policy.from_dict(policy_dict))
    assert outcome.certificate["units"]["exactness"] == "exact"
    constraint = outcome.certificate["constraints"][0]
    assert constraint["measured"]["backend"] == "exact_jaccard"
    assert constraint["status"] == SATISFIED


def test_minhash_backend_downgrades_the_claim(policy_dict):
    outcome = api.make_split(
        Policy.from_dict(policy_dict), backend_overrides={"sequence": "minhash"}
    )
    constraint = outcome.certificate["constraints"][0]
    assert constraint["status"] == "satisfied_unproven"
    assert outcome.ok


def test_units_report_counts_merges(policy_dict):
    policy = Policy.from_dict(policy_dict)
    data = dataset_module.load_for(policy)
    graph = units_module.build(policy, data)
    payload = graph.to_dict()
    assert payload["units"] < len(data)
    assert payload["entities"] == len(data)
    assert payload["largest_unit"] >= 5


def test_split_file_with_unknown_ids_is_rejected(policy_dict, tmp_path: Path):
    policy = Policy.from_dict(policy_dict)
    data = dataset_module.load_for(policy)
    bad = tmp_path / "bad.csv"
    bad.write_text("accession,fold\nNOPE,train\n", encoding="utf-8")
    with pytest.raises(DataError):
        api.load_assignment(data, split_path=bad)


def test_certificate_round_trips_as_json(policy_dict, tmp_path: Path):
    outcome = api.make_split(Policy.from_dict(policy_dict))
    path = tmp_path / "c.json"
    cert_module.write(path, outcome.certificate)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded == outcome.certificate
    assert cert_module.read(path)["policy_digest"] == outcome.certificate["policy_digest"]
