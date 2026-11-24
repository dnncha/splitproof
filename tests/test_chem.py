from __future__ import annotations

from pathlib import Path

import pytest

from cutsite import api
from cutsite.evaluate import SATISFIED, VIOLATED
from cutsite.policy import Policy
from cutsite.sim import EXACT, TooManyPairs
from cutsite.sim.chem import MorganTanimoto, RingSystems

rdkit = pytest.importorskip("rdkit", reason="RDKit is not installed on this machine")

BENZODIAZEPINES = [
    "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1",
    "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1",
    "CN1c2ccc(Br)cc2C(=NCC1=O)c1ccccc1",
]
PYRIDINES = [
    "Cc1ccncc1",
    "CCc1ccncc1",
    "Clc1ccncc1",
]
ALKANES = ["CCCC", "CCCCC", "CCCCCC"]


def _write(path: Path, groups) -> Path:
    lines = ["id,smiles,series,activity"]
    for name, members in groups:
        for position, smiles in enumerate(members):
            lines.append(f"{name}{position},{smiles},{name},{position}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _policy(path: Path, metric: str, threshold: float):
    return {
        "cutsite": 1,
        "dataset": {
            "path": str(path),
            "id": "id",
            "label": "activity",
            "similarity": {"molecule": {"column": "smiles", "kind": "molecule"}},
            "groups": [],
            "dates": {},
        },
        "scheme": {"kind": "holdout", "fractions": {"train": 0.6, "test": 0.4}},
        "constraints": [
            {"kind": "similarity", "key": "molecule", "metric": metric, "threshold": threshold}
        ],
        "seed": 7,
    }


def test_tanimoto_is_exact_and_self_consistent():
    result = MorganTanimoto().pairs_above(BENZODIAZEPINES, "molecule", 0.5)
    assert result.exactness == EXACT
    assert len(result.pairs) >= 1
    for _left, _right, similarity in result.pairs:
        assert 0.5 <= similarity <= 1.0


def test_tanimoto_separates_unrelated_series():
    values = BENZODIAZEPINES + ALKANES
    result = MorganTanimoto().pairs_above(values, "molecule", 0.7)
    for left, right, _similarity in result.pairs:
        assert (left < 3) == (right < 3)


def test_unparsable_smiles_is_distinguished_from_an_absent_one():
    result = MorganTanimoto().pairs_above(["CCO", "not a molecule", None], "molecule", 0.5)
    assert result.unparsable == [1]
    assert result.missing == [1, 2]
    assert any("could not parse" in note for note in result.notes)
    assert any("no SMILES at all" in note for note in result.notes)


def test_a_partly_unparsable_column_surfaces_the_count(tmp_path: Path):
    path = tmp_path / "half.csv"
    rows = ["id,smiles,activity"]
    for position, smiles in enumerate(BENZODIAZEPINES + PYRIDINES):
        rows.append(f"g{position},{smiles},{position}")
    for position in range(4):
        rows.append(f"bad{position},not a molecule,{position}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    outcome = api.make_split(Policy.from_dict(_policy(path, "tanimoto", 0.5)))
    measured = outcome.reports[0].measured
    assert measured["values_unparsable"] == 4
    assert measured["values_absent"] == 4


def test_scaffold_groups_a_shared_core():
    result = RingSystems().pairs_above(BENZODIAZEPINES, "molecule", 1.0)
    assert result.groups is not None
    assert len(set(result.groups)) == 1
    assert len(result.pairs) == 0


def test_scaffold_keeps_different_cores_apart():
    result = RingSystems().pairs_above(BENZODIAZEPINES + PYRIDINES, "molecule", 1.0)
    assert result.groups[:3] == [result.groups[0]] * 3
    assert result.groups[3:] == [result.groups[3]] * 3
    assert result.groups[0] != result.groups[3]


def test_tanimoto_nearest_finds_the_closest_analogue():
    neighbours = MorganTanimoto().nearest(
        BENZODIAZEPINES + ALKANES, "molecule", [0], list(range(1, 6))
    )
    best = neighbours[0]
    assert best is not None
    assert best.index in {1, 2}


def test_scaffold_split_keeps_a_series_together(tmp_path: Path):
    path = _write(tmp_path / "mols.csv", [("bzd", BENZODIAZEPINES), ("pyr", PYRIDINES)])
    outcome = api.make_split(Policy.from_dict(_policy(path, "ring_systems", 1.0)))
    series = outcome.data.column("series")
    seen = {}
    for index, fold in outcome.assignment.fold_of.items():
        seen.setdefault(series[index], set()).add(fold)
    assert all(len(folds) == 1 for folds in seen.values())
    assert outcome.certificate["status"] == SATISFIED


def test_a_scaffold_leak_is_caught(tmp_path: Path):
    path = tmp_path / "leaky.csv"
    lines = ["id,smiles,series,activity,fold"]
    for position, smiles in enumerate(BENZODIAZEPINES):
        lines.append(f"b{position},{smiles},bzd,{position},{'train' if position % 2 == 0 else 'test'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outcome = api.audit(Policy.from_dict(_policy(path, "ring_systems", 1.0)), split_column="fold")
    assert outcome.reports[0].status == VIOLATED


def test_tanimoto_charges_the_pair_budget_but_scaffold_needs_no_pairs():
    values = BENZODIAZEPINES * 8
    with pytest.raises(TooManyPairs):
        MorganTanimoto().pairs_above(values, "molecule", 0.1, max_pairs=5)
    classes = RingSystems().pairs_above(values, "molecule", 1.0, max_pairs=5)
    assert len(classes.pairs) == 0
    assert classes.groups is not None


def test_an_acyclic_only_dataset_refuses_rather_than_passing_vacuously(tmp_path: Path):
    from cutsite.errors import VacuousConstraint

    path = _write(tmp_path / "alkanes.csv", [("alk", ALKANES)])
    with pytest.raises(VacuousConstraint) as error:
        api.make_split(Policy.from_dict(_policy(path, "ring_systems", 1.0)))
    assert error.value.code == "CONSTRAINT_VACUOUS"
    assert error.value.details["backend"] == "ring_systems"


def test_ring_systems_ignore_halogen_and_smiles_ordering():
    from rdkit import Chem

    from cutsite.sim.chem import ring_systems

    keys = {ring_systems(Chem, Chem.MolFromSmiles(s)) for s in BENZODIAZEPINES}
    assert len(keys) == 1


def test_ring_systems_are_none_for_a_molecule_with_no_rings():
    from rdkit import Chem

    from cutsite.sim.chem import ring_systems

    assert ring_systems(Chem, Chem.MolFromSmiles("CCCC")) is None


def test_unparsable_input_downgrades_proven_to_unproven(tmp_path: Path):
    path = tmp_path / "mixed.csv"
    lines = ["id,smiles,series,activity"]
    for position, smiles in enumerate(BENZODIAZEPINES):
        lines.append(f"b{position},{smiles},bzd,{position}")
    for position, smiles in enumerate(PYRIDINES):
        lines.append(f"p{position},{smiles},pyr,{position}")
    for position in range(3):
        lines.append(f"junk{position},@@@not-a-molecule@@@,junk,{position}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outcome = api.make_split(Policy.from_dict(_policy(path, "tanimoto", 0.5)))
    report = outcome.reports[0]
    assert report.measured["values_unparsable"] == 3
    assert report.status == "satisfied_unproven"
    assert any("compared to nothing" in note for note in report.notes)
    assert outcome.certificate["status"] == "satisfied_unproven"


REAL_DRUGS = {
    "indole": "c1ccc2[nH]ccc2c1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "tryptophan": "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "adenine": "Nc1ncnc2[nH]cnc12",
    "guanine": "Nc1nc2[nH]cnc2c(=O)[nH]1",
    "thymine": "Cc1c[nH]c(=O)[nH]c1=O",
    "sildenafil": "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
    "omeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "morphine": "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "metformin": "CN(C)C(=N)NC(N)=N",
}


@pytest.mark.parametrize("name,smiles", sorted(REAL_DRUGS.items()))
def test_ring_systems_survives_real_drugs(name, smiles):
    """The guard for an escape: the previous hand-rolled Murcko reduction crashed.

    Stripping explicit hydrogens from pyrrole-type aromatic nitrogen broke
    kekulisation, so indole, caffeine, tryptophan and every nucleobase raised
    KekulizeException. Seven of nine common compounds failed.
    """
    from rdkit import Chem

    from cutsite.sim.chem import ring_systems

    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None, name
    key = ring_systems(Chem, molecule)
    assert key is None or isinstance(key, str)


def test_every_real_drug_goes_through_the_backend_without_failing():
    values = list(REAL_DRUGS.values())
    result = RingSystems().pairs_above(values, "molecule", 1.0)
    assert result.groups is not None
    assert len(result.missing) == 1


def test_an_equivalence_metric_refuses_a_threshold_below_one(tmp_path: Path):
    from cutsite.errors import PolicyError

    path = _write(tmp_path / "m.csv", [("bzd", BENZODIAZEPINES)])
    payload = _policy(path, "ring_systems", 1.0)
    payload["constraints"][0]["threshold"] = 0.1
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert "always 1" in error.value.message
