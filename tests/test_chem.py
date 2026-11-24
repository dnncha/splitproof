from __future__ import annotations

from pathlib import Path

import pytest

from cutsite import api
from cutsite.evaluate import SATISFIED, VIOLATED
from cutsite.policy import Policy
from cutsite.sim import EXACT, TooManyPairs
from cutsite.sim.chem import MorganTanimoto, MurckoScaffold

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


def test_unparsable_smiles_is_missing_not_fatal():
    result = MorganTanimoto().pairs_above(["CCO", "not a molecule", None], "molecule", 0.5)
    assert 1 in result.missing
    assert 2 in result.missing
    assert any("parsable" in note for note in result.notes)


def test_scaffold_groups_a_shared_core():
    result = MurckoScaffold().pairs_above(BENZODIAZEPINES, "molecule", 1.0)
    assert len(result.pairs) == 3
    assert all(similarity == 1.0 for _l, _r, similarity in result.pairs)


def test_scaffold_keeps_different_cores_apart():
    result = MurckoScaffold().pairs_above(BENZODIAZEPINES + PYRIDINES, "molecule", 1.0)
    for left, right, _similarity in result.pairs:
        assert (left < 3) == (right < 3)


def test_tanimoto_nearest_finds_the_closest_analogue():
    neighbours = MorganTanimoto().nearest(
        BENZODIAZEPINES + ALKANES, "molecule", [0], list(range(1, 6))
    )
    best = neighbours[0]
    assert best is not None
    assert best.index in {1, 2}


def test_scaffold_split_keeps_a_series_together(tmp_path: Path):
    path = _write(tmp_path / "mols.csv", [("bzd", BENZODIAZEPINES), ("pyr", PYRIDINES)])
    outcome = api.make_split(Policy.from_dict(_policy(path, "scaffold", 1.0)))
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
    outcome = api.audit(Policy.from_dict(_policy(path, "scaffold", 1.0)), split_column="fold")
    assert outcome.reports[0].status == VIOLATED


def test_chem_backends_charge_the_pair_budget():
    values = BENZODIAZEPINES * 8
    with pytest.raises(TooManyPairs):
        MorganTanimoto().pairs_above(values, "molecule", 0.1, max_pairs=5)
    with pytest.raises(TooManyPairs):
        MurckoScaffold().pairs_above(values, "molecule", 1.0, max_pairs=5)


def test_an_acyclic_only_dataset_refuses_rather_than_passing_vacuously(tmp_path: Path):
    from cutsite.errors import VacuousConstraint

    path = _write(tmp_path / "alkanes.csv", [("alk", ALKANES)])
    with pytest.raises(VacuousConstraint) as error:
        api.make_split(Policy.from_dict(_policy(path, "scaffold", 1.0)))
    assert error.value.code == "CONSTRAINT_VACUOUS"
    assert error.value.details["backend"] == "scaffold"


def test_framework_ignores_halogen_and_smiles_ordering():
    from rdkit import Chem

    from cutsite.sim.chem import murcko_framework

    frameworks = {murcko_framework(Chem, Chem.MolFromSmiles(s)) for s in BENZODIAZEPINES}
    assert len(frameworks) == 1


def test_framework_is_none_for_a_molecule_with_no_rings():
    from rdkit import Chem

    from cutsite.sim.chem import murcko_framework

    assert murcko_framework(Chem, Chem.MolFromSmiles("CCCC")) is None
