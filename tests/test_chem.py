from __future__ import annotations

from pathlib import Path

import pytest

from splitproof import api
from splitproof.evaluate import SATISFIED, VIOLATED
from splitproof.policy import Policy
from splitproof.sim import EXACT, TooManyPairs
from splitproof.sim.chem import MorganTanimoto, RingSystems

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
        "splitproof": 1,
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
    assert measured["values_absent"] == 0


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
    from splitproof.errors import VacuousConstraint

    path = _write(tmp_path / "alkanes.csv", [("alk", ALKANES)])
    with pytest.raises(VacuousConstraint) as error:
        api.make_split(Policy.from_dict(_policy(path, "ring_systems", 1.0)))
    assert error.value.code == "CONSTRAINT_VACUOUS"
    assert error.value.details["backend"] == "ring_systems"


def test_ring_systems_ignore_halogen_and_smiles_ordering():
    from rdkit import Chem

    from splitproof.sim.chem import ring_systems

    keys = {ring_systems(Chem, Chem.MolFromSmiles(s)) for s in BENZODIAZEPINES}
    assert len(keys) == 1


def test_ring_systems_are_none_for_a_molecule_with_no_rings():
    from rdkit import Chem

    from splitproof.sim.chem import ring_systems

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

    from splitproof.sim.chem import ring_systems

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
    from splitproof.errors import PolicyError

    path = _write(tmp_path / "m.csv", [("bzd", BENZODIAZEPINES)])
    payload = _policy(path, "ring_systems", 1.0)
    payload["constraints"][0]["threshold"] = 0.1
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert "always 1" in error.value.message


def test_unconstrained_counts_are_not_double_counted(tmp_path: Path):
    from splitproof import report as report_module

    path = tmp_path / "mixed.csv"
    lines = ["id,smiles,series,activity"]
    for position, smiles in enumerate(BENZODIAZEPINES + PYRIDINES):
        lines.append(f"g{position},{smiles},s,{position}")
    for position in range(4):
        lines.append(f"bad{position},@@@nope@@@,s,{position}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outcome = api.make_split(Policy.from_dict(_policy(path, "tanimoto", 0.5)))
    summary = report_module.summary(outcome.certificate)
    entities = outcome.certificate["dataset"]["entities"]
    assert entities == 10
    assert "4 entities had no usable value" in summary


EQUIVALENT_PAIRS = [
    ("indole", "c1ccc2[nH]ccc2c1", "N-methylindole", "Cn1ccc2ccccc21"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O"),
    ("adenine", "Nc1ncnc2[nH]cnc12", "9-methyladenine", "Cn1cnc2c(N)ncnc21"),
    ("pyridine", "c1ccncc1", "pyridinium", "c1cc[nH+]cc1"),
    (
        "morphine",
        "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
        "ent-morphine",
        "CN1CC[C@]23c4c5ccc(O)c4O[C@@H]2[C@H](O)C=C[C@H]3[C@H]1C5",
    ),
    (
        "aspirin",
        "CC(=O)Oc1ccccc1C(=O)O",
        "aspirin tosylate",
        "CC(=O)Oc1ccccc1C(=O)O.Cc1ccc(S(=O)(=O)O)cc1",
    ),
]

DISTINCT_PAIRS = [
    ("benzene", "c1ccccc1", "pyridine", "c1ccncc1"),
    ("indole", "c1ccc2[nH]ccc2c1", "benzimidazole", "c1ccc2[nH]cnc2c1"),
    ("naphthalene", "c1ccc2ccccc2c1", "quinoline", "c1ccc2ncccc2c1"),
]


def _key(smiles: str):
    from rdkit import Chem

    from splitproof.sim.chem import ring_systems

    return ring_systems(Chem, Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("left_name,left,right_name,right", EQUIVALENT_PAIRS)
def test_near_duplicates_share_a_ring_key(left_name, left, right_name, right):
    """N-substitution, protonation, stereochemistry and salt form must not separate.

    The first implementation keyed on the parent molecule's own SMILES, so
    N-methylindole, theophylline, pyridinium, an enantiomer and a tosylate salt all
    landed in different classes and could cross folds freely.
    """
    assert _key(left) == _key(right), (left_name, right_name)


@pytest.mark.parametrize("left_name,left,right_name,right", DISTINCT_PAIRS)
def test_genuinely_different_skeletons_stay_apart(left_name, left, right_name, right):
    assert _key(left) != _key(right), (left_name, right_name)


def test_a_salt_and_its_parent_land_in_the_same_fold(tmp_path: Path):
    path = tmp_path / "salts.csv"
    path.write_text(
        "id,smiles,activity,fold\n"
        "parent,CC(=O)Oc1ccccc1C(=O)O,1,train\n"
        "salt,CC(=O)Oc1ccccc1C(=O)O.Cc1ccc(S(=O)(=O)O)cc1,1,test\n"
        "other,c1ccc2ncccc2c1,2,train\n"
        "other2,c1ccc2ncccc2c1,2,test\n",
        encoding="utf-8",
    )
    outcome = api.audit(
        Policy.from_dict(_policy(path, "ring_systems", 1.0)),
        split_column="fold",
        skip_neighbours=True,
    )
    assert outcome.reports[0].status == VIOLATED
    assert outcome.reports[0].measured["cross_fold_pairs_at_or_above_threshold"] == 2


def test_canonical_smiles_groups_spellings_and_salts_but_not_enantiomers():
    from splitproof.sim.chem import CanonicalSmiles

    values = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "O=C(C)Oc1ccccc1C(=O)O",
        "CC(=O)Oc1ccccc1C(=O)O.Cc1ccc(S(=O)(=O)O)cc1",
        "C[C@H](N)C(=O)O",
        "C[C@@H](N)C(=O)O",
    ]
    result = CanonicalSmiles().pairs_above(values, "molecule", 1.0)
    assert result.groups[0] == result.groups[1] == result.groups[2]
    assert result.groups[3] != result.groups[4]


def test_canonical_smiles_reports_unparsable_input():
    from splitproof.sim.chem import CanonicalSmiles

    result = CanonicalSmiles().pairs_above(["CCO", "@@@nope@@@", None], "molecule", 1.0)
    assert result.unparsable == [1]
    assert result.missing == [1, 2]


def test_different_inorganic_salts_are_not_merged():
    from splitproof.sim.chem import CanonicalSmiles

    backend = CanonicalSmiles()
    if not backend.available():
        pytest.skip("rdkit is not installed")
    groups = backend.pairs_above(["[Na+].[Cl-]", "[Na+].[Br-]", "[K+].[Cl-]", "[Cl-].[Na+]"], "molecule", 1.0).groups
    assert groups is not None
    assert len({groups[0], groups[1], groups[2]}) == 3
    assert groups[0] == groups[3]


def test_the_salt_dictionary_is_parsed_once_however_many_molecules(monkeypatch):
    pytest.importorskip("rdkit")
    from rdkit.Chem import SaltRemover

    from splitproof.sim import chem
    from splitproof.sim.chem import CanonicalSmiles

    built = []
    original = SaltRemover.SaltRemover.__init__

    def counting(self, *args, **kwargs):
        built.append(1)
        original(self, *args, **kwargs)

    getattr(chem._standardisers, "cache_clear", lambda: None)()
    monkeypatch.setattr(SaltRemover.SaltRemover, "__init__", counting)
    CanonicalSmiles().pairs_above(["CCO", "CCN.Cl", "c1ccccc1"] * 50, "molecule", 1.0)
    assert len(built) == 1
