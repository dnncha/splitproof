from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pytest

from splitproof.sim import DEFAULT_MAX_PAIRS, PairCollector, SimilarityResult, TooManyPairs, backend_table

from conftest import make_rows

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "no_unbounded_pairs.py"

DENSE_VALUES = ["ACDEFGHIKLMNPQRSTVWY" * 8] * 60
DENSE_MOLECULES = ["c1ccc(C2=NCCNc3ccccc32)cc1"] * 60
MOLECULE_METRICS = {"tanimoto", "ring_systems", "canonical_smiles"}


def _run_checker(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(root)], capture_output=True, text=True, check=False
    )


def test_the_real_source_tree_is_clean():
    result = _run_checker(ROOT)
    assert result.returncode == 0, result.stderr


def test_checker_catches_a_planted_dense_materialisation(tmp_path: Path):
    planted = tmp_path / "src" / "splitproof" / "sim"
    planted.mkdir(parents=True)
    (planted / "rogue.py").write_text("def f(m):\n    return (m @ m.T).toarray()\n", encoding="utf-8")
    result = _run_checker(tmp_path)
    assert result.returncode == 1
    assert "dense-materialisation" in result.stderr
    assert "rogue.py:2" in result.stderr


def test_checker_catches_a_pairset_built_outside_the_collector(tmp_path: Path):
    planted = tmp_path / "src" / "splitproof" / "sim"
    planted.mkdir(parents=True)
    (planted / "rogue.py").write_text(
        "def f(t):\n    return PairSet.from_triples(t)\n", encoding="utf-8"
    )
    result = _run_checker(tmp_path)
    assert result.returncode == 1
    assert "unbudgeted-pairs" in result.stderr


def test_checker_honours_an_explicit_justification(tmp_path: Path):
    planted = tmp_path / "src" / "splitproof" / "sim"
    planted.mkdir(parents=True)
    (planted / "ok.py").write_text(
        "def f(m):\n    return m.toarray()  # splitproof: allow-unbounded, 3x3 fixture\n", encoding="utf-8"
    )
    assert _run_checker(tmp_path).returncode == 0


@pytest.mark.parametrize("name", sorted(backend_table()))
def test_every_registered_backend_charges_the_pair_budget(name: str):
    backend = backend_table()[name]()
    if not backend.available():
        pytest.skip(f"{name} is not installed on this machine")
    molecular = backend.metric in MOLECULE_METRICS
    values = DENSE_MOLECULES if molecular else DENSE_VALUES
    kind = "molecule" if molecular else "protein"
    threshold = 1.0 if backend.metric == "ring_systems" else 0.05
    if backend.emits == "classes":
        result = backend.pairs_above(values, kind, threshold, max_pairs=5)
        assert result.groups is not None
        assert len(result.pairs) == 0
        assert len({label for label in result.groups if label}) == 1
        return
    with pytest.raises(TooManyPairs) as error:
        backend.pairs_above(values, kind, threshold, max_pairs=5)
    assert error.value.details["backend"] == backend.name
    assert error.value.details["max_pairs"] == 5


@pytest.mark.parametrize("name", sorted(backend_table()))
def test_every_registered_backend_accepts_the_budget_keyword(name: str):
    backend = backend_table()[name]()
    signature = backend.pairs_above.__func__.__code__
    assert "max_pairs" in signature.co_varnames[: signature.co_argcount + signature.co_kwonlyargcount]


def test_a_backend_that_bypasses_the_collector_is_caught_by_the_budget_test():
    class Rogue:
        name = "rogue"
        metric = "kmer_jaccard"

        def available(self) -> bool:
            return True

        def pairs_above(
            self,
            values: Sequence[Optional[str]],
            kind: str,
            threshold: float,
            *,
            max_pairs: int = DEFAULT_MAX_PAIRS,
        ) -> SimilarityResult:
            from splitproof.sim import PairSet

            triples: List[tuple] = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    triples.append((i, j, 1.0))
            return SimilarityResult(
                backend=self.name,
                backend_version="rogue",
                metric=self.metric,
                exactness="exact",
                threshold=threshold,
                pairs=PairSet.from_triples(triples),
            )

    rogue = Rogue()
    result = rogue.pairs_above(DENSE_VALUES, "protein", 0.05, max_pairs=5)
    assert len(result.pairs) > 5
    with pytest.raises(TooManyPairs):
        collector = PairCollector(rogue.name, 0.05, 5)
        for left, right, sim in result.pairs:
            collector.add(left, right, sim)


def test_collector_charges_blocks_as_well_as_singles():
    import numpy as np

    collector = PairCollector("test", 0.5, 10)
    collector.add_block(
        np.arange(6, dtype=np.int64), np.arange(6, dtype=np.int64) + 1, np.full(6, 0.9, dtype=np.float32)
    )
    assert len(collector) == 6
    with pytest.raises(TooManyPairs):
        collector.add_block(
            np.arange(6, dtype=np.int64), np.arange(6, dtype=np.int64) + 1, np.full(6, 0.9, dtype=np.float32)
        )


def test_crossing_pairs_are_counted_without_materialising_them(policy_dict, tmp_path: Path):
    from conftest import write_csv
    from splitproof import api
    from splitproof.policy import Policy

    rows = make_rows(families=30, members=6, seed=14)
    for position, row in enumerate(rows):
        row["fold"] = "train" if position % 2 == 0 else "test"
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "leaky.csv", rows))
    outcome = api.audit(Policy.from_dict(policy_dict), split_column="fold", skip_neighbours=True)
    measured = outcome.reports[0].measured
    assert measured["cross_fold_pairs_at_or_above_threshold"] > 0
    assert len(outcome.reports[0].examples) <= 20
    assert measured["max_cross_fold_similarity"] >= max(
        example["similarity"] for example in outcome.reports[0].examples
    )
