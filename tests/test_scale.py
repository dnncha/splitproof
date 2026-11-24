from __future__ import annotations

import numpy as np
import pytest

from cutsite.sim import DEFAULT_MAX_PAIRS, PairSet, TooManyPairs, guard_pair_budget
from cutsite.sim.duplicate import ExactDuplicate
from cutsite.sim.kmer import ExactJaccard, MinHashJaccard

from conftest import make_rows


def test_pairset_behaves_like_a_sequence_of_triples():
    pairs = PairSet.from_triples([(2, 3, 0.5), (0, 1, 0.9)])
    assert len(pairs) == 2
    assert pairs[0] == (0, 1, pytest.approx(0.9))
    assert [row for row, _c, _s in pairs] == [0, 2]
    assert pairs.max_similarity == pytest.approx(0.9)


def test_empty_pairset_has_no_maximum():
    pairs = PairSet.empty()
    assert len(pairs) == 0
    assert pairs.max_similarity is None
    assert list(pairs) == []


def test_pairset_stores_edges_compactly():
    count = 100_000
    rows = np.arange(count, dtype=np.int64)
    pairs = PairSet(rows, rows + 1, np.full(count, 0.5, dtype=np.float32))
    assert pairs.nbytes() < 2_200_000
    assert len(pairs) == count


def test_guard_allows_a_count_within_budget():
    guard_pair_budget(10, 10, "exact_jaccard", 0.3)


def test_guard_refuses_beyond_budget_with_a_remedy():
    with pytest.raises(TooManyPairs) as error:
        guard_pair_budget(11, 10, "exact_jaccard", 0.3)
    assert error.value.code == "TOO_MANY_PAIRS"
    assert error.value.details["pairs"] == 11
    assert "Raise the threshold" in error.value.remedy


def test_dense_data_refuses_instead_of_exhausting_memory():
    rows = make_rows(families=1, members=400, mutation_rate=0.05, seed=4)
    values = [row["sequence"] for row in rows]
    with pytest.raises(TooManyPairs) as error:
        ExactJaccard().pairs_above(values, "protein", 0.2, max_pairs=1000)
    assert error.value.details["backend"] == "exact_jaccard"
    assert error.value.details["max_pairs"] == 1000


def test_minhash_respects_the_same_budget():
    rows = make_rows(families=1, members=300, mutation_rate=0.05, seed=4)
    values = [row["sequence"] for row in rows]
    with pytest.raises(TooManyPairs):
        MinHashJaccard().pairs_above(values, "protein", 0.2, max_pairs=500)


def test_an_equivalence_class_backend_handles_a_huge_bucket_without_pairs():
    values = ["ACDEFGHIKLMNP"] * 20000
    result = ExactDuplicate().pairs_above(values, "protein", 1.0, max_pairs=10)
    assert result.groups is not None
    assert len(result.pairs) == 0
    assert len({label for label in result.groups if label}) == 1


def test_default_budget_is_generous_enough_for_ordinary_work():
    rows = make_rows(families=60, members=5, seed=8)
    values = [row["sequence"] for row in rows]
    result = ExactJaccard().pairs_above(values, "protein", 0.3)
    assert len(result.pairs) > 0
    assert len(result.pairs) < DEFAULT_MAX_PAIRS


def test_exact_and_minhash_agree_on_the_strongest_pair():
    rows = make_rows(families=10, members=4, seed=6)
    values = [row["sequence"] for row in rows]
    exact = ExactJaccard().pairs_above(values, "protein", 0.3)
    approximate = MinHashJaccard().pairs_above(values, "protein", 0.3)
    assert exact.pairs.max_similarity == pytest.approx(approximate.pairs.max_similarity, abs=1e-6)
