from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from splitproof import evaluate
from splitproof.assign import DROPPED, Assignment
from splitproof.dataset import Dataset
from splitproof.sim import PairSet

FOLDS = ["test", "train", "valid"]


@st.composite
def _cases(draw):
    size = draw(st.integers(2, 30))
    ids = draw(st.permutations([f"e{i:02d}" for i in range(size)]))
    labels = draw(st.lists(st.sampled_from([None, "a", "b", "c", "d"]), min_size=size, max_size=size))
    folds = draw(st.lists(st.sampled_from(FOLDS + [DROPPED]), min_size=size, max_size=size))
    return list(ids), labels, folds


def _setting(ids, folds):
    data = Dataset(ids=ids, columns={"id": list(ids)})
    assignment = Assignment(fold_of=dict(enumerate(folds)), folds=list(FOLDS), stats=[])
    return data, assignment


def _brute(pairs, folds):
    total = len(pairs)
    crossing = [(i, j) for i, j in pairs if DROPPED not in (folds[i], folds[j]) and folds[i] != folds[j]]
    by_pair = {}
    for i, j in crossing:
        key = "|".join(sorted((folds[i], folds[j])))
        by_pair[key] = by_pair.get(key, 0) + 1
    return total, crossing, by_pair


@settings(max_examples=300, deadline=None)
@given(case=_cases())
def test_class_crossings_match_brute_force_enumeration(case):
    ids, labels, folds = case
    data, assignment = _setting(ids, folds)
    kept = [i for i in range(len(ids)) if labels[i] is not None and folds[i] != DROPPED]
    pairs = [(i, j) for i, j in combinations(kept, 2) if labels[i] == labels[j]]
    total, crossing, by_pair = _brute(pairs, folds)
    result = SimpleNamespace(groups=labels)
    got_total, got_count, worst, examples, got_by_pair = evaluate._crossing_classes(result, data, assignment)
    assert (got_total, got_count, got_by_pair) == (total, len(crossing), dict(sorted(by_pair.items())))
    assert worst == (1.0 if crossing else None)
    assert len(examples) <= evaluate.MAX_EXAMPLES
    assert bool(examples) == bool(crossing)
    for left, right, similarity in examples:
        assert similarity == 1.0
        assert labels[left] == labels[right]
        assert folds[left] != folds[right] and DROPPED not in (folds[left], folds[right])


@settings(max_examples=300, deadline=None)
@given(case=_cases(), sims=st.lists(st.sampled_from([0.5, 0.75, 0.9, 1.0]), min_size=900, max_size=900))
def test_pair_crossings_match_brute_force_enumeration(case, sims):
    ids, labels, folds = case
    data, assignment = _setting(ids, folds)
    triples = [
        (i, j, sims[(i * 31 + j) % len(sims)])
        for i, j in combinations(range(len(ids)), 2)
        if labels[i] is not None and labels[i] == labels[j]
    ]
    result = SimpleNamespace(pairs=PairSet.from_triples(triples))
    _, crossing, by_pair = _brute([(i, j) for i, j, _ in triples], folds)
    count, worst, examples, got_by_pair = evaluate._crossing(result, data, assignment)
    assert (count, got_by_pair) == (len(crossing), dict(sorted(by_pair.items())))
    weight = {(i, j): s for i, j, s in triples}
    ranked = sorted(
        ((-weight[(i, j)], *sorted((ids[i], ids[j]))) for i, j in crossing)
    )[: evaluate.MAX_EXAMPLES]
    assert [(-s, a, b) for s, a, b in ranked] == [(s, ids[x], ids[y]) for x, y, s in examples]
    assert worst == (max(weight[p] for p in crossing) if crossing else None)
