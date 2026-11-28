from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

from hypothesis import example, given, settings
from hypothesis import strategies as st

from splitproof import evaluate
from splitproof.assign import DROPPED, Assignment, fold_order
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
    got_total, got_count, worst, examples, got_by_pair = evaluate._crossing_classes(labels, data, assignment)
    assert (got_total, got_count, got_by_pair) == (total, len(crossing), dict(sorted(by_pair.items())))
    assert worst == (1.0 if crossing else None)
    assert len(examples) <= evaluate.MAX_EXAMPLES
    assert bool(examples) == bool(crossing)
    named = [(ids[left], ids[right]) for left, right, _ in examples]
    assert all(a < b for a, b in named), named
    assert named == sorted(named)
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


@st.composite
def _labelled(draw):
    size = draw(st.integers(1, 25))
    ids = draw(st.permutations([f"e{i:02d}" for i in range(size)]))
    values = draw(st.lists(st.sampled_from([None, "d1", "d2", "d3", "NA"]), min_size=size, max_size=size))
    days = draw(st.lists(st.one_of(st.none(), st.integers(0, 40)), min_size=size, max_size=size))
    folds = draw(st.lists(st.sampled_from(FOLDS + [DROPPED]), min_size=size, max_size=size))
    return list(ids), values, days, folds


def _report_or_refusal(check):
    from splitproof.errors import VacuousConstraint

    try:
        return check(), None
    except VacuousConstraint as error:
        return None, error


@settings(max_examples=300, deadline=None)
@given(case=_labelled())
def test_disjoint_check_matches_brute_force(case):
    from splitproof.evaluate import SATISFIED, SATISFIED_UNPROVEN, VIOLATED
    from splitproof.policy import DisjointConstraint

    ids, values, _days, folds = case
    data = Dataset(ids=ids, columns={"id": list(ids), "grp": values})
    assignment = Assignment(fold_of=dict(enumerate(folds)), folds=list(FOLDS), stats=[])
    live = [i for i in range(len(ids)) if folds[i] != DROPPED]
    bound = [i for i in live if values[i] is not None]
    spread = {}
    for i in bound:
        spread.setdefault(values[i], set()).add(folds[i])
    offenders = sorted(value for value, where in spread.items() if len(where) > 1)
    report, refusal = _report_or_refusal(lambda: evaluate._check_disjoint(DisjointConstraint(key="grp"), data, assignment))
    if not bound:
        assert refusal is not None
        return
    assert report is not None
    expected = VIOLATED if offenders else (SATISFIED_UNPROVEN if len(bound) < len(live) else SATISFIED)
    assert report.status == expected
    assert report.measured == {
        "distinct_values": len(spread),
        "values_in_more_than_one_fold": len(offenders),
        "missing_values": sum(value is None for value in values),
    }
    assert report.examples == [{"value": value, "folds": sorted(spread[value])} for value in offenders]
    assert bool(report.notes) == (not offenders and len(bound) < len(live))


@settings(max_examples=300, deadline=None)
@given(case=_labelled(), cutoff=st.integers(0, 40))
def test_temporal_check_matches_brute_force(case, cutoff):
    import datetime

    from splitproof.evaluate import SATISFIED, SATISFIED_UNPROVEN, VIOLATED
    from splitproof.policy import TemporalConstraint

    ids, _values, days, folds = case
    origin = datetime.date(2020, 1, 1)
    dates = [None if day is None else (origin + datetime.timedelta(days=day)).isoformat() for day in days]
    data = Dataset(ids=ids, columns={"id": list(ids), "when": dates})
    assignment = Assignment(fold_of=dict(enumerate(folds)), folds=list(FOLDS), stats=[])
    constraint = TemporalConstraint(key="w", cutoff=origin + datetime.timedelta(days=cutoff))
    policy = SimpleNamespace(dataset=SimpleNamespace(dates={"w": "when"}))
    live = [i for i in range(len(ids)) if folds[i] != DROPPED]
    dated = [i for i in live if days[i] is not None]
    wrong = sorted(ids[i] for i in dated if days[i] >= cutoff and folds[i] == "train")
    report, refusal = _report_or_refusal(lambda: evaluate._check_temporal(constraint, policy, data, assignment))
    if not dated:
        assert refusal is not None
        return
    assert report is not None
    expected = VIOLATED if wrong else (SATISFIED_UNPROVEN if len(dated) < len(live) else SATISFIED)
    assert report.status == expected
    assert report.measured["items_checked"] == len(dated)
    assert report.measured["items_after_the_cutoff_in_train"] == len(wrong)
    assert report.measured["items_without_a_date"] == len(live) - len(dated)
    assert [example["id"] for example in report.examples] == wrong[: evaluate.MAX_EXAMPLES]
    assert all(example["fold"] == "train" for example in report.examples)
    assert bool(report.notes) == (not wrong and len(dated) < len(live))


@settings(max_examples=400, deadline=None)
@given(
    folds=st.lists(st.sampled_from(["train", "test", "", "__dropped__", "__unassigned__"]), min_size=1, max_size=40),
    allow_dropped=st.booleans(),
)
@example(folds=["train", "test"] + ["__unassigned__"] * 12, allow_dropped=False)
def test_audit_fold_check_refuses_exactly_the_ungradable_splits(folds, allow_dropped):
    from splitproof.assign import UNASSIGNED, check_audit_folds
    from splitproof.errors import DataError

    ids = [f"e{i:02d}" for i in range(len(folds))]
    data = Dataset(ids=ids, columns={"id": ids})
    named = {fold for fold in folds if fold != DROPPED}
    should_refuse = (
        UNASSIGNED in folds
        or (DROPPED in folds and not allow_dropped)
        or "" in named
        or len(named) < 2
    )
    try:
        check_audit_folds(data, dict(enumerate(folds)), "column 'fold'", allow_dropped=allow_dropped)
    except DataError as error:
        assert should_refuse, (folds, allow_dropped, error.message)
        if UNASSIGNED in folds:
            blank = sorted(ids[i] for i, fold in enumerate(folds) if fold == UNASSIGNED)
            assert error.details == {"missing": len(blank), "examples": blank[:10]}
        return
    assert not should_refuse, (folds, allow_dropped)


def test_crossing_fold_pairs_have_distinct_keys_in_both_backends():
    ids = ["a", "b", "c", "d", "e", "f", "g", "h"]
    folds = ["a", "b|c", "a|b", "c", "fold2", "fold10", "a", "b%7Cc"]
    data = Dataset(ids=ids, columns={"id": ids})
    assignment = Assignment(fold_of=dict(enumerate(folds)), folds=sorted(set(folds), key=fold_order), stats=[])
    labels = ["one", "one", "two", "two", "three", "three", "four", "four"]
    result = SimpleNamespace(pairs=PairSet.from_triples([
        (0, 1, 1.0), (2, 3, 1.0), (4, 5, 1.0), (6, 7, 1.0),
    ]))
    expected = {"a|b%7Cc": 1, "a%7Cb|c": 1, "a|b%257Cc": 1, "fold2|fold10": 1}
    assert evaluate._crossing_classes(labels, data, assignment)[-1] == expected
    assert evaluate._crossing(result, data, assignment)[-1] == expected
