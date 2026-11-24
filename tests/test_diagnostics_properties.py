from __future__ import annotations

import math
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from splitproof.evaluate import _distribution, _nn_baseline, _spearman

_FINITE = st.floats(-1e6, 1e6, allow_nan=False)


@settings(max_examples=300, deadline=None)
@given(left=st.lists(_FINITE, min_size=3, max_size=30), right=st.lists(_FINITE, min_size=3, max_size=30))
def test_spearman_matches_scipy_on_finite_input(left, right):
    from scipy.stats import spearmanr

    size = min(len(left), len(right))
    left, right = left[:size], right[:size]
    got = _spearman(left, right)
    if len(set(left)) < 2 or len(set(right)) < 2:
        assert got is None
        return
    expected = spearmanr(left, right).statistic
    if expected != expected:
        assert got is None
    else:
        assert got == round(float(expected), 6)


@settings(max_examples=300, deadline=None)
@given(values=st.lists(_FINITE, min_size=1, max_size=40))
def test_distribution_summaries_are_drawn_from_the_values_and_ordered(values):
    summary = _distribution(values)
    assert summary["min"] == round(min(values), 6)
    assert summary["max"] == round(max(values), 6)
    assert summary["mean"] == round(sum(sorted(values)) / len(values), 6)
    rounded = {round(value, 6) for value in values}
    for key in ("p50", "p90", "p99"):
        assert summary[key] in rounded
    assert summary["min"] <= summary["p50"] <= summary["p90"] <= summary["p99"] <= summary["max"]


def test_distribution_of_nothing_is_empty():
    assert _distribution([]) == {}


def test_one_non_finite_label_does_not_erase_the_numeric_baseline():
    labels = ["1", "2", "3", "4", "5", "nan", "1", "2", "3", "4", "5", "inf"]
    neighbours = {query: SimpleNamespace(index=query + 6) for query in range(6)}
    result = _nn_baseline(neighbours, labels, "numeric", list(range(6)))
    assert result["compared"] == 5
    assert result["value"] == 1.0
    assert math.isfinite(result["value"])


def test_categorical_baseline_counts_exact_matches():
    labels = ["a", "b", "a", None, "a", "b", "b", "a"]
    neighbours = {0: SimpleNamespace(index=4), 1: SimpleNamespace(index=5), 2: SimpleNamespace(index=6), 3: SimpleNamespace(index=7)}
    result = _nn_baseline(neighbours, labels, "categorical", [0, 1, 2, 3])
    assert (result["compared"], result["metric"], result["value"]) == (3, "accuracy", round(2 / 3, 6))
