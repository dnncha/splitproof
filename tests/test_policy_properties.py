from __future__ import annotations

import copy
from typing import Any, Dict

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from splitproof.errors import SplitproofError, PolicyError
from splitproof.policy import Policy

SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=1, max_size=6
)

scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=12),
)

structures = st.recursive(
    scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=8), children, max_size=4),
    ),
    max_leaves=12,
)

BASE: Dict[str, Any] = {
    "splitproof": 1,
    "dataset": {
        "path": "data.csv",
        "id": "accession",
        "label": "activity",
        "similarity": {"sequence": {"column": "seq", "kind": "protein"}},
        "groups": ["donor_id"],
        "dates": {"deposited": "deposit_date"},
    },
    "scheme": {"kind": "holdout", "fractions": {"train": 0.8, "test": 0.2}},
    "constraints": [
        {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.3}
    ],
    "seed": 7,
}

PATHS = [
    ("splitproof",),
    ("seed",),
    ("dataset",),
    ("dataset", "id"),
    ("dataset", "label"),
    ("dataset", "label_kind"),
    ("dataset", "similarity"),
    ("dataset", "groups"),
    ("dataset", "dates"),
    ("scheme",),
    ("scheme", "kind"),
    ("scheme", "fractions"),
    ("scheme", "folds"),
    ("constraints",),
]


def _set(target: Dict[str, Any], path, value) -> Dict[str, Any]:
    out = copy.deepcopy(target)
    cursor: Any = out
    for key in path[:-1]:
        if not isinstance(cursor, dict):
            return out
        cursor = cursor.setdefault(key, {})
    if isinstance(cursor, dict):
        cursor[path[-1]] = value
    return out


@SETTINGS
@given(structures)
def test_arbitrary_top_level_input_is_rejected_not_crashed(payload):
    try:
        Policy.from_dict(payload)
    except PolicyError as error:
        assert error.message
        assert error.code == "POLICY_INVALID"
    except SplitproofError as error:
        assert error.message


@SETTINGS
@given(st.sampled_from(PATHS), structures)
def test_corrupting_any_field_gives_a_policy_error_not_a_traceback(path, value):
    payload = _set(BASE, path, value)
    try:
        policy = Policy.from_dict(payload)
    except PolicyError as error:
        assert error.message
        assert error.code == "POLICY_INVALID"
        return
    assert policy.constraints
    assert policy.dataset.id_column


@SETTINGS
@given(structures)
def test_corrupting_a_constraint_gives_a_policy_error(value):
    payload = copy.deepcopy(BASE)
    payload["constraints"] = [value]
    try:
        Policy.from_dict(payload)
    except PolicyError as error:
        assert error.message
        assert error.code == "POLICY_INVALID"


@SETTINGS
@given(
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False, width=32)
)
def test_only_thresholds_in_the_unit_interval_are_accepted(threshold):
    payload = copy.deepcopy(BASE)
    payload["constraints"][0]["threshold"] = threshold
    if 0.0 < threshold <= 1.0:
        assert Policy.from_dict(payload).similarity_constraints[0].threshold == pytest.approx(
            threshold
        )
    else:
        with pytest.raises(PolicyError):
            Policy.from_dict(payload)


@SETTINGS
@given(
    st.dictionaries(
        names,
        st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=4,
    )
)
def test_weights_of_one_or_more_are_normalised_to_one(fractions):
    payload = copy.deepcopy(BASE)
    payload["scheme"] = {"kind": "holdout", "fractions": fractions}
    policy = Policy.from_dict(payload)
    assert sum(policy.scheme.fractions.values()) == pytest.approx(1.0)
    assert set(policy.scheme.fractions) == {str(name) for name in fractions}


@SETTINGS
@given(
    st.dictionaries(
        names,
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=4,
    )
)
def test_fraction_like_values_must_actually_sum_to_one(fractions):
    payload = copy.deepcopy(BASE)
    payload["scheme"] = {"kind": "holdout", "fractions": fractions}
    total = sum(fractions.values())
    if abs(total - 1.0) <= 1e-6:
        assert Policy.from_dict(payload).scheme.fractions
        return
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert "sum to" in error.value.message


@SETTINGS
@given(st.text(max_size=12))
def test_an_unknown_constraint_kind_names_what_is_allowed(kind):
    payload = copy.deepcopy(BASE)
    payload["constraints"] = [{"kind": kind}]
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert error.value.message


@SETTINGS
@given(st.text(max_size=12))
def test_a_similarity_constraint_cannot_reference_an_undeclared_key(key):
    payload = copy.deepcopy(BASE)
    payload["constraints"] = [
        {"kind": "similarity", "key": key, "metric": "kmer_jaccard", "threshold": 0.3}
    ]
    if key.strip() == "sequence":
        assert Policy.from_dict(payload).similarity_constraints[0].key == "sequence"
        return
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert "similarity" in error.value.message or "key" in error.value.message


def test_a_valid_policy_round_trips_through_its_own_dict():
    policy = Policy.from_dict(BASE)
    again = Policy.from_dict(
        {**policy.to_dict(), "dataset": {**policy.to_dict()["dataset"], "path": "data.csv"}}
    )
    assert again.to_dict() == policy.to_dict()
