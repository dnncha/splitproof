from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from splitproof.cert import first_difference

_SCALARS = st.none() | st.booleans() | st.integers(-3, 3) | st.sampled_from([0.0, 1.0, 2.5, -1.0]) | st.sampled_from(["a", "b", ""])
_JSON = st.recursive(
    _SCALARS,
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.sampled_from(["x", "y", "z"]), children, max_size=3),
    max_leaves=10,
)


def _same(left, right) -> bool:
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, list) or isinstance(right, list):
        return isinstance(left, list) and isinstance(right, list) and len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    numeric = (int, float)
    if isinstance(left, numeric) and isinstance(right, numeric):
        return left == right
    return type(left) is type(right) and left == right


@settings(max_examples=500, deadline=None)
@given(stored=_JSON, fresh=_JSON)
def test_a_difference_is_reported_exactly_when_the_values_differ(stored, fresh):
    difference = first_difference(stored, fresh)
    assert (difference is None) == _same(stored, fresh), (stored, fresh, difference)
    if difference is not None:
        assert difference


@settings(max_examples=300, deadline=None)
@given(value=_JSON)
def test_a_value_never_differs_from_itself(value):
    assert first_difference(value, value) is None
