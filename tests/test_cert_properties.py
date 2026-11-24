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


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key in value:
            yield from _leaves(value[key], path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, path + (index,))
    else:
        yield path, value


def _edited(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "x"
    return 0


def _with(document, path, replacement):
    import copy

    out = copy.deepcopy(document)
    target = out
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = replacement
    return out


def _without(document, path):
    import copy

    out = copy.deepcopy(document)
    target = out
    for step in path[:-1]:
        target = target[step]
    del target[path[-1]]
    return out


def _real_certificate(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    from splitproof import api
    from splitproof.policy import Policy

    examples = Path(__file__).resolve().parents[1] / "examples"
    for name in ("proteins.yaml", "proteins.csv"):
        shutil.copy(examples / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    return api.make_split(Policy.load(tmp_path / "proteins.yaml")).certificate


UNVERIFIED = ("created", "environment", "audited", "source")


def test_only_provenance_fields_escape_verification():
    from splitproof.cert import VOLATILE

    assert set(VOLATILE) == set(UNVERIFIED)


def test_every_edit_to_a_verified_field_fails_verification(tmp_path, monkeypatch):
    from splitproof.cert import verify

    VOLATILE = UNVERIFIED

    certificate = _real_certificate(tmp_path, monkeypatch)
    assert verify(certificate, certificate).ok
    checked = 0
    for path, value in _leaves(certificate):
        if path[0] in VOLATILE:
            continue
        result = verify(_with(certificate, path, _edited(value)), certificate)
        assert not result.ok, f"editing {path} from {value!r} still verifies"
        checked += 1
    assert checked > 100


def test_every_removed_or_added_field_fails_verification(tmp_path, monkeypatch):
    from splitproof.cert import verify

    VOLATILE = UNVERIFIED

    certificate = _real_certificate(tmp_path, monkeypatch)
    for path, _value in _leaves(certificate):
        if path[0] in VOLATILE or not isinstance(path[-1], str):
            continue
        assert not verify(_without(certificate, path), certificate).ok, f"removing {path} still verifies"
    assert not verify({**certificate, "forged": True}, certificate).ok
    nested = _with(certificate, ("units", "forged"), 1)
    assert not verify(nested, certificate).ok
