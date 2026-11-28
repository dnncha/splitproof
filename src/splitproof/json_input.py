from __future__ import annotations

import json


def _unique_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    members: dict[str, object] = {}
    for key, value in pairs:
        if key in members:
            raise ValueError(f"duplicate JSON member {key!r}")
        members[key] = value
    return members


def load_json(text: str) -> object:
    result: object = json.loads(text, object_pairs_hook=_unique_members)
    return result
