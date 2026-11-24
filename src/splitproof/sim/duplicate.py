from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import (
    DEFAULT_MAX_PAIRS,
    EXACT,
    Neighbour,
    PairSet,
    SimilarityBackend,
    SimilarityResult,
    PairCollector,
    register,
)


SEQUENCE_KINDS = {"protein", "dna", "rna"}


def _normalise(value: str, kind: str) -> str:
    """Strip whitespace, and fold case only where case carries no meaning.

    In SMILES case is semantic: lowercase means aromatic, so upper-casing made
    benzene and cyclohexane the same molecule. Sequence alphabets are
    case-insensitive, so folding there is correct.
    """
    text = "".join(value.split())
    return text.upper() if kind in SEQUENCE_KINDS else text


@register("exact_duplicate")
class ExactDuplicate(SimilarityBackend):
    name = "exact_duplicate"
    metric = "exact_duplicate"
    exactness = EXACT
    emits = "classes"

    def version(self) -> str:
        return "builtin-1"

    def install_hint(self) -> str:
        return "pip install splitproof"

    def pairs_above(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        threshold: float,
        *,
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> SimilarityResult:
        buckets: Dict[str, List[int]] = {}
        missing: List[int] = []
        for i, value in enumerate(values):
            if value is None:
                missing.append(i)
                continue
            buckets.setdefault(_normalise(value, kind), []).append(i)
        groups: List[Optional[str]] = [None] * len(values)
        for label, members in buckets.items():
            for member in members:
                groups[member] = label
        folded = "and case-folded " if kind in SEQUENCE_KINDS else ""
        notes = [
            f"compares whitespace-stripped {folded}values for exact equality; equal values "
            "form an equivalence class, so no pairwise edge list is ever built"
        ]
        if missing:
            notes.append(f"{len(missing)} entities have no value for this key and are treated as unrelated")
        return SimilarityResult(
            backend=self.name,
            backend_version=self.version(),
            metric=self.metric,
            exactness=EXACT,
            threshold=threshold,
            groups=groups,
            missing=missing,
            notes=notes,
            params={"distinct_values": len(buckets)},
        )

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
        order: Optional[Sequence[str]] = None,
    ) -> Dict[int, Optional[Neighbour]]:
        key = (lambda index: str(order[index])) if order is not None else (lambda index: index)
        lookup: Dict[str, int] = {}
        for i in sorted(set(int(t) for t in targets), key=key):
            value = values[i]
            if value is not None:
                lookup.setdefault(_normalise(value, kind), i)
        out: Dict[int, Optional[Neighbour]] = {}
        for raw in queries:
            query = int(raw)
            value = values[query]
            match = lookup.get(_normalise(value, kind)) if value is not None else None
            if match is not None and match != query:
                out[query] = Neighbour(index=match, similarity=1.0)
            else:
                out[query] = None
        return out
