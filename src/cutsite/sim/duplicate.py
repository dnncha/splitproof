from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import EXACT, Neighbour, SimilarityBackend, SimilarityResult


def _normalise(value: str) -> str:
    return "".join(value.split()).upper()


class ExactDuplicate(SimilarityBackend):
    name = "exact_duplicate"
    metric = "exact_duplicate"
    exactness = EXACT

    def version(self) -> str:
        return "builtin-1"

    def install_hint(self) -> str:
        return "pip install cutsite"

    def pairs_above(self, values: Sequence[Optional[str]], kind: str, threshold: float) -> SimilarityResult:
        buckets: Dict[str, List[int]] = {}
        missing: List[int] = []
        for i, value in enumerate(values):
            if value is None:
                missing.append(i)
                continue
            buckets.setdefault(_normalise(value), []).append(i)
        pairs: List[Tuple[int, int, float]] = []
        for members in buckets.values():
            for a_pos in range(len(members)):
                for b_pos in range(a_pos + 1, len(members)):
                    pairs.append((members[a_pos], members[b_pos], 1.0))
        pairs.sort()
        notes = ["compares whitespace-stripped, upper-cased values for exact equality"]
        if missing:
            notes.append(f"{len(missing)} entities have no value for this key and are treated as unrelated")
        return SimilarityResult(
            backend=self.name,
            backend_version=self.version(),
            metric=self.metric,
            exactness=EXACT,
            threshold=threshold,
            pairs=pairs,
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
    ) -> Dict[int, Optional[Neighbour]]:
        lookup: Dict[str, int] = {}
        for i in sorted(set(int(t) for t in targets)):
            value = values[i]
            if value is not None:
                lookup.setdefault(_normalise(value), i)
        out: Dict[int, Optional[Neighbour]] = {}
        for raw in queries:
            query = int(raw)
            value = values[query]
            match = lookup.get(_normalise(value)) if value is not None else None
            if match is not None and match != query:
                out[query] = Neighbour(index=match, similarity=1.0)
            else:
                out[query] = None
        return out
