from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import BackendUnavailable, CutsiteError

EXACT = "exact"
HEURISTIC = "heuristic"
LOWER_BOUND = "lower_bound"

EXACTNESS_RANK = {EXACT: 2, HEURISTIC: 1, LOWER_BOUND: 0}

DEFAULT_MAX_PAIRS = 20_000_000


class TooManyPairs(CutsiteError):
    code = "TOO_MANY_PAIRS"


@dataclass
class Neighbour:
    index: int
    similarity: float


class PairSet:
    __slots__ = ("rows", "cols", "sims")

    def __init__(self, rows: np.ndarray, cols: np.ndarray, sims: np.ndarray):
        self.rows = rows
        self.cols = cols
        self.sims = sims

    @classmethod
    def empty(cls) -> "PairSet":
        return cls(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        )

    @classmethod
    def from_triples(cls, triples: Iterable[Tuple[int, int, float]]) -> "PairSet":
        materialised = list(triples)
        if not materialised:
            return cls.empty()
        rows = np.fromiter((item[0] for item in materialised), dtype=np.int64, count=len(materialised))
        cols = np.fromiter((item[1] for item in materialised), dtype=np.int64, count=len(materialised))
        sims = np.fromiter((item[2] for item in materialised), dtype=np.float32, count=len(materialised))
        return cls(rows, cols, sims).sorted()

    @classmethod
    def from_blocks(cls, blocks: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> "PairSet":
        if not blocks:
            return cls.empty()
        return cls(
            np.concatenate([block[0] for block in blocks]),
            np.concatenate([block[1] for block in blocks]),
            np.concatenate([block[2] for block in blocks]),
        ).sorted()

    def sorted(self) -> "PairSet":
        order = np.lexsort((self.cols, self.rows))
        return PairSet(self.rows[order], self.cols[order], self.sims[order])

    def __len__(self) -> int:
        return int(self.rows.size)

    def __iter__(self) -> Iterator[Tuple[int, int, float]]:
        rows, cols, sims = self.rows.tolist(), self.cols.tolist(), self.sims.tolist()
        return iter(zip(rows, cols, sims))

    def __getitem__(self, position: int) -> Tuple[int, int, float]:
        return (int(self.rows[position]), int(self.cols[position]), float(self.sims[position]))

    @property
    def max_similarity(self) -> Optional[float]:
        return float(self.sims.max()) if self.sims.size else None

    def nbytes(self) -> int:
        return int(self.rows.nbytes + self.cols.nbytes + self.sims.nbytes)


def guard_pair_budget(count: int, max_pairs: int, backend: str, threshold: float) -> None:
    if count <= max_pairs:
        return
    raise TooManyPairs(
        f"{backend} found {count} pairs at or above {threshold:g}, over the {max_pairs} the run allows",
        remedy=(
            "Almost every record is related to almost every other at this threshold, so the pair "
            "graph is the size of the problem. Raise the threshold, split on a coarser key such as "
            "a group or a scaffold, or raise --max-pairs if you have the memory for it."
        ),
        details={"pairs": count, "max_pairs": max_pairs, "backend": backend, "threshold": threshold},
    )


@dataclass
class SimilarityResult:
    backend: str
    backend_version: str
    metric: str
    exactness: str
    threshold: float
    pairs: PairSet = field(default_factory=PairSet.empty)
    missing: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    params: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "metric": self.metric,
            "exactness": self.exactness,
            "threshold": self.threshold,
            "edges": len(self.pairs),
            "missing_values": len(self.missing),
            "params": dict(sorted(self.params.items())),
            "notes": list(self.notes),
        }


class SimilarityBackend:
    name = "abstract"
    metric = "abstract"
    exactness = LOWER_BOUND

    def version(self) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        return True

    def install_hint(self) -> str:
        return "pip install cutsite"

    def pairs_above(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        threshold: float,
        *,
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> SimilarityResult:
        raise NotImplementedError

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
    ) -> Dict[int, Optional[Neighbour]]:
        raise NotImplementedError


def resolve(metric: str, kind: str, count: int, *, requested: Optional[str] = None, exact_max_n: int = 20000):
    from . import chem, duplicate, kmer, mmseqs

    table = {
        "exact_jaccard": kmer.ExactJaccard,
        "minhash": kmer.MinHashJaccard,
        "mmseqs2": mmseqs.MMseqs2Identity,
        "exact_duplicate": duplicate.ExactDuplicate,
        "tanimoto": chem.MorganTanimoto,
        "scaffold": chem.MurckoScaffold,
    }
    if requested:
        if requested not in table:
            raise BackendUnavailable(
                f"unknown similarity backend {requested!r}",
                remedy=f"Available backends: {sorted(table)}.",
                details={"requested": requested, "available": sorted(table)},
            )
        backend = table[requested]()
    elif metric == "identity":
        backend = mmseqs.MMseqs2Identity()
    elif metric == "kmer_jaccard":
        backend = kmer.ExactJaccard() if count <= exact_max_n else kmer.MinHashJaccard()
    elif metric == "exact_duplicate":
        backend = duplicate.ExactDuplicate()
    elif metric == "tanimoto":
        backend = chem.MorganTanimoto()
    elif metric == "scaffold":
        backend = chem.MurckoScaffold()
    else:
        raise BackendUnavailable(
            f"no backend implements metric {metric!r}",
            details={"metric": metric},
        )
    if backend.metric != metric:
        raise BackendUnavailable(
            f"backend {backend.name!r} computes {backend.metric!r}, not the requested {metric!r}",
            remedy="Drop the explicit 'backend' from the constraint, or change its 'metric'.",
            details={"backend": backend.name, "backend_metric": backend.metric, "metric": metric},
        )
    if not backend.available():
        raise BackendUnavailable(
            f"backend {backend.name!r} is not available on this machine",
            remedy=backend.install_hint(),
            details={"backend": backend.name, "metric": metric},
        )
    return backend
