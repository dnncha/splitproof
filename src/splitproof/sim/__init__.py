from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Mapping, Iterable, Iterator, List, Optional, Sequence, Tuple

import math

import numpy as np

from ..errors import BackendUnavailable, SplitproofError

EXACT = "exact"
HEURISTIC = "heuristic"
LOWER_BOUND = "lower_bound"

EXACTNESS_RANK = {EXACT: 2, HEURISTIC: 1, LOWER_BOUND: 0}

DEFAULT_MAX_PAIRS = 20_000_000


class TooManyPairs(SplitproofError):
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
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        )

    @classmethod
    def from_triples(cls, triples: Iterable[Tuple[int, int, float]]) -> "PairSet":
        materialised = list(triples)
        if not materialised:
            return cls.empty()
        rows = np.fromiter((item[0] for item in materialised), dtype=np.int64, count=len(materialised))
        cols = np.fromiter((item[1] for item in materialised), dtype=np.int64, count=len(materialised))
        sims = np.fromiter((item[2] for item in materialised), dtype=np.float64, count=len(materialised))
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

    CHUNK = 1 << 20

    def __iter__(self) -> Iterator[Tuple[int, int, float]]:
        total = int(self.rows.size)
        for start in range(0, total, self.CHUNK):
            stop = min(start + self.CHUNK, total)
            rows = self.rows[start:stop].tolist()
            cols = self.cols[start:stop].tolist()
            sims = self.sims[start:stop].tolist()
            for triple in zip(rows, cols, sims, strict=True):
                yield triple

    def edges(self) -> Iterator[Tuple[int, int]]:
        total = int(self.rows.size)
        for start in range(0, total, self.CHUNK):
            stop = min(start + self.CHUNK, total)
            rows = self.rows[start:stop].tolist()
            cols = self.cols[start:stop].tolist()
            for pair in zip(rows, cols, strict=True):
                yield pair

    def __getitem__(self, position: int) -> Tuple[int, int, float]:
        return (int(self.rows[position]), int(self.cols[position]), float(self.sims[position]))

    @property
    def max_similarity(self) -> Optional[float]:
        return float(self.sims.max()) if self.sims.size else None

    def nbytes(self) -> int:
        return int(self.rows.nbytes + self.cols.nbytes + self.sims.nbytes)


class PairCollector:
    """The only sanctioned way to accumulate similarity pairs.

    Every add checks the budget, so a backend cannot build an unbounded pair list
    by forgetting a guard: there is no other route to a SimilarityResult's pairs.
    """

    __slots__ = ("_backend", "_threshold", "_max_pairs", "_count", "_blocks", "_triples")

    def __init__(self, backend: str, threshold: float, max_pairs: int = DEFAULT_MAX_PAIRS):
        self._backend = backend
        self._threshold = threshold
        self._max_pairs = max_pairs
        self._count = 0
        self._blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._triples: List[Tuple[int, int, float]] = []

    def __len__(self) -> int:
        return self._count

    def _charge(self, amount: int) -> None:
        self._count += amount
        guard_pair_budget(self._count, self._max_pairs, self._backend, self._threshold)

    def add_block(self, rows: np.ndarray, cols: np.ndarray, sims: np.ndarray) -> None:
        if rows.size == 0:
            return
        self._charge(int(rows.size))
        self._blocks.append(
            (
                rows.astype(np.int64, copy=False),
                cols.astype(np.int64, copy=False),
                sims.astype(np.float64, copy=False),
            )
        )

    def add(self, left: int, right: int, similarity: float) -> None:
        self._charge(1)
        self._triples.append((int(left), int(right), float(similarity)))

    def finish(self) -> PairSet:
        if self._triples:
            self._blocks.append(
                (
                    np.fromiter((t[0] for t in self._triples), dtype=np.int64, count=len(self._triples)),
                    np.fromiter((t[1] for t in self._triples), dtype=np.int64, count=len(self._triples)),
                    np.fromiter((t[2] for t in self._triples), dtype=np.float64, count=len(self._triples)),
                )
            )
            self._triples = []
        return PairSet.from_blocks(self._blocks)


SEQUENCE_KINDS = frozenset({"protein", "dna", "rna"})


def normalise_sequence(value: str) -> str:
    """The one residue string every sequence backend compares.

    Whitespace anywhere is removed, because UniProt and GenBank print sequences in
    spaced blocks, and case is folded, because sequence alphabets carry no meaning in
    case. Backends that each normalised their own way let an exact backend miss a pair
    that another backend found.
    """
    return "".join(value.split()).upper()


def guard_candidate_budget(count: int, max_pairs: int, backend: str, threshold: float) -> None:
    """Refuse before enumerating a quadratic candidate set.

    LSH candidate generation happens before any similarity is known, so it has to be
    bounded on its own. The count is what the set already holds plus what the bucket
    about to be enumerated implies, charged before a single tuple is built, because a
    bucket of m members implies m(m-1)/2 distinct pairs and one bucket alone can
    exhaust memory.
    """
    if count <= max_pairs:
        return
    raise TooManyPairs(
        f"{backend} would consider at least {count} candidate pairs at {threshold:g}, over the "
        f"{max_pairs} the run allows",
        remedy=(
            "Candidate generation is bounded separately from the result, because it happens "
            "before any similarity is known. Almost everything is colliding in the same LSH "
            "bucket, so raise the threshold, split on a coarser key, or raise --max-pairs."
        ),
        details={
            "candidate_pairs": count,
            "max_pairs": max_pairs,
            "backend": backend,
            "threshold": threshold,
            "stage": "candidate_generation",
        },
    )


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
    groups: Optional[List[Optional[str]]] = None
    missing: List[int] = field(default_factory=list)
    unparsable: List[int] = field(default_factory=list)
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
            "equivalence_classes": (
                len({label for label in self.groups if label is not None})
                if self.groups is not None
                else None
            ),
            "missing_values": len(self.missing),
            "unparsable_values": len(self.unparsable),
            "params": dict(sorted(self.params.items())),
            "notes": list(self.notes),
        }


class SimilarityBackend:
    name = "abstract"
    metric = "abstract"
    exactness = LOWER_BOUND
    emits = "pairs"
    policy_params: ClassVar[Mapping[str, Tuple[type, Any, Any]]] = {}

    def version(self) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        return True

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
        raise NotImplementedError

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
        order: Optional[Sequence[str]] = None,
    ) -> Dict[int, Optional[Neighbour]]:
        raise NotImplementedError


def tie_ranks(order: Optional[Sequence[str]], size: int) -> "np.ndarray":
    """Stable ranks used only to break equal-similarity ties.

    Without them the nearest neighbour of a query with two equally similar targets
    depends on row order, which changed the 1-NN baseline and broke verification on
    permuted data.
    """
    if order is None:
        return np.arange(size, dtype=np.int64)
    keys = [str(order[index]) for index in range(size)]
    return np.argsort(np.argsort(np.asarray(keys))).astype(np.int64)


BACKENDS: Dict[str, type] = {}


def register(name: str):
    def decorate(cls):
        BACKENDS[name] = cls
        return cls

    return decorate


def backend_table() -> Dict[str, type]:
    from . import chem, duplicate, kmer, mmseqs  # noqa: F401

    return dict(BACKENDS)


def construct(cls, settings: Dict[str, Any]):
    """Build a backend from policy params, which are untrusted input.

    A certificate carries its own policy and `verify` instantiates backends from it,
    so anything reachable here is reachable by whoever wrote the certificate. Only
    the names a backend lists in policy_params are accepted, and each is range
    checked. Executable paths are deliberately not among them.
    """
    allowed = dict(getattr(cls, "policy_params", {}) or {})
    name = getattr(cls, "name", cls.__name__)
    unknown = sorted(set(settings) - set(allowed))
    if unknown:
        raise BackendUnavailable(
            f"backend {name!r} does not take {unknown} from a policy",
            remedy=(
                f"It takes {sorted(allowed)}. Parameters that could name a program or a path "
                "are never settable from a policy, because verifying a certificate would then "
                "run whatever its author chose."
            ),
            details={"backend": name, "given": sorted(settings), "accepted": sorted(allowed)},
        )
    checked: Dict[str, Any] = {}
    for key, value in settings.items():
        kind, low, high = allowed[key]
        if isinstance(value, bool):
            raise BackendUnavailable(
                f"{name}.{key} must be a number, got a boolean",
                details={"backend": name, "param": key},
            )
        if kind is int and isinstance(value, float) and value != int(value):
            raise BackendUnavailable(
                f"{name}.{key} must be a whole number, got {value!r}",
                details={"backend": name, "param": key, "value": str(value)},
            )
        try:
            coerced = kind(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BackendUnavailable(
                f"{name}.{key} must be {kind.__name__}, got {value!r}",
                details={"backend": name, "param": key, "value": str(value)},
            ) from exc
        if isinstance(coerced, float) and not math.isfinite(coerced):
            raise BackendUnavailable(
                f"{name}.{key} must be a finite number, got {value!r}",
                details={"backend": name, "param": key},
            )
        if (low is not None and coerced < low) or (high is not None and coerced > high):
            raise BackendUnavailable(
                f"{name}.{key} must be between {low} and {high}, got {coerced}",
                details={"backend": name, "param": key, "value": coerced, "min": low, "max": high},
            )
        checked[key] = coerced
    return cls(**checked)


def resolve(
    metric: str,
    kind: str,
    count: int,
    *,
    requested: Optional[str] = None,
    exact_max_n: int = 20000,
    settings: Optional[Dict[str, Any]] = None,
):
    table = backend_table()
    settings = settings or {}
    if requested:
        if requested not in table:
            raise BackendUnavailable(
                f"unknown similarity backend {requested!r}",
                remedy=f"Available backends: {sorted(table)}.",
                details={"requested": requested, "available": sorted(table)},
            )
        backend = construct(table[requested], settings)
    elif metric == "identity":
        backend = construct(table["mmseqs2"], settings)
    elif metric == "kmer_jaccard":
        backend = construct(
            table["exact_jaccard"] if count <= exact_max_n else table["minhash"], settings
        )
    elif metric == "exact_duplicate":
        backend = construct(table["exact_duplicate"], settings)
    elif metric == "tanimoto":
        backend = construct(table["tanimoto"], settings)
    elif metric == "canonical_smiles":
        backend = construct(table["canonical_smiles"], settings)
    elif metric == "ring_systems":
        backend = construct(table["ring_systems"], settings)
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
