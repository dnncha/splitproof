from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..errors import BackendUnavailable

EXACT = "exact"
HEURISTIC = "heuristic"
LOWER_BOUND = "lower_bound"

EXACTNESS_RANK = {EXACT: 2, HEURISTIC: 1, LOWER_BOUND: 0}


@dataclass
class Neighbour:
    index: int
    similarity: float


@dataclass
class SimilarityResult:
    backend: str
    backend_version: str
    metric: str
    exactness: str
    threshold: float
    pairs: List[Tuple[int, int, float]] = field(default_factory=list)
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

    def pairs_above(self, values: Sequence[Optional[str]], kind: str, threshold: float) -> SimilarityResult:
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
