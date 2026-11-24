from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import (
    DEFAULT_MAX_PAIRS,
    EXACT,
    LOWER_BOUND,
    Neighbour,
    PairSet,
    SimilarityBackend,
    SimilarityResult,
    PairCollector,
    guard_candidate_budget,
    tie_ranks,
    register,
)

DEFAULT_K = {"protein": 5, "dna": 12, "rna": 12, "text": 4, "molecule": 4}

_MERSENNE = (1 << 61) - 1


def stable_hash(token: str, salt: bytes) -> int:
    """A token hash that does not change between processes.

    Python's built-in hash() is salted per process, so using it here made the MinHash
    signatures, and therefore the split, depend on which interpreter run produced them.
    """
    return int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=salt).digest(), "little"
    )


def kmer_size(kind: str) -> int:
    return DEFAULT_K.get(kind, 5)


def tokenise(value: str, k: int) -> List[str]:
    text = value.strip().upper()
    if not text:
        return []
    if len(text) <= k:
        return [text]
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def _matrix(values: Sequence[Optional[str]], k: int):
    from scipy.sparse import csr_matrix

    vocab: Dict[str, int] = {}
    indices: List[int] = []
    indptr: List[int] = [0]
    missing: List[int] = []
    for i, value in enumerate(values):
        if value is None:
            missing.append(i)
            indptr.append(len(indices))
            continue
        tokens = tokenise(value, k)
        if not tokens:
            missing.append(i)
            indptr.append(len(indices))
            continue
        cols = set()
        for token in tokens:
            col = vocab.get(token)
            if col is None:
                col = len(vocab)
                vocab[token] = col
            cols.add(col)
        indices.extend(sorted(cols))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    shape = (len(values), max(len(vocab), 1))
    matrix = csr_matrix((data, np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)), shape=shape)
    return matrix, np.asarray(matrix.sum(axis=1)).ravel(), missing, len(vocab)


def _chunk_rows(nnz_per_row: np.ndarray, budget: int = 4_000_000) -> int:
    mean_nnz = float(nnz_per_row.mean()) if nnz_per_row.size else 1.0
    return max(1, min(2048, int(budget / max(mean_nnz, 1.0))))


@register("exact_jaccard")
class ExactJaccard(SimilarityBackend):
    name = "exact_jaccard"
    metric = "kmer_jaccard"
    exactness = EXACT
    policy_params = {"k": (int, 1, 64)}

    def __init__(self, k: Optional[int] = None):
        self.k = k

    def size_for(self, kind: str) -> int:
        return self.k if self.k else kmer_size(kind)

    def version(self) -> str:
        import scipy

        return f"scipy-{scipy.__version__}"

    def install_hint(self) -> str:
        return "pip install splitproof (scipy is a core dependency)"

    def pairs_above(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        threshold: float,
        *,
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> SimilarityResult:
        k = self.size_for(kind)
        matrix, sizes, missing, vocab_size = _matrix(values, k)
        sizes = sizes.astype(np.float64)
        collector = PairCollector(self.name, threshold, max_pairs)
        n = matrix.shape[0]
        transposed = matrix.T.tocsr()
        step = _chunk_rows(np.diff(matrix.indptr))
        for start in range(0, n, step):
            stop = min(start + step, n)
            block = (matrix[start:stop] @ transposed).tocoo()
            if block.nnz == 0:
                continue
            rows = block.row.astype(np.int64) + start
            cols = block.col.astype(np.int64)
            inter = block.data
            keep = rows < cols
            if not keep.any():
                continue
            rows, cols, inter = rows[keep], cols[keep], inter[keep]
            inter = inter.astype(np.float64)
            union = sizes[rows] + sizes[cols] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(union > 0, inter / union, 0.0)
            hit = sim >= threshold
            if not hit.any():
                continue
            collector.add_block(rows[hit], cols[hit], sim[hit])
        pairs = collector.finish()
        notes: List[str] = []
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
            params={"k": k, "vocabulary": vocab_size},
        )

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
        order: Optional[Sequence[str]] = None,
    ) -> Dict[int, Optional[Neighbour]]:
        k = self.size_for(kind)
        ranks = tie_ranks(order, len(values))
        matrix, sizes, _missing, _vocab = _matrix(values, k)
        sizes = sizes.astype(np.float64)
        target_index = np.asarray(sorted(set(targets)), dtype=np.int64)
        out: Dict[int, Optional[Neighbour]] = {int(q): None for q in queries}
        if target_index.size == 0:
            return out
        target_matrix = matrix[target_index].T.tocsr()
        target_sizes = sizes[target_index].astype(np.float64)
        query_index = np.asarray(sorted(set(int(q) for q in queries)), dtype=np.int64)
        step = _chunk_rows(np.diff(matrix.indptr))
        for start in range(0, query_index.size, step):
            rows = query_index[start : start + step]
            block = (matrix[rows] @ target_matrix).tocoo()
            if block.nnz == 0:
                continue
            local_rows = block.row.astype(np.int64)
            local_cols = block.col.astype(np.int64)
            global_rows = rows[local_rows]
            global_cols = target_index[local_cols]
            inter = block.data.astype(np.float64)
            union = sizes[global_rows] + target_sizes[local_cols] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(union > 0, inter / union, 0.0)
            keep = global_rows != global_cols
            if not keep.any():
                continue
            global_rows, global_cols, sim = global_rows[keep], global_cols[keep], sim[keep]
            ordering = np.lexsort((ranks[global_cols], -sim, global_rows))
            global_rows, global_cols, sim = (
                global_rows[ordering],
                global_cols[ordering],
                sim[ordering],
            )
            _unique, first = np.unique(global_rows, return_index=True)
            for position in first:
                out[int(global_rows[position])] = Neighbour(
                    index=int(global_cols[position]), similarity=float(sim[position])
                )
        return out


@register("minhash")
class MinHashJaccard(SimilarityBackend):
    name = "minhash"
    metric = "kmer_jaccard"
    exactness = LOWER_BOUND
    policy_params = {
        "k": (int, 1, 64),
        "num_perm": (int, 16, 4096),
        "seed": (int, 0, 2**31 - 1),
    }

    def __init__(self, num_perm: int = 128, seed: int = 7, k: Optional[int] = None):
        self.num_perm = num_perm
        self.seed = seed
        self.k = k

    def size_for(self, kind: str) -> int:
        return self.k if self.k else kmer_size(kind)

    def version(self) -> str:
        return f"builtin-minhash-{self.num_perm}"

    def install_hint(self) -> str:
        return "pip install splitproof"

    def _bands(self, threshold: float) -> Tuple[int, int]:
        best = (1, self.num_perm)
        best_gap = 2.0
        for rows in range(1, self.num_perm + 1):
            if self.num_perm % rows:
                continue
            bands = self.num_perm // rows
            estimate = (1.0 / bands) ** (1.0 / rows)
            gap = abs(estimate - threshold)
            if estimate <= threshold and gap < best_gap:
                best_gap = gap
                best = (bands, rows)
        return best

    def _signatures(self, values: Sequence[Optional[str]], k: int):
        salt = self.seed.to_bytes(8, "little", signed=False)
        cache: Dict[str, int] = {}

        def cache_hash(token: str) -> int:
            value = cache.get(token)
            if value is None:
                value = stable_hash(token, salt) & 0xFFFFFFFFFFFFFFFF
                cache[token] = value
            return value

        rng = np.random.default_rng(self.seed)
        coeff_a = rng.integers(1, _MERSENNE, size=self.num_perm, dtype=np.uint64)
        coeff_b = rng.integers(0, _MERSENNE, size=self.num_perm, dtype=np.uint64)
        signatures = np.full((len(values), self.num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
        token_sets: List[Optional[frozenset]] = []
        missing: List[int] = []
        for i, value in enumerate(values):
            if value is None:
                token_sets.append(None)
                missing.append(i)
                continue
            tokens = tokenise(value, k)
            if not tokens:
                token_sets.append(None)
                missing.append(i)
                continue
            unique = frozenset(tokens)
            token_sets.append(unique)
            hashes = np.fromiter(
                (np.uint64(cache_hash(token)) for token in unique),
                dtype=np.uint64,
                count=len(unique),
            )
            permuted = (coeff_a[None, :] * hashes[:, None] + coeff_b[None, :]) % np.uint64(_MERSENNE)
            signatures[i] = permuted.min(axis=0)
        return signatures, token_sets, missing

    def pairs_above(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        threshold: float,
        *,
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> SimilarityResult:
        k = self.size_for(kind)
        bands, rows = self._bands(threshold)
        signatures, token_sets, missing = self._signatures(values, k)
        candidates: set = set()
        for band in range(bands):
            buckets: Dict[bytes, List[int]] = {}
            lo, hi = band * rows, (band + 1) * rows
            for i, tokens in enumerate(token_sets):
                if tokens is None:
                    continue
                buckets.setdefault(signatures[i, lo:hi].tobytes(), []).append(i)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                implied = len(members) * (len(members) - 1) // 2
                guard_candidate_budget(
                    len(candidates) + implied, max_pairs, self.name, threshold
                )
                for a_pos in range(len(members)):
                    for b_pos in range(a_pos + 1, len(members)):
                        candidates.add((members[a_pos], members[b_pos]))
        collector = PairCollector(self.name, threshold, max_pairs)
        for i, j in sorted(candidates):
            left, right = token_sets[i], token_sets[j]
            if left is None or right is None:
                continue
            union = len(left | right)
            if union == 0:
                continue
            sim = len(left & right) / union
            if sim >= threshold:
                collector.add(i, j, sim)
        pairs = collector.finish()
        notes = [
            "MinHash LSH finds most but not provably all pairs above the threshold, so the measured "
            "maximum is a lower bound; reported similarities themselves are exact",
        ]
        if missing:
            notes.append(f"{len(missing)} entities have no value for this key and are treated as unrelated")
        return SimilarityResult(
            backend=self.name,
            backend_version=self.version(),
            metric=self.metric,
            exactness=LOWER_BOUND,
            threshold=threshold,
            pairs=pairs,
            missing=missing,
            notes=notes,
            params={"k": k, "num_perm": self.num_perm, "bands": bands, "rows_per_band": rows},
        )

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
        order: Optional[Sequence[str]] = None,
    ) -> Dict[int, Optional[Neighbour]]:
        return ExactJaccard(k=self.k).nearest(values, kind, queries, targets, order)
