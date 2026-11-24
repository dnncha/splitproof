from __future__ import annotations

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
    guard_pair_budget,
)

DEFAULT_K = {"protein": 5, "dna": 12, "rna": 12, "text": 4, "molecule": 4}

_MERSENNE = (1 << 61) - 1


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


class ExactJaccard(SimilarityBackend):
    name = "exact_jaccard"
    metric = "kmer_jaccard"
    exactness = EXACT

    def version(self) -> str:
        import scipy

        return f"scipy-{scipy.__version__}"

    def install_hint(self) -> str:
        return "pip install cutsite (scipy is a core dependency)"

    def pairs_above(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        threshold: float,
        *,
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> SimilarityResult:
        k = kmer_size(kind)
        matrix, sizes, missing, vocab_size = _matrix(values, k)
        blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        found = 0
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
            union = sizes[rows] + sizes[cols] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(union > 0, inter / union, 0.0)
            hit = sim >= threshold
            if not hit.any():
                continue
            found += int(hit.sum())
            guard_pair_budget(found, max_pairs, self.name, threshold)
            blocks.append((rows[hit], cols[hit], sim[hit].astype(np.float32)))
        pairs = PairSet.from_blocks(blocks)
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
    ) -> Dict[int, Optional[Neighbour]]:
        k = kmer_size(kind)
        matrix, sizes, _missing, _vocab = _matrix(values, k)
        target_index = np.asarray(sorted(set(targets)), dtype=np.int64)
        out: Dict[int, Optional[Neighbour]] = {int(q): None for q in queries}
        if target_index.size == 0:
            return out
        target_matrix = matrix[target_index].T.tocsr()
        target_sizes = sizes[target_index]
        query_index = np.asarray(sorted(set(int(q) for q in queries)), dtype=np.int64)
        step = _chunk_rows(np.diff(matrix.indptr))
        for start in range(0, query_index.size, step):
            rows = query_index[start : start + step]
            block = (matrix[rows] @ target_matrix).toarray()
            union = sizes[rows][:, None] + target_sizes[None, :] - block
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(union > 0, block / union, 0.0)
            for local, global_row in enumerate(rows):
                row = sim[local]
                mask = target_index != global_row
                if not mask.any():
                    continue
                masked = np.where(mask, row, -1.0)
                best = int(np.argmax(masked))
                if masked[best] < 0:
                    continue
                out[int(global_row)] = Neighbour(index=int(target_index[best]), similarity=float(masked[best]))
        return out


class MinHashJaccard(SimilarityBackend):
    name = "minhash"
    metric = "kmer_jaccard"
    exactness = LOWER_BOUND

    def __init__(self, num_perm: int = 128, seed: int = 7):
        self.num_perm = num_perm
        self.seed = seed

    def version(self) -> str:
        return f"builtin-minhash-{self.num_perm}"

    def install_hint(self) -> str:
        return "pip install cutsite"

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
                (np.uint64(hash(token) & 0xFFFFFFFFFFFFFFFF) for token in unique),
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
        k = kmer_size(kind)
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
                for a_pos in range(len(members)):
                    for b_pos in range(a_pos + 1, len(members)):
                        candidates.add((members[a_pos], members[b_pos]))
            guard_pair_budget(len(candidates), max_pairs, self.name, threshold)
        triples: List[Tuple[int, int, float]] = []
        for i, j in sorted(candidates):
            left, right = token_sets[i], token_sets[j]
            if left is None or right is None:
                continue
            union = len(left | right)
            if union == 0:
                continue
            sim = len(left & right) / union
            if sim >= threshold:
                triples.append((i, j, float(sim)))
        pairs = PairSet.from_triples(triples)
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
    ) -> Dict[int, Optional[Neighbour]]:
        return ExactJaccard().nearest(values, kind, queries, targets)
