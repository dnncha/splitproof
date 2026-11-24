from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import (
    DEFAULT_MAX_PAIRS,
    HEURISTIC,
    Neighbour,
    PairSet,
    SimilarityBackend,
    SimilarityResult,
    guard_pair_budget,
)

COVERAGE_DEFAULT = 0.8
SENSITIVITY_DEFAULT = 7.5


def parse_search_tsv(text: str, id_to_index: Dict[str, int]) -> List[Tuple[int, int, float]]:
    pairs: Dict[Tuple[int, int], float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        query, target, identity_text = fields[0], fields[1], fields[2]
        if query == target:
            continue
        left, right = id_to_index.get(query), id_to_index.get(target)
        if left is None or right is None:
            continue
        try:
            identity = float(identity_text)
        except ValueError:
            continue
        if identity > 1.0:
            identity = identity / 100.0
        key = (left, right) if left < right else (right, left)
        if identity > pairs.get(key, -1.0):
            pairs[key] = identity
    return [(a, b, value) for (a, b), value in sorted(pairs.items())]


class MMseqs2Identity(SimilarityBackend):
    name = "mmseqs2"
    metric = "identity"
    exactness = HEURISTIC

    def __init__(self, binary: str = "mmseqs", coverage: float = COVERAGE_DEFAULT, sensitivity: float = SENSITIVITY_DEFAULT):
        self.binary = binary
        self.coverage = coverage
        self.sensitivity = sensitivity

    def executable(self) -> Optional[str]:
        return shutil.which(self.binary)

    def available(self) -> bool:
        return self.executable() is not None

    def install_hint(self) -> str:
        return (
            "Install MMseqs2 to measure true sequence identity: 'brew install mmseqs2', "
            "'conda install -c bioconda mmseqs2', or see github.com/soedinglab/MMseqs2. "
            "Without it, use metric 'kmer_jaccard', which cutsite computes exactly but which is "
            "not sequence identity."
        )

    def version(self) -> str:
        executable = self.executable()
        if executable is None:
            return "unavailable"
        try:
            result = subprocess.run([executable, "version"], capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return (result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr) else "unknown"

    def command(self, query: Path, target: Path, out: Path, tmp: Path, threshold: float) -> List[str]:
        return [
            self.executable() or self.binary,
            "easy-search",
            str(query),
            str(target),
            str(out),
            str(tmp),
            "--min-seq-id",
            f"{threshold:g}",
            "-c",
            f"{self.coverage:g}",
            "--cov-mode",
            "0",
            "-s",
            f"{self.sensitivity:g}",
            "--max-seqs",
            "10000",
            "--threads",
            "1",
            "--format-output",
            "query,target,fident",
            "-v",
            "1",
        ]

    def _write_fasta(self, path: Path, values: Sequence[Optional[str]], indices: Sequence[int]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for i in indices:
                handle.write(f">s{i}\n{values[i]}\n")

    def pairs_above(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        threshold: float,
        *,
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> SimilarityResult:
        present = [i for i, value in enumerate(values) if value]
        missing = [i for i, value in enumerate(values) if not value]
        notes = [
            "MMseqs2 is a sensitive but heuristic search, so the measured maximum identity is not a "
            "proof of absence; raise --mmseqs-sensitivity for a stricter search",
        ]
        if missing:
            notes.append(f"{len(missing)} entities have no sequence and are treated as unrelated")
        triples: List[Tuple[int, int, float]] = []
        if len(present) > 1:
            with tempfile.TemporaryDirectory(prefix="cutsite-mmseqs-") as workdir:
                root = Path(workdir)
                fasta = root / "seqs.fasta"
                out = root / "hits.tsv"
                self._write_fasta(fasta, values, present)
                subprocess.run(
                    self.command(fasta, fasta, out, root / "tmp", threshold),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                id_to_index = {f"s{i}": i for i in present}
                triples = parse_search_tsv(out.read_text(encoding="utf-8"), id_to_index)
                guard_pair_budget(len(triples), max_pairs, self.name, threshold)
        return SimilarityResult(
            backend=self.name,
            backend_version=self.version(),
            metric=self.metric,
            exactness=HEURISTIC,
            threshold=threshold,
            pairs=PairSet.from_triples(triples),
            missing=missing,
            notes=notes,
            params={"coverage": self.coverage, "sensitivity": self.sensitivity, "cov_mode": 0},
        )

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
    ) -> Dict[int, Optional[Neighbour]]:
        out: Dict[int, Optional[Neighbour]] = {int(q): None for q in queries}
        query_list = [i for i in sorted(set(int(q) for q in queries)) if values[i]]
        target_list = [i for i in sorted(set(int(t) for t in targets)) if values[i]]
        if not query_list or not target_list:
            return out
        with tempfile.TemporaryDirectory(prefix="cutsite-mmseqs-nn-") as workdir:
            root = Path(workdir)
            query_fasta, target_fasta, hits = root / "q.fasta", root / "t.fasta", root / "hits.tsv"
            self._write_fasta(query_fasta, values, query_list)
            self._write_fasta(target_fasta, values, target_list)
            subprocess.run(
                self.command(query_fasta, target_fasta, hits, root / "tmp", 0.0),
                capture_output=True,
                text=True,
                check=True,
            )
            id_to_index = {f"s{i}": i for i in set(query_list) | set(target_list)}
            target_set = set(target_list)
            for line in hits.read_text(encoding="utf-8").splitlines():
                fields = line.strip().split("\t")
                if len(fields) < 3:
                    continue
                left, right = id_to_index.get(fields[0]), id_to_index.get(fields[1])
                if left is None or right is None or left == right or right not in target_set:
                    continue
                try:
                    identity = float(fields[2])
                except ValueError:
                    continue
                if identity > 1.0:
                    identity /= 100.0
                current = out.get(left)
                if current is None or identity > current.similarity:
                    out[left] = Neighbour(index=right, similarity=identity)
        return out
