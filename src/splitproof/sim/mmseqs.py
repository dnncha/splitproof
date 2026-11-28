from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Mapping, List, Optional, Sequence, Tuple

from ..errors import BackendUnavailable
from . import (
    DEFAULT_MAX_PAIRS,
    BackendParameter,
    HEURISTIC,
    Neighbour,
    PairSet,
    SimilarityBackend,
    SimilarityResult,
    PairCollector,
    guard_pair_budget,
    normalise_sequence,
    register,
)

COVERAGE_DEFAULT = 0.8
SENSITIVITY_DEFAULT = 7.5
EVALUE_DEFAULT = 0.001
COV_MODE_DEFAULT = 0


def _run(command: Sequence[str], backend: str) -> subprocess.CompletedProcess[str]:

    from ..errors import SplitproofError

    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise BackendUnavailable(
            f"could not run {command[0]!r}",
            remedy="Install MMseqs2 and make sure 'mmseqs' is on PATH.",
            details={"backend": backend},
        ) from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "").strip().splitlines()
        raise SplitproofError(
            f"MMseqs2 exited with status {exc.returncode}",
            remedy=(
                "Check the sequences are valid protein or nucleotide records, and that the "
                "constraint's coverage and sensitivity parameters are sensible."
            ),
            details={
                "backend": backend,
                "returncode": exc.returncode,
                "stderr": tail[-5:] if tail else [],
            },
        ) from exc


def _collect(backend: str, threshold: float, max_pairs: int, triples: Iterable[Tuple[int, int, float]]) -> PairSet:
    collector = PairCollector(backend, threshold, max_pairs)
    for left, right, identity in triples:
        collector.add(left, right, identity)
    return collector.finish()


def parse_search_tsv(
    text: str,
    id_to_index: Dict[str, int],
    *,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    backend: str = "mmseqs2",
    threshold: float = 0.0,
) -> List[Tuple[int, int, float]]:
    pairs: Dict[Tuple[int, int], float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
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
        if key not in pairs:
            guard_pair_budget(len(pairs) + 1, max_pairs, backend, threshold)
        if identity > pairs.get(key, -1.0):
            pairs[key] = identity
    return [(a, b, value) for (a, b), value in sorted(pairs.items())]


@register("mmseqs2")
class MMseqs2Identity(SimilarityBackend):
    name = "mmseqs2"
    metric = "identity"
    exactness = HEURISTIC
    policy_params: ClassVar[Mapping[str, BackendParameter]] = {
        "coverage": (float, 0.0, 1.0),
        "sensitivity": (float, 1.0, 10.0),
        "evalue": (float, 0.0, 1e6),
        "cov_mode": (int, 0, 5),
    }

    def __init__(
        self,
        binary: str = "mmseqs",
        coverage: float = COVERAGE_DEFAULT,
        sensitivity: float = SENSITIVITY_DEFAULT,
        evalue: float = EVALUE_DEFAULT,
        cov_mode: int = COV_MODE_DEFAULT,
    ):
        self.binary = binary
        self.coverage = coverage
        self.sensitivity = sensitivity
        self.evalue = evalue
        self.cov_mode = cov_mode

    def executable(self) -> Optional[str]:
        return shutil.which(self.binary)

    def available(self) -> bool:
        return self.executable() is not None

    def install_hint(self) -> str:
        return (
            "Install MMseqs2 to measure true sequence identity: 'brew install mmseqs2', "
            "'conda install -c bioconda mmseqs2', or see github.com/soedinglab/MMseqs2. "
            "Without it, use metric 'kmer_jaccard', which splitproof computes exactly but which is "
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

    def command(
        self,
        query: Path,
        target: Path,
        out: Path,
        tmp: Path,
        threshold: float,
        kind: str = "protein",
    ) -> List[str]:
        nucleotide = kind in {"dna", "rna"}
        return [
            self.executable() or self.binary,
            "easy-search",
            str(query),
            str(target),
            str(out),
            str(tmp),
            "--min-seq-id",
            repr(float(threshold)),
            "-c",
            repr(float(self.coverage)),
            "--cov-mode",
            str(self.cov_mode),
            "-e",
            repr(float(self.evalue)),
            "-s",
            repr(float(self.sensitivity)),
            "--max-seqs",
            "10000",
            "--threads",
            "1",
            "--format-output",
            "query,target,fident",
            "-v",
            "1",
        ] + (["--search-type", "3"] if nucleotide else [])

    def _write_fasta(self, path: Path, values: Sequence[Optional[str]], indices: Sequence[int]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for i in indices:
                sequence = normalise_sequence(str(values[i]))
                handle.write(f">s{i}\n{sequence}\n")

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
            "MMseqs2 is a sensitive but heuristic search, so a clean result is evidence and not "
            "proof of absence; raise the constraint's 'sensitivity' parameter for a stricter search",
        ]
        if missing:
            notes.append(f"{len(missing)} entities have no sequence and are treated as unrelated")
        triples: List[Tuple[int, int, float]] = []
        if len(present) > 1:
            with tempfile.TemporaryDirectory(prefix="splitproof-mmseqs-") as workdir:
                root = Path(workdir)
                fasta = root / "seqs.fasta"
                out = root / "hits.tsv"
                self._write_fasta(fasta, values, present)
                _run(self.command(fasta, fasta, out, root / "tmp", threshold, kind), self.name)
                id_to_index = {f"s{i}": i for i in present}
                triples = parse_search_tsv(
                    out.read_text(encoding="utf-8"),
                    id_to_index,
                    max_pairs=max_pairs,
                    backend=self.name,
                    threshold=threshold,
                )
        return SimilarityResult(
            backend=self.name,
            backend_version=self.version(),
            metric=self.metric,
            exactness=HEURISTIC,
            threshold=threshold,
            pairs=_collect(self.name, threshold, max_pairs, triples),
            missing=missing,
            notes=notes,
            params={
                "coverage": self.coverage,
                "sensitivity": self.sensitivity,
                "cov_mode": self.cov_mode,
                "evalue": self.evalue,
            },
        )

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
        order: Optional[Sequence[str]] = None,
    ) -> Dict[int, Optional[Neighbour]]:
        out: Dict[int, Optional[Neighbour]] = {int(q): None for q in queries}
        query_list = [i for i in sorted(set(int(q) for q in queries)) if values[i]]
        target_list = [i for i in sorted(set(int(t) for t in targets)) if values[i]]
        if not query_list or not target_list:
            return out
        with tempfile.TemporaryDirectory(prefix="splitproof-mmseqs-nn-") as workdir:
            root = Path(workdir)
            query_fasta, target_fasta, hits = root / "q.fasta", root / "t.fasta", root / "hits.tsv"
            self._write_fasta(query_fasta, values, query_list)
            self._write_fasta(target_fasta, values, target_list)
            _run(self.command(query_fasta, target_fasta, hits, root / "tmp", 0.0, kind), self.name)
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
                better = current is None or identity > current.similarity
                if not better and current is not None and identity == current.similarity:
                    better = str(order[right]) < str(order[current.index]) if order else right < current.index
                if better:
                    out[left] = Neighbour(index=right, similarity=identity)
        return out
