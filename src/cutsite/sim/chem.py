from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import EXACT, Neighbour, SimilarityBackend, SimilarityResult

RADIUS_DEFAULT = 2
BITS_DEFAULT = 2048

_HINT = (
    "Molecule similarity needs RDKit: pip install 'cutsite[chem]', or "
    "conda install -c conda-forge rdkit."
)


def _rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold as _Murcko
    except ModuleNotFoundError:
        return None
    RDLogger.DisableLog("rdApp.*")
    return Chem, rdFingerprintGenerator, _Murcko


class _ChemBackend(SimilarityBackend):
    def available(self) -> bool:
        return _rdkit() is not None

    def install_hint(self) -> str:
        return _HINT

    def version(self) -> str:
        try:
            import rdkit

            return f"rdkit-{rdkit.__version__}"
        except ModuleNotFoundError:
            return "unavailable"


class MorganTanimoto(_ChemBackend):
    name = "tanimoto"
    metric = "tanimoto"
    exactness = EXACT

    def __init__(self, radius: int = RADIUS_DEFAULT, bits: int = BITS_DEFAULT):
        self.radius = radius
        self.bits = bits

    def _fingerprints(self, values: Sequence[Optional[str]]):
        Chem, generators, _ = _rdkit()
        generator = generators.GetMorganGenerator(radius=self.radius, fpSize=self.bits)
        prints: List[Optional[object]] = []
        missing: List[int] = []
        for i, value in enumerate(values):
            molecule = Chem.MolFromSmiles(value) if value else None
            if molecule is None:
                prints.append(None)
                missing.append(i)
                continue
            prints.append(generator.GetFingerprint(molecule))
        return prints, missing

    def pairs_above(self, values: Sequence[Optional[str]], kind: str, threshold: float) -> SimilarityResult:
        from rdkit import DataStructs

        prints, missing = self._fingerprints(values)
        pairs: List[Tuple[int, int, float]] = []
        for i in range(len(prints)):
            if prints[i] is None:
                continue
            others = [p for p in prints[i + 1 :]]
            if not others:
                continue
            valid = [(i + 1 + offset, p) for offset, p in enumerate(others) if p is not None]
            if not valid:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(prints[i], [p for _, p in valid])
            for (j, _), sim in zip(valid, sims):
                if sim >= threshold:
                    pairs.append((i, j, float(sim)))
        notes = ["Morgan/ECFP fingerprints compared by Tanimoto; computed over all pairs, so exact"]
        if missing:
            notes.append(f"{len(missing)} entities have no parsable SMILES and are treated as unrelated")
        return SimilarityResult(
            backend=self.name,
            backend_version=self.version(),
            metric=self.metric,
            exactness=EXACT,
            threshold=threshold,
            pairs=pairs,
            missing=missing,
            notes=notes,
            params={"radius": self.radius, "bits": self.bits},
        )

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
    ) -> Dict[int, Optional[Neighbour]]:
        from rdkit import DataStructs

        prints, _missing = self._fingerprints(values)
        target_list = [i for i in sorted(set(int(t) for t in targets)) if prints[i] is not None]
        out: Dict[int, Optional[Neighbour]] = {}
        for raw in queries:
            query = int(raw)
            out[query] = None
            if prints[query] is None:
                continue
            pool = [i for i in target_list if i != query]
            if not pool:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(prints[query], [prints[i] for i in pool])
            best = max(range(len(pool)), key=lambda position: sims[position])
            out[query] = Neighbour(index=pool[best], similarity=float(sims[best]))
        return out


class MurckoScaffold(_ChemBackend):
    name = "scaffold"
    metric = "scaffold"
    exactness = EXACT

    def _scaffolds(self, values: Sequence[Optional[str]]) -> Tuple[List[Optional[str]], List[int]]:
        Chem, _generators, murcko = _rdkit()
        out: List[Optional[str]] = []
        missing: List[int] = []
        for i, value in enumerate(values):
            molecule = Chem.MolFromSmiles(value) if value else None
            if molecule is None:
                out.append(None)
                missing.append(i)
                continue
            try:
                scaffold = murcko.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
            except Exception:
                out.append(None)
                missing.append(i)
                continue
            out.append(scaffold or None)
            if not scaffold:
                missing.append(i)
        return out, missing

    def pairs_above(self, values: Sequence[Optional[str]], kind: str, threshold: float) -> SimilarityResult:
        scaffolds, missing = self._scaffolds(values)
        buckets: Dict[str, List[int]] = {}
        for i, scaffold in enumerate(scaffolds):
            if scaffold:
                buckets.setdefault(scaffold, []).append(i)
        pairs: List[Tuple[int, int, float]] = []
        for members in buckets.values():
            for a_pos in range(len(members)):
                for b_pos in range(a_pos + 1, len(members)):
                    pairs.append((members[a_pos], members[b_pos], 1.0))
        pairs.sort()
        notes = ["molecules sharing a Bemis-Murcko scaffold are treated as one unit"]
        if missing:
            notes.append(f"{len(missing)} entities have no scaffold and are treated as unrelated")
        return SimilarityResult(
            backend=self.name,
            backend_version=self.version(),
            metric=self.metric,
            exactness=EXACT,
            threshold=threshold,
            pairs=pairs,
            missing=missing,
            notes=notes,
            params={"distinct_scaffolds": len(buckets)},
        )

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
    ) -> Dict[int, Optional[Neighbour]]:
        scaffolds, _missing = self._scaffolds(values)
        lookup: Dict[str, int] = {}
        for i in sorted(set(int(t) for t in targets)):
            scaffold = scaffolds[i]
            if scaffold:
                lookup.setdefault(scaffold, i)
        out: Dict[int, Optional[Neighbour]] = {}
        for raw in queries:
            query = int(raw)
            scaffold = scaffolds[query]
            match = lookup.get(scaffold) if scaffold else None
            out[query] = Neighbour(index=match, similarity=1.0) if match is not None and match != query else None
        return out
