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
    except ModuleNotFoundError:
        return None
    RDLogger.DisableLog("rdApp.*")
    return Chem, rdFingerprintGenerator


def murcko_framework(chemistry, molecule) -> Optional[str]:
    """The ring systems and the linkers between them, as canonical SMILES.

    Computed by repeatedly deleting terminal atoms that are not in a ring, which
    leaves exactly the ring systems plus the paths joining them. Exocyclic
    substituents go, including exocyclic double-bonded atoms, so a lactam and its
    parent ring share a framework. That groups more molecules together than
    RDKit's MurckoScaffoldSmiles does, which is the safe direction for a split.

    Returns None for a molecule with no rings, which has no framework. It does
    not catch toolkit failures: a split must not quietly lose a constraint
    because a scaffold could not be computed.
    """
    editable = chemistry.RWMol(molecule)
    chemistry.FastFindRings(editable)
    while True:
        victim = None
        for atom in editable.GetAtoms():
            if atom.GetDegree() <= 1 and not atom.IsInRing():
                victim = atom.GetIdx()
                break
        if victim is None:
            break
        editable.RemoveAtom(victim)
        chemistry.FastFindRings(editable)
    if editable.GetNumAtoms() == 0:
        return None
    for atom in editable.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    framework = editable.GetMol()
    chemistry.SanitizeMol(framework)
    return chemistry.MolToSmiles(framework)


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


@register("tanimoto")
class MorganTanimoto(_ChemBackend):
    name = "tanimoto"
    metric = "tanimoto"
    exactness = EXACT

    def __init__(self, radius: int = RADIUS_DEFAULT, bits: int = BITS_DEFAULT):
        self.radius = radius
        self.bits = bits

    def _fingerprints(self, values: Sequence[Optional[str]]):
        Chem, generators = _rdkit()
        generator = generators.GetMorganGenerator(radius=self.radius, fpSize=self.bits)
        prints: List[Optional[object]] = []
        missing: List[int] = []
        unparsable: List[int] = []
        for i, value in enumerate(values):
            molecule = Chem.MolFromSmiles(value) if value else None
            if molecule is None:
                prints.append(None)
                missing.append(i)
                if value:
                    unparsable.append(i)
                continue
            prints.append(generator.GetFingerprint(molecule))
        return prints, missing, unparsable

    def pairs_above(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        threshold: float,
        *,
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> SimilarityResult:
        from rdkit import DataStructs

        prints, missing, unparsable = self._fingerprints(values)
        collector = PairCollector(self.name, threshold, max_pairs)
        present = [i for i, fingerprint in enumerate(prints) if fingerprint is not None]
        for position, i in enumerate(present):
            tail = present[position + 1 :]
            if not tail:
                break
            sims = DataStructs.BulkTanimotoSimilarity(prints[i], [prints[j] for j in tail])
            for j, sim in zip(tail, sims):
                if sim >= threshold:
                    collector.add(i, j, sim)
        notes = ["Morgan/ECFP fingerprints compared by Tanimoto; computed over all pairs, so exact"]
        if unparsable:
            notes.append(f"{len(unparsable)} entities hold SMILES that RDKit could not parse")
        absent = len(missing) - len(unparsable)
        if absent:
            notes.append(f"{absent} entities have no SMILES at all and are treated as unrelated")
        return SimilarityResult(
            backend=self.name,
            backend_version=self.version(),
            metric=self.metric,
            exactness=EXACT,
            threshold=threshold,
            pairs=collector.finish(),
            missing=missing,
            unparsable=unparsable,
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

        prints, _missing, _unparsable = self._fingerprints(values)
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


@register("scaffold")
class MurckoScaffold(_ChemBackend):
    name = "scaffold"
    metric = "scaffold"
    exactness = EXACT
    emits = "classes"

    def _scaffolds(
        self, values: Sequence[Optional[str]]
    ) -> Tuple[List[Optional[str]], List[int], List[int]]:
        Chem, _generators = _rdkit()
        out: List[Optional[str]] = []
        missing: List[int] = []
        unparsable: List[int] = []
        for i, value in enumerate(values):
            molecule = Chem.MolFromSmiles(value) if value else None
            if molecule is None:
                out.append(None)
                missing.append(i)
                if value:
                    unparsable.append(i)
                continue
            framework = murcko_framework(Chem, molecule)
            out.append(framework)
            if framework is None:
                missing.append(i)
        return out, missing, unparsable

    def pairs_above(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        threshold: float,
        *,
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> SimilarityResult:
        scaffolds, missing, unparsable = self._scaffolds(values)
        buckets: Dict[str, List[int]] = {}
        for i, scaffold in enumerate(scaffolds):
            if scaffold:
                buckets.setdefault(scaffold, []).append(i)
        groups: List[Optional[str]] = [None] * len(values)
        for label, members in buckets.items():
            for member in members:
                groups[member] = label
        notes = [
            "molecules sharing a Murcko framework (ring systems plus their linkers, with all "
            "exocyclic substituents removed) form an equivalence class, so no pairwise edge "
            "list is ever built"
        ]
        if unparsable:
            notes.append(f"{len(unparsable)} entities hold SMILES that RDKit could not parse")
        acyclic = len(missing) - len(unparsable)
        if acyclic:
            notes.append(
                f"{acyclic} entities have no ring system, so no framework, and are treated "
                "as unrelated"
            )
        return SimilarityResult(
            backend=self.name,
            backend_version=self.version(),
            metric=self.metric,
            exactness=EXACT,
            threshold=threshold,
            groups=groups,
            missing=missing,
            unparsable=unparsable,
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
        scaffolds, _missing, _unparsable = self._scaffolds(values)
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
