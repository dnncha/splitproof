from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..errors import BackendFailed
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
    "Molecule similarity needs RDKit: pip install 'splitproof[chem]', or "
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


def ring_systems(chemistry, molecule) -> Optional[str]:
    """A canonical key for the molecule's ring skeleton, or None if it has no rings.

    Salts and solvates are reduced to the largest covalent fragment. Fused rings are
    merged into one system, and each system is written as a canonical SMILES fragment
    of the parent molecule using element and aromaticity only, so the key ignores
    stereochemistry, formal charge, isotope and hydrogen count. Nothing is edited, so
    there is no re-sanitising and no kekulisation to break.

    That makes the key deliberately coarse. Indole and N-methylindole share it, so do
    caffeine and theophylline, adenine and 9-methyladenine, pyridine and pyridinium,
    a pair of enantiomers, and a drug and its tosylate salt. Grouping those together
    is the safe direction for a split: it never lets a near-duplicate cross folds.
    Benzene and pyridine stay apart, because their elements differ.

    It is not the Bemis-Murcko scaffold, which keeps inter-ring linkers and
    distinguishes all of the above, and it is not computed with RDKit's
    MurckoScaffoldSmiles, whose MurckoDecompose raises on every input in RDKit
    2026.09.
    """
    fragments = chemistry.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    if len(fragments) > 1:
        molecule = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())
    chemistry.FastFindRings(molecule)
    chemistry.RemoveStereochemistry(molecule)
    info = molecule.GetRingInfo()
    if info.NumRings() == 0:
        return None
    systems: List[set] = []
    for ring in info.AtomRings():
        current = set(ring)
        overlapping = [existing for existing in systems if existing & current]
        systems = [existing for existing in systems if not (existing & current)]
        for existing in overlapping:
            current |= existing
        systems.append(current)
    symbols = [
        f"[{atom.GetSymbol().lower() if atom.GetIsAromatic() else atom.GetSymbol()}]"
        for atom in molecule.GetAtoms()
    ]
    return "|".join(
        sorted(
            chemistry.MolFragmentToSmiles(
                molecule,
                atomsToUse=sorted(system),
                atomSymbols=symbols,
                canonical=True,
                isomericSmiles=False,
            )
            for system in systems
        )
    )


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
    policy_params = {"radius": (int, 0, 8), "bits": (int, 32, 65536)}

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
        order: Optional[Sequence[str]] = None,
    ) -> Dict[int, Optional[Neighbour]]:
        from rdkit import DataStructs

        prints, _missing, _unparsable = self._fingerprints(values)
        key = (lambda index: str(order[index])) if order is not None else (lambda index: index)
        target_list = [
            i for i in sorted(set(int(t) for t in targets), key=key) if prints[i] is not None
        ]
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
            best = min(range(len(pool)), key=lambda position: (-sims[position], key(pool[position])))
            out[query] = Neighbour(index=pool[best], similarity=float(sims[best]))
        return out


@register("ring_systems")
class RingSystems(_ChemBackend):
    name = "ring_systems"
    metric = "ring_systems"
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
            try:
                framework = ring_systems(Chem, molecule)
            except Exception as exc:
                raise BackendFailed(
                    f"could not extract ring systems from {value!r}",
                    remedy=(
                        "This is a bug in splitproof's framework reduction, not in your data. "
                        "Please report the SMILES."
                    ),
                    details={"backend": self.name, "smiles": value, "reason": str(exc)},
                ) from exc
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
            "molecules sharing the same set of ring systems form an equivalence class, so no "
            "pairwise edge list is ever built; linkers between ring systems are not part of the "
            "key, so this groups more widely than a Bemis-Murcko scaffold"
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
            params={"distinct_ring_system_sets": len(buckets)},
        )

    def nearest(
        self,
        values: Sequence[Optional[str]],
        kind: str,
        queries: Sequence[int],
        targets: Sequence[int],
        order: Optional[Sequence[str]] = None,
    ) -> Dict[int, Optional[Neighbour]]:
        scaffolds, _missing, _unparsable = self._scaffolds(values)
        key = (lambda index: str(order[index])) if order is not None else (lambda index: index)
        lookup: Dict[str, int] = {}
        for i in sorted(set(int(t) for t in targets), key=key):
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
