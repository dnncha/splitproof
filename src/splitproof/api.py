from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import assign as assign_module
from . import cert as cert_module
from . import dataset as dataset_module
from . import evaluate as evaluate_module
from . import units as units_module
from .assign import Assignment, DROPPED, FoldStat
from .dataset import Dataset
from .errors import DataError
from .policy import Policy
from .sim import DEFAULT_MAX_PAIRS
from .units import UnitGraph

DEFAULT_TOLERANCE = 0.1
DEFAULT_EXACT_MAX_N = 20000


@dataclass
class Outcome:
    policy: Policy
    data: Dataset
    graph: UnitGraph
    assignment: Assignment
    reports: List[evaluate_module.ConstraintReport]
    certificate: Dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.certificate.get("status") != evaluate_module.VIOLATED


def make_split(
    policy: Policy,
    *,
    data_path: Optional[Path] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    exact_max_n: int = DEFAULT_EXACT_MAX_N,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    skip_neighbours: bool = False,
    backend_overrides: Optional[Dict[str, str]] = None,
) -> Outcome:
    data = dataset_module.load_for(policy, data_path)
    graph = units_module.build(
        policy,
        data,
        exact_max_n=exact_max_n,
        max_pairs=max_pairs,
        backend_overrides=backend_overrides,
    )
    assignment = assign_module.assign(policy, graph, tolerance=tolerance)
    assignment.stats = assign_module.describe(policy, data, assignment, graph)
    reports = evaluate_module.check(policy, data, assignment, graph)
    diagnostics = evaluate_module.diagnostics(
        policy,
        data,
        assignment,
        exact_max_n=exact_max_n,
        skip_neighbours=skip_neighbours,
        backend_overrides=backend_overrides,
    )
    certificate = cert_module.build(policy, data, graph, assignment, reports, diagnostics)
    return Outcome(
        policy=policy,
        data=data,
        graph=graph,
        assignment=assignment,
        reports=reports,
        certificate=certificate,
    )


def audit(
    policy: Policy,
    *,
    data_path: Optional[Path] = None,
    split_column: Optional[str] = None,
    split_path: Optional[Path] = None,
    exact_max_n: int = DEFAULT_EXACT_MAX_N,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    skip_neighbours: bool = False,
    backend_overrides: Optional[Dict[str, str]] = None,
) -> Outcome:
    data = dataset_module.load_for(policy, data_path)
    assignment = load_assignment(data, split_column=split_column, split_path=split_path)
    graph = units_module.build(
        policy,
        data,
        exact_max_n=exact_max_n,
        max_pairs=max_pairs,
        backend_overrides=backend_overrides,
    )
    assignment.stats = assign_module.describe(policy, data, assignment, graph)
    reports = evaluate_module.check(policy, data, assignment, graph)
    diagnostics = evaluate_module.diagnostics(
        policy,
        data,
        assignment,
        exact_max_n=exact_max_n,
        skip_neighbours=skip_neighbours,
        backend_overrides=backend_overrides,
    )
    certificate = cert_module.build(policy, data, graph, assignment, reports, diagnostics)
    certificate["audited"] = True
    return Outcome(
        policy=policy,
        data=data,
        graph=graph,
        assignment=assignment,
        reports=reports,
        certificate=certificate,
    )


def verify(
    certificate_path: Path,
    *,
    data_path: Optional[Path] = None,
    split_path: Optional[Path] = None,
    split_column: Optional[str] = None,
    exact_max_n: int = DEFAULT_EXACT_MAX_N,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    skip_neighbours: bool = False,
    backend_overrides: Optional[Dict[str, str]] = None,
) -> cert_module.VerifyResult:
    certificate = cert_module.read(Path(certificate_path))
    policy = cert_module.policy_from(certificate)
    target = data_path
    if target is None and certificate.get("source", {}).get("path"):
        target = Path(certificate["source"]["path"])
    data = dataset_module.load_for(policy, target)
    assignment = load_assignment(data, split_column=split_column, split_path=split_path)
    graph = units_module.build(
        policy,
        data,
        exact_max_n=exact_max_n,
        max_pairs=max_pairs,
        backend_overrides=backend_overrides,
    )
    assignment.stats = assign_module.describe(policy, data, assignment, graph)
    reports = evaluate_module.check(policy, data, assignment, graph)
    diagnostics = evaluate_module.diagnostics(
        policy,
        data,
        assignment,
        exact_max_n=exact_max_n,
        skip_neighbours=skip_neighbours,
        backend_overrides=backend_overrides,
    )
    recomputed = cert_module.build(policy, data, graph, assignment, reports, diagnostics)
    return cert_module.verify(certificate, recomputed, reports)


def load_assignment(
    data: Dataset,
    *,
    split_column: Optional[str] = None,
    split_path: Optional[Path] = None,
) -> Assignment:
    if split_column:
        return assign_module.from_column(data, split_column)
    if split_path:
        return _from_split_file(data, Path(split_path))
    raise DataError(
        "no existing split given",
        remedy="Pass --split-column for a column in the data file, or --split for a two-column file.",
    )


def _from_split_file(data: Dataset, path: Path) -> Assignment:
    if not path.exists():
        raise DataError(f"split file not found: {path}", details={"path": str(path)})
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        raise DataError(f"split file {path} is empty")
    header = len(rows[0]) > 1 and rows[0][1].strip().lower() in {"fold", "split", "set"}
    duplicates: List[str] = []
    for row in rows[1 if header else 0 :]:
        if len(row) < 2:
            raise DataError(
                f"split file {path} needs two columns: id,fold",
                details={"path": str(path), "row": row},
            )
        entity, fold = row[0].strip(), row[1].strip()
        if entity in mapping and mapping[entity] != fold:
            duplicates.append(entity)
        mapping[entity] = fold
    if duplicates:
        raise DataError(
            f"split file {path} assigns {len(duplicates)} ids to more than one fold",
            remedy=(
                "An id in two folds is not a split. The later row silently won, and the audit "
                "graded only that one."
            ),
            details={"examples": sorted(set(duplicates))[:10]},
        )
    known = set(data.ids)
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise DataError(
            f"the split file names {len(unknown)} ids that are not in the data",
            remedy="Check that the split file's first column holds the same ids as the data file.",
            details={"examples": unknown[:10]},
        )
    uncovered = sorted(known - set(mapping))
    if uncovered:
        raise DataError(
            f"the split file covers {len(mapping)} of {len(known)} entities, leaving "
            f"{len(uncovered)} unassigned",
            remedy=(
                "Every entity must be in a fold, or the audit would silently grade only the part "
                "the file mentions. Add the missing ids, or drop them from the data file."
            ),
            details={"covered": len(mapping), "entities": len(known), "examples": uncovered[:10]},
        )
    fold_of: Dict[int, str] = {}
    counts: Dict[str, int] = {}
    for i, entity_id in enumerate(data.ids):
        fold = mapping.get(entity_id, DROPPED)
        fold_of[i] = fold
        counts[fold] = counts.get(fold, 0) + 1
    assign_module.check_audit_folds(data, fold_of, f"split file {path.name}")
    folds = sorted((name for name in counts if name != DROPPED), key=assign_module.fold_order)
    stats = [FoldStat(name=name, target=counts[name], count=counts[name], units=0) for name in folds]
    return Assignment(fold_of=fold_of, folds=folds, stats=stats, dropped=[])


def write_split(path: Path, data: Dataset, assignment: Assignment, id_column: str) -> None:
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow([id_column, "fold"])
        for entity_id, fold in assignment.to_rows(data):
            writer.writerow([entity_id, fold])
