from __future__ import annotations

import csv
import datetime as _dt
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .digest import digest
from .errors import DataError
from .policy import Policy


@dataclass
class Dataset:
    ids: List[str]
    columns: Dict[str, List[Optional[str]]]
    path: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def index(self) -> Dict[str, int]:
        return {entity_id: i for i, entity_id in enumerate(self.ids)}

    def column(self, name: str) -> List[Optional[str]]:
        if name not in self.columns:
            raise DataError(
                f"column '{name}' is not present in the data",
                remedy="Check the column names in your policy against the file header.",
                details={"column": name, "available": sorted(self.columns)},
            )
        return self.columns[name]

    def values(self, name: str) -> List[Optional[str]]:
        return self.column(name)

    def dates(self, name: str) -> List[Optional[_dt.date]]:
        out: List[Optional[_dt.date]] = []
        for raw in self.column(name):
            if raw is None or raw == "":
                out.append(None)
                continue
            text = str(raw)
            if len(text) > 10 and text[10] not in {"T", " "}:
                raise DataError(
                    f"column '{name}' holds {raw!r}, which has trailing characters after the date",
                    remedy="Use YYYY-MM-DD, optionally followed by T or a space and a time.",
                    details={"column": name, "value": text},
                )
            try:
                out.append(_dt.date.fromisoformat(text[:10]))
            except ValueError as exc:
                raise DataError(
                    f"column '{name}' holds {raw!r}, which is not an ISO date",
                    remedy="Dates must be YYYY-MM-DD (a longer ISO timestamp is fine).",
                    details={"column": name, "value": str(raw)},
                ) from exc
        return out

    def digest_for(self, policy: Policy) -> str:
        relevant: List[str] = [policy.dataset.id_column]
        relevant += [key.column for key in policy.dataset.similarity.values()]
        relevant += list(policy.dataset.groups)
        relevant += list(policy.dataset.dates.values())
        if policy.dataset.label_column:
            relevant.append(policy.dataset.label_column)
        ordered = sorted(dict.fromkeys(relevant))
        rows = []
        for i, entity_id in enumerate(self.ids):
            rows.append([entity_id] + [self.columns.get(c, [None] * len(self.ids))[i] for c in ordered])
        rows.sort(key=lambda row: row[0])
        return digest({"columns": ordered, "rows": rows})


def _normalise(value: Any) -> Optional[str]:
    """Only an empty cell is missing.

    'NA' is North America, neuraminidase, sodium and the Asn-Ala dipeptide. 'None'
    is the unperturbed arm of a Perturb-seq screen. 'Null' is a knockout genotype.
    Treating those as missing silently unconstrains real groups, so splitproof does not.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    return text or None


def _from_records(records: Sequence[Dict[str, Any]], fieldnames: Sequence[str], id_column: str, path: Optional[Path]) -> Dataset:
    if id_column not in fieldnames:
        raise DataError(
            f"id column '{id_column}' is not in the file header",
            remedy="Set dataset.id to one of the columns listed below.",
            details={"id": id_column, "available": sorted(fieldnames)},
        )
    columns: Dict[str, List[Optional[str]]] = {name: [] for name in fieldnames}
    ids: List[str] = []
    seen: Dict[str, int] = {}
    for row_number, record in enumerate(records, start=2):
        raw_id = _normalise(record.get(id_column))
        if raw_id is None:
            raise DataError(
                f"row {row_number} has an empty id in column '{id_column}'",
                remedy="Every row needs a unique, non-empty id.",
                details={"row": row_number, "id_column": id_column},
            )
        if raw_id in seen:
            raise DataError(
                f"id {raw_id!r} appears on rows {seen[raw_id]} and {row_number}",
                remedy="Ids must be unique. Deduplicate the file, or pick a different id column.",
                details={"id": raw_id, "rows": [seen[raw_id], row_number]},
            )
        seen[raw_id] = row_number
        ids.append(raw_id)
        for name in fieldnames:
            columns[name].append(_normalise(record.get(name)))
    if not ids:
        raise DataError("the dataset has no rows", remedy="Point splitproof at a file with data in it.")
    return Dataset(ids=ids, columns=columns, path=path)


def load_delimited(path: Path, id_column: str, delimiter: str) -> Dataset:
    text = Path(path).read_text(encoding="utf-8-sig")
    if "\x00" in text:
        raise DataError(
            f"{path} contains NUL bytes, so it is not valid text data",
            remedy=(
                "The file is probably binary, or truncated mid-write. Check it is the delimited "
                "text file you meant to pass."
            ),
            details={"path": str(path)},
        )
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if reader.fieldnames is None:
            raise DataError(f"{path} has no header row")
        fieldnames = [name.strip() for name in reader.fieldnames]
        records = [dict(zip(fieldnames, row.values())) for row in reader]
    except csv.Error as exc:
        raise DataError(
            f"{path} is not readable as delimited text: {exc}",
            remedy="Check the delimiter and that the file is not binary.",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    return _from_records(records, fieldnames, id_column, Path(path))


def load_parquet(path: Path, id_column: str) -> Dataset:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise DataError(
            "reading Parquet needs pyarrow",
            remedy="pip install 'splitproof[parquet]', or convert the file to CSV.",
            details={"path": str(path)},
        ) from exc
    table = pq.read_table(path)
    fieldnames = list(table.column_names)
    records = table.to_pylist()
    return _from_records(records, fieldnames, id_column, Path(path))


def load_fasta(path: Path, id_column: str, sequence_column: str = "sequence") -> Dataset:
    records: List[Dict[str, Any]] = []
    current_id: Optional[str] = None
    chunks: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if current_id is not None:
                records.append({id_column: current_id, sequence_column: "".join(chunks)})
            current_id = line[1:].strip().split()[0] if line[1:].strip() else ""
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if current_id is not None:
        records.append({id_column: current_id, sequence_column: "".join(chunks)})
    if not records:
        raise DataError(f"{path} contains no FASTA records")
    return _from_records(records, [id_column, sequence_column], id_column, Path(path))


def load(path: Path, id_column: str) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"data file not found: {path}",
            remedy="Check dataset.path in the policy, or pass --data.",
            details={"path": str(path)},
        )
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return load_delimited(path, id_column, ",")
    if suffix in {".tsv", ".tab"}:
        return load_delimited(path, id_column, "\t")
    if suffix in {".parquet", ".pq"}:
        return load_parquet(path, id_column)
    if suffix in {".fasta", ".fa", ".faa", ".fna"}:
        return load_fasta(path, id_column)
    raise DataError(
        f"unrecognised data format: {path.suffix or path.name}",
        remedy="splitproof reads .csv, .tsv, .parquet and .fasta.",
        details={"path": str(path)},
    )


def load_for(policy: Policy, override: Optional[Path] = None) -> Dataset:
    target = override or (Path(policy.dataset.path) if policy.dataset.path else None)
    if target is None:
        raise DataError(
            "no data file given",
            remedy="Set dataset.path in the policy, or pass --data.",
        )
    if override is None and policy.source_path is not None and not Path(target).is_absolute():
        candidate = policy.source_path.parent / target
        if candidate.exists():
            target = candidate
    return load(Path(target), policy.dataset.id_column)
