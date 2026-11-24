from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import PolicyError

POLICY_VERSION = 1

SIMILARITY_METRICS = {
    "identity": {"sequence"},
    "kmer_jaccard": {"sequence"},
    "tanimoto": {"molecule"},
    "scaffold": {"molecule"},
    "exact_duplicate": {"sequence", "molecule", "text"},
}

KEY_KINDS = {"protein", "dna", "rna", "molecule", "text"}

SPAN_ACTIONS = {"test", "train", "drop", "error"}


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise PolicyError(
            f"{where} is missing required field '{key}'",
            remedy=f"Add '{key}' to {where} in your policy file.",
            details={"where": where, "field": key},
        )
    return mapping[key]


def _as_date(value: Any, where: str) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise PolicyError(
            f"{where} is not an ISO date: {value!r}",
            remedy="Use YYYY-MM-DD.",
            details={"where": where, "value": str(value)},
        ) from exc


@dataclass(frozen=True)
class SimilarityKey:
    name: str
    column: str
    kind: str

    @staticmethod
    def parse(name: str, raw: Any) -> "SimilarityKey":
        where = f"dataset.similarity.{name}"
        if isinstance(raw, str):
            raw = {"column": raw}
        if not isinstance(raw, dict):
            raise PolicyError(f"{where} must be a column name or a mapping", details={"where": where})
        kind = str(raw.get("kind", "protein"))
        if kind not in KEY_KINDS:
            raise PolicyError(
                f"{where}.kind must be one of {sorted(KEY_KINDS)}, got {kind!r}",
                details={"where": where, "value": kind},
            )
        return SimilarityKey(name=name, column=str(_require(raw, "column", where)), kind=kind)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "column": self.column, "kind": self.kind}


@dataclass(frozen=True)
class SimilarityConstraint:
    key: str
    metric: str
    threshold: float
    backend: Optional[str] = None
    kind: str = "similarity"

    def to_dict(self) -> Dict[str, Any]:
        out = {"kind": "similarity", "key": self.key, "metric": self.metric, "threshold": self.threshold}
        if self.backend:
            out["backend"] = self.backend
        return out

    def describe(self) -> str:
        return f"no pair above {self.metric} {self.threshold:g} on '{self.key}' may cross folds"


@dataclass(frozen=True)
class DisjointConstraint:
    key: str
    kind: str = "disjoint"

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "disjoint", "key": self.key}

    def describe(self) -> str:
        return f"no value of '{self.key}' may appear in more than one fold"


@dataclass(frozen=True)
class TemporalConstraint:
    key: str
    cutoff: _dt.date
    before: str = "train"
    after: str = "test"
    on_span: str = "test"
    kind: str = "temporal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "temporal",
            "key": self.key,
            "cutoff": self.cutoff.isoformat(),
            "before": self.before,
            "after": self.after,
            "on_span": self.on_span,
        }

    def describe(self) -> str:
        return (
            f"'{self.key}' before {self.cutoff.isoformat()} goes to '{self.before}', "
            f"on or after goes to '{self.after}'"
        )


Constraint = Any


@dataclass(frozen=True)
class Scheme:
    kind: str
    fractions: Dict[str, float] = field(default_factory=dict)
    folds: int = 0

    @property
    def fold_names(self) -> List[str]:
        if self.kind == "kfold":
            return [f"fold{i}" for i in range(self.folds)]
        if self.kind == "leave-unit-out":
            return []
        return sorted(self.fractions)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        if self.fractions:
            out["fractions"] = {k: round(float(v), 10) for k, v in sorted(self.fractions.items())}
        if self.folds:
            out["folds"] = self.folds
        return out

    @staticmethod
    def parse(raw: Any) -> "Scheme":
        where = "scheme"
        if raw is None:
            return Scheme(kind="holdout", fractions={"train": 0.8, "test": 0.2})
        if not isinstance(raw, dict):
            raise PolicyError("scheme must be a mapping", details={"where": where})
        kind = str(raw.get("kind", "holdout"))
        if kind == "kfold":
            folds = int(raw.get("folds", 5))
            if folds < 2:
                raise PolicyError("scheme.folds must be at least 2", details={"where": where, "value": folds})
            return Scheme(kind=kind, folds=folds)
        if kind == "leave-unit-out":
            return Scheme(kind=kind)
        if kind != "holdout":
            raise PolicyError(
                f"scheme.kind must be holdout, kfold or leave-unit-out, got {kind!r}",
                details={"where": where, "value": kind},
            )
        fractions_raw = raw.get("fractions") or {"train": 0.8, "test": 0.2}
        if not isinstance(fractions_raw, dict) or len(fractions_raw) < 2:
            raise PolicyError(
                "scheme.fractions must map at least two fold names to fractions",
                details={"where": f"{where}.fractions"},
            )
        fractions = {str(k): float(v) for k, v in fractions_raw.items()}
        for name, value in fractions.items():
            if value <= 0:
                raise PolicyError(
                    f"scheme.fractions.{name} must be positive",
                    details={"where": f"{where}.fractions.{name}", "value": value},
                )
        total = sum(fractions.values())
        if abs(total - 1.0) > 1e-6:
            fractions = {k: v / total for k, v in fractions.items()}
        return Scheme(kind=kind, fractions=fractions)


@dataclass(frozen=True)
class DatasetSpec:
    path: Optional[str]
    id_column: str
    label_column: Optional[str] = None
    label_kind: str = "auto"
    similarity: Dict[str, SimilarityKey] = field(default_factory=dict)
    groups: List[str] = field(default_factory=list)
    dates: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id_column,
            "label": self.label_column,
            "label_kind": self.label_kind,
            "similarity": {k: v.to_dict() for k, v in sorted(self.similarity.items())},
            "groups": sorted(self.groups),
            "dates": dict(sorted(self.dates.items())),
        }

    @staticmethod
    def parse(raw: Any) -> "DatasetSpec":
        where = "dataset"
        if not isinstance(raw, dict):
            raise PolicyError("dataset must be a mapping", details={"where": where})
        similarity_raw = raw.get("similarity") or {}
        if not isinstance(similarity_raw, dict):
            raise PolicyError("dataset.similarity must be a mapping", details={"where": f"{where}.similarity"})
        groups_raw = raw.get("groups") or []
        if isinstance(groups_raw, str):
            groups_raw = [groups_raw]
        dates_raw = raw.get("dates") or {}
        if isinstance(dates_raw, str):
            dates_raw = {dates_raw: dates_raw}
        label_kind = str(raw.get("label_kind", "auto"))
        if label_kind not in {"auto", "categorical", "numeric"}:
            raise PolicyError(
                "dataset.label_kind must be auto, categorical or numeric",
                details={"where": f"{where}.label_kind", "value": label_kind},
            )
        return DatasetSpec(
            path=str(raw["path"]) if raw.get("path") else None,
            id_column=str(_require(raw, "id", where)),
            label_column=str(raw["label"]) if raw.get("label") else None,
            label_kind=label_kind,
            similarity={name: SimilarityKey.parse(name, value) for name, value in similarity_raw.items()},
            groups=[str(g) for g in groups_raw],
            dates={str(k): str(v) for k, v in dates_raw.items()},
        )


@dataclass(frozen=True)
class Policy:
    version: int
    dataset: DatasetSpec
    scheme: Scheme
    constraints: List[Constraint]
    seed: int = 7
    source_path: Optional[Path] = None

    @property
    def similarity_constraints(self) -> List[SimilarityConstraint]:
        return [c for c in self.constraints if isinstance(c, SimilarityConstraint)]

    @property
    def disjoint_constraints(self) -> List[DisjointConstraint]:
        return [c for c in self.constraints if isinstance(c, DisjointConstraint)]

    @property
    def temporal_constraints(self) -> List[TemporalConstraint]:
        return [c for c in self.constraints if isinstance(c, TemporalConstraint)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cutsite": self.version,
            "dataset": self.dataset.to_dict(),
            "scheme": self.scheme.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any], *, source_path: Optional[Path] = None) -> "Policy":
        if not isinstance(raw, dict):
            raise PolicyError("policy must be a mapping at the top level")
        version = int(raw.get("cutsite", POLICY_VERSION))
        if version != POLICY_VERSION:
            raise PolicyError(
                f"policy version {version} is not supported by this cutsite build",
                remedy=f"Set 'cutsite: {POLICY_VERSION}' or upgrade cutsite.",
                details={"value": version, "supported": POLICY_VERSION},
            )
        dataset = DatasetSpec.parse(_require(raw, "dataset", "policy"))
        scheme = Scheme.parse(raw.get("scheme"))
        constraints = _parse_constraints(raw.get("constraints") or [], dataset)
        if not constraints:
            raise PolicyError(
                "policy declares no constraints, so it would prove nothing",
                remedy="Add at least one similarity, disjoint or temporal constraint.",
            )
        seed = int(raw.get("seed", 7))
        return Policy(
            version=version,
            dataset=dataset,
            scheme=scheme,
            constraints=constraints,
            seed=seed,
            source_path=source_path,
        )

    @staticmethod
    def load(path: Path) -> "Policy":
        text = Path(path).read_text(encoding="utf-8")
        return Policy.from_dict(_parse_structured(text, path), source_path=Path(path))


def _parse_structured(text: str, path: Path) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        import json

        return json.loads(text)
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise PolicyError(
            "reading a YAML policy needs PyYAML",
            remedy="Install cutsite normally (pip install cutsite) or use a .json policy.",
        ) from exc
    loaded = yaml.safe_load(text)
    if loaded is None:
        raise PolicyError(f"policy file {path} is empty")
    return loaded


def _parse_constraints(raw: Any, dataset: DatasetSpec) -> List[Constraint]:
    if not isinstance(raw, list):
        raise PolicyError("constraints must be a list", details={"where": "constraints"})
    out: List[Constraint] = []
    for index, item in enumerate(raw):
        where = f"constraints[{index}]"
        if not isinstance(item, dict):
            raise PolicyError(f"{where} must be a mapping", details={"where": where})
        kind = str(_require(item, "kind", where))
        if kind == "similarity":
            out.append(_parse_similarity(item, where, dataset))
        elif kind == "disjoint":
            key = str(_require(item, "key", where))
            if key not in dataset.groups:
                raise PolicyError(
                    f"{where} references group '{key}' which is not listed in dataset.groups",
                    remedy=f"Add '{key}' to dataset.groups.",
                    details={"where": where, "key": key, "known": sorted(dataset.groups)},
                )
            out.append(DisjointConstraint(key=key))
        elif kind == "temporal":
            out.append(_parse_temporal(item, where, dataset))
        else:
            raise PolicyError(
                f"{where}.kind must be similarity, disjoint or temporal, got {kind!r}",
                details={"where": where, "value": kind},
            )
    return out


def _parse_similarity(item: Dict[str, Any], where: str, dataset: DatasetSpec) -> SimilarityConstraint:
    key = str(_require(item, "key", where))
    if key not in dataset.similarity:
        raise PolicyError(
            f"{where} references similarity key '{key}' which is not declared in dataset.similarity",
            remedy=f"Declare '{key}' under dataset.similarity.",
            details={"where": where, "key": key, "known": sorted(dataset.similarity)},
        )
    metric = str(_require(item, "metric", where))
    if metric not in SIMILARITY_METRICS:
        raise PolicyError(
            f"{where}.metric must be one of {sorted(SIMILARITY_METRICS)}, got {metric!r}",
            details={"where": where, "value": metric},
        )
    kind = dataset.similarity[key].kind
    family = "molecule" if kind == "molecule" else ("text" if kind == "text" else "sequence")
    if family not in SIMILARITY_METRICS[metric]:
        raise PolicyError(
            f"{where}.metric '{metric}' does not apply to a '{kind}' key",
            remedy=f"Metric '{metric}' applies to {sorted(SIMILARITY_METRICS[metric])} keys.",
            details={"where": where, "metric": metric, "key_kind": kind},
        )
    threshold = float(_require(item, "threshold", where))
    if not 0.0 < threshold <= 1.0:
        raise PolicyError(
            f"{where}.threshold must be in (0, 1], got {threshold}",
            details={"where": where, "value": threshold},
        )
    backend = str(item["backend"]) if item.get("backend") else None
    return SimilarityConstraint(key=key, metric=metric, threshold=threshold, backend=backend)


def _parse_temporal(item: Dict[str, Any], where: str, dataset: DatasetSpec) -> TemporalConstraint:
    key = str(_require(item, "key", where))
    if key not in dataset.dates:
        raise PolicyError(
            f"{where} references date '{key}' which is not listed in dataset.dates",
            remedy=f"Add '{key}' to dataset.dates.",
            details={"where": where, "key": key, "known": sorted(dataset.dates)},
        )
    cutoff = _as_date(_require(item, "cutoff", where), f"{where}.cutoff")
    before = str(item.get("before", "train"))
    after = str(item.get("after", "test"))
    if before == after:
        raise PolicyError(
            f"{where}.before and {where}.after must name different folds",
            details={"where": where, "value": before},
        )
    on_span = str(item.get("on_span", "test"))
    if on_span not in SPAN_ACTIONS:
        raise PolicyError(
            f"{where}.on_span must be one of {sorted(SPAN_ACTIONS)}, got {on_span!r}",
            details={"where": where, "value": on_span},
        )
    return TemporalConstraint(key=key, cutoff=cutoff, before=before, after=after, on_span=on_span)
