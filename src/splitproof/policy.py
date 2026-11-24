from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import PolicyError

POLICY_VERSION = 1

SIMILARITY_METRICS = {
    "identity": {"sequence"},
    "kmer_jaccard": {"sequence"},
    "tanimoto": {"molecule"},
    "ring_systems": {"molecule"},
    "canonical_smiles": {"molecule"},
    "exact_duplicate": {"sequence", "molecule", "text"},
}

KEY_KINDS = {"protein", "dna", "rna", "molecule", "text"}

EQUIVALENCE_METRICS = {"exact_duplicate", "ring_systems", "canonical_smiles"}

SPAN_ACTIONS = {"test", "drop", "error"}


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise PolicyError(
            f"{where} is missing required field '{key}'",
            remedy=f"Add '{key}' to {where} in your policy file.",
            details={"where": where, "field": key},
        )
    return mapping[key]


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PolicyError(
            f"{where} must be a whole number, got {type(value).__name__}",
            details={"where": where, "value": str(value)},
        )
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        raise PolicyError(
            f"{where} must be a finite whole number, got {value!r}",
            details={"where": where, "value": str(value)},
        )
    if isinstance(value, float) and value != int(value):
        raise PolicyError(
            f"{where} must be a whole number, got {value!r}",
            details={"where": where, "value": str(value)},
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PolicyError(
            f"{where} must be a whole number, got {value!r}",
            details={"where": where, "value": str(value)},
        ) from exc


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PolicyError(
            f"{where} must be a number, got {type(value).__name__}",
            details={"where": where, "value": str(value)},
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PolicyError(
            f"{where} must be a number, got {value!r}",
            details={"where": where, "value": str(value)},
        ) from exc
    if number != number or number in {float("inf"), float("-inf")}:
        raise PolicyError(
            f"{where} must be a finite number, got {value!r}",
            details={"where": where, "value": str(value)},
        )
    return number


def _as_text(value: Any, where: str) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        raise PolicyError(
            f"{where} must be a name, got {type(value).__name__}",
            details={"where": where, "value": str(value)},
        )
    text = str(value).strip()
    if not text:
        raise PolicyError(
            f"{where} must not be empty",
            details={"where": where},
        )
    return text


def _as_mapping(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(
            f"{where} must be a mapping, got {type(value).__name__}",
            details={"where": where, "value": str(value)},
        )
    return value


def _as_list(value: Any, where: str) -> List[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise PolicyError(
        f"{where} must be a list, got {type(value).__name__}",
        details={"where": where, "value": str(value)},
    )


def _as_date(value: Any, where: str) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value)
    if len(text) > 10 and text[10] not in {"T", " "}:
        raise PolicyError(
            f"{where} has trailing characters after the date: {value!r}",
            remedy="Use YYYY-MM-DD, optionally followed by T or a space and a time.",
            details={"where": where, "value": text},
        )
    try:
        return _dt.date.fromisoformat(text[:10])
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
        raw = _as_mapping(raw, where)
        kind = _as_text(raw.get("kind", "protein"), f"{where}.kind")
        if kind not in KEY_KINDS:
            raise PolicyError(
                f"{where}.kind must be one of {sorted(KEY_KINDS)}, got {kind!r}",
                details={"where": where, "value": kind},
            )
        column = _as_text(_require(raw, "column", where), f"{where}.column")
        return SimilarityKey(name=name, column=column, kind=kind)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "column": self.column, "kind": self.kind}


@dataclass(frozen=True)
class SimilarityConstraint:
    key: str
    metric: str
    threshold: float
    backend: Optional[str] = None
    params: Tuple[Tuple[str, Any], ...] = ()
    kind: str = "similarity"

    @property
    def settings(self) -> Dict[str, Any]:
        return dict(self.params)

    def to_dict(self) -> Dict[str, Any]:
        out = {"kind": "similarity", "key": self.key, "metric": self.metric, "threshold": self.threshold}
        if self.backend:
            out["backend"] = self.backend
        if self.params:
            out["params"] = dict(sorted(self.params))
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
            f"nothing dated on or after {self.cutoff.isoformat()} by '{self.key}' may appear "
            f"in '{self.before}'"
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
        raw = _as_mapping(raw, where)
        kind = _as_text(raw.get("kind", "holdout"), f"{where}.kind")
        if kind == "kfold":
            folds = _as_int(raw.get("folds", 5), f"{where}.folds")
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
        fractions_raw = _as_mapping(
            raw.get("fractions") or {"train": 0.8, "test": 0.2}, f"{where}.fractions"
        )
        if len(fractions_raw) < 2:
            raise PolicyError(
                "scheme.fractions must map at least two fold names to fractions",
                details={"where": f"{where}.fractions"},
            )
        fractions = {
            _as_text(k, f"{where}.fractions"): _as_float(v, f"{where}.fractions.{k}")
            for k, v in fractions_raw.items()
        }
        for name, value in fractions.items():
            if value <= 0:
                raise PolicyError(
                    f"scheme.fractions.{name} must be positive",
                    details={"where": f"{where}.fractions.{name}", "value": value},
                )
        total = sum(fractions.values())
        if abs(total - 1.0) > 1e-6:
            if all(value < 1.0 for value in fractions.values()):
                raise PolicyError(
                    f"scheme.fractions look like fractions but sum to {total:g}, not 1",
                    remedy=(
                        "Make them sum to 1, or express them as weights greater than 1 "
                        "(for example 8 and 2) if you meant a ratio."
                    ),
                    details={"where": f"{where}.fractions", "sum": total},
                )
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
        raw = _as_mapping(raw, where)
        similarity_raw = _as_mapping(raw.get("similarity") or {}, f"{where}.similarity")
        groups_raw = _as_list(raw.get("groups") or [], f"{where}.groups")
        dates_value = raw.get("dates") or {}
        if isinstance(dates_value, str):
            dates_value = {dates_value: dates_value}
        dates_raw = _as_mapping(dates_value, f"{where}.dates")
        label_kind = _as_text(raw.get("label_kind", "auto"), f"{where}.label_kind")
        if label_kind not in {"auto", "categorical", "numeric"}:
            raise PolicyError(
                "dataset.label_kind must be auto, categorical or numeric",
                details={"where": f"{where}.label_kind", "value": label_kind},
            )
        return DatasetSpec(
            path=_as_text(raw["path"], f"{where}.path") if raw.get("path") else None,
            id_column=_as_text(_require(raw, "id", where), f"{where}.id"),
            label_column=_as_text(raw["label"], f"{where}.label") if raw.get("label") else None,
            label_kind=label_kind,
            similarity={name: SimilarityKey.parse(name, value) for name, value in similarity_raw.items()},
            groups=[_as_text(g, f"{where}.groups") for g in groups_raw],
            dates={
                _as_text(k, f"{where}.dates"): _as_text(v, f"{where}.dates.{k}")
                for k, v in dates_raw.items()
            },
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
            "splitproof": self.version,
            "dataset": self.dataset.to_dict(),
            "scheme": self.scheme.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any], *, source_path: Optional[Path] = None) -> "Policy":
        if not isinstance(raw, dict):
            raise PolicyError("policy must be a mapping at the top level")
        version = _as_int(raw.get("splitproof", POLICY_VERSION), "policy.splitproof")
        if version != POLICY_VERSION:
            raise PolicyError(
                f"policy version {version} is not supported by this splitproof build",
                remedy=f"Set 'splitproof: {POLICY_VERSION}' or upgrade splitproof.",
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
        seed = _as_int(raw.get("seed", 7), "policy.seed")
        _check_temporal_fits_scheme(constraints, scheme)
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
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise PolicyError(
                f"could not read the policy at {path}",
                remedy="Check the path. 'splitproof init <data>' writes a starter policy.",
                details={"path": str(path), "reason": str(exc)},
            ) from exc
        return Policy.from_dict(_parse_structured(text, path), source_path=Path(path))


def _parse_structured(text: str, path: Path) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        import json

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise PolicyError(
                f"policy file {path} is not valid JSON",
                remedy="Fix the syntax. The error below names the position.",
                details={"path": str(path), "reason": str(exc)},
            ) from exc
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise PolicyError(
            "reading a YAML policy needs PyYAML",
            remedy="Install splitproof normally (pip install splitproof) or use a .json policy.",
        ) from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError(
            f"policy file {path} is not valid YAML",
            remedy="Fix the syntax. The error below names the line.",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    if loaded is None:
        raise PolicyError(f"policy file {path} is empty")
    return loaded


def _check_temporal_fits_scheme(constraints: List[Constraint], scheme: Scheme) -> None:
    temporal = [c for c in constraints if isinstance(c, TemporalConstraint)]
    if not temporal:
        return
    if scheme.kind != "holdout":
        raise PolicyError(
            f"a temporal constraint needs named folds, so it cannot be used with the "
            f"'{scheme.kind}' scheme",
            remedy=(
                "Use scheme.kind: holdout with named fractions, so the constraint has a 'before' "
                "fold and an 'after' fold to talk about. Under kfold or leave-unit-out the fold "
                "names are generated, and a temporal constraint there would check nothing."
            ),
            details={"scheme": scheme.kind},
        )
    names = set(scheme.fold_names)
    for constraint in temporal:
        for role, side in (("before", constraint.before), ("after", constraint.after)):
            if side not in names:
                raise PolicyError(
                    f"the temporal constraint's '{role}' names fold '{side}', which the scheme "
                    f"does not define",
                    remedy=f"Scheme folds are {sorted(names)}. Rename, or add that fraction.",
                    details={"role": role, "fold": side, "available": sorted(names)},
                )


def _parse_constraints(raw: Any, dataset: DatasetSpec) -> List[Constraint]:
    if not isinstance(raw, list):
        raise PolicyError("constraints must be a list", details={"where": "constraints"})
    out: List[Constraint] = []
    for index, item in enumerate(raw):
        where = f"constraints[{index}]"
        item = _as_mapping(item, where)
        kind = _as_text(_require(item, "kind", where), f"{where}.kind")
        if kind == "similarity":
            out.append(_parse_similarity(item, where, dataset))
        elif kind == "disjoint":
            key = _as_text(_require(item, "key", where), f"{where}.key")
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
    key = _as_text(_require(item, "key", where), f"{where}.key")
    if key not in dataset.similarity:
        raise PolicyError(
            f"{where} references similarity key '{key}' which is not declared in dataset.similarity",
            remedy=f"Declare '{key}' under dataset.similarity.",
            details={"where": where, "key": key, "known": sorted(dataset.similarity)},
        )
    metric = _as_text(_require(item, "metric", where), f"{where}.metric")
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
    if metric in EQUIVALENCE_METRICS:
        threshold = _as_float(item.get("threshold", 1.0), f"{where}.threshold")
        if threshold != 1.0:
            raise PolicyError(
                f"{where}.metric '{metric}' compares values for equality, so its threshold is "
                f"always 1, not {threshold:g}",
                remedy="Drop the threshold, or set it to 1.",
                details={"where": where, "metric": metric, "value": threshold},
            )
    else:
        threshold = _as_float(_require(item, "threshold", where), f"{where}.threshold")
        if not 0.0 < threshold <= 1.0:
            raise PolicyError(
                f"{where}.threshold must be in (0, 1], got {threshold}",
                details={"where": where, "value": threshold},
            )
    backend = _as_text(item["backend"], f"{where}.backend") if item.get("backend") else None
    params_raw = _as_mapping(item.get("params") or {}, f"{where}.params")
    params: Dict[str, Any] = {}
    for name, value in params_raw.items():
        label = _as_text(name, f"{where}.params")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise PolicyError(
                f"{where}.params.{label} must be a number or a name, got {type(value).__name__}",
                details={"where": f"{where}.params.{label}"},
            )
        params[label] = value
    return SimilarityConstraint(
        key=key,
        metric=metric,
        threshold=threshold,
        backend=backend,
        params=tuple(sorted(params.items())),
    )


def _parse_temporal(item: Dict[str, Any], where: str, dataset: DatasetSpec) -> TemporalConstraint:
    key = _as_text(_require(item, "key", where), f"{where}.key")
    if key not in dataset.dates:
        raise PolicyError(
            f"{where} references date '{key}' which is not listed in dataset.dates",
            remedy=f"Add '{key}' to dataset.dates.",
            details={"where": where, "key": key, "known": sorted(dataset.dates)},
        )
    cutoff = _as_date(_require(item, "cutoff", where), f"{where}.cutoff")
    before = _as_text(item.get("before", "train"), f"{where}.before")
    after = _as_text(item.get("after", "test"), f"{where}.after")
    if before == after:
        raise PolicyError(
            f"{where}.before and {where}.after must name different folds",
            details={"where": where, "value": before},
        )
    on_span = _as_text(item.get("on_span", "test"), f"{where}.on_span")
    if on_span not in SPAN_ACTIONS:
        extra = (
            " Pinning a spanning unit to the 'before' fold would put post-cutoff items there, "
            "which is the very thing the constraint forbids."
            if on_span == "train"
            else ""
        )
        raise PolicyError(
            f"{where}.on_span must be one of {sorted(SPAN_ACTIONS)}, got {on_span!r}.{extra}",
            details={"where": where, "value": on_span},
        )
    return TemporalConstraint(key=key, cutoff=cutoff, before=before, after=after, on_span=on_span)
