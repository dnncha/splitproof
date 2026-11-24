from __future__ import annotations

import datetime as _dt
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .assign import Assignment, DROPPED
from .dataset import Dataset
from .digest import digest
from .errors import CertificateMismatch
from .evaluate import SATISFIED, SATISFIED_UNPROVEN, VIOLATED, ConstraintReport
from .policy import Policy
from .units import UnitGraph

SCHEMA = 1


def assignment_digest(data: Dataset, assignment: Assignment) -> str:
    rows = sorted((data.ids[i], assignment.fold_of.get(i, DROPPED)) for i in range(len(data)))
    return digest({"rows": rows})


def overall_status(reports: List[ConstraintReport]) -> str:
    if any(report.status == VIOLATED for report in reports):
        return VIOLATED
    if any(report.status == SATISFIED_UNPROVEN for report in reports):
        return SATISFIED_UNPROVEN
    return SATISFIED


def build(
    policy: Policy,
    data: Dataset,
    graph: UnitGraph,
    assignment: Assignment,
    reports: List[ConstraintReport],
    diagnostics: Dict[str, Any],
    *,
    created: Optional[str] = None,
) -> Dict[str, Any]:
    policy_payload = policy.to_dict()
    return {
        "cutsite": {"schema": SCHEMA, "version": __version__},
        "created": created or _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "status": overall_status(reports),
        "policy": policy_payload,
        "policy_digest": digest(policy_payload),
        "dataset": {
            "digest": data.digest_for(policy),
            "entities": len(data),
            "path": str(data.path) if data.path else None,
            "id_column": policy.dataset.id_column,
        },
        "assignment": {
            "digest": assignment_digest(data, assignment),
            **assignment.to_dict(),
        },
        "units": graph.to_dict(),
        "constraints": [report.to_dict() for report in reports],
        "diagnostics": diagnostics,
        "seed": policy.seed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system().lower(),
        },
    }


def write(path: Path, certificate: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read(path: Path) -> Dict[str, Any]:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CertificateMismatch(
            f"could not read certificate {path}",
            remedy="Pass the cutsite.cert.json produced by 'cutsite split'.",
            details={"path": str(path)},
        ) from exc
    if not isinstance(loaded, dict) or "cutsite" not in loaded:
        raise CertificateMismatch(
            f"{path} is not a cutsite certificate",
            details={"path": str(path)},
        )
    schema = int(loaded.get("cutsite", {}).get("schema", 0))
    if schema != SCHEMA:
        raise CertificateMismatch(
            f"certificate schema {schema} is not supported by cutsite {__version__}",
            remedy=f"This build reads schema {SCHEMA}.",
            details={"schema": schema, "supported": SCHEMA},
        )
    return loaded


@dataclass
class VerifyResult:
    ok: bool
    checks: List[Dict[str, Any]]
    status: str
    reports: List[ConstraintReport]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "checks": self.checks,
            "constraints": [report.to_dict() for report in self.reports],
        }


def policy_from(certificate: Dict[str, Any]) -> Policy:
    return Policy.from_dict(certificate["policy"])


def verify(
    certificate: Dict[str, Any],
    data: Dataset,
    assignment: Assignment,
    reports: List[ConstraintReport],
    policy: Policy,
) -> VerifyResult:
    checks: List[Dict[str, Any]] = []

    recomputed_policy = digest(policy.to_dict())
    checks.append(
        _check(
            "policy_digest",
            certificate.get("policy_digest"),
            recomputed_policy,
            "the policy in the certificate still hashes to the recorded digest",
        )
    )
    checks.append(
        _check(
            "dataset_digest",
            certificate.get("dataset", {}).get("digest"),
            data.digest_for(policy),
            "the data file is byte-for-byte the one the split was built from",
        )
    )
    checks.append(
        _check(
            "assignment_digest",
            certificate.get("assignment", {}).get("digest"),
            assignment_digest(data, assignment),
            "the split assignment is the one the certificate describes",
        )
    )

    recorded = {item["id"]: item.get("status") for item in certificate.get("constraints", [])}
    for report in reports:
        expected = recorded.get(report.id)
        checks.append(
            {
                "name": f"constraint:{report.id}",
                "ok": expected == report.status and report.status != VIOLATED,
                "expected": expected,
                "measured": report.status,
                "describes": report.description,
            }
        )
    missing = sorted(set(recorded) - {report.id for report in reports})
    for identifier in missing:
        checks.append(
            {
                "name": f"constraint:{identifier}",
                "ok": False,
                "expected": recorded[identifier],
                "measured": "not_evaluated",
                "describes": "a constraint in the certificate was not re-checked",
            }
        )

    status = overall_status(reports)
    return VerifyResult(
        ok=all(check["ok"] for check in checks),
        checks=checks,
        status=status,
        reports=reports,
    )


def _check(name: str, expected: Any, measured: Any, describes: str) -> Dict[str, Any]:
    return {
        "name": name,
        "ok": expected == measured,
        "expected": expected,
        "measured": measured,
        "describes": describes,
    }
