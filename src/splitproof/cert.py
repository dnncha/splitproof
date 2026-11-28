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
from .json_input import load_json
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
        "splitproof": {"schema": SCHEMA, "version": __version__},
        "created": created or _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "status": overall_status(reports),
        "policy": policy_payload,
        "policy_digest": digest(policy_payload),
        "dataset": {
            "digest": data.digest_for(policy),
            "entities": len(data),
            "id_column": policy.dataset.id_column,
        },
        "source": {"path": _relative_source(data.path)},
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


def _relative_source(path: Optional[Path]) -> Optional[str]:
    """Where the data was, recorded relative to the working directory when it is inside it.

    verify will follow a relative path but not an absolute one, because a certificate
    is untrusted input and must not be able to point verify at a file of its choosing.
    """
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(Path(path).resolve())


def write(path: Path, certificate: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read(path: Path) -> Dict[str, Any]:
    try:
        loaded = load_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise CertificateMismatch(
            f"could not read certificate {path}: {exc}",
            remedy="Pass the splitproof.cert.json produced by 'splitproof split'.",
            details={"path": str(path)},
        ) from exc
    if not isinstance(loaded, dict) or "splitproof" not in loaded:
        raise CertificateMismatch(
            f"{path} is not a splitproof certificate",
            details={"path": str(path)},
        )
    header = loaded.get("splitproof")
    if (
        not isinstance(header, dict)
        or not isinstance(header.get("schema"), int)
        or isinstance(header.get("schema"), bool)
    ):
        raise CertificateMismatch(
            f"{path} has no readable splitproof schema header",
            remedy="Pass the splitproof.cert.json produced by 'splitproof split'.",
            details={"path": str(path), "header": str(header)},
        )
    schema = header["schema"]
    if schema != SCHEMA:
        raise CertificateMismatch(
            f"certificate schema {schema} is not supported by splitproof {__version__}",
            remedy=f"This build reads schema {SCHEMA}.",
            details={"schema": schema, "supported": SCHEMA},
        )
    for section, shape in SECTION_SHAPES.items():
        if not isinstance(loaded.get(section), shape):
            raise CertificateMismatch(
                f"{path} has no readable '{section}' section",
                remedy="Pass the splitproof.cert.json produced by 'splitproof split', unedited.",
                details={"path": str(path), "section": section, "expected": shape.__name__},
            )
    return loaded


SECTION_SHAPES: Dict[str, type] = {
    "assignment": dict,
    "constraints": list,
    "dataset": dict,
    "diagnostics": dict,
    "policy": dict,
    "status": str,
    "units": dict,
}


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
    policy = certificate.get("policy")
    if not isinstance(policy, dict):
        raise CertificateMismatch(
            "the certificate carries no policy, so there is nothing to verify against",
            remedy="Pass a certificate produced by 'splitproof split' or 'splitproof audit'.",
        )
    return Policy.from_dict(policy)


VOLATILE = ("created", "environment", "audited", "source")

SECTIONS = (
    ("splitproof", "the certificate was produced by this build of splitproof"),
    ("policy", "the embedded policy is the one being verified against"),
    ("policy_digest", "the policy still hashes to the recorded digest"),
    ("dataset", "the data holds the same ids and values the split was built from"),
    ("assignment", "the split assignment is the one the certificate describes"),
    ("units", "the independent units, backends and their parameters are unchanged"),
    ("constraints", "every constraint re-measures to exactly the recorded values"),
    ("diagnostics", "the leakage diagnostics re-measure to exactly the recorded values"),
    ("status", "the overall verdict is unchanged"),
    ("seed", "the seed is unchanged"),
)


def comparable(certificate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in certificate.items() if key not in VOLATILE}


def first_difference(stored: Any, fresh: Any, path: str = "") -> Optional[str]:
    if isinstance(stored, dict) and isinstance(fresh, dict):
        for key in sorted(set(stored) | set(fresh)):
            here = f"{path}.{key}" if path else str(key)
            if key not in stored:
                return f"{here} is present now but not in the certificate"
            if key not in fresh:
                return f"{here} is in the certificate but not recomputed"
            deeper = first_difference(stored[key], fresh[key], here)
            if deeper:
                return deeper
        return None
    if isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            return f"{path or 'value'} has {len(stored)} entries in the certificate and {len(fresh)} now"
        for index, (left, right) in enumerate(zip(stored, fresh, strict=True)):
            deeper = first_difference(left, right, f"{path}[{index}]")
            if deeper:
                return deeper
        return None
    if type(stored) is not type(fresh) and not (
        isinstance(stored, (int, float))
        and isinstance(fresh, (int, float))
        and not isinstance(stored, bool)
        and not isinstance(fresh, bool)
    ):
        return (
            f"{path or 'value'}: certificate says {stored!r} ({type(stored).__name__}), "
            f"recomputed {fresh!r} ({type(fresh).__name__})"
        )
    if stored != fresh:
        return f"{path or 'value'}: certificate says {stored!r}, recomputed {fresh!r}"
    return None


def verify(
    certificate: Dict[str, Any],
    recomputed: Dict[str, Any],
    reports: Optional[List[ConstraintReport]] = None,
) -> VerifyResult:
    """Compare a stored certificate against one rebuilt from the data and the split.

    Everything the certificate asserts is recomputed and compared, not just the
    digests: per-constraint measured values, the unit structure, the backends and
    their parameters, and the leakage diagnostics. A certificate whose numbers were
    edited by hand does not verify.
    """
    stored = comparable(certificate)
    fresh = comparable(recomputed)
    checks: List[Dict[str, Any]] = []
    for name, describes in SECTIONS:
        difference = first_difference(stored.get(name), fresh.get(name), name)
        checks.append(
            {
                "name": name,
                "ok": difference is None,
                "describes": describes,
                **({"difference": difference} if difference else {}),
            }
        )
    extra = sorted((set(stored) | set(fresh)) - {name for name, _ in SECTIONS})
    for name in extra:
        if name not in fresh:
            difference = f"{name} is in the certificate but splitproof does not produce it"
        else:
            difference = first_difference(stored.get(name), fresh.get(name), name)
        checks.append(
            {
                "name": name,
                "ok": difference is None,
                "describes": "an unrecognised certificate field is unchanged",
                **({"difference": difference} if difference else {}),
            }
        )
    return VerifyResult(
        ok=all(check["ok"] for check in checks),
        checks=checks,
        status=str(fresh.get("status", "unknown")),
        reports=list(reports or []),
    )
