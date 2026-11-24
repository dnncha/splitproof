from __future__ import annotations

from typing import Any, Dict, Optional


class SplitproofError(Exception):
    code = "SPLITPROOF_ERROR"

    def __init__(self, message: str, *, remedy: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.remedy = remedy
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "remedy": self.remedy,
            "details": self.details,
        }


class PolicyError(SplitproofError):
    code = "POLICY_INVALID"


class DataError(SplitproofError):
    code = "DATA_INVALID"


class BackendUnavailable(SplitproofError):
    code = "BACKEND_UNAVAILABLE"


class InfeasibleSplit(SplitproofError):
    code = "SPLIT_INFEASIBLE"


class GiantUnit(InfeasibleSplit):
    code = "GIANT_UNIT"


class BackendFailed(SplitproofError):
    code = "BACKEND_FAILED"


class VacuousConstraint(SplitproofError):
    code = "CONSTRAINT_VACUOUS"


class CertificateMismatch(SplitproofError):
    code = "CERT_MISMATCH"


class ConstraintViolated(SplitproofError):
    code = "CONSTRAINT_VIOLATED"


class FileProblem(SplitproofError):
    code = "FILE_ERROR"


class Unexpected(SplitproofError):
    code = "UNEXPECTED"


class ArgumentsInvalid(SplitproofError):
    code = "ARGUMENTS_INVALID"
