from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

from . import __version__, api, cert as cert_module, dataset as dataset_module
from . import report as report_module
from . import units as units_module
from .errors import ArgumentsInvalid, SplitproofError
from .policy import Policy

FALLBACK_PROTOCOL = "2025-06-18"

def _prop(kind: str, description: str) -> Dict[str, str]:
    return {"type": kind, "description": description}


def _tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: List[str],
    *,
    read_only: bool = False,
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    }


TOOLS: List[Dict[str, Any]] = [
    _tool(
        "splitproof_init",
        "Inspect a CSV, TSV, FASTA, or Parquet file and write a starter splitproof policy YAML, "
        "guessing the id column, similarity keys, group columns, and date columns. A guess is "
        "not a policy: read and edit the generated file before building a split. Sequence "
        "constraints default to metric 'identity', which needs MMseqs2 on PATH.",
        {
            "data": _prop("string", "Path to the input dataset (.csv, .tsv, .fasta, or .parquet)."),
            "out": _prop("string", "Output policy path inside the working directory (default: policy.yaml)."),
            "id_column": _prop("string", "Explicit entity id column name, skipping header guessing."),
        },
        ["data"],
    ),
    _tool(
        "splitproof_split",
        "Build a deterministic train/test or k-fold split from a policy and write a verifiable "
        "certificate. Returns fold sizes, independent unit statistics, per-constraint status, "
        "and leakage diagnostics. Status 'satisfied' means an exhaustive check passed on the "
        "declared constraints; 'satisfied_unproven' means a heuristic/approximate backend ran "
        "or some rows had no usable key value (evidence, not proof). Set require_exact=true to "
        "fail unless every constraint is 'satisfied'. Output paths must stay inside the working directory.",
        {
            "policy": _prop("string", "Path to the splitproof YAML or JSON policy file."),
            "data": _prop("string", "Optional dataset path overriding dataset.path in the policy."),
            "out": _prop("string", "Output path for the id,fold split file inside the working directory (default: split.csv)."),
            "cert": _prop("string", "Output path for the certificate JSON inside the working directory (default: splitproof.cert.json)."),
            "report": _prop("string", "Optional output path for a Markdown report inside the working directory."),
            "tolerance": _prop("number", "How far the largest unit may exceed a fold's target size before failing with GIANT_UNIT (default: 0.1)."),
            "require_exact": _prop("boolean", "Fail (ok=false) unless every constraint status is 'satisfied' rather than 'satisfied_unproven'."),
            "skip_neighbours": _prop("boolean", "Skip nearest-neighbour similarity and 1-NN baseline diagnostics for faster runs."),
        },
        ["policy"],
    ),
    _tool(
        "splitproof_audit",
        "Measure leakage in an existing split, such as a published benchmark's fold column or "
        "a two-column id,fold file. Returns per-constraint status, crossing pairs ranked by "
        "similarity, nearest-neighbour similarity percentiles, and a 1-NN similarity baseline "
        "to report beside any model score (a model that does not beat it has not shown it "
        "learned beyond similarity). Pass either split_column or split.",
        {
            "policy": _prop("string", "Path to the splitproof YAML or JSON policy file."),
            "data": _prop("string", "Optional dataset path overriding dataset.path in the policy."),
            "split_column": _prop("string", "Column name inside the dataset file containing fold assignments."),
            "split": _prop("string", "Path to a two-column id,fold CSV or TSV file."),
            "cert": _prop("string", "Optional output path inside the working directory to save the audit certificate JSON."),
            "report": _prop("string", "Optional output path inside the working directory to write a Markdown leakage report."),
            "skip_neighbours": _prop("boolean", "Skip nearest-neighbour similarity and 1-NN baseline diagnostics for faster runs."),
        },
        ["policy"],
    ),
    _tool(
        "splitproof_verify",
        "Re-prove a splitproof certificate from scratch against the data and split assignment. "
        "Recomputes digests, independent units, every constraint measurement, and leakage "
        "diagnostics, failing on any drift or manual edit. Certificates are treated as untrusted "
        "input: pass data explicitly if the dataset is outside the working directory.",
        {
            "cert": _prop("string", "Path to the splitproof certificate JSON file to verify."),
            "data": _prop("string", "Optional dataset path; required if the certificate's recorded path is outside the working directory."),
            "split": _prop("string", "Path to the two-column id,fold split file to verify."),
            "split_column": _prop("string", "Column name inside the dataset file containing fold assignments."),
            "require_exact": _prop("boolean", "Fail (ok=false) unless every constraint is proven exhaustively ('satisfied')."),
            "skip_neighbours": _prop("boolean", "Skip nearest-neighbour diagnostics during recomputation (must match how the certificate was built)."),
        },
        ["cert"],
        read_only=True,
    ),
    _tool(
        "splitproof_units",
        "Inspect the independent connected-component units a policy implies without building a "
        "split. Returns total unit count, largest unit size, largest_unit_fraction, and size "
        "percentiles. Call this first to check effective sample size or when splitproof_split "
        "fails with GIANT_UNIT or SPLIT_INFEASIBLE.",
        {
            "policy": _prop("string", "Path to the splitproof YAML or JSON policy file."),
            "data": _prop("string", "Optional dataset path overriding dataset.path in the policy."),
        },
        ["policy"],
        read_only=True,
    ),
    _tool(
        "splitproof_explain",
        "Format an existing splitproof certificate as plain text or Markdown without "
        "recomputing from the dataset. Use splitproof_verify to re-prove a certificate.",
        {
            "cert": _prop("string", "Path to an existing splitproof certificate JSON file."),
            "markdown": _prop("boolean", "Return a full Markdown report instead of a plain-text summary."),
        },
        ["cert"],
        read_only=True,
    ),
]


def _optional_text(arguments: Dict[str, Any], name: str) -> Optional[str]:
    value = arguments.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ArgumentsInvalid(
            f"argument '{name}' must be a string, got {type(value).__name__}", details={"argument": name}
        )
    return value


def _optional_path(arguments: Dict[str, Any], name: str) -> Optional[Path]:
    value = _optional_text(arguments, name)
    return Path(value) if value else None


def _writable(value: Optional[str], default: str) -> Path:
    """Where a tool may write: inside the working tree, and nowhere else.

    An agent reaches these arguments from whatever it was reading, which may be an
    untrusted report or certificate, so an unconfined path is a way to overwrite files
    elsewhere on the machine.
    """
    candidate = Path(str(value)) if value else Path(default)
    root = Path.cwd().resolve()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if root != resolved and root not in resolved.parents:
        raise SplitproofError(
            f"refusing to write outside the working directory: {candidate}",
            remedy=(
                "Pass a path inside the current directory. Writes are confined because these "
                "arguments can come from content the agent was reading."
            ),
            details={"requested": str(candidate), "working_directory": str(root)},
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _summarise(certificate: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "status": certificate["status"],
        "summary": report_module.summary(certificate),
        "folds": certificate["assignment"]["stats"],
        "units": certificate["units"],
        "constraints": certificate["constraints"],
        "diagnostics": certificate["diagnostics"],
    }
    payload.update(extra or {})
    return payload


def _required(arguments: Dict[str, Any], name: str) -> Any:
    value = arguments.get(name)
    if value is None or value == "":
        raise ArgumentsInvalid(f"argument '{name}' is required", details={"argument": name})
    if not isinstance(value, str):
        raise ArgumentsInvalid(
            f"argument '{name}' must be a string, got {type(value).__name__}", details={"argument": name}
        )
    return value


def _number(arguments: Dict[str, Any], name: str, default: float) -> float:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ArgumentsInvalid(f"argument '{name}' must be a number", details={"argument": name, "value": str(value)})
    return float(value)


def _flag(arguments: Dict[str, Any], name: str) -> bool:
    value = arguments.get(name, False)
    if not isinstance(value, bool):
        raise ArgumentsInvalid(f"argument '{name}' must be true or false", details={"argument": name, "value": str(value)})
    return value


def _do_init(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from .cli import guess_policy, _write_policy

    out = _writable(_optional_text(arguments, "out"), "policy.yaml")
    policy, seen = guess_policy(Path(_required(arguments, "data")), _optional_text(arguments, "id_column"))
    _write_policy(out, policy)
    return {
        "ok": True,
        "policy": str(out),
        "guessed": policy,
        "seen": seen,
        "summary": (
            f"Guessed a policy for {seen['rows']} rows: id '{seen['id']}', similarity keys "
            f"{seen['similarity']}, groups {seen['groups']}, dates {seen['dates']}. A guess is "
            "not a policy: read it and correct it before trusting a split built from it."
        ),
    }


def _do_split(arguments: Dict[str, Any]) -> Dict[str, Any]:
    policy = Policy.load(Path(_required(arguments, "policy")))
    outcome = api.make_split(
        policy,
        data_path=_optional_path(arguments, "data"),
        tolerance=_number(arguments, "tolerance", api.DEFAULT_TOLERANCE),
        skip_neighbours=_flag(arguments, "skip_neighbours"),
    )
    out = _writable(_optional_text(arguments, "out"), "split.csv")
    certificate_path = _writable(_optional_text(arguments, "cert"), "splitproof.cert.json")
    api.write_split(out, outcome.data, outcome.assignment, policy.dataset.id_column)
    cert_module.write(certificate_path, outcome.certificate)
    if _optional_text(arguments, "report"):
        _writable(_optional_text(arguments, "report"), "report.md").write_text(
            report_module.markdown(outcome.certificate), encoding="utf-8"
        )
    ok = outcome.ok
    if _flag(arguments, "require_exact") and outcome.certificate["status"] == "satisfied_unproven":
        ok = False
    return _summarise(
        outcome.certificate,
        {"ok": ok, "split": str(out), "certificate": str(certificate_path)},
    )


def _do_audit(arguments: Dict[str, Any]) -> Dict[str, Any]:
    policy = Policy.load(Path(_required(arguments, "policy")))
    outcome = api.audit(
        policy,
        data_path=_optional_path(arguments, "data"),
        split_column=_optional_text(arguments, "split_column"),
        split_path=_optional_path(arguments, "split"),
        skip_neighbours=_flag(arguments, "skip_neighbours"),
    )
    if _optional_text(arguments, "cert"):
        cert_module.write(_writable(_required(arguments, "cert"), "audit.cert.json"), outcome.certificate)
    if _optional_text(arguments, "report"):
        _writable(_optional_text(arguments, "report"), "leakage.md").write_text(
            report_module.markdown(outcome.certificate), encoding="utf-8"
        )
    return _summarise(outcome.certificate, {"ok": outcome.ok})


def _do_verify(arguments: Dict[str, Any]) -> Dict[str, Any]:
    result = api.verify(
        Path(_required(arguments, "cert")),
        data_path=_optional_path(arguments, "data"),
        split_path=_optional_path(arguments, "split"),
        split_column=_optional_text(arguments, "split_column"),
        skip_neighbours=_flag(arguments, "skip_neighbours"),
    )
    payload = result.to_dict()
    unproven = _flag(arguments, "require_exact") and result.status == "satisfied_unproven"
    payload["ok"] = result.ok and result.status != "violated" and not unproven
    payload["verified"] = result.ok
    payload["summary"] = (
        "the certificate re-derives exactly"
        if result.ok
        else "the certificate does not re-derive from this data and split"
    )
    if result.ok and result.status == "violated":
        payload["summary"] += ", and it records a violated constraint"
    return payload


def _do_units(arguments: Dict[str, Any]) -> Dict[str, Any]:
    policy = Policy.load(Path(_required(arguments, "policy")))
    data = dataset_module.load_for(policy, _optional_path(arguments, "data"))
    return units_module.build(policy, data).to_dict()


def _do_explain(arguments: Dict[str, Any]) -> Dict[str, Any]:
    certificate = cert_module.read(Path(_required(arguments, "cert")))
    body = (
        report_module.markdown(certificate)
        if _flag(arguments, "markdown")
        else report_module.summary(certificate)
    )
    return {"status": certificate.get("status"), "text": body}


HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "splitproof_init": _do_init,
    "splitproof_split": _do_split,
    "splitproof_audit": _do_audit,
    "splitproof_verify": _do_verify,
    "splitproof_units": _do_units,
    "splitproof_explain": _do_explain,
}


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    handler = HANDLERS.get(name)
    if handler is None:
        return {
            "isError": True,
            "content": [{"type": "text", "text": json.dumps({"error": {"code": "UNKNOWN_TOOL", "message": name}})}],
        }
    try:
        payload = handler(arguments)
        failed = payload.get("ok") is False
    except SplitproofError as error:
        payload = {"ok": False, "error": error.to_dict()}
        failed = True
    except Exception as error:
        payload = {
            "ok": False,
            "error": {
                "code": "UNEXPECTED",
                "message": f"{type(error).__name__}: {error}",
                "remedy": "This is a bug in splitproof. Please report it with the inputs.",
            },
        }
        failed = True
    return {
        "isError": failed,
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, sort_keys=True)}],
        "structuredContent": payload,
    }


def handle(message: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "a JSON-RPC message must be an object"},
        }
    method = message.get("method")
    if "id" not in message:
        return None
    identifier = message.get("id")
    if method == "initialize":
        params = message.get("params")
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        if not isinstance(requested, str) or not requested:
            requested = FALLBACK_PROTOCOL
        return _result(
            identifier,
            {
                "protocolVersion": requested,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "splitproof", "version": __version__},
                "instructions": (
                    "splitproof builds, audits, and re-verifies leakage-free train/test splits for "
                    "biological and chemical datasets. Workflow: start with splitproof_init on a "
                    "data file, review and correct the guessed policy, optionally inspect "
                    "splitproof_units to check effective sample size, then run splitproof_split. "
                    "Use splitproof_audit on any existing or published split before trusting "
                    "benchmark numbers, and report nearest_neighbour_baseline beside any model "
                    "score. Always check per-constraint status: 'satisfied' means an exhaustive "
                    "check passed on the declared constraints; 'satisfied_unproven' means a "
                    "heuristic or approximate backend ran or some rows were unconstrained "
                    "(evidence, not proof; pass require_exact=true to fail on unproven results); "
                    "'violated' lists the worst crossing pairs. Never call kmer_jaccard sequence "
                    "identity. Use splitproof_verify to re-derive a certificate from scratch."
                ),
            },
        )
    if method == "ping":
        return _result(identifier, {})
    if method == "tools/list":
        return _result(identifier, {"tools": TOOLS})
    if method == "tools/call":
        params = message.get("params")
        if not isinstance(params, dict):
            params = {}
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        return _result(identifier, call_tool(str(params.get("name") or ""), arguments))
    return {
        "jsonrpc": "2.0",
        "id": identifier,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def _result(identifier: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": identifier, "result": result}


def serve(stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None) -> int:
    source = stdin or sys.stdin
    sink = stdout or sys.stdout
    for raw_line in source:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            sink.write(
                json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n"
            )
            sink.flush()
            continue
        try:
            response = handle(message)
        except Exception as error:
            identifier = message.get("id") if isinstance(message, dict) else None
            response = {
                "jsonrpc": "2.0",
                "id": identifier,
                "error": {"code": -32603, "message": f"internal error: {type(error).__name__}: {error}"},
            }
        if response is not None:
            sink.write(json.dumps(response) + "\n")
            sink.flush()
    return 0
