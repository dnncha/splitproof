from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

from . import __version__, api, cert as cert_module, dataset as dataset_module
from . import report as report_module
from . import units as units_module
from .errors import SplitproofError
from .policy import Policy

FALLBACK_PROTOCOL = "2025-06-18"

STRING = {"type": "string"}
BOOLEAN = {"type": "boolean"}
NUMBER = {"type": "number"}


def _tool(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOLS: List[Dict[str, Any]] = [
    _tool(
        "splitproof_init",
        "Inspect a data file and write a starter splitproof policy, guessing the id column, "
        "similarity keys, group columns and date columns. A guess is not a policy: read it "
        "and correct it before trusting a split built from it.",
        {"data": STRING, "out": STRING, "id_column": STRING},
        ["data"],
    ),
    _tool(
        "splitproof_split",
        "Build a leakage-free split from a policy and write a certificate proving it. Returns "
        "the fold sizes, the per-constraint status, the unit structure and the leakage "
        "diagnostics. A 'satisfied' status means every pair at or above the declared threshold "
        "was compared and none crosses folds; 'satisfied_unproven' means the backend is "
        "heuristic so absence is evidence, not proof.",
        {
            "policy": STRING,
            "data": STRING,
            "out": STRING,
            "cert": STRING,
            "report": STRING,
            "tolerance": NUMBER,
            "require_exact": BOOLEAN,
            "skip_neighbours": BOOLEAN,
        },
        ["policy"],
    ),
    _tool(
        "splitproof_audit",
        "Measure leakage in a split that already exists, such as a published benchmark's "
        "train/test column. Returns the crossing pairs ranked by similarity, the "
        "nearest-neighbour similarity distribution, and the nearest-neighbour baseline score "
        "that any real model must beat.",
        {
            "policy": STRING,
            "data": STRING,
            "split_column": STRING,
            "split": STRING,
            "cert": STRING,
            "report": STRING,
            "skip_neighbours": BOOLEAN,
        },
        ["policy"],
    ),
    _tool(
        "splitproof_verify",
        "Re-prove a certificate against the data and the split. Recomputes the data digest, the "
        "assignment digest and every constraint measurement, and reports which checks failed. "
        "Use this to confirm a split has not drifted since it was certified.",
        {
            "cert": STRING,
            "data": STRING,
            "split": STRING,
            "split_column": STRING,
            "require_exact": BOOLEAN,
            "skip_neighbours": BOOLEAN,
        },
        ["cert"],
    ),
    _tool(
        "splitproof_units",
        "Report the independent units a policy implies, without building a split: how many "
        "units exist, how large the biggest one is, and what fraction of the data it holds. "
        "Use this first when a split is refused as infeasible.",
        {"policy": STRING, "data": STRING},
        ["policy"],
    ),
    _tool(
        "splitproof_explain",
        "Summarise an existing certificate in plain language, or as Markdown.",
        {"cert": STRING, "markdown": BOOLEAN},
        ["cert"],
    ),
]


def _path(value: Optional[str]) -> Optional[Path]:
    return Path(value) if value else None


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


def _do_init(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from .cli import guess_policy, _write_policy

    out = Path(arguments.get("out") or "policy.yaml")
    policy, seen = guess_policy(Path(arguments["data"]), arguments.get("id_column"))
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
    policy = Policy.load(Path(arguments["policy"]))
    outcome = api.make_split(
        policy,
        data_path=_path(arguments.get("data")),
        tolerance=float(arguments.get("tolerance", api.DEFAULT_TOLERANCE)),
        skip_neighbours=bool(arguments.get("skip_neighbours", False)),
    )
    out = Path(arguments.get("out") or "split.csv")
    certificate_path = Path(arguments.get("cert") or "splitproof.cert.json")
    api.write_split(out, outcome.data, outcome.assignment, policy.dataset.id_column)
    cert_module.write(certificate_path, outcome.certificate)
    if arguments.get("report"):
        Path(arguments["report"]).write_text(
            report_module.markdown(outcome.certificate), encoding="utf-8"
        )
    ok = outcome.ok
    if arguments.get("require_exact") and outcome.certificate["status"] == "satisfied_unproven":
        ok = False
    return _summarise(
        outcome.certificate,
        {"ok": ok, "split": str(out), "certificate": str(certificate_path)},
    )


def _do_audit(arguments: Dict[str, Any]) -> Dict[str, Any]:
    policy = Policy.load(Path(arguments["policy"]))
    outcome = api.audit(
        policy,
        data_path=_path(arguments.get("data")),
        split_column=arguments.get("split_column"),
        split_path=_path(arguments.get("split")),
        skip_neighbours=bool(arguments.get("skip_neighbours", False)),
    )
    if arguments.get("cert"):
        cert_module.write(Path(arguments["cert"]), outcome.certificate)
    if arguments.get("report"):
        Path(arguments["report"]).write_text(
            report_module.markdown(outcome.certificate), encoding="utf-8"
        )
    return _summarise(outcome.certificate, {"ok": outcome.ok})


def _do_verify(arguments: Dict[str, Any]) -> Dict[str, Any]:
    result = api.verify(
        Path(arguments["cert"]),
        data_path=_path(arguments.get("data")),
        split_path=_path(arguments.get("split")),
        split_column=arguments.get("split_column"),
        skip_neighbours=bool(arguments.get("skip_neighbours", False)),
    )
    payload = result.to_dict()
    unproven = arguments.get("require_exact") and result.status == "satisfied_unproven"
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
    policy = Policy.load(Path(arguments["policy"]))
    data = dataset_module.load_for(policy, _path(arguments.get("data")))
    return units_module.build(policy, data).to_dict()


def _do_explain(arguments: Dict[str, Any]) -> Dict[str, Any]:
    certificate = cert_module.read(Path(arguments["cert"]))
    body = (
        report_module.markdown(certificate)
        if arguments.get("markdown")
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
    except (OSError, ValueError, KeyError) as error:
        payload = {"ok": False, "error": {"code": "UNEXPECTED", "message": str(error)}}
        failed = True
    return {
        "isError": failed,
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, sort_keys=True)}],
        "structuredContent": payload,
    }


def handle(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = message.get("method")
    identifier = message.get("id")
    if method == "initialize":
        requested = (message.get("params") or {}).get("protocolVersion") or FALLBACK_PROTOCOL
        return _result(
            identifier,
            {
                "protocolVersion": requested,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "splitproof", "version": __version__},
                "instructions": (
                    "splitproof builds, audits and proves leakage-free train/test splits for "
                    "biological data. Start with splitproof_init on a data file, correct the "
                    "guessed policy, then splitproof_split. Audit any split you did not build "
                    "yourself with splitproof_audit and treat its nearest-neighbour baseline as "
                    "the score a model must beat."
                ),
            },
        )
    if method in {"notifications/initialized", "initialized"}:
        return None
    if method == "ping":
        return _result(identifier, {})
    if method == "tools/list":
        return _result(identifier, {"tools": TOOLS})
    if method == "tools/call":
        params = message.get("params") or {}
        return _result(identifier, call_tool(params.get("name", ""), params.get("arguments") or {}))
    if identifier is None:
        return None
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
    for line in source:
        line = line.strip()
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
        response = handle(message)
        if response is not None:
            sink.write(json.dumps(response) + "\n")
            sink.flush()
    return 0
