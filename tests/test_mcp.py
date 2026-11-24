from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

from splitproof import mcp
from splitproof.policy import Policy


def _drive(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stdin = io.StringIO("\n".join(json.dumps(message) for message in messages) + "\n")
    stdout = io.StringIO()
    mcp.serve(stdin=stdin, stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_initialize_echoes_the_clients_protocol_version():
    responses = _drive(
        [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}]
    )
    assert responses[0]["result"]["protocolVersion"] == "2024-11-05"
    assert responses[0]["result"]["serverInfo"]["name"] == "splitproof"
    assert "capabilities" in responses[0]["result"]


def test_notifications_get_no_response():
    responses = _drive(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ]
    )
    assert len(responses) == 1


def test_tools_list_exposes_every_handler():
    responses = _drive([{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    names = {tool["name"] for tool in responses[0]["result"]["tools"]}
    assert names == set(mcp.HANDLERS)
    for tool in responses[0]["result"]["tools"]:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"


def test_unknown_method_returns_a_json_rpc_error():
    responses = _drive([{"jsonrpc": "2.0", "id": 3, "method": "nope/at/all"}])
    assert responses[0]["error"]["code"] == -32601


def test_malformed_line_returns_a_parse_error():
    stdout = io.StringIO()
    mcp.serve(stdin=io.StringIO("{not json\n"), stdout=stdout)
    assert json.loads(stdout.getvalue())["error"]["code"] == -32700


def test_split_tool_writes_files_and_reports_status(policy_dict, tmp_path: Path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy_dict), encoding="utf-8")
    result = mcp.call_tool(
        "splitproof_split",
        {
            "policy": str(policy_path),
            "out": str(tmp_path / "split.csv"),
            "cert": str(tmp_path / "cert.json"),
        },
    )
    assert not result["isError"]
    payload = result["structuredContent"]
    assert payload["status"] == "satisfied"
    assert (tmp_path / "split.csv").exists()
    assert (tmp_path / "cert.json").exists()
    assert "summary" in payload


def test_audit_tool_flags_a_leaky_split(policy_dict, tmp_path: Path, rows):
    from conftest import write_csv

    for position, row in enumerate(rows):
        row["fold"] = "train" if position % 2 == 0 else "test"
    policy_dict["dataset"]["path"] = str(write_csv(tmp_path / "leaky.csv", rows))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy_dict), encoding="utf-8")
    result = mcp.call_tool("splitproof_audit", {"policy": str(policy_path), "split_column": "fold"})
    assert result["isError"]
    assert result["structuredContent"]["status"] == "violated"


def test_tool_errors_come_back_structured_not_raised(tmp_path: Path):
    result = mcp.call_tool("splitproof_split", {"policy": str(tmp_path / "missing.yaml")})
    assert result["isError"]
    assert result["structuredContent"]["ok"] is False
    assert "error" in result["structuredContent"]


def test_unknown_tool_is_an_error_not_a_crash():
    result = mcp.call_tool("splitproof_teleport", {})
    assert result["isError"]
    assert "UNKNOWN_TOOL" in result["content"][0]["text"]


def test_units_tool_describes_the_blob(policy_dict, tmp_path: Path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy_dict), encoding="utf-8")
    result = mcp.call_tool("splitproof_units", {"policy": str(policy_path)})
    assert not result["isError"]
    assert result["structuredContent"]["units"] > 0
    assert "largest_unit_fraction" in result["structuredContent"]


def test_verify_tool_round_trips(policy_dict, tmp_path: Path):
    from splitproof import api, cert as cert_module

    policy = Policy.from_dict(policy_dict)
    outcome = api.make_split(policy)
    split_path = tmp_path / "split.csv"
    cert_path = tmp_path / "cert.json"
    api.write_split(split_path, outcome.data, outcome.assignment, "accession")
    cert_module.write(cert_path, outcome.certificate)
    result = mcp.call_tool("splitproof_verify", {"cert": str(cert_path), "split": str(split_path)})
    assert not result["isError"]
    assert result["structuredContent"]["ok"] is True
