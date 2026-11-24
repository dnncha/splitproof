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
        assert "annotations" in tool
        for prop in tool["inputSchema"]["properties"].values():
            assert prop.get("description")


def test_unknown_method_returns_a_json_rpc_error():
    responses = _drive([{"jsonrpc": "2.0", "id": 3, "method": "nope/at/all"}])
    assert responses[0]["error"]["code"] == -32601


def test_malformed_line_returns_a_parse_error():
    stdout = io.StringIO()
    mcp.serve(stdin=io.StringIO("{not json\n"), stdout=stdout)
    assert json.loads(stdout.getvalue())["error"]["code"] == -32700


def test_split_tool_writes_files_and_reports_status(policy_dict, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
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


def test_audit_tool_flags_a_leaky_split(policy_dict, tmp_path: Path, rows, monkeypatch):
    monkeypatch.chdir(tmp_path)
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
    result = mcp.call_tool(
        "splitproof_verify",
        {
            "cert": str(cert_path),
            "split": str(split_path),
            "data": str(outcome.data.path),
        },
    )
    assert not result["isError"]
    assert result["structuredContent"]["ok"] is True


def test_a_tool_refuses_to_write_outside_the_working_directory(policy_dict, tmp_path: Path, monkeypatch):
    """The guard for a real exposure: these arguments can come from untrusted content.

    An agent reading a report or certificate someone else wrote could be steered into
    passing an absolute path, and the server would have written there.
    """
    monkeypatch.chdir(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy_dict), encoding="utf-8")
    result = mcp.call_tool(
        "splitproof_split",
        {"policy": str(policy_path), "out": "/tmp/splitproof-escape.csv"},
    )
    assert result["isError"]
    assert "outside the working directory" in result["structuredContent"]["error"]["message"]
    assert not Path("/tmp/splitproof-escape.csv").exists()


def test_a_non_object_message_does_not_kill_the_server():
    stdout = io.StringIO()
    mcp.serve(stdin=io.StringIO('["not", "an", "object"]\n'), stdout=stdout)
    assert json.loads(stdout.getvalue())["error"]["code"] == -32600


def test_a_malformed_tool_call_returns_an_error_not_a_crash(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    responses = []
    stdout = io.StringIO()
    mcp.serve(
        stdin=io.StringIO(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "splitproof_split", "arguments": "not-a-mapping"},
                }
            )
            + "\n"
        ),
        stdout=stdout,
    )
    responses = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert responses[0]["result"]["isError"]


from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=8),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=8), children, max_size=3),
    max_leaves=8,
)
_METHODS = st.sampled_from(["initialize", "ping", "tools/list", "tools/call", "nope", "notifications/initialized"])
_ARGUMENTS = st.dictionaries(
    st.sampled_from(["policy", "data", "cert", "split", "out", "tolerance", "require_exact", "markdown", "skip_neighbours"]),
    _JSON,
    max_size=4,
)


@st.composite
def _messages(draw):
    message: Dict[str, Any] = {"jsonrpc": "2.0", "method": draw(_METHODS)}
    if draw(st.booleans()):
        message["id"] = draw(st.integers(0, 10**6) | st.text(min_size=1, max_size=6))
    if message["method"] == "tools/call" and draw(st.booleans()):
        message["params"] = {"name": draw(st.sampled_from(sorted(mcp.HANDLERS))), "arguments": draw(_ARGUMENTS)}
    else:
        message["params"] = draw(_JSON)
    return message


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(batch=st.lists(_messages(), min_size=1, max_size=4))
def test_no_message_kills_the_server_and_only_requests_are_answered(tmp_path, monkeypatch, batch):
    monkeypatch.chdir(tmp_path)
    responses = _drive(batch + [{"jsonrpc": "2.0", "id": "last", "method": "ping"}])
    expected = [message["id"] for message in batch if "id" in message] + ["last"]
    assert [response["id"] for response in responses] == expected
    for response in responses:
        if "result" in response and "structuredContent" in response["result"]:
            error = response["result"]["structuredContent"].get("error") or {}
            assert error.get("code") != "UNEXPECTED", error


ACCESSORS = {"_required", "_number", "_flag", "_optional_text"}


def _raw_argument_reads(source: str):
    import ast

    tree = ast.parse(source)
    hits = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name in ACCESSORS:
            continue
        for node in ast.walk(function):
            target = None
            if isinstance(node, ast.Subscript):
                target = node.value
            elif isinstance(node, ast.Attribute) and node.attr == "get":
                target = node.value
            if isinstance(target, ast.Name) and target.id == "arguments":
                hits.append(f"{function.name}:{node.lineno}")
    return hits


def test_tool_arguments_are_read_only_through_typed_accessors():
    source = (Path(mcp.__file__)).read_text(encoding="utf-8")
    assert _raw_argument_reads(source) == []


def test_the_accessor_rule_catches_a_planted_raw_read():
    planted = "def _do_x(arguments):\n    return Path(arguments.get('data'))\n"
    assert _raw_argument_reads(planted) == ["_do_x:2"]
