from __future__ import annotations

import json
from pathlib import Path

import pytest

from splitproof import cli

from conftest import make_rows, write_csv


def _policy(tmp_path: Path, data: Path, *, exact: bool) -> Path:
    path = tmp_path / "p.yaml"
    backend = "" if exact else ", backend: minhash"
    path.write_text(
        "splitproof: 1\n"
        f"dataset: {{path: {data.name}, id: accession, similarity: {{s: {{column: sequence, kind: protein}}}}}}\n"
        "scheme: {kind: holdout, fractions: {train: 0.7, test: 0.3}}\n"
        f"constraints: [{{kind: similarity, key: s, metric: kmer_jaccard, threshold: 0.4{backend}}}]\n",
        encoding="utf-8",
    )
    return path


def _run(capsys, argv):
    code = cli.main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_json_ok_is_true_exactly_when_the_exit_code_is_zero(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = write_csv(tmp_path / "d.csv", make_rows(families=6, members=3, seed=5))
    policy = _policy(tmp_path, data, exact=False)
    argv = ["split", str(policy), "--out", "s.csv", "--cert", "c.json", "--no-neighbours", "--json"]
    codes = []
    for extra in ([], ["--require-exact"]):
        code, payload = _run(capsys, argv + extra)
        assert payload["status"] == "satisfied_unproven"
        assert payload["ok"] is (code == 0), (extra, code)
        codes.append(code)
    assert codes == [cli.EXIT_OK, cli.EXIT_VIOLATED]
    code, payload = _run(capsys, ["verify", "c.json", "--split", "s.csv", "--no-neighbours", "--json", "--require-exact"])
    assert payload["ok"] is (code == 0)
    assert payload["verified"] is True


@pytest.mark.parametrize(
    "argv,setup",
    [
        (["init", "empty.csv"], lambda d: (d / "empty.csv").write_text("")),
        (["init", "absent.csv"], lambda d: None),
        (["verify", "bad.json", "--json"], lambda d: (d / "bad.json").write_text(json.dumps({"splitproof": {"schema": 1}, "assignment": []}))),
        (["explain", "bad.json", "--json"], lambda d: (d / "bad.json").write_text(json.dumps({"splitproof": {"schema": 1}, "assignment": []}))),
    ],
)
def test_bad_input_is_a_structured_error_with_the_error_exit_code(tmp_path, capsys, monkeypatch, argv, setup):
    monkeypatch.chdir(tmp_path)
    setup(tmp_path)
    code = cli.main(argv + (["--json"] if "--json" not in argv and argv[0] != "init" else []))
    captured = capsys.readouterr()
    assert code == cli.EXIT_ERROR, captured
    assert "Traceback" not in captured.err
