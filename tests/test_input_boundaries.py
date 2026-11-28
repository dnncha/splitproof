from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from splitproof import api, cert, mcp
from splitproof.dataset import load
from splitproof.errors import CertificateMismatch, DataError, InfeasibleSplit, PolicyError
from splitproof.policy import Policy


@pytest.mark.parametrize("format", ["yaml", "json"])
def test_policy_refuses_repeated_mapping_keys(tmp_path: Path, format: str) -> None:
    document = {
        "json": '{"dataset":{"id":"discarded","id":"id","groups":["g"]},'
                '"constraints":[{"kind":"disjoint","key":"g"}]}',
        "yaml": 'dataset: {id: discarded, id: id, groups: [g]}\n'
                'constraints: [{kind: disjoint, key: g}]\n',
    }
    path = tmp_path / f"policy.{format}"
    path.write_text(document[format], encoding="utf-8")
    with pytest.raises(PolicyError, match="duplicate.*id"):
        Policy.load(path)


def test_yaml_merge_allows_an_explicit_override(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        'defaults: &defaults {id: base, groups: [g]}\n'
        'dataset: {<<: *defaults, id: id}\n'
        'constraints: [{kind: disjoint, key: g}]\n',
        encoding="utf-8",
    )
    assert Policy.load(path).dataset.id_column == "id"


def test_certificate_refuses_a_repeated_field(tmp_path: Path) -> None:
    path = tmp_path / "certificate.json"
    path.write_text(
        '{"splitproof":{"schema":0,"schema":1},"assignment":{},"constraints":[],'
        '"dataset":{},"diagnostics":{},"policy":{},"status":"satisfied","units":{}}',
        encoding="utf-8",
    )
    with pytest.raises(CertificateMismatch, match="duplicate.*schema"):
        cert.read(path)


def test_mcp_rejects_duplicate_members_and_keeps_serving() -> None:
    source = io.StringIO(
        '{"jsonrpc":"2.0","id":1,"method":"ping","method":"ping"}\n'
        '{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
    )
    output = io.StringIO()
    assert mcp.serve(stdin=source, stdout=output) == 0
    rejected, response = map(json.loads, output.getvalue().splitlines())
    assert rejected["error"]["code"] == -32700
    assert response["id"] == 2 and response["result"] == {}


@pytest.mark.parametrize("format", ["csv", "tsv"])
def test_dataset_rejects_an_unterminated_quoted_field(tmp_path: Path, format: str) -> None:
    delimiter = "," if format == "csv" else "\t"
    path = tmp_path / f"data.{format}"
    path.write_text(f'id{delimiter}group\na{delimiter}"g', encoding="utf-8")
    with pytest.raises(DataError, match="delimited text"):
        load(path, "id")


def test_split_file_rejects_an_unterminated_quoted_field(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("id\na\nb\n", encoding="utf-8")
    split = tmp_path / "split.csv"
    split.write_text('id,fold\na,train\nb,"test', encoding="utf-8")
    with pytest.raises(DataError, match="delimited text"):
        api.load_assignment(load(dataset, "id"), split_path=split)


def test_fasta_id_cannot_replace_the_sequence_column(tmp_path: Path) -> None:
    path = tmp_path / "data.fasta"
    path.write_text(">row-a\nAAAA\n>row-b\nCCCC\n", encoding="utf-8")
    with pytest.raises(DataError, match="column 'sequence' twice"):
        load(path, "sequence")


def test_leave_unit_out_refuses_a_single_independent_unit(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("id,group\na,one\nb,one\n", encoding="utf-8")
    policy = Policy.from_dict({
        "dataset": {"id": "id", "groups": ["group"]},
        "scheme": {"kind": "leave-unit-out"},
        "constraints": [{"kind": "disjoint", "key": "group"}],
    })
    with pytest.raises(InfeasibleSplit, match="at least two independent units"):
        api.make_split(policy, data_path=path)


@pytest.mark.parametrize("fold_name", ["fold", "split", "set"])
def test_headerless_split_keeps_the_first_known_entity(tmp_path: Path, fold_name: str) -> None:
    path = tmp_path / "data.csv"
    path.write_text("id\na\nb\n", encoding="utf-8")
    split = tmp_path / "split.csv"
    split.write_text(f"a,{fold_name}\nb,train\n", encoding="utf-8")
    data = load(path, "id")
    assignment = api.load_assignment(data, split_path=split)
    assert assignment.fold_of == {0: fold_name, 1: "train"}


def test_written_header_can_match_a_real_entity_id(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("id,fold\nid,test\nb,train\n", encoding="utf-8")
    data = load(path, "id")
    expected = api.load_assignment(data, split_column="fold")
    output = tmp_path / "written.csv"
    api.write_split(output, data, expected, "id")
    assert api.load_assignment(data, split_path=output).fold_of == expected.fold_of


@pytest.mark.parametrize("extra", ["extra", ""])
def test_split_rejects_an_extra_column(tmp_path: Path, extra: str) -> None:
    path = tmp_path / "data.csv"
    path.write_text("id\na\nb\n", encoding="utf-8")
    split = tmp_path / "split.csv"
    split.write_text(f"id,fold\na,test,{extra}\nb,train\n", encoding="utf-8")
    with pytest.raises(DataError, match="needs two columns"):
        api.load_assignment(load(path, "id"), split_path=split)


@pytest.mark.parametrize(
    "key,aliases",
    [("dates", {"day": "original", " day": "replacement"}),
     ("similarity", {"match": {"column": "a"}, " match": {"column": "b"}})],
)
def test_policy_refuses_colliding_normalised_aliases(key, aliases) -> None:
    payload = {
        "dataset": {"id": "id", "groups": ["g"], key: aliases},
        "constraints": [{"kind": "disjoint", "key": "g"}],
    }
    with pytest.raises(PolicyError, match="duplicate.*after normalisation"):
        Policy.from_dict(payload)


def test_policy_refuses_colliding_normalised_backend_params() -> None:
    payload = {
        "dataset": {"id": "id", "similarity": {"s": {"column": "s"}}},
        "constraints": [{
            "kind": "similarity", "key": "s", "metric": "kmer_jaccard", "threshold": 0.5,
            "params": {"k": 2, " k": 3},
        }],
    }
    with pytest.raises(PolicyError, match="duplicate.*after normalisation"):
        Policy.from_dict(payload)


@pytest.mark.parametrize("suffix", ["T24:00", "T12:60", "T12:00:61", "T12:00+24:00", "T12:00-0060"])
def test_policy_refuses_an_impossible_clock_or_offset(suffix: str) -> None:
    payload = {
        "dataset": {"id": "id", "dates": {"day": "day"}},
        "constraints": [{"kind": "temporal", "key": "day", "cutoff": f"2024-06-01{suffix}"}],
    }
    with pytest.raises(PolicyError, match="not an ISO date"):
        Policy.from_dict(payload)


def test_summary_attributes_unusable_data_to_individual_constraints(tmp_path: Path) -> None:
    from splitproof.report import summary

    path = tmp_path / "data.csv"
    path.write_text("id,left,right,fold\na,g,,train\nb,,g,test\n", encoding="utf-8")
    policy = Policy.from_dict({
        "dataset": {"id": "id", "groups": ["left", "right"]},
        "constraints": [{"kind": "disjoint", "key": key} for key in ("left", "right")],
    })
    outcome = api.audit(policy, data_path=path, split_column="fold")
    assert "For at least one declared constraint, 1 entity had no usable value" in summary(outcome.certificate)


def test_yaml_reuses_an_anchor_with_nested_merges(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        'one: &one {id: first}\n'
        'two: &two {id: second}\n'
        'combined: &combined {<<: [*one, *two], id: chosen, groups: [g]}\n'
        'dataset: {<<: *combined}\n'
        'constraints: [{kind: disjoint, key: g}]\n',
        encoding="utf-8",
    )
    assert Policy.load(path).dataset.id_column == "chosen"
