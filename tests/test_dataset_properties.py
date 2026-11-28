from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from splitproof import dataset as dataset_module
from splitproof.errors import SplitproofError, DataError

SETTINGS = settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

blobs = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=400,
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@SETTINGS
@given(blobs)
def test_arbitrary_csv_bytes_give_a_data_error_not_a_traceback(tmp_path, text):
    path = _write(tmp_path, "fuzz.csv", text)
    try:
        dataset_module.load(path, "id")
    except DataError as error:
        assert error.message
        assert error.code == "DATA_INVALID"
    except SplitproofError as error:
        assert error.message


@SETTINGS
@given(blobs)
def test_arbitrary_fasta_bytes_give_a_data_error_not_a_traceback(tmp_path, text):
    path = _write(tmp_path, "fuzz.fasta", text)
    try:
        dataset_module.load(path, "id")
    except DataError as error:
        assert error.message
    except SplitproofError as error:
        assert error.message


@SETTINGS
@given(blobs)
def test_arbitrary_tsv_bytes_give_a_data_error_not_a_traceback(tmp_path, text):
    path = _write(tmp_path, "fuzz.tsv", text)
    try:
        dataset_module.load(path, "id")
    except SplitproofError as error:
        assert error.message


column_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=1, max_size=8
)


@SETTINGS
@given(st.lists(column_names, min_size=1, max_size=5, unique=True))
def test_a_header_only_file_is_refused_as_empty(tmp_path, header):
    path = _write(tmp_path, "head.csv", ",".join(header) + "\n")
    with pytest.raises(DataError) as error:
        dataset_module.load(path, header[0])
    assert "no rows" in error.value.message


def test_an_unknown_extension_is_named_in_the_error(tmp_path):
    path = _write(tmp_path, "data.xlsx", "whatever")
    with pytest.raises(DataError) as error:
        dataset_module.load(path, "id")
    assert ".csv" in error.value.remedy


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(DataError) as error:
        dataset_module.load(tmp_path / "nope.csv", "id")
    assert "not found" in error.value.message


def test_a_bad_date_names_the_column_and_the_value(tmp_path):
    path = _write(tmp_path, "d.csv", "id,when\na,not-a-date\n")
    data = dataset_module.load(path, "id")
    with pytest.raises(DataError) as error:
        data.dates("when")
    assert error.value.details["column"] == "when"
    assert error.value.details["value"] == "not-a-date"


def test_an_unknown_column_lists_what_is_available(tmp_path):
    path = _write(tmp_path, "c.csv", "id,seq\na,ACDEF\n")
    data = dataset_module.load(path, "id")
    with pytest.raises(DataError) as error:
        data.column("missing")
    assert "seq" in error.value.details["available"]


@pytest.mark.parametrize(
    "body,line,found",
    [
        ("a,kinase,G1,MKV\nb,kinase, putative,G1,MKA\nc,phos,G2,MKL\n", 3, 5),
        ("a,kinase,G1,MKV\nb,kinase,G1\n", 3, 3),
        ("a,kinase,G1,MKV\nb\n", 3, 1),
    ],
)
def test_a_ragged_row_is_refused_rather_than_shifted_into_the_wrong_columns(tmp_path, body, line, found):
    path = _write(tmp_path, "r.csv", "id,name,group,sequence\n" + body)
    with pytest.raises(DataError) as error:
        dataset_module.load(path, "id")
    assert error.value.details["line"] == line
    assert error.value.details["found"] == found
    assert error.value.details["expected"] == 4


def test_a_trailing_empty_value_is_a_field_not_a_ragged_row(tmp_path):
    path = _write(tmp_path, "r.csv", "id,group,sequence\na,G1,\nb,,MKV\n")
    data = dataset_module.load(path, "id")
    assert data.column("sequence") == [None, "MKV"]
    assert data.column("group") == ["G1", None]


def test_only_an_empty_cell_is_missing(tmp_path):
    """The guard for an escape: 'NA', 'None' and 'Null' are real biological values.

    'NA' is North America, neuraminidase, sodium or the Asn-Ala dipeptide; 'None' is
    the unperturbed arm of a screen; 'Null' is a knockout genotype. Treating them as
    missing silently unconstrained whole groups and still certified the split clean.
    """
    path = _write(tmp_path, "n.csv", "id,v\n1,NA\n2,None\n3,NULL\n4,\n5,   \n6,real\n")
    data = dataset_module.load(path, "id")
    assert data.column("v") == ["NA", "None", "NULL", None, None, "real"]


def test_a_group_named_na_is_still_a_group(tmp_path):
    from splitproof import api
    from splitproof.policy import Policy

    path = _write(
        tmp_path,
        "sites.csv",
        "id,seq,site,fold\n"
        "a,ACDEFGHIKLMNPQRST,NA,train\n"
        "b,ACDEFGHIKLMNPQRST,NA,test\n"
        "c,WWWWWWWWWWWWWWWWW,None,train\n"
        "d,WWWWWWWWWWWWWWWWW,None,test\n",
    )
    payload = {
        "splitproof": 1,
        "dataset": {
            "path": str(path),
            "id": "id",
            "similarity": {"sequence": {"column": "seq", "kind": "protein"}},
            "groups": ["site"],
            "dates": {},
        },
        "scheme": {"kind": "holdout", "fractions": {"train": 0.5, "test": 0.5}},
        "constraints": [{"kind": "disjoint", "key": "site"}],
        "seed": 7,
    }
    outcome = api.audit(Policy.from_dict(payload), split_column="fold", skip_neighbours=True)
    assert outcome.reports[0].status == "violated"
    assert outcome.reports[0].measured["values_in_more_than_one_fold"] == 2


def test_a_utf8_bom_does_not_corrupt_the_first_column_name(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_text("id,seq\na,ACDEF\n", encoding="utf-8-sig")
    data = dataset_module.load(path, "id")
    assert data.ids == ["a"]


def test_fasta_without_a_terminating_newline_keeps_the_last_record(tmp_path):
    path = _write(tmp_path, "f.fasta", ">one\nACDEF\n>two\nGHIKL")
    data = dataset_module.load(path, "id")
    assert data.ids == ["one", "two"]
    assert data.column("sequence") == ["ACDEF", "GHIKL"]


def test_fasta_wraps_are_joined(tmp_path):
    path = _write(tmp_path, "w.fasta", ">one\nACDEF\nGHIKL\n\nMNPQR\n")
    data = dataset_module.load(path, "id")
    assert data.column("sequence") == ["ACDEFGHIKLMNPQR"]


def test_nul_bytes_are_refused_the_same_way_on_every_python(tmp_path):
    """Python 3.10's csv module raises on NUL and 3.13's does not.

    splitproof refuses NUL itself so the behaviour does not depend on the interpreter.
    """
    with_nul = tmp_path / "nul.csv"
    with_nul.write_bytes("id,seq\na\x00b,ACDEF\n".encode("utf-8"))
    with pytest.raises(DataError) as error:
        dataset_module.load(with_nul, "id")
    assert "NUL" in error.value.message

    binary = tmp_path / "binary.csv"
    binary.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(DataError):
        dataset_module.load(binary, "id")


@pytest.mark.parametrize("value", ["20200105", "2020-W01-1", "2020-01-05 garbage", "2020-1-5", "2020-001", "2020-01-05T"])
def test_only_an_iso_calendar_date_is_a_date_on_every_python(tmp_path, value):
    path = _write(tmp_path, "d.csv", f"id,when\na,{value}\n")
    with pytest.raises(DataError) as error:
        dataset_module.load(path, "id").dates("when")
    assert error.value.details["value"] == value


@pytest.mark.parametrize(
    "value",
    ["2020-01-05", "2020-01-05T10:00", "2020-01-05 10:00:00", "2020-01-05T10:00:00.123Z", "2020-01-05T10:00:00+01:00"],
)
def test_an_iso_date_or_timestamp_reads_as_its_calendar_day(tmp_path, value):
    import datetime

    path = _write(tmp_path, "d.csv", f"id,when\na,{value}\n")
    assert dataset_module.load(path, "id").dates("when") == [datetime.date(2020, 1, 5)]


def test_a_repeated_column_name_is_refused(tmp_path):
    path = _write(tmp_path, "r.csv", "id,group,group,sequence\nx1,G1,G2,ACGT\nx2,G3,G4,TTTT\n")
    with pytest.raises(DataError) as error:
        dataset_module.load(path, "id")
    assert error.value.details["column"] == "group"
    assert error.value.details["positions"] == [2, 3]


def test_a_fasta_bom_does_not_lose_the_first_record(tmp_path):
    path = tmp_path / "bom.fasta"
    path.write_text("﻿>seq1\nACGTACGTACGT\n>seq2\nTTTTGGGGCCCC\n", encoding="utf-8")
    assert dataset_module.load(path, "id").ids == ["seq1", "seq2"]


def test_fasta_comment_lines_are_not_sequence(tmp_path):
    path = _write(tmp_path, "c.fasta", ";comment\n>s1\n;another\nACGT\n")
    assert dataset_module.load(path, "id").column("sequence") == ["ACGT"]


def test_fasta_sequence_before_any_header_is_refused(tmp_path):
    path = _write(tmp_path, "o.fasta", "ACGT\n>s1\nACGT\n")
    with pytest.raises(DataError) as error:
        dataset_module.load(path, "id")
    assert error.value.details["line"] == 1


def test_parquet_and_csv_produce_the_same_normalised_dataset(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet = tmp_path / "data.parquet"
    pq.write_table(pa.table({"id": ["a", "b", "c"], "group": ["NA", None, "group-2"]}), parquet)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,group\na,NA\nb,\nc,group-2\n", encoding="utf-8")

    from splitproof.dataset import load

    actual = load(parquet, "id")
    expected = load(csv_path, "id")
    assert actual.ids == expected.ids
    assert actual.column("group") == expected.column("group")
