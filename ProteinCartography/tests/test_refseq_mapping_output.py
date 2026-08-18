"""Tests for the RefSeq-to-UniProt mapping side-channel.

``map_refseq_ids`` computes which RefSeq accession became which UniProt entry
and then writes only the UniProt side, discarding the correspondence. That
correspondence is the only way to key a BLAST e-value -- reported against a
RefSeq accession -- to a cohort candidate, which is keyed on UniProt. These
tests cover the optional output that keeps it.

The file lands in the run directory, so the tests that matter are about
*stability*: an unordered dump of an API response is the shape of defect PR #106
had to go back and fix.
"""

import pandas as pd
import pytest
from map_refseq_ids import _write_mapping


def read(path):
    return path.read_text()


def test_the_pairs_are_written_with_a_header(tmp_path):
    frame = pd.DataFrame([{"from": "NP_001", "to": "P12345"}])
    out = tmp_path / "map.tsv"
    _write_mapping(frame, str(out))
    assert read(out) == "from\tto\nNP_001\tP12345\n"


def test_the_bioservices_column_name_is_accepted(tmp_path):
    """`json_normalize` renders the nested response as `to.primaryAccession`."""
    frame = pd.DataFrame([{"from": "NP_001", "to.primaryAccession": "P12345"}])
    out = tmp_path / "map.tsv"
    _write_mapping(frame, str(out))
    assert "NP_001\tP12345" in read(out)


def test_the_output_is_sorted_and_therefore_stable(tmp_path):
    """Two API responses in different orders must produce the same bytes."""
    pairs = [("NP_003", "C3"), ("NP_001", "A1"), ("NP_002", "B2")]
    first, second = tmp_path / "a.tsv", tmp_path / "b.tsv"
    _write_mapping(pd.DataFrame([{"from": a, "to": b} for a, b in pairs]), str(first))
    _write_mapping(pd.DataFrame([{"from": a, "to": b} for a, b in reversed(pairs)]), str(second))
    assert read(first) == read(second)
    assert read(first).splitlines()[1:] == ["NP_001\tA1", "NP_002\tB2", "NP_003\tC3"]


def test_duplicate_pairs_are_collapsed(tmp_path):
    """The two query databases can both return the same pair."""
    frame = pd.DataFrame([{"from": "NP_001", "to": "P1"}] * 3)
    out = tmp_path / "map.tsv"
    _write_mapping(frame, str(out))
    assert read(out).splitlines() == ["from\tto", "NP_001\tP1"]


def test_one_refseq_mapping_to_two_accessions_keeps_both(tmp_path):
    """Real UniProt responses do this; dropping either would lose evidence."""
    frame = pd.DataFrame(
        [{"from": "NP_001009784", "to": "P60713"}, {"from": "NP_001009784", "to": "D7RIF5"}]
    )
    out = tmp_path / "map.tsv"
    _write_mapping(frame, str(out))
    assert read(out).splitlines()[1:] == ["NP_001009784\tD7RIF5", "NP_001009784\tP60713"]


def test_an_empty_frame_still_writes_a_header(tmp_path):
    """A zero-hit query must produce a readable file, not a missing one."""
    out = tmp_path / "map.tsv"
    _write_mapping(pd.DataFrame(), str(out))
    assert read(out) == "from\tto\n"


def test_nothing_is_written_without_a_path(tmp_path):
    """The output is opt-in: the existing rule must be unaffected."""
    _write_mapping(pd.DataFrame([{"from": "a", "to": "b"}]), None)
    assert list(tmp_path.iterdir()) == []


def test_non_string_entries_are_skipped(tmp_path):
    """A NaN target is an unmapped accession, not a protein named 'nan'."""
    frame = pd.DataFrame([{"from": "NP_001", "to": None}, {"from": "NP_002", "to": "P2"}])
    out = tmp_path / "map.tsv"
    _write_mapping(frame, str(out))
    assert read(out).splitlines()[1:] == ["NP_002\tP2"]


@pytest.mark.parametrize("column", ["to", "to.primaryAccession"])
def test_the_mapping_feeds_blast_significance(tmp_path, column):
    """The point of the file: it is what keys a BLAST e-value to a candidate."""
    import constants
    from hit_significance import blast_significance

    mapping_path = tmp_path / "map.tsv"
    _write_mapping(pd.DataFrame([{"from": "NP_001", column: "P12345"}]), str(mapping_path))

    fields = [name for name in constants.BLAST_OUTFMT.split(" ") if name != "6"]
    row = ["-"] * len(fields)
    row[fields.index("sacc")] = "NP_001"
    row[fields.index("evalue")] = "1e-50"
    row[fields.index("bitscore")] = "400"
    blast_path = tmp_path / "q.tsv"
    blast_path.write_text("\t".join(row) + "\n")

    table = blast_significance([str(blast_path)], [str(mapping_path)])
    assert list(table["protid"]) == ["P12345"]
