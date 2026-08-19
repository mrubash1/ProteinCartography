"""A domain id is a pattern, not a substring.

`DOMAIN_ID_SEP` is `"__d"`, and two call sites tested for it with `in`. Both
failures are silent: a protein whose accession contains those characters loses
its fident and concordance layers from the protein map, and a *path* containing
them makes the mock hit-list writer fire on a protein-path invocation.

Neither is reachable with today's UniProt accessions. Both are one character of
carelessness away from being reachable, and neither would raise.
"""

from __future__ import annotations

import pytest
from domain_utils import DOMAIN_ID_SEP, is_domain_id


def test_a_real_domain_id_is_recognized():
    assert is_domain_id("P60709__d01")
    assert is_domain_id("A0A286Q506__d12")


def test_a_protein_that_merely_contains_the_separator_is_not_a_domain():
    """The whole point. `"__d" in protid` says yes to every one of these."""
    for protid in ("P60709__domain", "X__db", "AB__d", "Q9__disorder", "__d01x"):
        assert DOMAIN_ID_SEP in protid, "fixture must contain the separator to be a test"
        assert not is_domain_id(protid), f"{protid!r} is not a domain instance id"


def test_an_ordinary_accession_is_not_a_domain():
    for protid in ("P60709", "A0A286Q506", "Q6QAQ1"):
        assert not is_domain_id(protid)


def test_the_domain_index_must_be_numeric_and_terminal():
    assert not is_domain_id("P60709__dab")
    assert not is_domain_id("P60709__d01x")
    assert not is_domain_id("P60709__d01.blast_hits")


# --------------------------------------------------------------------------
# the mock writer keys on the filename, not on any parent directory
# --------------------------------------------------------------------------


def test_the_mock_fires_for_a_real_domain_hit_file(tmp_path):
    mock_domain_hits = pytest.importorskip("mock_domain_hits")
    out = tmp_path / "P60709__d01.blast_hits.refseq.txt"
    assert mock_domain_hits.maybe_write_per_domain_hits(str(out)) is True
    assert out.read_text().strip(), "it claimed to write a hit list and wrote nothing"


def test_a_directory_containing_the_token_does_not_trigger_the_mock(tmp_path):
    """The protein path must not silently receive a domain hit list because
    somebody's working directory is named after a domain."""
    mock_domain_hits = pytest.importorskip("mock_domain_hits")
    workdir = tmp_path / "run__d01.attempt2"
    workdir.mkdir()
    out = workdir / "P60709.blast_hits.refseq.txt"
    assert mock_domain_hits.maybe_write_per_domain_hits(str(out)) is False
    assert not out.exists(), "a protein-path output was written by the domain mock"
