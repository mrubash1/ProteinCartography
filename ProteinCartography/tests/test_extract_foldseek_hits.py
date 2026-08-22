"""Tests for `extract_foldseek_hits`, which had none.

This module decides WHICH hits reach the map: it concatenates one frame per
Foldseek database, drops non-AlphaFold targets, filters on e-value, and then
truncates. The truncation is the part worth pinning, and it is pinned here as a
DEFECT rather than as behaviour worth keeping -- see the docstring on
`test_truncation_keeps_the_first_n_in_database_order`.

`pd.read_csv` is real here rather than mocked. The module's whole job is what
pandas does to a `.m8`, and a mocked reader would test the mock.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")

import constants  # noqa: E402
from extract_foldseek_hits import extract_foldseekhits  # noqa: E402


def m8_row(accession: str, evalue: float, bits: int = 500) -> str:
    """One `.m8` line, with everything but the fields under test left as filler."""
    row = ["-"] * len(constants.FOLDSEEK_COLUMN_NAMES)
    index = {name: i for i, name in enumerate(constants.FOLDSEEK_COLUMN_NAMES)}
    row[index["query"]] = "job.pdb"
    row[index["target"]] = f"AF-{accession}-F1-model_v4 Some protein"
    row[index["evalue"]] = str(evalue)
    row[index["bits"]] = str(bits)
    return "\t".join(row)


def write_m8(path, rows) -> str:
    path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return str(path)


def hits_in(path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# the empty-file guard (FOLLOWUPS #24)
# ---------------------------------------------------------------------------


def test_a_zero_byte_m8_is_skipped_and_produces_an_empty_output(tmp_path, capsys):
    """The guard's own case, now that it runs before the read rather than after.

    Inert under the pinned pandas 2.0.1 either way -- reading an empty file with
    `names=` supplied returns a 0-row frame instead of raising -- so this test
    passes before and after the move. It is here to say what the behaviour IS,
    because #24 asserted `EmptyDataError` and that is not what happens.
    """
    empty = write_m8(tmp_path / "alis_afdb50.m8", [])
    out = tmp_path / "hits.txt"
    extract_foldseekhits([empty], str(out))
    assert hits_in(out) == []
    assert "No matching foldseek hits found" in capsys.readouterr().out


def test_an_empty_file_beside_a_real_one_does_not_lose_the_real_one(tmp_path):
    """The case the guard exists for: skipping one input must not skip the rest,
    including when the empty one comes FIRST and the accumulator is still
    unset."""
    empty = write_m8(tmp_path / "alis_afdb50.m8", [])
    real = write_m8(tmp_path / "alis_afdb-swissprot.m8", [m8_row("P00001", 1e-40)])
    out = tmp_path / "hits.txt"
    extract_foldseekhits([empty, real], str(out))
    assert hits_in(out) == ["P00001"]


def test_a_file_whose_rows_are_all_filtered_out_yields_an_empty_output(tmp_path, capsys):
    """Distinct from the empty-file case: the file has rows, and none survive."""
    weak = write_m8(tmp_path / "alis_afdb50.m8", [m8_row("P00001", 10.0)])
    out = tmp_path / "hits.txt"
    extract_foldseekhits([weak], str(out), evalue=1e-3)
    assert hits_in(out) == []
    assert "No matching foldseek hits found" in capsys.readouterr().out


def test_a_non_alphafold_target_is_dropped(tmp_path):
    """Only AlphaFold models are downloadable, so only they are candidates."""
    row = m8_row("P00001", 1e-40).replace("AF-P00001-F1-model_v4", "1abc_A")
    path = write_m8(tmp_path / "alis_afdb50.m8", [row, m8_row("P00002", 1e-40)])
    out = tmp_path / "hits.txt"
    extract_foldseekhits([path], str(out))
    assert hits_in(out) == ["P00002"]


def test_a_repeated_accession_is_written_once(tmp_path):
    """`unique()` across the concatenated frames, so one protein is one hit."""
    path = write_m8(
        tmp_path / "alis_afdb50.m8",
        [m8_row("P00001", 1e-40), m8_row("P00001", 1e-90), m8_row("P00002", 1e-40)],
    )
    out = tmp_path / "hits.txt"
    extract_foldseekhits([path], str(out))
    assert hits_in(out) == ["P00001", "P00002"]


# ---------------------------------------------------------------------------
# what truncation does today (FOLLOWUPS #23)
# ---------------------------------------------------------------------------


def test_truncation_keeps_the_first_n_in_database_order(tmp_path):
    """THIS PINS A DEFECT. It is not a description of behaviour worth keeping.

    `max_foldseek_hits` keeps the first N accessions in the order the frames
    were concatenated, which is the order the Snakefile expands
    `db=FOLDSEEK_DATABASES` -- so it is the order of `foldseek_databases` in
    `config.yml`, today `afdb50`, `afdb-swissprot`, `afdb-proteome`. It is NOT
    the N most significant hits, and the e-values needed to rank them are in the
    frame the whole time.

    The fixture makes the two rules disagree on purpose: the weakest hit is
    first in database order and the strongest is last, so a run that kept the
    top 2 by e-value would return `{P00003, P00004}` and this returns
    `{P00001, P00002}`.

    **PC-014 phase 4 is what changes this**, and it cannot land under the
    parity requirement without a decision, because it changes which proteins
    reach the map. Until then the rule is at least written down and a change to
    it cannot be silent. A test that blesses a defect without saying so is worse
    than no test.
    """
    first = write_m8(
        tmp_path / "alis_afdb50.m8",
        [m8_row("P00001", 1e-5), m8_row("P00002", 1e-6)],
    )
    second = write_m8(
        tmp_path / "alis_afdb-swissprot.m8",
        [m8_row("P00003", 1e-90), m8_row("P00004", 1e-80)],
    )
    out = tmp_path / "hits.txt"
    extract_foldseekhits([first, second], str(out), max_num_hits=2)

    assert hits_in(out) == ["P00001", "P00002"], "database order, not significance"
    # State the counterfactual, so the defect is legible from the test alone.
    by_significance = ["P00003", "P00004"]
    assert hits_in(out) != by_significance


def test_truncation_is_not_applied_when_no_maximum_is_given(tmp_path):
    """The other half: the default keeps everything that survived the filters."""
    path = write_m8(
        tmp_path / "alis_afdb50.m8",
        [m8_row(f"P{i:05d}", 1e-40) for i in range(5)],
    )
    out = tmp_path / "hits.txt"
    extract_foldseekhits([path], str(out))
    assert len(hits_in(out)) == 5
