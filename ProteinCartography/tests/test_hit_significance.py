"""Tests for per-hit significance aggregation.

The point of this table is to let ``cohort.select`` rank candidates, so the
tests that matter are about *reduction*: a hit found by five queries has five
rows of evidence and must come out as one row holding the strongest of them.
Getting the reduction backwards -- keeping the worst e-value rather than the
best -- would rank a protein by the query that barely found it, and would look
perfectly reasonable in the output file.

The other half is keying. BLAST reports RefSeq accessions and the cohort is
keyed on UniProt, so an unmapped BLAST hit must be dropped rather than joined
on the wrong key.
"""

import constants
import pytest
from hit_significance import (
    TmalignOutputError,
    aggregate_significance,
    blast_significance,
    foldseek_significance,
)

FOLDSEEK_NCOL = len(constants.FOLDSEEK_COLUMN_NAMES)
BLAST_FIELDS = [name for name in constants.BLAST_OUTFMT.split(" ") if name != "6"]


def foldseek_row(accession, evalue, bits):
    """One .m8 line, with everything but the fields under test left as filler."""
    row = ["-"] * FOLDSEEK_NCOL
    index = {name: i for i, name in enumerate(constants.FOLDSEEK_COLUMN_NAMES)}
    row[index["query"]] = "job.pdb"
    row[index["target"]] = f"AF-{accession}-F1-model_v4 Some protein"
    row[index["evalue"]] = str(evalue)
    row[index["bits"]] = str(bits)
    return "\t".join(row)


def write_m8(path, rows):
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def blast_row(refseq, evalue, bitscore):
    row = ["-"] * len(BLAST_FIELDS)
    index = {name: i for i, name in enumerate(BLAST_FIELDS)}
    row[index["qseqid"]] = "query"
    row[index["sacc"]] = refseq
    row[index["evalue"]] = str(evalue)
    row[index["bitscore"]] = str(bitscore)
    return "\t".join(row)


def write_blast(path, rows):
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def write_mapping(path, pairs):
    lines = ["from\tto"] + [f"{a}\t{b}" for a, b in pairs]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# reduction across queries
# ---------------------------------------------------------------------------


def test_the_best_evalue_across_queries_wins(tmp_path):
    """A protein is ranked by the best evidence anyone has for it, not the worst."""
    first = write_m8(tmp_path / "a.m8", [foldseek_row("P1", 1e-3, 100)])
    second = write_m8(tmp_path / "b.m8", [foldseek_row("P1", 1e-90, 3000)])
    table = foldseek_significance([first, second]).set_index("protid")
    assert table.loc["P1", "evalue"] == pytest.approx(1e-90)


def test_the_best_bit_score_across_queries_wins(tmp_path):
    """Opposite direction from the e-value, in the same reduction."""
    first = write_m8(tmp_path / "a.m8", [foldseek_row("P1", 1e-3, 100)])
    second = write_m8(tmp_path / "b.m8", [foldseek_row("P1", 1e-90, 3000)])
    table = foldseek_significance([first, second]).set_index("protid")
    assert table.loc["P1", "bits"] == 3000


def test_the_number_of_queries_that_found_a_hit_is_recorded(tmp_path):
    first = write_m8(tmp_path / "a.m8", [foldseek_row("P1", 1e-3, 10)])
    second = write_m8(tmp_path / "b.m8", [foldseek_row("P1", 1e-4, 20), foldseek_row("P2", 1.0, 5)])
    table = foldseek_significance([first, second]).set_index("protid")
    assert table.loc["P1", "n_queries"] == 2
    assert table.loc["P2", "n_queries"] == 1


def test_repeated_alignments_within_one_file_count_as_one_query(tmp_path):
    """Foldseek can report a protein more than once; that is one piece of evidence."""
    path = write_m8(tmp_path / "a.m8", [foldseek_row("P1", 1e-3, 10), foldseek_row("P1", 1e-9, 90)])
    table = foldseek_significance([path]).set_index("protid")
    assert table.loc["P1", "n_queries"] == 1
    assert table.loc["P1", "evalue"] == pytest.approx(1e-9)


def test_the_table_is_ordered_by_accession(tmp_path):
    """Byte-stable output, so it can live in the run directory."""
    path = write_m8(
        tmp_path / "a.m8",
        [foldseek_row("Z9", 1e-3, 10), foldseek_row("A1", 1e-3, 10), foldseek_row("M5", 1e-3, 10)],
    )
    table = foldseek_significance([path])
    assert list(table["protid"]) == ["A1", "M5", "Z9"]


# ---------------------------------------------------------------------------
# what is skipped, and why
# ---------------------------------------------------------------------------


def test_a_non_alphafold_target_is_skipped(tmp_path):
    """The pipeline only downloads AlphaFold models, so only they are candidates."""
    row = foldseek_row("P1", 1e-9, 90).replace("AF-P1-F1-model_v4", "1abc_A")
    path = write_m8(tmp_path / "a.m8", [row, foldseek_row("P2", 1e-9, 90)])
    table = foldseek_significance([path])
    assert list(table["protid"]) == ["P2"]


def test_missing_and_empty_files_are_skipped(tmp_path):
    empty = tmp_path / "empty.m8"
    empty.write_text("")
    real = write_m8(tmp_path / "a.m8", [foldseek_row("P1", 1e-9, 90)])
    table = foldseek_significance([str(tmp_path / "nope.m8"), str(empty), real])
    assert list(table["protid"]) == ["P1"]


def test_no_input_at_all_yields_an_empty_table_rather_than_an_error(tmp_path):
    table = aggregate_significance([], [], [])
    assert list(table.columns) == ["protid", "evalue", "bits", "n_queries", "sources"]
    assert table.empty


# ---------------------------------------------------------------------------
# BLAST needs the RefSeq mapping, and says so
# ---------------------------------------------------------------------------


def test_blast_hits_are_keyed_through_the_refseq_mapping(tmp_path):
    blast = write_blast(tmp_path / "q.tsv", [blast_row("NP_001", 1e-50, 400)])
    mapping = write_mapping(tmp_path / "map.tsv", [("NP_001", "P12345")])
    table = blast_significance([blast], [mapping])
    assert list(table["protid"]) == ["P12345"]


def test_an_unmapped_blast_hit_is_dropped_rather_than_keyed_on_refseq(tmp_path):
    """Joining on the wrong key would rank a protein that is not in the cohort."""
    blast = write_blast(
        tmp_path / "q.tsv", [blast_row("NP_001", 1e-50, 400), blast_row("NP_999", 1e-60, 500)]
    )
    mapping = write_mapping(tmp_path / "map.tsv", [("NP_001", "P12345")])
    table = blast_significance([blast], [mapping])
    assert list(table["protid"]) == ["P12345"]


def test_blast_without_a_mapping_is_skipped_with_a_warning(tmp_path, capsys):
    blast = write_blast(tmp_path / "q.tsv", [blast_row("NP_001", 1e-50, 400)])
    table = blast_significance([blast], [])
    assert table.empty
    assert "--refseq-mapping" in capsys.readouterr().err


def test_two_refseqs_mapping_to_one_accession_are_reduced_together(tmp_path):
    """Many-to-one is normal, and means more evidence for the same protein."""
    blast = write_blast(
        tmp_path / "q.tsv", [blast_row("NP_001", 1e-50, 400), blast_row("NP_002", 1e-90, 900)]
    )
    mapping = write_mapping(tmp_path / "map.tsv", [("NP_001", "P12345"), ("NP_002", "P12345")])
    table = blast_significance([blast], [mapping]).set_index("protid")
    assert len(table) == 1
    assert table.loc["P12345", "evalue"] == pytest.approx(1e-90)
    assert table.loc["P12345", "bits"] == 900


# ---------------------------------------------------------------------------
# tmalign mode reuses the column positions for different quantities
# ---------------------------------------------------------------------------
#
# `foldseek_apiquery.py --mode tmalign` returns the same 21 columns and the
# server puts a TM-score where the e-value goes. Nothing renames, so nothing
# errors; the polarity just inverts and the worst structures rank first.
#
# The numbers below are from a live tmalign query against afdb-swissprot with
# the demo actin structure: 938 hits, e-value column 0.402 to 0.9999, bit scores
# 22 to 99.


def tmalign_row(accession, tmscore):
    """A row shaped like real tmalign output: TM-score in the e-value column."""
    return foldseek_row(accession, tmscore, int(round(tmscore * 100)))


def test_tmalign_output_is_refused_rather_than_ranked_backwards(tmp_path):
    path = write_m8(
        tmp_path / "a.m8",
        [tmalign_row("P1", 0.9999), tmalign_row("P2", 0.71), tmalign_row("P3", 0.402)],
    )
    with pytest.raises(TmalignOutputError, match="tmalign"):
        foldseek_significance([path])


def test_the_refusal_explains_the_polarity_consequence(tmp_path):
    path = write_m8(tmp_path / "a.m8", [tmalign_row("P1", 0.99), tmalign_row("P2", 0.42)])
    with pytest.raises(TmalignOutputError, match="least"):
        foldseek_significance([path])


def test_a_normal_search_is_not_mistaken_for_tmalign(tmp_path):
    """The real 3diaa fixture reaches e-values many orders below any TM-score."""
    path = write_m8(
        tmp_path / "a.m8", [foldseek_row("P1", 1.647e-79, 3274), foldseek_row("P2", 1e-20, 300)]
    )
    assert list(foldseek_significance([path])["protid"]) == ["P1", "P2"]


def test_a_weak_search_with_large_bit_scores_is_not_mistaken_for_tmalign(tmp_path):
    """Both conditions are required, so a weak 3diaa run is not refused.

    E-values near 1 are plausible for a weak search; TM-score-scale bit scores
    alongside them are not.
    """
    path = write_m8(tmp_path / "a.m8", [foldseek_row("P1", 0.8, 900), foldseek_row("P2", 0.5, 750)])
    assert list(foldseek_significance([path])["protid"]) == ["P1", "P2"]


def test_a_search_reaching_small_evalues_is_not_refused_despite_small_bits(tmp_path):
    """The other half of the pair: small bit scores alone are not enough."""
    path = write_m8(
        tmp_path / "a.m8", [foldseek_row("P1", 1e-30, 90), foldseek_row("P2", 1e-8, 40)]
    )
    assert list(foldseek_significance([path])["protid"]) == ["P1", "P2"]


# ---------------------------------------------------------------------------
# combining the two methods
# ---------------------------------------------------------------------------


def test_a_hit_found_by_both_methods_records_both_sources(tmp_path):
    m8 = write_m8(tmp_path / "a.m8", [foldseek_row("P12345", 1e-20, 200)])
    blast = write_blast(tmp_path / "q.tsv", [blast_row("NP_001", 1e-50, 400)])
    mapping = write_mapping(tmp_path / "map.tsv", [("NP_001", "P12345")])
    table = aggregate_significance([m8], [blast], [mapping]).set_index("protid")
    assert table.loc["P12345", "sources"] == "blast+foldseek"
    assert table.loc["P12345", "evalue"] == pytest.approx(1e-50)
    assert table.loc["P12345", "bits"] == 400


def test_a_hit_found_by_one_method_records_only_that_source(tmp_path):
    m8 = write_m8(tmp_path / "a.m8", [foldseek_row("P1", 1e-20, 200)])
    blast = write_blast(tmp_path / "q.tsv", [blast_row("NP_001", 1e-50, 400)])
    mapping = write_mapping(tmp_path / "map.tsv", [("NP_001", "P2")])
    table = aggregate_significance([m8], [blast], [mapping]).set_index("protid")
    assert table.loc["P1", "sources"] == "foldseek"
    assert table.loc["P2", "sources"] == "blast"


# ---------------------------------------------------------------------------
# the table is only useful if cohort selection can consume it
# ---------------------------------------------------------------------------


def test_the_table_drives_significance_selection(tmp_path):
    """End to end: strongest hits survive truncation, weakest do not."""
    from cohort import select

    path = write_m8(
        tmp_path / "a.m8",
        [
            foldseek_row("WEAK", 1.0, 10),
            foldseek_row("STRONG", 1e-99, 3000),
            foldseek_row("MIDDLE", 1e-20, 300),
        ],
    )
    table = foldseek_significance([path])
    scores = dict(zip(table["protid"], table["evalue"]))
    selection = select(
        ["WEAK", "STRONG", "MIDDLE"],
        max_structures=2,
        rule="significance",
        scores=scores,
        measure="evalue",
    )
    assert selection.retained == ("STRONG", "MIDDLE")
    assert selection.discarded == ("WEAK",)
    assert selection.reproducible


def test_a_candidate_with_no_evidence_is_absent_from_the_table(tmp_path):
    """Absent, not worst-scored -- `cohort.select` needs to tell them apart."""
    path = write_m8(tmp_path / "a.m8", [foldseek_row("P1", 1e-9, 90)])
    table = foldseek_significance([path])
    assert "P2" not in set(table["protid"])
