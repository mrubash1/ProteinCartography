"""Tests for the `enrich_clusters` entry point.

Driven through `main()` with an argv, for the reason `test_coregister.py` is:
the entry point is where the config meets the statistic, and a defect that
lives in the caller is invisible to every test of the callee. `test_enrichment`
already proves the statistic finds the planted signal when it is handed the
right counts; this file proves the counts it gets handed are the right ones.

The cohort is `annotated_cohort`, written to disk in the two-table shape the
pipeline produces -- `leiden_features.tsv` beside `uniprot_features.tsv` --
because the join between them is one of the things that can go wrong.
"""

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from annotated_cohort import annotated_cohort
from config_schema import ConfigError, EnrichmentConfig
from enrich_clusters import (
    TABLE_COLUMNS,
    EnrichmentError,
    join_on_protid,
    main,
    numeric_column,
    read_keyed_table,
    to_frame,
)


@pytest.fixture(scope="module")
def cohort():
    return annotated_cohort()


def write_input_tables(root, cohort):
    """The two tables the pipeline writes, in the two places it writes them.

    The annotation table keeps the cluster column, because the pipeline's does:
    `aggregated_features.tsv` is built by joining `leiden_features.tsv` into
    everything else. Writing it without was what let a crash reach the demo run.
    """
    frame = cohort.frame
    clusters = frame[["protid", cohort.cluster_column]]
    annotations = frame

    output = root / "output"
    clusters_path = output / "foldseek_clustering_results" / "leiden_features.tsv"
    clusters_path.parent.mkdir(parents=True)
    clusters.to_csv(clusters_path, sep="\t", index=False)

    annotations_path = output / "protein_features" / "uniprot_features.tsv"
    annotations_path.parent.mkdir(parents=True)
    annotations.to_csv(annotations_path, sep="\t", index=False)

    return root, clusters_path, annotations_path


@pytest.fixture
def run_dir(tmp_path, cohort):
    """A private input tree, for the tests that write into it.

    Function-scoped on purpose. Every test that writes a second output
    directory, a second config file, or a second run's tables needs a tree
    nobody else is reading; `default_run` below covers only the read-only case.
    Building it costs ~1.8 ms, so there is nothing to win by sharing it.
    """
    return write_input_tables(tmp_path, cohort)


DEFAULT_ENRICHMENT = {
    "cluster_column": "LeidenCluster",
    "categorical": ["Lineage", "Pfam", "Organism"],
    "continuous": ["Length", "pdb_confidence", "Annotation"],
    "min_term_count": 3,
    "fdr": 0.05,
}


def write_config(tmp_path, enrichment=None, name="config.json"):
    config = {"enrichment": DEFAULT_ENRICHMENT if enrichment is None else enrichment}
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return path


def enrich_argv(config_path, output_dir, clusters, annotations):
    return [
        "enrich_clusters.py",
        "--configfile",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--clusters",
        str(clusters),
        "--annotations",
        str(annotations),
    ]


def run(monkeypatch, config_path, output_dir, clusters, annotations):
    monkeypatch.setattr(sys, "argv", enrich_argv(config_path, output_dir, clusters, annotations))
    return main()


def read_table(output_dir):
    return pd.read_csv(
        output_dir / "enrichment" / "cluster_enrichment.tsv", sep="\t", keep_default_na=False
    )


def numeric_table(output_dir):
    """The same table with the numeric columns as numbers, for arithmetic."""
    frame = read_table(output_dir)
    for column in ("effect", "p_value", "q_value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


class DefaultRun(NamedTuple):
    status: int
    output: Path
    clusters: Path
    annotations: Path
    stderr: str


@pytest.fixture(scope="module")
def default_run(tmp_path_factory, cohort):
    """One `main()` over `DEFAULT_ENRICHMENT`, for the tests that only read it.

    Fourteen tests here ran that identical command -- same cohort, same config,
    same argv -- and then asserted on different parts of the one table it wrote.
    At ~50 ms a call that was ~700 ms of the file's ~1.15 s; running it once
    takes the file to ~0.5 s.

    Sharing is safe because what escapes is paths and a string: every consumer
    re-parses `cluster_enrichment.tsv` or `manifest.json` from disk into its own
    frame, so no test can hand another a mutated object. What is *not* safe is
    sharing the directory with anything that writes, which is why `run_dir`
    stays function-scoped and why the tests that run a second, differently
    configured `main()` keep using it.

    `test_two_runs_over_the_same_input_produce_the_same_bytes` must never take
    this fixture. Its two `main()` calls are the assertion itself -- collapsing
    them to one would leave it comparing a file against itself and passing
    unconditionally.

    stderr is captured with `redirect_stderr` rather than `capsys`, which is
    function-scoped and so cannot see a module-scoped fixture's output.
    `enrich_clusters` resolves `sys.stderr` at each `print` call, so the two
    capture the same bytes.
    """
    root = tmp_path_factory.mktemp("enrich_default")
    _, clusters, annotations = write_input_tables(root, cohort)
    output = root / "output"
    config = write_config(root)

    captured = io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", enrich_argv(config, output, clusters, annotations))
        with contextlib.redirect_stderr(captured):
            status = main()

    return DefaultRun(
        status=status,
        output=output,
        clusters=clusters,
        annotations=annotations,
        stderr=captured.getvalue(),
    )


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------


def test_it_writes_a_tidy_table_and_a_manifest(default_run):
    output = default_run.output
    assert default_run.status == 0

    table = read_table(output)
    assert list(table.columns) == list(TABLE_COLUMNS)
    assert (table["clustering"] == "LeidenCluster").all()
    assert set(table["kind"]) == {"categorical", "continuous"}
    assert (output / "enrichment" / "manifest.json").exists()


def test_every_planted_signal_survives_the_entry_point(default_run, cohort):
    """The claim that matters. `test_enrichment` shows the statistic finds
    these when handed the counts; this shows the counts arrive intact through
    the config, the join, the encoding detection and the correction."""
    table = numeric_table(default_run.output)

    significant = table[table["significant"].astype(str) == "True"]
    found = {
        (row["annotation"], row["term"] or None, row["cluster"])
        for _, row in significant.iterrows()
    }
    assert cohort.signals() <= found, cohort.signals() - found


def test_nothing_is_found_in_the_column_generated_null(default_run):
    table = numeric_table(default_run.output)
    organism = table[table["annotation"] == "Organism"]
    assert len(organism) > 0
    assert not (organism["significant"].astype(str) == "True").any()


def test_the_planted_effects_point_the_way_they_were_planted(default_run, cohort):
    table = numeric_table(default_run.output)

    for shift in cohort.planted_shifts:
        row = table[
            (table["annotation"] == shift.column) & (table["cluster"] == shift.cluster)
        ].iloc[0]
        assert np.sign(row["effect"]) == np.sign(shift.effect_size)
        assert row["effect_kind"] == "rank_biserial"

    for planted in cohort.planted_terms:
        row = table[
            (table["annotation"] == planted.column)
            & (table["cluster"] == planted.cluster)
            & (table["term"] == planted.term)
        ].iloc[0]
        assert row["effect"] > 2.0
        assert row["effect_kind"] == "fold_enrichment"


def test_the_counts_on_a_row_reconstruct_its_own_test(default_run, cohort):
    """A row has to be checkable without rerunning anything. If `n_term_cluster`
    and `n_cluster` do not agree with the table the numbers came from, the
    p-value is unauditable however correct it happens to be."""
    annotations = default_run.annotations
    table = read_table(default_run.output)

    planted = cohort.planted_terms[1]  # Pfam / PF00022
    row = table[
        (table["annotation"] == "Pfam")
        & (table["cluster"] == planted.cluster)
        & (table["term"] == planted.term)
    ].iloc[0]

    # Reconstructed from the file the entry point read, not from the in-memory
    # cohort: a protein carrying no families is written as an empty field and
    # read back as NaN, so the two differ. See the round-trip test below.
    on_disk = pd.read_csv(annotations, sep="\t")
    annotated = on_disk[on_disk["Pfam"].notna()]
    members = annotated[annotated[cohort.cluster_column] == planted.cluster]
    carries = annotated["Pfam"].str.contains(planted.term, regex=False)
    assert int(row["n_universe"]) == len(annotated)
    assert int(row["n_cluster"]) == len(members)
    assert int(row["n_term_universe"]) == int(carries.sum())
    assert int(row["n_term_cluster"]) == int(
        members["Pfam"].str.contains(planted.term, regex=False).sum()
    )


def test_the_universe_shrinks_to_the_annotated_background_and_says_so(default_run, cohort):
    """The consequence of the file format, not of the statistic.

    A protein carrying no Pfam families is written as an empty field and read
    back as NaN, so after a round-trip through TSV "carries no families" and
    "was never annotated" are the same bytes. Both leave the universe, which
    takes Pfam from 400 proteins to 232 and raises every fold enrichment
    accordingly. That is the conventional choice -- an enrichment background is
    the annotated background -- but it is not free, so it has to be visible
    rather than inferred from a number nobody compares against anything.
    """
    assert (cohort.frame["Pfam"] == "").sum() == 168
    assert pd.read_csv(default_run.annotations, sep="\t")["Pfam"].isna().sum() == 168

    manifest = json.loads((default_run.output / "enrichment" / "manifest.json").read_text())
    assert manifest["extra"]["categorical"]["Pfam"]["universe"] == 232
    assert manifest["extra"]["n_proteins"] == 400
    assert "Pfam: tested against 232 of 400 proteins" in default_run.stderr


# ---------------------------------------------------------------------------
# the parts that must be reported rather than skipped
# ---------------------------------------------------------------------------


def test_an_untested_hypothesis_appears_with_its_reason(default_run):
    """The cluster with no pLDDT and the constant column. Absent rows would
    read as "nothing to say here"; these say why there is nothing."""
    table = read_table(default_run.output)

    confidence = table[(table["annotation"] == "pdb_confidence") & (table["cluster"] == "LC7")]
    assert len(confidence) == 1
    assert "no measurements in the cluster" in confidence.iloc[0]["note"]

    constant = table[table["annotation"] == "Annotation"]
    assert len(constant) == 8
    assert all("no ordering to test" in note for note in constant["note"])


def test_an_untested_row_is_never_significant_and_has_no_q(default_run):
    table = numeric_table(default_run.output)
    untested = table[table["note"] != ""]
    assert len(untested) > 0
    assert untested["q_value"].isna().all()
    assert not (untested["significant"].astype(str) == "True").any()


def test_a_requested_column_that_is_absent_is_named_not_skipped(monkeypatch, run_dir, capsys):
    """FOLLOWUPS #35: `ec` and `cc_subcellular_location` are never fetched, so
    two of the four categories PLAN names have no column. "No enrichment for
    localization" and "localization was never in the table" are different
    facts."""
    tmp_path, clusters, annotations = run_dir
    output = tmp_path / "output"
    config = write_config(
        tmp_path,
        {
            **DEFAULT_ENRICHMENT,
            "categorical": ["Lineage", "EC", "Subcellular location"],
            "continuous": [],
        },
    )
    assert run(monkeypatch, config, output, clusters, annotations) == 0

    assert "EC" in capsys.readouterr().err
    manifest = json.loads((output / "enrichment" / "manifest.json").read_text())
    assert manifest["extra"]["columns_absent"] == ["EC", "Subcellular location"]
    assert read_table(output)["annotation"].unique().tolist() == ["Lineage"]


def test_terms_below_the_minimum_count_are_dropped_and_counted(default_run):
    """The singleton Pfam term. Testing it would cost every other term in the
    family a larger correction and it cannot reach significance anyway."""
    output = default_run.output
    assert "PF99999" not in set(read_table(output)["term"])
    manifest = json.loads((output / "enrichment" / "manifest.json").read_text())
    assert "PF99999" in manifest["extra"]["categorical"]["Pfam"]["dropped"]


def test_lowering_the_minimum_count_brings_the_singleton_back(monkeypatch, run_dir):
    """Guards the guard: without this, a filter that dropped everything would
    pass the test above."""
    tmp_path, clusters, annotations = run_dir
    output = tmp_path / "output"
    config = write_config(tmp_path, {**DEFAULT_ENRICHMENT, "min_term_count": 1})
    run(monkeypatch, config, output, clusters, annotations)
    table = numeric_table(output)
    singleton = table[table["term"] == "PF99999"]
    assert len(singleton) == 8
    # Present, and still not significant. That is the point of the filter: it
    # saves the correction, it does not change the answer.
    assert not (singleton["significant"].astype(str) == "True").any()


def test_the_report_reaches_stderr(default_run):
    err = default_run.stderr
    assert "hypotheses over 400 proteins in 8 clusters" in err
    assert "significant at q<=0.05" in err
    assert "untested" in err


def test_the_manifest_records_what_the_table_rests_on(default_run):
    manifest = json.loads((default_run.output / "enrichment" / "manifest.json").read_text())

    assert manifest["params"]["min_term_count"] == 3
    assert set(manifest["inputs"]) == {"clusters", "annotations"}
    # A changed annotation table must change the cache key rather than leave a
    # stale enrichment looking fresh.
    assert all(digest.startswith("sha256:") for digest in manifest["inputs"].values())
    assert manifest["extra"]["categorical"]["Lineage"]["encoding"] == "list_repr"
    assert manifest["extra"]["categorical"]["Pfam"]["encoding"] == "delimited"
    assert manifest["extra"]["n_proteins"] == 400


# ---------------------------------------------------------------------------
# the correction
# ---------------------------------------------------------------------------


def test_the_correction_family_is_one_annotation_column(monkeypatch, tmp_path, default_run):
    """Pooling would let a large vocabulary set the correction for a small one,
    so a finding's q would depend on which other columns were configured."""
    together = numeric_table(default_run.output)

    # The Pfam-only run has to read the same two input tables as the shared run,
    # or the comparison would be between two different cohorts. It writes into
    # its own `tmp_path`, so nothing lands in the shared tree.
    alone = tmp_path / "alone"
    config = write_config(
        tmp_path, {**DEFAULT_ENRICHMENT, "categorical": ["Pfam"], "continuous": []}, "alone.json"
    )
    run(monkeypatch, config, alone, default_run.clusters, default_run.annotations)
    separate = numeric_table(alone)

    def pfam_q(frame):
        rows = frame[frame["annotation"] == "Pfam"].sort_values(["cluster", "term"])
        return rows["q_value"].to_numpy()

    np.testing.assert_allclose(pfam_q(together), pfam_q(separate), rtol=1e-12)


def test_a_q_value_is_never_below_its_p_value(default_run):
    table = numeric_table(default_run.output)
    tested = table[table["note"] == ""]
    assert (tested["q_value"] >= tested["p_value"] - 1e-12).all()


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_two_runs_over_the_same_input_produce_the_same_bytes(monkeypatch, run_dir):
    """Do NOT route this through `default_run`.

    The two `main()` calls below are the assertion, not setup for it. Replacing
    either with the shared run would leave this comparing one file against
    itself, which passes whatever `main()` does.
    """
    tmp_path, clusters, annotations = run_dir
    config = write_config(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    run(monkeypatch, config, first, clusters, annotations)
    run(monkeypatch, config, second, clusters, annotations)
    assert (first / "enrichment" / "cluster_enrichment.tsv").read_bytes() == (
        second / "enrichment" / "cluster_enrichment.tsv"
    ).read_bytes()


def test_the_table_is_sorted_by_q_with_untested_rows_last(default_run):
    table = numeric_table(default_run.output)
    q_values = table["q_value"].to_numpy()
    tested = ~np.isnan(q_values)
    assert np.all(np.diff(q_values[tested]) >= -1e-12)
    # Every untested row sits below every tested one.
    assert np.all(np.flatnonzero(~tested) > np.max(np.flatnonzero(tested)))


# ---------------------------------------------------------------------------
# the inputs, and refusing to guess
# ---------------------------------------------------------------------------


def test_a_table_without_a_protid_column_is_refused(tmp_path):
    path = tmp_path / "bad.tsv"
    pd.DataFrame({"accession": ["P1"], "LeidenCluster": ["LC0"]}).to_csv(
        path, sep="\t", index=False
    )
    with pytest.raises(EnrichmentError, match="no `protid` column"):
        read_keyed_table(str(path), "the cluster table")


def test_a_duplicated_protid_is_refused(tmp_path):
    """It would count one protein twice in every universe it appears in."""
    path = tmp_path / "dup.tsv"
    pd.DataFrame({"protid": ["P1", "P1"], "x": [1, 2]}).to_csv(path, sep="\t", index=False)
    with pytest.raises(EnrichmentError, match="more than once"):
        read_keyed_table(str(path), "the cluster table")


def test_an_unknown_cluster_column_is_refused_by_name():
    clusters = pd.DataFrame({"LeidenCluster": ["LC0"]}, index=pd.Index(["P1"], name="protid"))
    annotations = pd.DataFrame({"Length": [1.0]}, index=pd.Index(["P1"], name="protid"))
    with pytest.raises(EnrichmentError, match="no 'StruCluster' column"):
        join_on_protid(clusters, annotations, "StruCluster")


def test_two_tables_that_share_no_proteins_are_refused():
    clusters = pd.DataFrame({"LeidenCluster": ["LC0"]}, index=pd.Index(["P1"], name="protid"))
    annotations = pd.DataFrame({"Length": [1.0]}, index=pd.Index(["Q9"], name="protid"))
    with pytest.raises(EnrichmentError, match="share no proteins"):
        join_on_protid(clusters, annotations, "LeidenCluster")


def test_the_join_enumerates_what_each_side_lost():
    """A count says something happened; the list says whether it matters. Same
    argument as ADR 0011 §1, one layer out."""
    clusters = pd.DataFrame(
        {"LeidenCluster": ["LC0", "LC0"]}, index=pd.Index(["P1", "P2"], name="protid")
    )
    annotations = pd.DataFrame({"Length": [1.0, 2.0]}, index=pd.Index(["P1", "Q9"], name="protid"))
    joined, dropped = join_on_protid(clusters, annotations, "LeidenCluster")
    assert list(joined.index) == ["P1"]
    assert dropped == {"unannotated": ["P2"], "unclustered": ["Q9"]}


def test_the_annotation_table_may_already_carry_the_cluster_column():
    """It does, in the pipeline. `aggregated_features.tsv` is built by joining
    `leiden_features.tsv` into everything else, so both inputs name the column.

    This is the defect the demo run found and the unit tests did not: the
    fixture wrote an annotation table with the cluster column dropped, which is
    a table the pipeline never produces. Group 6's lesson again -- test the
    caller's path, not only the callee's.
    """
    clusters = pd.DataFrame(
        {"LeidenCluster": ["LC0", "LC1"]}, index=pd.Index(["P1", "P2"], name="protid")
    )
    annotations = pd.DataFrame(
        {"LeidenCluster": ["LC0", "LC1"], "Length": [1.0, 2.0]},
        index=pd.Index(["P1", "P2"], name="protid"),
    )
    joined, _ = join_on_protid(clusters, annotations, "LeidenCluster")
    assert list(joined.columns) == ["LeidenCluster", "Length"]
    assert list(joined["LeidenCluster"]) == ["LC0", "LC1"]


def test_two_tables_that_disagree_about_the_clustering_are_refused():
    """Silently preferring one would enrich one run's clusters against another
    run's annotations, which is wrong in no visible way."""
    clusters = pd.DataFrame(
        {"LeidenCluster": ["LC0", "LC1"]}, index=pd.Index(["P1", "P2"], name="protid")
    )
    annotations = pd.DataFrame(
        {"LeidenCluster": ["LC0", "LC9"], "Length": [1.0, 2.0]},
        index=pd.Index(["P1", "P2"], name="protid"),
    )
    with pytest.raises(EnrichmentError, match="describe different runs"):
        join_on_protid(clusters, annotations, "LeidenCluster")


def test_a_continuous_column_holding_no_numbers_is_refused():
    frame = pd.DataFrame({"Organism": ["Homo sapiens", "Mus musculus"]})
    with pytest.raises(EnrichmentError, match="holds no numbers"):
        numeric_column(frame, "Organism")


def test_unparseable_entries_in_a_numeric_column_are_counted_not_hidden():
    frame = pd.DataFrame({"Length": [1.0, "n/a", 3.0]})
    values, unparseable = numeric_column(frame, "Length")
    assert unparseable == 1
    assert np.isnan(values[1])


def test_enrichment_that_names_no_column_says_how_to_opt_out(monkeypatch, run_dir):
    tmp_path, clusters, annotations = run_dir
    config = write_config(tmp_path, {"categorical": [], "continuous": []})
    with pytest.raises(SystemExit, match="does not enter the DAG"):
        run(monkeypatch, config, tmp_path / "output", clusters, annotations)


# ---------------------------------------------------------------------------
# the config
# ---------------------------------------------------------------------------


def test_a_column_cannot_be_both_kinds():
    with pytest.raises(ConfigError, match="both categorical and continuous"):
        EnrichmentConfig.from_dict({"categorical": ["Length"], "continuous": ["Length"]})


def test_an_encoding_override_must_name_a_categorical_column():
    with pytest.raises(ConfigError, match="not listed under enrichment.categorical"):
        EnrichmentConfig.from_dict({"continuous": ["Length"], "encodings": {"Length": "delimited"}})


def test_an_unknown_encoding_is_refused():
    with pytest.raises(ConfigError, match="is not valid"):
        EnrichmentConfig.from_dict({"categorical": ["Pfam"], "encodings": {"Pfam": "semicolon"}})


@pytest.mark.parametrize(
    "data,message",
    [
        ({"min_term_count": 0}, "at least 1"),
        ({"fdr": 0}, r"must be in \(0, 1\]"),
        ({"fdr": 1.5}, r"must be in \(0, 1\]"),
        ({"cluster_column": "  "}, "must not be empty"),
        ({"nonsense": 1}, "unknown key"),
    ],
)
def test_the_config_refuses_what_it_cannot_honour(data, message):
    with pytest.raises(ConfigError, match=message):
        EnrichmentConfig.from_dict(data)


def test_enrichment_is_off_unless_a_column_is_named():
    assert not EnrichmentConfig().enabled
    assert EnrichmentConfig.from_dict({"continuous": ["Length"]}).enabled


def test_an_encoding_override_beats_detection(monkeypatch, run_dir):
    """The escape hatch for a single-valued column that happens to contain a
    semicolon, which detection would read as a list of two."""
    tmp_path, clusters, annotations = run_dir
    output = tmp_path / "output"
    config = write_config(
        tmp_path,
        {
            **DEFAULT_ENRICHMENT,
            "categorical": ["Pfam"],
            "continuous": [],
            "encodings": {"Pfam": "single"},
        },
    )
    run(monkeypatch, config, output, clusters, annotations)
    manifest = json.loads((output / "enrichment" / "manifest.json").read_text())
    assert manifest["extra"]["categorical"]["Pfam"]["encoding"] == "single"
    # Read whole, `PF00022;PF00125;` is one term rather than two.
    assert any(";" in term for term in read_table(output)["term"])


# ---------------------------------------------------------------------------
# the empty table
# ---------------------------------------------------------------------------


def test_an_empty_result_still_has_the_columns():
    """A downstream reader must not have to special-case a run that tested
    nothing."""
    frame = to_frame([], "LeidenCluster")
    assert list(frame.columns) == list(TABLE_COLUMNS)
    assert frame.empty
