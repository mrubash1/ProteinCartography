"""The parity tests: proof that the default pipeline still produces what it did.

This is the evidence behind the backwards-compatibility claim, so it is written
to be hard to satisfy accidentally.

Three tests, in dependency order:

1. **The baseline is deterministic.** Two runs of the *same* code, compared. What
   differs here is inherently nondeterministic and is recorded as the floor.
   Without this the parity comparison could pass simply because everything is
   noisy.
2. **HEAD is deterministic**, to the same floor. A change that introduced
   nondeterminism would show up here even if it happened to match the baseline
   once.
3. **HEAD matches the baseline** on everything outside that floor.

Each run takes about a minute, so these are marked ``slow`` and skipped unless
``--runslow`` is given. CI runs them on every pull request.

The comparison logic and the reasoning behind each exclusion live in
``parity.py``.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from parity import (
    ADDITIVE_OUTPUTS,
    CRITICAL_OUTPUTS,
    assert_critical_outputs_compared,
    compare_trees,
    normalize_bytes,
    run_pipeline,
)

BASELINE_TAG = "multispace-base"

#: Files whose content legitimately varies between two runs of identical code.
#: Anything else that differs run-to-run is a defect, not a fact of life.
_EXPECTED_RUN_TO_RUN = (
    "semantic_analysis",  # wordcloud layout is stochastic
    "/temp/",  # foldseek internal split databases
    "/temp_tm/",  # ditto, key-protid TM-scores
    "all_by_all_tmscore.tsv",  # raw pair list, emitted in shard order
    "key_protid_tmscores.tsv",  # raw pair list, as above
)


def _run_to_run_expected(relpath: str) -> bool:
    normalized = "/" + relpath.replace(os.sep, "/")
    return any(token in normalized for token in _EXPECTED_RUN_TO_RUN)


@pytest.fixture(scope="module")
def baseline_repo(repo_dirpath):
    """The checkout of tag `multispace-base` used as the parity reference.

    Conventionally a git worktree at ../pc-baseline. Skipped rather than failed
    when absent, because a contributor without the worktree should still be able
    to run the suite.
    """
    candidate = Path(repo_dirpath).parent / "pc-baseline"
    if not (candidate / "Snakefile").exists():
        pytest.skip(
            f"no baseline checkout at {candidate}. Create one with:\n"
            f"  git worktree add ../pc-baseline {BASELINE_TAG}"
        )
    return candidate


@pytest.fixture(scope="module")
def conda_prefix(repo_dirpath):
    """Shared conda environments, so the baseline checkout does not rebuild them."""
    prefix = Path(repo_dirpath) / ".snakemake" / "conda"
    if not prefix.exists():
        pytest.skip(f"no built conda environments at {prefix}; run the pipeline once first")
    return prefix


@pytest.fixture(scope="module")
def runs(tmp_path_factory, repo_dirpath, baseline_repo, conda_prefix):
    """Three pipeline runs: HEAD twice and the baseline once.

    Module-scoped because each run takes about a minute and all three tests
    share them.
    """
    work = tmp_path_factory.mktemp("parity")
    head_a = run_pipeline(Path(repo_dirpath), work / "head_a", conda_prefix=conda_prefix)
    head_b = run_pipeline(Path(repo_dirpath), work / "head_b", conda_prefix=conda_prefix)
    base_a = run_pipeline(baseline_repo, work / "base_a", conda_prefix=conda_prefix)
    base_b = run_pipeline(baseline_repo, work / "base_b", conda_prefix=conda_prefix)
    return {"head_a": head_a, "head_b": head_b, "base_a": base_a, "base_b": base_b}


@pytest.fixture(scope="module")
def nondeterminism_floor(runs):
    """Files that differ between two runs of identical code.

    Established empirically rather than declared, so the parity test cannot be
    weakened by quietly adding an exclusion.
    """
    base_self = compare_trees(runs["base_a"], runs["base_b"])
    head_self = compare_trees(runs["head_a"], runs["head_b"])
    return set(base_self.differing) | set(head_self.differing)


@pytest.mark.slow
def test_the_baseline_is_deterministic(runs):
    """Two runs of the baseline must agree on every scientific output.

    PLAN.md makes this a precondition: a parity test against a nondeterministic
    baseline proves nothing. If this fails, PR #106 did not fully land and the
    port must not proceed.
    """
    report = compare_trees(runs["base_a"], runs["base_b"])

    # Two classes of file are allowed to differ between runs of identical code,
    # and they are named rather than pattern-excluded so that a *new* source of
    # nondeterminism fails this test instead of being absorbed silently:
    #   - the semantic-analysis wordcloud, whose layout is stochastic
    #   - foldseek's internal split databases and the raw pair lists it writes,
    #     which distribute records across shards by thread scheduling. Their
    #     sorted derivatives -- the pivoted matrix and the key-protid features --
    #     are compared, and those are exactly what PR #106 made deterministic.
    unexplained = [r for r in report.differing if not _run_to_run_expected(r)]
    assert not unexplained, (
        "the baseline is not deterministic, so it cannot serve as a parity "
        f"reference:\n{report.describe()}"
    )
    assert not report.only_in_a and not report.only_in_b, report.describe()


@pytest.mark.slow
def test_head_is_deterministic(runs, nondeterminism_floor):
    """The branch must be no less deterministic than the baseline."""
    report = compare_trees(runs["head_a"], runs["head_b"])
    unexplained = [r for r in report.differing if not _run_to_run_expected(r)]
    assert not unexplained, f"this branch introduced nondeterminism:\n{report.describe()}"


@pytest.mark.slow
def test_default_output_is_unchanged_from_the_baseline(runs, nondeterminism_floor):
    """The central claim: the default configuration produces what it always did.

    Everything outside the empirically-established nondeterminism floor must be
    identical, byte for byte, after normalizing Plotly's random figure uuid.
    """
    report = compare_trees(
        runs["head_a"], runs["base_a"], ignore=nondeterminism_floor, baseline="b"
    )
    assert report.ok, f"default output changed:\n{report.describe()}"

    # Guard against the test passing because it compared nothing. A count alone
    # is weak -- it can be satisfied by forty irrelevant files -- so the
    # artifacts that carry the promise are named and checked to be in reach.
    assert_critical_outputs_compared(report, "parity")
    assert (
        report.n_compared >= 35
    ), f"only {report.n_compared} files were compared:\n{report.describe()}"


@pytest.mark.slow
def test_the_scientific_outputs_are_byte_identical(runs, nondeterminism_floor):
    """Name the artifacts explicitly, so nobody has to trust the glob logic.

    `test_default_output_is_unchanged_from_the_baseline` compares everything and
    is the stronger test. This one is unmissable in a review: these specific
    files, byte for byte.
    """
    head, base = Path(runs["head_a"]), Path(runs["base_a"])
    must_match = [
        "final_results/parity_aggregated_features.tsv",
        "final_results/parity_leiden_similarity.tsv",
        "final_results/parity_strucluster_similarity.tsv",
        "foldseek_clustering_results/all_by_all_tmscore_pivoted.tsv",
        "foldseek_clustering_results/all_by_all_tmscore_pivoted_pca_umap.tsv",
        "foldseek_clustering_results/leiden_features.tsv",
        "foldseek_clustering_results/struclusters_features.tsv",
        "protein_features/key_protid_tmscore_features.tsv",
        "protein_features/pdb_features.tsv",
        "protein_features/source_features.tsv",
        "protein_features/uniprot_features.tsv",
    ]
    mismatched = []
    for rel in must_match:
        a, b = head / rel, base / rel
        assert a.exists(), f"{rel} was not produced by this branch"
        assert b.exists(), f"{rel} was not produced by the baseline"
        if a.read_bytes() != b.read_bytes():
            mismatched.append(rel)
    assert not mismatched, f"these outputs changed: {mismatched}"


@pytest.mark.slow
def test_the_pivoted_matrix_is_self_consistent(runs):
    """The matrix must satisfy its own invariants, not merely match the baseline.

    Matching a baseline that was itself wrong would be no comfort, so the header
    order, the label diagonal and the censoring semantics are checked directly.
    """
    from matrix_io import load_labeled_matrix

    path = Path(runs["head_a"]) / "foldseek_clustering_results" / "all_by_all_tmscore_pivoted.tsv"
    matrix = load_labeled_matrix(path)  # raises unless header order == row order
    assert matrix.is_aligned
    assert matrix.is_square
    assert (matrix.aligned_diagonal() == 1.0).all()
    assert matrix.measured_zero_count() == 0


@pytest.mark.slow
def test_the_cohort_report_records_the_truncation_the_baseline_hid(runs):
    """The demo fixture truncates, and the baseline says nothing about it.

    Worth asserting on the real run rather than a unit fixture, because the
    numbers here are the pipeline's own: 24 hits, 20 surviving the metadata
    filter, 10 admitted by `max_structures`. Half the candidates are dropped and
    before ADR 0008 no file recorded that.
    """
    import json

    head, base = Path(runs["head_a"]), Path(runs["base_a"])
    relpath = "protein_features/cohort_report.json"
    assert not (base / relpath).exists(), "the baseline is not supposed to have this file"

    payload = json.loads((head / relpath).read_text())
    assert payload["rule"] == "as_filtered"
    assert payload["truncation_fired"] is True
    assert payload["reproducible"] is False
    assert payload["n_retained"] == payload["max_structures"] == 10
    assert payload["n_discarded"] > 0
    assert payload["n_candidates"] == payload["n_retained"] + payload["n_discarded"]
    assert payload["n_candidates_before_filtering"] >= payload["n_candidates"]
    assert any("not controlled by this pipeline" in w for w in payload["warnings"])


@pytest.mark.slow
def test_the_cohort_the_report_describes_is_the_cohort_that_was_built(runs):
    """A report that disagrees with the run is worse than no report.

    The retained count has to match the structures actually downloaded, or the
    diagnostic is describing a decision the pipeline did not make.
    """
    import json

    head = Path(runs["head_a"])
    payload = json.loads((head / "protein_features" / "cohort_report.json").read_text())
    matrix = head / "foldseek_clustering_results" / "all_by_all_tmscore_pivoted.tsv"
    n_in_map = sum(1 for _ in matrix.read_text().splitlines()) - 1

    # The map also carries the search-mode input proteins, which arrive through
    # `copy_pdb` and were never cohort candidates.
    assert n_in_map == payload["n_retained"] + 1


@pytest.mark.slow
def test_significance_selection_produces_a_reproducible_cohort(
    tmp_path_factory, repo_dirpath, conda_prefix
):
    """The opt-in rule, exercised through the DAG rather than in isolation.

    This is the only coverage of `aggregate_hit_significance` actually running:
    the rule is deliberately absent from the default DAG, so nothing else
    reaches it. Asserted on the report rather than on the outputs, because the
    fixture only carries AlphaFold responses for the ten proteins the *default*
    rule selects -- a different cohort means most downloads fail, which is
    itself confirmation that the rule selects differently.
    """
    import json

    work = tmp_path_factory.mktemp("significance")
    output = run_pipeline(
        Path(repo_dirpath),
        work / "run",
        conda_prefix=conda_prefix,
        extra_config={"cohort": {"selection": "significance", "max_structures": 10}},
    )

    scores = (output / "protein_features" / "hit_significance.tsv").read_text().splitlines()
    assert scores[0] == "protid\tevalue\tbits\tn_queries\tsources"
    assert len(scores) > 100, "the Foldseek results should score many accessions"

    payload = json.loads((output / "protein_features" / "cohort_report.json").read_text())
    assert payload["rule"] == "significance"
    assert payload["measure"] == "evalue"
    assert payload["truncation_fired"] is True
    # The point of the rule: truncation fired and the result is still reproducible.
    assert payload["reproducible"] is True
    assert not any("not controlled by this pipeline" in w for w in payload["warnings"])


@pytest.mark.slow
def test_the_cohort_report_is_deterministic(runs, nondeterminism_floor):
    """It lands in the output tree, so it must not become a source of churn."""
    relpath = "protein_features/cohort_report.json"
    assert relpath not in nondeterminism_floor
    a = (Path(runs["head_a"]) / relpath).read_bytes()
    b = (Path(runs["head_b"]) / relpath).read_bytes()
    assert a == b


# --------------------------------------------------------------------------
# tests of the harness itself
# --------------------------------------------------------------------------


def test_plotly_uuid_normalization_is_narrow():
    """It must strip the figure uuid and nothing that carries meaning."""
    a = b'<div id="1f2e3d4c-5b6a-7980-a1b2-c3d4e5f60718" class="plotly">DATA</div>'
    b = b'<div id="99999999-8888-7777-6666-555555555555" class="plotly">DATA</div>'
    assert normalize_bytes("x.html", a) == normalize_bytes("x.html", b)

    # A real difference in the payload must survive normalization.
    c = b'<div id="99999999-8888-7777-6666-555555555555" class="plotly">OTHER</div>'
    assert normalize_bytes("x.html", b) != normalize_bytes("x.html", c)

    # Non-HTML is untouched.
    assert normalize_bytes("x.tsv", a) == a


def test_compare_trees_detects_a_changed_file(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "final_results").mkdir(parents=True)
    (b / "final_results").mkdir(parents=True)
    (a / "final_results" / "x.tsv").write_text("1\n")
    (b / "final_results" / "x.tsv").write_text("2\n")
    report = compare_trees(a, b)
    assert not report.ok
    assert report.differing == ["final_results/x.tsv"]


def test_compare_trees_detects_a_missing_file(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "only_here.tsv").write_text("1\n")
    report = compare_trees(a, b)
    assert not report.ok
    assert report.only_in_a == ["only_here.tsv"]


def test_compare_trees_applies_exclusions_with_reasons(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "benchmarks").mkdir(parents=True)
    (b / "benchmarks").mkdir(parents=True)
    (a / "benchmarks" / "t.txt").write_text("0.1\n")
    (b / "benchmarks" / "t.txt").write_text("0.2\n")
    report = compare_trees(a, b)
    assert report.ok
    assert "wall-clock" in report.excluded["benchmarks/t.txt"]


def test_ignore_set_is_reported_as_excluded(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "noisy.tsv").write_text("1\n")
    (b / "noisy.tsv").write_text("2\n")
    report = compare_trees(a, b, ignore={"noisy.tsv"})
    assert report.ok
    assert "self-diff" in report.excluded["noisy.tsv"]


COHORT_REPORT = "protein_features/cohort_report.json"


def _trees_where(tmp_path, *, in_a=(), in_b=()):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, relpaths in ((a, in_a), (b, in_b)):
        root.mkdir()
        for rel in relpaths:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
    return a, b


def test_an_additive_output_missing_from_the_baseline_is_allowed(tmp_path):
    """A file this work adds, where the baseline has nothing, is not a failure."""
    a, b = _trees_where(tmp_path, in_a=[COHORT_REPORT])
    report = compare_trees(a, b, baseline="b")
    assert report.ok
    assert "ADR 0008" in report.added[COHORT_REPORT]


def test_the_allowance_follows_the_named_baseline_and_not_the_argument_order(tmp_path):
    """Both call orders are in use, so the direction cannot be inferred.

    This is the case that got the first version of the allowance wrong: it
    assumed the second argument was always the new tree, and the parity test
    passes it first.
    """
    a, b = _trees_where(tmp_path, in_b=[COHORT_REPORT])
    assert compare_trees(a, b, baseline="a").ok
    assert not compare_trees(a, b, baseline="b").ok


def test_an_additive_output_missing_from_the_branch_is_a_removal(tmp_path):
    """Losing a file is never additive, whatever the file is called."""
    a, b = _trees_where(tmp_path, in_b=[COHORT_REPORT])
    report = compare_trees(a, b, baseline="b")
    assert not report.ok
    assert report.only_in_b == [COHORT_REPORT]


def test_the_allowance_does_not_apply_between_two_runs_of_the_same_code(tmp_path):
    """With no baseline named, both trees must agree -- self-diffs included."""
    a, b = _trees_where(tmp_path, in_a=[COHORT_REPORT])
    report = compare_trees(a, b)
    assert not report.ok
    assert report.only_in_a == [COHORT_REPORT]


def test_an_additive_output_is_still_compared_when_both_trees_have_it(tmp_path):
    """The allowance is about absence only. If both runs write it, it must match."""
    a, b = _trees_where(tmp_path, in_a=[COHORT_REPORT], in_b=[COHORT_REPORT])
    (a / COHORT_REPORT).write_text('{"n_retained": 10}\n')
    (b / COHORT_REPORT).write_text('{"n_retained": 11}\n')
    report = compare_trees(a, b, baseline="b")
    assert not report.ok
    assert report.differing == [COHORT_REPORT]


def test_an_unlisted_new_file_still_fails(tmp_path):
    """The allowance is a fixed list, not a rule about new files in general."""
    a, b = _trees_where(tmp_path, in_a=["protein_features/surprise.json"])
    report = compare_trees(a, b, baseline="b")
    assert not report.ok
    assert report.only_in_a == ["protein_features/surprise.json"]


def test_an_unknown_baseline_side_is_rejected():
    with pytest.raises(ValueError, match="baseline must be"):
        compare_trees(Path("."), Path("."), baseline="head")


def test_every_additive_output_carries_a_reason():
    for path, reason in ADDITIVE_OUTPUTS:
        assert path and reason, path


def test_no_additive_pattern_can_swallow_a_critical_output():
    """The allowance must not be able to excuse a missing promised artifact.

    A pattern loose enough to match one of the CRITICAL_OUTPUTS would turn
    "the baseline produced this and we do not" into an allowed difference for
    exactly the files whose byte-identity is the promise.
    """
    import fnmatch

    for pattern, _reason in ADDITIVE_OUTPUTS:
        for critical in CRITICAL_OUTPUTS:
            relpath = critical.format(name="parity")
            assert not fnmatch.fnmatch(relpath, pattern), f"{pattern} matches {relpath}"


def test_a_per_protein_additive_pattern_matches_its_files(tmp_path):
    """The mapping file is one per query protein, so its entry is a glob."""
    a, b = _trees_where(tmp_path, in_a=["blast_results/P60709.blast_hits.mapping.tsv"])
    report = compare_trees(a, b, baseline="b")
    assert report.ok
    assert "discards" in report.added["blast_results/P60709.blast_hits.mapping.tsv"]


def test_a_tree_compared_against_itself_is_clean(tmp_path):
    a = tmp_path / "a"
    (a / "d").mkdir(parents=True)
    (a / "d" / "x.tsv").write_text("hello\n")
    b = tmp_path / "b"
    shutil.copytree(a, b)
    assert compare_trees(a, b).ok


# ==========================================================================
# Component parity at N > 500
# ==========================================================================
#
# Mutation testing showed the 11-protein end-to-end fixture cannot see several
# realistic refactor errors, because at that size the PCA component count, the
# UMAP neighbour count and Leiden's n_pcs are all clamped to the same value
# whatever the config says, and the censoring fill never appears at all. These
# tests run the reduction step alone on a seeded 750-protein matrix, which is
# above the 500-row threshold where `svd_solver="auto"` would switch to the
# randomized solver -- the regime PR #106 exists for.
#
# The fixture is generated, not stored: the seed is the fixture. That also keeps
# it clear of the publishability question hanging over the real-data slice.


@pytest.fixture(scope="module")
def big_matrix(tmp_path_factory):
    from parity import synthetic_matrix

    return synthetic_matrix(tmp_path_factory.mktemp("big") / "all_by_all_tmscore_pivoted.tsv")


@pytest.fixture(scope="module")
def analysis_python(repo_dirpath):
    """An interpreter with scikit-learn and umap-learn.

    The reduction step runs in `envs/analysis.yml`, not in the test environment,
    so the test has to find that environment rather than use its own.
    """
    for candidate in sorted((Path(repo_dirpath) / ".snakemake" / "conda").glob("*/bin/python")):
        probe = subprocess.run([str(candidate), "-c", "import sklearn, umap"], capture_output=True)
        if probe.returncode == 0:
            return str(candidate)
    pytest.skip(
        "no environment with scikit-learn and umap-learn was found under "
        ".snakemake/conda; run the pipeline once so snakemake builds them"
    )


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["pca_umap", "pca_tsne"])
def test_reduction_at_n750_matches_the_baseline(
    mode, runs, big_matrix, baseline_repo, repo_dirpath, analysis_python, tmp_path_factory
):
    """The port must preserve the reduction at a size where clamping does not hide it."""
    from parity import compare_trees, run_reducer

    work = tmp_path_factory.mktemp(f"reduce_{mode}")
    head = run_reducer(
        Path(repo_dirpath), big_matrix, work / "head", mode=mode, python=analysis_python
    )
    base = run_reducer(baseline_repo, big_matrix, work / "base", mode=mode, python=analysis_python)

    report = compare_trees(head, base, baseline="b")
    assert report.ok, f"{mode} output changed at N=750:\n{report.describe()}"
    # The input matrix, the intermediate PCA, and the final embedding.
    assert report.n_compared == 3, report.describe()


@pytest.mark.slow
def test_reduction_at_n750_is_deterministic(
    big_matrix, repo_dirpath, analysis_python, tmp_path_factory
):
    """Above 500 rows is exactly where determinism used to fail."""
    from parity import compare_trees, run_reducer

    work = tmp_path_factory.mktemp("reduce_det")
    a = run_reducer(
        Path(repo_dirpath), big_matrix, work / "a", mode="pca_umap", python=analysis_python
    )
    b = run_reducer(
        Path(repo_dirpath), big_matrix, work / "b", mode="pca_umap", python=analysis_python
    )
    report = compare_trees(a, b)
    assert report.ok, f"reduction is not reproducible at N=750:\n{report.describe()}"


def test_the_synthetic_fixture_has_the_shape_it_claims(tmp_path):
    """The fixture is only useful if it reproduces the real matrix's structure.

    Cheap enough to run in the normal suite, and it fails loudly if a change to
    the generator quietly makes the N>500 tests test something else.
    """
    from matrix_io import load_labeled_matrix, summarize_censoring
    from parity import synthetic_matrix

    matrix = load_labeled_matrix(synthetic_matrix(tmp_path / "m.tsv", n=200, cap=80))
    summary = summarize_censoring(matrix)

    assert matrix.is_aligned
    assert (matrix.aligned_diagonal() == 1.0).all()
    # A per-query cap: rows uniform at the cap, columns free to vary.
    assert summary["cap_detected"]
    assert summary["measured_per_row"]["min"] == summary["measured_per_row"]["max"]
    assert summary["measured_per_col"]["min"] < summary["measured_per_col"]["max"]
    # No measured zeros, so every zero is unambiguously the fill.
    assert summary["measured_zero_count"] == 0
    assert 0.5 < summary["censoring_rate"] < 0.7


def test_the_synthetic_fixture_can_reproduce_the_106_defect(tmp_path):
    from matrix_io import MatrixAlignmentError, load_labeled_matrix
    from parity import synthetic_matrix

    path = synthetic_matrix(tmp_path / "p.tsv", n=50, cap=20, permute_columns=True)
    with pytest.raises(MatrixAlignmentError):
        load_labeled_matrix(path)

    # Permuted, but not corrupt: read by label, the diagonal is still exact.
    matrix = load_labeled_matrix(path, require_alignment=False)
    assert (matrix.aligned_diagonal() == 1.0).all()


def test_the_synthetic_fixture_is_reproducible(tmp_path):
    from parity import synthetic_matrix

    a = synthetic_matrix(tmp_path / "a.tsv", n=40, cap=10).read_bytes()
    b = synthetic_matrix(tmp_path / "b.tsv", n=40, cap=10).read_bytes()
    assert a == b
