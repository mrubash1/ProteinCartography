#!/usr/bin/env python
"""The diagnostics entry point, driven through its real interface.

Written because the pattern has now cost four defects: a passing unit suite is
not evidence the entry point works. Group 8a's most recent was a signature
change that nineteen passing tests could not see, because the caller lived in
another file (REVIEW_LOG G8.6).

So these tests write real blocks through the real :class:`~spaces.store.BlockStore`,
write a real config, and call :func:`diagnose_space.main` with real arguments.
The config is JSON rather than YAML: ``envs/analysis.yml`` carries no PyYAML,
and JSON is what the Snakefile hands these scripts anyway.
"""

from __future__ import annotations
import json
import sys

import numpy as np
import pytest
from embedding_cohort import FOLD, ISOMETRIC, SPLIT, embedding_cohort
from fusion_cohort import NARROW_BLOCK, RESCALED_BLOCK, WIDE_BLOCK, fusion_cohort


def _write_config(path, spaces, blocks):
    path.write_text(json.dumps({"blocks": blocks, "spaces": spaces}))
    return str(path)


def _run(argv):
    import diagnose_space

    original = sys.argv
    try:
        sys.argv = ["diagnose_space.py", *argv]
        return diagnose_space.main()
    finally:
        sys.argv = original


def _report(output_dir, space_id="structure") -> dict:
    with open(output_dir / "spaces" / space_id / "diagnostics.json") as handle:
        return json.load(handle)


# --- a multi-block space ----------------------------------------------------


@pytest.fixture(scope="module")
def fusion(tmp_path_factory):
    """Two redundant blocks and one independent one, on disk."""
    from fusion_cohort import write_fusion_cohort

    root = tmp_path_factory.mktemp("diagnose_fusion")
    output_dir = root / "output"
    cohort = fusion_cohort()
    write_fusion_cohort(output_dir, cohort, [WIDE_BLOCK, NARROW_BLOCK, RESCALED_BLOCK])
    config = _write_config(
        root / "config.json",
        {
            "redundant": {
                "blocks": [NARROW_BLOCK, RESCALED_BLOCK],
                "strategy": "late",
                "reducers": ["pca"],
            },
            "independent": {
                "blocks": [WIDE_BLOCK, NARROW_BLOCK],
                "strategy": "late",
                "reducers": ["pca"],
            },
        },
        {b: {"provider": "tmscore"} for b in (WIDE_BLOCK, NARROW_BLOCK, RESCALED_BLOCK)},
    )
    return root, output_dir, config


def test_a_space_of_two_redundant_blocks_says_so(fusion):
    """`narrow_rescaled` is `narrow` times 1000, so the entry point has to see
    through the unit change.

    Note what is *not* asserted: exact equality with 1.0. The unit test on the
    same two blocks gets exactly 1.0 and this one gets 0.9999999999949,
    because the block store writes float32 (ADR 0004) and quantizing two copies
    at scales 0.01 and 10 is not a proportional operation -- measured deviation
    9.7e-8 relative, one float32 epsilon, which is enough to swap a handful of
    the 28,680 distance ranks. The exactness is a property of the arrays, not
    of the pipeline, and asserting it here would be asserting that the store
    does something it deliberately does not.
    """
    root, output_dir, config = fusion
    assert _run(["-c", config, "-s", "redundant", "-o", str(output_dir)]) == 0
    report = _report(output_dir, "redundant")
    assert report["redundancy"]["pairs"][0]["redundant"] is True
    assert report["redundancy"]["pairs"][0]["spearman"] > 1.0 - 1e-9
    assert any("counts one view twice" in w for w in report["redundancy"]["warnings"])


@pytest.fixture(scope="module")
def independent(fusion):
    """One run of ``-s independent``, shared by the five tests that read it.

    All five invoked the entry point with byte-identical argv into the same
    module-scoped output dir, so four of them were doing nothing but
    overwriting the first one's files -- measured 5 x ~130 ms, about 0.5 s of
    this file's 2.3 s.

    The return code is handed back rather than asserted here so
    ``test_a_space_of_independent_blocks_says_that_too`` keeps making that
    assertion: a fixture that swallowed a nonzero exit would turn one clear
    failure into four confusing KeyErrors.

    Read-only. The output tree is shared, and `spaces/independent/` is written
    by nothing else in this module -- the `redundant` space is a different argv
    and stays where it is, run once, in its own test.
    """
    root, output_dir, config = fusion
    return output_dir, _run(["-c", config, "-s", "independent", "-o", str(output_dir)])


def test_a_space_of_independent_blocks_says_that_too(independent):
    output_dir, code = independent
    assert code == 0
    report = _report(output_dir, "independent")
    assert report["redundancy"]["pairs"][0]["redundant"] is False
    assert any("something the others do not" in w for w in report["redundancy"]["warnings"])


def test_the_report_records_what_the_space_is(independent):
    output_dir, _ = independent
    report = _report(output_dir, "independent")
    assert report["space_id"] == "independent"
    assert report["strategy"] == "late"
    assert report["blocks"] == [WIDE_BLOCK, NARROW_BLOCK]
    assert report["n_proteins"] == 240


def test_a_manifest_lands_beside_the_report(independent):
    output_dir, _ = independent
    path = output_dir / "spaces" / "independent" / "manifest_diagnostics.json"
    manifest = json.loads(path.read_text())
    assert manifest["provider"] == "diagnose_space"
    assert "redundancy" in manifest["extra"]["sections"]


def test_the_manifest_names_only_sections_that_are_sections(independent):
    """`strategy` and `n_proteins` describe the space rather than diagnosing
    it, and listing them as sections would tell a reader that four diagnostics
    ran when two did. A section's *absence* is information -- this space has no
    embedding passed in and no cohort report -- so the set has to be accurate."""
    from diagnose_space import SECTIONS

    output_dir, _ = independent
    path = output_dir / "spaces" / "independent" / "manifest_diagnostics.json"
    sections = json.loads(path.read_text())["extra"]["sections"]
    assert set(sections) <= set(SECTIONS)
    assert "strategy" not in sections
    assert "n_proteins" not in sections
    # This space is two feature blocks with no censoring channel, no embedding
    # argument and no cohort report. Redundancy and stability are the two that
    # can still be answered: stability needs only the space's own distances,
    # and the two partition sections are opt-in and unrequested here.
    assert sections == ["redundancy", "stability"]


def test_a_section_that_could_not_be_answered_is_absent_from_the_manifest(single, tmp_path):
    """The other direction: a single-block space cannot report redundancy, and
    the manifest has to say so by omission rather than by listing an empty one."""
    from diagnose_space import SECTIONS

    root, output_dir, config, cohort = single
    path = tmp_path / "iso.tsv"
    cohort.write_embedding(path, ISOMETRIC)
    _run(
        [
            "-c",
            config,
            "-s",
            "structure",
            "-o",
            str(output_dir),
            "--embedding",
            f"pca_umap={path}",
        ]
    )
    manifest = output_dir / "spaces" / "structure" / "manifest_diagnostics.json"
    sections = json.loads(manifest.read_text())["extra"]["sections"]
    assert sections == ["faithfulness", "stability"]
    assert set(sections) < set(SECTIONS)


# --- a single-block space ---------------------------------------------------


@pytest.fixture(scope="module")
def single(tmp_path_factory):
    """One block, with a planted-faithfulness embedding to score."""
    from spaces.store import BlockStore

    root = tmp_path_factory.mktemp("diagnose_single")
    output_dir = root / "output"
    cohort = embedding_cohort()
    BlockStore(str(output_dir)).write_block(cohort.block_result("truth"))
    config = _write_config(
        root / "config.json",
        {"structure": {"blocks": ["truth"], "strategy": "none", "reducers": ["pca_umap"]}},
        {"truth": {"provider": "tmscore"}},
    )
    return root, output_dir, config, cohort


@pytest.fixture(scope="module")
def single_bare(single):
    """One bare ``-s structure`` run over the single-block space, snapshotted.

    Two tests assert on this run -- no redundancy section, no censoring section
    -- and each used to invoke the entry point itself, ~150 ms together.

    The parsed *report* is captured here rather than the output dir on purpose.
    The tests in between run the same space with ``--embedding`` and overwrite
    `spaces/structure/diagnostics.json`, so a fixture handing back only the
    directory would let the second consumer read a different run's file and
    still pass -- the assertions are absence assertions, which is exactly the
    shape that passes against the wrong file. Read-only: the dict is shared.
    """
    root, output_dir, config, _ = single
    return _run(["-c", config, "-s", "structure", "-o", str(output_dir)]), _report(output_dir)


def test_a_single_block_space_reports_no_redundancy_section(single_bare):
    code, report = single_bare
    assert code == 0
    assert "redundancy" not in report


def test_a_faithful_layout_is_reported_as_faithful(single, tmp_path):
    root, output_dir, config, cohort = single
    path = tmp_path / "isometric.tsv"
    cohort.write_embedding(path, ISOMETRIC)
    _run(
        [
            "-c",
            config,
            "-s",
            "structure",
            "-o",
            str(output_dir),
            "--embedding",
            f"pca_umap={path}",
        ]
    )
    entry = _report(output_dir)["faithfulness"][0]
    assert entry["reducer"] == "pca_umap"
    assert entry["trustworthiness_mean"] == 1.0
    assert entry["continuity_mean"] == 1.0
    assert entry["n_unreliable"] == 0


@pytest.mark.parametrize(
    "case,expected",
    [(FOLD, "over-compressed"), (SPLIT, "torn rather than crowded")],
)
def test_a_distorted_layout_is_reported_in_the_right_direction(single, tmp_path, case, expected):
    """The planted 2x2, carried all the way through the entry point. A swap
    anywhere between the fixture and the JSON transposes this."""
    root, output_dir, config, cohort = single
    path = tmp_path / f"{case}.tsv"
    cohort.write_embedding(path, case)
    _run(
        [
            "-c",
            config,
            "-s",
            "structure",
            "-o",
            str(output_dir),
            "--embedding",
            f"pca_umap={path}",
        ]
    )
    entry = _report(output_dir)["faithfulness"][0]
    assert any(expected in w for w in entry["warnings"])


def test_a_per_protein_table_is_written_beside_the_report(single, tmp_path):
    import pandas as pd

    root, output_dir, config, cohort = single
    path = tmp_path / "fold.tsv"
    cohort.write_embedding(path, FOLD)
    _run(
        [
            "-c",
            config,
            "-s",
            "structure",
            "-o",
            str(output_dir),
            "--embedding",
            f"pca_umap={path}",
        ]
    )
    written = output_dir / "spaces" / "structure" / "faithfulness_pca_umap.tsv"
    frame = pd.read_csv(written, sep="\t", index_col=0)
    assert frame.index.name == "protid"
    assert list(frame.columns) == ["trustworthiness", "continuity"]
    assert list(frame.index) == cohort.protids


def test_an_embedding_missing_proteins_is_refused(single, tmp_path):
    """Silently scoring the overlap would report a faithful map of a cohort
    nobody chose."""
    root, output_dir, config, cohort = single
    path = tmp_path / "short.tsv"
    cohort.embedding_frame(ISOMETRIC).iloc[:100].to_csv(path, sep="\t")
    with pytest.raises(SystemExit, match="disagree about the cohort"):
        _run(
            [
                "-c",
                config,
                "-s",
                "structure",
                "-o",
                str(output_dir),
                "--embedding",
                f"pca_umap={path}",
            ]
        )


# --- censoring, wired in rather than rebuilt --------------------------------


@pytest.fixture(scope="module")
def censored(tmp_path_factory):
    """A profile block carrying a censoring channel, as `tmscore` produces."""
    from spaces.base import BlockResult, BlockSpec
    from spaces.store import BlockStore

    root = tmp_path_factory.mktemp("diagnose_censored")
    output_dir = root / "output"
    rng = np.random.RandomState(0)
    n = 60
    protids = [f"CN{i:03d}" for i in range(n)]
    labels = np.repeat(np.arange(3), n // 3)
    values = np.where(labels[:, None] == labels[None, :], 0.85, 0.15)
    values = values + rng.normal(0, 0.02, size=(n, n))
    values = 0.5 * (values + values.T)
    np.fill_diagonal(values, 1.0)
    # Censor the weakest pairs, which is what a per-query cap does: it takes
    # the between-cluster edges first.
    mask = values < 0.2
    np.fill_diagonal(mask, False)
    values = np.where(mask, 0.0, values)

    BlockStore(str(output_dir)).write_block(
        BlockResult(
            spec=BlockSpec(
                id="tmscore",
                kind="features",
                fusable=True,
                metric="euclidean",
                normalization="none",
                provider="tmscore",
            ),
            protids=protids,
            features=values.astype(np.float32),
            channels={"censored": mask},
        )
    )
    clusters = root / "clusters.tsv"
    clusters.write_text(
        "protid\tLeidenCluster\n" + "".join(f"{p}\t{c}\n" for p, c in zip(protids, labels))
    )
    config = _write_config(
        root / "config.json",
        {"structure": {"blocks": ["tmscore"], "strategy": "none", "reducers": ["pca"]}},
        {"tmscore": {"provider": "tmscore"}},
    )
    return root, output_dir, config, clusters


def test_censoring_is_reported_for_a_block_that_carries_a_mask(censored):
    """Group 8c changed what this test can assert, and the change is the point.

    Before spaces clustered, omitting ``--clusters`` left the space with no
    partition and cross-cluster edge retention was absent. Now the space
    supplies its own, so the retention is reported *from the space's own
    grouping* -- which is FOLLOWUPS #41: the legacy structural partition is the
    right one for `structure` and the wrong one for `physicochemistry`, and
    until now every space got the structural one. The section is therefore
    present exactly when the space could be clustered.
    """
    from clustering import is_available

    root, output_dir, config, _ = censored
    assert _run(["-c", config, "-s", "structure", "-o", str(output_dir)]) == 0
    report = _report(output_dir)
    section = report["censoring"]
    assert len(section) == 1
    assert section[0]["block_id"] == "tmscore"
    assert section[0]["matrix"]["censoring_rate"] > 0.3
    if is_available()[0]:
        assert report["partition"]["source"] == "this space"
        assert "cross_cluster_edge_retention" in section[0]
    else:
        assert "cross_cluster_edge_retention" not in section[0]


def test_supplying_clusters_turns_on_cross_cluster_edge_retention(censored):
    """The number worth reading, and the reason `--clusters` exists.

    Between-cluster pairs are the weakest, so they are censored first. Since
    group 8c the supplied partition is the *fallback* rather than the only
    source: a space that can cluster itself uses its own, and a space that
    cannot uses this one and says so in the report's `partition.caveat`.
    """
    root, output_dir, config, clusters = censored
    _run(
        [
            "-c",
            config,
            "-s",
            "structure",
            "-o",
            str(output_dir),
            "--clusters",
            str(clusters),
        ]
    )
    retention = _report(output_dir)["censoring"][0]["cross_cluster_edge_retention"]
    assert retention["n_clusters"] == 3
    assert retention["between_retention"] < retention["within_retention"]
    assert retention["between_over_within"] < 0.5


def test_a_block_with_no_censoring_channel_gets_no_censoring_section(single_bare):
    _, report = single_bare
    assert "censoring" not in report


def test_a_cluster_table_without_a_protid_column_is_refused(censored, tmp_path):
    root, output_dir, config, _ = censored
    bad = tmp_path / "bad.tsv"
    bad.write_text("name\tcluster\nCN000\t0\n")
    with pytest.raises(SystemExit, match="no 'protid' column"):
        _run(
            [
                "-c",
                config,
                "-s",
                "structure",
                "-o",
                str(output_dir),
                "--clusters",
                str(bad),
            ]
        )


# --- the cohort report, passed through rather than recomputed ---------------


def test_the_cohort_report_is_carried_into_the_space_report(single, tmp_path):
    root, output_dir, config, _ = single
    cohort_report = tmp_path / "cohort_report.json"
    cohort_report.write_text(
        json.dumps(
            {
                "rule": "as_filtered",
                "n_retained": 240,
                "warnings": ["truncation was not reproducible"],
            }
        )
    )
    _run(
        [
            "-c",
            config,
            "-s",
            "structure",
            "-o",
            str(output_dir),
            "--cohort-report",
            str(cohort_report),
        ]
    )
    report = _report(output_dir)
    assert report["cohort"]["rule"] == "as_filtered"
    assert report["cohort"]["warnings"] == ["truncation was not reproducible"]


def test_a_missing_cohort_report_is_simply_absent(single):
    root, output_dir, config, _ = single
    _run(
        [
            "-c",
            config,
            "-s",
            "structure",
            "-o",
            str(output_dir),
            "--cohort-report",
            "/nonexistent/cohort_report.json",
        ]
    )
    assert "cohort" not in _report(output_dir)


# --- the guards -------------------------------------------------------------


def test_an_undefined_space_is_refused(single):
    root, output_dir, config, _ = single
    with pytest.raises(SystemExit, match="is not defined"):
        _run(["-c", config, "-s", "nope", "-o", str(output_dir)])


def test_a_malformed_embedding_argument_is_refused(single):
    root, output_dir, config, _ = single
    with pytest.raises(SystemExit, match="expects NAME=PATH"):
        _run(["-c", config, "-s", "structure", "-o", str(output_dir), "--embedding", "justapath"])


def test_the_report_is_plain_json(independent):
    """numpy scalars are not JSON, and the report is assembled from four
    modules that each produce their own. A crashed rule is a worse diagnostic
    than a rounded one."""
    output_dir, _ = independent
    raw = (output_dir / "spaces" / "independent" / "diagnostics.json").read_text()
    assert json.loads(raw) == json.loads(json.dumps(json.loads(raw)))


# --- group 8c: stability, the sweep, and the negative controls ---------------


def _diag_config(path, spaces, blocks, diagnostics):
    path.write_text(json.dumps({"blocks": blocks, "spaces": spaces, "diagnostics": diagnostics}))
    return str(path)


@pytest.fixture(scope="module")
def crossed(tmp_path_factory):
    """A space fusing both blocks of ``fusion_cohort``, so its right answer is
    the twelve cells of the crossed partition rather than either block's four
    or three."""
    from fusion_cohort import write_fusion_cohort

    root = tmp_path_factory.mktemp("diagnose_8c")
    output_dir = root / "output"
    cohort = fusion_cohort()
    write_fusion_cohort(output_dir, cohort, [WIDE_BLOCK, NARROW_BLOCK])
    return root, output_dir, cohort


def _run_crossed(root, output_dir, diagnostics, name="fused"):
    config = _diag_config(
        root / f"config_{name}.json",
        {name: {"blocks": [WIDE_BLOCK, NARROW_BLOCK], "strategy": "late", "reducers": ["pca"]}},
        {WIDE_BLOCK: {"provider": "tmscore"}, NARROW_BLOCK: {"provider": "tmscore"}},
        diagnostics,
    )
    assert _run(["-c", config, "-s", name, "-o", str(output_dir)]) == 0
    return _report(output_dir, name)


def test_stability_is_reported_for_a_space_with_no_optional_dependency(crossed):
    """It needs only the space's own distances, so it is the one section that
    is always available."""
    root, output_dir, _ = crossed
    report = _run_crossed(root, output_dir, {"k": 10}, name="always")
    stability = report["stability"][0]
    assert stability["n_proteins"] == 240
    assert stability["k"] == 10
    assert 0.0 <= stability["stability_mean"] <= 1.0
    assert stability["n_measured"] == 240


def test_the_configured_replicate_count_and_fraction_reach_the_statistic(crossed):
    """Both fields were dead until this commit; asserting they round-trip
    through `from_legacy` would not have noticed."""
    root, output_dir, _ = crossed
    report = _run_crossed(
        root,
        output_dir,
        {"k": 10, "bootstrap_replicates": 7, "subsample_fraction": 0.5},
        name="tuned",
    )
    stability = report["stability"][0]
    assert stability["replicates"] == 7
    assert stability["subsample_fraction"] == 0.5


def test_zero_replicates_turns_stability_off(crossed):
    root, output_dir, _ = crossed
    report = _run_crossed(root, output_dir, {"bootstrap_replicates": 0}, name="off")
    assert "stability" not in report


def test_stability_k_is_clamped_to_the_subsample_not_the_cohort(crossed):
    """A 50% subsample of 240 leaves 120, so k cannot exceed 119 whatever the
    cohort could support. Group 8b's defect, one level deeper."""
    root, output_dir, _ = crossed
    report = _run_crossed(root, output_dir, {"k": 150, "subsample_fraction": 0.5}, name="clamped")
    stability = report["stability"][0]
    assert stability["k"] == 119
    assert stability["k_requested"] == 150
    assert any("reduced from 150" in note for note in stability["warnings"])


def test_the_partition_section_records_which_partition_was_used(crossed):
    """Whether a space clustered in its own right or borrowed the legacy
    structural clustering changes what every partition-dependent number means,
    so it is recorded rather than inferable."""
    from clustering import is_available

    root, output_dir, _ = crossed
    report = _run_crossed(root, output_dir, {"k": 10}, name="whichpart")
    available, _ = is_available()
    if available:
        assert report["partition"]["source"] == "this space"
        assert report["partition"]["n_clusters"] >= 2
    else:
        assert "partition" not in report


def test_an_unclusterable_space_still_produces_the_other_sections(crossed):
    """ADR 0006 rule 2, at the level of a rule rather than a provider: without
    scanpy the partition sections vanish and nothing else does."""
    root, output_dir, _ = crossed
    report = _run_crossed(
        root,
        output_dir,
        {"k": 10, "leiden_resolution_sweep": [0.5, 1.0], "negative_controls": ["shuffled_labels"]},
        name="degraded",
    )
    from clustering import is_available

    assert "stability" in report and "redundancy" in report
    if not is_available()[0]:
        assert "resolution_sweep" not in report
        assert "negative_controls" not in report


@pytest.mark.skipif(not __import__("clustering").is_available()[0], reason="needs scanpy")
def test_the_space_is_clustered_and_the_partition_written_beside_the_report(crossed):
    root, output_dir, cohort = crossed
    _run_crossed(root, output_dir, {"k": 10}, name="clustered")
    path = output_dir / "spaces" / "clustered" / "clusters.tsv"
    assert path.exists()
    lines = path.read_text().splitlines()
    assert lines[0] == "protid\tcluster"
    assert len(lines) == cohort.n_proteins + 1


@pytest.mark.skipif(not __import__("clustering").is_available()[0], reason="needs scanpy")
def test_a_late_fusion_of_two_crossed_blocks_recovers_twelve_cells(crossed):
    """The planted answer, recovered through the entry point.

    `wide` shows four fold groups and `narrow` three chemistry groups, crossed
    exactly. A space fusing both should see the twelve cells -- not four, not
    three. Nothing about that is inferable from either block alone, which is
    what makes it a test of the whole path rather than of the clusterer.
    """
    root, output_dir, _ = crossed
    report = _run_crossed(root, output_dir, {"k": 10}, name="twelve")
    assert report["partition"]["n_clusters"] == 12


@pytest.mark.skipif(not __import__("clustering").is_available()[0], reason="needs scanpy")
def test_the_resolution_sweep_lands_in_the_report(crossed):
    root, output_dir, _ = crossed
    report = _run_crossed(
        root, output_dir, {"k": 10, "leiden_resolution_sweep": [0.25, 1.0, 4.0]}, name="swept"
    )
    sweep = report["resolution_sweep"]
    assert [step["resolution"] for step in sweep["steps"]] == [0.25, 1.0, 4.0]
    assert len(sweep["adjacent_ari"]) == 2
    assert sweep["plateau"]["n_resolutions"] >= 1


@pytest.mark.skipif(not __import__("clustering").is_available()[0], reason="needs scanpy")
def test_the_negative_controls_land_in_the_report(crossed):
    """Both controls, and the observed partition beating both. On a fixture
    with a planted answer that is the expected result; a space where it did not
    hold is exactly what item 8 exists to surface."""
    root, output_dir, _ = crossed
    report = _run_crossed(
        root,
        output_dir,
        {"k": 10, "negative_controls": ["shuffled_labels", "random_distances"]},
        name="controlled",
    )
    controls = report["negative_controls"]
    assert {c["name"] for c in controls["controls"]} == {"shuffled_labels", "random_distances"}
    assert all(margin > 0 for margin in controls["margins"].values())
    assert controls["observed"]["silhouette_mean"] > 0.5


@pytest.mark.skipif(not __import__("clustering").is_available()[0], reason="needs scanpy")
def test_the_manifest_lists_the_new_sections(crossed):
    from diagnose_space import SECTIONS

    root, output_dir, _ = crossed
    _run_crossed(
        root,
        output_dir,
        {"k": 10, "leiden_resolution_sweep": [0.5, 1.0], "negative_controls": ["shuffled_labels"]},
        name="manifested",
    )
    path = output_dir / "spaces" / "manifested" / "manifest_diagnostics.json"
    sections = json.loads(path.read_text())["extra"]["sections"]
    assert set(sections) <= set(SECTIONS)
    assert {"stability", "resolution_sweep", "negative_controls"} <= set(sections)
    assert "partition" not in sections


# --- the config keys these sections read -------------------------------------


def test_an_unknown_negative_control_is_refused():
    from config_schema import ConfigError, from_legacy

    with pytest.raises(ConfigError, match="unknown control"):
        from_legacy({"diagnostics": {"negative_controls": ["shufled_labels"]}})


def test_a_sweep_of_one_resolution_is_refused():
    """It has no adjacent pair, so it measures nothing -- the same reasoning
    that makes a single-block space produce no redundancy section."""
    from config_schema import ConfigError, from_legacy

    with pytest.raises(ConfigError, match="no adjacent pair"):
        from_legacy({"diagnostics": {"leiden_resolution_sweep": [1.0]}})


def test_a_repeated_resolution_is_refused():
    from config_schema import ConfigError, from_legacy

    with pytest.raises(ConfigError, match="distinct"):
        from_legacy({"diagnostics": {"leiden_resolution_sweep": [1.0, 1.0]}})


def test_a_non_positive_resolution_is_refused():
    from config_schema import ConfigError, from_legacy

    with pytest.raises(ConfigError, match="must be positive"):
        from_legacy({"diagnostics": {"leiden_resolution_sweep": [0.5, 0.0]}})
