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


def test_a_space_of_independent_blocks_says_that_too(fusion):
    root, output_dir, config = fusion
    assert _run(["-c", config, "-s", "independent", "-o", str(output_dir)]) == 0
    report = _report(output_dir, "independent")
    assert report["redundancy"]["pairs"][0]["redundant"] is False
    assert any("something the others do not" in w for w in report["redundancy"]["warnings"])


def test_the_report_records_what_the_space_is(fusion):
    root, output_dir, config = fusion
    _run(["-c", config, "-s", "independent", "-o", str(output_dir)])
    report = _report(output_dir, "independent")
    assert report["space_id"] == "independent"
    assert report["strategy"] == "late"
    assert report["blocks"] == [WIDE_BLOCK, NARROW_BLOCK]
    assert report["n_proteins"] == 240


def test_a_manifest_lands_beside_the_report(fusion):
    root, output_dir, config = fusion
    _run(["-c", config, "-s", "independent", "-o", str(output_dir)])
    path = output_dir / "spaces" / "independent" / "manifest_diagnostics.json"
    manifest = json.loads(path.read_text())
    assert manifest["provider"] == "diagnose_space"
    assert "redundancy" in manifest["extra"]["sections"]


def test_the_manifest_names_only_sections_that_are_sections(fusion):
    """`strategy` and `n_proteins` describe the space rather than diagnosing
    it, and listing them as sections would tell a reader that four diagnostics
    ran when two did. A section's *absence* is information -- this space has no
    embedding passed in and no cohort report -- so the set has to be accurate."""
    from diagnose_space import SECTIONS

    root, output_dir, config = fusion
    _run(["-c", config, "-s", "independent", "-o", str(output_dir)])
    path = output_dir / "spaces" / "independent" / "manifest_diagnostics.json"
    sections = json.loads(path.read_text())["extra"]["sections"]
    assert set(sections) <= set(SECTIONS)
    assert "strategy" not in sections
    assert "n_proteins" not in sections
    # This space is two feature blocks with no censoring channel, no embedding
    # argument and no cohort report, so exactly one of the four can be answered.
    assert sections == ["redundancy"]


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
    assert sections == ["faithfulness"]
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


def test_a_single_block_space_reports_no_redundancy_section(single):
    root, output_dir, config, _ = single
    assert _run(["-c", config, "-s", "structure", "-o", str(output_dir)]) == 0
    assert "redundancy" not in _report(output_dir)


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
    root, output_dir, config, _ = censored
    assert _run(["-c", config, "-s", "structure", "-o", str(output_dir)]) == 0
    section = _report(output_dir)["censoring"]
    assert len(section) == 1
    assert section[0]["block_id"] == "tmscore"
    assert section[0]["matrix"]["censoring_rate"] > 0.3
    assert "cross_cluster_edge_retention" not in section[0]


def test_supplying_clusters_turns_on_cross_cluster_edge_retention(censored):
    """The number worth reading, and the reason `--clusters` exists. Between-
    cluster pairs are the weakest, so they are censored first."""
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


def test_a_block_with_no_censoring_channel_gets_no_censoring_section(single):
    root, output_dir, config, _ = single
    _run(["-c", config, "-s", "structure", "-o", str(output_dir)])
    assert "censoring" not in _report(output_dir)


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


def test_the_report_is_plain_json(fusion):
    """numpy scalars are not JSON, and the report is assembled from four
    modules that each produce their own. A crashed rule is a worse diagnostic
    than a rounded one."""
    root, output_dir, config = fusion
    _run(["-c", config, "-s", "independent", "-o", str(output_dir)])
    raw = (output_dir / "spaces" / "independent" / "diagnostics.json").read_text()
    assert json.loads(raw) == json.loads(json.dumps(json.loads(raw)))
