"""Regression tests for defects found by the adversarial review gates.

Every test here corresponds to a finding in ``docs/REVIEW_LOG.md``. They are
kept together rather than filed under the module they exercise because the
useful property is the provenance: each one is a bug that was actually present
and actually produced a wrong answer, so a failure here means a real regression
rather than a changed opinion about design.

Two of these (B1, B2) were "blocks" severity -- they returned wrong numbers and
reported success.
"""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from config_schema import ConfigError, MultispaceConfig
from index import IndexAlignmentError, ProteinIndex
from spaces.base import BlockResult, BlockSpec, BlockSpecError, NotFusableError
from spaces.manifest import Manifest
from spaces.store import BlockStore

REPO_ROOT = Path(__file__).resolve().parents[2]

FILL = "0.0"


def fmt(v):
    return f"{v:.3E}"


def write(path, row_labels, col_labels, cells):
    lines = ["\t".join(["protid"] + list(col_labels))]
    for label, row in zip(row_labels, cells):
        lines.append("\t".join([label] + list(row)))
    path.write_text("\n".join(lines) + "\n")
    return path


# ==========================================================================
# B2 (blocks) -- align silently took the last duplicate from the SOURCE labels
# ==========================================================================


def test_b2_duplicate_source_labels_are_refused():
    """Was: `{p: i for i, p in enumerate(protids)}` kept the last occurrence, so
    aligning source ['A','B','A'] to index ('A','B') returned source row 2 for
    'A' -- silently, because the length check still passed.
    """
    idx = ProteinIndex.from_iterable(["A", "B"])
    values = np.array([[10.0, 10.0], [20.0, 20.0], [99.0, 99.0]])
    with pytest.raises(IndexAlignmentError) as excinfo:
        idx.align(["A", "B", "A"], values)
    message = str(excinfo.value)
    assert "duplicate protid" in message
    assert "'A'" in message
    assert "discard the rest" in message


def test_b2_align_frame_inherits_the_guard():
    import pandas as pd

    idx = ProteinIndex.from_iterable(["A", "B"])
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.Index(["A", "B", "A"]))
    with pytest.raises(IndexAlignmentError, match="duplicate protid"):
        idx.align_frame(frame)


def test_b2_a_duplicate_that_also_hides_an_absence_reports_both():
    """A row duplicated in place of a missing one is the common shape of this
    mistake, so the error names both problems rather than making the caller fix
    one to discover the next.
    """
    idx = ProteinIndex.from_iterable(["A", "B", "C"])
    values = np.zeros((3, 1))
    with pytest.raises(IndexAlignmentError) as excinfo:
        idx.align(["A", "B", "B"], values)
    message = str(excinfo.value)
    assert "duplicate protid" in message
    assert "3 proteins are missing" in message or "missing from the source" in message
    assert "'C'" in message


# ==========================================================================
# B3 -- is_fresh could never return True across processes
# ==========================================================================


def pairwise_spec(**overrides):
    kwargs = {
        "id": "tmscore",
        "kind": "pairwise",
        "fusable": True,
        "metric": "precomputed",
        "normalization": "unit_mean_distance",
        "provider": "tmscore",
        "symmetrization": "mean",
    }
    kwargs.update(overrides)
    return BlockSpec(**kwargs)


def test_b3_cache_hits_for_an_independently_rebuilt_manifest(tmp_path):
    """Was: write_block injected values_digest/spec into manifest.extra, which
    feeds cache_key, so only a caller who already had the output could match --
    i.e. never the caller the cache exists for.
    """
    store = BlockStore(str(tmp_path))
    built = Manifest.build("block", "tmscore", params={"v": 1}, protids=["A", "B", "C"])
    store.write_block(
        BlockResult(spec=pairwise_spec(), protids=["A", "B", "C"], distances=np.zeros(3)),
        built,
    )
    # Rebuilt from the same inputs by a fresh process that has NOT computed the
    # block. This is the only case that matters.
    rebuilt = Manifest.build("block", "tmscore", params={"v": 1}, protids=["A", "B", "C"])
    assert store.is_fresh("tmscore", rebuilt)


def test_b3_write_block_does_not_mutate_the_callers_manifest(tmp_path):
    """Was: the passing freshness tests passed *because* of this mutation."""
    store = BlockStore(str(tmp_path))
    manifest = Manifest.build("block", "tmscore", protids=["A", "B"])
    before = copy.deepcopy(manifest)
    store.write_block(
        BlockResult(spec=pairwise_spec(), protids=["A", "B"], distances=np.zeros(1)),
        manifest,
    )
    assert manifest.extra == before.extra
    assert manifest.derived == before.derived
    assert manifest.cache_key == before.cache_key


def test_b3_changed_inputs_still_miss(tmp_path):
    store = BlockStore(str(tmp_path))
    store.write_block(
        BlockResult(spec=pairwise_spec(), protids=["A", "B"], distances=np.zeros(1)),
        Manifest.build("block", "tmscore", params={"v": 1}, protids=["A", "B"]),
    )
    assert not store.is_fresh(
        "tmscore", Manifest.build("block", "tmscore", params={"v": 2}, protids=["A", "B"])
    )


def test_b3_values_digest_describes_the_file_as_stored(tmp_path):
    """Was: the digest hashed the pre-cast float64 array while the file on disk
    was float32, so it could never verify its own file.
    """
    from spaces.manifest import values_digest

    store = BlockStore(str(tmp_path))
    store.write_block(
        BlockResult(
            spec=pairwise_spec(),
            protids=["A", "B", "C"],
            distances=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        )
    )
    stored = store.read_block("tmscore")
    recorded = store.read_manifest("tmscore").derived["values_digest"]
    assert values_digest(stored.distances) == recorded


# ==========================================================================
# B6 -- the fusable protection was keyed on a user-chosen block id
# ==========================================================================


def _fuse(block_id, block_body):
    return MultispaceConfig.from_dict(
        {
            "blocks": {"tmscore": {"provider": "tmscore"}, block_id: block_body},
            "spaces": {"s": {"blocks": ["tmscore", block_id], "strategy": "late"}},
        }
    )


@pytest.mark.parametrize("block_id", ["tax", "lineage", "my_taxonomy_block"])
def test_b6_protection_follows_the_provider_not_the_block_name(block_id):
    """Was: keyed on the block id, so it protected `taxonomy:` and missed
    `tax:` -- protecting exactly the users who already knew about it.
    """
    with pytest.raises(NotFusableError, match="circular"):
        _fuse(block_id, {"provider": "uniprot_lineage"})


def test_b6_protection_is_case_insensitive_on_the_block_id():
    with pytest.raises(NotFusableError):
        _fuse("Taxonomy", {"provider": "some_unknown_provider"})


def test_b6_explicit_override_requires_a_written_justification():
    """Was: `fusable: true` silently defeated the protection."""
    with pytest.raises(ConfigError) as excinfo:
        _fuse("taxonomy", {"provider": "uniprot_lineage", "fusable": True})
    message = str(excinfo.value)
    assert "known overlay-only signal" in message
    assert "fusable_override_reason" in message


def test_b6_override_with_a_justification_is_allowed_and_recorded():
    """A maintainer may have a reason; it just has to be written down."""
    config = _fuse(
        "taxonomy",
        {
            "provider": "uniprot_lineage",
            "fusable": True,
            "fusable_override_reason": "deliberate positive control for the circularity check",
        },
    )
    block = config.blocks["taxonomy"]
    assert block.fusable
    assert "positive control" in block.params["fusable_override_reason"]


def test_b6_struclusters_provider_is_covered():
    with pytest.raises(NotFusableError, match="amino-acid identity"):
        _fuse("sc", {"provider": "struclusters"})


# ==========================================================================
# B7 -- the alignment_verified gate accepted any truthy value
# ==========================================================================


@pytest.mark.parametrize("value", ["false", "no", "0", 1, []])
def test_b7_non_boolean_alignment_verified_is_rejected(value):
    """Was: the string "false" is truthy, so it switched the gate OFF -- the
    opposite of what it says, on the gate standing between the user and a
    99.92%-wrong direct read.
    """
    with pytest.raises(ConfigError):
        MultispaceConfig.from_dict(
            {
                "blocks": {
                    "tmscore": {
                        "provider": "tmscore",
                        "representation": "direct",
                        "alignment_verified": value,
                    }
                },
                "spaces": {"structure": {"blocks": ["tmscore"]}},
            }
        )


def test_b7_real_true_still_opens_the_gate():
    config = MultispaceConfig.from_dict(
        {
            "blocks": {
                "tmscore": {
                    "provider": "tmscore",
                    "representation": "direct",
                    "alignment_verified": True,
                }
            },
            "spaces": {"structure": {"blocks": ["tmscore"]}},
        }
    )
    assert config.blocks["tmscore"].representation == "direct"


# ==========================================================================
# Notes -- lower severity, same origin
# ==========================================================================


def test_note_with_protids_refuses_to_relabel_with_a_different_length():
    result = BlockResult(spec=pairwise_spec(), protids=["A", "B", "C"], distances=np.zeros(3))
    with pytest.raises(BlockSpecError, match="renames rows in place"):
        result.with_protids(["A", "B"])


def test_note_with_protids_refuses_duplicates():
    result = BlockResult(spec=pairwise_spec(), protids=["A", "B", "C"], distances=np.zeros(3))
    with pytest.raises(BlockSpecError, match="doubles that protein's weight"):
        result.with_protids(["A", "A", "C"])


# ==========================================================================
# Design findings from the API-probe pass
# ==========================================================================


def test_asymmetric_pairwise_block_can_be_represented():
    """Was: the condensed-triangle shape check made symmetry a type-level
    assertion, so half of every directed measurement had to be discarded with
    no record that it had been.
    """
    values = np.array([[1.0, 0.8, 0.5], [0.7, 1.0, 0.4], [0.5, 0.45, 1.0]])
    result = BlockResult(
        spec=pairwise_spec(kind="pairwise_directed", symmetrization=None),
        protids=["A", "B", "C"],
        distances=values,
    )
    assert result.distances.shape == (3, 3)
    assert result.distances[0][1] != result.distances[1][0]


def test_symmetric_pairwise_must_declare_its_symmetrization():
    with pytest.raises(BlockSpecError) as excinfo:
        pairwise_spec(symmetrization=None)
    assert "modelling choice" in str(excinfo.value)


def test_directed_block_may_not_claim_a_symmetrization():
    with pytest.raises(BlockSpecError, match="contradictory"):
        pairwise_spec(kind="pairwise_directed", symmetrization="mean")


def test_absence_and_censoring_are_different_channels():
    """A predictor that has no value is not the same as a pair that lost a cut,
    and one anonymous mask could not tell them apart.
    """
    features = np.zeros((3, 2))
    absent = np.zeros((3, 2), dtype=bool)
    absent[2, :] = True
    result = BlockResult(
        spec=BlockSpec(
            id="deeploc",
            kind="features",
            fusable=True,
            metric="euclidean",
            normalization="zscore_within",
            provider="deeploc",
        ),
        protids=["A", "B", "C"],
        features=features,
        channels={"absent": absent},
    )
    assert result.absent_rate == pytest.approx(1 / 3)
    assert result.censoring_rate is None  # no claim made about censoring


def test_confidence_is_a_channel_not_a_feature_column():
    """Encoding confidence into the value would let low confidence move a point."""
    result = BlockResult(
        spec=BlockSpec(
            id="clean_ec",
            kind="features",
            fusable=True,
            metric="cosine",
            normalization="none",
            provider="clean",
        ),
        protids=["A", "B"],
        features=np.zeros((2, 4)),
        channels={"confidence": np.full((2, 4), 0.8)},
    )
    assert result.channel("confidence").mean() == pytest.approx(0.8)


def test_unknown_channel_names_are_rejected():
    with pytest.raises(BlockSpecError, match="unknown channel"):
        BlockResult(
            spec=pairwise_spec(),
            protids=["A", "B"],
            distances=np.zeros(1),
            channels={"vibes": np.zeros(1, dtype=bool)},
        )


def test_object_dtype_features_are_rejected_at_construction():
    """Was: accepted here and failed much later inside np.save."""
    ragged = np.empty((2, 1), dtype=object)
    ragged[0, 0] = ["PF00069", "PF00017"]
    ragged[1, 0] = ["PF00069"]
    with pytest.raises(BlockSpecError, match="floating-point array"):
        BlockResult(
            spec=BlockSpec(
                id="domains",
                kind="features",
                fusable=True,
                metric="jaccard",
                normalization="none",
                provider="domains",
            ),
            protids=["A", "B"],
            features=ragged,
        )


def test_unexplained_nan_is_rejected():
    features = np.zeros((2, 2))
    features[0, 0] = np.nan
    with pytest.raises(BlockSpecError, match="not covered by a 'censored' or 'absent'"):
        BlockResult(
            spec=BlockSpec(
                id="x",
                kind="features",
                fusable=True,
                metric="euclidean",
                normalization="none",
                provider="x",
            ),
            protids=["A", "B"],
            features=features,
        )


def test_declared_nan_is_accepted():
    features = np.zeros((2, 2))
    features[0, 0] = np.nan
    absent = np.zeros((2, 2), dtype=bool)
    absent[0, 0] = True
    result = BlockResult(
        spec=BlockSpec(
            id="x",
            kind="features",
            fusable=True,
            metric="euclidean",
            normalization="none",
            provider="x",
        ),
        protids=["A", "B"],
        features=features,
        channels={"absent": absent},
    )
    assert np.isnan(result.features[0, 0])


def test_channels_round_trip_through_the_store_with_their_names(tmp_path):
    store = BlockStore(str(tmp_path))
    censored = np.zeros((2, 2), dtype=bool)
    censored[0, 0] = True
    absent = np.zeros((2, 2), dtype=bool)
    absent[1, 1] = True
    store.write_block(
        BlockResult(
            spec=BlockSpec(
                id="multi",
                kind="features",
                fusable=True,
                metric="euclidean",
                normalization="none",
                provider="x",
            ),
            protids=["A", "B"],
            features=np.zeros((2, 2)),
            channels={"censored": censored, "absent": absent},
        )
    )
    loaded = store.read_block("multi")
    assert set(loaded.channels) == {"censored", "absent"}
    np.testing.assert_array_equal(loaded.channels["censored"], censored)
    np.testing.assert_array_equal(loaded.channels["absent"], absent)


# ==========================================================================
# GE.2 (blocks) -- a checkpoint output nothing consumes silently drops a rule
# ==========================================================================
#
# `download_pdbs` is a checkpoint. Adding `cohort_report.json` to its outputs
# looked additive and was not: on an output tree produced before this branch the
# structures directory exists and the report does not, and because *no job
# requests the report*, snakemake never re-runs the checkpoint to produce it.
# `checkpoints.download_pdbs.get()` then raises, `get_pdb_filepaths` contributes
# no `copy_pdb` job, and the run proceeds without the query proteins -- silently,
# not with an error. Reproduced in isolation against snakemake 7.25.3: the same
# tree plans 3 jobs including `copy_pdb` with a single-output checkpoint and 2
# jobs without, and the surviving job receives the directory rather than the
# file list.
#
# The fix is to declare each new output only when the rule that reads it is in
# the DAG, so it is never an orphan. These tests assert that, not the prose.


def _planned_outputs(config_path):
    """The output-file set snakemake plans for a config, via `--summary`."""
    completed = subprocess.run(
        [
            "snakemake",
            "--configfile",
            str(config_path),
            "--summary",
            "--rerun-incomplete",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        pytest.fail(f"snakemake --summary failed:\n{completed.stdout}\n{completed.stderr}")
    rows = completed.stdout.splitlines()[1:]
    return {row.split("\t")[0] for row in rows if "\t" in row}


def _write_config(tmp_path, name, extra):
    body = {
        "mode": "search",
        "analysis_name": "ge2",
        "input_dir": "demo/search-mode/input/",
        "output_dir": str(tmp_path / "out") + "/",
        "max_blast_hits": 10,
        "max_foldseek_hits": 10,
        "max_structures": 10,
        "plotting_modes": ["pca_umap"],
    }
    body.update(extra)
    path = tmp_path / name
    # JSON is valid YAML, and no environment here has PyYAML alongside the rest
    # of the stack -- the Snakefile reads either.
    path.write_text(json.dumps(body))
    return path


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_the_default_search_path_declares_neither_new_output(tmp_path):
    """The regression itself. Both files are absent from the legacy plan."""
    planned = _planned_outputs(_write_config(tmp_path, "legacy.json", {}))
    assert planned, "the summary returned no rows, so this test proves nothing"
    orphans = [p for p in planned if p.endswith(("cohort_report.json", ".mapping.tsv"))]
    assert orphans == [], (
        f"{orphans} are declared on the default path but no rule consumes them. "
        "An unconsumed output on the `download_pdbs` checkpoint drops `copy_pdb` "
        "on any pre-existing output tree -- see REVIEW_LOG GE.2."
    )


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_the_cohort_report_appears_exactly_when_a_space_reads_it(tmp_path):
    spaces = {
        "blocks": {"tmscore": {"provider": "tmscore", "representation": "profile"}},
        "spaces": {"legacy": {"blocks": ["tmscore"], "strategy": "none", "reducers": ["pca"]}},
    }
    planned = _planned_outputs(_write_config(tmp_path, "spaces.json", spaces))
    assert any(p.endswith("cohort_report.json") for p in planned), (
        "`diagnose_space` reads the cohort report, so enabling spaces must declare it. "
        "If this fails the report is never written and the diagnostic loses its input."
    )


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_the_refseq_mapping_appears_exactly_when_significance_reads_it(tmp_path):
    significance = {"cohort": {"selection": "significance"}}
    planned = _planned_outputs(_write_config(tmp_path, "significance.json", significance))
    assert any(p.endswith(".mapping.tsv") for p in planned), (
        "`aggregate_hit_significance` reads the RefSeq mapping, so significance "
        "selection must declare it."
    )
