#!/usr/bin/env python
"""Per-space Leiden, and the one test that justifies not reimplementing it.

Every other numeric component in this work is hand-written numpy checked
against a reference. This one *is* the reference -- it calls the same scanpy
that ``rule leiden_clustering`` calls -- so the check that replaces a
cross-implementation comparison is a cross-*path* one:
``test_a_space_and_the_legacy_path_return_the_same_partition`` requires this
module and ``leiden_clustering.scanpy_leiden_cluster`` to label the same matrix
identically. If they ever drift, a diagnostic about "the partition" stops being
about the partition the pipeline ships, which is the failure ADR 0015 is
written against.

Everything that does not need scanpy is tested without it, so the bare
environment still exercises the availability contract, the argument checking
and the small-N short circuit.
"""

from __future__ import annotations

import numpy as np
import pytest
from clustering import (
    DEFAULT_N_NEIGHBORS,
    DEFAULT_N_PCS,
    ClusteringError,
    Partition,
    is_available,
    leiden_partition,
    sweep_resolutions,
)

AVAILABLE, EXPLANATION = is_available()
needs_scanpy = pytest.mark.skipif(not AVAILABLE, reason=EXPLANATION)


# --- the availability contract, testable everywhere --------------------------


def test_is_available_returns_a_flag_and_an_explanation():
    available, explanation = is_available()
    assert isinstance(available, bool)
    assert isinstance(explanation, str) and explanation


def test_the_explanation_names_the_environment_that_has_it():
    """ADR 0006 rule 2: the string says what is missing and how to get it."""
    available, explanation = is_available()
    if not available:
        assert "envs/analysis.yml" in explanation


@pytest.mark.skipif(AVAILABLE, reason="scanpy is installed here")
def test_clustering_without_scanpy_raises_rather_than_returning_nonsense():
    with pytest.raises(ClusteringError, match="scanpy"):
        leiden_partition(np.zeros((10, 4)), [f"P{i}" for i in range(10)])


# --- the small-N short circuit, which needs no scanpy ------------------------


@pytest.mark.parametrize("n", [0, 1, 2])
def test_below_three_proteins_everything_goes_in_one_cluster(n):
    """What ``leiden_clustering`` does, so what the agreement requires."""
    result = leiden_partition(np.zeros((n, 3)), [f"P{i}" for i in range(n)])
    assert result.labels == ["LC0"] * n
    assert result.n_clusters == (1 if n else 0)


def test_a_feature_matrix_with_the_wrong_number_of_rows_is_refused():
    with pytest.raises(ClusteringError, match="one per protid"):
        leiden_partition(np.zeros((5, 3)), ["A", "B", "C"])


def test_a_non_positive_resolution_is_refused():
    with pytest.raises(ClusteringError, match="resolution must be positive"):
        leiden_partition(np.zeros((10, 3)), [f"P{i}" for i in range(10)], resolution=0.0)


def test_the_partition_maps_protids_to_labels():
    partition = Partition(["A", "B"], ["LC0", "LC1"], 1.0, 10, 30, 0)
    assert partition.as_mapping() == {"A": "LC0", "B": "LC1"}
    assert partition.n_clusters == 2


# --- the agreement test, which is the argument -------------------------------


@needs_scanpy
def test_a_space_and_the_legacy_path_return_the_same_partition(tmp_path):
    """Identical labels, not merely a high ARI.

    ``leiden_clustering.scanpy_leiden_cluster`` reads a similarity matrix from
    disk; this module takes a space's features in memory. They are two call
    sites for one algorithm, and the only thing keeping them one algorithm is
    this assertion -- ``clustering._clamped`` duplicates the legacy clamping
    rather than importing it, because importing it would pull scanpy in at
    module scope.
    """
    import pandas as pd
    from leiden_clustering import scanpy_leiden_cluster
    from parity import synthetic_matrix

    matrix = synthetic_matrix(tmp_path / "matrix.tsv", n=250)
    legacy = scanpy_leiden_cluster(str(matrix))
    frame = pd.read_csv(matrix, sep="\t", index_col=0)

    mine = leiden_partition(frame.to_numpy(dtype=float), list(frame.index))
    by_protid = dict(zip(legacy["protid"], legacy["LeidenCluster"]))
    assert [by_protid[p] for p in mine.protids] == mine.labels


@needs_scanpy
@pytest.mark.slow
def test_the_two_paths_still_agree_above_five_hundred_proteins(tmp_path):
    """N > 500 is where the reducers' solver switches; the graph build is a
    different code path in scanpy too, so the agreement is checked on both
    sides of it rather than assumed to carry."""
    import pandas as pd
    from leiden_clustering import scanpy_leiden_cluster
    from parity import synthetic_matrix

    matrix = synthetic_matrix(tmp_path / "matrix.tsv", n=750)
    legacy = scanpy_leiden_cluster(str(matrix))
    frame = pd.read_csv(matrix, sep="\t", index_col=0)
    mine = leiden_partition(frame.to_numpy(dtype=float), list(frame.index))
    by_protid = dict(zip(legacy["protid"], legacy["LeidenCluster"]))
    assert [by_protid[p] for p in mine.protids] == mine.labels


# --- behavior on a cohort with a planted answer ------------------------------


@needs_scanpy
def test_it_recovers_a_planted_partition():
    """A clusterer returns clusters whatever it is given, so "it ran" is not
    evidence. ``fusion_cohort``'s `wide` block was built to show `fold` and to
    be blind to `chemistry`; recovering the first and not the second is the
    only result that cannot be produced by accident."""
    from diagnostics.partition import adjusted_rand_index
    from fusion_cohort import WIDE_BLOCK, fusion_cohort

    cohort = fusion_cohort()
    partition = leiden_partition(cohort.values(WIDE_BLOCK), cohort.protids)
    assert adjusted_rand_index(partition.labels, cohort.partitions["fold"].labels) > 0.9
    assert abs(adjusted_rand_index(partition.labels, cohort.partitions["chemistry"].labels)) < 0.1


@needs_scanpy
def test_the_same_seed_gives_the_same_partition():
    """Leiden is stochastic. A diagnostic that moved between two runs of one
    input would be indistinguishable from the instability it measures."""
    from fusion_cohort import WIDE_BLOCK, fusion_cohort

    cohort = fusion_cohort()
    first = leiden_partition(cohort.values(WIDE_BLOCK), cohort.protids)
    second = leiden_partition(cohort.values(WIDE_BLOCK), cohort.protids)
    assert first.labels == second.labels


@needs_scanpy
def test_a_higher_resolution_never_returns_fewer_clusters():
    from fusion_cohort import WIDE_BLOCK, fusion_cohort

    cohort = fusion_cohort()
    counts = [
        leiden_partition(cohort.values(WIDE_BLOCK), cohort.protids, resolution=r).n_clusters
        for r in (0.2, 0.5, 1.0, 2.0, 4.0)
    ]
    assert counts == sorted(counts), counts
    assert counts[-1] > counts[0], counts


@needs_scanpy
def test_the_clamped_settings_are_recorded_not_the_requested_ones():
    """A run at N=20 that asked for 30 principal components and got 19 should
    say so, for the reason ``ReducerResult.params_used`` exists."""
    from fusion_cohort import NARROW_BLOCK, fusion_cohort

    cohort = fusion_cohort(n=24)
    partition = leiden_partition(cohort.values(NARROW_BLOCK), cohort.protids)
    assert partition.n_pcs < DEFAULT_N_PCS
    assert partition.n_neighbors >= DEFAULT_N_NEIGHBORS


@needs_scanpy
def test_sweep_resolutions_is_keyed_and_ordered_by_resolution():
    from fusion_cohort import WIDE_BLOCK, fusion_cohort

    cohort = fusion_cohort()
    swept = sweep_resolutions(cohort.values(WIDE_BLOCK), cohort.protids, [2.0, 0.5, 1.0])
    assert list(swept) == [0.5, 1.0, 2.0]
    assert all(len(labels) == cohort.n_proteins for labels in swept.values())


# ==========================================================================
# The sidecar manifest. Four archived leiden_features.tsv files in this project
# could not be regenerated from their own inputs and none carried any record of
# the settings that made them, so the disagreement could only be observed, not
# diagnosed (FOLLOWUPS #78).
# ==========================================================================


@needs_scanpy
def test_the_manifest_records_the_parameters_actually_used(tmp_path):
    """The USED values, not the requested ones.

    `n_neighbors` is raised to `round(n/10)` and both are clamped to the matrix,
    so two runs invoked identically on different-sized cohorts do different
    things (FOLLOWUPS #79). A manifest echoing the request would have recorded
    nothing worth having.
    """
    import json

    from leiden_clustering import scanpy_leiden_cluster
    from parity import synthetic_matrix

    matrix = synthetic_matrix(tmp_path / "matrix.tsv", n=250)
    out = tmp_path / "leiden_features.tsv"
    manifest = tmp_path / "leiden_features.manifest.json"
    scanpy_leiden_cluster(str(matrix), str(out), manifest_path=str(manifest))

    assert manifest.exists(), "no manifest was written"
    payload = json.loads(manifest.read_text())
    assert payload["requested"]["n_neighbors"] == 10
    # n/10 for a 250-protein matrix is 25, which is what actually ran.
    assert payload["used"]["n_neighbors"] == 25, payload["used"]
    assert payload["n_proteins"] == 250
    assert payload["n_clusters"] >= 1
    assert payload["input"]["sha256"], "the input is not pinned by digest"
    assert payload["versions"]["scanpy"], "the library that decides the answer is unrecorded"


@needs_scanpy
def test_the_manifest_names_the_resolution_nobody_passes(tmp_path):
    """`sc.tl.leiden` is called with no resolution, so it takes scanpy's default.

    That is the parameter most likely to differ between two runs that look
    identical from the command line, and the one that would have explained the
    archives. Recording that it was NOT set is the honest entry.
    """
    import json

    from leiden_clustering import scanpy_leiden_cluster
    from parity import synthetic_matrix

    matrix = synthetic_matrix(tmp_path / "matrix.tsv", n=60)
    manifest = tmp_path / "m.json"
    scanpy_leiden_cluster(str(matrix), str(tmp_path / "o.tsv"), manifest_path=str(manifest))
    payload = json.loads(manifest.read_text())
    assert "default" in payload["resolution"].lower()
    assert "passes none" in payload["resolution"]


@needs_scanpy
def test_no_manifest_is_written_unless_one_is_asked_for(tmp_path):
    """This script runs on the DEFAULT output path.

    Writing a file unconditionally would add an output to the default DAG, and
    the parity suite would see it as a differing file. Opt-in is what keeps the
    default path byte-identical.
    """
    from leiden_clustering import scanpy_leiden_cluster
    from parity import synthetic_matrix

    matrix = synthetic_matrix(tmp_path / "matrix.tsv", n=60)
    scanpy_leiden_cluster(str(matrix), str(tmp_path / "o.tsv"))
    assert not list(tmp_path.glob("*.json")), "a manifest appeared without being asked for"


def test_the_snakefile_does_not_pass_manifest_on_the_default_path():
    """Pinned against the Snakefile, so the parity guarantee is not just a habit."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    snakefile = (root / "Snakefile").read_text()
    block = snakefile[snakefile.index("leiden_clustering.py") :][:600]
    assert "--manifest" not in block, (
        "the default DAG now writes a Leiden manifest, which adds an output and "
        "will show up in parity as a differing file"
    )
