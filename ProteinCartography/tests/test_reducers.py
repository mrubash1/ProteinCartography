"""Tests for the shared reducer core.

These need scikit-learn and umap-learn, which the bare test environment does not
have -- see ``test_optional_dependencies.py``. They are skipped there and run in
the analysis environment.

The point of this module is that `dim_reduction.py` and the multi-space
machinery call the *same* functions, so the tests that matter most are the ones
asserting the numerics and the column naming match what the legacy script has
always produced. Renaming a column here renames a column in
`aggregated_features.tsv`.
"""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from spaces.reducers.core import (  # noqa: E402
    ReducerResult,
    reduce_pca,
    reduce_tsne,
    reduce_umap,
    small_n_layout,
)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    return rng.uniform(0.2, 0.9, size=(30, 30))


@pytest.fixture
def protids():
    return [f"P{i:03d}" for i in range(30)]


# --------------------------------------------------------------------------
# naming -- downstream reads these positionally, so they are load-bearing
# --------------------------------------------------------------------------


def test_pca_columns_are_zero_based(data, protids):
    """PC0, not PC1. Legacy naming; changing it renames a shipped column."""
    result = reduce_pca(data, protids, n_components=3)
    assert result.column_names == ["PC0", "PC1", "PC2"]


def test_umap_and_tsne_columns_are_one_based(data, protids):
    assert reduce_umap(data, protids).column_names == ["UMAP1", "UMAP2"]
    assert reduce_tsne(data, protids).column_names == ["tSNE1", "tSNE2"]


# --------------------------------------------------------------------------
# determinism -- the property PR #106 added
# --------------------------------------------------------------------------


def test_pca_is_reproducible(data, protids):
    a = reduce_pca(data, protids, n_components=5, random_state=123456)
    b = reduce_pca(data, protids, n_components=5, random_state=123456)
    np.testing.assert_array_equal(a.coordinates, b.coordinates)


def test_pca_uses_the_exact_solver(data, protids):
    """`auto` switches to a randomized solver above 500 rows or columns and draws
    from the global numpy state, which is what made large maps irreproducible.
    """
    result = reduce_pca(data, protids, n_components=2)
    assert result.params_used["svd_solver"] == "full"


def test_umap_is_reproducible(data, protids):
    pytest.importorskip("umap")
    a = reduce_umap(data, protids, random_state=123456)
    b = reduce_umap(data, protids, random_state=123456)
    np.testing.assert_allclose(a.coordinates, b.coordinates)


def test_tsne_is_reproducible(data, protids):
    a = reduce_tsne(data, protids, random_state=123456, n_iter=250)
    b = reduce_tsne(data, protids, random_state=123456, n_iter=250)
    np.testing.assert_allclose(a.coordinates, b.coordinates)


# --------------------------------------------------------------------------
# clamping -- the PR #103 small-N behavior, preserved exactly
# --------------------------------------------------------------------------


def test_pca_clamps_components_to_the_matrix(protids):
    values = np.random.RandomState(0).uniform(size=(5, 5))
    result = reduce_pca(values, protids[:5], n_components=30)
    assert result.coordinates.shape[1] == 5
    assert result.params_used["n_components_requested"] == 30
    assert result.params_used["n_components_used"] == 5


def test_umap_clamps_neighbours_below_n(protids):
    pytest.importorskip("umap")
    values = np.random.RandomState(0).uniform(size=(6, 6))
    result = reduce_umap(values, protids[:6], n_neighbors=80)
    assert result.params_used["n_neighbors_requested"] == 80
    assert result.params_used["n_neighbors_used"] == 5


def test_umap_falls_back_below_three_points(protids):
    values = np.random.RandomState(0).uniform(size=(2, 4))
    result = reduce_umap(values, protids[:2], n_components=2)
    assert result.coordinates.shape == (2, 2)
    assert result.column_names == ["UMAP1", "UMAP2"]
    assert result.params_used["reducer"] == "small_n_layout"


def test_umap_single_point_gives_the_origin(protids):
    result = reduce_umap(np.array([[1.0, 2.0]]), protids[:1], n_components=2)
    np.testing.assert_array_equal(result.coordinates, np.zeros((1, 2)))


def test_tsne_below_two_points_is_degenerate(protids):
    result = reduce_tsne(np.array([[1.0, 2.0]]), protids[:1])
    np.testing.assert_array_equal(result.coordinates, np.zeros((1, 2)))
    assert result.params_used["degenerate"]


def test_tsne_clamps_perplexity_only_for_the_sklearn_constraint(protids):
    """It must not scale perplexity for every small N.

    Doing so would move ordinary maps -- N=100 would go from 50 to 21.
    """
    values = np.random.RandomState(0).uniform(size=(20, 20))
    result = reduce_tsne(values, protids[:20], perplexity=50, n_iter=250)
    assert result.params_used["perplexity_used"] == 19.0

    bigger = np.random.RandomState(0).uniform(size=(100, 20))
    result = reduce_tsne(bigger, [f"Q{i}" for i in range(100)], perplexity=50, n_iter=250)
    assert result.params_used["perplexity_used"] == 50


def test_small_n_layout_pads_to_the_requested_width(protids):
    values = np.random.RandomState(0).uniform(size=(2, 1))
    result = small_n_layout(values, protids[:2], 3, "UMAP")
    assert result.coordinates.shape == (2, 3)
    assert result.column_names == ["UMAP1", "UMAP2", "UMAP3"]


def test_empty_matrix_raises(protids):
    with pytest.raises(ValueError, match="empty"):
        reduce_pca(np.zeros((0, 0)), [], n_components=2)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_params_used_records_post_clamp_values(protids):
    """Two runs with the same config and different N are not the same run."""
    pytest.importorskip("umap")
    small = reduce_umap(np.random.RandomState(0).uniform(size=(6, 6)), protids[:6])
    large = reduce_umap(np.random.RandomState(0).uniform(size=(30, 30)), protids)
    assert small.params_used["n_neighbors_used"] != large.params_used["n_neighbors_used"]
    assert (
        small.params_used["n_neighbors_requested"] == (large.params_used["n_neighbors_requested"])
    )


def test_result_rejects_a_protid_count_mismatch():
    with pytest.raises(ValueError, match="protids"):
        ReducerResult(coordinates=np.zeros((3, 2)), protids=["A", "B"], column_names=["x", "y"])


def test_result_rejects_a_column_name_mismatch():
    with pytest.raises(ValueError, match="column names"):
        ReducerResult(coordinates=np.zeros((2, 2)), protids=["A", "B"], column_names=["x"])


def test_to_frame_is_labeled(data, protids):
    frame = reduce_pca(data, protids, n_components=2).to_frame()
    assert list(frame.index) == protids
    assert frame.index.name == "protid"
    assert list(frame.columns) == ["PC0", "PC1"]
