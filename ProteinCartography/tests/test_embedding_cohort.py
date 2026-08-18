#!/usr/bin/env python
"""The embedding fixture's own tests.

These assert the *construction*, not any statistic computed from it. Nothing in
`diagnostics/` exists yet, deliberately: group 7b's fixture caught two generator
defects before the statistic it was built for was written, and both would
otherwise have surfaced later as "the statistic missed the signal"
for the same reason.

The two that matter are :func:`test_the_fold_moves_no_pair_apart` and
:func:`test_the_split_moves_no_pair_together`. Between them they prove the
planted 2x2 answer is a property of the geometry rather than a label somebody
typed into a dataclass -- a reflection can only bring points together, and a
rigid translation of a subset can only push them apart.
"""

from __future__ import annotations

import numpy as np
import pytest
from embedding_cohort import (
    DEFAULT_N,
    FOLD,
    HIGH_DIMENSIONS,
    ISOMETRIC,
    SHUFFLE,
    SPLIT,
    SPLIT_OFFSET,
    EmbeddingCohort,
    embedding_cohort,
)

CASES = (ISOMETRIC, FOLD, SPLIT, SHUFFLE)


@pytest.fixture(scope="module")
def cohort() -> EmbeddingCohort:
    return embedding_cohort()


def _distances(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


# --- determinism ------------------------------------------------------------


def test_the_same_seed_gives_the_same_cohort():
    a, b = embedding_cohort(seed=3), embedding_cohort(seed=3)
    assert a.protids == b.protids
    np.testing.assert_array_equal(a.high, b.high)
    for name in CASES:
        np.testing.assert_array_equal(a.case(name).low, b.case(name).low)


def test_a_different_seed_gives_a_different_cohort():
    a, b = embedding_cohort(seed=0), embedding_cohort(seed=1)
    assert not np.allclose(a.high, b.high)


def test_n_below_twelve_is_refused():
    with pytest.raises(ValueError, match="at least 12"):
        embedding_cohort(n=8)


# --- the truth and its rotation ---------------------------------------------


def test_the_cohort_has_the_advertised_shape(cohort):
    assert cohort.n_proteins == DEFAULT_N
    assert len(cohort.protids) == DEFAULT_N
    assert len(set(cohort.protids)) == DEFAULT_N
    assert cohort.truth.shape == (DEFAULT_N, 2)
    assert cohort.high.shape == (DEFAULT_N, HIGH_DIMENSIONS)


def test_the_rotation_into_high_dimensions_changes_no_distance(cohort):
    """The whole fixture rests on this. If the rotation is not an isometry,
    `isometric` is not isometric and its exact 1.0 means nothing."""
    error = np.abs(_distances(cohort.high) - _distances(cohort.truth)).max()
    assert error < 1e-12, f"rotation moved points by up to {error:.2e}"


def test_the_high_dimensional_truth_is_really_two_dimensional(cohort):
    """20 columns spanning a 2-D subspace, which is what makes a faithful 2-D
    map possible at all."""
    assert np.linalg.matrix_rank(cohort.high - cohort.high.mean(axis=0)) == 2


def test_no_two_distances_are_exactly_tied(cohort):
    """Generic position, asserted rather than hoped for.

    Ties make a k-nearest set depend on how the sort breaks them. On an earlier
    integer-lattice draft that cost the `isometric` case its exact 1.0 and
    produced a 2.6e-3 disagreement with scikit-learn that looked like a formula
    error and was not.
    """
    for values in (cohort.high, *(cohort.case(name).low for name in CASES)):
        # Each unordered pair appears twice off the diagonal, so with nothing
        # tied the number of distinct values is exactly the number of pairs.
        n = values.shape[0]
        distinct = len(np.unique(_off_diagonal(_distances(values))))
        assert distinct == n * (n - 1) // 2, f"{n * (n - 1) // 2 - distinct} tied pair(s)"


# --- the planted 2x2 --------------------------------------------------------


def test_every_cell_of_the_two_by_two_is_occupied(cohort):
    """The reason the fixture exists. A diagnostic that computes one statistic
    twice can only produce the diagonal of this table."""
    table = {(c.trustworthy, c.continuous) for c in cohort.cases.values()}
    assert table == {(True, True), (False, True), (True, False), (False, False)}


def test_each_case_carries_a_description(cohort):
    for name in CASES:
        assert cohort.case(name).description
        assert cohort.case(name).name == name


def test_only_the_isometric_case_is_faithful(cohort):
    faithful = [name for name in CASES if cohort.case(name).faithful]
    assert faithful == [ISOMETRIC]


# --- what each layout does to the geometry ----------------------------------


def test_the_isometric_case_is_the_truth(cohort):
    np.testing.assert_array_equal(cohort.case(ISOMETRIC).low, cohort.truth)


def test_the_fold_moves_no_pair_apart(cohort):
    """A reflection is a contraction, so `fold` cannot break continuity.

    ``| |a| - |b| | <= |a - b|`` for every pair, with equality exactly when a
    and b share a sign. That is why `fold`'s planted answer is "untrustworthy
    but continuous" rather than an empirical observation about one seed: no
    true neighbor can be separated by an operation that moves nothing apart.
    """
    before, after = _distances(cohort.truth), _distances(cohort.case(FOLD).low)
    assert (after <= before + 1e-12).all()


def test_the_fold_brings_distant_points_into_contact(cohort):
    """The other half of `fold`: it must actually invent neighbors, or its
    planted "untrustworthy" is a claim about nothing."""
    before, after = _distances(cohort.truth), _distances(cohort.case(FOLD).low)
    contracted = _off_diagonal(before) - _off_diagonal(after)
    assert contracted.max() > 1.0, "the reflection barely moved anything"
    # Pairs that were in the far half of the distance distribution and are now
    # in the near half -- the false neighbors trustworthiness has to catch.
    far = _off_diagonal(before) > np.median(_off_diagonal(before))
    near_now = _off_diagonal(after) < np.percentile(_off_diagonal(after), 10)
    assert (far & near_now).sum() > 0


def test_the_split_moves_no_pair_together(cohort):
    """A rigid translation of one half cannot invent a neighbor.

    This is the mirror of the fold's contraction property, and it is why
    `split`'s planted answer is "trustworthy but discontinuous". Within-half
    distances are preserved *exactly* -- not approximately -- and every
    cross-half distance grows.
    """
    before, after = _distances(cohort.truth), _distances(cohort.case(SPLIT).low)
    assert (after >= before - 1e-12).all()


def test_the_split_is_exactly_a_translation_of_each_half(cohort):
    """Bitwise exact, and this is the claim that carries the fixture's meaning.

    Rigidity is a property of the construction: every protein's displacement is
    one of exactly two vectors. Distances *recomputed* from the translated
    coordinates are a different matter -- see the next test.
    """
    low = cohort.case(SPLIT).low
    displacement = low - cohort.truth
    distinct = np.unique(displacement, axis=0)
    assert distinct.shape == (2, 2)
    np.testing.assert_array_equal(distinct, np.array([[0.0, 0.0], [SPLIT_OFFSET, 0.0]]))


def test_the_split_preserves_within_half_distances(cohort):
    """To 1e-11 relative, and the residual is measurement, not construction.

    ``(a + 500) - (b + 500)`` is not ``a - b`` in floating point when a and b
    are of order 1: the offset is added exactly and cancels inexactly. Measured
    residual is 8.1e-13 relative, five orders below anything a rank statistic
    can see, and it is the recomputation that carries it rather than the
    layout. The previous test is the exact one.
    """
    low = cohort.case(SPLIT).low
    # Recover the halves from the layout: the offset is far larger than the
    # cloud, so the x coordinate separates them cleanly.
    half = low[:, 0] > SPLIT_OFFSET / 2
    before, after = _distances(cohort.truth), _distances(low)
    for members in (half, ~half):
        assert members.sum() > 1
        np.testing.assert_allclose(
            after[np.ix_(members, members)],
            before[np.ix_(members, members)],
            rtol=1e-11,
            atol=1e-11,
        )


def test_the_split_separates_the_halves_beyond_every_retained_neighbor(cohort):
    """Every cross-half distance exceeds every within-half distance, so a
    displaced true neighbor is ranked behind all of the retained ones."""
    low = cohort.case(SPLIT).low
    half = low[:, 0] > SPLIT_OFFSET / 2
    after = _distances(low)
    within = np.concatenate(
        [
            _off_diagonal(after[np.ix_(half, half)]),
            _off_diagonal(after[np.ix_(~half, ~half)]),
        ]
    )
    between = after[np.ix_(half, ~half)].ravel()
    assert between.min() > within.max()


def test_the_split_scatters_each_protein_s_neighbors_across_both_halves(cohort):
    """The mechanism behind the continuity collapse, stated as a number.

    A split that happened to keep every protein with its neighbors would leave
    continuity untouched, and the planted answer would be wrong without any
    test noticing.
    """
    low = cohort.case(SPLIT).low
    half = low[:, 0] > SPLIT_OFFSET / 2
    truth_distances = _distances(cohort.truth)
    np.fill_diagonal(truth_distances, np.inf)
    neighbors = np.argsort(truth_distances, axis=1, kind="stable")[:, :10]
    same_half = (half[neighbors] == half[:, None]).mean()
    assert 0.4 < same_half < 0.6, f"{same_half:.2f} of true neighbors stayed"


def test_the_shuffle_is_a_permutation_of_the_truth(cohort):
    """Same points, different assignment. The point cloud is unchanged, so the
    map is as pretty as the truth and carries none of it."""
    shuffled = cohort.case(SHUFFLE).low
    order = np.lexsort((cohort.truth[:, 1], cohort.truth[:, 0]))
    shuffled_order = np.lexsort((shuffled[:, 1], shuffled[:, 0]))
    np.testing.assert_array_equal(cohort.truth[order], shuffled[shuffled_order])


def test_the_shuffle_actually_moves_proteins(cohort):
    moved = (cohort.case(SHUFFLE).low != cohort.truth).any(axis=1).mean()
    assert moved > 0.9


# --- the shapes the pipeline consumes ---------------------------------------


def test_the_block_result_carries_the_high_dimensional_truth(cohort):
    result = cohort.block_result()
    assert result.protids == cohort.protids
    assert result.features.shape == (DEFAULT_N, HIGH_DIMENSIONS)
    assert result.spec.id == "truth"
    assert result.spec.kind == "features"


def test_the_embedding_frame_matches_what_reduce_space_writes(cohort):
    frame = cohort.embedding_frame(ISOMETRIC)
    assert frame.index.name == "protid"
    assert list(frame.index) == cohort.protids
    assert list(frame.columns) == ["UMAP1", "UMAP2"]


def test_the_embedding_round_trips_through_a_file(cohort, tmp_path):
    import pandas as pd

    path = cohort.write_embedding(tmp_path / "embedding_umap.tsv", SPLIT)
    reloaded = pd.read_csv(path, sep="\t", index_col=0)
    assert list(reloaded.index) == cohort.protids
    np.testing.assert_allclose(reloaded.to_numpy(), cohort.case(SPLIT).low)
