"""The fusion fixture has to be right before anything is tested against it.

Group 7b's fixture tests caught two generator defects -- planted labels that did
not match the padding rule, and a recorded rate that was conditional rather than
marginal -- both before any statistic existed, and both of which would have
presented later as "the statistic missed the signal". The same
reasoning applies here and the stakes are higher: every claim this fixture
supports is a claim about *which block* a fused geometry came from, and if the
two partitions are not really independent then every one of those claims is
measuring the fixture.

So this file asserts the construction, not the statistics:

* the two partitions are exactly balanced and exactly independent;
* each structured block shows its own partition and is blind to the other;
* the scale gap between `wide` and `narrow` is real and large;
* the four pathology blocks are each pathological in the specific way claimed.
"""

from __future__ import annotations

import numpy as np
import pytest
from fusion_cohort import (
    CHEMISTRY_GROUPS,
    CONSTANT_COLUMN_BLOCK,
    DEGENERATE_BLOCK,
    FOLD_GROUPS,
    NARROW_BLOCK,
    NOISE_BLOCK,
    RESCALE_FACTOR,
    RESCALED_BLOCK,
    WIDE_BLOCK,
    fusion_cohort,
    separation,
)


@pytest.fixture(scope="module")
def cohort():
    return fusion_cohort()


# -- reproducibility --------------------------------------------------------


def test_the_cohort_is_reproducible_from_the_seed_alone():
    a, b = fusion_cohort(), fusion_cohort()
    assert a.protids == b.protids
    for block_id, values in a.blocks.items():
        np.testing.assert_array_equal(values, b.blocks[block_id])
    for name, planted in a.partitions.items():
        np.testing.assert_array_equal(planted.labels, b.partitions[name].labels)


def test_a_different_seed_gives_a_different_cohort():
    a, b = fusion_cohort(seed=0), fusion_cohort(seed=1)
    assert not np.array_equal(a.blocks[WIDE_BLOCK], b.blocks[WIDE_BLOCK])
    # The protid list is positional, so it must NOT depend on the seed -- two
    # cohorts at different seeds still describe the same slots.
    assert a.protids == b.protids


def test_an_n_that_cannot_cross_exactly_is_refused():
    with pytest.raises(ValueError, match="multiple of 12"):
        fusion_cohort(n=250)


# -- the partitions ---------------------------------------------------------


def test_both_partitions_are_exactly_balanced(cohort):
    fold = cohort.partitions["fold"].labels
    chemistry = cohort.partitions["chemistry"].labels
    assert sorted(np.bincount(fold)) == [cohort.n_proteins // FOLD_GROUPS] * FOLD_GROUPS
    assert (
        sorted(np.bincount(chemistry)) == [cohort.n_proteins // CHEMISTRY_GROUPS] * CHEMISTRY_GROUPS
    )


def test_the_partitions_are_exactly_independent(cohort):
    """Every (fold, chemistry) cell holds the same count.

    Independence in expectation is not enough. If the contingency table were
    lopsided, a fusion that recovered `fold` would score above chance on
    `chemistry` too, and the tests asserting that the second partition stays
    null would be measuring the draw rather than the construction.
    """
    fold = cohort.partitions["fold"].labels
    chemistry = cohort.partitions["chemistry"].labels
    table = np.zeros((FOLD_GROUPS, CHEMISTRY_GROUPS), dtype=int)
    for f, c in zip(fold, chemistry):
        table[f, c] += 1
    expected = cohort.n_proteins // (FOLD_GROUPS * CHEMISTRY_GROUPS)
    assert (table == expected).all(), table


def test_each_partition_names_the_block_built_to_show_it(cohort):
    assert cohort.partitions["fold"].block_id == WIDE_BLOCK
    assert cohort.partitions["chemistry"].block_id == NARROW_BLOCK


# -- what each block does and does not carry --------------------------------


def test_the_wide_block_shows_fold_and_is_blind_to_chemistry(cohort):
    assert cohort.separation(WIDE_BLOCK, "fold") > 2.0
    assert cohort.separation(WIDE_BLOCK, "chemistry") == pytest.approx(1.0, abs=0.05)


def test_the_narrow_block_shows_chemistry_and_is_blind_to_fold(cohort):
    assert cohort.separation(NARROW_BLOCK, "chemistry") > 2.0
    assert cohort.separation(NARROW_BLOCK, "fold") == pytest.approx(1.0, abs=0.05)


def test_the_two_blocks_separate_their_partitions_about_equally_well(cohort):
    """Comparable information, incommensurate units.

    If `wide` separated `fold` far better than `narrow` separates `chemistry`,
    then a fusion favouring `wide` could be defended as favouring the better
    block, and the fixture would no longer isolate scale. The separations are
    ratios, so they are comparable across the four-order-of-magnitude scale gap
    the very next test asserts.
    """
    wide = cohort.separation(WIDE_BLOCK, "fold")
    narrow = cohort.separation(NARROW_BLOCK, "chemistry")
    assert 0.5 < wide / narrow < 2.0, (wide, narrow)


def test_the_scale_gap_is_real_and_large(cohort):
    def mean_distance(values):
        d = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
        return float(d[~np.eye(len(values), dtype=bool)].mean())

    wide = mean_distance(cohort.blocks[WIDE_BLOCK])
    narrow = mean_distance(cohort.blocks[NARROW_BLOCK])
    assert wide / narrow > 1000.0, (wide, narrow)


def test_the_noise_block_carries_neither_partition(cohort):
    assert cohort.separation(NOISE_BLOCK, "fold") == pytest.approx(1.0, abs=0.05)
    assert cohort.separation(NOISE_BLOCK, "chemistry") == pytest.approx(1.0, abs=0.05)


# -- the pathologies --------------------------------------------------------


def test_the_rescaled_block_is_exactly_a_constant_multiple(cohort):
    np.testing.assert_array_equal(
        cohort.blocks[RESCALED_BLOCK], cohort.blocks[NARROW_BLOCK] * RESCALE_FACTOR
    )


def test_the_rescaled_block_carries_identical_information(cohort):
    """Separation is scale-free, so rescaling must not move it at all."""
    assert cohort.separation(RESCALED_BLOCK, "chemistry") == pytest.approx(
        cohort.separation(NARROW_BLOCK, "chemistry"), rel=1e-9
    )


def test_the_degenerate_block_has_every_distance_zero(cohort):
    values = cohort.blocks[DEGENERATE_BLOCK]
    assert values.shape[0] == cohort.n_proteins
    np.testing.assert_array_equal(values, np.tile(values[0], (cohort.n_proteins, 1)))


def test_the_constant_column_block_has_exactly_one_constant_column(cohort):
    variances = cohort.blocks[CONSTANT_COLUMN_BLOCK].var(axis=0)
    assert (variances == 0).sum() == 1
    assert (variances[[0, 2, 3, 4]] > 0).all()


# -- the measuring stick itself ---------------------------------------------


def test_separation_is_one_when_the_labels_are_meaningless():
    rng = np.random.RandomState(7)
    values = rng.normal(size=(120, 6))
    labels = rng.randint(0, 3, size=120)
    assert separation(values, labels) == pytest.approx(1.0, abs=0.08)


def test_separation_is_scale_invariant():
    rng = np.random.RandomState(7)
    values = rng.normal(size=(60, 4))
    labels = np.repeat([0, 1], 30)
    values[labels == 1] += 5.0
    assert separation(values * 1e6, labels) == pytest.approx(separation(values, labels), rel=1e-9)


def test_separation_needs_two_groups():
    with pytest.raises(ValueError, match="two groups"):
        separation(np.zeros((5, 2)), np.zeros(5, dtype=int))


def test_separation_of_identical_rows_is_infinite_only_when_groups_differ():
    values = np.zeros((6, 2))
    labels = np.repeat([0, 1], 3)
    # Every distance is zero, within and between: the grouping is invisible,
    # not infinitely visible. Returning inf here would make the degenerate
    # block look like the most informative one in the cohort.
    assert separation(values, labels) == 1.0


# -- the shape the pipeline reads -------------------------------------------


def test_block_result_is_a_real_block_result(cohort):
    result = cohort.block_result(WIDE_BLOCK)
    assert result.spec.id == WIDE_BLOCK
    assert result.spec.kind == "features"
    assert result.protids == cohort.protids
    assert result.features.shape == (cohort.n_proteins, 200)
    assert result.features.dtype == np.float64


def test_the_cohort_round_trips_through_the_store(tmp_path, cohort):
    """The store is float32, so this also pins what precision fusion actually sees."""
    from fusion_cohort import write_fusion_cohort
    from spaces.store import BlockStore

    write_fusion_cohort(tmp_path, cohort, block_ids=[WIDE_BLOCK, NARROW_BLOCK])
    store = BlockStore(str(tmp_path))
    assert sorted(store.list_blocks()) == [NARROW_BLOCK, WIDE_BLOCK]
    loaded = store.read_block(NARROW_BLOCK)
    assert loaded.protids == cohort.protids
    np.testing.assert_allclose(loaded.features, cohort.blocks[NARROW_BLOCK], rtol=1e-6, atol=1e-12)
