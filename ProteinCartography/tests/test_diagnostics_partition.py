#!/usr/bin/env python
"""ARI, silhouette, the resolution sweep and the negative controls.

The two statistics have reference implementations in scikit-learn, so they are
checked against it -- including in the degenerate cases where its answer is a
convention rather than a formula. Group 8b's clearest evidence that its
arithmetic was right was exact agreement with sklearn's trustworthiness, and
the same lever is available here.

The reports on top of them have no reference, so they are checked against
planted partitions instead: ``fusion_cohort`` carries two exactly crossed
groupings, so a statistic that recovers one must be blind to the other, and a
report that confuses them cannot produce that pattern by accident.
"""

from __future__ import annotations

import numpy as np
import pytest
from diagnostics.partition import (
    MEANINGFUL_MARGIN,
    PLATEAU_THRESHOLD,
    PartitionError,
    adjusted_rand_index,
    contingency_table,
    negative_controls,
    resolution_sweep,
    silhouette,
)
from fusion_cohort import NARROW_BLOCK, NOISE_BLOCK, WIDE_BLOCK, fusion_cohort

COHORT = fusion_cohort()
FOLD = COHORT.partitions["fold"].labels
CHEMISTRY = COHORT.partitions["chemistry"].labels


def distances_of(block_id) -> np.ndarray:
    values = COHORT.values(block_id)
    return np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))


WIDE = distances_of(WIDE_BLOCK)


# --- adjusted Rand index ------------------------------------------------------


def test_a_partition_agrees_perfectly_with_itself():
    assert adjusted_rand_index(FOLD, FOLD) == pytest.approx(1.0)


def test_renaming_the_clusters_changes_nothing():
    """Leiden returns strings like ``LC03``; the legacy path prefixes them
    again. Comparing partitions must not depend on either."""
    renamed = [f"LC{value:02d}" for value in FOLD]
    assert adjusted_rand_index(FOLD, renamed) == pytest.approx(1.0)


def test_the_index_is_symmetric():
    assert adjusted_rand_index(FOLD, CHEMISTRY) == pytest.approx(
        adjusted_rand_index(CHEMISTRY, FOLD)
    )


def test_two_exactly_crossed_partitions_score_about_zero():
    """The fixture's construction, read back out.

    `fold` and `chemistry` are built as an exact cross-product, so neither
    carries information about the other and a chance-corrected index must say
    so. An uncorrected Rand index would score these around 0.7.
    """
    assert abs(adjusted_rand_index(FOLD, CHEMISTRY)) < 0.02


def test_a_systematically_crossed_labeling_goes_negative():
    """Below chance is a real answer, not a clamp."""
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([0, 0, 0, 0], [0, 0, 0, 0], 1.0),
        ([0, 1, 2, 3], [0, 1, 2, 3], 1.0),
        ([0, 0, 0, 0], [0, 0, 1, 1], 0.0),
        ([0, 1, 2, 3], [0, 0, 1, 1], 0.0),
    ],
)
def test_the_degenerate_cases_follow_sklearn_s_convention(left, right, expected):
    """0/0 by the formula, fixed by convention, and the pipeline reaches it.

    ``leiden_clustering`` short-circuits to a single cluster below three
    proteins, so two such spaces compared to each other hit the all-in-one case
    on any small cohort.
    """
    assert adjusted_rand_index(left, right) == pytest.approx(expected)


def test_adjusted_rand_index_agrees_with_sklearn():
    """Relative tolerance, not absolute: an absolute 1e-12 is vacuous against
    values that are legitimately near zero, which is most of a random pair."""
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.RandomState(0)
    for _ in range(40):
        n = int(rng.randint(4, 120))
        left = rng.randint(0, rng.randint(2, 9), size=n)
        right = rng.randint(0, rng.randint(2, 9), size=n)
        mine = adjusted_rand_index(left, right)
        theirs = float(metrics.adjusted_rand_score(left, right))
        assert abs(mine - theirs) <= 1e-12 + 1e-12 * abs(theirs), (mine, theirs)


def test_the_planted_partitions_agree_with_sklearn_too():
    metrics = pytest.importorskip("sklearn.metrics")
    for left, right in ((FOLD, CHEMISTRY), (FOLD, FOLD), (CHEMISTRY, FOLD)):
        assert adjusted_rand_index(left, right) == pytest.approx(
            float(metrics.adjusted_rand_score(left, right)), rel=1e-12, abs=1e-12
        )


def test_mismatched_lengths_are_refused():
    with pytest.raises(PartitionError, match="different numbers of proteins"):
        adjusted_rand_index([0, 1, 2], [0, 1])


def test_the_contingency_table_counts_the_cross_product():
    """4 fold groups x 3 chemistry groups, every cell equally filled."""
    table = contingency_table(FOLD, CHEMISTRY)
    assert table.shape == (4, 3)
    assert table.sum() == COHORT.n_proteins
    assert len(np.unique(table)) == 1


# --- silhouette ----------------------------------------------------------------


def test_the_block_that_shows_a_partition_separates_it():
    assert silhouette(WIDE, FOLD).mean() > 0.5


def test_the_same_block_is_blind_to_the_crossed_partition():
    assert abs(silhouette(WIDE, CHEMISTRY).mean()) < 0.05


def test_each_block_separates_only_its_own_partition():
    """The 2x2 the fixture was built for, read through the silhouette."""
    narrow = distances_of(NARROW_BLOCK)
    assert silhouette(WIDE, FOLD).mean() > 0.5
    assert abs(silhouette(WIDE, CHEMISTRY).mean()) < 0.05
    assert silhouette(narrow, CHEMISTRY).mean() > 0.5
    assert abs(silhouette(narrow, FOLD).mean()) < 0.05


def test_a_partition_of_noise_scores_about_zero():
    assert abs(silhouette(distances_of(NOISE_BLOCK), FOLD).mean()) < 0.05


def test_a_singleton_cluster_scores_zero_rather_than_one():
    """Otherwise a clustering is rewarded for producing singletons."""
    distances = np.array([[0, 1, 5, 6], [1, 0, 5, 6], [5, 5, 0, 1], [6, 6, 1, 0]], dtype=float)
    assert list(silhouette(distances, [0, 0, 1, 2])) == [0.8, 0.8, 0.0, 0.0]


def test_silhouette_agrees_with_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.RandomState(1)
    for block_id in (WIDE_BLOCK, NARROW_BLOCK, NOISE_BLOCK):
        distances = distances_of(block_id)
        for labels in (FOLD, CHEMISTRY, rng.randint(0, 5, size=COHORT.n_proteins)):
            mine = silhouette(distances, labels)
            theirs = metrics.silhouette_samples(distances, labels, metric="precomputed")
            assert np.allclose(mine, theirs, rtol=1e-12, atol=1e-12)


def test_silhouette_needs_two_clusters():
    with pytest.raises(PartitionError, match="at least two clusters"):
        silhouette(WIDE, np.zeros(COHORT.n_proteins, dtype=int))


def test_silhouette_rejects_a_mismatched_matrix():
    with pytest.raises(PartitionError, match="must be square"):
        silhouette(WIDE[:10, :10], FOLD)


# --- the resolution sweep -------------------------------------------------------


def _nested(labels, n_groups):
    """A coarsening of ``labels`` into ``n_groups`` groups."""
    return np.asarray(labels) % n_groups


def test_a_plateau_is_found_where_the_partition_stops_moving():
    """Three resolutions returning the same grouping is the level to report."""
    partitions = {
        0.5: _nested(FOLD, 2),
        1.0: FOLD,
        1.5: FOLD,
        2.0: FOLD,
        4.0: np.arange(COHORT.n_proteins) % 40,
    }
    sweep = resolution_sweep("wide", WIDE, partitions)
    low, high, width = sweep.plateau()
    assert (low, high, width) == (1.0, 2.0, 3)
    assert sweep.warnings() == []


def test_no_plateau_is_reported_when_every_resolution_disagrees():
    """The finding that matters: the cluster count is tracking the knob."""
    rng = np.random.RandomState(2)
    partitions = {r: rng.randint(0, 6, size=COHORT.n_proteins) for r in (0.5, 1.0, 1.5, 2.0)}
    sweep = resolution_sweep("noise", distances_of(NOISE_BLOCK), partitions)
    assert sweep.plateau()[2] == 1
    assert any("tracking the resolution parameter" in note for note in sweep.warnings())


def test_the_sweep_records_the_cluster_count_and_separation_per_rung():
    sweep = resolution_sweep("wide", WIDE, {1.0: FOLD, 2.0: CHEMISTRY})
    assert [step.n_clusters for step in sweep.steps] == [4, 3]
    assert sweep.steps[0].silhouette_mean > 0.5
    assert abs(sweep.steps[1].silhouette_mean) < 0.05


def test_the_sweep_is_ordered_by_resolution_however_it_is_given():
    sweep = resolution_sweep("wide", WIDE, {2.0: CHEMISTRY, 1.0: FOLD})
    assert [step.resolution for step in sweep.steps] == [1.0, 2.0]


def test_an_unmoved_sweep_says_the_range_may_be_too_narrow():
    sweep = resolution_sweep("wide", WIDE, {1.0: FOLD, 1.1: FOLD})
    assert any("too narrow" in note for note in sweep.warnings())


def test_adjacent_agreement_uses_the_threshold_it_documents():
    partitions = {1.0: FOLD, 2.0: FOLD}
    sweep = resolution_sweep("wide", WIDE, partitions)
    assert sweep.adjacent[0][2] >= PLATEAU_THRESHOLD


def test_the_sweep_survives_json():
    import json

    payload = json.dumps(resolution_sweep("wide", WIDE, {1.0: FOLD, 2.0: CHEMISTRY}).to_dict())
    assert "plateau" in payload


# --- the negative controls --------------------------------------------------------


def test_shuffling_the_labels_destroys_the_separation():
    report = negative_controls("wide", WIDE, FOLD)
    assert report.observed.silhouette_mean > 0.5
    assert abs(report.margin("shuffled_labels") - report.observed.silhouette_mean) < 0.05
    assert report.warnings() == []


def test_the_permutation_control_keeps_every_marginal_fixed():
    """It changes only the correspondence, so a warning from it cannot be
    explained by cluster count or cluster sizes."""
    report = negative_controls("wide", WIDE, FOLD)
    control = next(c for c in report.controls if c.name == "shuffled_labels")
    assert control.n_clusters == report.observed.n_clusters


def test_a_partition_no_better_than_its_control_is_called_out():
    """The headline of item 8. Here the observed partition *is* noise."""
    report = negative_controls("noise", distances_of(NOISE_BLOCK), FOLD)
    assert report.margin("shuffled_labels") < MEANINGFUL_MARGIN
    assert any("should not be reported as findings" in note for note in report.warnings()) or any(
        "weak evidence" in note for note in report.warnings()
    )


def test_a_clusterer_finds_attractive_clusters_in_pure_noise():
    """The uncomfortable half of item 8, demonstrated rather than asserted.

    Fitting k-means to a block with no partition in it returns clusters with a
    clearly positive silhouette -- higher, here, than the *correct* partition of
    the same block scores. That is the number a reader has to see beside a real
    one, and it is why the control is a fitted clustering rather than a
    permutation.
    """
    cluster = pytest.importorskip("sklearn.cluster")
    noise = COHORT.values(NOISE_BLOCK)
    fitted = cluster.KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(noise)
    distances = distances_of(NOISE_BLOCK)
    report = negative_controls(
        "wide",
        WIDE,
        FOLD,
        extra=[("random_distances", "a block with no partition in it", distances, fitted)],
    )
    control = next(c for c in report.controls if c.name == "random_distances")
    assert control.silhouette_mean > 0.05
    assert control.silhouette_mean > silhouette(distances, FOLD).mean()
    # The real partition still beats it, which is what makes the number useful.
    assert report.margin("random_distances") > MEANINGFUL_MARGIN


def test_an_unknown_control_name_is_an_error():
    report = negative_controls("wide", WIDE, FOLD)
    with pytest.raises(KeyError, match="no control named"):
        report.margin("nonesuch")


def test_the_controls_survive_json():
    import json

    payload = json.dumps(negative_controls("wide", WIDE, FOLD).to_dict())
    assert "shuffled_labels" in payload


def test_the_permutation_control_is_reproducible():
    first = negative_controls("wide", WIDE, FOLD, seed=5)
    second = negative_controls("wide", WIDE, FOLD, seed=5)
    assert first.margin("shuffled_labels") == second.margin("shuffled_labels")
