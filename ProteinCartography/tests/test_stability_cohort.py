#!/usr/bin/env python
"""The stability fixture's geometry, asserted before any statistic reads it.

Everything here is about distances and construction. What the diagnostic
*scores* on this cohort is pinned in ``test_diagnostics_stability.py`` -- the
same split ``test_embedding_cohort.py`` makes, and for the same reason: a
fixture whose properties are only ever checked through the statistic under test
cannot tell you which of the two is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from stability_cohort import (
    DIFFUSE,
    GAPPED,
    GAPPED_CLUSTERS,
    GAPPED_RADIUS,
    TIED,
    TIED_N,
    chance_jaccard,
    stability_cohort,
)

COHORT = stability_cohort()
DISTANCES = COHORT.distances()


def _neighbors(distances, index, k):
    masked = distances[index].copy()
    masked[index] = np.inf
    return set(np.argsort(masked, kind="stable")[:k].tolist())


def _gapped_cluster_of(position):
    """The block of ``k + 1`` consecutive positions this protein was built in."""
    size = COHORT.k + 1
    start = (position // size) * size
    return set(range(start, start + size))


# --- shape and reproducibility ----------------------------------------------


def test_the_cohort_has_the_advertised_size():
    assert COHORT.n_proteins == GAPPED_CLUSTERS * (COHORT.k + 1) + 60 + TIED_N
    assert len(COHORT.protids) == COHORT.n_proteins
    assert COHORT.features.shape[0] == COHORT.n_proteins


def test_the_protids_are_unique():
    assert len(set(COHORT.protids)) == len(COHORT.protids)


def test_the_groups_partition_the_cohort():
    counts = {g: int((COHORT.groups == g).sum()) for g in (GAPPED, DIFFUSE, TIED)}
    assert sum(counts.values()) == COHORT.n_proteins
    assert counts[GAPPED] == GAPPED_CLUSTERS * (COHORT.k + 1)
    assert counts[TIED] == TIED_N


def test_the_same_seed_gives_bitwise_identical_features():
    assert np.array_equal(stability_cohort().features, stability_cohort().features)


def test_a_different_seed_gives_different_features():
    assert not np.array_equal(stability_cohort(seed=1).features, COHORT.features)


def test_k_sizes_the_gapped_clusters():
    """The cluster size is ``k + 1`` by construction, not by coincidence.

    If these drifted apart the gap would land at some rank other than k and
    every band in the fixture's table would move without anything failing.
    """
    for k in (5, 10, 14):
        cohort = stability_cohort(k=k)
        assert int((cohort.groups == GAPPED).sum()) == GAPPED_CLUSTERS * (k + 1)


def test_members_rejects_an_unknown_group():
    with pytest.raises(ValueError, match="no group named"):
        COHORT.members("nonesuch")


def test_the_block_result_carries_the_features_unchanged():
    result = COHORT.block_result()
    assert result.protids == COHORT.protids
    assert np.array_equal(result.features, COHORT.features)
    assert result.spec.kind == "features"


# --- the planted geometry, which is the fixture ------------------------------


def test_a_gapped_protein_s_k_nearest_are_exactly_its_clustermates():
    """The strongest statement the `gapped` band rests on.

    Not "mostly clustermates" and not "well separated" -- the k-nearest *set*
    is forced, so with no noise there is nothing left for the statistic to be
    uncertain about and it must return exactly 1.0.
    """
    for position in COHORT.members(GAPPED):
        clustermates = _gapped_cluster_of(position) - {position}
        assert _neighbors(DISTANCES, position, COHORT.k) == clustermates


def test_the_gapped_clusters_have_a_real_gap_at_k():
    """The (k+1)-th neighbor is an order of magnitude further than the k-th.

    This is the quantity the noise term has to overcome, so it is measured
    rather than asserted qualitatively. Anything near 1.0 would mean the band
    is decided by sampling noise.
    """
    ratios = []
    for position in COHORT.members(GAPPED):
        ordered = np.sort(np.delete(DISTANCES[position], position))
        ratios.append(ordered[COHORT.k] / ordered[COHORT.k - 1])
    assert min(ratios) > 10.0, f"smallest gap ratio was {min(ratios):.2f}"


def test_the_diffuse_group_has_no_gap_at_k():
    """The contrast that makes the middle band mean something.

    `diffuse` is not "less separated" than `gapped`; it has no rank-k gap at
    all, so an arbitrarily small perturbation reorders it.
    """
    ratios = []
    for position in COHORT.members(DIFFUSE):
        ordered = np.sort(np.delete(DISTANCES[position], position))
        ratios.append(ordered[COHORT.k] / ordered[COHORT.k - 1])
    assert max(ratios) < 1.5, f"largest gap ratio was {max(ratios):.2f}"


def test_every_tied_pair_is_exactly_equidistant():
    """One distinct value, not one value within a tolerance.

    A near-tie would make the group's chance level depend on the size of the
    near-ness relative to the noise, and the closed-form prediction would stop
    holding. Asserted as an exact count of distinct floats because that is the
    property; the value itself is 1.0 only up to the representation of
    ``sqrt(2)/2``.
    """
    positions = COHORT.members(TIED)
    block = DISTANCES[np.ix_(positions, positions)]
    off_diagonal = block[~np.eye(len(positions), dtype=bool)]
    assert len(np.unique(off_diagonal)) == 1


def test_no_tied_protein_has_a_non_tied_neighbor():
    """Otherwise the group is a mixture and its chance level is not the formula."""
    tied = set(COHORT.members(TIED).tolist())
    for position in tied:
        assert _neighbors(DISTANCES, position, COHORT.k) <= tied


def test_the_three_groups_share_one_distance_scale():
    """The confound that inverted an earlier draft of this fixture.

    The noise level is a fraction of the median pairwise distance, so if any
    group sat at its own scale that median would be a between-group distance
    and the shared sigma would be meaningless for the tight group. Stated as a
    bound on the ratio of each group's internal median to the global one.
    """
    off = DISTANCES[~np.eye(COHORT.n_proteins, dtype=bool)]
    overall = float(np.median(off))
    for group in (GAPPED, DIFFUSE, TIED):
        positions = COHORT.members(group)
        block = DISTANCES[np.ix_(positions, positions)]
        internal = float(np.median(block[~np.eye(len(positions), dtype=bool)]))
        assert 0.1 < internal / overall < 10.0, (
            f"{group} sits at {internal:.3f} against a global median of {overall:.3f}; "
            "one shared noise level cannot be fair to groups at different scales."
        )


def test_the_gapped_radius_is_small_against_the_cluster_spacing():
    """States the construction's margin rather than leaving it implicit."""
    centres = []
    size = COHORT.k + 1
    for start in range(0, GAPPED_CLUSTERS * size, size):
        centres.append(COHORT.features[start : start + size].mean(axis=0))
    centres = np.array(centres)
    spacing = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2))
    closest = spacing[~np.eye(GAPPED_CLUSTERS, dtype=bool)].min()
    assert closest > 20 * GAPPED_RADIUS


# --- the chance level, which the `tied` band is measured against -------------


def test_chance_jaccard_is_one_when_the_pool_is_the_neighborhood():
    """With exactly k candidates both subsets must be all of them."""
    assert chance_jaccard(10, 10) == pytest.approx(1.0)


def test_chance_jaccard_falls_as_the_pool_grows():
    values = [chance_jaccard(pool, 10) for pool in (10, 20, 40, 200)]
    assert values == sorted(values, reverse=True)
    assert values[-1] < 0.03


def test_chance_jaccard_matches_a_monte_carlo_draw():
    """The closed form against the thing it is a closed form for.

    Group 6's lesson inverted: there, comparing derived quantities hid a wrong
    constant table. Here the formula is the only place an error can hide, so it
    is checked against a direct simulation of the process it describes.
    """
    rng = np.random.RandomState(3)
    pool, k, trials = 31, 10, 20000
    total = 0.0
    for _ in range(trials):
        a = set(rng.choice(pool, size=k, replace=False).tolist())
        b = set(rng.choice(pool, size=k, replace=False).tolist())
        total += len(a & b) / len(a | b)
    assert total / trials == pytest.approx(chance_jaccard(pool, k), abs=0.01)


def test_chance_jaccard_refuses_a_pool_smaller_than_k():
    with pytest.raises(ValueError, match="1 <= k <= pool"):
        chance_jaccard(5, 10)
