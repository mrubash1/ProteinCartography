"""What the four fusion strategies do, checked three ways.

1. **Against the definition.** Small hand-computable cases where the answer can
   be worked out on paper, plus scipy behind `importorskip` for the two pieces
   that have a standard implementation.
2. **Against the fixture, in both directions.** `tests/fusion_cohort.py` plants
   two crossed partitions in two different blocks, so every strategy can be
   asked both "did you find what is there" and "did you stay blind to what is
   not". Only the second half catches a fusion that mixed everything into mush
   and scored well by accident.
3. **Against the contract.** ADR 0002 promises normalized scale, shares that sum
   to 1, a loud warning above 70%, and an error rather than a NaN on a
   degenerate block. Each of those is a test rather than a sentence.

The scipy-gated tests do not run in `cartography_tidy`, which has no scipy. Run
them somewhere they do not skip -- see PLAN §0.4 and REVIEW_LOG G7.12.
"""

from __future__ import annotations

import numpy as np
import pytest
from fusion import (
    DEFAULT_GRAPH_K,
    DOMINANCE_THRESHOLD,
    BlockContribution,
    FusionError,
    FusionInput,
    FusionResult,
    fuse,
    fuse_early,
    fuse_graph,
    fuse_late,
    fuse_none,
    standardize,
    unit_mean_distance,
)
from fusion_cohort import (
    DEGENERATE_BLOCK,
    NARROW_BLOCK,
    NOISE_BLOCK,
    RESCALED_BLOCK,
    WIDE_BLOCK,
    fusion_cohort,
    separation,
)


@pytest.fixture(scope="module")
def cohort():
    return fusion_cohort()


@pytest.fixture(scope="module")
def wide(cohort):
    return FusionInput(WIDE_BLOCK, cohort.blocks[WIDE_BLOCK])


@pytest.fixture(scope="module")
def narrow(cohort):
    return FusionInput(NARROW_BLOCK, cohort.blocks[NARROW_BLOCK])


@pytest.fixture(scope="module")
def noise(cohort):
    return FusionInput(NOISE_BLOCK, cohort.blocks[NOISE_BLOCK])


# The three default `graph` fusions, computed once each rather than once per
# test. `fuse_graph` is by some distance the most expensive call in this file --
# ~101 ms for two blocks, ~153 ms for three -- and these three argument sets
# account for eight calls between them, so sharing them takes the file from
# 2.4 s to 1.9 s.
#
# Sharing is safe only because `FusionResult` hands out its numpy arrays without
# copying (`frozen=True` protects the binding, not the buffer). Verified as of
# this commit: no test in this module writes into `result.values`,
# `result.contributions` or `params_used` -- the only `fill_diagonal` calls here
# are on locally built arrays. A single in-place write into one of these
# fixtures would silently corrupt every later test in the module, so treat them
# as read-only; a test that needs to mutate must call `fuse_graph` itself.
#
# The calls that pass a non-default `k`, `mu` or `iterations` deliberately keep
# their own `fuse_graph`: the parameter *is* the thing under test there.
@pytest.fixture(scope="module")
def graph_wide_narrow(wide, narrow):
    return fuse_graph([wide, narrow])


@pytest.fixture(scope="module")
def graph_wide_noise(wide, noise):
    return fuse_graph([wide, noise])


@pytest.fixture(scope="module")
def graph_wide_narrow_noise(wide, narrow, noise):
    return fuse_graph([wide, narrow, noise])


@pytest.fixture(scope="module")
def fold(cohort):
    return cohort.partitions["fold"].labels


@pytest.fixture(scope="module")
def chemistry(cohort):
    return cohort.partitions["chemistry"].labels


# ===========================================================================
# inputs
# ===========================================================================


def test_a_block_must_be_a_two_dimensional_matrix():
    with pytest.raises(FusionError, match="expected an .N, D. feature matrix"):
        FusionInput("x", np.zeros(5))


def test_a_block_may_not_carry_a_nan():
    values = np.zeros((3, 2))
    values[1, 1] = np.nan
    with pytest.raises(FusionError, match="NaN or infinity"):
        FusionInput("x", values)


def test_a_negative_weight_is_refused():
    with pytest.raises(FusionError, match="negative weight"):
        FusionInput("x", np.zeros((3, 2)), weight=-1.0)


def test_blocks_of_different_lengths_are_refused():
    a = FusionInput("a", np.zeros((4, 2)))
    b = FusionInput("b", np.zeros((5, 2)))
    with pytest.raises(FusionError, match="same proteins in the same order"):
        fuse_late([a, b])


def test_a_block_listed_twice_is_refused():
    a = FusionInput("a", np.zeros((4, 2)))
    with pytest.raises(FusionError, match="listed more than once"):
        fuse_late([a, a])


def test_all_zero_weights_are_refused():
    a = FusionInput("a", np.eye(4), weight=0.0)
    b = FusionInput("b", np.eye(4), weight=0.0)
    with pytest.raises(FusionError, match="every weight is zero"):
        fuse_late([a, b])


# ===========================================================================
# the normalization contract (ADR 0002)
# ===========================================================================


def test_unit_mean_distance_puts_the_mean_at_exactly_one():
    rng = np.random.RandomState(0)
    distances = np.abs(rng.normal(size=(30, 30))) * 17.0
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)
    normalized = unit_mean_distance(distances)
    off = normalized[~np.eye(30, dtype=bool)]
    assert off.mean() == pytest.approx(1.0, rel=1e-12)


def test_unit_mean_distance_excludes_the_diagonal():
    """Including the structural zeros would make the normalization depend on N."""
    distances = np.full((4, 4), 2.0)
    np.fill_diagonal(distances, 0.0)
    normalized = unit_mean_distance(distances)
    # Every off-diagonal entry is 2.0, so every one must come back exactly 1.0.
    # If the diagonal were in the mean, the mean would be 1.5 and these 4/3.
    off = normalized[~np.eye(4, dtype=bool)]
    np.testing.assert_allclose(off, 1.0)


def test_a_block_with_no_geometry_is_an_error_not_a_nan(cohort):
    block = FusionInput(DEGENERATE_BLOCK, cohort.blocks[DEGENERATE_BLOCK])
    other = FusionInput(WIDE_BLOCK, cohort.blocks[WIDE_BLOCK])
    with pytest.raises(FusionError, match="every pairwise distance is zero"):
        fuse_late([block, other])


def test_fusion_needs_at_least_two_proteins():
    with pytest.raises(FusionError, match="at least two proteins"):
        unit_mean_distance(np.zeros((1, 1)))


def test_standardize_leaves_a_constant_column_at_zero_and_counts_it(cohort):
    values = cohort.blocks["constant_column"]
    standardized, n_constant = standardize(values)
    assert n_constant == 1
    assert (standardized[:, 1] == 0).all()
    np.testing.assert_allclose(standardized[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized[:, 0].std(), 1.0, rtol=1e-12)


def test_standardizing_an_entirely_constant_block_is_an_error():
    with pytest.raises(FusionError, match="every column is constant"):
        standardize(np.ones((6, 3)), "flat")


def test_standardize_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.RandomState(3)
    values = rng.normal(loc=4.0, scale=9.0, size=(50, 7))
    standardized, n_constant = standardize(values)
    assert n_constant == 0
    np.testing.assert_allclose(standardized, scipy_stats.zscore(values, axis=0), rtol=1e-12)


def test_the_late_distances_match_scipy(cohort):
    """The Gram identity has to agree with a direct pdist, including near zero."""
    spatial = pytest.importorskip("scipy.spatial.distance")
    values = cohort.blocks[NARROW_BLOCK][:60]
    from fusion import pairwise_distances

    ours = pairwise_distances(values)
    theirs = spatial.squareform(spatial.pdist(values))
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-15)


# ===========================================================================
# shares are checked, not assumed
# ===========================================================================


def _contribution(block_id, share, realized=None):
    return BlockContribution(
        block_id=block_id,
        weight=1.0,
        n_features=1,
        share=share,
        realized_share=share if realized is None else realized,
    )


def test_shares_that_do_not_sum_to_one_are_refused():
    with pytest.raises(FusionError, match="sums to"):
        FusionResult(
            values=np.zeros((2, 2)),
            strategy="late",
            representation="distance_profile",
            contributions=(_contribution("a", 0.3), _contribution("b", 0.3)),
        )


def test_realized_shares_are_checked_separately_from_nominal_ones():
    """Two numbers, two checks. A sum enforced on one is not enforced on both."""
    with pytest.raises(FusionError, match="realized_share"):
        FusionResult(
            values=np.zeros((2, 2)),
            strategy="late",
            representation="distance_profile",
            contributions=(
                _contribution("a", 0.5, realized=0.9),
                _contribution("b", 0.5, realized=0.9),
            ),
        )


def test_a_result_must_name_a_contributing_block():
    with pytest.raises(FusionError, match="at least one contributing block"):
        FusionResult(
            values=np.zeros((2, 2)),
            strategy="none",
            representation="features",
            contributions=(),
        )


# ===========================================================================
# strategy `none`
# ===========================================================================


def test_none_returns_the_block_untouched(wide):
    result = fuse_none([wide])
    assert result.values is wide.values
    assert result.representation == "features"
    assert result.shares == {WIDE_BLOCK: 1.0}


def test_none_preserves_the_stores_precision(cohort):
    """float32 in, float32 out, bit for bit.

    The block store writes float32 (ADR 0004) and `none` hands its block
    straight to the reducer. Promoting to float64 here would move every existing
    single-block embedding the moment `reduce_space` started routing through
    this module -- a numeric change to a working path, arriving as a side effect
    of adding fusion.
    """
    stored = cohort.blocks[WIDE_BLOCK].astype(np.float32)
    result = fuse_none([FusionInput(WIDE_BLOCK, stored)])
    assert result.values.dtype == np.float32
    np.testing.assert_array_equal(result.values, stored)


def test_an_integer_block_is_promoted_rather_than_left_to_wrap():
    result = fuse_none([FusionInput("counts", np.arange(12).reshape(6, 2))])
    assert np.issubdtype(result.values.dtype, np.floating)


def test_none_refuses_more_than_one_block(wide, narrow):
    with pytest.raises(FusionError, match="strategy 'none' means a single block"):
        fuse_none([wide, narrow])


# ===========================================================================
# strategy `early` -- the scale failure ADR 0002 is written about
# ===========================================================================


def test_early_concatenates_every_column(wide, narrow):
    result = fuse_early([wide, narrow])
    assert result.values.shape == (wide.n_proteins, wide.n_features + narrow.n_features)
    assert result.representation == "features"


def test_early_gives_the_wide_block_the_geometry(wide, narrow):
    """The headline number in ADR 0002, reproduced on a fixture built for it.

    200 columns against 4, both standardized, both weighted 1.0: the wide block
    takes 98% of the variance. Nothing about the two blocks' information
    justifies that -- the fixture's own tests establish that they separate their
    partitions about equally well -- so it is column count and nothing else.
    """
    result = fuse_early([wide, narrow])
    assert result.shares[WIDE_BLOCK] == pytest.approx(0.98, abs=0.01)
    assert result.shares[NARROW_BLOCK] == pytest.approx(0.02, abs=0.01)


def test_early_finds_the_wide_blocks_partition_and_loses_the_narrow_one(
    wide, narrow, fold, chemistry
):
    result = fuse_early([wide, narrow])
    assert separation(result.values, fold) > 2.0
    assert separation(result.values, chemistry) == pytest.approx(1.0, abs=0.05)


def test_early_warns_with_the_runs_own_numbers(wide, narrow):
    """ADR 0002 requires the warning cite real numbers, not a generic caution."""
    result = fuse_early([wide, narrow])
    warning = next(w for w in result.warnings if "total variance" in w)
    assert "D=200" in warning and "D=4" in warning
    assert "98.0%" in warning
    assert "strategy: late" in warning


def test_early_also_gets_the_dominance_warning(wide, narrow):
    """ADR 0002's 70% rule applies to every strategy, not only the fused ones.

    `early` has a warning of its own about column count, and it is not the same
    statement: one says why the share came out that way, the other says what the
    share means for reading the map. On the demo cohort `early` lands at 73.3%,
    just over the line, which is exactly the case where one warning without the
    other would be easy to skim past.
    """
    result = fuse_early([wide, narrow])
    assert result.dominant.block_id == WIDE_BLOCK
    assert any("that block's map" in w for w in result.warnings)
    assert any("total variance" in w for w in result.warnings)


def test_early_says_when_it_left_a_constant_column_alone(cohort, wide):
    block = FusionInput("constant_column", cohort.blocks["constant_column"])
    result = fuse_early([wide, block])
    assert any("1 constant column" in w for w in result.warnings)


def test_early_weights_act_on_variance(narrow):
    """Weighting is sqrt(w) on the values because variance is quadratic in them.

    A block at weight 4 against an identical block at weight 1 must take 80% of
    the variance, not 4/(4+1) of the values' magnitude or 2/(2+1) of anything.
    """
    a = FusionInput("a", narrow.values, weight=4.0)
    b = FusionInput("b", narrow.values, weight=1.0)
    result = fuse_early([a, b])
    assert result.shares["a"] == pytest.approx(0.8, rel=1e-9)


# ===========================================================================
# strategy `late` -- the recommended path
# ===========================================================================


def test_late_returns_a_square_distance_profile(wide, narrow):
    result = fuse_late([wide, narrow])
    assert result.values.shape == (wide.n_proteins, wide.n_proteins)
    assert result.representation == "distance_profile"
    np.testing.assert_allclose(np.diag(result.values), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.values, result.values.T, atol=1e-12)


def test_late_finds_both_partitions(wide, narrow, fold, chemistry):
    """The point of fusing: neither block alone can see both.

    `wide` is blind to `chemistry` and `narrow` is blind to `fold` -- the
    fixture's own tests establish both -- so a fused geometry that separates
    both partitions has combined them rather than picked one.
    """
    result = fuse_late([wide, narrow])
    assert separation(result.values, fold) > 1.2
    assert separation(result.values, chemistry) > 1.2


def test_late_is_immune_to_the_scale_gap_that_defeats_early(wide, narrow):
    """A 7500x difference in units moves the shares by nothing."""
    result = fuse_late([wide, narrow])
    assert result.shares[WIDE_BLOCK] == pytest.approx(0.5, rel=1e-9)
    assert result.shares[NARROW_BLOCK] == pytest.approx(0.5, rel=1e-9)


def test_late_gives_an_identical_geometry_for_a_rescaled_block(cohort, wide):
    """The one test only a normalization applied in the wrong place can fail.

    `narrow_rescaled` is `narrow` times 1000: the same information in different
    units. If normalization happened after weighting, or after the squares were
    accumulated, these two fusions would differ.
    """
    plain = fuse_late([wide, FusionInput("chem", cohort.blocks[NARROW_BLOCK])])
    rescaled = fuse_late([wide, FusionInput("chem", cohort.blocks[RESCALED_BLOCK])])
    np.testing.assert_allclose(plain.values, rescaled.values, rtol=1e-10, atol=1e-12)
    assert plain.shares == pytest.approx(rescaled.shares)


def test_late_records_the_raw_mean_distance_that_normalization_removed(wide, narrow):
    """The units are gone from the geometry and must not be gone from the record."""
    result = fuse_late([wide, narrow])
    by_id = {c.block_id: c for c in result.contributions}
    assert by_id[WIDE_BLOCK].mean_distance > 100
    assert by_id[NARROW_BLOCK].mean_distance < 0.1


def test_the_nominal_and_realized_shares_are_different_numbers(wide, narrow):
    """ADR 0002's formula returns the weights; what the blocks supply does not.

    `mean(d̃)` is 1 by construction, so ADR 0002's share is the normalized weight
    vector and can say nothing the config did not. `mean(d̃²)` is `1 + var(d̃)`,
    so a block whose distances are more dispersed genuinely puts more into the
    fused squared distance. On this fixture the gap is a few percent; it is not
    zero, and a test asserting the two agree would have been asserting the
    formula against itself.
    """
    result = fuse_late([wide, narrow])
    assert result.shares[WIDE_BLOCK] == pytest.approx(0.5, rel=1e-9)
    assert result.realized_shares[WIDE_BLOCK] != pytest.approx(0.5, rel=1e-6)
    assert result.realized_shares[WIDE_BLOCK] == pytest.approx(0.477, abs=0.01)
    assert sum(result.realized_shares.values()) == pytest.approx(1.0, rel=1e-12)


def test_late_with_one_block_reproduces_that_blocks_normalized_geometry(narrow):
    """Why the formula divides by the total weight, which ADR 0002's does not.

    Without the division, fusing one block would scale its distances by
    `sqrt(w)` -- a constant that depends on nothing but how the config was
    written. With it, `late` over one block *is* the normalization.
    """
    from fusion import pairwise_distances

    expected = unit_mean_distance(pairwise_distances(narrow.values))
    for weight in (0.25, 1.0, 7.0):
        result = fuse_late([FusionInput("only", narrow.values, weight=weight)])
        np.testing.assert_allclose(result.values, expected, rtol=1e-12)


def test_a_dominant_block_is_named_in_a_warning(wide, narrow):
    result = fuse_late([FusionInput(WIDE_BLOCK, wide.values, weight=50.0), narrow])
    assert result.dominant is not None
    assert result.dominant.block_id == WIDE_BLOCK
    warning = next(w for w in result.warnings if "of the fused geometry" in w)
    assert WIDE_BLOCK in warning
    assert "that block's map" in warning


def test_no_dominance_warning_when_the_blocks_are_balanced(wide, narrow):
    result = fuse_late([wide, narrow])
    assert result.dominant is None
    assert not any("threshold" in w for w in result.warnings)
    assert max(result.realized_shares.values()) < DOMINANCE_THRESHOLD


def test_weights_move_the_late_geometry_toward_the_weighted_block(cohort, narrow, fold, chemistry):
    """Monotone in the weight, in both partitions at once."""
    seen = []
    for weight in (0.05, 1.0, 20.0):
        result = fuse_late([FusionInput(WIDE_BLOCK, cohort.blocks[WIDE_BLOCK], weight), narrow])
        seen.append((separation(result.values, fold), separation(result.values, chemistry)))
    (low_f, low_c), (mid_f, mid_c), (high_f, high_c) = seen
    assert low_f < mid_f < high_f
    assert low_c > mid_c > high_c


# ===========================================================================
# strategy `graph` -- SNF
# ===========================================================================


def test_graph_needs_two_blocks(wide):
    with pytest.raises(FusionError, match="at least 2 block"):
        fuse_graph([wide])


def test_graph_rejects_an_out_of_range_k(wide, narrow):
    with pytest.raises(FusionError, match="k must be between 1 and N"):
        fuse_graph([wide, narrow], k=0)
    with pytest.raises(FusionError, match="k must be between 1 and N"):
        fuse_graph([wide, narrow], k=wide.n_proteins + 1)


def test_graph_rejects_nonsense_hyperparameters(wide, narrow):
    with pytest.raises(FusionError, match="mu must be positive"):
        fuse_graph([wide, narrow], mu=0.0)
    with pytest.raises(FusionError, match="iterations must be at least 1"):
        fuse_graph([wide, narrow], iterations=0)


def test_graph_returns_a_symmetric_affinity_with_no_self_affinity(graph_wide_narrow):
    result = graph_wide_narrow
    assert result.representation == "affinity_profile"
    np.testing.assert_allclose(result.values, result.values.T, atol=1e-15)
    np.testing.assert_array_equal(np.diag(result.values), 0.0)
    assert (result.values >= 0).all()


def test_the_iterates_stay_row_stochastic():
    """Equation 8's property, which is what makes the per-step delta comparable."""
    import fusion

    rng = np.random.RandomState(1)
    distances = fusion.pairwise_distances(rng.normal(size=(40, 5)))
    normalized = unit_mean_distance(distances)
    affinity = fusion._scaled_exponential_kernel(normalized, DEFAULT_GRAPH_K, 0.5)
    full = fusion._full_normalize(affinity)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.diag(full), 0.5, rtol=1e-12)
    local = fusion._local_normalize(affinity, 8)
    np.testing.assert_allclose(local.sum(axis=1), 1.0, rtol=1e-12)
    assert (local == 0).sum() > 0, "sparsification kept every entry"


def test_graph_converges_and_says_so(graph_wide_narrow):
    result = graph_wide_narrow
    assert result.params_used["final_delta"] < 1e-6
    assert not any("not converged" in w for w in result.warnings)


def test_graph_warns_when_it_has_not_converged(wide, narrow):
    result = fuse_graph([wide, narrow], iterations=1)
    assert result.params_used["final_delta"] > 1e-6
    assert any("had not converged" in w for w in result.warnings)


def test_graph_finds_both_partitions(graph_wide_narrow, fold, chemistry):
    result = graph_wide_narrow
    assert separation(result.values, fold) > 1.2
    assert separation(result.values, chemistry) > 1.1


def test_graph_is_immune_to_the_scale_gap(cohort, wide):
    plain = fuse_graph([wide, FusionInput("chem", cohort.blocks[NARROW_BLOCK])])
    rescaled = fuse_graph([wide, FusionInput("chem", cohort.blocks[RESCALED_BLOCK])])
    np.testing.assert_allclose(plain.values, rescaled.values, atol=1e-14)


def test_two_identical_blocks_split_every_protein_exactly_evenly(wide):
    """The strongest exact statement available about the per-protein shares.

    SNF is a smoothing operation, so fusing two copies of a view does *not*
    return that view -- the diffusion moves it. What must hold exactly is the
    attribution: if two blocks are byte-identical then no protein has any reason
    to prefer one, and every per-protein share must be exactly one half. A
    per-protein share that drifted here would be measuring floating-point
    accumulation order and calling it evidence.
    """
    result = fuse_graph([FusionInput("a", wide.values), FusionInput("b", wide.values)])
    for contribution in result.contributions:
        np.testing.assert_allclose(contribution.per_protein_share, 0.5, rtol=1e-12)
        assert contribution.realized_share == pytest.approx(0.5, rel=1e-12)


def test_an_informative_block_outweighs_a_noise_block_per_protein(graph_wide_noise):
    """What `graph` has that `late` does not: an unconfigured, earned weight.

    Both blocks are weighted 1.0, so `late` would report exactly 50/50 whatever
    they contained. Here the block with structure comes out ahead because the
    fused network agrees with it more, and it does so for every protein rather
    than on average.
    """
    result = graph_wide_noise
    by_id = {c.block_id: c for c in result.contributions}
    assert by_id[WIDE_BLOCK].share == pytest.approx(0.5, rel=1e-12)
    assert by_id[WIDE_BLOCK].realized_share > 0.55
    assert (by_id[WIDE_BLOCK].per_protein_share > by_id[NOISE_BLOCK].per_protein_share).all()


def test_the_per_protein_shares_sum_to_one_for_every_protein(graph_wide_narrow_noise):
    result = graph_wide_narrow_noise
    stacked = np.stack([c.per_protein_share for c in result.contributions])
    np.testing.assert_allclose(stacked.sum(axis=0), 1.0, rtol=1e-12)


def test_the_per_protein_shares_vary_between_proteins(graph_wide_noise):
    """Otherwise they are a scalar with N copies, which is what `late` already has."""
    result = graph_wide_noise
    shares = result.contributions[0].per_protein_share
    assert shares.max() - shares.min() > 0.01


def test_only_graph_reports_per_protein_shares(wide, narrow):
    """`early` and `late` apply one weight to everyone, and say so by omission."""
    for result in (fuse_early([wide, narrow]), fuse_late([wide, narrow])):
        assert all(c.per_protein_share is None for c in result.contributions)


def test_weights_move_the_graph_geometry_toward_the_weighted_block(cohort, narrow, fold, chemistry):
    seen = []
    for weight in (0.1, 1.0, 10.0):
        result = fuse_graph([FusionInput(WIDE_BLOCK, cohort.blocks[WIDE_BLOCK], weight), narrow])
        seen.append(
            (
                separation(result.values, fold),
                separation(result.values, chemistry),
                result.realized_shares[WIDE_BLOCK],
            )
        )
    assert seen[0][0] < seen[1][0] < seen[2][0]
    assert seen[0][1] > seen[1][1] > seen[2][1]
    assert seen[0][2] < seen[1][2] < seen[2][2]


# ===========================================================================
# dispatch and reporting
# ===========================================================================


@pytest.mark.parametrize("strategy", ["early", "late", "graph"])
def test_fuse_dispatches_to_each_strategy(strategy, wide, narrow):
    result = fuse(strategy, [wide, narrow])
    assert result.strategy == strategy
    assert sum(result.shares.values()) == pytest.approx(1.0, rel=1e-12)


def test_fuse_rejects_an_unknown_strategy(wide, narrow):
    with pytest.raises(FusionError, match="unknown fusion strategy"):
        fuse("snf", [wide, narrow])


def test_fuse_rejects_a_misspelled_parameter(wide, narrow):
    """A dropped parameter leaves a run reporting the default and looking configured."""
    with pytest.raises(FusionError, match=r"does not take parameter\(s\) \['iteratons'\]"):
        fuse("graph", [wide, narrow], {"iteratons": 3})


def test_fuse_rejects_a_parameter_the_strategy_does_not_use(wide, narrow):
    with pytest.raises(FusionError, match="Allowed: \\(none\\)"):
        fuse("late", [wide, narrow], {"k": 5})


def test_graph_parameters_reach_the_algorithm(wide, narrow):
    result = fuse("graph", [wide, narrow], {"k": 7, "mu": 0.4, "iterations": 3})
    assert result.params_used["k"] == 7
    assert result.params_used["mu"] == 0.4
    assert result.params_used["iterations"] == 3
    # And they changed the answer, rather than being recorded and ignored.
    #
    # **Both sides go through `fuse`, and that is the test.** A speedup pass
    # once replaced this control with the module-scoped `graph_wide_narrow`
    # fixture, which calls `fuse_graph` directly, on the reasoning that `fuse`
    # with an empty params dict *is* `fuse_graph(inputs)`. That is true today
    # and it is exactly the assumption under test: with the fixture as the
    # control, a `fuse` that quietly substituted its own defaults would produce
    # two identical sides and pass. Gate E's adversarial pass demonstrated it
    # with that mutation -- caught before the change, missed after.
    default = fuse("graph", [wide, narrow], {})
    assert not np.allclose(result.values, default.values)


def test_describe_shows_the_weight_vector_and_the_shares(wide, narrow):
    """ADR 0002: the weight vector is a displayed object, not a buried config value."""
    text = fuse_late([FusionInput(WIDE_BLOCK, wide.values, weight=3.0), narrow]).describe()
    assert "weight 3" in text
    assert "share" in text and "realized" in text
    assert WIDE_BLOCK in text and NARROW_BLOCK in text


def test_the_result_serializes_everything_a_manifest_needs(graph_wide_narrow_noise):
    data = graph_wide_narrow_noise.to_dict()
    assert data["strategy"] == "graph"
    assert data["representation"] == "affinity_profile"
    assert {c["block_id"] for c in data["contributions"]} == {
        WIDE_BLOCK,
        NARROW_BLOCK,
        NOISE_BLOCK,
    }
    first = data["contributions"][0]
    assert set(first["per_protein_share"]) == {"min", "median", "max"}
    assert data["params_used"]["algorithm"].startswith("SNF")
    # The whole thing has to survive a round trip to JSON, which is where a
    # stray numpy scalar turns into a TypeError at the end of a long run.
    import json

    assert json.loads(json.dumps(data))["strategy"] == "graph"


# --- Gate D: a block with no geometry must not reach the map ------------------


@pytest.mark.parametrize("n", [24, 60, 120, 240])
def test_a_constant_block_is_refused_at_every_cohort_size(n):
    """Gate D found this passing at N=240 and silently failing at N=60.

    `pairwise_distances` uses the Gram identity, which loses about sqrt(eps)
    relative precision near zero, so a block whose rows are *bitwise identical*
    came back with a mean distance of 4.2e-08 rather than 0. The guard tested
    `mean <= 0.0`, missed it, divided by the residue, and handed pure
    cancellation noise a **46.7% contribution share** of the fused geometry.

    Whether it happened depended on the magnitude of the block's values, which
    is why one N raised and another did not -- so the size is a parameter now
    rather than a fixture constant.
    """
    from fusion_cohort import fusion_cohort

    cohort = fusion_cohort(n=n)
    blocks = [
        FusionInput(DEGENERATE_BLOCK, cohort.blocks[DEGENERATE_BLOCK]),
        FusionInput(WIDE_BLOCK, cohort.blocks[WIDE_BLOCK]),
    ]
    for strategy in ("late", "graph"):
        with pytest.raises(FusionError, match="carries no geometry"):
            fuse(strategy, blocks, {})


def test_the_noise_floor_names_the_scale_it_compared_against():
    """A "this is zero" message on a distance of 4e-08 would be confusing.

    Both branches are reachable and they need different wording: whether the
    Gram identity cancels to exactly zero or leaves a residue depends on the
    magnitude of the row, so a constant block hits one or the other with no
    pattern a reader could predict. This is the residue branch, using the exact
    cohort size Gate D found it at -- ``fusion_cohort(n=60)``'s degenerate block
    produces a mean distance of 4.2e-08 where ``n=240``'s produces 0.0.
    """
    from fusion import NOISE_FLOOR
    from fusion_cohort import fusion_cohort

    assert NOISE_FLOOR == 1e-6
    cohort = fusion_cohort(n=60)
    blocks = [
        FusionInput(DEGENERATE_BLOCK, cohort.blocks[DEGENERATE_BLOCK]),
        FusionInput(WIDE_BLOCK, cohort.blocks[WIDE_BLOCK]),
    ]
    with pytest.raises(FusionError, match="cancellation noise"):
        fuse("late", blocks, {})


def test_a_block_with_real_but_small_variation_is_still_accepted():
    """The other half of the guard. The floor is 1e-6 of the block's own scale,
    so a block whose geometry is genuinely small relative to its offset -- which
    is ordinary -- must survive."""
    rng = np.random.RandomState(0)
    values = 1000.0 + rng.normal(0.0, 0.01, size=(40, 4))
    result = fuse("late", [FusionInput("small", values), FusionInput("real", np.eye(40))], {})
    assert len(result.contributions) == 2
