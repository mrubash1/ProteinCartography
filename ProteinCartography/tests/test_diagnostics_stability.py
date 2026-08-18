#!/usr/bin/env python
"""What the stability statistic scores on a cohort with a planted answer.

``test_stability_cohort.py`` pins the fixture's geometry. This pins what the
diagnostic makes of it, and the two are separate so that a disagreement between
them names which one is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from diagnostics.stability import (
    COIN_FLIP_THRESHOLD,
    DEFAULT_NOISE,
    STABLE_THRESHOLD,
    NeighborhoodStability,
    StabilityError,
    jaccard_rows,
    largest_stable_k,
    neighborhood_stability,
)
from stability_cohort import DIFFUSE, GAPPED, TIED, chance_jaccard, stability_cohort

COHORT = stability_cohort()
DISTANCES = COHORT.distances()
K = COHORT.k


def run(**kwargs) -> NeighborhoodStability:
    options = {"k": K, "seed": 1}
    options.update(kwargs)
    return neighborhood_stability("planted", DISTANCES, COHORT.protids, **options)


def group_mean(result, group) -> float:
    return float(np.nanmean(result.stability[COHORT.members(group)]))


# --- the planted three bands, which are the point ----------------------------


def test_the_three_planted_bands_land_one_per_reporting_band():
    """The load-bearing test in this file.

    Shape assertions cannot fail on a statistic that returns a plausible
    constant; this can only pass if `gapped`, `diffuse` and `tied` are told
    apart, and each lands where its construction says it must.
    """
    result = run()
    gapped, diffuse, tied = (group_mean(result, g) for g in (GAPPED, DIFFUSE, TIED))
    assert gapped > STABLE_THRESHOLD, gapped
    assert COIN_FLIP_THRESHOLD < diffuse < STABLE_THRESHOLD, diffuse
    assert tied < COIN_FLIP_THRESHOLD, tied


def test_the_band_ordering_survives_the_seed():
    """Six cohorts and six replicate streams. A lucky draw is not a fixture."""
    for seed in range(6):
        cohort = stability_cohort(seed=seed)
        result = neighborhood_stability(
            "planted", cohort.distances(), cohort.protids, k=cohort.k, seed=seed + 100
        )
        means = [
            float(np.nanmean(result.stability[cohort.members(g)])) for g in (GAPPED, DIFFUSE, TIED)
        ]
        assert means == sorted(means, reverse=True), (seed, means)
        assert means[0] > STABLE_THRESHOLD
        assert COIN_FLIP_THRESHOLD < means[1] < STABLE_THRESHOLD
        assert means[2] < COIN_FLIP_THRESHOLD


def test_the_measured_table_is_what_the_fixture_documents():
    """The numbers in ``stability_cohort``'s docstring, to four places."""
    expected = {
        0.00: (1.0000, 1.0000, 1.0000),
        0.01: (0.9583, 0.7824, 0.1968),
        0.05: (0.9196, 0.6228, 0.1968),
        0.10: (0.8724, 0.4953, 0.1968),
    }
    for noise, (gapped, diffuse, tied) in expected.items():
        result = run(noise=noise)
        assert group_mean(result, GAPPED) == pytest.approx(gapped, abs=5e-5)
        assert group_mean(result, DIFFUSE) == pytest.approx(diffuse, abs=5e-5)
        assert group_mean(result, TIED) == pytest.approx(tied, abs=5e-5)


# --- the two invariants that make the perturbation interpretable -------------


def test_resampling_alone_scores_exactly_one_for_every_protein():
    """The null the whole design rests on.

    The reference neighbor set is recomputed inside each subsample, so dropping
    proteins cannot change the answer: both sides lose the same ones. Exact
    equality, not a tolerance -- this is set overlap, and any departure means
    the bookkeeping is measuring itself.
    """
    result = run(noise=0.0)
    assert np.all(result.stability == 1.0)
    assert result.mean_stability == 1.0


def test_the_tied_group_scores_the_closed_form_chance_level():
    """A neighborhood carrying no information must report chance, not 1.0.

    `argsort` breaks the fixture's exact ties by index, deterministically, so a
    diagnostic that perturbed nothing would call these forty proteins perfectly
    stable. Their pool under an 80% subsample is about 31 candidates, and
    ``chance_jaccard`` predicts 0.192 for that.
    """
    result = run()
    pool = round(result.subsample_fraction * len(COHORT.members(TIED))) - 1
    assert group_mean(result, TIED) == pytest.approx(chance_jaccard(pool, result.k), abs=0.02)


def test_the_tied_group_does_not_decay_with_noise():
    """Any sigma above zero randomizes an exact tie completely.

    So the tied band steps straight to chance and stays flat, where `diffuse`
    decays. A statistic that responded smoothly here would be reading the size
    of the perturbation rather than the presence of a gap.
    """
    values = [group_mean(run(noise=noise), TIED) for noise in (0.01, 0.05, 0.10, 0.25)]
    assert max(values) - min(values) < 0.02, values


def test_more_noise_never_raises_a_group_s_stability():
    for group in (GAPPED, DIFFUSE):
        values = [group_mean(run(noise=noise), group) for noise in (0.0, 0.02, 0.05, 0.1, 0.2)]
        assert values == sorted(values, reverse=True), (group, values)


# --- determinism -------------------------------------------------------------


def test_the_same_seed_gives_bitwise_identical_values():
    assert np.array_equal(run().stability, run().stability)


def test_a_different_seed_moves_the_values_but_not_the_bands():
    first, second = run(seed=1), run(seed=2)
    assert not np.array_equal(first.stability, second.stability)
    assert group_mean(first, GAPPED) == pytest.approx(group_mean(second, GAPPED), abs=0.05)


def test_more_replicates_tighten_the_estimate():
    """Twenty replicates of a chance-level group should sit nearer the closed
    form than five do. Not guaranteed for any single pair, so it is asserted on
    the tied group, whose expectation is known rather than estimated."""
    target = chance_jaccard(round(0.8 * len(COHORT.members(TIED))) - 1, K)
    few = abs(group_mean(run(replicates=3), TIED) - target)
    many = abs(group_mean(run(replicates=60), TIED) - target)
    assert many <= few


# --- k clamping, at the demo's scale rather than the fixture's ---------------


def test_k_is_clamped_to_what_a_subsample_can_supply():
    """Group 8b's fifth defect, met before it could ship again.

    ``DEFAULT_K = 15`` against the demo's eleven proteins broke all seven
    spaces while every unit test passed at N=240. Here the ceiling is tighter
    still, because k must fit the subsample: eleven proteins at 80% leave nine,
    which can supply eight neighbors.
    """
    small = stability_cohort(k=2)
    protids = small.protids[:11]
    distances = small.distances()[:11, :11]
    result = neighborhood_stability("demo", distances, protids, k=15, seed=0)
    assert result.k == 8
    assert result.k_requested == 15
    assert any("reduced from 15 to 8" in note for note in result.warnings())


def test_largest_stable_k_matches_the_clamp():
    assert largest_stable_k(11, 0.8) == 8
    assert largest_stable_k(240, 0.8) == 191
    assert largest_stable_k(1, 0.8) == 0


def test_a_cohort_too_small_to_supply_one_neighbor_is_refused():
    small = stability_cohort(k=2)
    with pytest.raises(StabilityError, match="cannot supply a single neighbor"):
        neighborhood_stability("tiny", small.distances()[:1, :1], small.protids[:1], k=1)


def test_a_full_subsample_uses_the_whole_cohort():
    result = run(subsample_fraction=1.0)
    assert result.k == K
    assert np.all(result.replicates_seen == result.replicates)


# --- the Jaccard kernel, against an obvious implementation -------------------


def test_jaccard_rows_agrees_with_python_sets():
    """The fast path is a sort-and-count. The slow path is the definition."""
    rng = np.random.RandomState(0)
    left = np.array([rng.choice(50, size=7, replace=False) for _ in range(40)])
    right = np.array([rng.choice(50, size=7, replace=False) for _ in range(40)])
    expected = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(left, right)]
    assert jaccard_rows(left, right) == pytest.approx(expected)


def test_jaccard_rows_is_one_for_identical_sets_in_any_order():
    left = np.array([[3, 1, 2], [9, 8, 7]])
    right = np.array([[2, 3, 1], [7, 9, 8]])
    assert np.all(jaccard_rows(left, right) == 1.0)


def test_jaccard_rows_is_zero_for_disjoint_sets():
    assert np.all(jaccard_rows(np.array([[1, 2]]), np.array([[3, 4]])) == 0.0)


def test_jaccard_rows_rejects_mismatched_shapes():
    with pytest.raises(StabilityError, match="same shape"):
        jaccard_rows(np.array([[1, 2]]), np.array([[1, 2, 3]]))


# --- the report ---------------------------------------------------------------


def test_the_report_survives_json():
    import json

    payload = json.dumps(run().to_dict())
    assert "stability_mean" in payload


def test_the_frame_is_indexed_by_protid():
    pytest.importorskip("pandas")
    frame = run().to_frame()
    assert frame.index.name == "protid"
    assert list(frame.columns) == ["stability", "replicates_seen"]
    assert len(frame) == COHORT.n_proteins


def test_the_coin_flip_list_names_the_tied_proteins():
    result = run()
    flips = set(result.coin_flips())
    assert set(COHORT.protids_in(TIED)) <= flips
    assert not (set(COHORT.protids_in(GAPPED)) & flips)


def test_the_stable_list_names_the_gapped_proteins():
    result = run()
    assert set(COHORT.protids_in(GAPPED)) <= set(result.stable())


def test_a_fully_stable_space_produces_no_warnings():
    """The other half of the guard. A diagnostic that always fires is noise."""
    assert run(noise=0.0).warnings() == []


def test_a_space_of_pure_ties_says_so():
    """Every protein equidistant from every other -- what a censored similarity
    matrix read without its mask looks like."""
    n = 30
    distances = np.ones((n, n)) - np.eye(n)
    result = neighborhood_stability("ties", distances, [f"P{i}" for i in range(n)], k=5, seed=0)
    assert result.mean_stability < COIN_FLIP_THRESHOLD
    assert any("dominated by exact ties" in note for note in result.warnings())


def test_the_default_noise_is_the_documented_stress_level():
    assert DEFAULT_NOISE == 0.10
    assert run().noise == DEFAULT_NOISE


# --- input validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"subsample_fraction": 0.0}, "subsample_fraction"),
        ({"subsample_fraction": 1.5}, "subsample_fraction"),
        ({"noise": -0.1}, "noise"),
        ({"replicates": 0}, "replicates"),
        ({"k": 0}, "k must be at least 1"),
    ],
)
def test_bad_arguments_are_refused(kwargs, message):
    with pytest.raises(StabilityError, match=message):
        run(**kwargs)


def test_a_non_square_distance_matrix_is_refused():
    with pytest.raises(StabilityError, match="must be square"):
        neighborhood_stability("bad", DISTANCES[:, :5], COHORT.protids, k=3)


# --- the vacuity guard, which the demo found ---------------------------------


def test_a_neighborhood_that_is_the_whole_subsample_says_so():
    """The defect the demo exposed, and the reason to run things end to end.

    At eleven proteins k clamps to 8 and a replicate holds 9, so every
    protein's k nearest are *all* the others and the Jaccard is 1.0 whatever
    the noise. All seven demo spaces reported perfect stability under a sigma
    half the size of the data. Reporting 1.000 there is the most confident
    possible way to say nothing, so the report now says which it is.
    """
    small = stability_cohort(k=2)
    result = neighborhood_stability("demo", small.distances()[:11, :11], small.protids[:11], k=15)
    assert result.mean_stability == 1.0
    assert result.k == 8
    assert result.subsample_size == 9
    assert result.neighborhood_fraction == 1.0
    assert not result.informative
    assert any("Jaccard is 1.0 by construction" in note for note in result.warnings())


def test_a_neighborhood_over_half_the_subsample_is_flagged_without_being_refused():
    """Between "local" and "vacuous" there is a band worth naming rather than
    erroring on -- the score is real, it is just not measuring anything local."""
    small = stability_cohort(k=2)
    result = neighborhood_stability("mid", small.distances()[:30, :30], small.protids[:30], k=15)
    assert result.k == 15
    assert 0.5 <= result.neighborhood_fraction < 1.0
    assert not result.informative
    assert any("most of the cohort" in note for note in result.warnings())
    assert not any("by construction" in note for note in result.warnings())


def test_a_local_neighborhood_is_reported_as_informative():
    """The other half of the guard: it must not fire at a sane k."""
    result = run()
    assert result.informative
    assert result.neighborhood_fraction < 0.1
    assert not any("most of the cohort" in note for note in result.warnings())
