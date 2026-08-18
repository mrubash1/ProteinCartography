#!/usr/bin/env python
"""Trustworthiness and continuity: against the definition, the fixture, sklearn.

The load-bearing test in this file is
:func:`test_the_planted_two_by_two_table_is_recovered`. Everything else checks
that the arithmetic is right; that one checks that the two statistics are
*different statistics*, which is the failure the formulas invite and which no
amount of shape checking can see.
"""

from __future__ import annotations

import numpy as np
import pytest
from diagnostics.embedding import (
    DEFAULT_K,
    DISTORTED_THRESHOLD,
    EmbeddingDiagnosticError,
    EmbeddingFaithfulness,
    continuity,
    faithfulness,
    largest_valid_k,
    trustworthiness,
)
from embedding_cohort import (
    FAITHFUL_THRESHOLD,
    FOLD,
    ISOMETRIC,
    SHUFFLE,
    SPLIT,
    embedding_cohort,
)

CASES = (ISOMETRIC, FOLD, SPLIT, SHUFFLE)
KS = (5, 10, 20)


def _distances(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))


@pytest.fixture(scope="module")
def cohort():
    return embedding_cohort()


@pytest.fixture(scope="module")
def high(cohort):
    return _distances(cohort.high)


@pytest.fixture(scope="module")
def low(cohort):
    return {name: _distances(cohort.case(name).low) for name in CASES}


# --- the planted answer -----------------------------------------------------


@pytest.mark.parametrize("k", KS)
def test_the_planted_two_by_two_table_is_recovered(cohort, high, low, k):
    """The reason the fixture exists.

    An implementation that computes one statistic twice produces only the
    diagonal of this table; one that swaps the two produces it transposed.
    Both pass every shape, range and symmetry check in this file.
    """
    for name in CASES:
        case = cohort.case(name)
        measured_trust = trustworthiness(high, low[name], k).mean() >= FAITHFUL_THRESHOLD
        measured_keep = continuity(high, low[name], k).mean() >= FAITHFUL_THRESHOLD
        assert measured_trust == case.trustworthy, f"{name} trustworthiness at k={k}"
        assert measured_keep == case.continuous, f"{name} continuity at k={k}"


@pytest.mark.parametrize("k", KS)
def test_an_isometric_map_scores_exactly_one(cohort, high, low, k):
    """Not 'close to 1'. The layout is the truth rotated by an orthonormal
    matrix, so every neighbor set matches and every penalty term is zero."""
    assert trustworthiness(high, low[ISOMETRIC], k).min() == 1.0
    assert continuity(high, low[ISOMETRIC], k).min() == 1.0


@pytest.mark.parametrize("k", KS)
def test_the_fold_and_the_split_fail_in_opposite_directions(high, low, k):
    """Stated as an ordering rather than as thresholds, so it survives a
    change to FAITHFUL_THRESHOLD that the 2x2 test would not."""
    fold_gap = trustworthiness(high, low[FOLD], k).mean() - continuity(high, low[FOLD], k).mean()
    split_gap = trustworthiness(high, low[SPLIT], k).mean() - continuity(high, low[SPLIT], k).mean()
    assert fold_gap < -0.15, f"the fold should be much less trustworthy: {fold_gap:+.3f}"
    assert split_gap > +0.15, f"the split should be much less continuous: {split_gap:+.3f}"


def test_a_shuffled_layout_scores_near_chance(high, low):
    for value in (
        trustworthiness(high, low[SHUFFLE], DEFAULT_K).mean(),
        continuity(high, low[SHUFFLE], DEFAULT_K).mean(),
    ):
        assert 0.4 < value < 0.6


# --- against the definition -------------------------------------------------


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("name", CASES)
def test_both_measures_stay_within_the_unit_interval(high, low, k, name):
    for values in (trustworthiness(high, low[name], k), continuity(high, low[name], k)):
        assert values.min() >= 0.0
        assert values.max() <= 1.0


@pytest.mark.parametrize("name", CASES)
def test_continuity_is_trustworthiness_with_the_spaces_exchanged(high, low, name):
    """The definition, asserted. This is what the implementation does, so the
    test is a statement of intent rather than a check on arithmetic -- the
    2x2 table is what would catch the exchange being wrong."""
    np.testing.assert_array_equal(
        continuity(high, low[name], DEFAULT_K),
        trustworthiness(low[name], high, DEFAULT_K),
    )


@pytest.mark.parametrize("name", CASES)
def test_the_global_value_is_exactly_the_mean_of_the_per_protein_ones(cohort, high, low, name):
    """What makes 'per protein' meaningful rather than a division of a total.

    Recomputed here from the unnormalized definition, so the test does not
    share the implementation's normalizing constant.
    """
    n, k = cohort.n_proteins, DEFAULT_K
    per_protein = trustworthiness(high, low[name], k)
    worst = k * (2.0 * n - 3.0 * k - 1.0) / 2.0
    excess = (1.0 - per_protein) * worst
    global_value = 1.0 - (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))) * excess.sum()
    assert abs(global_value - per_protein.mean()) < 1e-12


def test_a_layout_is_unchanged_by_relabeling_the_proteins(cohort, high, low):
    """Both statistics are functions of the geometry, so permuting the protein
    order permutes the results and changes nothing else."""
    rng = np.random.RandomState(7)
    order = rng.permutation(cohort.n_proteins)
    permuted_high = high[np.ix_(order, order)]
    permuted_low = low[FOLD][np.ix_(order, order)]
    np.testing.assert_allclose(
        trustworthiness(permuted_high, permuted_low, DEFAULT_K),
        trustworthiness(high, low[FOLD], DEFAULT_K)[order],
    )


def test_a_uniform_rescale_of_either_space_changes_nothing(high, low):
    """Both are rank statistics, so multiplying every distance by a constant
    cannot move them. Exactly, not approximately."""
    np.testing.assert_array_equal(
        trustworthiness(high * 1000.0, low[FOLD] * 0.001, DEFAULT_K),
        trustworthiness(high, low[FOLD], DEFAULT_K),
    )


# --- the guards -------------------------------------------------------------


def test_k_above_the_ceiling_is_refused(high, low):
    """2N - 3k - 1 must be positive or the statistic is undefined, and at the
    boundary the best and worst maps score the same."""
    n = high.shape[0]
    with pytest.raises(EmbeddingDiagnosticError, match=r"less than \(2N-1\)/3"):
        trustworthiness(high, low[FOLD], (2 * n - 1) // 3 + 1)


def test_k_below_one_is_refused(high, low):
    with pytest.raises(EmbeddingDiagnosticError, match="at least 1"):
        trustworthiness(high, low[FOLD], 0)


def test_mismatched_shapes_are_refused(high, low):
    with pytest.raises(EmbeddingDiagnosticError, match="same square distance matrix"):
        trustworthiness(high, low[FOLD][:10, :10], 5)


def test_a_non_square_matrix_is_refused():
    with pytest.raises(EmbeddingDiagnosticError, match="same square distance matrix"):
        trustworthiness(np.zeros((4, 5)), np.zeros((4, 5)), 2)


def test_a_protid_count_that_disagrees_with_the_matrix_is_refused(cohort, high, low):
    with pytest.raises(EmbeddingDiagnosticError, match="protids but the distance matrix"):
        faithfulness("s", "pca_umap", high, low[FOLD], cohort.protids[:10], k=5)


# --- the report -------------------------------------------------------------


@pytest.fixture(scope="module")
def reports(cohort, high, low):
    return {
        name: faithfulness("structure", "pca_umap", high, low[name], cohort.protids, k=10)
        for name in CASES
    }


def test_the_report_carries_what_it_measured(reports, cohort):
    report = reports[ISOMETRIC]
    assert isinstance(report, EmbeddingFaithfulness)
    assert report.space_id == "structure"
    assert report.reducer == "pca_umap"
    assert report.k == 10
    assert report.n_proteins == cohort.n_proteins
    assert report.protids == tuple(cohort.protids)


def test_the_faithful_map_produces_no_warning_but_still_says_so(reports):
    """A diagnostic that only speaks up when something is wrong leaves a reader
    unable to tell 'checked and fine' from 'not checked'."""
    notes = reports[ISOMETRIC].warnings()
    assert len(notes) == 1
    assert "faithful" in notes[0]


def test_the_fold_is_reported_as_crowded_and_not_as_torn(reports):
    joined = " ".join(reports[FOLD].warnings())
    assert "over-compressed" in joined
    assert "torn rather than crowded" not in joined


def test_the_split_is_reported_as_torn_and_not_as_crowded(reports):
    joined = " ".join(reports[SPLIT].warnings())
    assert "torn rather than crowded" in joined
    assert "over-compressed" not in joined


def test_the_shuffled_map_is_reported_as_failing_both_ways(reports):
    joined = " ".join(reports[SHUFFLE].warnings())
    assert "trustworthiness is" in joined
    assert "continuity is" in joined


def test_no_protein_is_called_unreliable_on_a_faithful_map(reports):
    assert reports[ISOMETRIC].unreliable() == []


def test_unreliable_proteins_are_listed_worst_first(reports):
    report = reports[SHUFFLE]
    unreliable = report.unreliable()
    assert unreliable
    worst = {
        protid: min(t, c)
        for protid, t, c in zip(report.protids, report.trustworthiness, report.continuity)
    }
    scores = [worst[p] for p in unreliable]
    assert scores == sorted(scores)
    assert max(scores) <= DISTORTED_THRESHOLD


def test_a_large_k_is_called_out_as_not_local(cohort, high, low):
    n = cohort.n_proteins
    report = faithfulness("s", "pca_umap", high, low[ISOMETRIC], cohort.protids, k=n // 2)
    assert any("not local neighborhoods" in note for note in report.warnings())


def test_the_report_serializes_to_plain_json_types(reports):
    import json

    payload = reports[FOLD].to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["trustworthiness_mean"] < payload["continuity_mean"]


def test_the_per_protein_frame_is_indexed_by_protid(reports, cohort):
    frame = reports[SPLIT].to_frame()
    assert frame.index.name == "protid"
    assert list(frame.index) == cohort.protids
    assert list(frame.columns) == ["trustworthiness", "continuity"]


# --- against scikit-learn ---------------------------------------------------


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("name", CASES)
def test_trustworthiness_agrees_with_sklearn(cohort, high, low, k, name):
    """To 1e-13 relative, and the tolerance is chosen from measurement.

    Exact equality is not available and the reason is worth recording. The
    global statistic is defined as one normalized sum over all N*k rank
    excesses, which is what scikit-learn computes; this module computes a
    per-protein value and averages, which is algebraically the same quantity
    associated differently. Measured across all four cases at k = 5, 10 and 20,
    the two agree exactly in seven of twelve and differ by exactly one unit in
    the last place in the other five -- worst relative difference 2.2e-16, or
    one machine epsilon.

    So 1e-13 sits three orders above the observed floating-point noise and
    thirteen orders below the smallest error that would indicate a real defect:
    the tie-breaking artifact this fixture was built to avoid moved the answer
    by 2.6e-3. An *absolute* tolerance is what would be vacuous here -- both
    quantities live in [0, 1] and a wholly wrong one is still only 0.5 away.
    """
    sklearn = pytest.importorskip("sklearn.manifold")
    theirs = sklearn.trustworthiness(
        high, cohort.case(name).low, n_neighbors=k, metric="precomputed"
    )
    mine = trustworthiness(high, low[name], k).mean()
    assert abs(mine - theirs) <= 1e-13 * abs(theirs)


def test_the_isometric_case_agrees_with_sklearn_exactly(cohort, high, low):
    """Where equality *is* available it is asserted, rather than being hidden
    under the tolerance the other cases need. Both sides are exactly 1.0."""
    sklearn = pytest.importorskip("sklearn.manifold")
    for k in KS:
        theirs = sklearn.trustworthiness(
            high, cohort.case(ISOMETRIC).low, n_neighbors=k, metric="precomputed"
        )
        assert trustworthiness(high, low[ISOMETRIC], k).mean() == theirs == 1.0


# --- small cohorts ----------------------------------------------------------
#
# Added after the demo. Nineteen unit tests passed and every space in the
# multispace demo failed at once, because every test here ran at N=240 and the
# demo cohort is eleven proteins against a default k of fifteen. Fifth sighting
# of "a passing test suite is not evidence the entry point works".


def test_k_is_clamped_to_the_cohort_rather_than_refused():
    """A cohort smaller than the configured k is ordinary, not an error.

    `reduce_pca` handles the same situation by clamping `n_components` with a
    warning. Refusing instead would mean the smallest cohorts -- the ones whose
    maps are least trustworthy -- are the only ones that get no diagnostics.
    """
    small = embedding_cohort(n=12)
    high = _distances(small.high)
    low = _distances(small.case(FOLD).low)
    report = faithfulness("s", "pca_umap", high, low, small.protids, k=DEFAULT_K)
    assert report.k_requested == DEFAULT_K
    assert report.k == largest_valid_k(12) == 7
    assert report.trustworthiness.shape == (12,)


def test_the_clamp_is_reported_rather_than_silent():
    """Two runs with identical configs and different N are not the same run."""
    small = embedding_cohort(n=12)
    high = _distances(small.high)
    low = _distances(small.case(FOLD).low)
    report = faithfulness("s", "pca_umap", high, low, small.protids, k=DEFAULT_K)
    assert any("k was reduced from 15 to 7" in note for note in report.warnings())
    assert report.to_dict()["k_requested"] == DEFAULT_K
    assert report.to_dict()["k"] == 7


def test_a_k_that_already_fits_is_not_reported_as_clamped():
    """The other half: a diagnostic that always fires is noise."""
    small = embedding_cohort(n=12)
    high = _distances(small.high)
    low = _distances(small.case(ISOMETRIC).low)
    report = faithfulness("s", "pca_umap", high, low, small.protids, k=3)
    assert report.k == report.k_requested == 3
    assert not any("k was reduced" in note for note in report.warnings())


@pytest.mark.parametrize("n,expected", [(3, 1), (11, 6), (12, 7), (240, 159)])
def test_the_largest_valid_k_is_the_largest_one_that_works(n, expected):
    """Pinned against the constraint itself rather than against a formula
    rewritten from the same source."""
    assert largest_valid_k(n) == expected
    assert 2 * n - 3 * expected - 1 > 0, "the claimed maximum is not actually valid"
    assert 2 * n - 3 * (expected + 1) - 1 <= 0, "a larger k would also have worked"


@pytest.mark.parametrize("n", [1, 2])
def test_a_cohort_too_small_for_any_neighborhood_is_refused(n):
    high = np.zeros((n, n))
    with pytest.raises(EmbeddingDiagnosticError, match="no valid neighborhood size"):
        faithfulness("s", "pca_umap", high, high, [f"P{i}" for i in range(n)], k=1)


# --- Gate D: non-finite input --------------------------------------------------


def test_a_nan_distance_is_refused_rather_than_ranked():
    """Gate D produced a trustworthiness of 0.489 from one NaN cell.

    `argsort` sorts NaN last rather than raising, so one non-finite feature
    makes one protein's distances all NaN, that protein sorts last for
    everybody, and the statistic returns a plausible number for a neighbor list
    that is an artifact of the sort. 0.489 is indistinguishable from "this is a
    mediocre map".
    """
    from diagnostics.embedding import EmbeddingDiagnosticError, faithfulness, require_finite

    high = np.abs(np.random.RandomState(0).normal(size=(20, 20)))
    high = (high + high.T) / 2
    np.fill_diagonal(high, 0.0)
    low = high.copy()
    high[3, 7] = np.nan
    with pytest.raises(EmbeddingDiagnosticError, match="NaN or infinite"):
        faithfulness("s", "pca", high, low, [f"P{i}" for i in range(20)], k=4)
    assert require_finite(low, "fine") is not None


def test_an_infinite_distance_is_refused_too():
    from diagnostics.embedding import EmbeddingDiagnosticError, require_finite

    matrix = np.zeros((4, 4))
    matrix[1, 2] = np.inf
    with pytest.raises(EmbeddingDiagnosticError, match="NaN or infinite"):
        require_finite(matrix, "test")


def test_require_finite_passes_a_clean_matrix_through_unchanged():
    from diagnostics.embedding import require_finite

    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert np.array_equal(require_finite(matrix, "test"), matrix)
